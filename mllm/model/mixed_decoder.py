from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.distributed as dist
from transformers import PreTrainedTokenizer

from mllm.config.model import MixedDecoderCfg, MixedDecoderDsType, MixedDecoderType, BertEmbType
from mllm.model.encdec_ranker_hg import EncoderBert
from mllm.model.gpt2 import GPT2LMHeadModel
from mllm.model.losses import EncdecMaskPadItemLoss
from mllm.train.encdec_graph_bert import MaskedCiteBatch
from mllm.train.next_tok_wiki import NextTokBatch
from mllm.train.qna_cite import QnaCiteBatch


class MixedDecoder(nn.Module):
    cfg: MixedDecoderCfg
    tkz: PreTrainedTokenizer
    # enc: EncoderBert
    # pos_emb: nn.Embedding

    def __init__(self, cfg: MixedDecoderCfg, tkz: PreTrainedTokenizer):
        super().__init__()
        self.cfg = cfg
        self.tkz = tkz
        self.enc = EncoderBert(cfg.enc_bert)

        if self.cfg.decoder_type == MixedDecoderType.Gpt2:
            self.decoder = GPT2LMHeadModel.from_pretrained(self.cfg.decoder_model_name)
            self.word_embeddings = self.decoder.transformer.wte
            self.sep_token_id = tkz.sep_token_id if tkz.sep_token_id is not None else tkz.eos_token_id
            d_dec = self.decoder.config.n_embd
        elif self.cfg.decoder_type == MixedDecoderType.BertDec:
            from mllm.model.bert_generation import BertGenerationDecoder
            self.decoder = BertGenerationDecoder.from_pretrained(
                self.cfg.decoder_model_name, is_decoder=True, add_cross_attention=False,
            )
            self.word_embeddings = self.decoder.bert.embeddings.word_embeddings
            self.sep_token_id = tkz.sep_token_id if tkz.sep_token_id is not None else tkz.cls_token_id
            d_dec = self.decoder.config.hidden_size
        else:
            raise ValueError(f'Decoder type {self.cfg.decoder_type} is not supported.')

        # Embedding expansion: each CLS embedding (d_model) → (emb_exp_rate, d_dec)
        self.emb_exp = None
        if self.cfg.emb_exp_rate > 0:
            self.emb_exp = nn.Linear(self.cfg.d_model, self.cfg.emb_exp_rate * d_dec, bias=False)

        # If encoder d_model differs from decoder d_model, add a projection layer (only when no emb expansion)
        self.enc_proj = None
        if self.cfg.emb_exp_rate <= 0 and self.cfg.d_model != d_dec:
            self.enc_proj = nn.Linear(self.cfg.d_model, d_dec, bias=False)

        self.d_dec = d_dec
        # Learnable positional embeddings over the full combined sequence
        self.pos_emb = nn.Embedding(cfg.max_seq_len, d_dec)

        # Mask-aware loss: gives higher weight to [MASK] positions, lower to special tokens
        self.mask_loss_fn = None
        if self.cfg.train_cfg.mask_cfg is not None:
            self.mask_loss_fn = EncdecMaskPadItemLoss(
                msk_tok_id=cast(int, tkz.mask_token_id),
                spc_tok_ids=[cast(int, tkz.pad_token_id), cast(int, tkz.cls_token_id), cast(int, tkz.sep_token_id)],
                reg_weight=1, msk_weight=5, spc_weight=0.1,
            )

    def load_pretrained(self, checkpoint: Optional[Dict[str, Any]] = None):
        if checkpoint is not None:
            checkpt_dict = checkpoint['model']
            cleaned_dict = {}
            for key, val in checkpt_dict.items():
                if key.startswith('module.'):
                    key = key[7:]
                cleaned_dict[key] = val
            print(f'Load {len(cleaned_dict)}')
            self.load_state_dict(cleaned_dict, strict=True)
        else:
            pretrained_mixed_decoder_model_path = self.cfg.train_cfg.pretrained_mixed_decoder_model_path
            pretrained_encdec_model_path = self.cfg.train_cfg.pretrained_encdec_model_path
            rank = dist.get_rank() if dist.is_initialized() else 0

            if pretrained_mixed_decoder_model_path and pretrained_mixed_decoder_model_path.exists():
                # Load full MixedDecoder model with strict mode
                print(f'R{rank}. Loading MixedDecoder checkpoint with strict mode from {pretrained_mixed_decoder_model_path}')
                pretrained_checkpoint = torch.load(pretrained_mixed_decoder_model_path)
                checkpt_dict = pretrained_checkpoint['model']

                cleaned_dict = {}
                for key, val in checkpt_dict.items():
                    if key.startswith('module.'):
                        key = key[7:]
                    if key.startswith('model.'):
                        key = key[6:]
                    cleaned_dict[key] = val

                self.load_state_dict(cleaned_dict, strict=True)
            elif pretrained_encdec_model_path and pretrained_encdec_model_path.exists():
                # Load only encoder weights from an EncdecBert checkpoint
                print(f'R{rank}. Loading encoder checkpoint from {pretrained_encdec_model_path}')
                pretrained_checkpoint = torch.load(pretrained_encdec_model_path)
                checkpt_dict = pretrained_checkpoint['model']

                enc_checkpt_dict = {}
                for key, val in checkpt_dict.items():
                    if key.startswith('module.'):
                        key = key[7:]
                    if key.startswith('model.'):
                        key = key[6:]
                    if key.startswith('enc_bert.'):
                        new_key = key[9:]
                        enc_checkpt_dict[new_key] = val

                self.enc.load_state_dict(enc_checkpt_dict, strict=True)
            else:
                print(f'R{rank}. No pretrained model path provided or file does not exist')

    # inp_toks: (batch_size, inp_len)
    # inp_mask: (batch_size, inp_len)
    # returns: (batch_size, d_model)
    def run_enc(self, inp_toks: Tensor, inp_mask: Tensor) -> Tensor:
        out_enc = self.enc(inp_toks, inp_mask)
        out_enc_last_hidden_state, out_enc_pooler = out_enc
        if self.cfg.enc_bert.emb_type == BertEmbType.Cls:
            return out_enc_last_hidden_state[:, 0]
        elif self.cfg.enc_bert.emb_type == BertEmbType.Pooler:
            return out_enc_pooler
        else:
            raise ValueError(f'Encoder BERT embedding type {self.cfg.enc_bert.emb_type} is not supported')

    def build_decoder_input(
            self, ctx_embs: Tensor, prompt_toks: Tensor, prompt_att_mask: Tensor,
            target_toks: Tensor, target_att_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, int]:
        """Build the concatenated decoder input sequence.

        Layout (use_sep=True):
            [CtxEmb_1, ..., CtxEmb_N, SEP_emb, PromptTokEmb_1, ..., PromptTokEmb_P, TargetTokEmb_1, ..., TargetTokEmb_{T-1}]

        Layout (use_sep=False):
            [CtxEmb_1, ..., CtxEmb_N, PromptTokEmb_1, ..., PromptTokEmb_P, TargetTokEmb_1, ..., TargetTokEmb_{T-1}]

        Args:
            ctx_embs: (batch_size, n_ctx, d_model) - CLS embeddings from encoder
            prompt_toks: (batch_size, prompt_len) - prompt token ids
            prompt_att_mask: (batch_size, prompt_len) - prompt attention mask
            target_toks: (batch_size, target_len) - target token ids (full sequence incl. CLS/SEP)
            target_att_mask: (batch_size, target_len) - target attention mask

        Returns:
            input_embs: (batch_size, total_len, d_dec)
            attention_mask: (batch_size, total_len)
            labels: (batch_size, total_len) - labels with -100 for non-target positions
            target_start_idx: int - start index of target tokens in the combined sequence
        """
        batch_size = ctx_embs.shape[0]
        n_ctx = ctx_embs.shape[1]
        device = ctx_embs.device

        # Project encoder embeddings if needed
        if self.enc_proj is not None:
            ctx_embs = self.enc_proj(ctx_embs)

        parts_embs = [ctx_embs]
        parts_mask = [torch.ones((batch_size, n_ctx), dtype=torch.long, device=device)]
        prefix_len = n_ctx

        # Optional SEP between context and prompt
        if self.cfg.use_sep:
            sep_tok = torch.full((batch_size, 1), self.sep_token_id, dtype=torch.long, device=device)
            sep_emb = self.word_embeddings(sep_tok)  # (batch_size, 1, d_dec)
            parts_embs.append(sep_emb)
            parts_mask.append(torch.ones((batch_size, 1), dtype=torch.long, device=device))
            prefix_len += 1

        # Prompt token embeddings
        prompt_embs = self.word_embeddings(prompt_toks)  # (batch_size, prompt_len, d_dec)
        parts_embs.append(prompt_embs)
        parts_mask.append(prompt_att_mask)
        prompt_len = prompt_toks.shape[1]
        prefix_len += prompt_len

        # Target token embeddings (shifted: input is target[:-1], labels are target)
        target_inp_toks = target_toks[:, :-1]  # (batch_size, target_len - 1)
        target_inp_embs = self.word_embeddings(target_inp_toks)  # (batch_size, target_len - 1, d_dec)
        target_inp_mask = target_att_mask[:, :-1]  # (batch_size, target_len - 1)
        parts_embs.append(target_inp_embs)
        parts_mask.append(target_inp_mask)

        target_start_idx = prefix_len - 1  # Target starts right after the prompt (or SEP if used). The first target token corresponds to the prediction at this position.
        target_inp_len = target_inp_toks.shape[1]
        total_len = prefix_len + target_inp_len

        assert total_len <= self.cfg.max_seq_len, \
            f'Total sequence length {total_len} exceeds max_seq_len={self.cfg.max_seq_len}'

        # Concatenate everything
        input_embs = torch.cat(parts_embs, dim=1)  # (batch_size, total_len, d_dec)
        attention_mask = torch.cat(parts_mask, dim=1)  # (batch_size, total_len)

        # Add positional embeddings
        pos_ids = torch.arange(total_len, device=device).unsqueeze(0)  # (1, total_len)
        pos_embs = self.pos_emb(pos_ids)  # (1, total_len, d_dec)
        input_embs = input_embs + pos_embs

        # Build labels: -100 for prefix (context + sep + prompt), actual token ids for target
        labels = torch.full((batch_size, total_len), -100, dtype=torch.long, device=device)
        # Target labels at positions [target_start_idx, target_start_idx + target_len)
        # The label at position i corresponds to the prediction at position i (next token)
        target_labels = target_toks.clone()
        # Mask padding positions in target labels
        target_labels[target_att_mask == 0] = -100
        labels[:, target_start_idx:target_start_idx + target_toks.shape[1]] = target_labels

        return input_embs, attention_mask, labels, target_start_idx

    def run_decoder(self, input_embs: Tensor, attention_mask: Tensor) -> Tensor:
        """Run decoder forward and return logits.

        Args:
            input_embs: (batch_size, total_len, d_dec)
            attention_mask: (batch_size, total_len)

        Returns:
            logits: (batch_size, total_len, n_vocab)
        """
        if self.cfg.decoder_type == MixedDecoderType.Gpt2:
            out = self.decoder(
                inputs_embeds=input_embs, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            )
        elif self.cfg.decoder_type == MixedDecoderType.BertDec:
            out = self.decoder(
                inputs_embeds=input_embs, attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            raise ValueError(f'Decoder type {self.cfg.decoder_type} is not supported.')
        return out.logits

    def calc_loss(
            self, logits: Tensor, labels: Tensor,
            tokens_inp: Optional[Tensor] = None, tokens_tgt: Optional[Tensor] = None,
            target_start_idx: int = 0,
    ) -> Dict[str, Tensor]:
        """Compute loss on target positions.

        When mask_loss_fn is available and tokens_inp/tokens_tgt are provided,
        uses EncdecMaskPadItemLoss which weights masked positions higher.
        Otherwise falls back to standard cross-entropy.

        Args:
            logits: (batch_size, total_len, n_vocab)
            labels: (batch_size, total_len) with -100 for ignored positions
            tokens_inp: (batch_size, target_len) masked input tokens for mask-aware loss
            tokens_tgt: (batch_size, target_len) original target tokens for mask-aware loss
            target_start_idx: start index of target tokens in the logits sequence

        Returns:
            Dict with 'loss' key (and 'reg_toks_loss', 'msk_toks_loss', 'spc_toks_loss' when mask loss is used).
        """
        if self.mask_loss_fn is not None and tokens_inp is not None and tokens_tgt is not None:
            # Extract target logits: (batch_size, target_len, n_vocab)
            target_len = tokens_tgt.shape[1]
            target_logits = logits[:, target_start_idx:target_start_idx + target_len]
            return self.mask_loss_fn(target_logits, tokens_inp, tokens_tgt)

        # Flatten
        logits_flat = logits.view(-1, logits.shape[-1])  # (batch_size * total_len, n_vocab)
        labels_flat = labels.view(-1)  # (batch_size * total_len)
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        return {'loss': loss}

    def run_on_text_citation(self, batch: MaskedCiteBatch, epoch: int = -1) -> Tuple[Dict[str, Tensor], Tensor]:
        """Main training method.

        1. Encode all input chunks to CLS embeddings
        2. Build decoder input: [ctx_embs, (sep), prompt_toks, target_toks]
        3. Run decoder with causal attention
        4. Compute loss on target positions

        Args:
            batch: MaskedCiteBatch with inp_toks, prompts_toks, etc.
            epoch: Current epoch number.

        Returns:
            Tuple of (loss_dict, logits).
        """
        batch_size = batch.inp_toks.shape[0]

        assert torch.all(batch.inp_toks[:, 0] == self.tkz.cls_token_id), 'Input tokens must start with CLS token'

        # Encode context chunks
        freeze_enc = self.cfg.train_cfg.freeze_encoder
        enc_ctx = torch.no_grad() if freeze_enc else nullcontext()

        with enc_ctx:
            # inp_enc_embs: (batch_size, d_model)
            inp_enc_embs = self.run_enc(batch.inp_masked_toks, batch.inp_att_mask)

        # Context embeddings: all batch CLS embeddings as prefix for each sample
        # Determine embedding window
        emb_win_active = (self.cfg.emb_win_min_size <= self.cfg.emb_win_max_size and self.cfg.emb_win_max_size > 0)
        win_indices = None  # None means use all embeddings
        if emb_win_active:
            win_min = max(self.cfg.emb_win_min_size, 1)
            win_max = min(self.cfg.emb_win_max_size, batch_size)
            win_min = min(win_min, win_max)
            win_size = torch.randint(win_min, win_max + 1, (1,)).item()
            if win_size < batch_size:
                offset_before = torch.randint(0, win_size, (1,)).item() if win_size > 1 else 0
                sample_idx = torch.arange(batch_size, device=inp_enc_embs.device)
                j_idx = torch.arange(win_size, device=inp_enc_embs.device)
                # win_indices[i, j] = (i - offset_before + j) % batch_size
                # Ensures each sample's own embedding is always at position offset_before
                win_indices = (sample_idx.unsqueeze(1) - offset_before + j_idx.unsqueeze(0)) % batch_size

        if self.cfg.emb_exp_rate > 0:
            # Expand each CLS embedding from 1 vector to emb_exp_rate vectors
            # inp_enc_embs: (batch_size, d_model) -> (batch_size, emb_exp_rate * d_dec)
            exp_embs = self.emb_exp(inp_enc_embs)
            # (batch_size, emb_exp_rate * d_dec) -> (batch_size, emb_exp_rate, d_dec)
            exp_embs = exp_embs.view(batch_size, self.cfg.emb_exp_rate, self.d_dec)
            if win_indices is not None:
                # (batch_size, win_size, emb_exp_rate, d_dec) -> (batch_size, win_size * emb_exp_rate, d_dec)
                ctx_embs = exp_embs[win_indices]
                ctx_embs = ctx_embs.reshape(batch_size, win_indices.shape[1] * self.cfg.emb_exp_rate, self.d_dec)
            else:
                # (batch_size, emb_exp_rate, d_dec) -> (1, batch_size * emb_exp_rate, d_dec)
                exp_embs = exp_embs.reshape(1, batch_size * self.cfg.emb_exp_rate, self.d_dec)
                # -> (batch_size, batch_size * emb_exp_rate, d_dec)
                ctx_embs = exp_embs.expand(batch_size, -1, -1)
        else:
            if win_indices is not None:
                # ctx_embs: (batch_size, win_size, d_model)
                ctx_embs = inp_enc_embs[win_indices]
            else:
                # ctx_embs: (batch_size, batch_size, d_model)
                ctx_embs = inp_enc_embs.unsqueeze(0).expand(batch_size, -1, -1)

        # Select target based on prompt_all config
        if self.cfg.prompt_all:
            # Target is the whole input chunk with tags (inp_toks), starts with CLS
            target_toks = batch.inp_toks  # (batch_size, inp_len)
            target_att_mask = batch.inp_att_mask  # (batch_size, inp_len)
            # Strip leading CLS token - it's an encoder special token, not a generation target
            assert torch.all(target_toks[:, 0] == self.tkz.cls_token_id), \
                'Target tokens (inp_toks) must start with CLS token'
            target_toks = target_toks[:, 1:]
            target_att_mask = target_att_mask[:, 1:]
        else:
            # Target is just the citation between tags (cites_toks) - no CLS prefix
            target_toks = batch.cites_toks  # (batch_size, cite_len)
            target_att_mask = batch.cites_att_mask  # (batch_size, cite_len)

        # Build decoder input
        input_embs, attention_mask, labels, target_start_idx = self.build_decoder_input(
            ctx_embs, batch.prompts_toks, batch.prompts_att_mask,
            target_toks, target_att_mask,
        )

        # Run decoder
        logits = self.run_decoder(input_embs, attention_mask)

        # Compute loss
        if self.mask_loss_fn is not None:
            if self.cfg.prompt_all:
                # target_toks was stripped of leading CLS, so masked tokens must match
                masked_target_toks = batch.inp_masked_toks[:, 1:]  # (batch_size, inp_len - 1)
                orig_target_toks = batch.inp_toks[:, 1:]  # (batch_size, inp_len - 1)
            else:
                masked_target_toks = batch.cites_masked_toks  # (batch_size, cite_len)
                orig_target_toks = batch.cites_toks  # (batch_size, cite_len)
            loss_dict = self.calc_loss(logits, labels, masked_target_toks, orig_target_toks, target_start_idx)
        else:
            loss_dict = self.calc_loss(logits, labels)

        return loss_dict, logits

    def run_on_qna(self, batch: QnaCiteBatch, epoch: int = -1) -> Tuple[Dict[str, Tensor], Tensor]:
        """Training method for QnA (SQuAD v2) data.

        1. Encode all context chunks to CLS embeddings
        2. For each sample, gather its own context embeddings; fill remaining
           slots (up to emb_win_max_size) with random embeddings from the batch,
           placed before or after the sample's own embeddings (random coin flip).
        3. Apply emb_exp expansion if configured.
        4. Build decoder input: [ctx_embs, (sep), prompt_toks, ans_toks]
        5. Run decoder with causal attention
        6. Compute loss on answer positions

        Args:
            batch: QnaCiteBatch with context chunks, prompts, and answer tokens.
            epoch: Current epoch number.

        Returns:
            Tuple of (loss_dict, logits).
        """
        batch_size = len(batch.ctx_chunk_counts)
        device = batch.ctx_chunks_toks.device

        # 1. Encode all context chunks
        freeze_enc = self.cfg.train_cfg.freeze_encoder
        enc_ctx = torch.no_grad() if freeze_enc else nullcontext()

        with enc_ctx:
            # all_enc_embs: (total_chunks, d_model)
            all_enc_embs = self.run_enc(batch.ctx_chunks_toks, batch.ctx_chunks_att_mask)

        total_chunks = all_enc_embs.shape[0]

        # 2. Determine target context window size per sample
        emb_win_max = self.cfg.emb_win_max_size
        emb_win_min = max(self.cfg.emb_win_min_size, 1)
        if emb_win_max <= 0:
            # No windowing: use max chunks in this batch
            emb_win_max = max(batch.ctx_chunk_counts)
            emb_win_min = emb_win_max

        # Optionally randomize window size
        win_min = min(emb_win_min, emb_win_max)
        target_win_size = torch.randint(win_min, emb_win_max + 1, (1,)).item()

        # 3. Build ctx_embs per sample: own chunks + filler from other samples
        # Split all_enc_embs by chunk_counts
        chunk_offsets = [0]
        for c in batch.ctx_chunk_counts:
            chunk_offsets.append(chunk_offsets[-1] + c)

        # Collect per-sample own embeddings (before expansion)
        own_embs_list = []  # list of (n_own_i, d_model) tensors
        for i in range(batch_size):
            start, end = chunk_offsets[i], chunk_offsets[i + 1]
            own_embs_list.append(all_enc_embs[start:end])  # (n_own_i, d_model)

        # Build padded context embeddings of shape (batch_size, target_win_size, d_model)
        ctx_embs_raw = torch.zeros((batch_size, target_win_size, self.cfg.d_model), device=device)
        for i in range(batch_size):
            own = own_embs_list[i]
            n_own = min(own.shape[0], target_win_size)

            if n_own >= target_win_size:
                # Have enough own context chunks
                ctx_embs_raw[i] = own[:target_win_size]
            else:
                # Need filler embeddings from other samples in the batch
                n_filler = target_win_size - n_own

                # Collect all embeddings except this sample's
                other_inds = list(range(0, chunk_offsets[i])) + list(range(chunk_offsets[i + 1], total_chunks))
                if len(other_inds) >= n_filler:
                    perm = torch.randperm(len(other_inds), device=device)[:n_filler]
                    filler_inds = torch.tensor(other_inds, device=device)[perm]
                else:
                    # Not enough other embeddings; repeat with replacement
                    filler_inds = torch.tensor(other_inds, device=device)
                    if len(other_inds) > 0:
                        extra = torch.randint(0, len(other_inds), (n_filler - len(other_inds),), device=device)
                        filler_inds = torch.cat([filler_inds, torch.tensor(other_inds, device=device)[extra]])
                    else:
                        # Only one sample with all chunks; pad with zeros
                        filler_inds = torch.tensor([], dtype=torch.long, device=device)

                filler_embs = all_enc_embs[filler_inds] if filler_inds.numel() > 0 else torch.zeros((n_filler, self.cfg.d_model), device=device)

                # Random coin flip: place own embeddings first or last
                if torch.rand(1).item() < 0.5:
                    # Own first, filler after
                    ctx_embs_raw[i, :n_own] = own[:n_own]
                    ctx_embs_raw[i, n_own:n_own + filler_embs.shape[0]] = filler_embs
                else:
                    # Filler first, own after
                    n_fill = filler_embs.shape[0]
                    ctx_embs_raw[i, :n_fill] = filler_embs
                    ctx_embs_raw[i, n_fill:n_fill + n_own] = own[:n_own]

        # 4. Apply emb_exp expansion or projection
        if self.cfg.emb_exp_rate > 0:
            # ctx_embs_raw: (batch_size, target_win_size, d_model)
            # Expand each embedding from d_model to emb_exp_rate * d_dec
            exp_embs = self.emb_exp(ctx_embs_raw)  # (batch_size, target_win_size, emb_exp_rate * d_dec)
            exp_embs = exp_embs.view(batch_size, target_win_size * self.cfg.emb_exp_rate, self.d_dec)
            ctx_embs = exp_embs
        else:
            ctx_embs = ctx_embs_raw  # (batch_size, target_win_size, d_model)

        # 5. Build decoder input per sample (prompts have variable lengths)
        # Project encoder embeddings if needed
        if self.enc_proj is not None:
            ctx_embs = self.enc_proj(ctx_embs)

        n_ctx = ctx_embs.shape[1]
        sep_len = 1 if self.cfg.use_sep else 0

        # Pre-compute all embeddings
        prompt_embs_all = self.word_embeddings(batch.prompt_toks)    # (bs, max_prompt_len, d_dec)
        target_inp_embs_all = self.word_embeddings(batch.ans_toks[:, :-1])  # (bs, max_ans_len-1, d_dec)
        if self.cfg.use_sep:
            sep_emb = self.word_embeddings(
                torch.full((1, 1), self.sep_token_id, dtype=torch.long, device=device)
            )  # (1, 1, d_dec)

        prompt_lens = batch.prompt_lengths
        ans_lens = [int(batch.ans_att_mask[i].sum().item()) for i in range(batch_size)]

        # Compute per-sample total length and max
        total_lens = [n_ctx + sep_len + prompt_lens[i] + max(ans_lens[i] - 1, 0) for i in range(batch_size)]
        max_total_len = max(total_lens)
        assert max_total_len <= self.cfg.max_seq_len, \
            f'Total sequence length {max_total_len} exceeds max_seq_len={self.cfg.max_seq_len}'

        input_embs = torch.zeros((batch_size, max_total_len, self.d_dec), device=device)
        attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_total_len), -100, dtype=torch.long, device=device)

        for i in range(batch_size):
            pos = 0
            # Context embeddings
            input_embs[i, :n_ctx] = ctx_embs[i]
            attention_mask[i, :n_ctx] = 1
            pos = n_ctx

            # Optional SEP
            if self.cfg.use_sep:
                input_embs[i, pos:pos + 1] = sep_emb[0]
                attention_mask[i, pos:pos + 1] = 1
                pos += 1

            # Prompt tokens (actual length, no padding)
            pl = prompt_lens[i]
            input_embs[i, pos:pos + pl] = prompt_embs_all[i, :pl]
            attention_mask[i, pos:pos + pl] = 1

            target_start_i = pos + pl - 1  # last prompt pos predicts first answer token
            pos += pl

            # Target input tokens (shifted: first ans_len-1 tokens)
            til = max(ans_lens[i] - 1, 0)
            if til > 0:
                input_embs[i, pos:pos + til] = target_inp_embs_all[i, :til]
                attention_mask[i, pos:pos + til] = 1
            pos += til

            # Labels: place actual answer tokens at [target_start_i, target_start_i + ans_len)
            al = ans_lens[i]
            labels[i, target_start_i:target_start_i + al] = batch.ans_toks[i, :al]

        # Positional embeddings
        pos_ids = torch.arange(max_total_len, device=device).unsqueeze(0)
        input_embs = input_embs + self.pos_emb(pos_ids)

        # 6. Run decoder and compute loss
        logits = self.run_decoder(input_embs, attention_mask)
        loss_dict = self.calc_loss(logits, labels)

        return loss_dict, logits

    def run_on_next(self, batch: NextTokBatch, epoch: int = -1) -> Tuple[Dict[str, Tensor], Tensor]:
        """Training method for next-token prediction with Wikipedia data.

        1. Encode all context chunks to CLS embeddings.
        2. Per-sample: use exactly the sample's own context embeddings (no filler).
           Pad to the maximum chunk count across the batch.
        3. Apply emb_exp expansion if configured.
        4. Build decoder input: [ctx_embs, (sep), prompt_embs, target_inp_embs].
        5. Run decoder with causal attention.
        6. Compute loss on target positions.

        Args:
            batch: NextTokBatch with context chunks, prompt, and target tokens.
            epoch: Current epoch number.

        Returns:
            Tuple of (loss_dict, logits).
        """
        batch_size = len(batch.ctx_chunk_counts)
        device = batch.ctx_chunks_toks.device

        # 1. Encode all context chunks
        freeze_enc = self.cfg.train_cfg.freeze_encoder
        enc_ctx = torch.no_grad() if freeze_enc else nullcontext()

        with enc_ctx:
            # all_enc_embs: (total_chunks, d_model)
            all_enc_embs = self.run_enc(batch.ctx_chunks_toks, batch.ctx_chunks_att_mask)

        # 2. Split encoded embeddings per sample and pad to max window size
        chunk_offsets = [0]
        for c in batch.ctx_chunk_counts:
            chunk_offsets.append(chunk_offsets[-1] + c)

        max_win = max(batch.ctx_chunk_counts)

        ctx_embs_raw = torch.zeros((batch_size, max_win, self.cfg.d_model), device=device)
        for i in range(batch_size):
            start, end = chunk_offsets[i], chunk_offsets[i + 1]
            n_own = end - start
            ctx_embs_raw[i, :n_own] = all_enc_embs[start:end]

        # 3. Apply emb_exp expansion or projection
        if self.cfg.emb_exp_rate > 0:
            exp_embs = self.emb_exp(ctx_embs_raw)  # (batch_size, max_win, emb_exp_rate * d_dec)
            exp_embs = exp_embs.view(batch_size, max_win * self.cfg.emb_exp_rate, self.d_dec)
            ctx_embs = exp_embs
        else:
            ctx_embs = ctx_embs_raw

        # 4. Build decoder input per sample
        if self.enc_proj is not None:
            ctx_embs = self.enc_proj(ctx_embs)

        n_ctx = ctx_embs.shape[1]
        sep_len = 1 if self.cfg.use_sep else 0

        prompt_embs_all = self.word_embeddings(batch.prompt_toks)        # (bs, max_prompt_len, d_dec)
        target_inp_embs_all = self.word_embeddings(batch.target_toks[:, :-1])  # (bs, max_target_len-1, d_dec)
        if self.cfg.use_sep:
            sep_emb = self.word_embeddings(
                torch.full((1, 1), self.sep_token_id, dtype=torch.long, device=device)
            )  # (1, 1, d_dec)

        prompt_lens = batch.prompt_lengths
        target_lens = [int(batch.target_att_mask[i].sum().item()) for i in range(batch_size)]

        total_lens = [n_ctx + sep_len + prompt_lens[i] + max(target_lens[i] - 1, 0) for i in range(batch_size)]
        max_total_len = max(total_lens)
        assert max_total_len <= self.cfg.max_seq_len, \
            f'Total sequence length {max_total_len} exceeds max_seq_len={self.cfg.max_seq_len}'

        input_embs = torch.zeros((batch_size, max_total_len, self.d_dec), device=device)
        attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_total_len), -100, dtype=torch.long, device=device)

        for i in range(batch_size):
            pos = 0
            # Context embeddings
            input_embs[i, :n_ctx] = ctx_embs[i]
            attention_mask[i, :n_ctx] = 1
            pos = n_ctx

            # Optional SEP
            if self.cfg.use_sep:
                input_embs[i, pos:pos + 1] = sep_emb[0]
                attention_mask[i, pos:pos + 1] = 1
                pos += 1

            # Prompt tokens
            pl = prompt_lens[i]
            input_embs[i, pos:pos + pl] = prompt_embs_all[i, :pl]
            attention_mask[i, pos:pos + pl] = 1

            target_start_i = pos + pl - 1  # last prompt pos predicts first target token
            pos += pl

            # Target input tokens (shifted: first target_len-1 tokens)
            til = max(target_lens[i] - 1, 0)
            if til > 0:
                input_embs[i, pos:pos + til] = target_inp_embs_all[i, :til]
                attention_mask[i, pos:pos + til] = 1
            pos += til

            # Labels: actual target tokens at [target_start_i, target_start_i + target_len)
            al = target_lens[i]
            labels[i, target_start_i:target_start_i + al] = batch.target_toks[i, :al]

        # Positional embeddings
        pos_ids = torch.arange(max_total_len, device=device).unsqueeze(0)
        input_embs = input_embs + self.pos_emb(pos_ids)

        # 5. Run decoder and compute loss
        logits = self.run_decoder(input_embs, attention_mask)
        loss_dict = self.calc_loss(logits, labels)

        return loss_dict, logits

    def forward(self, batch: Union[MaskedCiteBatch, QnaCiteBatch, NextTokBatch], epoch: int = -1) -> Tuple[Dict[str, Tensor], Tensor]:
        if self.cfg.train_ds_type == MixedDecoderDsType.Next:
            return self.run_on_next(batch, epoch)
        if self.cfg.train_ds_type == MixedDecoderDsType.Qna:
            return self.run_on_qna(batch, epoch)
        return self.run_on_text_citation(batch, epoch)

