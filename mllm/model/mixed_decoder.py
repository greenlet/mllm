from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.distributed as dist
from transformers import PreTrainedTokenizer

from mllm.config.model import MixedDecoderCfg, MixedDecoderDsType, MixedDecoderType, BertEmbType, \
    InteractiveExtractorAttnType
from mllm.model.encdec_ranker_hg import EncoderBert
from mllm.model.gpt2 import GPT2LMHeadModel
from mllm.model.losses import EncdecMaskPadItemLoss
from mllm.train.encdec_graph_bert import MaskedCiteBatch
from mllm.train.next_tok_wiki import NextTokBatch
from mllm.data.qna.batch import QnaBatch


class InteractiveExtractor(nn.Module):
    """Query-conditioned soft-token bridge (v1).

    Replaces the plain linear `emb_exp` expansion. Each of the ``N`` context
    embeddings (``d_model``) is first expanded into ``exp_rate`` decoder-space
    slots (``N -> N*exp_rate`` vectors of ``d_dec``); the slots then VISIT the
    prompt through ``num_layers`` attention blocks before being prepended to the
    decoder as soft tokens.

    No pooling is performed in this version: the output keeps all
    ``N*exp_rate`` slots.

    forward:
        chunk_embs:  (B, N, d_model)        raw context embeddings (pre-expansion)
        prompt_embs: (B, L_q, d_dec)        prompt token embeddings (decoder space)
        prompt_pad:  (B, L_q) bool or None  True at padding positions to ignore
      returns:       (B, N*exp_rate, d_dec) query-conditioned soft tokens
    """

    def __init__(
            self, d_model: int, d_dec: int, exp_rate: int, num_layers: int,
            attn_type: InteractiveExtractorAttnType, n_heads: int, mlp_ratio: float,
            dropout: float, norm_first: bool, max_ctx: int, max_prompt_len: int,
    ):
        super().__init__()
        assert exp_rate > 0, 'InteractiveExtractor requires exp_rate > 0'
        assert num_layers > 0, 'InteractiveExtractor requires num_layers > 0'
        assert d_dec % n_heads == 0, f'd_dec ({d_dec}) must be divisible by n_heads ({n_heads})'
        assert max_ctx > 0, 'InteractiveExtractor requires max_ctx > 0'
        assert max_prompt_len > 0, 'InteractiveExtractor requires max_prompt_len > 0'
        self.d_model = d_model
        self.d_dec = d_dec
        self.exp_rate = exp_rate
        self.num_layers = num_layers
        self.attn_type = attn_type

        # EXPAND: one embedding -> exp_rate decoder-space slots.
        self.expand = nn.Linear(d_model, exp_rate * d_dec, bias=False)

        # Single learned absolute positional table shared by the soft-token slots
        # and the prompt. The first `max_slots = max_ctx * exp_rate` rows position
        # the expanded context slots (so each slot knows which of the N context
        # chunks it came from AND which of the exp_rate copies it is); the next
        # `max_prompt_len` rows position the prompt tokens before the VISIT step.
        self.max_slots = max_ctx * exp_rate
        self.max_prompt_len = max_prompt_len
        self.pos_emb = nn.Parameter(torch.zeros(self.max_slots + max_prompt_len, d_dec))

        self.attns = nn.ModuleList(
            nn.MultiheadAttention(d_dec, n_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        )
        self.attn_norms = nn.ModuleList(nn.LayerNorm(d_dec) for _ in range(num_layers))

        # Optional per-block feed-forward network (mlp_ratio <= 0 disables it).
        self.use_ffn = mlp_ratio > 0
        if self.use_ffn:
            hidden = int(round(mlp_ratio * d_dec))
            self.ffns = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(d_dec, hidden), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden, d_dec), nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            )
            self.ffn_norms = nn.ModuleList(nn.LayerNorm(d_dec) for _ in range(num_layers))
        self.norm_first = norm_first
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot init for all linear projections (zero bias) and a small
        normal init for the learned positional table."""
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.xavier_uniform_(self.expand.weight)
        for attn in self.attns:
            # MultiheadAttention packs q/k/v into in_proj_weight; out_proj is a Linear.
            nn.init.xavier_uniform_(attn.in_proj_weight)
            if attn.in_proj_bias is not None:
                nn.init.zeros_(attn.in_proj_bias)
            nn.init.xavier_uniform_(attn.out_proj.weight)
            if attn.out_proj.bias is not None:
                nn.init.zeros_(attn.out_proj.bias)
        if self.use_ffn:
            for ffn in self.ffns:
                for module in ffn:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    def _attend(
            self, idx: int, x: Tensor, prompt_embs: Tensor, prompt_pad: Optional[Tensor],
    ) -> Tensor:
        """Run one VISIT attention block on the slots ``x``."""
        attn = self.attns[idx]
        if self.attn_type == InteractiveExtractorAttnType.Cross:
            # Slots query the prompt (keys/values = prompt).
            out, _ = attn(query=x, key=prompt_embs, value=prompt_embs, key_padding_mask=prompt_pad)
            return out
        # Self-attention over concat[slots, prompt]; slice the slots back out.
        n_slots = x.shape[1]
        joint = torch.cat([x, prompt_embs], dim=1)
        if prompt_pad is not None:
            slots_pad = torch.zeros((x.shape[0], n_slots), dtype=torch.bool, device=x.device)
            joint_pad = torch.cat([slots_pad, prompt_pad], dim=1)
        else:
            joint_pad = None
        out, _ = attn(query=joint, key=joint, value=joint, key_padding_mask=joint_pad)
        return out[:, :n_slots]

    def forward(
            self, chunk_embs: Tensor, prompt_embs: Tensor, prompt_pad: Optional[Tensor] = None,
    ) -> Tensor:
        b, n, _ = chunk_embs.shape
        n_slots = n * self.exp_rate
        l_q = prompt_embs.shape[1]
        assert n_slots <= self.max_slots, (
            f'InteractiveExtractor: {n} context embeddings * exp_rate {self.exp_rate} = '
            f'{n_slots} slots exceed max_slots={self.max_slots} (increase ie_max_ctx).'
        )
        assert l_q <= self.max_prompt_len, (
            f'InteractiveExtractor: prompt length {l_q} exceeds '
            f'max_prompt_len={self.max_prompt_len} (increase ie_max_prompt_len).'
        )

        # EXPAND -> (B, N, exp_rate, d_dec) -> (B, N*exp_rate, d_dec)
        x = self.expand(chunk_embs).view(b, n, self.exp_rate, self.d_dec)
        x = x.reshape(b, n_slots, self.d_dec)
        # Add the slot region of the shared positional table.
        x = x + self.pos_emb[:n_slots]

        # Add the prompt region of the shared positional table (once, before VISIT).
        prompt_embs = prompt_embs + self.pos_emb[self.max_slots:self.max_slots + l_q]

        for i in range(self.num_layers):
            # VISIT (cross- or self-attention), pre-/post-LN residual.
            if self.norm_first:
                x = x + self.dropout(self._attend(i, self.attn_norms[i](x), prompt_embs, prompt_pad))
            else:
                x = self.attn_norms[i](x + self.dropout(self._attend(i, x, prompt_embs, prompt_pad)))
            if self.use_ffn:
                if self.norm_first:
                    x = x + self.ffns[i](self.ffn_norms[i](x))
                else:
                    x = self.ffn_norms[i](x + self.ffns[i](x))
        return x


class MixedDecoder(nn.Module):
    cfg: MixedDecoderCfg
    tkz_enc: PreTrainedTokenizer
    tkz_dec: PreTrainedTokenizer
    # enc: EncoderBert
    # pos_emb: nn.Embedding

    def __init__(self, cfg: MixedDecoderCfg, tkz_enc: PreTrainedTokenizer, tkz_dec: PreTrainedTokenizer):
        super().__init__()
        self.cfg = cfg
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec
        # Backward-compat alias: legacy code may still read self.tkz.
        self.tkz = tkz_enc
        self.decoder_only = cfg.decoder_only
        if not self.decoder_only:
            self.enc = EncoderBert(cfg.enc_bert)
        else:
            self.enc = None

        if self.cfg.decoder_type == MixedDecoderType.Gpt2:
            self.decoder = GPT2LMHeadModel.from_pretrained(self.cfg.decoder_model_name)
            self.word_embeddings = self.decoder.transformer.wte
            # GPT-2 has only the eos special token; use it as SEP/delimiter and as the target ending.
            self.sep_token_id = tkz_dec.sep_token_id if tkz_dec.sep_token_id is not None else tkz_dec.eos_token_id
            d_dec = self.decoder.config.n_embd
        elif self.cfg.decoder_type == MixedDecoderType.BertDec:
            from mllm.model.bert_generation import BertGenerationDecoder
            self.decoder = BertGenerationDecoder.from_pretrained(
                self.cfg.decoder_model_name, is_decoder=True, add_cross_attention=False,
            )
            self._extend_bert_position_embeddings(self.cfg.max_seq_len)
            self.word_embeddings = self.decoder.bert.embeddings.word_embeddings
            self.sep_token_id = tkz_dec.sep_token_id if tkz_dec.sep_token_id is not None else tkz_dec.cls_token_id
            d_dec = self.decoder.config.hidden_size
        elif self.cfg.decoder_type == MixedDecoderType.Qwen:
            from transformers import AutoModelForCausalLM
            # Always load decoder weights in fp32; mixed-precision compute is handled
            # via torch.cuda.amp.autocast at training time, keyed off cfg.decoder_dtype.
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.cfg.decoder_model_name, torch_dtype=torch.float32,
            )
            # Reduce activation memory: critical for fp32 1.5B+ on 32GB GPUs.
            # use_reentrant=False is required for DDP compatibility: the legacy
            # reentrant checkpoint replays autograd inside backward, which makes
            # DDP's per-parameter ready hooks fire twice and crash with
            # "Expected to mark a variable ready only once".
            self.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False},
            )
            self.decoder.config.use_cache = False
            self.word_embeddings = self.decoder.get_input_embeddings()
            # Qwen has no SEP; use eos as delimiter (also serves as pad in Qwen2.5/Qwen3).
            self.sep_token_id = tkz_dec.eos_token_id
            d_dec = self.decoder.config.hidden_size
            # Apply self-attention dropout if requested (Qwen-only knob; Qwen2/2.5/3
            # expose only `attention_dropout` — no separate residual/MLP dropout).
            attn_dp = self.cfg.train_cfg.attention_dropout
            if attn_dp > 0:
                self.decoder.config.attention_dropout = attn_dp
                for layer in self.decoder.model.layers:
                    layer.self_attn.attention_dropout = attn_dp
            # LoRA (parameter-efficient fine-tuning). Wrap AFTER the attention-dropout
            # loop above, which relies on the un-wrapped `self.decoder.model.layers`
            # path (peft nests the base model under `base_model.model`). When enabled,
            # the decoder base weights are frozen and only the low-rank adapters train;
            # the encoder / emb_exp / InteractiveExtractor bridges keep training.
            if self.cfg.train_cfg.use_lora:
                from peft import LoraConfig, get_peft_model
                target_modules = list(self.cfg.train_cfg.lora_target_modules) or [
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj',
                ]
                lora_cfg = LoraConfig(
                    task_type='CAUSAL_LM',
                    r=self.cfg.train_cfg.lora_rank,
                    lora_alpha=self.cfg.train_cfg.lora_alpha,
                    lora_dropout=self.cfg.train_cfg.lora_dropout,
                    target_modules=target_modules,
                    bias='none',
                )
                self.decoder = get_peft_model(self.decoder, lora_cfg)
                # With gradient checkpointing + a frozen base, ensure gradients can
                # still reach the LoRA adapters (standard peft recipe; a no-op on the
                # inputs_embeds path but harmless).
                self.decoder.enable_input_require_grads()
                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    self.decoder.print_trainable_parameters()
                # Re-fetch the input embedding module through the peft wrapper so the
                # sanity check / token embedding lookups below use the live object.
                self.word_embeddings = self.decoder.get_input_embeddings()
            # Qwen vocab differs from BERT, so the mask-aware decoder loss cannot be
            # used (mask positions / vocab IDs do not transfer). mask_cfg is still
            # honored on the encoder side (encoder receives inp_masked_toks); the
            # decoder simply trains with plain cross-entropy on the unmasked target.
        else:
            raise ValueError(f'Decoder type {self.cfg.decoder_type} is not supported.')

        # Sanity check: decoder embedding table must match tkz_dec vocab size.
        assert self.word_embeddings.num_embeddings >= len(tkz_dec), (
            f'Decoder embedding size {self.word_embeddings.num_embeddings} is smaller than '
            f'tkz_dec vocab size {len(tkz_dec)}. Check that --decoder-model-name and the '
            f'decoder tokenizer come from the same model family.'
        )

        # Embedding expansion: each CLS embedding (d_model) → (emb_exp_rate, d_dec).
        # Skipped entirely when the InteractiveExtractor is enabled (it replaces emb_exp).
        self.emb_exp = None
        if not self.decoder_only and not self.cfg.use_interactive_extractor and self.cfg.emb_exp_rate > 0:
            self.emb_exp = nn.Linear(self.cfg.d_model, self.cfg.emb_exp_rate * d_dec, bias=False)

        # InteractiveExtractor: query-conditioned soft-token bridge. When enabled it
        # replaces both the plain emb_exp expansion and the enc_proj projection
        # (its output is already in decoder space d_dec).
        self.interactive_extractor = None
        if not self.decoder_only and self.cfg.use_interactive_extractor:
            self.interactive_extractor = InteractiveExtractor(
                d_model=self.cfg.d_model, d_dec=d_dec, exp_rate=self.cfg.ie_exp_rate,
                num_layers=self.cfg.ie_num_layers, attn_type=self.cfg.ie_attn_type,
                n_heads=self.cfg.ie_n_heads, mlp_ratio=self.cfg.ie_mlp_ratio,
                dropout=self.cfg.ie_dropout, norm_first=self.cfg.ie_norm_first,
                max_ctx=self.cfg.ie_max_ctx, max_prompt_len=self.cfg.ie_max_prompt_len,
            )

        # If encoder d_model differs from decoder d_model, add a projection layer (only when no emb expansion)
        self.enc_proj = None
        if (not self.decoder_only and self.interactive_extractor is None
                and self.cfg.emb_exp_rate <= 0 and self.cfg.d_model != d_dec):
            self.enc_proj = nn.Linear(self.cfg.d_model, d_dec, bias=False)

        self.d_dec = d_dec
        # Learnable positional embeddings over the full combined sequence.
        # Qwen uses RoPE inside its attention layers, so adding a learned absolute
        # positional embedding on top of inputs_embeds would corrupt the geometry.
        if self.cfg.decoder_type == MixedDecoderType.Qwen:
            self.pos_emb = None
        else:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, d_dec)

        # Mask-aware loss: gives higher weight to [MASK] positions, lower to special tokens.
        # Special-token ids come from tkz_dec because the loss is computed over decoder targets.
        # The mask-aware loss requires the encoder and decoder to share a tokenizer
        # (so mask positions / vocab IDs align). When they don't (e.g. BERT encoder
        # + GPT-2 / Qwen decoder) we still honor mask_cfg on the encoder side
        # (encoder receives inp_masked_toks) but fall back to standard cross-entropy
        # on the unmasked decoder target.
        self.mask_loss_fn = None
        if self.cfg.train_cfg.mask_cfg is not None:
            same_vocab = tkz_enc is tkz_dec or len(tkz_enc) == len(tkz_dec)
            if same_vocab and tkz_dec.mask_token_id is not None:
                spc_ids = [
                    tid for tid in (tkz_dec.pad_token_id, tkz_dec.cls_token_id, tkz_dec.sep_token_id)
                    if tid is not None
                ]
                self.mask_loss_fn = EncdecMaskPadItemLoss(
                    msk_tok_id=cast(int, tkz_dec.mask_token_id),
                    spc_tok_ids=spc_ids,
                    reg_weight=1, msk_weight=5, spc_weight=0.1,
                )

    def _extend_bert_position_embeddings(self, new_max_len: int):
        """Extend BertGenerationDecoder positional embeddings if new_max_len exceeds
        the model's current max_position_embeddings. New positions are initialized by
        tiling the existing pretrained position embeddings.
        """
        embeddings = self.decoder.bert.embeddings
        old_pe: nn.Embedding = embeddings.position_embeddings
        old_max, hidden = old_pe.weight.shape
        if new_max_len <= old_max:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f'R{rank}. Extending BertDec position embeddings from {old_max} to {new_max_len}')

        new_pe = nn.Embedding(new_max_len, hidden)
        new_pe.to(device=old_pe.weight.device, dtype=old_pe.weight.dtype)
        with torch.no_grad():
            reps = (new_max_len + old_max - 1) // old_max
            tiled = old_pe.weight.repeat(reps, 1)[:new_max_len]
            new_pe.weight.copy_(tiled)
        embeddings.position_embeddings = new_pe

        embeddings.register_buffer(
            'position_ids', torch.arange(new_max_len).expand((1, -1)), persistent=False,
        )
        self.decoder.config.max_position_embeddings = new_max_len

    def _adapt_emb_exp_weight(self, state_dict: Dict[str, Any]) -> None:
        """Reconcile a checkpoint's `emb_exp.weight` with the current emb_exp_rate.

        `emb_exp` is Linear(d_model, emb_exp_rate * d_dec, bias=False), so its weight
        has shape (emb_exp_rate * d_dec, d_model) laid out as `emb_exp_rate` blocks of
        `d_dec` rows (one block per expansion copy, matching the later
        view(batch, emb_exp_rate, d_dec)). When a checkpoint was trained with a
        different rate we keep the shared d_dec block size and:
          - truncate the extra blocks when the new rate is smaller, or
          - repeat the existing blocks cyclically when the new rate is larger.
        Edits `state_dict` in place; no-op if emb_exp is absent or already matches.
        """
        key = 'emb_exp.weight'
        if self.emb_exp is None or key not in state_dict:
            return
        src = state_dict[key]
        tgt = self.emb_exp.weight
        if src.shape == tgt.shape:
            return
        d_dec = self.d_dec
        # Only the expansion-rate (row) dimension may differ; d_model and d_dec are fixed.
        if src.shape[1] != tgt.shape[1] or src.shape[0] % d_dec != 0 or tgt.shape[0] % d_dec != 0:
            return
        src_rate = src.shape[0] // d_dec
        tgt_rate = tgt.shape[0] // d_dec
        src_blocks = src.reshape(src_rate, d_dec, src.shape[1])
        block_idx = [i % src_rate for i in range(tgt_rate)]
        new_blocks = src_blocks[block_idx]  # truncates (tgt<src) or repeats cyclically (tgt>src)
        adapted = new_blocks.reshape(tgt_rate * d_dec, src.shape[1]).to(dtype=src.dtype)
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f'R{rank}. Adapted emb_exp.weight from rate {src_rate} to {tgt_rate} '
              f'({"truncated" if tgt_rate <= src_rate else "repeated"}).')
        state_dict[key] = adapted

    def load_pretrained(self, checkpoint: Optional[Dict[str, Any]] = None):
        # Older checkpoints were saved with the BertModel pooler enabled
        # (enc.bert_model.pooler.dense.{weight,bias}). EncoderBert now constructs
        # BertModel with add_pooling_layer=False, so those keys are unexpected and
        # must be stripped from any state-dict before strict loading.
        def _is_stale_pooler_key(k: str) -> bool:
            return 'bert_model.pooler.' in k

        if checkpoint is not None:
            checkpt_dict = checkpoint['model']
            cleaned_dict = {}
            for key, val in checkpt_dict.items():
                if key.startswith('module.'):
                    key = key[7:]
                if self.decoder_only and (key.startswith('enc.') or key.startswith('emb_exp.') or key.startswith('enc_proj.')):
                    continue
                if _is_stale_pooler_key(key):
                    continue
                cleaned_dict[key] = val
            print(f'Load {len(cleaned_dict)}')
            self._adapt_emb_exp_weight(cleaned_dict)
            # self.load_state_dict(cleaned_dict, strict=not self.decoder_only)
            self.load_state_dict(cleaned_dict, strict=False)
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
                    if self.decoder_only and (key.startswith('enc.') or key.startswith('emb_exp.') or key.startswith('enc_proj.')):
                        continue
                    if _is_stale_pooler_key(key):
                        continue
                    cleaned_dict[key] = val

                self._adapt_emb_exp_weight(cleaned_dict)
                # self.load_state_dict(cleaned_dict, strict=not self.decoder_only)
                self.load_state_dict(cleaned_dict, strict=False)
            elif pretrained_encdec_model_path and pretrained_encdec_model_path.exists():
                if self.decoder_only:
                    print(f'R{rank}. decoder_only=True: skipping encoder load from {pretrained_encdec_model_path}')
                    return
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
                        if _is_stale_pooler_key(new_key):
                            continue
                        enc_checkpt_dict[new_key] = val

                self.enc.load_state_dict(enc_checkpt_dict, strict=False)
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
            if out_enc_pooler is None:
                raise ValueError(
                    'enc_bert.emb_type=pooler requires the BertModel pooler, but it was '
                    'disabled (add_pooling_layer=False in EncoderBert). Re-enable the pooler '
                    'or switch emb_type to cls.'
                )
            return out_enc_pooler
        else:
            raise ValueError(f'Encoder BERT embedding type {self.cfg.enc_bert.emb_type} is not supported')

    def build_decoder_input(
            self, ctx_embs: Tensor, prompt_toks: Tensor, prompt_att_mask: Tensor,
            target_toks: Tensor, target_att_mask: Tensor, include_prompt: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, int]:
        """Build the concatenated decoder input sequence.

        Layout (prompt_first=False, use_sep=True):
            [CtxEmb_1, ..., CtxEmb_N, SEP_emb, PromptTokEmb_1, ..., PromptTokEmb_P, TargetTokEmb_1, ..., TargetTokEmb_{T-1}]

        Layout (prompt_first=False, use_sep=False):
            [CtxEmb_1, ..., CtxEmb_N, PromptTokEmb_1, ..., PromptTokEmb_P, TargetTokEmb_1, ..., TargetTokEmb_{T-1}]

        Layout (prompt_first=True, use_sep=True):
            [PromptTokEmb_1, ..., PromptTokEmb_P, SEP_emb, CtxEmb_1, ..., CtxEmb_N, TargetTokEmb_1, ..., TargetTokEmb_{T-1}]

        Layout (prompt_first=True, use_sep=False):
            [PromptTokEmb_1, ..., PromptTokEmb_P, CtxEmb_1, ..., CtxEmb_N, TargetTokEmb_1, ..., TargetTokEmb_{T-1}]

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

        # Context-embedding block.
        ctx_embs_part = [ctx_embs]
        ctx_mask_part = [torch.ones((batch_size, n_ctx), dtype=torch.long, device=device)]
        ctx_prefix_len = n_ctx

        # Optional SEP delimiter; always sits between the context and prompt blocks.
        sep_embs_part: list = []
        sep_mask_part: list = []
        sep_prefix_len = 0
        if self.cfg.use_sep:
            sep_tok = torch.full((batch_size, 1), self.sep_token_id, dtype=torch.long, device=device)
            sep_emb = self.word_embeddings(sep_tok)  # (batch_size, 1, d_dec)
            sep_embs_part.append(sep_emb)
            sep_mask_part.append(torch.ones((batch_size, 1), dtype=torch.long, device=device))
            sep_prefix_len = 1

        # Prompt token embeddings.
        prompt_embs = self.word_embeddings(prompt_toks)  # (batch_size, prompt_len, d_dec)
        prompt_len = prompt_toks.shape[1]
        prompt_embs_part: list = []
        prompt_mask_part: list = []
        prompt_prefix_len = 0
        if include_prompt:
            prompt_embs_part.append(prompt_embs)
            prompt_mask_part.append(prompt_att_mask)
            prompt_prefix_len = prompt_len

        # Order the prompt and context blocks (SEP stays between them); target is appended last.
        if self.cfg.prompt_first:
            parts_embs = prompt_embs_part + sep_embs_part + ctx_embs_part
            parts_mask = prompt_mask_part + sep_mask_part + ctx_mask_part
        else:
            parts_embs = ctx_embs_part + sep_embs_part + prompt_embs_part
            parts_mask = ctx_mask_part + sep_mask_part + prompt_mask_part
        prefix_len = ctx_prefix_len + sep_prefix_len + prompt_prefix_len

        # Target token embeddings (shifted: input is target[:-1], labels are target)
        target_inp_toks = target_toks[:, :-1]  # (batch_size, target_len - 1)
        target_inp_embs = self.word_embeddings(target_inp_toks)  # (batch_size, target_len - 1, d_dec)
        target_inp_mask = target_att_mask[:, :-1]  # (batch_size, target_len - 1)
        parts_embs.append(target_inp_embs)
        parts_mask.append(target_inp_mask)

        target_start_idx = prefix_len - 1  # Target starts right after the prefix (ctx + sep + prompt, in either order). The first target token corresponds to the prediction at this position.


        # Deterministically truncate the target if the combined sequence would
        # exceed max_seq_len. We truncate the *target* (not the context/prompt)
        # because (a) ctx_embs and prompt_toks are needed for conditioning and
        # (b) target_toks comes from variable-length real text and is the only
        # piece that can blow past max_seq_len in practice. The truncation is
        # purely a function of tensor shapes, so it is identical on every rank
        # and does not break DDP/FSDP gradient sync. Without this, an
        # over-length sample crashes one rank, the others hang in NCCL waiting
        # for it, and `mp.spawn` eventually SIGTERMs the whole job.
        max_target_inp_len = self.cfg.max_seq_len - prefix_len
        if max_target_inp_len < 1:
            # Pathological case: even the prefix exceeds max_seq_len. Nothing
            # sensible to train on; surface a clear error.
            raise RuntimeError(
                f'Decoder prefix length {prefix_len} (ctx={n_ctx}, sep={int(self.cfg.use_sep)}, '
                f'prompt={prompt_len}) already meets or exceeds max_seq_len={self.cfg.max_seq_len}; '
                f'no room for any target tokens.'
            )
        # target_inp = target_toks[:, :-1]; we need target_inp_len <= max_target_inp_len,
        # which means target_toks length <= max_target_inp_len + 1.
        max_target_full_len = max_target_inp_len + 1
        if target_toks.shape[1] > max_target_full_len:
            target_toks = target_toks[:, :max_target_full_len]
            target_att_mask = target_att_mask[:, :max_target_full_len]
            # Recompute target inputs after truncation.
            target_inp_toks = target_toks[:, :-1]
            target_inp_embs = self.word_embeddings(target_inp_toks)
            target_inp_mask = target_att_mask[:, :-1]
            parts_embs[-1] = target_inp_embs
            parts_mask[-1] = target_inp_mask

        target_inp_len = target_inp_toks.shape[1]
        total_len = prefix_len + target_inp_len

        assert total_len <= self.cfg.max_seq_len, \
            f'Total sequence length {total_len} exceeds max_seq_len={self.cfg.max_seq_len} ' \
            f'after truncation (this is a bug)'

        # Concatenate everything
        input_embs = torch.cat(parts_embs, dim=1)  # (batch_size, total_len, d_dec)
        attention_mask = torch.cat(parts_mask, dim=1)  # (batch_size, total_len)

        # Add positional embeddings (skipped for decoders with built-in positional encoding, e.g. RoPE in Qwen).
        if self.pos_emb is not None:
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
        elif self.cfg.decoder_type == MixedDecoderType.Qwen:
            out = self.decoder(
                inputs_embeds=input_embs, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
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
        loss = F.cross_entropy(
            logits_flat, labels_flat, ignore_index=-100,
            label_smoothing=self.cfg.train_cfg.label_smoothing,
        )
        return {'loss': loss}

    def _build_ctx_tokens_decoder_only(
            self, ctx_chunks_toks: Tensor, ctx_chunks_att_mask: Tensor, chunk_counts: list,
    ) -> Tuple[list, list]:
        """For decoder-only mode: per-sample, concatenate raw chunk content tokens
        (strip per-chunk leading CLS / trailing SEP via attention mask) into a single
        1-D tensor of decoder-vocab IDs.

        Chunk tokens originate from the data loader using the encoder tokenizer
        (``tkz_enc``). When ``tkz_enc`` and ``tkz_dec`` differ we cannot feed those
        IDs to the decoder embedding layer; we decode the per-chunk content text
        with ``tkz_enc`` and re-tokenize it with ``tkz_dec``.
        """
        chunk_offsets = [0]
        for c in chunk_counts:
            chunk_offsets.append(chunk_offsets[-1] + c)

        cls_id = self.tkz_enc.cls_token_id
        sep_id = self.tkz_enc.sep_token_id
        device = ctx_chunks_toks.device
        same_vocab = self.tkz_enc is self.tkz_dec or len(self.tkz_enc) == len(self.tkz_dec) and \
            self.tkz_enc.get_vocab() == self.tkz_dec.get_vocab()

        ctx_tok_ids_list = []
        ctx_lens = []
        for i in range(len(chunk_counts)):
            start, end = chunk_offsets[i], chunk_offsets[i + 1]
            parts: list = []
            for ci in range(start, end):
                n_valid = int(ctx_chunks_att_mask[ci].sum().item())
                toks = ctx_chunks_toks[ci, :n_valid]
                # Strip leading CLS and trailing SEP if present (encoder vocab).
                if toks.numel() > 0 and cls_id is not None and toks[0].item() == cls_id:
                    toks = toks[1:]
                if toks.numel() > 0 and sep_id is not None and toks[-1].item() == sep_id:
                    toks = toks[:-1]
                if toks.numel() == 0:
                    continue
                if same_vocab:
                    parts.append(toks)
                else:
                    text = self.tkz_enc.decode(toks.tolist(), skip_special_tokens=True)
                    dec_ids = self.tkz_dec(text, add_special_tokens=False).input_ids
                    if len(dec_ids) > 0:
                        parts.append(torch.tensor(dec_ids, dtype=torch.long, device=device))
            if parts:
                ctx_ids = torch.cat(parts, dim=0)
            else:
                ctx_ids = torch.empty((0,), dtype=torch.long, device=device)
            ctx_tok_ids_list.append(ctx_ids)
            ctx_lens.append(int(ctx_ids.numel()))
        return ctx_tok_ids_list, ctx_lens

    def _decoder_only_forward(
            self,
            ctx_tok_ids_list: list, ctx_lens: list,
            prompt_toks: Tensor, prompt_lengths: list,
            target_toks: Tensor, target_lengths: list,
            masked_target_toks: Optional[Tensor] = None,
            orig_target_toks: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """Decoder-only forward: build per-sample sequence
        [ctx_tok_embs, sep?, prompt_embs, target_inp_embs], add positional
        embeddings, run decoder, compute loss on target positions.

        For mask-aware loss, the answer/target region must be a contiguous slice
        and start aligned across the batch in the logits tensor; this helper does
        not support mask-aware loss in that case (falls back to standard CE).
        """
        batch_size = prompt_toks.shape[0]
        device = prompt_toks.device
        sep_len = 1 if self.cfg.use_sep else 0

        # Per-sample lengths
        total_lens = [
            ctx_lens[i] + sep_len + prompt_lengths[i] + max(target_lengths[i] - 1, 0)
            for i in range(batch_size)
        ]
        max_total_len = max(total_lens) if total_lens else 0
        assert max_total_len <= self.cfg.max_seq_len, \
            f'Total sequence length {max_total_len} exceeds max_seq_len={self.cfg.max_seq_len}'

        # Pre-compute all embeddings we can in batched form
        prompt_embs_all = self.word_embeddings(prompt_toks)  # (B, max_prompt, d_dec)
        emb_dtype = prompt_embs_all.dtype
        target_inp_embs_all = self.word_embeddings(target_toks[:, :-1]) if target_toks.shape[1] > 1 \
            else torch.zeros((batch_size, 0, self.d_dec), device=device, dtype=emb_dtype)
        if self.cfg.use_sep:
            sep_emb = self.word_embeddings(
                torch.full((1, 1), self.sep_token_id, dtype=torch.long, device=device)
            )

        input_embs = torch.zeros((batch_size, max_total_len, self.d_dec), device=device, dtype=emb_dtype)
        attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_total_len), -100, dtype=torch.long, device=device)

        for i in range(batch_size):
            cl = ctx_lens[i]
            pl = prompt_lengths[i]
            ctx_embs_i = self.word_embeddings(ctx_tok_ids_list[i].unsqueeze(0))[0] if cl > 0 else None

            pos = 0
            if self.cfg.prompt_first:
                if pl > 0:
                    input_embs[i, pos:pos + pl] = prompt_embs_all[i, :pl]
                    attention_mask[i, pos:pos + pl] = 1
                    pos += pl
                if self.cfg.use_sep:
                    input_embs[i, pos:pos + 1] = sep_emb[0]
                    attention_mask[i, pos:pos + 1] = 1
                    pos += 1
                if cl > 0:
                    input_embs[i, pos:pos + cl] = ctx_embs_i
                    attention_mask[i, pos:pos + cl] = 1
                    pos += cl
            else:
                if cl > 0:
                    input_embs[i, pos:pos + cl] = ctx_embs_i
                    attention_mask[i, pos:pos + cl] = 1
                    pos += cl
                if self.cfg.use_sep:
                    input_embs[i, pos:pos + 1] = sep_emb[0]
                    attention_mask[i, pos:pos + 1] = 1
                    pos += 1
                if pl > 0:
                    input_embs[i, pos:pos + pl] = prompt_embs_all[i, :pl]
                    attention_mask[i, pos:pos + pl] = 1
                    pos += pl

            target_start_i = pos - 1  # last prefix pos predicts first target token

            til = max(target_lengths[i] - 1, 0)
            if til > 0:
                input_embs[i, pos:pos + til] = target_inp_embs_all[i, :til]
                attention_mask[i, pos:pos + til] = 1
            pos += til

            tl = target_lengths[i]
            labels[i, target_start_i:target_start_i + tl] = target_toks[i, :tl]

        if self.pos_emb is not None:
            pos_ids = torch.arange(max_total_len, device=device).unsqueeze(0)
            input_embs = input_embs + self.pos_emb(pos_ids)

        logits = self.run_decoder(input_embs, attention_mask)

        # Mask-aware loss requires a fixed target_start across the batch. In
        # decoder-only mode per-sample target_start varies, so fall back to
        # standard CE in that case.
        loss_dict = self.calc_loss(logits, labels)
        return loss_dict, logits

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

        assert torch.all(batch.inp_toks[:, 0] == self.tkz_enc.cls_token_id), 'Input tokens must start with encoder CLS token'

        if self.decoder_only:
            # Treat each sample's inp_toks as a single chunk (already shape (B, inp_len)).
            ctx_tok_ids_list, ctx_lens = self._build_ctx_tokens_decoder_only(
                batch.inp_toks, batch.inp_att_mask, [1] * batch_size,
            )

            if self.cfg.prompt_all:
                # prompt_all target is the same content as the encoder input but in decoder vocab.
                target_toks = batch.inp_toks_dec
                target_att_mask = batch.inp_dec_att_mask
            else:
                target_toks = batch.cites_toks
                target_att_mask = batch.cites_att_mask
            target_lengths = [int(target_att_mask[i].sum().item()) for i in range(batch_size)]

            return self._decoder_only_forward(
                ctx_tok_ids_list, ctx_lens,
                batch.prompts_toks, [int(batch.prompts_att_mask[i].sum().item()) for i in range(batch_size)],
                target_toks, target_lengths,
            )

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

        if self.interactive_extractor is not None:
            # Build the raw windowed context (B, N, d_model), then let the extractor
            # expand each embedding into ie_exp_rate slots that VISIT the prompt.
            if win_indices is not None:
                ctx_embs_raw = inp_enc_embs[win_indices]  # (batch_size, win_size, d_model)
            else:
                ctx_embs_raw = inp_enc_embs.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, batch_size, d_model)
            prompt_embs = self.word_embeddings(batch.prompts_toks)
            prompt_pad = batch.prompts_att_mask == 0
            ctx_embs = self.interactive_extractor(ctx_embs_raw, prompt_embs, prompt_pad)
        elif self.cfg.emb_exp_rate > 0:
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
            # Target is the whole input chunk content with tags, in decoder vocab.
            target_toks = batch.inp_toks_dec  # (batch_size, inp_dec_len)
            target_att_mask = batch.inp_dec_att_mask  # (batch_size, inp_dec_len)
        else:
            # Target is just the citation between tags (cites_toks) - decoder vocab
            target_toks = batch.cites_toks  # (batch_size, cite_len)
            target_att_mask = batch.cites_att_mask  # (batch_size, cite_len)

        # Build decoder input. When the InteractiveExtractor already consumed the
        # prompt and ie_prompt_in_stream is False, omit the prompt from the causal
        # stream so the answer must flow through the extracted soft tokens.
        include_prompt = self.interactive_extractor is None or self.cfg.ie_prompt_in_stream
        input_embs, attention_mask, labels, target_start_idx = self.build_decoder_input(
            ctx_embs, batch.prompts_toks, batch.prompts_att_mask,
            target_toks, target_att_mask, include_prompt=include_prompt,
        )

        # Run decoder
        logits = self.run_decoder(input_embs, attention_mask)

        # Compute loss
        if self.mask_loss_fn is not None:
            if self.cfg.prompt_all:
                masked_target_toks = batch.inp_masked_toks_dec
                orig_target_toks = batch.inp_toks_dec
            else:
                masked_target_toks = batch.cites_masked_toks  # (batch_size, cite_len)
                orig_target_toks = batch.cites_toks  # (batch_size, cite_len)
            loss_dict = self.calc_loss(logits, labels, masked_target_toks, orig_target_toks, target_start_idx)
        else:
            loss_dict = self.calc_loss(logits, labels)

        return loss_dict, logits

    def run_on_qna(self, batch: QnaBatch, epoch: int = -1) -> Tuple[Dict[str, Tensor], Tensor]:
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
            batch: QnaBatch with context chunks, prompts, and answer tokens.
            epoch: Current epoch number.

        Returns:
            Tuple of (loss_dict, logits).
        """
        batch_size = len(batch.ctx_chunk_counts)
        device = batch.ctx_chunks_toks.device

        if self.decoder_only:
            ctx_tok_ids_list, ctx_lens = self._build_ctx_tokens_decoder_only(
                batch.ctx_chunks_toks, batch.ctx_chunks_att_mask, batch.ctx_chunk_counts,
            )
            ans_lens = [int(batch.ans_att_mask[i].sum().item()) for i in range(batch_size)]
            return self._decoder_only_forward(
                ctx_tok_ids_list, ctx_lens,
                batch.prompt_toks, list(batch.prompt_lengths),
                batch.ans_toks, ans_lens,
            )

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
        ctx_embs_raw = torch.zeros((batch_size, target_win_size, self.cfg.d_model), device=device, dtype=all_enc_embs.dtype)
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

                filler_embs = all_enc_embs[filler_inds] if filler_inds.numel() > 0 else torch.zeros((n_filler, self.cfg.d_model), device=device, dtype=all_enc_embs.dtype)

                off = torch.randint(0, target_win_size - n_own + 1, (1,)).item()
                ctx_embs_raw[i, off:off + n_own] = own[:n_own]
                ctx_embs_raw[i, :off] = filler_embs[:off]
                if off + n_own < target_win_size:
                    ctx_embs_raw[i, off + n_own:] = filler_embs[off:n_filler]

        # 4. Apply InteractiveExtractor, emb_exp expansion, or projection
        if self.interactive_extractor is not None:
            # ctx_embs_raw: (batch_size, target_win_size, d_model)
            prompt_embs = self.word_embeddings(batch.prompt_toks)
            max_pl = batch.prompt_toks.shape[1]
            prompt_lens_t = torch.as_tensor(batch.prompt_lengths, device=device).unsqueeze(1)
            prompt_pad = torch.arange(max_pl, device=device).unsqueeze(0) >= prompt_lens_t  # (bs, max_pl)
            ctx_embs = self.interactive_extractor(ctx_embs_raw, prompt_embs, prompt_pad)
        elif self.cfg.emb_exp_rate > 0:
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

        # When the InteractiveExtractor already consumed the prompt and
        # ie_prompt_in_stream is False, omit the prompt from the causal stream.
        prompt_in_stream = self.interactive_extractor is None or self.cfg.ie_prompt_in_stream
        stream_prompt_lens = prompt_lens if prompt_in_stream else [0] * batch_size

        # Compute per-sample total length and max
        total_lens = [n_ctx + sep_len + stream_prompt_lens[i] + max(ans_lens[i] - 1, 0) for i in range(batch_size)]
        max_total_len = max(total_lens)
        assert max_total_len <= self.cfg.max_seq_len, \
            f'Total sequence length {max_total_len} exceeds max_seq_len={self.cfg.max_seq_len}'

        input_embs = torch.zeros((batch_size, max_total_len, self.d_dec), device=device, dtype=prompt_embs_all.dtype)
        attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_total_len), -100, dtype=torch.long, device=device)

        for i in range(batch_size):
            pl = stream_prompt_lens[i]  # prompt length in stream (0 when IE omits the prompt)
            pos = 0
            if self.cfg.prompt_first:
                # Prompt tokens (actual length, no padding); optionally omitted from the stream
                if pl > 0:
                    input_embs[i, pos:pos + pl] = prompt_embs_all[i, :pl]
                    attention_mask[i, pos:pos + pl] = 1
                    pos += pl
                # Optional SEP
                if self.cfg.use_sep:
                    input_embs[i, pos:pos + 1] = sep_emb[0]
                    attention_mask[i, pos:pos + 1] = 1
                    pos += 1
                # Context embeddings
                input_embs[i, pos:pos + n_ctx] = ctx_embs[i]
                attention_mask[i, pos:pos + n_ctx] = 1
                pos += n_ctx
            else:
                # Context embeddings
                input_embs[i, pos:pos + n_ctx] = ctx_embs[i]
                attention_mask[i, pos:pos + n_ctx] = 1
                pos += n_ctx
                # Optional SEP
                if self.cfg.use_sep:
                    input_embs[i, pos:pos + 1] = sep_emb[0]
                    attention_mask[i, pos:pos + 1] = 1
                    pos += 1
                # Prompt tokens (actual length, no padding); optionally omitted from the stream
                if pl > 0:
                    input_embs[i, pos:pos + pl] = prompt_embs_all[i, :pl]
                    attention_mask[i, pos:pos + pl] = 1
                    pos += pl

            # The last prefix position predicts the first answer token.
            target_start_i = pos - 1

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
        if self.pos_emb is not None:
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

        if self.decoder_only:
            ctx_tok_ids_list, ctx_lens = self._build_ctx_tokens_decoder_only(
                batch.ctx_chunks_toks, batch.ctx_chunks_att_mask, batch.ctx_chunk_counts,
            )
            target_lens = [int(batch.target_att_mask[i].sum().item()) for i in range(batch_size)]
            return self._decoder_only_forward(
                ctx_tok_ids_list, ctx_lens,
                batch.prompt_toks, list(batch.prompt_lengths),
                batch.target_toks, target_lens,
            )

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

        ctx_embs_raw = torch.zeros((batch_size, max_win, self.cfg.d_model), device=device, dtype=all_enc_embs.dtype)
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

        input_embs = torch.zeros((batch_size, max_total_len, self.d_dec), device=device, dtype=prompt_embs_all.dtype)
        attention_mask = torch.zeros((batch_size, max_total_len), dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_total_len), -100, dtype=torch.long, device=device)

        for i in range(batch_size):
            pl = prompt_lens[i]
            pos = 0
            if self.cfg.prompt_first:
                # Prompt tokens
                input_embs[i, pos:pos + pl] = prompt_embs_all[i, :pl]
                attention_mask[i, pos:pos + pl] = 1
                pos += pl
                # Optional SEP
                if self.cfg.use_sep:
                    input_embs[i, pos:pos + 1] = sep_emb[0]
                    attention_mask[i, pos:pos + 1] = 1
                    pos += 1
                # Context embeddings
                input_embs[i, pos:pos + n_ctx] = ctx_embs[i]
                attention_mask[i, pos:pos + n_ctx] = 1
                pos += n_ctx
            else:
                # Context embeddings
                input_embs[i, pos:pos + n_ctx] = ctx_embs[i]
                attention_mask[i, pos:pos + n_ctx] = 1
                pos += n_ctx
                # Optional SEP
                if self.cfg.use_sep:
                    input_embs[i, pos:pos + 1] = sep_emb[0]
                    attention_mask[i, pos:pos + 1] = 1
                    pos += 1
                # Prompt tokens
                input_embs[i, pos:pos + pl] = prompt_embs_all[i, :pl]
                attention_mask[i, pos:pos + pl] = 1
                pos += pl

            target_start_i = pos - 1  # last prefix pos predicts first target token

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
        if self.pos_emb is not None:
            pos_ids = torch.arange(max_total_len, device=device).unsqueeze(0)
            input_embs = input_embs + self.pos_emb(pos_ids)

        # 5. Run decoder and compute loss
        logits = self.run_decoder(input_embs, attention_mask)
        loss_dict = self.calc_loss(logits, labels)

        return loss_dict, logits

    def forward(self, batch: Union[MaskedCiteBatch, QnaBatch, NextTokBatch], epoch: int = -1) -> Tuple[Dict[str, Tensor], Tensor]:
        # Dispatch by batch class so compound (multi-dataset) training routes each
        # batch to the correct run_* regardless of the configured train_ds_types.
        if isinstance(batch, QnaBatch) or hasattr(batch, 'ans_toks'):
            return self.run_on_qna(batch, epoch)
        if isinstance(batch, NextTokBatch):
            return self.run_on_next(batch, epoch)
        # MaskedCiteBatch and synthetic-extraction batches.
        return self.run_on_text_citation(batch, epoch)

