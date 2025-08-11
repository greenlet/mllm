import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.onnx.symbolic_opset12 import dropout
from transformers import BatchEncoding, GenerationConfig
# from transformers import BertModel, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPoolingAndCrossAttentions

from mllm.config.model import GenmixBertCfg, GenmixEmbExpType, GenmixEmbAggType, GenmixembBertCfg, TokensAggType, \
    EncPyrCfg, VocabEncoderCfg, PosEncType, HgReductType, BertAggType
from mllm.data.itsquadv2 import QnaBatchV2
from mllm.model.at2_decoder import BertGenerationEmbeddings
from mllm.model.bert import BertModel, BertTokenizer
from mllm.model.bert_generation import BertGenerationEncoder, BertGenerationDecoder
from mllm.model.encdec_ranker_hg import EncoderPyramid
from mllm.model.encoder_decoder import EncoderDecoderModel
from mllm.model.utils import get_top_vects
from mllm.train.utils import WordToks
from mllm.data.wiki.itwiki import WikiBatch


class GenmixembBert(nn.Module):
    cfg: GenmixembBertCfg
    device: torch.device
    agg: nn.Module
    gen: EncoderDecoderModel

    def __init__(self, cfg: GenmixembBertCfg, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.tkz = BertTokenizer.from_pretrained(self.cfg.bert_model_name)
        encoder: BertGenerationEncoder = BertGenerationEncoder.from_pretrained(
            self.cfg.bert_model_name, bos_token_id=self.tkz.bos_token_id, eos_token_id=self.tkz.eos_token_id,
            device_map=self.device,
        )
        decoder: BertGenerationDecoder = BertGenerationDecoder.from_pretrained(
            self.cfg.bert_model_name, add_cross_attention=True, is_decoder=True,
            bos_token_id=self.tkz.bos_token_id, eos_token_id=self.tkz.eos_token_id, device_map=self.device,
        )

        if self.cfg.add_token_type_ids:
            tt_embs_0 = torch.randn((1, 1, self.cfg.d_model), device=self.device)
            tt_embs_1 = tt_embs_0.clone()
            nn.init.xavier_uniform(tt_embs_0)
            nn.init.xavier_uniform(tt_embs_1)
            self.tt_embs_0 = nn.Parameter(tt_embs_0)
            self.tt_embs_1 = nn.Parameter(tt_embs_1)
        else:
            self.tt_embs_0 = None
            self.tt_embs_1 = None

        self.agg = self.create_agg_model(encoder)
        self.gen = EncoderDecoderModel(encoder=encoder, decoder=decoder)


    def create_agg_model(self, encoder: BertGenerationEncoder) -> Optional[nn.Module]:
        if not self.need_run_agg:
            return
        if self.cfg.toks_agg_type == TokensAggType.Bert:
            agg = BertModel.from_pretrained(
                self.cfg.bert_model_name, torch_dtype=torch.float32, device_map=self.device,
            )
            if self.cfg.share_agg_enc_token_embeds:
                agg.embeddings.word_embeddings = encoder.embeddings.word_embeddings
                agg.embeddings.position_embeddings = encoder.embeddings.position_embeddings
                agg.embeddings.LayerNorm = encoder.embeddings.LayerNorm
                agg.embeddings.dropout = encoder.embeddings.dropout
            else:
                agg.embeddings.load_state_dict(encoder.embeddings.state_dict(), strict=False)
        elif self.cfg.toks_agg_type == TokensAggType.Pyramid:
            d_model = self.cfg.d_model
            pad_idx = self.tkz.pad_token_id
            bert_cfg = encoder.config
            n_vocab = bert_cfg.vocab_size
            n_heads = bert_cfg.num_attention_heads
            d_word_vec = d_model
            d_k = d_v = d_model // n_heads
            n_layers = self.cfg.pyr_agg_n_levels
            n_similar_layers = self.cfg.pyr_agg_n_layers_per_level
            share_layer_weights = self.cfg.pyr_share_layer_weights
            dropout_rate = bert_cfg.hidden_dropout_prob
            pos_enc_type = PosEncType.Emb
            inp_len = 512
            d_inner = bert_cfg.intermediate_size
            step = self.cfg.pyr_agg_step
            reduct_type = self.cfg.pyr_agg_type
            temperature = 0
            cfg_vocab_enc = VocabEncoderCfg(
                n_vocab=n_vocab, d_word_vec=d_word_vec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len,
                dropout_rate=dropout_rate, pos_enc_type=pos_enc_type,
            )
            cfg_enc = EncPyrCfg(
                vocab_encoder=cfg_vocab_enc, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                d_inner=d_inner, inp_len=inp_len, step=step, n_layers=n_layers, dropout_rate=dropout_rate,
                n_similar_layers=n_similar_layers, reduct_type=reduct_type, temperature=temperature, share_layer_weights=share_layer_weights,
            )
            if self.cfg.share_agg_enc_token_embeds:
                encoder_embeddings = encoder.embeddings
            else:
                encoder_embeddings = BertGenerationEmbeddings(encoder.config).to(self.device)
                encoder_embeddings.load_state_dict(encoder.embeddings.state_dict())
            agg = EncoderPyramid(cfg_enc, bert_encoder=encoder_embeddings).to(self.device)
            for n, p in agg.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p, -0.1, 0.1)
        else:
            raise Exception(f'Tokens aggregation type {self.cfg.toks_agg_type} is not supported.')

        return agg

    # logits: [n_batch, n_seq, n_vocab]
    # labels: [n_batch, n_seq]
    # mask: [n_batch, n_seq]
    def calc_loss(
            self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None, mask_weight: float = 0.5,
    ) -> torch.Tensor:
        if mask is None or not mask.any():
            logits = logits.view(-1, self.gen.decoder.config.vocab_size)
            labels = labels.reshape(-1)
            loss = F.cross_entropy(logits, labels)
        else:
            nmask = ~mask
            mask_logits, mask_labels = logits[mask], labels[mask]
            toks_logits, toks_labels = logits[nmask], labels[nmask]
            mask_loss = torch.zeros(size=(1,), device=self.device)
            if len(mask_logits) > 0:
                mask_loss = F.cross_entropy(mask_logits, mask_labels)
            assert len(toks_logits) > 0
            toks_loss = F.cross_entropy(toks_logits, toks_labels)
            nmask_weight = 1 - mask_weight
            loss = mask_weight * mask_loss + nmask_weight * toks_loss
        return loss

    def prefix_token(self, toks: torch.Tensor, tok_id: int) -> torch.Tensor:
        if toks.ndim == 1:
            has_prefix = toks[0] == tok_id
        elif toks.ndim == 2:
            mask = toks[:, 0] == tok_id
            has_any, has_all = mask.any(), mask.all()
            assert has_any == has_all, (f'Either all starting toks are expected to be equal to {tok_id} or none. '
                                        f'Got partial match instead: {mask}.')
            has_prefix = has_all
        else:
            raise Exception(f'Expected 1 or 2 dimensional tensor, got shape = {toks.shape}.')
        if not has_prefix:
            toks = F.pad(toks, (1, 0), value=tok_id)
        return toks

    # toks: [n_batch, n_seq] -> [n_batch, n_chunks, d_model]
    def run_agg(self, toks: torch.Tensor):
        inp_shape = toks.shape
        if self.cfg.toks_agg_type == TokensAggType.Bert:
            if self.cfg.bert_agg_type == BertAggType.Sep:
                n_subseq = self.cfg.bert_agg_n_subseq_toks
                n_batch, n_seq = toks.shape
                n_seq_mod = n_seq % n_subseq
                if n_seq_mod > 0:
                    pad_size = n_subseq - n_seq_mod
                    toks = F.pad(toks, (0, pad_size), 'constant', self.tkz.pad_token_id)
                    n_seq += pad_size
                # [n_batch * n_chunks, n_subseq]
                toks = toks.reshape((-1, n_subseq))
                # [n_batch * n_chunks, 1 + n_subseq]
                toks = torch.pad(toks, (1, 0), 'constant', self.tkz.cls_token_id)
                # [n_batch * n_chunks, 1 + n_subseq]
                mask = toks != self.tkz.pad_token_id
                out = self.agg(input_ids=toks, attention_mask=mask)
                # [n_batch * n_chunks, 1 + n_subseq, d_model]
                emb = out.last_hidden_state
                # [n_batch * n_chunks, 1, d_model]
                emb = emb[:, :1, :]
                n_chunks = n_seq // n_subseq
                # [n_batch, n_chunks, d_model]
                emb = emb.reshape((n_batch, n_chunks, self.cfg.d_model))
            elif self.cfg.bert_agg_type in (BertAggType.Topcos, BertAggType.Topdot):
                n_subseq = self.cfg.bert_agg_n_subseq_toks
                # [n_batch, n_seq]
                toks = self.prefix_token(toks, self.tkz.cls_token_id)
                n_batch, n_seq = toks.shape
                n_chunks = n_seq // n_subseq
                # [n_batch, n_seq]
                mask = toks != self.tkz.pad_token_id
                out = self.agg(input_ids=toks, attention_mask=mask)
                # [n_batch, n_seq, d_model]
                emb = out.last_hidden_state
                # [n_batch, n_chunks, d_model]
                emb = get_top_vects(emb, n_chunks, calc_cos=self.cfg.bert_agg_type == BertAggType.Topcos)
            else:
                raise Exception(f'Bert aggregation type {self.cfg.bert_agg_type} is not supported.')
        elif self.cfg.toks_agg_type == TokensAggType.Pyramid:
            emb = self.agg(toks)
        else:
            raise Exception(f'Tokens aggregation type {self.cfg.toks_agg_type} is not supported.')
        # print(f'Agg {self.cfg.toks_agg_type.value}. toks {inp_shape} --> emb {emb.shape}')
        return emb

    @property
    def need_run_agg(self) -> bool:
        return self.cfg.toks_agg_type == TokensAggType.Bert and self.cfg.bert_agg_n_subseq_toks > 0 \
            or self.cfg.toks_agg_type == TokensAggType.Pyramid and self.cfg.pyr_agg_step > 0 and self.cfg.pyr_agg_n_levels > 0

    def run_on_wiki(self, batch: WikiBatch) -> torch.Tensor:
        # toks: [n_batch, max_len]
        # masked_toks: [n_batch, max_len]
        # mask: [n_batch, max_len]
        # tgt_toks: [n_batch, tgt_len]
        toks, masked_toks, mask, tgt_toks = batch.get_tensors()
        if tgt_toks is not None:
            # [n_batch, tgt_len]
            target_ids = tgt_toks
        else:
            # [n_batch, tgt_len]
            target_ids = toks[:, :self.cfg.max_out_toks]
        # [n_batch, tgt_len + 1]
        target_ids = self.prefix_token(target_ids, self.tkz.cls_token_id)
        # [n_batch * tgt_len, n_vocab]
        tgt_inp_ids, tgt_out_ids = target_ids[:, :-1], target_ids[:, 1:]

        if not self.need_run_agg:
            att_mask = masked_toks != self.tkz.pad_token_id
            # print(f'input_ids: {masked_toks.shape}. attention_mask: {att_mask.shape}. decoder_input_ids: {tgt_inp_ids.shape}')
            gen_out: Seq2SeqLMOutput = self.gen(
                input_ids=masked_toks, attention_mask=att_mask, decoder_input_ids=tgt_inp_ids, use_cache=False,
            )
        else:
            if self.training and not self.cfg.train_agg_model:
                with torch.no_grad():
                    # [n_batch, n_chunks, d_model]
                    emb = self.run_agg(toks)
            else:
                # [n_batch, n_chunks, d_model]
                emb = self.run_agg(toks)
            gen_out: Seq2SeqLMOutput = self.gen(
                inputs_embeds=emb, decoder_input_ids=tgt_inp_ids, use_cache=False,
            )

        # # [n_batch, tgt_len, n_vocab]
        # logits = gen_out.logits
        # # [n_batch * tgt_len, n_vocab]
        # logits = logits.view(-1, self.gen.decoder.config.vocab_size)
        # # [n_batch * tgt_len]
        # labels = tgt_out_ids.reshape(-1)
        # loss = F.cross_entropy(logits, labels)

        # [n_batch, tgt_len, n_vocab]
        logits = gen_out.logits
        # [n_batch, tgt_len]
        labels = tgt_out_ids
        label_mask = mask[:, :self.cfg.max_out_toks]
        loss = self.calc_loss(logits, labels, label_mask)
        return loss

    # logits [n_batch, tgt_len, n_vocab]
    # labels [n_batch, tgt_len]
    def calc_gen_loss(self, b_logits: torch.Tensor, b_labels: torch.Tensor) -> torch.Tensor:
        n_batch = len(b_logits)
        b_mask = (b_labels != self.tkz.pad_token_id) & (b_labels != self.tkz.sep_token_id)
        b_nmask = ~b_mask
        b_loss = torch.zeros(size=(1,), device=self.device)
        for i in range(n_batch):
            # logits: [tgt_len, n_vocab]
            # labels, mask, nmask: [tgt_len]
            logits, labels, mask, nmask = b_logits[i], b_labels[i], b_mask[i], b_nmask[i]
            mask_logits, mask_labels = logits[mask], labels[mask]
            mask_loss = F.cross_entropy(mask_logits, mask_labels)
            nmask_logits, nmask_labels = logits[nmask], labels[nmask]
            # print(f'{i}. mask_loss: {mask_loss}')
            if len(nmask_logits) > 0:
                nmask_loss = F.cross_entropy(nmask_logits, nmask_labels)
                # print(f'{i}. nmask_loss: {nmask_loss}')
                loss = 0.95 * mask_loss + 0.05 * nmask_loss
            else:
                loss = mask_loss
            b_loss = b_loss + loss
        b_loss = b_loss / n_batch
        return b_loss

    def run_on_qna(self, batch: QnaBatchV2) -> torch.Tensor:
        # ctx_toks: [n_batch, ctx_len]
        # que_toks: [n_batch, que_len]
        # ans_toks: [n_batch, ans_len]
        # cq_toks: [n_batch, cq_len]
        ctx_toks, que_toks, ans_toks, cq_toks = batch.get_tensors()

        # [n_batch, tgt_len]
        target_ids = ans_toks
        # # [n_batch, tgt_len + 1]
        # target_ids = self.prefix_token(target_ids, self.tkz.cls_token_id)
        # [n_batch, tgt_len']
        tgt_inp_ids, tgt_out_ids = target_ids[:, :-1], target_ids[:, 1:]

        if not self.need_run_agg:
            # cq_toks: [n_batch, cq_len]
            att_mask = cq_toks != self.tkz.pad_token_id
            gen_out: Seq2SeqLMOutput = self.gen(
                input_ids=cq_toks, attention_mask=att_mask, decoder_input_ids=tgt_inp_ids, use_cache=False,
            )
        else:
            if self.training and not self.cfg.train_agg_model:
                with torch.no_grad():
                    # [n_batch, n_ctx_chunks, d_model]
                    ctx_emb = self.run_agg(ctx_toks)
            else:
                # [n_batch, n_ctx_chunks, d_model]
                ctx_emb = self.run_agg(ctx_toks)
            # que_toks: [n_batch, que_len]
            que_toks = self.prefix_token(que_toks, self.tkz.sep_token_id)
            # [n_batch, que_len, d_model]
            que_emb = self.gen.encoder.embeddings.word_embeddings(que_toks)

            if self.cfg.add_token_type_ids:
                ctx_emb = ctx_emb + self.tt_embs_0[:, :ctx_emb.shape[1]]
                que_emb = que_emb + self.tt_embs_1[:, :que_emb.shape[1]]

            # [n_batch, n_ctx_chunks + nque_len, d_model]
            emb = torch.cat((ctx_emb, que_emb), dim=-2)

            # [n_batch, ctx_len]
            ctx_att_mask = torch.full((ctx_emb.shape[0], ctx_emb.shape[1]), True, device=self.device)
            # [n_batch, que_len]
            que_att_mask = que_toks != self.tkz.pad_token_id
            att_mask = torch.cat((ctx_att_mask, que_att_mask), dim=-1)

            gen_out: Seq2SeqLMOutput = self.gen(
                inputs_embeds=emb, attention_mask=att_mask, decoder_input_ids=tgt_inp_ids, use_cache=False,
            )

        # [n_batch, tgt_len, n_vocab]
        logits = gen_out.logits
        # [n_batch, tgt_len]
        labels = tgt_out_ids
        loss = self.calc_gen_loss(logits, labels)
        return loss

    # TODO: renovate before use
    def run_on_qna_v2(self, batch: QnaBatchV2) -> torch.Tensor:
        if not self.need_run_agg:
            # ctx_toks: [n_batch, ctx_len]
            # que_toks: [n_batch, que_len]
            # ans_toks: [n_batch, ans_len]
            # cq_toks: [n_batch, cq_len]
            ctx_toks, que_toks, ans_toks, cq_toks = batch.get_tensors()

            # [n_batch, tgt_len]
            target_ids = ans_toks
            # [n_batch, tgt_len + 1]
            target_ids = self.prefix_token(target_ids, self.tkz.cls_token_id)
            # [n_batch, tgt_len]
            tgt_inp_ids, tgt_out_ids = target_ids[:, :-1], target_ids[:, 1:]

            # cq_toks: [n_batch, cq_len]
            att_mask = cq_toks != self.tkz.pad_token_id
            gen_out: Seq2SeqLMOutput = self.gen(
                input_ids=cq_toks, attention_mask=att_mask, decoder_input_ids=tgt_inp_ids, use_cache=False,
            )
            # [n_batch, tgt_len, n_vocab]
            logits = gen_out.logits
            # [n_batch, tgt_len]
            labels = tgt_out_ids
            loss = self.calc_gen_loss(logits, labels)
            return loss

        batch_loss = torch.zeros(size=(1,), device=self.device)
        for item in batch.items:
            # [ctx_len], [que_len], [ans_len]
            ctx_toks, que_toks, ans_toks = item.get_tensors()
            # [1, ctx_len], [1, que_len], [1, ans_len]
            ctx_toks, que_toks, ans_toks = ctx_toks.unsqueeze(0), que_toks.unsqueeze(0), ans_toks.unsqueeze(0)

            # [1, tgt_len]
            target_ids = ans_toks
            # [1, tgt_len + 1]
            target_ids = self.prefix_token(target_ids, self.tkz.cls_token_id)
            # [n_batch, tgt_len]
            tgt_inp_ids, tgt_out_ids = target_ids[:, :-1], target_ids[:, 1:]

            if self.training and not self.cfg.train_agg_model:
                with torch.no_grad():
                    # [1, n_ctx_chunks, d_model]
                    ctx_emb = self.run_agg(ctx_toks)
            else:
                # [1, n_ctx_chunks, d_model]
                ctx_emb = self.run_agg(ctx_toks)
            # [1, que_len]
            que_toks = self.prefix_token(que_toks, self.tkz.sep_token_id)
            # [1, que_len, d_model]
            que_emb = self.gen.encoder.embeddings(que_toks)
            # [1, n_ctx_chunks + nque_len, d_model]
            emb = torch.cat((ctx_emb, que_emb), dim=-2)
            gen_out: Seq2SeqLMOutput = self.gen(
                inputs_embeds=emb, decoder_input_ids=tgt_inp_ids, use_cache=False,
            )

            # [1, tgt_len, n_vocab]
            logits = gen_out.logits
            # [1, tgt_len]
            labels = tgt_out_ids
            loss = self.calc_gen_loss(logits, labels)

            batch_loss = batch_loss + loss

        batch_loss = batch_loss / len(batch.items)
        return batch_loss

    # toks: [max_len]
    def gen_on_wiki(self, toks: torch.Tensor) -> torch.Tensor:
        # [1, max_len]
        if toks.ndim == 1:
            toks = toks.unsqueeze(0)

        # gen_cfg = GenerationConfig(
        #     max_new_tokens=self.cfg.max_out_toks, early_stopping=True, num_beams=5, no_repeat_ngram_size=4,
        # )
        gen_cfg = GenerationConfig(
            max_new_tokens=self.cfg.max_out_toks,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            # temperature=0.6,
        )

        if not self.need_run_agg:
            # [1, max_len]
            att_mask = toks != self.tkz.pad_token_id
            out_toks = self.gen.generate(
                input_ids=toks, attention_mask=att_mask, decoder_start_token_id=self.tkz.cls_token_id, generation_config=gen_cfg,
            )
        else:
            # [n_batch, n_chunks, d_model]
            emb = self.run_agg(toks)
            out_toks = self.gen.generate(
                inputs_embeds=emb, decoder_start_token_id=self.tkz.cls_token_id, generation_config=gen_cfg,
            )
        return out_toks

    # toks: [max_len]
    def gen_on_qna(self, ctx_toks: torch.Tensor, que_toks: torch.Tensor, cq_toks: Optional[torch.Tensor] = None) -> torch.Tensor:
        # [1, max_len]
        if ctx_toks.ndim == 1:
            ctx_toks = ctx_toks.unsqueeze(0)
        if que_toks.ndim == 1:
            que_toks = que_toks.unsqueeze(0)

        gen_cfg = GenerationConfig(
            max_new_tokens=self.cfg.max_out_toks,
            eos_token_id=self.tkz.sep_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            # temperature=0.6,
        )

        if not self.need_run_agg:
            if cq_toks is None:
                cq_toks = []
                if ctx_toks[0, 0] != self.tkz.cls_token_id:
                    cq_toks.append([[self.tkz.cls_token_id]])
                cq_toks.append(ctx_toks)
                if ctx_toks[0, -1] != self.tkz.sep_token_id and que_toks[0, 0] != self.tkz.sep_token_id:
                    cq_toks.append([[self.tkz.sep_token_id]])
                cq_toks.append(que_toks)
                if que_toks[0, -1] != self.tkz.sep_token_id:
                    cq_toks.append([[self.tkz.sep_token_id]])
                cq_toks = torch.concatenate(cq_toks, dim=1)
            elif cq_toks.ndim == 1:
                cq_toks = cq_toks.unsqueeze(0)
            att_mask = cq_toks != self.tkz.pad_token_id
            out_toks = self.gen.generate(
                input_ids=cq_toks, attention_mask=att_mask, decoder_start_token_id=self.tkz.cls_token_id, generation_config=gen_cfg,
            )
        else:
            # [n_batch, n_ctx_chunks, d_model]
            ctx_emb = self.run_agg(ctx_toks)
            que_toks = self.prefix_token(que_toks, self.tkz.sep_token_id)
            # [n_batch, que_len, d_model]
            que_emb = self.gen.encoder.embeddings.word_embeddings(que_toks)

            if self.cfg.add_token_type_ids:
                ctx_emb = ctx_emb + self.tt_embs_0[:, :ctx_emb.shape[1]]
                que_emb = que_emb + self.tt_embs_1[:, :que_emb.shape[1]]

            # [n_batch, n_ctx_chunks + nque_len, d_model]
            emb = torch.cat((ctx_emb, que_emb), dim=-2)

            # [n_batch, ctx_len]
            ctx_att_mask = torch.full((ctx_emb.shape[0], ctx_emb.shape[1]), True, device=self.device)
            # [n_batch, que_len]
            que_att_mask = que_toks != self.tkz.pad_token_id
            att_mask = torch.cat((ctx_att_mask, que_att_mask), dim=-1)

            out_toks = self.gen.generate(
                inputs_embeds=emb, attention_mask=att_mask, decoder_start_token_id=self.tkz.cls_token_id, generation_config=gen_cfg,
            )

        return out_toks


def test_train():
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased",
                                                                "google-bert/bert-base-uncased")
    model.train()
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    tkz_inp: BatchEncoding = tokenizer(
        ("The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
         "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."),
        return_tensors="pt",
    )
    input_ids = tkz_inp.input_ids

    tkz_lbl: BatchEncoding = tokenizer(
        ("the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
         "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris."),
        return_tensors="pt",
    )
    print(type(tkz_lbl))
    print(tkz_lbl)
    decoder_input_ids = tkz_lbl.input_ids

    # the forward function automatically creates the correct decoder_input_ids
    out = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    print(type(out))
    print('loss:', out.loss)


def test_generate():
    from transformers import AutoTokenizer

    # load a fine-tuned seq2seq model and corresponding tokenizer
    model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
    tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

    # let's perform inference on a long piece of text
    ARTICLE_TO_SUMMARIZE = (
        "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )
    input_ids = tokenizer(ARTICLE_TO_SUMMARIZE, return_tensors="pt").input_ids

    # autoregressively generate summary (uses greedy decoding by default)
    generated_ids = model.generate(input_ids)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


if __name__ == '__main__':
    # test_train()
    test_generate()

