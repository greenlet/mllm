import math
import os
from enum import Enum
from pathlib import Path
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from click.termui import hidden_prompt_func
from torch import nn
from torch.onnx.symbolic_opset12 import dropout
from transformers import BatchEncoding, GenerationConfig, GPT2LMHeadModel, GPT2Tokenizer
# from transformers import BertModel, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPoolingAndCrossAttentions, \
    CausalLMOutputWithCrossAttentions

from mllm.config.model import GenmixBertCfg, GenmixEmbExpType, GenmixEmbAggType, GenmixembCfg, TokensAggType, \
    EncPyrCfg, VocabEncoderCfg, PosEncType, HgReductType, BertAggType, CtxQuePromptType, EncoderConvCfg
from mllm.data.itsquadv2 import QnaBatchV2
from mllm.model.at2_decoder import BertGenerationEmbeddings
from mllm.model.bert import BertModel, BertTokenizer
from mllm.model.bert_generation import BertGenerationEncoder, BertGenerationDecoder
from mllm.model.encdec_ranker_hg import EncoderPyramid
from mllm.model.encoder_conv import EncoderConv
from mllm.model.encoder_decoder import EncoderDecoderModel
from mllm.model.utils import get_top_vects
from mllm.train.utils import WordToks
from mllm.data.wiki.itwiki import WikiBatch


class CtxQuePlaceholder(str, Enum):
    Ctx = 'ctx'
    Que = 'que'


CtxQuePromptTemplateType = list[Union[torch.Tensor, CtxQuePromptType]]


class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Genmixemb(nn.Module):
    cfg: GenmixembCfg
    device: torch.device
    agg: nn.Module
    gen: EncoderDecoderModel
    ctx_que_prompt_templates: list[CtxQuePromptTemplateType]

    def __init__(self, cfg: GenmixembCfg, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        if self.cfg.is_bert:
            self.tkz = BertTokenizer.from_pretrained(self.cfg.model_name)
        else:
            self.tkz = GPT2Tokenizer.from_pretrained(self.cfg.model_name)
            # self.tkz.pad_token_id = -100
            self.tkz.cls_token_id = self.tkz.eos_token_id
            self.tkz.pad_token_id = self.tkz.eos_token_id

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

        # str -> [n_batch, n_toks]
        def tokenize(s: str) -> torch.Tensor:
            toks = self.tkz(s, add_special_tokens=False, return_tensors='pt').input_ids
            return toks.to(self.device)

        tok_pat = [CtxQuePlaceholder.Ctx, tokenize('[SEP]'), CtxQuePlaceholder.Que]
        cq_pat = [tokenize('Context:'), CtxQuePlaceholder.Ctx, tokenize('Question:'), CtxQuePlaceholder.Que]
        qc_pat = [tokenize('Question:'), CtxQuePlaceholder.Ctx, tokenize('Context:'), CtxQuePlaceholder.Que]

        if self.cfg.ctx_que_prompt_type == CtxQuePromptType.Tok:
            self.ctx_que_prompt_templates = [tok_pat]
        elif self.cfg.ctx_que_prompt_type == CtxQuePromptType.Cq:
            self.ctx_que_prompt_templates = [cq_pat]
        elif self.cfg.ctx_que_prompt_type == CtxQuePromptType.Qc:
            self.ctx_que_prompt_templates = [qc_pat]
        elif self.cfg.ctx_que_prompt_type == CtxQuePromptType.Cqqc:
            self.ctx_que_prompt_templates = [cq_pat, qc_pat]
        else:
            raise Exception(f'Context-Query prompt type {self.cfg.ctx_que_prompt_type} is not supported.')

        if self.cfg.is_bert:
            encoder: BertGenerationEncoder = BertGenerationEncoder.from_pretrained(
                self.cfg.model_name, bos_token_id=self.tkz.bos_token_id, eos_token_id=self.tkz.eos_token_id,
                device_map=self.device,
            )
            decoder: BertGenerationDecoder = BertGenerationDecoder.from_pretrained(
                self.cfg.model_name, add_cross_attention=True, is_decoder=True,
                bos_token_id=self.tkz.bos_token_id, eos_token_id=self.tkz.eos_token_id, device_map=self.device,
            )
            gen_model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        elif self.cfg.is_gpt2:
            gen_model = GPT2LMHeadModel.from_pretrained(self.cfg.model_name, device_map=self.device)
        else:
            raise Exception(f'Model type {self.cfg.model_name} is not supported.')

        self.agg = self.create_agg_model(gen_model)
        self.gen = gen_model

    # ctx_toks: [n_batch, n_ctx]
    # que_toks: [n_batch, n_que]
    def toks_template_to_emb_tensor(self, prompt_template: CtxQuePromptTemplateType, ctx_toks: torch.Tensor, que_toks: torch.Tensor) -> torch.Tensor:

        # if self.cfg.add_token_type_ids:
        #     ctx_emb = ctx_emb + self.tt_embs_0[:, :ctx_emb.shape[1]]
        #     que_emb = que_emb + self.tt_embs_1[:, :que_emb.shape[1]]

        embs = []
        n_batch = ctx_toks.shape[0]
        for pat in prompt_template:
            if isinstance(pat, torch.Tensor):
                # [1, n_toks]
                toks = pat
                # [1, n_toks, d_model]
                emb = self.gen.encoder.embeddings.word_embeddings(toks)
                # [n_batch, n_toks, d_model]
                emb = emb.repeat((n_batch, 1, 1))
                if self.cfg.add_token_type_ids:
                    emb = emb + self.tt_embs_1
            elif pat == CtxQuePlaceholder.Ctx:
                # [n_batch, n_ctx, d_model]
                emb = self.run_agg(ctx_toks)
                if self.cfg.add_token_type_ids:
                    emb = emb + self.tt_embs_0
            elif pat == CtxQuePlaceholder.Que:
                # [n_batch, n_que, d_model]
                emb = self.gen.encoder.embeddings.word_embeddings(que_toks)
                if self.cfg.add_token_type_ids:
                    emb = emb + self.tt_embs_1
            else:
                raise Exception(f'Prompt must consist of either tokens or placeholders, got {prompt_template}.')
            embs.append(emb)
        # [n_batch, n_ctx + n_que + n_template_toks, d_model]
        embs = torch.cat(embs, dim=1)
        return embs

    # ctx_toks: [n_batch, n_ctx]
    # que_toks: [n_batch, n_que]
    def prompt_emb(self, ctx_toks: torch.Tensor, que_toks: torch.Tensor) -> torch.Tensor:
        prompt_template = random.choice(self.ctx_que_prompt_templates)
        emb = self.toks_template_to_emb_tensor(prompt_template, ctx_toks, que_toks)
        return emb

    def create_agg_model(self, gen_model: Union[EncoderDecoderModel, GPT2LMHeadModel]) -> Optional[nn.Module]:
        if not self.need_run_agg:
            return

        if self.cfg.is_bert:
            gen_model: BertGenerationEncoder = gen_model.encoder
            n_vocab = gen_model.config.vocab_size
            n_heads = gen_model.config.num_attention_heads
            dropout_rate = gen_model.config.hidden_dropout_prob
            d_inner = gen_model.config.intermediate_size
            hidden_dropout_prob = gen_model.config.hidden_dropout_prob
            word_embeddings = gen_model.embeddings.word_embeddings
            position_embeddings = gen_model.embeddings.position_embeddings
        elif self.cfg.is_gpt2:
            gen_model: GPT2LMHeadModel = gen_model
            n_vocab = gen_model.config.vocab_size
            n_heads = gen_model.config.n_head
            dropout_rate = gen_model.config.resid_pdrop
            d_inner = gen_model.transformer.h[0].mlp.c_proj.out_features
            hidden_dropout_prob = gen_model.config.embd_pdrop
            word_embeddings = gen_model.transformer.wte
            position_embeddings = gen_model.transformer.wpe
        else:
            raise

        if self.cfg.toks_agg_type == TokensAggType.Bert:
            agg = BertModel.from_pretrained(
                self.cfg.model_name, torch_dtype=torch.float32, device_map=self.device,
            )
            if self.cfg.share_agg_enc_token_embeds:
                agg.embeddings.word_embeddings = word_embeddings
                agg.embeddings.position_embeddings = position_embeddings
            else:
                # agg.embeddings.word_embeddings.load_state_dict(word_embeddings.state_dict(), strict=False)
                # agg.embeddings.position_embeddings.load_state_dict(position_embeddings.state_dict(), strict=False)
                agg.embeddings.word_embeddings = word_embeddings.clone()
                agg.embeddings.position_embeddings = position_embeddings.clone()
        elif self.cfg.toks_agg_type == TokensAggType.Pyramid:
            d_model = self.cfg.d_model
            pad_idx = self.tkz.pad_token_id
            d_word_vec = d_model
            d_k = d_v = d_model // n_heads
            n_layers = self.cfg.pyr_agg_n_levels
            n_similar_layers = self.cfg.pyr_agg_n_layers_per_level
            share_layer_weights = self.cfg.pyr_share_layer_weights
            pos_enc_type = PosEncType.Emb
            inp_len = 512
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
            emb_cfg = AttributeDict(
                vocab_size=n_vocab, hidden_size=d_model, pad_token_id=pad_idx, max_position_embeddings=position_embeddings.num_embeddings,
                layer_norm_eps=1e-12, hidden_dropout_prob=dropout_rate,
            )
            encoder_embeddings = BertGenerationEmbeddings(config=emb_cfg).to(self.device)
            if self.cfg.share_agg_enc_token_embeds:
                encoder_embeddings.word_embeddings = word_embeddings
                encoder_embeddings.position_embeddings = position_embeddings
            else:
                encoder_embeddings.word_embeddings = word_embeddings.clone()
                encoder_embeddings.position_embeddings = position_embeddings.clone()
            agg = EncoderPyramid(cfg_enc, bert_encoder=encoder_embeddings).to(self.device)
        elif self.cfg.toks_agg_type == TokensAggType.Conv:
            conv_cfg = EncoderConvCfg(
                n_levels=self.cfg.cnv_n_levels, n_layers_per_level=self.cfg.cnv_n_layers_per_level, d_model=self.cfg.d_model,
                conv_kernel_size=self.cfg.cnv_conv_kernel_size, pool_kernel_size=self.cfg.cnv_pool_kernel_size, pool_stride=self.cfg.cnv_pool_stride,
                dropout_rate=hidden_dropout_prob, share_layer_weights=self.cfg.cnv_share_layer_weights,
            )
            agg = EncoderConv(conv_cfg).to(self.device)
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
            else:
                mask_weight = 0.0
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
            # [n_batch, n_seq_new, d_model]
            emb = self.agg(toks)
        elif self.cfg.toks_agg_type == TokensAggType.Conv:
            # [n_batch, n_seq, d_model]
            emb = self.gen.encoder.embeddings(toks)
            # [n_batch, n_seq // steps_sum, d_model]
            emb = self.agg(emb)
        else:
            raise Exception(f'Tokens aggregation type {self.cfg.toks_agg_type} is not supported.')
        # print(f'Agg {self.cfg.toks_agg_type.value}. toks {inp_shape} --> emb {emb.shape}')
        return emb

    @property
    def need_run_agg(self) -> bool:
        return self.cfg.toks_agg_type == TokensAggType.Bert and self.cfg.bert_agg_n_subseq_toks > 0 \
            or self.cfg.toks_agg_type == TokensAggType.Pyramid and self.cfg.pyr_agg_step > 0 and self.cfg.pyr_agg_n_levels > 0 \
            or self.cfg.toks_agg_type == TokensAggType.Conv and self.cfg.cnv_n_levels > 0 and self.cfg.cnv_pool_stride > 0

    def run_on_wiki(self, batch: WikiBatch) -> torch.Tensor:
        # toks: [n_batch, max_len]
        # masked_toks: [n_batch, max_len]
        # mask: [n_batch, max_len]
        # tgt_toks: [n_batch, tgt_len]
        toks, masked_toks, mask, tgt_toks = batch.get_tensors()

        if not self.need_run_agg:
            if self.cfg.is_gpt2:
                max_len = 0
                n_toks_max = self.cfg.max_inp_toks + self.cfg.max_out_toks
                for item in batch.items:
                    cur_len = min(len(item.src_toks), n_toks_max)
                    max_len = max(max_len, cur_len)
                max_len += 1
                n_batch = len(batch.items)
                input_toks = np.full((n_batch, max_len), self.tkz.pad_token_id, dtype=int)
                target_toks = []
                for ib, item in enumerate(batch.items):
                    toks = item.src_toks
                    n_toks = len(toks)
                    if n_toks > max_len:
                        i = np.random.randint(n_toks - max_len + 1)
                        toks = toks[i:i + max_len]
                    input_toks[ib, :len(toks) - 1] = toks[:-1]
                    target_toks.append(torch.from_numpy(toks[1:]).to(self.device))
                input_toks = torch.from_numpy(input_toks).to(self.device)
                att_mask = input_toks != self.tkz.pad_token_id
                gen_out = self.gen(
                    input_ids=input_toks, attention_mask=att_mask,
                )
            else:
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
                input_toks = masked_toks
                att_mask = input_toks != self.tkz.pad_token_id
                gen_out = self.gen(
                    input_ids=input_toks, attention_mask=att_mask, decoder_input_ids=tgt_inp_ids, use_cache=False,
                )
        else:
            if self.training and not self.cfg.train_agg_model:
                with torch.no_grad():
                    # [n_batch, n_chunks, d_model]
                    emb = self.run_agg(toks)
            else:
                # [n_batch, n_chunks, d_model]
                emb = self.run_agg(toks)
            gen_out = self.gen(
                inputs_embeds=emb, decoder_input_ids=tgt_inp_ids, use_cache=False,
            )

        # # [n_batch, tgt_len, n_vocab]
        # if self.cfg.is_bert:
        #     gen_out: Seq2SeqLMOutput = gen_out
        #     logits = gen_out.logits
        # elif self.cfg.is_gpt2:
        #     gen_out: CausalLMOutputWithCrossAttentions = gen_out
        #     logits = gen_out.logits
        # else:
        #     raise

        if self.cfg.is_bert:
            # [n_batch, tgt_len, n_vocab]
            logits = gen_out.logits
            # [n_batch, tgt_len]
            labels = tgt_out_ids
            label_mask = mask[:, :self.cfg.max_out_toks]
            loss = self.calc_loss(logits, labels, label_mask)
        elif self.cfg.is_gpt2:
            loss = torch.zeros(size=(1,), device=self.device)
            for ib, item in enumerate(batch.items):
                tgt_toks = target_toks[ib]
                # [n_tgt]
                n_toks = len(tgt_toks)
                # [n_tgt, n_vocab]
                logits = gen_out.logits[ib][:n_toks]
                # [n_tgt]
                labels = tgt_toks
                item_loss = F.cross_entropy(logits, labels)
                loss = loss + item_loss
            loss = loss / len(batch.items)
        else:
            raise
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
                    # [n_batch, n_toks, d_model]
                    emb = self.prompt_emb(ctx_toks, que_toks)
            else:
                # [n_batch, n_toks, d_model]
                emb = self.prompt_emb(ctx_toks, que_toks)

            gen_out: Seq2SeqLMOutput = self.gen(
                inputs_embeds=emb, decoder_input_ids=tgt_inp_ids, use_cache=False,
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

        gen_cfg = GenerationConfig(
            max_new_tokens=self.cfg.max_out_toks,
            eos_token_id=self.tkz.sep_token_id,
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
            # [n_batch, n_toks, d_model]
            emb = self.prompt_emb(ctx_toks, que_toks)

            out_toks = self.gen.generate(
                inputs_embeds=emb, decoder_start_token_id=self.tkz.cls_token_id, generation_config=gen_cfg,
            )

        return out_toks


def run_encdec_bert_train():
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


def run_generate():
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


def run_gpt2_train():
    import os
    from pathlib import Path

    from datasets import load_dataset
    from regex import T
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

    DATA_PATH = Path(os.path.expandvars('$HOME')) / 'data'

    gpt2_train_path = DATA_PATH / 'gpt2_train'

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'
    # dataset = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(DATA_PATH))
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=str(DATA_PATH))

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

    # Set the EOS token as the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_function(examples):
        inputs =  tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
        inputs['labels'] = inputs['input_ids'].copy()
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(gpt2_train_path),
        # evaluation_strategy='epoch',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(gpt2_train_path),
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    # Train the model
    # trainer.train(resume_from_checkpoint=str(gpt2_train_path))
    trainer.train()

    # save the model and tokenizer explicitly
    model_output_dir = str(gpt2_train_path)

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)




if __name__ == '__main__':
    # run_encdec_bert_train()
    # run_generate()
    run_gpt2_train()

