import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from nbconvert.nbconvertapp import nbconvert_flags
from torch import nn
from transformers import BertModel, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, \
    BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPoolingAndCrossAttentions

from mllm.config.model import GenmixBertCfg, GenmixEmbExpType, GenmixEmbAggType


class GenmixBert(nn.Module):
    cfg: GenmixBertCfg
    device: torch.device
    enc: BertModel
    gen: EncoderDecoderModel
    n_first_embs: int
    n_second_embs: int

    def __init__(self, cfg: GenmixBertCfg, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.tkz = BertTokenizer.from_pretrained(self.cfg.tokenizer_name)
        self.enc = BertModel.from_pretrained(
            self.cfg.pretrained_model_name, torch_dtype=torch.float32, device_map=self.device,
        )
        encoder: BertGenerationEncoder = BertGenerationEncoder.from_pretrained(
            self.cfg.pretrained_model_name, bos_token_id=self.tkz.bos_token_id, eos_token_id=self.tkz.eos_token_id,
            device_map=self.device,
        )
        decoder: BertGenerationDecoder = BertGenerationDecoder.from_pretrained(
            self.cfg.pretrained_model_name, add_cross_attention=True, is_decoder=True,
            bos_token_id=self.tkz.bos_token_id, eos_token_id=self.tkz.eos_token_id, device_map=self.device,
        )
        self.gen = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        if self.cfg.n_first_embs > 0:
            self.n_first_embs = min(self.cfg.n_first_embs, self.cfg.inp_len)
        elif self.cfg.emb_agg_type == GenmixEmbAggType.Avg:
            self.n_first_embs = 1
        else:
            self.n_first_embs = self.cfg.inp_len

        if self.cfg.n_second_embs > 0:
            self.n_second_embs = self.cfg.n_second_embs
        else:
            self.n_second_embs = self.cfg.inp_len

        if self.cfg.emb_agg_type == GenmixEmbAggType.Fst:
            # Leave n_first_embs intact
            pass
        elif self.cfg.emb_agg_type == GenmixEmbAggType.Avg:
            assert self.n_first_embs == 1, f'For embeddings aggregation type {self.cfg.emb_agg_type} n_first_embs (={self.n_first_embs} must be equal to 1)'
            assert self.cfg.emb_exp_type == GenmixEmbExpType.Mat or self.n_second_embs == 1, \
                f'For embeddings aggregation type {self.cfg.emb_agg_type} and expansion type {self.cfg.emb_exp_type} n_second_embs (={self.n_second_embs}) must be equal to 1'
        elif self.cfg.emb_agg_type == GenmixEmbAggType.Mat:
            in_features = self.cfg.inp_len * self.cfg.d_model
            out_features = self.n_first_embs * self.cfg.d_model
            self.emb_agg = nn.Linear(in_features, out_features, bias=False, device=self.device)
        else:
            raise Exception(f'Embedding aggregation type {self.cfg.emb_agg_type} is not supported')

        if self.cfg.emb_exp_type == GenmixEmbExpType.Non:
            pass
        elif self.cfg.emb_exp_type == GenmixEmbExpType.Mat:
            in_features = self.n_first_embs * self.cfg.d_model
            out_features = self.n_second_embs * self.cfg.d_model
            self.emb_exp = nn.Linear(in_features, out_features, bias=False, device=self.device)
        else:
            raise Exception(f'Embedding expansion type {self.cfg.emb_exp_type} is not supported')


    # input_ids: [src_len]
    # target_ids: [tgt_len]
    def run_train(self, input_ids: torch.Tensor, target_ids: torch.Tensor):
        if input_ids[0] != self.tkz.cls_token_id:
            input_ids = F.pad(input_ids, (1, 0), 'constant', self.tkz.cls_token_id)
        # [n_seq * inp_len]
        src_len = len(input_ids)
        pad_size = src_len - src_len // self.cfg.inp_len
        if pad_size > 0:
            input_ids = F.pad(input_ids, (0, pad_size), 'constant', self.tkz.pad_token_id)
        # [n_seq, inp_len]
        input_ids = input_ids.reshape(-1, self.cfg.inp_len)
        input_mask = input_ids != self.tkz.pad_token_id
        enc_out: BaseModelOutputWithPoolingAndCrossAttentions = self.enc(input_ids=input_ids, attention_mask=input_mask)

        # [n_seq, inp_len, d_model]
        emb = enc_out.last_hidden_state
        # [n_seq, d_model]
        emb = emb[:, 0]
        # [1, n_seq, d_model]
        emb = emb.unsqueeze(0)

        if target_ids[0] != self.tkz.cls_token_id:
            target_ids = F.pad(target_ids, (1, 0), 'constant', self.tkz.cls_token_id)
        # [1, tgt_len]
        target_ids = target_ids.unsqueeze(0)

        gen_out: Seq2SeqLMOutput = self.gen(inputs_embeds=emb, decoder_input_ids=target_ids)
        # [1, tgt_len, n_vocab]
        gen_logits = gen_out.logits

        logits = gen_logits.view(-1, self.gen.decoder.config.vocab_size)[:-1]
        labels = target_ids[0][1:]
        loss = F.cross_entropy(logits, labels)
        return loss

    def _to_toks(self, s: str, inp_len: Optional[int] = None) -> torch.Tensor:
        t = self.tkz(s, return_tensors='pt').input_ids.to(self.device)
        assert t[0][0] == self.tkz.cls_token_id and t[0][-1] == self.tkz.sep_token_id
        if inp_len is not None:
            t = t.reshape(-1)
            mod = len(t) % inp_len
            if mod > 0:
                pad_size = inp_len - mod
                t = F.pad(t, (0, pad_size), 'constant', self.tkz.pad_token_id)
            t = t.reshape(-1, inp_len)
        return t

    def context_question_to_emb(self, context: str, question: str) -> torch.Tensor:
        # [n_ctx, inp_len]
        ctx_toks = self._to_toks(context, inp_len=self.cfg.inp_len)
        # [n_qst, inp_len]
        qst_toks = self._to_toks(question, inp_len=self.cfg.inp_len)
        # [n_ctx + n_qst, inp_len]
        cq_inp = torch.concat([ctx_toks, qst_toks])
        # [n_ctx + n_qst, inp_len]
        inp_mask = cq_inp != self.tkz.pad_token_id

        enc_out: BaseModelOutputWithPoolingAndCrossAttentions = self.enc(input_ids=cq_inp, attention_mask=inp_mask)
        # n_cq = n_ctx + n_qst
        # [n_cq, inp_len, d_model]
        emb = enc_out.last_hidden_state

        # # [n_cq d_model]
        # emb = emb[:, 0]
        # # [1, n_cq, d_model]
        # emb = emb.unsqueeze(0)

        if self.cfg.n_first_embs > 0:
            n_first_embs = self.cfg.n_first_embs
        else:
            n_first_embs = emb.shape[1]

        # [n_cq n_first_embs, d_model]
        emb = emb[:, :n_first_embs]

        if self.cfg.emb_agg_type == GenmixEmbAggType.Fst:
            # Just leave n_first_embeddings intact
            pass
        elif self.cfg.emb_agg_type == GenmixEmbAggType.Avg:
            assert self.cfg
            pass
        elif self.cfg.emb_agg_type == GenmixEmbAggType.Mat:
            pass
        else:
            raise Exception(f'Embedding aggregation type {self.cfg.emb_agg_type} is not supported')

        if self.cfg.emb_exp_type == GenmixEmbExpType.Non:
            pass
        elif self.cfg.emb_exp_type == GenmixEmbExpType.Mat:
            pass
        else:
            raise Exception(f'Embedding expansion type {self.cfg.emb_exp_type} is not supported')

        # [n_cq * n_first_embs, d_model]
        emb = emb.reshape((1, -1, self.cfg.d_model))

        return emb

    def prompt_to_emb(self, prompt: str) -> torch.Tensor:
        # [n_prompt, inp_len]
        prompt_toks = self._to_toks(prompt, inp_len=self.cfg.inp_len)
        if self.cfg.max_inp_chunks > 0:
            prompt_toks = prompt_toks[:self.cfg.max_inp_chunks]
        # [n_prompt, inp_len]
        prompt_mask = prompt_toks != self.tkz.pad_token_id

        enc_out: BaseModelOutputWithPoolingAndCrossAttentions = self.enc(
            input_ids=prompt_toks, attention_mask=prompt_mask,
        )
        # [n_prompt, inp_len, d_model]
        emb = enc_out.last_hidden_state

        # # [n_prompt, d_model]
        # emb = emb[:, 0]
        # # [1, n_prompt, d_model]
        # emb = emb.unsqueeze(0)

        n_prompt = emb.shape[0]
        if self.cfg.emb_agg_type == GenmixEmbAggType.Fst:
            # [n_prompt, n_first_embs, d_model]
            emb = emb[:, :self.n_first_embs]
            # [1, n_prompt * n_first_embs, d_model]
            emb = emb.reshape((1, -1, self.cfg.d_model))
        elif self.cfg.emb_agg_type == GenmixEmbAggType.Avg:
            assert self.n_first_embs == 1
            # [n_prompt * 1, d_model]
            emb = torch.mean(emb, dim=1, keepdim=False)
            # [1, n_prompt * 1, d_model]
            emb = emb.unsqueeze(0)
        elif self.cfg.emb_agg_type == GenmixEmbAggType.Mat:
            # [n_prompt, inp_len * d_model]
            emb = emb.reshape((n_prompt, self.cfg.inp_len * self.cfg.d_model))
            # [n_prompt, n_first_embs * d_model]
            emb = self.emb_agg(emb)
            # [1, n_prompt * n_first_embs, d_model]
            emb = emb.reshape((1, n_prompt * self.n_first_embs, self.cfg.d_model))
        else:
            raise

        # emb: [1, n_prompt * n_first_embs, d_model]
        if self.cfg.emb_exp_type == GenmixEmbExpType.Non:
            assert self.n_first_embs == self.n_second_embs, f'n_first_embs (={self.n_first_embs}) != n_second_embs (={self.n_second_embs})'
            pass
        elif self.cfg.emb_exp_type == GenmixEmbExpType.Mat:
            # [1, n_prompt, n_first_embs * d_model]
            emb = emb.reshape((1, n_prompt, self.n_first_embs * self.cfg.d_model))
            # [1, n_prompt, n_second_embs * d_model]
            emb = self.emb_exp(emb)
            # [1, n_prompt, n_second_embs, d_model]
            emb = emb.reshape(1, n_prompt, self.n_second_embs, self.cfg.d_model)
            # [1, n_second_embs, d_model]
            emb = torch.mean(emb, dim=1, keepdim=False)
        else:
            raise

        # emb: [1, n_embs, d_model]
        return emb

    def text_title_to_emb(self, text: str, title: str) -> torch.Tensor:
        prompt = f'Summarize following text. Title: {title}. Text: {text}'
        emb = self.prompt_to_emb(prompt)
        return emb

    def run_on_qna_txt(self, context: str, question: str, answer: str) -> torch.Tensor:
        emb = self.context_question_to_emb(context=context, question=question)

        # [1, n_ans]
        ans_toks = self._to_toks(answer)
        # [n_ans]
        ans_toks = ans_toks[0]
        # tgt_len = n_ans - 1
        # [tgt_len]
        target_ids = ans_toks[:-1]
        if target_ids[0] != self.tkz.cls_token_id:
            target_ids = F.pad(target_ids, (1, 0), 'constant', self.tkz.cls_token_id)
        # [1, tgt_len]
        target_ids = target_ids.unsqueeze(0)

        gen_out: Seq2SeqLMOutput = self.gen(inputs_embeds=emb, decoder_input_ids=target_ids, use_cache=False)
        # [1, tgt_len, n_vocab]
        gen_logits = gen_out.logits

        # [tgt_len, n_vocab]
        logits = gen_logits.view(-1, self.gen.decoder.config.vocab_size)
        # [tgt_len]
        labels = ans_toks[1:]
        # [tgt_len]
        loss = F.cross_entropy(logits, labels, reduction='none')
        # The last one is sep_token_id
        assert loss.shape[0] > 1
        loss_1, loss_2 = loss[:-1].mean(), loss[-1]
        w1, w2 = 50, 1
        loss = (loss_1 * w1 + loss_2 * w2) / (w1 + w2)
        return loss

    def gen_on_qna_txt(self, context: str, question: str) -> torch.Tensor:
        emb = self.context_question_to_emb(context=context, question=question)

        out_toks = self.gen.generate(inputs_embeds=emb, decoder_start_token_id=self.tkz.cls_token_id)
        return out_toks

    def run_on_sum_txt(self, text: str, summary: str, title: str) -> torch.Tensor:
        emb = self.text_title_to_emb(text=text, title=title)

        # [1, n_sum]
        sum_toks = self._to_toks(summary)
        if 0 < self.cfg.max_out_toks < sum_toks.shape[1]:
            sum_toks = sum_toks[:, :self.cfg.max_out_toks]
        # [n_sum]
        sum_toks = sum_toks[0]
        # tgt_len = n_sum - 1
        # [tgt_len]
        target_ids = sum_toks[:-1]
        if target_ids[0] != self.tkz.cls_token_id:
            target_ids = F.pad(target_ids, (1, 0), 'constant', self.tkz.cls_token_id)
        # [1, tgt_len]
        target_ids = target_ids.unsqueeze(0)

        gen_out: Seq2SeqLMOutput = self.gen(inputs_embeds=emb, decoder_input_ids=target_ids, use_cache=False)
        # [1, tgt_len, n_vocab]
        gen_logits = gen_out.logits

        # [tgt_len, n_vocab]
        logits = gen_logits.view(-1, self.gen.decoder.config.vocab_size)
        # [tgt_len]
        labels = sum_toks[1:]
        # [tgt_len]
        loss = F.cross_entropy(logits, labels, reduction='none')
        # The last one is sep_token_id
        assert loss.shape[0] > 1
        if labels[-1] == self.tkz.sep_token_id:
            loss_1, loss_2 = loss[:-1].mean(), loss[-1]
            w1, w2 = 50, 1
            loss = (loss_1 * w1 + loss_2 * w2) / (w1 + w2)
        else:
            loss = loss.mean()
        return loss

    def gen_on_sum_txt(self, text: str, title: str, max_len: int = 100) -> torch.Tensor:
        emb = self.text_title_to_emb(text=text, title=title)

        out_toks = self.gen.generate(inputs_embeds=emb, decoder_start_token_id=self.tkz.cls_token_id, max_length=max_len)
        return out_toks

    def run_on_wiki_txt(self, title: str, text: str) -> torch.Tensor:
        prompt = f'Cite the text starting from words "<words>". Text: {text}'
        emb = self.prompt_to_emb(prompt=prompt)

        # [1, n_sum]
        sum_toks = self._to_toks(summary)
        if 0 < self.cfg.max_out_toks < sum_toks.shape[1]:
            sum_toks = sum_toks[:, :self.cfg.max_out_toks]
        # [n_sum]
        sum_toks = sum_toks[0]
        # tgt_len = n_sum - 1
        # [tgt_len]
        target_ids = sum_toks[:-1]
        if target_ids[0] != self.tkz.cls_token_id:
            target_ids = F.pad(target_ids, (1, 0), 'constant', self.tkz.cls_token_id)
        # [1, tgt_len]
        target_ids = target_ids.unsqueeze(0)

        gen_out: Seq2SeqLMOutput = self.gen(inputs_embeds=emb, decoder_input_ids=target_ids, use_cache=False)
        # [1, tgt_len, n_vocab]
        gen_logits = gen_out.logits

        # [tgt_len, n_vocab]
        logits = gen_logits.view(-1, self.gen.decoder.config.vocab_size)
        # [tgt_len]
        labels = sum_toks[1:]
        # [tgt_len]
        loss = F.cross_entropy(logits, labels, reduction='none')
        # The last one is sep_token_id
        assert loss.shape[0] > 1
        if labels[-1] == self.tkz.sep_token_id:
            loss_1, loss_2 = loss[:-1].mean(), loss[-1]
            w1, w2 = 50, 1
            loss = (loss_1 * w1 + loss_2 * w2) / (w1 + w2)
        else:
            loss = loss.mean()
        return loss


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

