import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, \
    BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPoolingAndCrossAttentions

from mllm.config.model import GenmixBertCfg


class GenmixBert(nn.Module):
    cfg: GenmixBertCfg
    device: torch.device
    enc: BertModel
    gen: EncoderDecoderModel

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
        assert t[0] == self.tkz.cls_token_id and t[-1] == self.tkz.sep_token_id
        if inp_len is not None:
            div = len(t) % inp_len
            if div > 0:
                t = F.pad(t, (0, div), 'constant', self.tkz.pad_token_id)
            t = t.reshape(-1, inp_len)
        return t

    def run_qna_txt(self, context: str, question: str, answer: str) -> torch.Tensor:
        # [n_ctx, inp_len]
        ctx_toks = self._to_toks(context, inp_len=self.cfg.inp_len)
        # [n_qst, inp_len]
        qst_toks = self._to_toks(question, inp_len=self.cfg.inp_len)
        # [n_ans]
        ans_toks = self._to_toks(answer)
        # [n_ctx + n_qst, inp_len]
        cq_inp = torch.concat([ctx_toks, qst_toks])
        inp_mask = cq_inp != self.tkz.pad_token_id


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

