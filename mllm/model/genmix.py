import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import BertModel, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, \
    BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput

from mllm.config.model import GenmixBertCfg


class GenmixBert(nn.Module):
    cfg: GenmixBertCfg
    enc: BertModel
    gen: EncoderDecoderModel

    def __init__(self, cfg: GenmixBertCfg):
        super().__init__()
        self.cfg = cfg
        self.tkz = BertTokenizer.from_pretrained(self.cfg.tokenizer_name)
        self.enc = BertModel.from_pretrained(self.cfg.pretrained_model_name, torch_dtype=torch.float32)
        encoder: BertGenerationEncoder = BertGenerationEncoder.from_pretrained(
            self.cfg.pretrained_model_name, bos_token_id=self.tkz.bos_token_id, eos_token_id=self.tkz.eos_token_id,
        )
        decoder: BertGenerationDecoder = BertGenerationDecoder.from_pretrained(
            self.cfg.pretrained_model_name, add_cross_attention=True, is_decoder=True,
            bos_token_id=self.tkz.bos_token_id, eos_token_id=self.tkz.eos_token_id,
        )
        self.gen = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    def run_train(self, input_ids: torch.Tensor, labels: torch.Tensor):
        gen_out: Seq2SeqLMOutput = self.gen(input_ids=input_ids, decoder_input_ids=labels)
        return gen_out.loss



if __name__ == '__main__':
    from transformers import BertTokenizer, EncoderDecoderModel

    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased",
                                                                "google-bert/bert-base-uncased")

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    tkz_inp: BatchEncoding = tokenizer(
        "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
        return_tensors="pt",
    )
    input_ids = tkz_inp.input_ids

    tkz_lbl: BatchEncoding = tokenizer(
        "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
        return_tensors="pt",
    )
    print(type(tkz_lbl))
    print(tkz_lbl)
    labels = tkz_lbl.input_ids

    # the forward function automatically creates the correct decoder_input_ids
    out = model(input_ids=input_ids, labels=labels)
    print(type(out))
    print('loss:', out.loss)


