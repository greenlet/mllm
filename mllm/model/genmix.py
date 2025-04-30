import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import BertModel, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer

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
