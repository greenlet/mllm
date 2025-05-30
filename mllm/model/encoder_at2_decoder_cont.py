import os
from typing import Optional

from pydantic import BaseModel
from torch import nn

from mllm.model.encoder_at2_decoder_bert import EncoderAt2DecoderModel, EncoderAt2DecoderConfig


class EncAt2DecCfg(BaseModel):
    inp_len: int
    pretrained_model_name: str
    max_inp_chunks: int
    max_out_toks: int
    bert: EncoderAt2DecoderConfig

    @classmethod
    def create(
            cls, inp_len: int = 128, pretrained_model_name: str = 'bert-base-uncased', max_inp_chunks: int = 10, max_out_toks: int = 50,
            bert_cfg: Optional[EncoderAt2DecoderModel] = None,
    ) -> 'EncAt2Dec':
        pass




class EncAt2Dec(nn.Module):
    model: EncoderAt2DecoderModel

    def __init__(self):
        super().__init__()


