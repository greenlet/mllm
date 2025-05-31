import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict
from torch import nn
from transformers import BertTokenizer

from mllm.config.configuration_bert_at2_generation import BertAt2GenerationConfig
from mllm.model.at2_decoder import BertGenerationAt2Decoder, BertGenerationEncoder
from mllm.model.encoder_at2_decoder_bert import EncoderAt2DecoderModel, EncoderAt2DecoderConfig
from mllm.utils.utils import coalesce


@dataclass
class EncAt2DecCfg:
    inp_len: int
    pretrained_model_name: str
    max_inp_chunks: int
    max_out_toks: int
    bert: EncoderAt2DecoderConfig

    @classmethod
    def create(
            cls, inp_len: int = 128, pretrained_model_name: str = 'bert-base-uncased', max_inp_chunks: int = 10, max_out_toks: int = 50,
            enc_at2_enabled: bool = True, dec_at2_enabled: bool = True, last_dec_to_all_enc_at2_enabled: bool = True,
    ) -> 'EncAt2DecCfg':
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        print(tokenizer)
        enc_model: BertGenerationEncoder = BertGenerationEncoder.from_pretrained(
            pretrained_model_name, bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id,
        )
        dec_model: BertGenerationAt2Decoder = BertGenerationAt2Decoder.from_pretrained(
            pretrained_model_name, add_cross_attention=True, is_decoder=True, bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id, use_cache=False,
            enc_at2_enabled=enc_at2_enabled, dec_at2_enabled=dec_at2_enabled, last_dec_to_all_enc_at2_enabled=last_dec_to_all_enc_at2_enabled,
        )
        model = EncoderAt2DecoderModel(
            encoder=enc_model, decoder=dec_model,
        )
        model.train()
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        bert_cfg = model.config

        return EncAt2DecCfg(
            inp_len=inp_len, pretrained_model_name=pretrained_model_name, max_inp_chunks=max_inp_chunks,
            max_out_toks=max_out_toks, bert=bert_cfg,
        )

    @classmethod
    def copy_override(
            cls, cfg: 'EncAt2DecCfg', inp_len: int = 0, pretrained_model_name: str = '', max_inp_chunks: int = 0, max_out_toks: int = 0,
            enc_at2_enabled: Optional[bool] = None, dec_at2_enabled: Optional[bool] = None, last_dec_to_all_enc_at2_enabled: Optional[bool] = None,
    ) -> 'EncAt2DecCfg':
        inp_len = inp_len or cfg.inp_len
        pretrained_model_name = pretrained_model_name or cfg.pretrained_model_name
        max_inp_chunks = max_inp_chunks or cfg.max_inp_chunks
        max_out_toks = max_out_toks or cfg.max_out_toks
        bert_cfg = EncoderAt2DecoderConfig.from_dict(deepcopy(cfg.bert.to_dict()))
        dec_cfg: BertAt2GenerationConfig = bert_cfg.decoder
        dec_cfg.enc_at2_enabled = coalesce(enc_at2_enabled, dec_cfg.enc_at2_enabled)
        dec_cfg.dec_at2_enabled = coalesce(dec_at2_enabled, dec_cfg.dec_at2_enabled)
        dec_cfg.last_dec_to_all_enc_at2_enabled = coalesce(last_dec_to_all_enc_at2_enabled, dec_cfg.last_dec_to_all_enc_at2_enabled)

        return EncAt2DecCfg(
            inp_len=inp_len, pretrained_model_name=pretrained_model_name, max_inp_chunks=max_inp_chunks,
            max_out_toks=max_out_toks, bert=bert_cfg,
        )


class EncAt2Dec(nn.Module):
    model: EncoderAt2DecoderModel

    def __init__(self):
        super().__init__()



def run_create_config():
    cfg = EncAt2DecCfg.create()
    print(cfg)


if __name__ == '__main__':
    run_create_config()


