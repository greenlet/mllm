import dataclasses
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml
from torch import nn
from transformers import BertTokenizer

from mllm.config.configuration_bert_at2_generation import BertAt2GenerationConfig
from mllm.config.model import GenmixTrainDsType
from mllm.model.at2_decoder import BertGenerationAt2Decoder, BertGenerationEncoder
from mllm.model.encoder_at2_decoder_bert import EncoderAt2DecoderModel, EncoderAt2DecoderConfig
from mllm.utils.utils import coalesce, bool_to_str


@dataclass
class Genat2Cfg:
    inp_len: int
    pretrained_model_name: str
    max_inp_chunks: int
    max_out_toks: int
    bert: EncoderAt2DecoderConfig

    file_name = 'genat2_model_cfg.yaml'

    @classmethod
    def create(
            cls, inp_len: int = 128, pretrained_model_name: str = 'bert-base-uncased', max_inp_chunks: int = 10, max_out_toks: int = 50,
            enc_at2_enabled: bool = True, dec_at2_enabled: bool = True, last_dec_to_all_enc_at2_enabled: bool = True,
    ) -> 'Genat2Cfg':
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

        return Genat2Cfg(
            inp_len=inp_len, pretrained_model_name=pretrained_model_name, max_inp_chunks=max_inp_chunks,
            max_out_toks=max_out_toks, bert=bert_cfg,
        )

    @classmethod
    def copy_override(
            cls, cfg: Union['Genat2Cfg', Path], inp_len: int = 0, pretrained_model_name: str = '', max_inp_chunks: int = 0, max_out_toks: int = 0,
            enc_at2_enabled: Optional[bool] = None, dec_at2_enabled: Optional[bool] = None, last_dec_to_all_enc_at2_enabled: Optional[bool] = None,
    ) -> 'Genat2Cfg':
        if not isinstance(cfg, Genat2Cfg):
            fpath = Path(cfg)
            if fpath.is_dir():
                fpath /= cls.file_name
            cfg = cls.load_from_yaml(cfg)

        inp_len = inp_len or cfg.inp_len
        pretrained_model_name = pretrained_model_name or cfg.pretrained_model_name
        max_inp_chunks = max_inp_chunks or cfg.max_inp_chunks
        max_out_toks = max_out_toks or cfg.max_out_toks
        bert_cfg = EncoderAt2DecoderConfig.from_dict(deepcopy(cfg.bert.to_dict()))
        dec_cfg: BertAt2GenerationConfig = bert_cfg.decoder
        dec_cfg.enc_at2_enabled = coalesce(enc_at2_enabled, dec_cfg.enc_at2_enabled)
        dec_cfg.dec_at2_enabled = coalesce(dec_at2_enabled, dec_cfg.dec_at2_enabled)
        dec_cfg.last_dec_to_all_enc_at2_enabled = coalesce(last_dec_to_all_enc_at2_enabled, dec_cfg.last_dec_to_all_enc_at2_enabled)

        return Genat2Cfg(
            inp_len=inp_len, pretrained_model_name=pretrained_model_name, max_inp_chunks=max_inp_chunks,
            max_out_toks=max_out_toks, bert=bert_cfg,
        )

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data['bert'] = data['bert'].to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Genat2Cfg':
        data = deepcopy(data)
        data['bert'] = EncoderAt2DecoderConfig.from_dict(data['bert'])
        return cls(**data)

    def save_to_yaml(self, fpath: Path):
        data = self.to_dict()
        with open(fpath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load_from_yaml(cls, fpath: Path) -> 'Genat2Cfg':
        with open(fpath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class Genat2Model(nn.Module):
    cfg: Genat2Cfg
    model: EncoderAt2DecoderModel

    def __init__(self, cfg: Genat2Cfg):
        super().__init__()
        self.cfg = cfg


def gen_prefpostfix_genat2(model_cfg: Genat2Cfg, train_ds_type: Optional[GenmixTrainDsType] = None) -> tuple[str, str]:
    prefix, postfix_parts = f'genat2', []

    bert_str = model_cfg.pretrained_model_name.replace('_', '_')
    postfix_parts.append(bert_str)

    dec_cfg: BertAt2GenerationConfig = model_cfg.bert.decoder
    postfix_parts.append(f'd{dec_cfg.hidden_size}')

    postfix_parts.append(f'inp{model_cfg.inp_len}')

    postfix_parts.append(f'dat2{bool_to_str(dec_cfg.dec_at2_enabled, cap=False)}')
    postfix_parts.append(f'eat2{bool_to_str(dec_cfg.dec_at2_enabled, cap=False)}')
    postfix_parts.append(f'ldeat2{bool_to_str(dec_cfg.last_dec_to_all_enc_at2_enabled, cap=False)}')

    if train_ds_type is not None:
        postfix_parts.append(f'ds_{train_ds_type.value}')

    if model_cfg.max_inp_chunks > 0:
        postfix_parts.append(f'maxi{model_cfg.max_inp_chunks}')

    if model_cfg.max_out_toks > 0:
        postfix_parts.append(f'maxo{model_cfg.max_out_toks}')

    postfix = '-'.join(postfix_parts)
    return prefix, postfix


def run_create_config():
    cfg = Genat2Cfg.create()
    print(cfg)


if __name__ == '__main__':
    run_create_config()


