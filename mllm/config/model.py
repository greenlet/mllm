import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar, Union, Optional

import numpy as np
import torch
from jsonschema.validators import create
from pydantic import BaseModel, Field
from torchtext.datasets import dataset_module

from mllm.utils.utils import coalesce
from transformers import BertModel, BertConfig

T = TypeVar('T')
MS = Union[T, tuple[T, ...]]

class PosEncType(str, Enum):
    Num = 'num'
    Emb = 'emb'


def to_tuple(t: MS[T], n: int) -> tuple[T, ...]:
    if isinstance(t, tuple):
        return t
    return tuple(t for _ in range(n))


class CustomToken(BaseModel):
    name: str
    repr: str
    special: bool
    ind: int

    @staticmethod
    def create(name: str, special: bool) -> 'CustomToken':
        return CustomToken(name=name, repr=f'<|{name}|>', special=special, ind=-1)


class TokenizerCfg(BaseModel):
    name: str
    n_tokens_init: int
    model_max_length: int
    custom_tokens: dict[str, CustomToken]


class VocabEncoderCfg(BaseModel):
    n_vocab: int
    d_word_vec: int
    d_model: int
    pad_idx: int
    inp_len: int
    dropout_rate: float
    pos_enc_type: PosEncType = PosEncType.Num


class EncoderCfg(BaseModel):
    n_layers: int
    n_heads: int
    d_k: int
    d_v: int
    d_model: int
    d_inner: int
    pad_idx: int
    with_graph_mat: bool
    inp_len: int
    dropout_rate: float
    with_emb_mat: bool


class EmbDecoderCfg(BaseModel):
    d_emb: int
    n_layers: int
    n_heads: int
    d_hid: int
    seq_len: int
    dp_rate: float


class MllmEncdecCfg(BaseModel):
    vocab_encoder: VocabEncoderCfg
    encoders: list[EncoderCfg]
    decoders: list[EmbDecoderCfg]
    with_vocab_decoder: bool


class MllmRankerCfg(BaseModel):
    vocab_encoder: VocabEncoderCfg
    encoders: list[EncoderCfg]
    decoders: list[EncoderCfg]


def create_mllm_encdec_cfg(
        n_vocab: int, inp_len: int = 1000, d_word_wec: int = 512, dropout_rate: float = 0.1,
        n_levels: int = 2,
        enc_n_layers: MS[int] = (3, 2), n_heads: int = 8, d_model: int = 512,
        d_inner: int = 2048, enc_with_graph_mat: bool = False, enc_with_emb_mat: MS[bool] = False,
        dec_n_layers: MS[int] = 3, pad_idx: int = 0, with_vocab_decoder: bool = True,
) -> MllmEncdecCfg:
    enc_n_layers = to_tuple(enc_n_layers, n_levels)
    assert len(enc_n_layers) == n_levels
    dec_n_layers = to_tuple(dec_n_layers, n_levels)
    assert len(dec_n_layers) == n_levels
    enc_with_emb_mat = to_tuple(enc_with_emb_mat, n_levels)
    assert len(enc_with_emb_mat) == n_levels

    assert d_model % n_heads == 0
    d_k = d_v = d_model // n_heads

    cfg_vocab_enc = VocabEncoderCfg(
        n_vocab=n_vocab, d_word_vec=d_word_wec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
    )
    cfgs_enc = []
    for il, n_layers in enumerate(enc_n_layers):    
        cfg_enc = EncoderCfg(
            n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
            with_graph_mat=enc_with_graph_mat, inp_len=inp_len, dropout_rate=dropout_rate, with_emb_mat=enc_with_emb_mat[il],
        )
        cfgs_enc.append(cfg_enc)

    cfgs_dec = []
    for n_layers in dec_n_layers:
        cfg_dec = EmbDecoderCfg(
            d_emb=d_model, n_layers=n_layers, n_heads=n_heads, d_hid=d_inner,
            seq_len=inp_len, dp_rate=dropout_rate,
        )
        cfgs_dec.append(cfg_dec)

    cfg_mllm_encdec = MllmEncdecCfg(
        vocab_encoder=cfg_vocab_enc, encoders=cfgs_enc, decoders=cfgs_dec, with_vocab_decoder=with_vocab_decoder,
    )

    return cfg_mllm_encdec


def create_mllm_ranker_cfg(
        n_vocab: int, inp_len: int = 1000, d_word_wec: int = 512, dropout_rate: float = 0.1,
        n_levels: int = 2,
        enc_n_layers: MS[int] = (3, 2), n_heads: int = 8, d_model: int = 512,
        d_inner: int = 2048, enc_with_graph_mat: bool = False, enc_with_emb_mat: MS[bool] = False,
        dec_n_layers: MS[int] = 1, pad_idx: int = 0,
) -> MllmRankerCfg:
    enc_n_layers = to_tuple(enc_n_layers, n_levels)
    assert len(enc_n_layers) == n_levels
    dec_n_layers = to_tuple(dec_n_layers, n_levels)
    assert len(dec_n_layers) == n_levels
    enc_with_emb_mat = to_tuple(enc_with_emb_mat, n_levels)
    assert len(enc_with_emb_mat) == n_levels

    assert d_model % n_heads == 0
    d_k = d_v = d_model // n_heads

    cfg_vocab_enc = VocabEncoderCfg(
        n_vocab=n_vocab, d_word_vec=d_word_wec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
    )
    cfgs_enc = []
    for il, n_layers in enumerate(enc_n_layers):
        cfg_enc = EncoderCfg(
            n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
            with_graph_mat=enc_with_graph_mat, inp_len=inp_len, dropout_rate=dropout_rate, with_emb_mat=enc_with_emb_mat[il],
        )
        cfgs_enc.append(cfg_enc)

    cfgs_dec = []
    for n_layers in dec_n_layers:
        cfg_dec = EncoderCfg(
            n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
            with_graph_mat=False, inp_len=inp_len, dropout_rate=dropout_rate, with_emb_mat=False,
        )
        cfgs_dec.append(cfg_dec)

    cfg_mllm_ranker = MllmRankerCfg(
        vocab_encoder=cfg_vocab_enc, encoders=cfgs_enc, decoders=cfgs_dec,
    )

    return cfg_mllm_ranker


def gen_prefpostfix_level(model_cfg: Union[MllmEncdecCfg, MllmRankerCfg], model_level: int) -> tuple[str, str]:
    enc_cfg, dec_cfg = model_cfg.encoders[model_level], model_cfg.decoders[model_level]
    enc_str = f'enc-lrs{enc_cfg.n_layers}-embmat{enc_cfg.with_emb_mat}-d{enc_cfg.d_model}-h{enc_cfg.n_heads}'
    if isinstance(model_cfg, MllmEncdecCfg):
        assert isinstance(dec_cfg, EmbDecoderCfg)
        prefix = 'encdec'
        dec_str = f'dec-lrs{dec_cfg.n_layers}-seqlen{dec_cfg.seq_len}-d{dec_cfg.d_emb}-h{dec_cfg.n_heads}'
        if model_level == 0:
            dec_str = f'{dec_str}-vocdec{model_cfg.with_vocab_decoder}'
    elif isinstance(model_cfg, MllmRankerCfg):
        assert isinstance(dec_cfg, EncoderCfg)
        prefix = 'ranker'
        dec_str = f'dec-lrs{dec_cfg.n_layers}-d{dec_cfg.d_model}-h{dec_cfg.n_heads}'
    else:
        raise Exception(f'Unknown config type: {type(model_cfg)}.', model_cfg)
    prefix = f'{prefix}-lvl{model_level}'
    postfix = f'{enc_str}-{dec_str}'
    return prefix, postfix


class HgReductType(str, Enum):
    Matmul = 'matmul'
    Decim = 'decim'
    Avg = 'avg'
    Sub = 'sub'


class HgEnhanceType(str, Enum):
    Matmul = 'matmul'
    MatmulBegin = 'mmbeg'
    MatmulBeginBias = 'mmbb'


class EncPyrCfg(BaseModel):
    vocab_encoder: VocabEncoderCfg
    pad_idx: int
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_inner: int
    inp_len: int
    step: int
    n_layers: int
    dropout_rate: float
    n_similar_layers: int = 1
    reduct_type: HgReductType = HgReductType.Matmul
    temperature: float = 0


class BertEmbType(str, Enum):
    Cls = 'cls'
    Pooler = 'plr'


class EncBertCfg(BaseModel):
    d_model: int = 0
    pretrained_model_name: str = ''
    tokenizer_name: str = ''
    emb_type: BertEmbType = BertEmbType.Cls


class DecPyrCfg(BaseModel):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_inner: int
    inp_len: int
    step: int
    n_layers: int
    dropout_rate: float
    n_vocab: int
    n_similar_layers: int = 1
    enhance_type: HgEnhanceType = HgEnhanceType.Matmul
    temperature: float = 0


class EncdecHgCfg(BaseModel):
    enc_pyr: EncPyrCfg
    dec_pyr: DecPyrCfg


class EncdecBertCfg(BaseModel):
    enc_bert: EncBertCfg
    dec_pyr: DecPyrCfg


class DecRankHgCfg(BaseModel):
    d_model: int
    mlp_layers: str = ''


class RankerHgCfg(BaseModel):
    enc_pyr: EncPyrCfg
    dec_rank: DecRankHgCfg


class RankerBertCfg(BaseModel):
    enc_bert: EncBertCfg
    dec_rank: DecRankHgCfg


MLP_LAYERS_PAT = re.compile(r'^(?P<size>\d+)(?P<bias>b)?|(?P<act>[a-z]\w+)$')


@dataclass(kw_only=True)
class ParsedMlpLayer:
    size: int = 0
    bias: bool = False
    act: str = ''

def parse_mlp_layers(s: str) -> list[ParsedMlpLayer]:
    res = []
    parts = s.split(',')
    for i, part in enumerate(parts):
        m = MLP_LAYERS_PAT.match(part)
        assert m is not None, f'Cannot parse {i}th part of {parts} with {MLP_LAYERS_PAT} pattern. Original string: {s}'
        size = m.group('size')
        size = int(size) if size else 0
        bias = m.group('bias') is not None
        act = m.group('act') or ''
        pl = ParsedMlpLayer(size=size, bias=bias, act=act)
        res.append(pl)
    return res


def bool_to_char(b: bool) -> str:
    return 't' if b else 'f'


def create_encdec_hg_cfg(
        n_vocab: int, pad_idx: int, d_model: int = 256, n_heads: int = 8, d_inner: int = 1024, inp_len: int = 256,
        step: int = 2, dropout_rate: float = 0.0, n_similar_layers: int = 1, reduct_type: HgReductType = HgReductType.Matmul,
        enhance_type: HgEnhanceType = HgEnhanceType.Matmul, pos_enc_type: PosEncType = PosEncType.Num, dec_n_layers: int = 0,
        temperature: float = 0,
        ) -> EncdecHgCfg:
    d_word_vec = d_model
    d_k = d_v = d_model // n_heads
    n_layers = math.ceil(math.log(inp_len, step))
    cfg_vocab_enc = VocabEncoderCfg(
        n_vocab=n_vocab, d_word_vec=d_word_vec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
        pos_enc_type=pos_enc_type,
    )
    cfg_enc = EncPyrCfg(
        vocab_encoder=cfg_vocab_enc, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_inner=d_inner, inp_len=inp_len, step=step, n_layers=n_layers, dropout_rate=dropout_rate,
        n_similar_layers=n_similar_layers, reduct_type=reduct_type, temperature=temperature,
    )

    assert dec_n_layers >= 0, f'dec_n_layers (={dec_n_layers}) must be >= 0'
    dec_n_layers = dec_n_layers or n_layers
    cfg_dec = DecPyrCfg(
        d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_inner=d_inner, inp_len=inp_len, step=step, n_layers=dec_n_layers, dropout_rate=dropout_rate, n_vocab=n_vocab,
        n_similar_layers=n_similar_layers, enhance_type=enhance_type, temperature=temperature,
    )

    cfg_encdec_hg = EncdecHgCfg(enc_pyr=cfg_enc, dec_pyr=cfg_dec)
    return cfg_encdec_hg


def create_encdec_bert_cfg(
        pretrained_model_name: str = 'bert-base-uncased', tokenizer_name: str = '', emb_type: BertEmbType = BertEmbType.Cls,
        inp_len = 128, dec_enhance_type: HgEnhanceType = HgEnhanceType.Matmul,
        dec_n_layers: int = 7, dec_n_similar_layers: int = 1, dec_dropout_rate: float = 0.0, dec_temperature: float = 0,
):
    model = BertModel.from_pretrained(pretrained_model_name, torch_dtype=torch.float32)
    bert_cfg: BertConfig = model.config
    # BertConfig
    # {
    #     "_name_or_path": "bert-base-uncased",
    #     "architectures": [
    #         "BertForMaskedLM"
    #     ],
    #     "attention_probs_dropout_prob": 0.1,
    #     "classifier_dropout": null,
    #     "gradient_checkpointing": false,
    #     "hidden_act": "gelu",
    #     "hidden_dropout_prob": 0.1,
    #     "hidden_size": 768,
    #     "initializer_range": 0.02,
    #     "intermediate_size": 3072,
    #     "layer_norm_eps": 1e-12,
    #     "max_position_embeddings": 512,
    #     "model_type": "bert",
    #     "num_attention_heads": 12,
    #     "num_hidden_layers": 12,
    #     "pad_token_id": 0,
    #     "position_embedding_type": "absolute",
    #     "transformers_version": "4.42.4",
    #     "type_vocab_size": 2,
    #     "use_cache": true,
    #     "vocab_size": 30522
    # }
    d_model = bert_cfg.hidden_size
    n_heads = bert_cfg.num_attention_heads
    n_vocab = bert_cfg.vocab_size

    tokenizer_name = tokenizer_name or pretrained_model_name
    cfg_enc = EncBertCfg(
        d_model=d_model, pretrained_model_name=pretrained_model_name, tokenizer_name=tokenizer_name,
        emb_type=emb_type,
    )
    step = 2
    if dec_n_layers == 0:
        dec_n_layers = math.ceil(math.log(inp_len, step))
    d_inner = d_model * 4
    d_k = d_v = d_model // n_heads
    assert dec_n_layers > 0, f'n_layers (={dec_n_layers}) must be > 0'
    cfg_dec = DecPyrCfg(
        d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_inner=d_inner, inp_len=inp_len, step=step, n_layers=dec_n_layers, dropout_rate=dec_dropout_rate, n_vocab=n_vocab,
        n_similar_layers=dec_n_similar_layers, enhance_type=dec_enhance_type, temperature=dec_temperature,
    )

    cfg_encdec_bert = EncdecBertCfg(enc_bert=cfg_enc, dec_pyr=cfg_dec)
    return cfg_encdec_bert


def create_ranker_hg_cfg(
        n_vocab: int, pad_idx: int, d_model: int = 256, n_heads: int = 8, d_inner: int = 1024, inp_len: int = 256,
        step: int = 2, dropout_rate: float = 0.0, n_similar_layers: int = 1, reduct_type: HgReductType = HgReductType.Matmul,
        pos_enc_type: PosEncType = PosEncType.Num, dec_mlp_layers: str = '', temperature: float = 0,
        ) -> RankerHgCfg:
    d_word_vec = d_model
    d_k = d_v = d_model // n_heads
    n_layers = math.ceil(math.log(inp_len, step))
    cfg_vocab_enc = VocabEncoderCfg(
        n_vocab=n_vocab, d_word_vec=d_word_vec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
        pos_enc_type=pos_enc_type,
    )
    cfg_enc = EncPyrCfg(
        vocab_encoder=cfg_vocab_enc, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_inner=d_inner, inp_len=inp_len, step=step, n_layers=n_layers, dropout_rate=dropout_rate,
        n_similar_layers=n_similar_layers, reduct_type=reduct_type, temperature=temperature,
    )

    cfg_dec_rank = DecRankHgCfg(
        d_model=d_model, mlp_layers=dec_mlp_layers,
    )

    cfg_ranker_hg = RankerHgCfg(enc_pyr=cfg_enc, dec_rank=cfg_dec_rank)
    return cfg_ranker_hg


def copy_override_encdec_hg_cfg(
        cfg: EncdecHgCfg, inp_len: int = 0, n_similar_layers: int = 1, reduct_type: HgReductType = HgReductType.Matmul,
        enhance_type: HgEnhanceType = HgEnhanceType.Matmul, pos_enc_type: PosEncType = PosEncType.Num, dropout_rate: float = 0.0,
        dec_n_layers: int = 0, temperature: float = -1,
        ) -> EncdecHgCfg:
    n_vocab = cfg.enc_pyr.vocab_encoder.n_vocab
    pad_idx = cfg.enc_pyr.vocab_encoder.pad_idx
    d_model = cfg.enc_pyr.d_model
    n_heads = cfg.enc_pyr.n_heads
    d_inner = cfg.enc_pyr.d_inner
    step = cfg.enc_pyr.step
    if 0 < inp_len != cfg.enc_pyr.inp_len:
        assert inp_len & (inp_len - 1) == 0, f'inp_len = {inp_len} is not power of 2'
    else:
        inp_len = cfg.enc_pyr.inp_len

    if n_similar_layers != cfg.enc_pyr.n_similar_layers:
        assert n_similar_layers > 0, f'n_similar_layers = {n_similar_layers}, but must be > 0'
        assert cfg.enc_pyr.n_similar_layers == cfg.dec_pyr.n_similar_layers, \
            f'enc n_similar_layers = {cfg.enc_pyr.n_similar_layers} != dec n_similar_layers = {cfg.dec_pyr.n_similar_layers}'

    temperature = temperature if temperature >= 0 else cfg.enc_pyr.temperature

    return create_encdec_hg_cfg(
        n_vocab=n_vocab, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_inner=d_inner, inp_len=inp_len, step=step,
        dropout_rate=dropout_rate, n_similar_layers=n_similar_layers, reduct_type=reduct_type, enhance_type=enhance_type,
        pos_enc_type=pos_enc_type, dec_n_layers=dec_n_layers, temperature=temperature,
    )


def copy_override_encdec_bert_cfg(
        cfg: EncdecBertCfg, emb_type: Optional[BertEmbType] = None, inp_len: int = 0, dec_enhance_type: Optional[HgEnhanceType] = None,
        dec_n_layers: int = 0, dec_n_similar_layers: int = 0, dec_dropout_rate: Optional[float] = None,
        dec_temperature: Optional[float] = None,
        ) -> EncdecBertCfg:
    enc = cfg.enc_bert
    dec = cfg.dec_pyr
    emb_type = coalesce(emb_type, cfg.enc_bert.emb_type)
    inp_len = inp_len or dec.inp_len
    dec_enhance_type = coalesce(dec_enhance_type, dec.enhance_type)
    dec_n_layers = dec_n_layers or dec.n_layers
    dec_n_similar_layers = dec_n_similar_layers or dec.n_similar_layers
    dec_dropout_rate = coalesce(dec_dropout_rate, dec.dropout_rate)
    dec_temperature = coalesce(dec_temperature, dec.temperature)

    return create_encdec_bert_cfg(
        pretrained_model_name=enc.pretrained_model_name, tokenizer_name=enc.tokenizer_name, emb_type=emb_type,
        inp_len=inp_len, dec_enhance_type=dec_enhance_type, dec_n_layers=dec_n_layers, dec_n_similar_layers=dec_n_similar_layers,
        dec_dropout_rate=dec_dropout_rate, dec_temperature=dec_temperature,
    )


def copy_override_ranker_hg_cfg(
        cfg: RankerHgCfg, inp_len: int = 0, n_similar_layers: int = 1, reduct_type: HgReductType = HgReductType.Matmul,
        pos_enc_type: PosEncType = PosEncType.Num, dec_mlp_layers: str = '', temperature: float = -1, dropout_rate: float = -1,
        ) -> RankerHgCfg:
    n_vocab = cfg.enc_pyr.vocab_encoder.n_vocab
    pad_idx = cfg.enc_pyr.vocab_encoder.pad_idx
    d_model = cfg.enc_pyr.d_model
    n_heads = cfg.enc_pyr.n_heads
    d_inner = cfg.enc_pyr.d_inner
    step = cfg.enc_pyr.step

    if 0 < inp_len != cfg.enc_pyr.inp_len:
        assert inp_len & (inp_len - 1) == 0, f'inp_len = {inp_len} is not power of 2'
    else:
        inp_len = cfg.enc_pyr.inp_len

    if n_similar_layers != cfg.enc_pyr.n_similar_layers:
        assert n_similar_layers > 0, f'n_similar_layers = {n_similar_layers}, but must be > 0'

    temperature = temperature if temperature >= 0 else cfg.enc_pyr.temperature
    dropout_rate = dropout_rate if dropout_rate >=0 else cfg.enc_pyr.dropout_rate

    return create_ranker_hg_cfg(
        n_vocab=n_vocab, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_inner=d_inner, inp_len=inp_len, step=step,
        dropout_rate=dropout_rate, n_similar_layers=n_similar_layers, reduct_type=reduct_type,
        pos_enc_type=pos_enc_type, dec_mlp_layers=dec_mlp_layers, temperature=temperature,
    )


def gen_prefpostfix_encdec_hg(model_cfg: EncdecHgCfg) -> tuple[str, str]:
    prefix = f'encdechg'
    enc, dec = model_cfg.enc_pyr, model_cfg.dec_pyr
    dp_rate = np.round(enc.dropout_rate, 2)
    if dp_rate < 1e-6:
        dp_rate = 0
    n_layers_str = f'{enc.n_layers}'
    if dec.n_layers != enc.n_layers:
        n_layers_str = f'{n_layers_str}_{dec.n_layers}'
    assert enc.temperature == dec.temperature, f'Encoder\' temperature (={enc.temperature}) != decoder\'s temperature (={dec.temperature})'
    temp = np.round(enc.temperature, 2)
    postfix = (f'inp{enc.inp_len}-pos_{enc.vocab_encoder.pos_enc_type.value}-lrs{n_layers_str}x{enc.n_similar_layers}-'
               f'rdc_{enc.reduct_type.value}-enh_{dec.enhance_type.value}-step{enc.step}-d{enc.d_model}-h{enc.n_heads}-'
               f'dp{dp_rate}-t{temp}')
    return prefix, postfix


def gen_prefpostfix_encdec_bert(model_cfg: EncdecBertCfg) -> tuple[str, str]:
    prefix = f'encdecbert'
    enc, dec = model_cfg.enc_bert, model_cfg.dec_pyr
    dp_rate = np.round(dec.dropout_rate, 2)
    if dp_rate < 1e-6:
        dp_rate = 0
    temp = np.round(dec.temperature, 2)
    brt_str = enc.pretrained_model_name.replace('_', '_')
    if enc.tokenizer_name != enc.pretrained_model_name:
        tkz_name = enc.tokenizer_name.replace('-', '_')
        brt_str = f'{brt_str}-{tkz_name}'
    postfix = (f'{brt_str}-d{enc.d_model}-emb_{enc.emb_type}-inp{dec.inp_len}-lrs{dec.n_layers}x{dec.n_similar_layers}-enh_{dec.enhance_type.value}-step{dec.step}-h{dec.n_heads}-'
               f'dp{dp_rate}-t{temp}')
    return prefix, postfix


def gen_prefpostfix_ranker_hg(model_cfg: RankerHgCfg) -> tuple[str, str]:
    prefix = f'rankerhg'
    enc = model_cfg.enc_pyr
    dec = model_cfg.dec_rank
    dec_mlp_layers = dec.mlp_layers.replace(',', '_')

    dp_rate = np.round(enc.dropout_rate, 2)
    if dp_rate < 1e-6:
        dp_rate = 0

    temp = np.round(enc.temperature, 2)
    temp_round = np.round(temp)
    if temp - temp_round < 0.01:
        temp = int(temp_round)

    postfix = (f'inp{enc.inp_len}-pos_{enc.vocab_encoder.pos_enc_type.value}-lrs{enc.n_layers}x{enc.n_similar_layers}-'
               f'rdc_{enc.reduct_type.value}-step{enc.step}-d{enc.d_model}-h{enc.n_heads}-dp{dp_rate}-t{temp}-dmlp_{dec_mlp_layers}')
    return prefix, postfix

