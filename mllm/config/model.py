import math
from enum import Enum
from pathlib import Path
from typing import TypeVar, Union, Optional

from jsonschema.validators import create
from pydantic import BaseModel
from torchtext.datasets import dataset_module

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


class HgReductType(str, Enum):
    Matmul = 'matmul'
    Decim = 'decim'
    Avg = 'avg'


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


class EncdecHgCfg(BaseModel):
    enc_pyr: EncPyrCfg
    dec_pyr: DecPyrCfg


class DecRankType(str, Enum):
    Simple = 'smp'
    Trans = 'trn'


class DecRankSimpleCfg(BaseModel):
    d_model: int


class DecRankTransCfg(BaseModel):
    n_layers: int
    n_heads: int
    d_k: int
    d_v: int
    d_model: int
    d_inner: int
    inp_len: int
    dropout_rate: float


class RankerHgCfg(BaseModel):
    enc_pyr: EncPyrCfg
    dec_type: DecRankType
    dec_simple: Optional[DecRankSimpleCfg] = None
    dec_trans: Optional[DecRankTransCfg] = None


def create_encdec_hg_cfg(
        n_vocab: int, pad_idx: int, d_model: int = 256, n_heads: int = 8, d_inner: int = 1024, inp_len: int = 256,
        step: int = 2, dropout_rate: float = 0.0, n_similar_layers: int = 1, reduct_type: HgReductType = HgReductType.Matmul,
        enhance_type: HgEnhanceType = HgEnhanceType.Matmul, pos_enc_type: PosEncType = PosEncType.Num,
        ) -> EncdecHgCfg:
    d_word_vec = d_model
    d_k = d_v = d_model // n_heads
    n_layers = math.ceil(math.log(inp_len, step))
    cfg_vocab_enc = VocabEncoderCfg(
        n_vocab=n_vocab, d_word_vec=d_word_vec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
        pos_enc_type=pos_enc_type,
    )
    cfg_enc_pyr = EncPyrCfg(
        vocab_encoder=cfg_vocab_enc, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_inner=d_inner, inp_len=inp_len, step=step, n_layers=n_layers, dropout_rate=dropout_rate,
        n_similar_layers=n_similar_layers, reduct_type=reduct_type,
    )
    cfg_dec_pyr = DecPyrCfg(
        d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_inner=d_inner, inp_len=inp_len, step=step, n_layers=n_layers, dropout_rate=dropout_rate, n_vocab=n_vocab,
        n_similar_layers=n_similar_layers, enhance_type=enhance_type,
    )
    cfg_encdec_hg = EncdecHgCfg(enc_pyr=cfg_enc_pyr, dec_pyr=cfg_dec_pyr)
    return cfg_encdec_hg


def create_ranker_hg_cfg(
        n_vocab: int, pad_idx: int, d_model: int = 256, n_heads: int = 8, d_inner: int = 1024, inp_len: int = 256,
        step: int = 2, dropout_rate: float = 0.0, n_similar_layers: int = 1, reduct_type: HgReductType = HgReductType.Matmul,
        pos_enc_type: PosEncType = PosEncType.Num, dec_type: DecRankType = DecRankType.Simple, dec_n_layers: int = -1, dec_dropout_rate: float = - 1,
        ) -> RankerHgCfg:
    d_word_vec = d_model
    d_k = d_v = d_model // n_heads
    n_layers = math.ceil(math.log(inp_len, step))
    cfg_vocab_enc = VocabEncoderCfg(
        n_vocab=n_vocab, d_word_vec=d_word_vec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
        pos_enc_type=pos_enc_type,
    )
    cfg_enc_pyr = EncPyrCfg(
        vocab_encoder=cfg_vocab_enc, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_inner=d_inner, inp_len=inp_len, step=step, n_layers=n_layers, dropout_rate=dropout_rate,
        n_similar_layers=n_similar_layers, reduct_type=reduct_type,
    )

    cfg_dec_simple: Optional[DecRankSimpleCfg] = None
    cfg_dec_trans: Optional[DecRankTransCfg] = None
    if dec_type == DecRankType.Simple:
        cfg_dec_simple = DecRankSimpleCfg(d_model=d_model)
    elif dec_type == DecRankType.Trans:
        if dec_n_layers < 0:
            dec_n_layers = n_layers
        if dec_dropout_rate < 0:
            dec_dropout_rate = dropout_rate
        cfg_dec_trans = DecRankTransCfg(
            n_layers=dec_n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model,
            d_inner=d_inner, dec_dropout_rate=dec_dropout_rate,
        )
    else:
        raise f'Unsupported Ranker Decoder type: {dec_type}'
    cfg_encdec_hg = RankerHgCfg(enc_pyr=cfg_enc_pyr, dec_type=dec_type, dec_simple=cfg_dec_simple, dec_trans=cfg_dec_trans)
    return cfg_encdec_hg


def copy_override_encdec_hg_cfg(
        cfg: EncdecHgCfg, inp_len: int = 0, n_similar_layers: int = 1, reduct_type: HgReductType = HgReductType.Matmul,
        enhance_type: HgEnhanceType = HgEnhanceType.Matmul, pos_enc_type: PosEncType = PosEncType.Num,
        ) -> EncdecHgCfg:
    n_vocab = cfg.enc_pyr.vocab_encoder.n_vocab
    pad_idx = cfg.enc_pyr.vocab_encoder.pad_idx
    d_model = cfg.enc_pyr.d_model
    n_heads = cfg.enc_pyr.n_heads
    d_inner = cfg.enc_pyr.d_inner
    step = cfg.enc_pyr.step
    dropout_rate = cfg.enc_pyr.dropout_rate

    changed = False
    if 0 < inp_len != cfg.enc_pyr.inp_len:
        assert inp_len & (inp_len - 1) == 0, f'inp_len = {inp_len} is not power of 2'
        changed = True
    else:
        inp_len = cfg.enc_pyr.inp_len

    if n_similar_layers != cfg.enc_pyr.n_similar_layers:
        assert n_similar_layers > 0, f'n_similar_layers = {n_similar_layers}, but must be > 0'
        assert cfg.enc_pyr.n_similar_layers == cfg.dec_pyr.n_similar_layers, \
            f'enc n_similar_layers = {cfg.enc_pyr.n_similar_layers} != dec n_similar_layers = {cfg.dec_pyr.n_similar_layers}'
        changed = True

    if reduct_type != cfg.enc_pyr.reduct_type:
        changed = True

    if enhance_type != cfg.dec_pyr.enhance_type:
        changed = True

    if pos_enc_type != cfg.enc_pyr.vocab_encoder.pos_enc_type:
        changed = True

    if changed:
        return create_encdec_hg_cfg(
            n_vocab=n_vocab, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_inner=d_inner, inp_len=inp_len, step=step,
            dropout_rate=dropout_rate, n_similar_layers=n_similar_layers, reduct_type=reduct_type, enhance_type=enhance_type,
            pos_enc_type=pos_enc_type,
        )
    return cfg


def copy_override_ranker_hg_cfg(
        cfg: RankerHgCfg, inp_len: int = 0, n_similar_layers: int = 1, reduct_type: HgReductType = HgReductType.Matmul,
        pos_enc_type: PosEncType = PosEncType.Num, dec_type: DecRankType = DecRankType.Simple,
        dec_n_layers: int = -1, dec_dropout_rate: float = - 1,
        ) -> RankerHgCfg:
    n_vocab = cfg.enc_pyr.vocab_encoder.n_vocab
    pad_idx = cfg.enc_pyr.vocab_encoder.pad_idx
    d_model = cfg.enc_pyr.d_model
    n_heads = cfg.enc_pyr.n_heads
    d_inner = cfg.enc_pyr.d_inner
    step = cfg.enc_pyr.step
    dropout_rate = cfg.enc_pyr.dropout_rate

    if 0 < inp_len != cfg.enc_pyr.inp_len:
        assert inp_len & (inp_len - 1) == 0, f'inp_len = {inp_len} is not power of 2'
    else:
        inp_len = cfg.enc_pyr.inp_len

    if n_similar_layers != cfg.enc_pyr.n_similar_layers:
        assert n_similar_layers > 0, f'n_similar_layers = {n_similar_layers}, but must be > 0'

    return create_ranker_hg_cfg(
        n_vocab=n_vocab, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_inner=d_inner, inp_len=inp_len, step=step,
        dropout_rate=dropout_rate, n_similar_layers=n_similar_layers, reduct_type=reduct_type,
        pos_enc_type=pos_enc_type, dec_type=dec_type, dec_n_layers=dec_n_layers, dec_dropout_rate=dec_dropout_rate,
    )


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


def gen_prefpostfix_encdec_hg(model_cfg: EncdecHgCfg) -> tuple[str, str]:
    prefix = f'encdechg'
    enc, dec = model_cfg.enc_pyr, model_cfg.dec_pyr
    postfix = (f'inp{enc.inp_len}-pos_{enc.vocab_encoder.pos_enc_type.value}-lrs{enc.n_layers}x{enc.n_similar_layers}-'
               f'rdc_{enc.reduct_type.value}-enh_{dec.enhance_type.value}-step{enc.step}-d{enc.d_model}-h{enc.n_heads}')
    return prefix, postfix


def gen_prefpostfix_ranker_hg(model_cfg: RankerHgCfg) -> tuple[str, str]:
    prefix = f'rankerhg'
    enc = model_cfg.enc_pyr
    postfix = (f'inp{enc.inp_len}-pos_{enc.vocab_encoder.pos_enc_type.value}-lrs{enc.n_layers}x{enc.n_similar_layers}-'
               f'rdc_{enc.reduct_type.value}-step{enc.step}-d{enc.d_model}-h{enc.n_heads}-dec_{model_cfg.dec_type}')
    if model_cfg.dec_type == DecRankType.Trans:
        dec = model_cfg.dec_trans
        postfix = f'{postfix}-dlrs{dec.n_layers}'
    return prefix, postfix

