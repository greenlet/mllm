from pathlib import Path
from typing import TypeVar, Union

from pydantic import BaseModel, Field
from torch import nn



class EncdecExpCfg(BaseModel):
    train_dir_path: Path


class ExpCfg(BaseModel):
    version: str = '0.0.1'
    description: str = ''
    ds_dir_path: Path
    train_dir_path: Path


class ExpState(BaseModel):
    last_epoch: int
    val_loss_min: float



class ModelEncdecEncoderConfig(BaseModel):
    pass


class ModelEncdecDecoderConfig(BaseModel):
    pass


class ModelEncdecConfig(BaseModel):
    pass


class ModelRankerEncoderConfig(BaseModel):
    pass


class ModelRankerDecoderConfig(BaseModel):
    pass


class ModelRankerConfig(BaseModel):
    pass


class ModelEncdecEncoder(nn.Module):
    cfg: ModelEncdecEncoderConfig

    def __init__(self, cfg: ModelEncdecEncoderConfig):
        super().__init__()
        self.cfg = cfg


class ModelEncdecDecoder(nn.Module):
    cfg: ModelEncdecDecoderConfig

    def __init__(self, cfg: ModelEncdecDecoderConfig):
        super().__init__()
        self.cfg = cfg


class ModelEncdec(nn.Module):
    cfg: ModelEncdecConfig
    encoder: ModelEncdecEncoder
    decoder: ModelEncdecDecoder

    def __init__(self, cfg: ModelEncdecConfig):
        super().__init__()
        self.cfg = cfg


class ModelRankerEncoder(nn.Module):
    pass


class ModelRankerDecoder(nn.Module):
    pass


class ModelRanker(nn.Module):
    pass


T = TypeVar('T')
MS = Union[T, tuple[T]]


class CfgVocabEncoder(BaseModel):
    n_vocab: int
    d_word_vec: int
    d_model: int
    pad_idx: int
    inp_len: int
    dropout_rate: float


class CfgEncoder(BaseModel):
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


class CfgMllmRanker(BaseModel):
    vocab_encoder: CfgVocabEncoder
    encoders: list[CfgEncoder]
    decoders: list[CfgEncoder]


class CfgEmbDecoder(BaseModel):
    d_emb: int
    n_layers: int
    n_heads: int
    d_hid: int
    seq_len: int
    dp_rate: float


class CfgMllmEncdec(BaseModel):
    vocab_encoder: CfgVocabEncoder
    encoder: CfgEncoder
    decoder: CfgEmbDecoder


def create_mllm_encdec_cfg(
        n_vocab: int, d_word_wec: int = 512, inp_len: int = 1000, dropout_rate: float = 0.1,
        enc_n_layers: int = 3, n_heads: int = 8, d_model: int = 512,
        d_inner: int = 2048, enc_with_graph_mat: bool = False, enc_with_emb_mat: bool = False,
        dec_n_layers: int = 3, pad_idx: int = 0,
) -> CfgMllmEncdec:
    cfg_vocab_enc = CfgVocabEncoder(
        n_vocab=n_vocab, d_word_vec=d_word_wec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
    )
    assert d_model % n_heads == 0
    d_k = d_v = d_model // n_heads
    cfg_enc = CfgEncoder(
        n_layers=enc_n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
        with_graph_mat=enc_with_graph_mat, inp_len=inp_len, dropout_rate=dropout_rate, with_emb_mat=enc_with_emb_mat,
    )
    cfg_dec = CfgEmbDecoder(
        d_emb=d_model, n_layers=dec_n_layers, n_heads=n_heads, d_hid=d_inner,
        seq_len=inp_len, dp_rate=dropout_rate,
    )
    cfg_mllm_encdec = CfgMllmEncdec(
        vocab_encoder=cfg_vocab_enc, encoder=cfg_enc, decoder=cfg_dec,
    )

    return cfg_mllm_encdec


def create_mllm_ranker_cfg(
        n_vocab: int, d_word_wec: int = 512, inp_len: int = 1000, dropout_rate: float = 0.1,
        n_levels: int = 2,
        enc_n_layers: MS[int] = (3, 2), n_heads: int = 8, d_k: int = 64, d_v: int = 64, d_model: int = 512,
        d_inner: int = 2048, enc_with_graph_mat: bool = False, enc_with_emb_mat: bool = False,
        dec_n_layers: MS[int] = 1, pad_idx: int = 0,
) -> CfgMllmRanker:
    if not isinstance(enc_n_layers, tuple):
        enc_n_layers = tuple(enc_n_layers for _ in range(n_levels))
    assert len(enc_n_layers) == n_levels
    if not isinstance(dec_n_layers, tuple):
        dec_n_layers = tuple(dec_n_layers for _ in range(n_levels))
    assert len(dec_n_layers) == n_levels

    cfg_vocab_enc = CfgVocabEncoder(
        n_vocab=n_vocab, d_word_vec=d_word_wec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
    )
    cfgs_enc = []
    for n_layers in enc_n_layers:
        cfg_enc = CfgEncoder(
            n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
            with_graph_mat=enc_with_graph_mat, inp_len=inp_len, dropout_rate=dropout_rate, with_emb_mat=enc_with_emb_mat,
        )
        cfgs_enc.append(cfg_enc)

    cfgs_dec = []
    for n_layers in dec_n_layers:
        cfg_dec = CfgEncoder(
            n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
            with_graph_mat=False, inp_len=0, dropout_rate=dropout_rate, with_emb_mat=False,
        )
        cfgs_dec.append(cfg_dec)

    cfg_mllm_ranker = CfgMllmRanker(
        vocab_encoder=cfg_vocab_enc, encoders=cfgs_enc, decoders=cfgs_dec,
    )

    return cfg_mllm_ranker
