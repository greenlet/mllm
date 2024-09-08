from pathlib import Path
from typing import TypeVar, Union

from pydantic import BaseModel


T = TypeVar('T')
MS = Union[T, tuple[T]]


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


class MllmRankerCfg(BaseModel):
    vocab_encoder: VocabEncoderCfg
    encoders: list[EncoderCfg]
    decoders: list[EncoderCfg]


class EmbDecoderCfg(BaseModel):
    d_emb: int
    n_layers: int
    n_heads: int
    d_hid: int
    seq_len: int
    dp_rate: float


class MllmEncdecCfg(BaseModel):
    vocab_encoder: VocabEncoderCfg
    encoder: EncoderCfg
    decoder: EmbDecoderCfg


def create_mllm_encdec_cfg(
        n_vocab: int, d_word_wec: int = 512, inp_len: int = 1000, dropout_rate: float = 0.1,
        enc_n_layers: int = 3, n_heads: int = 8, d_model: int = 512,
        d_inner: int = 2048, enc_with_graph_mat: bool = False, enc_with_emb_mat: bool = False,
        dec_n_layers: int = 3, pad_idx: int = 0,
) -> MllmEncdecCfg:
    cfg_vocab_enc = VocabEncoderCfg(
        n_vocab=n_vocab, d_word_vec=d_word_wec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
    )
    assert d_model % n_heads == 0
    d_k = d_v = d_model // n_heads
    cfg_enc = EncoderCfg(
        n_layers=enc_n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
        with_graph_mat=enc_with_graph_mat, inp_len=inp_len, dropout_rate=dropout_rate, with_emb_mat=enc_with_emb_mat,
    )
    cfg_dec = EmbDecoderCfg(
        d_emb=d_model, n_layers=dec_n_layers, n_heads=n_heads, d_hid=d_inner,
        seq_len=inp_len, dp_rate=dropout_rate,
    )
    cfg_mllm_encdec = MllmEncdecCfg(
        vocab_encoder=cfg_vocab_enc, encoder=cfg_enc, decoder=cfg_dec,
    )

    return cfg_mllm_encdec


def create_mllm_ranker_cfg(
        n_vocab: int, inp_len: int = 1000, d_word_wec: int = 512, dropout_rate: float = 0.1,
        n_levels: int = 2,
        enc_n_layers: MS[int] = (3, 2), n_heads: int = 8, d_k: int = 64, d_v: int = 64, d_model: int = 512,
        d_inner: int = 2048, enc_with_graph_mat: bool = False, enc_with_emb_mat: bool = False,
        dec_n_layers: MS[int] = 1, pad_idx: int = 0,
) -> MllmRankerCfg:
    if not isinstance(enc_n_layers, tuple):
        enc_n_layers = tuple(enc_n_layers for _ in range(n_levels))
    assert len(enc_n_layers) == n_levels
    if not isinstance(dec_n_layers, tuple):
        dec_n_layers = tuple(dec_n_layers for _ in range(n_levels))
    assert len(dec_n_layers) == n_levels

    cfg_vocab_enc = VocabEncoderCfg(
        n_vocab=n_vocab, d_word_vec=d_word_wec, d_model=d_model, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
    )
    cfgs_enc = []
    for n_layers in enc_n_layers:
        cfg_enc = EncoderCfg(
            n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
            with_graph_mat=enc_with_graph_mat, inp_len=inp_len, dropout_rate=dropout_rate, with_emb_mat=enc_with_emb_mat,
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

