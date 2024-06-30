from dataclasses import dataclass
from typing import Optional, Any, Union, TypeVar

import numpy as np
from pydantic import BaseModel
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    temperature: float
    inp_len: int
    dropout_rate: float
    dropout: nn.Module

    def __init__(self, temperature: float, inp_len: int = 0,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.inp_len = inp_len
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, q, k, v, mask=None):
        attn = q / self.temperature
        attn = torch.matmul(attn, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    n_head: int
    d_model: int
    d_k: int
    d_v: int
    with_graph_mat: bool
    inp_len: int
    dropout_rate: float

    Q: Optional[nn.Parameter] = None
    K: Optional[nn.Parameter] = None
    V: Optional[nn.Parameter] = None

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int,
                 with_graph_mat: bool = False, inp_len: int = 0, dropout_rate: float = 0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.with_graph_mat = with_graph_mat
        self.inp_len = inp_len
        self.dropout_rate = dropout_rate

        if self.with_graph_mat:
            assert self.inp_len > 0
            self.Q = nn.Parameter(torch.empty((self.inp_len, self.inp_len)))
            self.K = nn.Parameter(torch.empty((self.inp_len, self.inp_len)))
            self.V = nn.Parameter(torch.empty((self.inp_len, self.inp_len)))

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5, inp_len=inp_len,
            dropout_rate=dropout_rate,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)
        if self.with_graph_mat:
            q = torch.matmul(self.Q, q)
            k = torch.matmul(self.K, k)
            v = torch.matmul(self.V, v)
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(1000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout_rate: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_head, d_model, d_inner, d_k, d_v, with_graph_mat: bool, inp_len: int, dropout_rate: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, with_graph_mat, inp_len, dropout_rate=dropout_rate)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout_rate=dropout_rate)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class VocabEncoder(nn.Module):
    n_vocab: int
    d_word_vec: int
    pad_idx: int
    inp_len: int
    dropout_rate: float

    def __init__(self, n_vocab: int, d_word_vec: int, pad_idx: int, inp_len: int, dropout_rate: float):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_word_vec = d_word_vec
        self.pad_idx = pad_idx
        self.inp_len = inp_len
        self.dropout_rate = dropout_rate
        self.src_word_emb = nn.Embedding(self.n_vocab, self.d_word_vec, padding_idx=self.pad_idx)
        self.position_enc = PositionalEncoding(self.d_word_vec, n_position=self.inp_len)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.d_word_vec, eps=1e-6)

    def forward(self, src_seq: Tensor) -> Tensor:
        enc_out = self.src_word_emb(src_seq)
        enc_out = self.position_enc(enc_out)
        enc_out = self.dropout(enc_out)
        enc_out = self.layer_norm(enc_out)
        return enc_out


class Encoder(nn.Module):
    n_layers: int
    n_head: int
    d_k: int
    d_v: int
    d_model: int
    d_inner: int
    pad_idx: int
    with_graph_mat: bool
    inp_len: int
    dropout_rate: float

    def __init__(self, n_layers: int,
                 n_head: int, d_k: int, d_v: int, d_model: int, d_inner: int, pad_idx: int,
                 with_graph_mat: bool, inp_len: int, dropout_rate: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.pad_idx = pad_idx
        self.with_graph_mat = with_graph_mat
        self.inp_len = inp_len
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.n_head, self.d_model, self.d_inner, self.d_k, self.d_v, self.with_graph_mat, self.inp_len, dropout_rate=dropout_rate)
            for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, src_seq: Tensor, src_mask: Tensor, return_attns: bool = False) -> tuple[Tensor, list[Tensor]]:
        enc_slf_attn_list = []

        enc_out = src_seq

        for enc_layer in self.layer_stack:
            enc_out, enc_slf_attn = enc_layer(enc_out, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        return enc_out, enc_slf_attn_list


class Decoder(nn.Module):
    n_layers: int
    n_head: int
    d_k: int
    d_v: int
    d_model: int
    d_inner: int
    dropout_rate: float

    def __init__(self, n_layers: int, n_head: int, d_k: int, d_v: int, d_model: int, d_inner: int, pad_idx: int,
                 with_graph_mat: bool, inp_len: int, dropout_rate: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.pad_idx = pad_idx
        self.with_graph_mat = with_graph_mat
        self.inp_len = inp_len
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(
                n_head=self.n_head, d_model=self.d_model, d_inner=self.d_inner, d_k=self.d_k, d_v=self.d_v,
                with_graph_mat=self.with_graph_mat, inp_len=self.inp_len, dropout_rate=dropout_rate,
            )
            for _ in range(n_layers)
        ])
        self.d_model = d_model
        self.rank_prj = nn.Linear(self.d_model, 1, bias=False)

    def forward(self, src_seq: Tensor, src_mask: Tensor, return_attns: bool = False) -> tuple[Tensor, list[Tensor]]:
        enc_slf_attn_list = []

        enc_out = src_seq

        for enc_layer in self.layer_stack:
            enc_out, enc_slf_attn = enc_layer(enc_out, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        rank_logit = self.rank_prj(enc_out)
        return rank_logit, enc_slf_attn_list


class CfgVocabEncoder(BaseModel):
    n_vocab: int
    d_word_vec: int
    pad_idx: int
    inp_len: int
    dropout_rate: float


class CfgEncoder(BaseModel):
    n_layers: int
    n_head: int
    d_k: int
    d_v: int
    d_model: int
    d_inner: int
    pad_idx: int
    with_graph_mat: bool
    inp_len: int
    dropout_rate: float


class CfgDecoder(BaseModel):
    pass


class CfgMllm(BaseModel):
    vocab_encoder: CfgVocabEncoder
    encoders: list[CfgEncoder]
    decoders: list[CfgEncoder]



class Mllm(nn.Module):
    cfg: CfgMllm
    vocab_encoder: VocabEncoder
    encoders: nn.ModuleList
    decoders: nn.ModuleList

    def __init__(self, cfg: CfgMllm):
        super().__init__()
        self.cfg = cfg.copy(deep=True)
        self.vocab_encoder = VocabEncoder(
            **cfg.vocab_encoder.dict(),
        )
        self.encoders = nn.ModuleList([
            Encoder(**cfg_enc.dict()) for cfg_enc in cfg.encoders
        ])
        self.decoders = nn.ModuleList([
            Encoder(**cfg_dec.dict()) for cfg_dec in cfg.decoders
        ])
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def run_vocab_encoder(self, inp: Tensor) -> Tensor:
        return self.vocab_encoder(inp)

    def run_encoder(self, level_num: int, inp: Tensor) -> tuple[Tensor, Tensor]:
        ind = level_num - 1
        out = self.encoders[ind](inp)[0]
        out_seq, out_emb = out[..., :-1], out[..., -1]
        return out_seq, out_emb

    def run_decoder(self, level_num: int, inp: Tensor) -> Tensor:
        ind = level_num - 1
        return self.decoders[ind](inp)[0]


T = TypeVar('T')
MS = Union[T, tuple[T]]


def create_mllm_cfg(
        n_vocab: int, d_word_wec: int = 512, inp_len: int = 1000, dropout_rate: float = 0.1,
        n_levels: int = 2,
        enc_n_layers: MS[int] = (3, 2), n_head: int = 8, d_k: int = 64, d_v: int = 64, d_model: int = 512,
        d_inner: int = 2048, enc_with_graph_mat: bool = False,
        dec_n_layers: MS[int] = 1, pad_idx = 0,
) -> CfgMllm:
    if not isinstance(enc_n_layers, tuple):
        enc_n_layers = tuple(enc_n_layers for _ in range(n_levels))
    assert len(enc_n_layers) == n_levels
    if not isinstance(dec_n_layers, tuple):
        dec_n_layers = tuple(dec_n_layers for _ in range(n_levels))
    assert len(dec_n_layers) == n_levels

    cfg_vocab_enc = CfgVocabEncoder(
        n_vocab=n_vocab, d_word_vec=d_word_wec, pad_idx=pad_idx, inp_len=inp_len, dropout_rate=dropout_rate,
    )
    cfgs_enc = []
    for n_layers in enc_n_layers:
        cfg_enc = CfgEncoder(
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
            with_graph_mat=enc_with_graph_mat, inp_len=inp_len, dropout_rate=dropout_rate,
        )
        cfgs_enc.append(cfg_enc)

    cfgs_dec = []
    for n_layers in dec_n_layers:
        cfg_dec = CfgEncoder(
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx,
            with_graph_mat=False, inp_len=0, dropout_rate=dropout_rate,
        )
        cfgs_dec.append(cfg_dec)

    cfg_mllm = CfgMllm(
        vocab_encoder=cfg_vocab_enc, encoders=cfgs_enc, decoders=cfgs_dec,
    )

    return cfg_mllm


def test_create_mllm_model():
    cfg_mllm = create_mllm_cfg(n_vocab=50_000)
    print(cfg_mllm)
    mllm = Mllm(cfg_mllm)
    print(mllm)


if __name__ == '__main__':
    test_create_mllm_model()

