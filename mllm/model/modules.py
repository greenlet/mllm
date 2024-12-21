from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mllm.config.model import PosEncType


class ScaledDotProductAttention(nn.Module):
    temperature: float
    inp_len: int
    dropout_rate: float
    # dropout: nn.Module

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
    n_heads: int
    d_model: int
    d_k: int
    d_v: int
    with_graph_mat: bool
    inp_len: int
    dropout_rate: float

    # Q: Optional[nn.Parameter] = None
    # K: Optional[nn.Parameter] = None
    # V: Optional[nn.Parameter] = None

    def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int,
                 with_graph_mat: bool = False, inp_len: int = 0, dropout_rate: float = 0.1):
        super().__init__()

        self.n_heads = n_heads
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

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        temp = d_k ** 0.5
        temp = 1
        self.attention = ScaledDotProductAttention(
            temperature=temp, inp_len=inp_len,
            dropout_rate=dropout_rate,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
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
        q = q.view(sz_b, len_q, n_heads, d_k)
        k = k.view(sz_b, len_k, n_heads, d_k)
        v = v.view(sz_b, len_v, n_heads, d_v)

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
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

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
        bias = True
        self.w_1 = nn.Linear(d_in, d_hid, bias=bias) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in, bias=bias) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.w_2(F.leaky_relu(self.w_1(x)))
        # x = self.w_2(F.sigmoid(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_heads, d_model, d_inner, d_k, d_v, with_graph_mat: bool, inp_len: int, dropout_rate: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, with_graph_mat, inp_len, dropout_rate=dropout_rate)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout_rate=dropout_rate)

    def forward(self, enc_input: Optional[Tensor], enc_input_kv: Optional[Tensor] = None, slf_attn_mask: Optional[Tensor] = None):
        if enc_input_kv is None:
            enc_input_kv = enc_input
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input_kv, enc_input_kv, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class VocabEncoder(nn.Module):
    n_vocab: int
    d_word_vec: int
    d_model: int
    pad_idx: int
    inp_len: int
    dropout_rate: float
    pos_enc_type: PosEncType

    def __init__(self, n_vocab: int, d_word_vec: int, d_model: int, pad_idx: int, inp_len: int, dropout_rate: float, pos_enc_type: PosEncType = PosEncType.Num):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_word_vec = d_word_vec
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.inp_len = inp_len
        self.dropout_rate = dropout_rate
        self.src_word_emb = nn.Embedding(self.n_vocab, self.d_word_vec, padding_idx=self.pad_idx)
        self.pos_enc_type = pos_enc_type
        if self.pos_enc_type == PosEncType.Num:
            self.position_enc = PositionalEncoding(self.d_word_vec, n_position=self.inp_len * 10)
        elif self.pos_enc_type == PosEncType.Emb:
            self.position_enc = nn.Embedding(self.inp_len, self.d_word_vec)
        else:
            raise Exception(f'Unspoorted position encoding type: {self.pos_enc_type}')
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.d_word_vec, eps=1e-6)

    def forward(self, src_seq: Tensor) -> Tensor:
        enc_out = self.src_word_emb(src_seq)
        enc_out *= self.d_model**0.5
        if self.pos_enc_type == PosEncType.Num:
            enc_out = self.position_enc(enc_out)
        elif self.pos_enc_type == PosEncType.Emb:
            # inds = torch.arange(self.inp_len).expand(len(src_seq), self.inp_len)
            # pos_embs = self.position_enc(inds)
            pos_embs = self.position_enc.weight.expand(len(src_seq), self.inp_len, self.d_word_vec)
            enc_out = enc_out + pos_embs
        enc_out = self.dropout(enc_out)
        enc_out = self.layer_norm(enc_out)
        return enc_out


class Encoder(nn.Module):
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
    # dropout: nn.Dropout
    # layer_stack: nn.ModuleList
    with_emb_mat: bool
    # w_em: Optional[nn.Linear] = None

    def __init__(self, n_layers: int,
                 n_heads: int, d_k: int, d_v: int, d_model: int, d_inner: int, pad_idx: int,
                 with_graph_mat: bool, inp_len: int, with_emb_mat: bool, dropout_rate: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.pad_idx = pad_idx
        self.with_graph_mat = with_graph_mat
        self.inp_len = inp_len
        self.with_emb_mat = with_emb_mat
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.n_heads, self.d_model, self.d_inner, self.d_k, self.d_v, self.with_graph_mat, self.inp_len, dropout_rate=dropout_rate)
            for _ in range(n_layers)])
        self.d_model = d_model
        if self.with_emb_mat:
            assert self.inp_len > 0
            self.w_em = nn.Linear(self.inp_len, 1, bias=False)
            # self.A_em = nn.Parameter(torch.zeros((self.inp_len, self.d_model), dtype=torch.float32))
        else:
            self.a_em = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def forward(self, src_seq: Tensor, src_mask: Optional[Tensor] = None, return_attns: bool = False) -> tuple[Tensor, list[Tensor]]:
        enc_slf_attn_list = []

        enc_out = src_seq

        for enc_layer in self.layer_stack:
            enc_out, enc_slf_attn = enc_layer(enc_out, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if self.with_emb_mat:
            enc_out = enc_out.transpose(1, 2)
            enc_out = self.w_em(enc_out)
            enc_out = enc_out.squeeze(-1)
            # enc_out = self.A_em.unsqueeze(0) * enc_out
            # enc_out = torch.sum(enc_out, dim=1)
        else:
            enc_out = torch.sum(enc_out, dim=1, keepdim=False) * self.a_em

        enc_out = self.layer_norm(enc_out)

        return enc_out, enc_slf_attn_list


class Decoder(nn.Module):
    n_layers: int
    n_heads: int
    d_k: int
    d_v: int
    d_model: int
    d_inner: int
    dropout_rate: float

    def __init__(self, n_layers: int, n_heads: int, d_k: int, d_v: int, d_model: int, d_inner: int, pad_idx: int,
                 with_graph_mat: bool, inp_len: int, dropout_rate: float = 0.1, with_emb_mat=False):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
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
                n_heads=self.n_heads, d_model=self.d_model, d_inner=self.d_inner, d_k=self.d_k, d_v=self.d_v,
                with_graph_mat=self.with_graph_mat, inp_len=self.inp_len, dropout_rate=dropout_rate,
            )
            for _ in range(n_layers)
        ])
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.rank_prj = nn.Linear(self.d_model, 1, bias=False)

    def forward(self, src_seq: Tensor, src_mask: Optional[Tensor] = None, return_attns: bool = False) -> tuple[Tensor, list[Tensor]]:
        enc_slf_attn_list = []

        enc_out = src_seq

        for enc_layer in self.layer_stack:
            enc_out, enc_slf_attn = enc_layer(enc_out, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # enc_out = self.layer_norm(enc_out)
        rank_logit = self.rank_prj(enc_out)

        # rank_prob = rank_logit
        # rank_prob = torch.sigmoid(rank_logit)
        rank_prob = torch.softmax(rank_logit, dim=-2)
        return rank_prob, enc_slf_attn_list


class EmbDecoder(nn.Module):
    d_emb: int
    n_layers: int
    n_heads: int
    d_hid: int
    seq_len: int
    dp_rate: float
    A_emb2sec: nn.Parameter
    att_layers: nn.ModuleList

    def __init__(self, d_emb: int, n_layers: int, n_heads: int, d_hid: int, seq_len: int, dp_rate: float = 0.0):
        super().__init__()
        self.d_emb = d_emb
        self.n_layers = n_layers
        self.n_heads = n_heads
        assert self.d_emb % self.n_heads == 0
        self.d_head = self.d_emb // self.n_heads
        self.d_hid = d_hid
        self.seq_len = seq_len
        self.dp_rate = dp_rate
        self.A_emb2sec = nn.Parameter(torch.empty((self.seq_len, self.d_emb, self.d_emb), dtype=torch.float32))
        self.att_layers = nn.ModuleList([
            EncoderLayer(
                n_heads=n_heads, d_model=self.d_emb, d_inner=self.d_hid, d_k=self.d_head, d_v=self.d_head,
                with_graph_mat=False, inp_len=self.seq_len, dropout_rate=self.dp_rate,
            ) for _ in range(self.n_layers)
        ])

    # inp has [batch, d_emb] dimensions
    def forward(self, inp: Tensor) -> Tensor:
        # Make inp [batch, 1, d_emb, 1] dimensions
        inp = inp.reshape((inp.shape[0], 1, self.d_emb, 1))
        # [seq_len, d_emb, d_emb] x [batch, 1, d_emb, 1] = [batch, seq_len, d_emb, 1]
        seq = self.A_emb2sec.matmul(inp)
        # Change [batch, seq_len, d_emb, 1] to [batch, seq_len, d_emb]
        seq = seq.squeeze(-1)

        for att_layer in self.att_layers:
            seq, _ = att_layer(seq)

        return seq


class VocabDecoder(nn.Module):
    d_model: int
    n_vocab: int
    word_prj: nn.Linear

    def __init__(self, d_model: int, n_vocab: int):
        super().__init__()
        self.d_model = d_model
        self.n_vocab = n_vocab
        self.word_prj = nn.Linear(d_model, n_vocab, bias=False)

    def forward(self, inp: Tensor) -> Tensor:
        out_logit = self.word_prj(inp)
        return out_logit


class DecoderRankSimple(nn.Module):
    d_model: int

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    # docs_chunks: (batch_size, docs_chunks_len, d_model)
    # query_chunks: (batch_size, query_chunks_len, d_model)
    def forward(self, docs_chunks: Tensor, query_chunks: Tensor) -> Tensor:
        # docs_chunks = self.layer_norm(docs_chunks)
        # query_chunks = self.layer_norm(query_chunks)
        # docs_chunks = docs_chunks / docs_chunks.norm(dim=-1, keepdim=True)
        # query_chunks = query_chunks / query_chunks.norm(dim=-1, keepdim=True)

        # (batch_size, query_chunks_len, d_model)
        query_chunks = self.w(query_chunks)

        # (batch_size, d_model, docs_chunks_len)
        docs_chunks = docs_chunks.transpose(-2, -1)

        # (batch_size, query_chunks_len, docs_chunks_len)
        ranks = torch.matmul(query_chunks, docs_chunks)

        # (batch_size, docs_chunks_len)
        ranks = torch.max(ranks, 1)[0]

        # (batch_size, docs_chunks_len)
        ranks_prob = torch.sigmoid(ranks)
        # ranks_prob = torch.softmax(ranks, dim=-1)

        return ranks_prob


class DecoderRankTrans(nn.Module):
    n_layers: int
    n_heads: int
    d_k: int
    d_v: int
    d_model: int
    d_inner: int
    inp_len: int
    dropout_rate: float
    dropout: nn.Dropout
    att_layers: nn.ModuleList
    w: nn.Linear
    layer_norm: nn.LayerNorm

    def __init__(self, n_layers: int, n_heads: int, d_k: int, d_v: int, d_model: int, d_inner: int, inp_len: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.inp_len = inp_len
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.att_layers = nn.ModuleList([
            EncoderLayer(self.n_heads, self.d_model, self.d_inner, self.d_k, self.d_v, False, self.inp_len, dropout_rate=self.dropout_rate)
            for _ in range(n_layers)])
        self.w = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    # docs_chunks: (batch_size, docs_chunks_len, d_model)
    # query_chunks: (batch_size, query_chunks_len, d_model)
    # res: (batch_size, 1)
    def forward(self, docs_chunks: Tensor, query_chunks: Tensor) -> Tensor:
        enc_out = docs_chunks
        for att_layer in self.att_layers:
            enc_out, _ = att_layer(enc_out)

        # if len(self.att_layers) > 0:
        #     enc_out = self.layer_norm(enc_out)

        # (batch_size, query_chunks_len, d_model)
        query_chunks = self.w(query_chunks)

        # (batch_size, d_model, docs_chunks_len)
        docs_chunks = enc_out.transpose(-2, -1)

        # (batch_size, query_chunks_len, docs_chunks_len)
        ranks = torch.matmul(query_chunks, docs_chunks)

        # (batch_size, docs_chunks_len)
        ranks = torch.max(ranks, 1)[0]

        # (batch_size, docs_chunks_len)
        ranks_prob = torch.sigmoid(ranks)
        # ranks_prob = torch.softmax(ranks, dim=-1)

        return ranks_prob


