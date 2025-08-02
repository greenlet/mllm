import re
import sys
from typing import Optional, Union

from mllm.model.bert_generation.modeling_bert_generation import BertGenerationEmbeddings
from mllm.train.utils import get_activation_module
from transformers import BertModel, PreTrainedModel, BertTokenizerFast

if '..' not in sys.path: sys.path.append('..')

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mllm.config.model import EncdecHgCfg, DecPyrCfg, EncPyrCfg, HgReductType, HgEnhanceType, RankerHgCfg, DecRankHgCfg, \
    parse_mlp_layers, ParsedMlpLayer, EncBertCfg, BertEmbType, EncdecBertCfg, RankerBertCfg
from mllm.model.modules import VocabEncoder, VocabDecoder


class ScaledDotProductAttention(nn.Module):
    temperature: float
    dropout_rate: float
    dropout: nn.Module

    def __init__(self, temperature: float,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, q, k, v, mask=None):
        attn = q / self.temperature
        attn = torch.matmul(attn, k.transpose(2, 3))

        if mask is not None:
            # print_dtype_shape(attn)
            # print_dtype_shape(mask)
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    n_heads: int
    d_model: int
    d_k: int
    d_v: int
    dropout_rate: float

    def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int, dropout_rate: float = 0.1, temperature: float = 0):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout_rate = dropout_rate

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        if temperature < 1e-8:
            temperature = d_k ** 0.5

        self.attention = ScaledDotProductAttention(
            temperature=temperature, dropout_rate=dropout_rate,
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
    def __init__(self, n_heads, d_model, d_inner, d_k, d_v, dropout_rate: float = 0.1, temperature: float = 0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout_rate=dropout_rate, temperature=temperature)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout_rate=dropout_rate)

    def forward(self, enc_input: Optional[Tensor], enc_input_kv: Optional[Tensor] = None, slf_attn_mask: Optional[Tensor] = None):
        if enc_input_kv is None:
            enc_input_kv = enc_input
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input_kv, enc_input_kv, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class ReduceLayer(nn.Module):
    d_model: int
    step: int
    reduct_type: HgReductType
    # reducer: nn.Linear

    def __init__(self, d_model: int, step: int, reduct_type: HgReductType) -> None:
        super().__init__()
        self.d_model = d_model
        self.step = step
        self.reduct_type = reduct_type
        if reduct_type == HgReductType.Matmul:
            self.reducer = nn.Linear(in_features=d_model * step, out_features=d_model, bias=True)
        else:
            self.reducer = None
        if reduct_type == HgReductType.Sub:
            assert self.step == 2

    # inp: [batch_size, seq_len, d_model]
    def forward(self, inp: Tensor) -> Tensor:
        batch_size, seq_len, d_model = inp.shape
        assert d_model == self.d_model, f'self.d_model = {self.d_model}. inp d_model = {d_model}'
        len_mod = seq_len % self.step
        # print_dtype_shape(inp, 'rdc_inp')
        if len_mod > 0:
            n_seq_add = self.step - len_mod
            inp = F.pad(inp, (0, 0, 0, n_seq_add), value=0)
            seq_len += n_seq_add
            # print_dtype_shape(inp, 'rdc_inp_pad')
        if self.reduct_type == HgReductType.Matmul:
            inp = inp.reshape(batch_size, seq_len // self.step, self.d_model * self.step)
            # print_dtype_shape(inp, 'rds_reshape')
            out = self.reducer(inp)
            # print_dtype_shape(out, 'rdc_reduce')
        elif self.reduct_type == HgReductType.Decim:
            out = inp[:, ::self.step]
        elif self.reduct_type == HgReductType.Avg:
            out = inp.reshape((batch_size, seq_len // self.step, self.step, d_model))
            out = torch.mean(out, dim=2, keepdim=False)
        elif self.reduct_type == HgReductType.Sub:
            out = inp.reshape((batch_size, seq_len // self.step, self.step, d_model))
            out = out[:, :, 1, :] - out[:, :, 0, :]
        else:
            raise Exception(f'Reduction type {self.reduct_type} is not supported')
        return out


class EncoderPyramid(nn.Module):
    cfg: EncPyrCfg
    # vocab_encoder: Union[VocabEncoder, BertGenerationEmbeddings]
    enc_layers: nn.ModuleList
    rdc_layers: nn.ModuleList
    inp_chunk_len: int

    def __init__(self, cfg: EncPyrCfg, bert_encoder: Optional[BertGenerationEmbeddings] = None):
        super().__init__()
        self.cfg = cfg
        if bert_encoder is None:
            self.vocab_encoder = VocabEncoder(**cfg.vocab_encoder.dict())
        else:
            self.vocab_encoder = bert_encoder
        self.enc_layers = nn.ModuleList([
            EncoderLayer(
                n_heads=cfg.n_heads, d_model=cfg.d_model, d_inner=cfg.d_inner, d_k=cfg.d_k, d_v=cfg.d_v,
                dropout_rate=cfg.dropout_rate, temperature=cfg.temperature,
            ) for _ in range(cfg.n_layers * cfg.n_similar_layers)
        ])
        self.rdc_layers = nn.ModuleList([
            ReduceLayer(d_model=cfg.d_model, step=cfg.step, reduct_type=cfg.reduct_type) for _ in range(cfg.n_layers)
        ])

    # Tensor of integer tokens: [batch_size, seq_len]
    def forward(self, inp: Tensor) -> Tensor:
        batch_size, seq_len = inp.shape
        # mask = (inp == self.cfg.pad_idx).to(torch.float32).to(inp.device)
        # mask = torch.matmul(mask.unsqueeze(-1), mask.unsqueeze(-2)).to(torch.int32)
        mask = None
        # assert self.cfg.inp_len == 0 or seq_len == self.cfg.inp_len, f'seq_len = {seq_len}. inp_len = {self.cfg.inp_len}'
        if isinstance(self.vocab_encoder, BertGenerationEmbeddings):
            # [batch_size, seq_len, d_model]
            # out = self.vocab_encoder(inp, do_not_transform_embeds=True)
            out = self.vocab_encoder(inp, do_not_transform_embeds=False)
        else:
            # [batch_size, seq_len, d_model]
            out = self.vocab_encoder(inp)
        # print_dtype_shape(out, 'vocab_enc')
        enc_layers_it = iter(self.enc_layers)
        for rdc_layer in self.rdc_layers:
            for _ in range(self.cfg.n_similar_layers):
                enc_layer = next(enc_layers_it)
                out, _ = enc_layer(out, slf_attn_mask=mask)
            # inds = slice(0, out.shape[1], 2)
            # print_dtype_shape(mask, 'mask 1')
            # mask = mask[:, inds, inds]
            # print_dtype_shape(mask, 'mask 2')
            out = rdc_layer(out)
        return out


class EncoderBert(nn.Module):
    cfg: EncBertCfg
    bert_model: BertModel

    def __init__(self, cfg: EncBertCfg):
        super().__init__()
        self.cfg = cfg
        self.bert_model = BertModel.from_pretrained(self.cfg.pretrained_model_name, torch_dtype=torch.float32)

    def forward(self, inp_toks: Tensor, inp_mask: Optional[Tensor] = None) -> Tensor:
        if inp_mask is None:
            inp_mask = inp_toks != self.cfg.pad_token_id
        out = self.bert_model(inp_toks, inp_mask)
        if self.cfg.emb_type == BertEmbType.Cls:
            out = out['last_hidden_state'][:, 0]
        elif self.cfg.emb_type == BertEmbType.Pooler:
            out = out['pooler_output']
        else:
            raise Exception(f'Bert embedding type "{self.cfg.emb_type}" is not supportd')
        return out


class EnhanceLayer(nn.Module):
    d_model: int
    step: int
    enhancer: nn.Linear

    def __init__(self, d_model: int, step: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.step = step
        self.enhancer = nn.Linear(in_features=d_model, out_features=d_model * step, bias=False)

    def forward(self, inp: Tensor) -> Tensor:
        batch_size, seq_len, d_model = inp.shape
        assert d_model == self.d_model, f'self.d_model = {self.d_model}. inp d_model = {d_model}'
        # print_dtype_shape(inp, 'enh_inp')
        out = self.enhancer(inp)
        # print_dtype_shape(out, 'enh_out')
        out = out.reshape(batch_size, seq_len * self.step, self.d_model)
        # print_dtype_shape(out, 'enh_out_reshape')
        return out


class DecoderPyramid(nn.Module):
    cfg: DecPyrCfg
    enc_layers: nn.ModuleList
    # enh_layers: nn.ModuleList
    inp_chunk_len: int
    vocab_decoder: VocabDecoder

    def __init__(self, cfg: DecPyrCfg):
        super().__init__()
        self.cfg = cfg
        self.enc_layers = nn.ModuleList([
            EncoderLayer(
                n_heads=cfg.n_heads, d_model=cfg.d_model, d_inner=cfg.d_inner, d_k=cfg.d_k, d_v=cfg.d_v,
                dropout_rate=cfg.dropout_rate, temperature=cfg.temperature,
            ) for _ in range(cfg.n_layers * cfg.n_similar_layers)
        ])
        self.enh_layers, self.enh_beg_layer = None, None
        if cfg.enhance_type == HgEnhanceType.Matmul:
            self.enh_layers = nn.ModuleList([
                EnhanceLayer(d_model=cfg.d_model, step=cfg.step) for _ in range(cfg.n_layers)
            ])
        elif cfg.enhance_type in (HgEnhanceType.MatmulBegin, HgEnhanceType.MatmulBeginBias):
            bias = cfg.enhance_type == HgEnhanceType.MatmulBeginBias
            self.enh_beg_layer = nn.Linear(in_features=cfg.d_model, out_features=cfg.d_model * cfg.inp_len, bias=bias)
        else:
            raise Exception(f'Enhance type {cfg.enhance_type} is not supported')
        self.vocab_decoder = VocabDecoder(d_model=self.cfg.d_model, n_vocab=self.cfg.n_vocab)

    # Tensor with embeddings: [batch_size, d_model]
    def forward(self, inp: Tensor) -> Tensor:
        batch_size, d_model = inp.shape
        # inp = inp.unsqueeze(1)
        out = inp
        if self.cfg.enhance_type in (HgEnhanceType.MatmulBegin, HgEnhanceType.MatmulBeginBias):
            # [batch_size, 1, d_model] -> [batch_size, 1, d_model * inp_len]
            out = self.enh_beg_layer(out)
            # [batch_size, 1, d_model * inp_len] -> [batch_size, d_model * inp_len]
            out = out.reshape((batch_size, self.cfg.inp_len, self.cfg.d_model))

        if self.cfg.enhance_type == HgEnhanceType.Matmul:
            enc_layers_it = iter(self.enc_layers)
            for enh_layer in self.enh_layers:
                out = enh_layer(out)
                for _ in range(self.cfg.n_similar_layers):
                    enc_layer = next(enc_layers_it)
                    out, _ = enc_layer(out)
        else:
            for enc_layer in self.enc_layers:
                out, _ = enc_layer(out)

        # [batch_size, inp_len, d_model] -> [batch_size, inp_len, n_vocab]
        out = self.vocab_decoder(out)

        return out


class EncdecHg(nn.Module):
    cfg: EncdecHgCfg
    enc_pyr: EncoderPyramid
    dec_pyr: DecoderPyramid

    def __init__(self, cfg: EncdecHgCfg):
        super().__init__()
        self.cfg = cfg
        self.enc_pyr = EncoderPyramid(cfg.enc_pyr)
        self.dec_pyr = DecoderPyramid(cfg.dec_pyr)

        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # else:
            #     nn.init.uniform_(p, -0.1, 0.1)
            # pnp = p.detach().cpu().numpy()
            # print(n, pnp.shape, pnp.min(), pnp.mean(), pnp.max())

    def forward(self, inp: Tensor, enc_only: bool = False) -> Tensor:
        out = inp
        out = self.enc_pyr(out)
        if not enc_only:
            out = self.dec_pyr(out)
        return out


class EncdecBert(nn.Module):
    cfg: EncdecBertCfg
    enc_bert: EncoderBert
    dec_pyr: DecoderPyramid

    def __init__(self, cfg: EncdecBertCfg):
        super().__init__()
        self.cfg = cfg
        self.enc_bert = EncoderBert(cfg.enc_bert)
        self.dec_pyr = DecoderPyramid(cfg.dec_pyr)

        for n, p in self.dec_pyr.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # else:
            #     nn.init.uniform_(p, -0.1, 0.1)
            # pnp = p.detach().cpu().numpy()
            # print(n, pnp.shape, pnp.min(), pnp.mean(), pnp.max())

    def forward(self, inp: Tensor, mask: Tensor, enc_only: bool = False) -> Tensor:
        out = self.enc_bert(inp, mask)
        if not enc_only:
            out = self.dec_pyr(out)
        return out


class DecoderRankHg(nn.Module):
    cfg: DecRankHgCfg
    # w: nn.Linear
    layer_norm: nn.LayerNorm
    cfg_mlp_layers: list[ParsedMlpLayer]
    # mlp_layers: list[nn.Linear]


    def __init__(self, cfg: DecRankHgCfg):
        super().__init__()
        self.cfg = cfg
        d_last = self.cfg.d_model
        mlp_layers = []
        self.cfg_mlp_layers = parse_mlp_layers(self.cfg.mlp_layers)
        for cfg_mlp_layer in self.cfg_mlp_layers:
            if cfg_mlp_layer.size > 0:
                mlp_layer = nn.Linear(d_last, cfg_mlp_layer.size, bias=cfg_mlp_layer.bias)
                d_last = cfg_mlp_layer.size
            else:
                act_module = get_activation_module(cfg_mlp_layer.act)
                mlp_layer = act_module()
            mlp_layers.append(mlp_layer)
        self.mlp_layers = nn.ModuleList(mlp_layers)

    # embs_batch: (batch_size, inp_len, d_model)
    def forward(self, embs_batch: Tensor) -> Tensor:
        out = embs_batch
        for mlp_layer in self.mlp_layers:
            out = mlp_layer(out)
        # out = self.layer_norm(out)
        return out


# emb1: (n1, d_model)
# emb2: (n2, d_model)
def cos_mat(emb1: Tensor, emb2: Tensor) -> Tensor:
    # emb1: (n1, d_model)
    emb1 = F.normalize(emb1, dim=-1)

    # emb2: (n2, d_model)
    emb2 = F.normalize(emb2, dim=-1)

    # emb2: (d_model, n2)
    emb2 = torch.transpose(emb2, 0, 1)
    # out: (n1, n2)
    out = emb1 @ emb2
    return out


class RankerHg(nn.Module):
    cfg: RankerHgCfg
    enc_pyr: EncoderPyramid
    dec_rank: DecoderRankHg

    def __init__(self, cfg: RankerHgCfg):
        super().__init__()
        self.cfg = cfg
        self.enc_pyr = EncoderPyramid(cfg.enc_pyr)
        self.dec_rank = DecoderRankHg(cfg.dec_rank)

        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # else:
            #     nn.init.uniform_(p, -0.1, 0.1)
            # pnp = p.detach().cpu().numpy()
            # print(n, pnp.shape, pnp.min(), pnp.mean(), pnp.max())

    # inp: (batch_size, inp_len)
    def run_encdec(self, inp: Tensor) -> Tensor:
        out = inp
        # out: (batch_size, 1, d_model)
        out = self.enc_pyr(inp)
        # out: (batch_size, d_model)
        out = out.squeeze(1)
        # out: (batch_size, d_model)
        out = self.dec_rank(out)
        return out

    # inp_docs: (n_docs, inp_len)
    # inp_qs: (n_qs, inp_len)
    def forward(self, inp_docs: Tensor, inp_qs: Tensor) -> Tensor:
        # out_docs: (n_docs, d_model)
        out_docs = self.run_encdec(inp_docs)

        # out_qs: (n_qs, d_model)
        out_qs = self.run_encdec(inp_qs)

        # out_rank: (n_docs, n_qs)
        # contains matrix of cosine distances between docs' and queries' embeddings
        out_rank = cos_mat(out_docs, out_qs)
        return out_rank


class RankerBert(nn.Module):
    cfg: RankerBertCfg
    enc_bert: EncoderBert
    dec_rank: DecoderRankHg

    def __init__(self, cfg: RankerBertCfg):
        super().__init__()
        self.cfg = cfg
        self.enc_bert = EncoderBert(cfg.enc_bert)
        self.dec_rank = DecoderRankHg(cfg.dec_rank)

        for n, p in self.dec_rank.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # inp: (batch_size, inp_len)
    def run_encdec(self, inp: Tensor) -> Tensor:
        out = inp
        # out: (batch_size, 1, d_model)
        out = self.enc_bert(out)
        # out: (batch_size, d_model)
        out = out.squeeze(1)
        out1 = out
        # out: (batch_size, d_model)
        out = self.dec_rank(out)
        return out

    # inp_docs: (n_docs, inp_len)
    # inp_qs: (n_qs, inp_len)
    def forward(self, inp_docs: Tensor, inp_qs: Tensor) -> Tensor:
        # out_docs: (n_docs, d_model)
        out_docs = self.run_encdec(inp_docs)

        # out_qs: (n_qs, d_model)
        out_qs = self.run_encdec(inp_qs)

        # out_rank: (n_docs, n_qs)
        # contains matrix of cosine distances between docs' and queries' embeddings
        out_rank = cos_mat(out_docs, out_qs)
        return out_rank


