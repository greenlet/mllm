import sys
from pathlib import Path
from typing import Optional, cast

from transformers import PreTrainedTokenizer

from mllm.model.bert import BertModel
from mllm.model.bert_generation.modeling_bert_generation import BertGenerationEmbeddings
from mllm.model.losses import EncdecMaskPadItemLoss, R2Loss, join_losses_dicts
from mllm.model.utils import get_top_vects
from mllm.train.utils import get_activation_module

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
        elif reduct_type == HgReductType.MaxPool:
            self.reducer = nn.MaxPool1d(self.step)
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
        is_top = self.reduct_type in (HgReductType.TopCos, HgReductType.TopDot)
        if len_mod > 0 and not is_top:
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
        elif is_top:
            n = seq_len // self.step
            calc_cos = self.reduct_type == HgReductType.TopCos
            out = get_top_vects(inp, n=n, calc_cos=calc_cos)
        elif self.reduct_type == HgReductType.MaxPool:
            # [batch_size, d_model, seq_len]
            x = inp.transpose(1, 2)
            # [batch_size, d_model, seq_len // step]
            x = self.reducer(x)
            # [batch_size, seq_len // step, d_model]
            out = x.transpose(1, 2)
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
        n_enc_layers = cfg.n_similar_layers if cfg.share_layer_weights else cfg.n_layers * cfg.n_similar_layers
        self.enc_layers = nn.ModuleList([
            EncoderLayer(
                n_heads=cfg.n_heads, d_model=cfg.d_model, d_inner=cfg.d_inner, d_k=cfg.d_k, d_v=cfg.d_v,
                dropout_rate=cfg.dropout_rate, temperature=cfg.temperature,
            ) for _ in range(n_enc_layers)
        ])
        n_rdc_layers = 1 if cfg.share_layer_weights else cfg.n_layers
        self.rdc_layers = nn.ModuleList([
            ReduceLayer(d_model=cfg.d_model, step=cfg.step, reduct_type=cfg.reduct_type) for _ in range(n_rdc_layers)
        ])
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -0.1, 0.1)

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

        # enc_layers_it = iter(self.enc_layers)
        # for rdc_layer in self.rdc_layers:
        #     for _ in range(self.cfg.n_similar_layers):
        #         enc_layer = next(enc_layers_it)
        #         out, _ = enc_layer(out, slf_attn_mask=mask)
        #     # inds = slice(0, out.shape[1], 2)
        #     # print_dtype_shape(mask, 'mask 1')
        #     # mask = mask[:, inds, inds]
        #     # print_dtype_shape(mask, 'mask 2')
        #     out = rdc_layer(out)

        for i_lv in range(self.cfg.n_layers):
            for i_lr in range(self.cfg.n_similar_layers):
                i = i_lv * self.cfg.n_similar_layers + i_lr
                i %= len(self.enc_layers)
                enc_layer = self.enc_layers[i]
                out, _ = enc_layer(out, slf_attn_mask=mask)
            j = i_lv % len(self.rdc_layers)
            out = self.rdc_layers[j](out)

        return out


class EncoderBert(nn.Module):
    cfg: EncBertCfg
    bert_model: BertModel

    def __init__(self, cfg: EncBertCfg):
        super().__init__()
        self.cfg = cfg
        self.bert_model = BertModel.from_pretrained(self.cfg.pretrained_model_name, torch_dtype=torch.float32)
        pos_embs = self.bert_model.embeddings.position_embeddings
        # pos_embs_len = pos_embs.weight.shape[0]
        # if self.cfg.inp_len > pos_embs_len:
        #     n_repeat = self.cfg.inp_len // pos_embs_len + min(self.cfg.inp_len % pos_embs_len, 1)
        #     pos_embs_shape = pos_embs.weight.shape
        #     pos_embs.weight = nn.Parameter(pos_embs.weight.repeat((n_repeat, 1)))
        #     print(f'Input length (={self.cfg.inp_len}) > position embeddings length (={pos_embs_len}). '
        #           f'Repeating position embeddings {n_repeat} times, changing shape: {pos_embs_shape} --> {pos_embs.weight.shape}')

    def forward(self, inp_toks: Tensor, inp_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if inp_mask is None:
            inp_mask = inp_toks != self.cfg.pad_token_id
        out = self.bert_model(inp_toks, inp_mask)
        return out['last_hidden_state'], out['pooler_output']


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
    enc_only: bool
    enc_bert: EncoderBert
    # dec_pyr: DecoderPyramid

    def __init__(self, cfg: EncdecBertCfg, enc_only: bool = False):
        super().__init__()
        self.cfg = cfg
        self.enc_only = enc_only
        self.enc_bert = EncoderBert(cfg.enc_bert)
        if not self.enc_only:
            self.dec_pyr = DecoderPyramid(cfg.dec_pyr)
            for n, p in self.dec_pyr.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                # else:
                #     nn.init.uniform_(p, -0.1, 0.1)
                # pnp = p.detach().cpu().numpy()
                # print(n, pnp.shape, pnp.min(), pnp.mean(), pnp.max())

    # inp: [batch_size, inp_len]
    # mask: [batch_size, inp_len]
    def forward(self, inp: Tensor, mask: Tensor, enc_only: bool = False) -> tuple[Tensor, Optional[Tensor]]:
        # out_dec: tuple[(batch_size, inp_len, d_model), (batch_size, d_model)]
        out_enc = self.enc_bert(inp, mask)
        # out_enc_last_hidden_state: (batch_size, inp_len, d_model)
        # out_enc_pooler: (batch_size, d_model)
        out_enc_last_hidden_state, out_enc_pooler = out_enc
        if self.cfg.enc_bert.emb_type == BertEmbType.Cls:
            # out_enc_emb: (batch_size, d_model)
            out_enc_emb = out_enc_last_hidden_state[:, 0]
        elif self.cfg.enc_bert.emb_type == BertEmbType.Pooler:
            # out_enc_emb: (batch_size, d_model)
            out_enc_emb = out_enc_pooler
        else:
            raise Exception(f'Encoder BERT embedding type {self.cfg.enc_bert.emb_type} is not supported')

        if not self.enc_only and not enc_only:
            # [batch_size, inp_len, n_vocab]
            out_dec = self.dec_pyr(out_enc_emb)
        else:
            out_dec = None
        return out_enc, out_dec


class EncdecBertAgg(nn.Module):
    cfg: EncdecBertCfg
    tkz: PreTrainedTokenizer
    model: EncdecBert
    enforce_enc_mask_understanding: bool
    next_tok_pred: bool
    masked_loss_for_encoder: bool
    emb_loss_weight: float
    vocab_loss_weight: float
    total_loss_weight: float

    def __init__(
            self, cfg: EncdecBertCfg, tkz: PreTrainedTokenizer, enforce_enc_mask_understanding: bool, next_tok_pred: bool,
            masked_loss_for_encoder: bool, emb_loss_weight: float = 1.0, vocab_loss_weight: float = 1.0,
        ):
        super().__init__()
        self.cfg = cfg
        self.tkz = tkz
        self.model = EncdecBert(cfg, enc_only=False)
        self.enforce_enc_mask_understanding = enforce_enc_mask_understanding
        self.next_tok_pred = next_tok_pred
        self.masked_loss_for_encoder = masked_loss_for_encoder
        self.emb_loss_weight = emb_loss_weight
        self.vocab_loss_weight = vocab_loss_weight
        self.total_loss_weight = self.emb_loss_weight + self.vocab_loss_weight
        self.vocab_loss_fn = EncdecMaskPadItemLoss(
            msk_tok_id=cast(int, tkz.mask_token_id), spc_tok_ids=[cast(int, tkz.pad_token_id), cast(int, tkz.cls_token_id), cast(int, tkz.sep_token_id)],
            reg_weight=1, msk_weight=1, spc_weight=0.1,
        )
        if self.enforce_enc_mask_understanding:
            self.emb_loss_fn = nn.CosineEmbeddingLoss()
            # self.emb_loss_fn = nn.L1Loss()
            # self.emb_loss_fn = nn.MSELoss()
            # self.emb_loss_fn = R2Loss()

    def load_pretrained(self, pretrained_model_path: Optional[Path]):
        if pretrained_model_path and pretrained_model_path.exists():
            print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
            pretrained_checkpoint = torch.load(pretrained_model_path)
            checkpt_dict = pretrained_checkpoint['model']

            checkpt_dict_renamed = {}
            for key, val in checkpt_dict.items():
                if key.startswith('model.'):
                    key = key[6:]
                if self.model.enc_only and key.startswith('dec_pyr.'):
                    continue
                if key.startswith('vocab_loss_fn.') or key.startswith('emb_loss_fn.'):
                    continue
                checkpt_dict_renamed[key] = val
            checkpt_dict = checkpt_dict_renamed

            self.model.load_state_dict(checkpt_dict, strict=True)

    def create_causal_mask(self, size: int, device: torch.device) -> Tensor:
        # (size, size)
        mask = torch.tril(torch.ones((size, size), device=device)).to(torch.int32)
        return mask

    # inp_masked_toks: (batch_size, inp_len)
    # inp_toks: (batch_size, inp_len)
    def forward(self, inp_masked_toks, inp_toks: Tensor) -> dict[str, Tensor]:
        if self.next_tok_pred:
            assert not self.enforce_enc_mask_understanding, 'Next token prediction together with enforcing encoder masked token understanding is not supported yet'
            batch_size, inp_len = inp_toks.shape
            device = inp_toks.device
            # (inp_len, inp_len)
            causal_mask = self.create_causal_mask(inp_len, device)
            # (1, inp_len, inp_len)
            causal_mask = causal_mask.unsqueeze(0)
            # out_enc_causal: tuple[(batch_size, inp_len, d_model), (batch_size, d_model)]
            # out_dec_causal: (batch_size, inp_len, n_vocab)
            out_enc_causal, out_dec_causal = self.model(inp_toks, causal_mask, enc_only=False)

            # tgt_toks: (batch_size, inp_len - 1)
            tgt_toks = inp_toks[:, 1:].contiguous()
            # logits: (batch_size, inp_len - 1, n_vocab)
            logits = out_dec_causal[:, :-1, :].contiguous()

            loss = torch.tensor(0.0, device=device)
            for ib in range(batch_size):
                n_nonpad = (tgt_toks[ib, :] != self.tkz.pad_token_id).sum().item()
                tgt_toks_i = tgt_toks[ib, :n_nonpad]
                logits_i = logits[ib, :n_nonpad, :]
                loss_i = F.cross_entropy(logits_i, tgt_toks_i, reduction='mean')
                loss += loss_i
            loss /= batch_size
            return {'loss': loss}

        if self.enforce_enc_mask_understanding:
            # (batch_size, inp_len)
            inp_att_mask = inp_toks != self.tkz.pad_token_id
            
            # out_enc: tuple[(batch_size, inp_len, d_model), (batch_size, d_model)]
            # out_dec: (batch_size, inp_len, n_vocab)
            out_enc, out_dec = self.model(inp_toks, inp_att_mask, enc_only=True)
            # out_enc_last_hidden_state: (batch_size, inp_len, d_model)
            # out_enc_pooler: (batch_size, d_model)
            out_enc_last_hidden_state, out_enc_pooler = out_enc
            # out_enc: (batch_size, d_model)
            out_enc_emb = out_enc_last_hidden_state[:, 0]
            
            # (batch_size, inp_len)
            inp_masked_att_mask = inp_masked_toks != self.tkz.pad_token_id
            # out_enc_masked: tuple[(batch_size, inp_len, d_model), (batch_size, d_model)]
            # out_dec_masked: (batch_size, inp_len, n_vocab)
            out_enc_masked, out_dec_masked = self.model(inp_masked_toks, inp_masked_att_mask, enc_only=False)
            # out_enc_masked_last_hidden_state: (batch_size, inp_len, d_model)
            # out_enc_masked_pooler: (batch_size, d_model)
            out_enc_masked_last_hidden_state, out_enc_masked_pooler = out_enc_masked
            # out_enc_masked_emb: (batch_size, d_model)
            out_enc_masked_emb = out_enc_masked_last_hidden_state[:, 0]
            
            vocab_loss_dict = self.vocab_loss_fn(out_dec_masked, inp_masked_toks, inp_toks)
            # (1,)
            vocab_loss = vocab_loss_dict['loss']

            # emb_loss = self.emb_loss_fn(out_enc_masked_emb, out_enc_emb)
            emb_loss = self.emb_loss_fn(out_enc_masked_emb, out_enc_emb, torch.ones((out_enc_emb.shape[0],), device=out_enc.device))

            # loss = (self.emb_loss_weight * emb_loss + self.vocab_loss_weight * vocab_loss) / self.total_loss_weight
            loss = self.emb_loss_weight * emb_loss + self.vocab_loss_weight * vocab_loss
            vocab_loss_dict = {f'vocab_{k}': v for k, v in vocab_loss_dict.items()}
            return {'loss': loss, 'emb_loss': emb_loss, **vocab_loss_dict}
        
        # (batch_size, inp_len)
        inp_masked_att_mask = inp_masked_toks != self.tkz.pad_token_id
        # out_enc: tuple[(batch_size, inp_len, d_model), (batch_size, d_model)]
        # out_dec: (batch_size, inp_len, n_vocab)
        out_enc, out_dec = self.model(inp_masked_toks, inp_masked_att_mask)
        # out_enc_last_hidden_state: (batch_size, inp_len, d_model)
        # out_enc_pooler: (batch_size, d_model)
        out_enc_last_hidden_state, out_enc_pooler = out_enc
        # out_enc_emb: (batch_size, d_model)
        out_enc_emb = out_enc_last_hidden_state[:, 0]

        if self.masked_loss_for_encoder:
            # (batch_size, inp_len - 1, n_vocab)
            out_logits = self.model.dec_pyr.vocab_decoder(out_enc_last_hidden_state[:, 1:])
            # (1,)
            enc_loss = self.vocab_loss_fn(out_logits, inp_masked_toks[:, 1:], inp_toks[:, 1:])
            dec_loss = self.vocab_loss_fn(out_dec, inp_masked_toks, inp_toks)
            loss = (enc_loss['loss'] + dec_loss['loss']) / 2
            encdec_loss = join_losses_dicts(['enc', 'dec'], [enc_loss, dec_loss])
            return {'loss': loss, **encdec_loss}

        # (1,)
        vocab_loss = self.vocab_loss_fn(out_dec, inp_masked_toks, inp_toks)
        return vocab_loss


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


