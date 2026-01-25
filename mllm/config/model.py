import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, TypeVar, Union, Optional

import numpy as np
import torch
from pydantic import BaseModel

from mllm.model.bert.configuration_bert import BertConfig
from mllm.model.bert.modeling_bert import BertModel
from mllm.train.mask_utils import MaskCfg
from mllm.utils.utils import coalesce
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config


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
    TopCos = 'topcos'
    TopDot = 'topdot'
    MaxPool = 'mxpl'


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
    share_layer_weights: bool = False


class BertEmbType(str, Enum):
    Cls = 'cls'
    Pooler = 'plr'


class RankCosLossType(str, Enum):
    Cos = 'cos'


class EncBertCfg(BaseModel):
    inp_len: int
    d_model: int
    pad_token_id: int
    pretrained_model_name: str = ''
    tokenizer_name: str = ''
    emb_type: BertEmbType = BertEmbType.Cls
    emb2_tok_name: str = ''


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


class EncdecMiddleType(str, Enum):
    Graph = 'graph'
    Attn = 'attn'
    Mlp = 'mlp'


class PyClassCfg(BaseModel):
    module_path: str
    cls_name: str
    params: Dict[str, Any]


def create_cls_params(cls_name: str, cls_name_to_defaults: Dict[str, Any], override_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = cls_name_to_defaults[cls_name].copy()
    if override_params:
        params.update(override_params)
    return params


def cls_cfg_to_str(cls_cfg: PyClassCfg, param_to_short_str: Optional[Dict[str, str]] = None) -> str:
    parts = [cls_cfg.cls_name]
    param_to_short_str = param_to_short_str or {}
    for param_name, param_value in cls_cfg.params.items():
        param_short_name = param_to_short_str.get(param_name, param_name)
        if param_value is None:
            parts.append(f'{param_short_name}None')
        elif isinstance(param_value, bool):
            parts.append(bool_param_to_str(param_short_name, param_value))
        elif isinstance(param_value, str):
            parts.append(f'{param_short_name}{param_value.capitalize()}')
        else:
            parts.append(f'{param_short_name}{param_value}')
    return '_'.join(parts)


gnn_conv_param_to_short_str = {
    'improved': 'impr',
    'cached': 'cach',
    'normalize': 'norm',
    'normalization': 'norm',
    'add_self_loops': 'asl',
    'root_weight': 'rw',
    'project': 'prj',
}

gnn_conv_name_to_defaults = {
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv
    'GCNConv': {
        'improved': False,
        'cached': False,
        'add_self_loops': None,
        'normalize': True,
        'bias': True,
    },
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.ChebConv.html#torch_geometric.nn.conv.ChebConv
    'ChebConv': {
        'K': 1,
        'normalization': 'sym',
        'bias': True,
    },
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html#torch_geometric.nn.conv.SAGEConv
    'SAGEConv': {
        'aggr': 'mean',
        'normalize': False,
        'root_weight': True,
        'project': False,
        'bias': True,
    },
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.CuGraphSAGEConv.html#torch_geometric.nn.conv.CuGraphSAGEConv
    'CuGraphSAGEConv': {
        'aggr': 'mean',
        'normalize': False,
        'root_weight': True,
        'project': False,
        'bias': True,
    }
}

optimizer_param_to_short_str = {}
lrs_param_to_short_str = {'factor': 'f', 'patience': 'p', 'threshold': 't', 'min_lr': 'minlr'}


class EmbGraphCfg(BaseModel):
    n_layers: int
    d_model: int
    hidden_dim: int
    gnn_conv: PyClassCfg


class EmbAttnCfg(BaseModel):
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    d_inner: int
    n_layers: int
    dropout_rate: float


class EmbMlpCfg(BaseModel):
    d_model: int
    d_out: int
    window_size: int = 3  # Number of embeddings to process together in a sliding window
    n_window_layers: int = 1
    n_out_layers: int = 1
    act_fn: str = 'gelu'


class EncdecCiteToksTargetType(str, Enum):
    All = 'all'
    Cite = 'cite'


class EncdecCiteEmbsTargetType(str, Enum):
    Cos = 'cos'
    Mse = 'mse'
    Sqrt = 'sqrt'
    R2 = 'r2'


class EncdecTrainCfg(BaseModel):
    pretrained_model_path: Optional[Path] = None
    mask_cfg: Optional[MaskCfg] = None
    cite_toks_target_weight: float = 1.0
    cite_toks_target_type: EncdecCiteToksTargetType = EncdecCiteToksTargetType.All
    cite_toks_target_scale: float = 1.0
    cite_embs_target_weight: float = 1.0
    cite_embs_target_type: EncdecCiteEmbsTargetType = EncdecCiteEmbsTargetType.R2
    cite_embs_target_scale: float = 1.0
    input_toks_target_weight: float = 1.0
    input_toks_target_scale: float = 1.0
    learning_rate: float = 1e-4
    optimizer: Optional[PyClassCfg] = None
    learning_rate_scheduler: Optional[PyClassCfg] = None
    batch_size: int = 10


class EncdecGraphBertCfg(BaseModel):
    enc_bert: EncBertCfg
    dec_pyr: DecPyrCfg
    share_enc_dec_proj_weights: bool
    middle_type: EncdecMiddleType
    emb_graph: EmbGraphCfg
    emb_attn: EmbAttnCfg
    emb_mlp: EmbMlpCfg
    train_cfg: EncdecTrainCfg


class DecRankHgCfg(BaseModel):
    d_model: int
    mlp_layers: str = ''


class RankerHgCfg(BaseModel):
    enc_pyr: EncPyrCfg
    dec_rank: DecRankHgCfg


class RankerBertCfg(BaseModel):
    enc_bert: EncBertCfg
    dec_rank: DecRankHgCfg


class EncdecRankBertCfg(BaseModel):
    enc_bert: EncBertCfg
    dec_pyr: DecPyrCfg
    dec_rank: DecRankHgCfg


class EmbGenBertCfg(BaseModel):
    inp_len: int
    d_model: int
    pad_token_id: int
    pretrained_model_name: str = ''
    tokenizer_name: str = ''
    emb_type: BertEmbType = BertEmbType.Cls


class EncmixOutEmbsType(str, Enum):
    Non = 'non'
    Inp = 'inp'
    New = 'new'


class EncmixBertCfg(BaseModel):
    inp_len: int
    d_model: int
    pretrained_model_name: str = ''
    tokenizer_name: str = ''
    out_embs_type: EncmixOutEmbsType = EncmixOutEmbsType.Non
    token_types_for_embs: bool = False


class EncmixTrainDsType(str, Enum):
    Msk = 'msk'
    Qna = 'qna'
    Sub = 'sub'


class EncmixModelType(str, Enum):
    One = 'one'
    Sep = 'sep'


class GenmixEmbAggType(str, Enum):
    Fst = 'fst'
    Avg = 'avg'
    Mat = 'mat'


class GenmixEmbExpType(str, Enum):
    Non = 'non'
    Mat = 'mat'
    Mtb = 'mtb'


class GenmixBertCfg(BaseModel):
    inp_len: int
    d_model: int
    pretrained_model_name: str = ''
    tokenizer_name: str = ''
    max_inp_chunks: int
    max_out_toks: int
    n_first_embs: int = -1
    n_second_embs: int = -1
    emb_agg_type: GenmixEmbAggType
    emb_exp_type: GenmixEmbExpType


class GenmixTrainDsType(str, Enum):
    Qna = 'qna'
    Sum = 'sum'
    Wki = 'wki'


class BertModelType(str, Enum):
    Enc = 'enc'
    EncDec = 'encdec'
    Dec = 'dec'


class TokensAggType(str, Enum):
    Bert = 'brt'
    Pyramid = 'pyr'
    Conv = 'cnv'


class BertAggType(str, Enum):
    Sep = 'sep'
    Topcos = 'topcos'
    Topdot = 'topdot'


class CtxQuePromptType(str, Enum):
    Tok = 'tok'
    Cq = 'cq'
    Qc = 'qc'
    Cqqc = 'cqqc'


class SelfSuperviseType(str, Enum):
    Input = 'inp'
    NextSent = 'nxtsnt'
    NextTok = 'nxttok'


class EncoderConvCfg(BaseModel):
    n_levels: int
    n_layers_per_level: int
    d_model: int
    conv_kernel_size: int
    pool_kernel_size: int
    pool_stride: int
    dropout_rate: float
    share_layer_weights: bool = False


class DecExpertType(str, Enum):
    Non = 'non'
    Ttid = 'ttid'
    Moe = 'moe'


class GenmixembCfg(BaseModel):
    model_name: str
    bert_model_type: BertModelType = BertModelType.EncDec
    bert_attention_prob_dropout_prob: float = 0.1
    bert_hidden_dropout_prob: float = 0.1
    gpt2_embd_pdrop: float = 0.1
    gpt2_attn_pdrop: float = 0.1
    gpt2_resid_pdrop: float = 0.1

    d_model: int
    max_inp_toks: int
    max_out_toks: int
    toks_agg_type: TokensAggType
    bert_agg_model_name: str = 'bert-base-uncased'
    bert_agg_type: BertAggType = BertAggType.Sep
    bert_agg_n_subseq_toks: int
    pyr_agg_type: HgReductType = HgReductType.Decim
    pyr_agg_step: int = 0
    pyr_agg_n_levels: int
    pyr_agg_n_layers_per_level: int
    pyr_share_layer_weights: bool = False

    cnv_n_levels: int = 0
    cnv_n_layers_per_level: int = 0
    cnv_conv_kernel_size: int = 0
    cnv_pool_kernel_size: int = 0
    cnv_pool_stride: int = 0
    cnv_share_layer_weights: bool = False

    train_agg_model: bool
    share_agg_enc_token_embeds: bool = False
    add_token_type_ids: bool = False
    join_ctx_que_agg: bool = False
    ctx_que_prompt_type: CtxQuePromptType = CtxQuePromptType.Tok

    dec_expert_type: DecExpertType = DecExpertType.Non
    moe_experts_num: int = 0
    moe_topk: int = 0

    dp_prob: float = 0.1

    @property
    def is_bert(self) -> bool:
        return self.model_name.startswith('bert')

    @property
    def is_gpt2(self) -> bool:
        return self.model_name.startswith('gpt2')


MLP_LAYERS_PAT = re.compile(r'^(?P<size>\d+)(?P<bias>b)?|(?P<act>[a-z]\w+)$')


@dataclass(kw_only=True)
class ParsedMlpLayer:
    size: int = 0
    bias: bool = False
    act: str = ''

def parse_mlp_layers(s: str) -> list[ParsedMlpLayer]:
    res = []
    if not s:
        return res
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
) -> EncdecBertCfg:
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
    pad_token_id = bert_cfg.pad_token_id

    tokenizer_name = tokenizer_name or pretrained_model_name
    cfg_enc = EncBertCfg(
        inp_len=inp_len, d_model=d_model, pad_token_id=pad_token_id, pretrained_model_name=pretrained_model_name,
        tokenizer_name=tokenizer_name, emb_type=emb_type,
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


def create_encdec_graph_bert_cfg(
        pretrained_model_name: str = 'bert-base-uncased', tokenizer_name: str = '', emb_type: BertEmbType = BertEmbType.Cls,
        inp_len = 128, dec_enhance_type: HgEnhanceType = HgEnhanceType.Matmul,
        dec_n_layers: int = 7, dec_n_similar_layers: int = 1, dec_dropout_rate: float = 0.0, dec_temperature: float = 0,
        share_enc_dec_proj_weights: bool = False, middle_type: EncdecMiddleType = EncdecMiddleType.Graph,
        n_graph_layers: int = 1, gnn_hidden_dim: int = 0, gnn_conv_name: str = 'GCNConv', gnn_conv_params: Optional[Dict[str, Any]] = None,
        n_emb_attn_layers: int = 2, emb_mlp_window_size: int = 3, emb_mlp_n_window_layers: int = 1, emb_mlp_n_out_layers: int = 1,
        emb_mlp_act_fn: str = 'gelu', pretrained_model_path: Optional[Path] = None, mask_cfg: Optional[MaskCfg] = None,
        cite_toks_target_weight: float = 1.0, cite_toks_target_type: EncdecCiteToksTargetType = EncdecCiteToksTargetType.All, cite_toks_target_scale: float = 1.0,
        cite_embs_target_weight: float = 1.0, cite_embs_target_type: EncdecCiteEmbsTargetType = EncdecCiteEmbsTargetType.R2, cite_embs_target_scale: float = 1.0,
        input_toks_target_weight: float = 1.0, input_toks_target_scale: float = 1.0, learning_rate: float = 1e-4, optimizer_name: str = 'AdamW',
        optimizer_params: Optional[Dict[str, Any]] = None, lrs_name: str = 'ReduceLROnPlateau',
        lrs_params: Optional[Dict[str, Any]] = None, batch_size: int = 10,
) -> EncdecGraphBertCfg:
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
    pad_token_id = bert_cfg.pad_token_id

    tokenizer_name = tokenizer_name or pretrained_model_name
    cfg_enc = EncBertCfg(
        inp_len=inp_len, d_model=d_model, pad_token_id=pad_token_id, pretrained_model_name=pretrained_model_name,
        tokenizer_name=tokenizer_name, emb_type=emb_type,
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
    gnn_conv_params = create_cls_params(gnn_conv_name, gnn_conv_name_to_defaults, gnn_conv_params)
    cfg_gnn_conv = PyClassCfg(
        module_path='torch_geometric.nn.conv',
        cls_name=gnn_conv_name,
        params=gnn_conv_params,
    )
    if gnn_hidden_dim <= 0:
        gnn_hidden_dim = d_model
    cfg_graph = EmbGraphCfg(
        n_layers=n_graph_layers, d_model=d_model, hidden_dim=gnn_hidden_dim,
        gnn_conv=cfg_gnn_conv,
    )

    cfg_attn = EmbAttnCfg(
        d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_inner=d_inner, n_layers=n_emb_attn_layers, dropout_rate=dec_dropout_rate,
    )

    cfg_mlp = EmbMlpCfg(
        d_model=d_model, d_out=cfg_dec.d_model, window_size=emb_mlp_window_size,
        n_window_layers=emb_mlp_n_window_layers, n_out_layers=emb_mlp_n_out_layers, act_fn=emb_mlp_act_fn,
    )

    cfg_train = EncdecTrainCfg(
        pretrained_model_path=pretrained_model_path, mask_cfg=mask_cfg,
        cite_toks_target_weight=cite_toks_target_weight, cite_toks_target_type=cite_toks_target_type, cite_toks_target_scale=cite_toks_target_scale,
        cite_embs_target_weight=cite_embs_target_weight, cite_embs_target_type=cite_embs_target_type, cite_embs_target_scale=cite_embs_target_scale,
        input_toks_target_weight=input_toks_target_weight, input_toks_target_scale=input_toks_target_scale,
        learning_rate=learning_rate,
        optimizer=PyClassCfg(
            module_path='torch.optim',
            cls_name=optimizer_name,
            params=optimizer_params or {},
        ),
        learning_rate_scheduler=PyClassCfg(
            module_path='torch.optim.lr_scheduler',
            cls_name=lrs_name,
            params=lrs_params or {},
        ),
        batch_size=batch_size,
    )

    cfg_encdec_bert = EncdecGraphBertCfg(
        enc_bert=cfg_enc, dec_pyr=cfg_dec, share_enc_dec_proj_weights=share_enc_dec_proj_weights,
        middle_type=middle_type, emb_graph=cfg_graph, emb_attn=cfg_attn, emb_mlp=cfg_mlp, train_cfg=cfg_train,
    )
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


def create_ranker_bert_cfg(
        pretrained_model_name: str = 'bert-base-uncased', tokenizer_name: str = '', emb_type: BertEmbType = BertEmbType.Cls,
        inp_len: int = 128, dec_mlp_layers: str = '',
) -> RankerBertCfg:
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
    pad_token_id = bert_cfg.pad_token_id

    tokenizer_name = tokenizer_name or pretrained_model_name
    cfg_enc = EncBertCfg(
        inp_len=inp_len, d_model=d_model, pad_token_id=pad_token_id, pretrained_model_name=pretrained_model_name,
        tokenizer_name=tokenizer_name, emb_type=emb_type,
    )

    cfg_dec = DecRankHgCfg(
        d_model=d_model, mlp_layers=dec_mlp_layers,
    )

    cfg_ranker_bert = RankerBertCfg(enc_bert=cfg_enc, dec_rank=cfg_dec)
    return cfg_ranker_bert


SPEC_TOK_PAT = re.compile(r'^\[[A-Z]+]$')


def load_bert_tokenizer_and_model(pretrained_model_name: str, tokenizer_name: str, emb2_tok_name: str) -> tuple[AutoTokenizer, BertModel]:
    tkz = AutoTokenizer.from_pretrained(tokenizer_name)
    model = BertModel.from_pretrained(pretrained_model_name, torch_dtype=torch.float32)
    if emb2_tok_name:
        assert SPEC_TOK_PAT.match(emb2_tok_name), f'Token name {emb2_tok_name} does not match {SPEC_TOK_PAT} pattern.'
        tkz.add_special_tokens({
            'additional_special_tokens': [emb2_tok_name],
        })
        model.resize_token_embeddings(len(tkz))
    return tkz, model


def create_encdecrnk_bert_cfg(
        pretrained_model_name: str = 'bert-base-uncased', tokenizer_name: str = '', emb_type: BertEmbType = BertEmbType.Cls,
        emb2_tok_name: str = '', inp_len = 128, dec_pyr_enhance_type: HgEnhanceType = HgEnhanceType.Matmul,
        dec_pyr_n_layers: int = 7, dec_pyr_n_similar_layers: int = 1, dec_pyr_dropout_rate: float = 0.0, dec_pyr_temperature: float = 0,
        dec_rank_mlp_layers: str = '',
) -> EncdecRankBertCfg:
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
    pad_token_id = bert_cfg.pad_token_id

    tokenizer_name = tokenizer_name or pretrained_model_name
    cfg_enc = EncBertCfg(
        inp_len=inp_len, d_model=d_model, pad_token_id=pad_token_id, pretrained_model_name=pretrained_model_name,
        tokenizer_name=tokenizer_name, emb_type=emb_type, emb2_tok_name=emb2_tok_name,
    )

    step = 2
    if dec_pyr_n_layers == 0:
        dec_pyr_n_layers = math.ceil(math.log(inp_len, step))
    d_inner = d_model * 4
    d_k = d_v = d_model // n_heads
    assert dec_pyr_n_layers > 0, f'n_layers (={dec_pyr_n_layers}) must be > 0'
    cfg_dec = DecPyrCfg(
        d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_inner=d_inner, inp_len=inp_len, step=step, n_layers=dec_pyr_n_layers, dropout_rate=dec_pyr_dropout_rate, n_vocab=n_vocab,
        n_similar_layers=dec_pyr_n_similar_layers, enhance_type=dec_pyr_enhance_type, temperature=dec_pyr_temperature,
    )

    cfg_rank = DecRankHgCfg(
        d_model=d_model, mlp_layers=dec_rank_mlp_layers,
    )

    cfg_encdecrnk_bert = EncdecRankBertCfg(enc_bert=cfg_enc, dec_pyr=cfg_dec, dec_rank=cfg_rank)
    return cfg_encdecrnk_bert


def create_encmix_bert_cfg(
        pretrained_model_name: str = 'bert-base-uncased', tokenizer_name: str = '', inp_len = 128,
        out_embs_type: EncmixOutEmbsType = EncmixOutEmbsType.Non, token_types_for_embs: bool = False,
) -> EncmixBertCfg:
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
    pad_token_id = bert_cfg.pad_token_id

    tokenizer_name = tokenizer_name or pretrained_model_name

    cfg_enc_mix_bert = EncmixBertCfg(
        inp_len=inp_len, d_model=d_model, pretrained_model_name=pretrained_model_name,
        tokenizer_name=tokenizer_name, out_embs_type=out_embs_type, token_types_for_embs=token_types_for_embs,
    )
    return cfg_enc_mix_bert


def create_genmix_bert_cfg(
        pretrained_model_name: str = 'bert-base-uncased', tokenizer_name: str = '', inp_len = 128, max_inp_chunks: int = 0,
        max_out_toks: int = 0, n_first_embs: int = 1, n_second_embs: int = 1, emb_agg_type: GenmixEmbAggType = GenmixEmbAggType.Fst,
        emb_exp_type: GenmixEmbExpType = GenmixEmbExpType.Non,
) -> GenmixBertCfg:
    model = BertModel.from_pretrained(pretrained_model_name, torch_dtype=torch.float32)
    bert_cfg: BertConfig = model.config
    d_model = bert_cfg.hidden_size

    tokenizer_name = tokenizer_name or pretrained_model_name

    cfg_gen_mix_bert = GenmixBertCfg(
        inp_len=inp_len, d_model=d_model, pretrained_model_name=pretrained_model_name,
        tokenizer_name=tokenizer_name, max_inp_chunks=max_inp_chunks, max_out_toks=max_out_toks,
        n_first_embs=n_first_embs, n_second_embs=n_second_embs, emb_agg_type=emb_agg_type, emb_exp_type=emb_exp_type,
    )
    return cfg_gen_mix_bert


def create_genmixemb_cfg(
        model_name: str = 'bert-base-uncased', max_inp_toks: int = 0, max_out_toks: int = 0, toks_agg_type: TokensAggType = TokensAggType.Bert,
        bert_agg_type: BertAggType = BertAggType.Sep, bert_agg_model_name: str = 'bert-base-uncased', bert_agg_n_subseq_toks: int = 0, pyr_agg_type: HgReductType = HgReductType.Decim,
        pyr_agg_step: int = 2, pyr_agg_n_levels: int = 0, pyr_agg_n_layers_per_level: int = 0, pyr_share_layer_weights: bool = False,
        cnv_n_levels: int = 0, cnv_n_layers_per_level: int = 0, cnv_conv_kernel_size: int = 0, cnv_pool_kernel_size: int = 0,
        cnv_pool_stride: int = 0, cnv_share_layer_weights: bool = False, train_agg_model: bool = False, add_token_type_ids: bool = False,
        share_agg_enc_token_embeds: bool = False, join_ctx_que_agg: bool = False, ctx_que_prompt_type: CtxQuePromptType = CtxQuePromptType.Tok,
        dec_expert_type: DecExpertType = DecExpertType.Non, moe_experts_num: int = 0, moe_topk: int = 0, bert_model_type: BertModelType = BertModelType.EncDec,
        bert_attention_prob_dropout_prob: float = 0.1,
        bert_hidden_dropout_prob: float = 0.1, gpt2_embd_pdrop: float = 0.1, gpt2_attn_pdrop: float = 0.1, gpt2_resid_pdrop: float = 0.1,
) -> GenmixembCfg:
    if model_name.startswith('bert'):
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

        model = BertModel.from_pretrained(model_name, torch_dtype=torch.float32)
        bert_cfg: BertConfig = model.config
        d_model = bert_cfg.hidden_size
    elif model_name.startswith('gpt2'):
        # GPT2Config
        # {
        #     "_attn_implementation_autoset": true,
        #     "activation_function": "gelu_new",
        #     "architectures": [
        #         "GPT2LMHeadModel"
        #     ],
        #     "attn_pdrop": 0.1,
        #     "bos_token_id": 50256,
        #     "embd_pdrop": 0.1,
        #     "eos_token_id": 50256,
        #     "initializer_range": 0.02,
        #     "layer_norm_epsilon": 1e-05,
        #     "model_type": "gpt2",
        #     "n_ctx": 1024,
        #     "n_embd": 768,
        #     "n_head": 12,
        #     "n_inner": null,
        #     "n_layer": 12,
        #     "n_positions": 1024,
        #     "reorder_and_upcast_attn": false,
        #     "resid_pdrop": 0.1,
        #     "scale_attn_by_inverse_layer_idx": false,
        #     "scale_attn_weights": true,
        #     "summary_activation": null,
        #     "summary_first_dropout": 0.1,
        #     "summary_proj_to_labels": true,
        #     "summary_type": "cls_index",
        #     "summary_use_proj": true,
        #     "task_specific_params": {
        #         "text-generation": {
        #             "do_sample": true,
        #             "max_length": 50
        #         }
        #     },
        #     "torch_dtype": "float32",
        #     "transformers_version": "4.51.3",
        #     "use_cache": true,
        #     "vocab_size": 50257
        # }

        model = GPT2LMHeadModel.from_pretrained(model_name)
        gpt2_cfg: GPT2Config = model.config
        d_model = gpt2_cfg.n_embd
    else:
        raise Exception(f'Model name {model_name} is not supported. Supported models are bert-* and gpt2-*')

    cfg = GenmixembCfg(
        model_name=model_name, d_model=d_model, max_inp_toks=max_inp_toks, max_out_toks=max_out_toks, toks_agg_type=toks_agg_type,
        bert_agg_type=bert_agg_type, bert_agg_model_name=bert_agg_model_name, bert_agg_n_subseq_toks=bert_agg_n_subseq_toks,
        pyr_agg_type=pyr_agg_type, pyr_agg_step=pyr_agg_step, pyr_agg_n_levels=pyr_agg_n_levels,
        pyr_agg_n_layers_per_level=pyr_agg_n_layers_per_level, pyr_share_layer_weights=pyr_share_layer_weights, cnv_n_levels=cnv_n_levels,
        cnv_n_layers_per_level=cnv_n_layers_per_level, cnv_conv_kernel_size=cnv_conv_kernel_size, cnv_pool_kernel_size=cnv_pool_kernel_size,
        cnv_pool_stride=cnv_pool_stride, cnv_share_layer_weights=cnv_share_layer_weights, train_agg_model=train_agg_model,
        share_agg_enc_token_embeds=share_agg_enc_token_embeds, add_token_type_ids=add_token_type_ids, join_ctx_que_agg=join_ctx_que_agg,
        ctx_que_prompt_type=ctx_que_prompt_type, dec_expert_type=dec_expert_type, moe_experts_num=moe_experts_num,
        bert_model_type=bert_model_type, bert_attention_prob_dropout_prob=bert_attention_prob_dropout_prob, bert_hidden_dropout_prob=bert_hidden_dropout_prob,
        gpt2_embd_pdrop=gpt2_embd_pdrop, gpt2_attn_pdrop=gpt2_attn_pdrop, gpt2_resid_pdrop=gpt2_resid_pdrop
    )
    return cfg


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
        cfg: EncdecBertCfg, pretrained_model_name: Optional[str], emb_type: Optional[BertEmbType] = None, inp_len: int = 0, dec_enhance_type: Optional[HgEnhanceType] = None,
        dec_n_layers: int = 0, dec_n_similar_layers: int = 0, dec_dropout_rate: Optional[float] = None,
        dec_temperature: Optional[float] = None,
) -> EncdecBertCfg:
    enc = cfg.enc_bert
    dec = cfg.dec_pyr
    pretrained_model_name = coalesce(pretrained_model_name, enc.pretrained_model_name)
    tokenizer_name = pretrained_model_name
    emb_type = coalesce(emb_type, cfg.enc_bert.emb_type)
    inp_len = inp_len or enc.inp_len
    dec_enhance_type = coalesce(dec_enhance_type, dec.enhance_type)
    dec_n_layers = dec_n_layers or dec.n_layers
    dec_n_similar_layers = dec_n_similar_layers or dec.n_similar_layers
    dec_dropout_rate = coalesce(dec_dropout_rate, dec.dropout_rate)
    dec_temperature = coalesce(dec_temperature, dec.temperature)

    return create_encdec_bert_cfg(
        pretrained_model_name=pretrained_model_name, tokenizer_name=tokenizer_name, emb_type=emb_type,
        inp_len=inp_len, dec_enhance_type=dec_enhance_type, dec_n_layers=dec_n_layers, dec_n_similar_layers=dec_n_similar_layers,
        dec_dropout_rate=dec_dropout_rate, dec_temperature=dec_temperature,
    )


def copy_override_encdec_graph_bert_cfg(
        cfg: EncdecGraphBertCfg, pretrained_model_name: Optional[str], emb_type: Optional[BertEmbType] = None, inp_len: Optional[int] = None,
        dec_enhance_type: Optional[HgEnhanceType] = None, dec_n_layers: Optional[int] = None, dec_n_similar_layers: Optional[int] = None,
        dec_dropout_rate: Optional[float] = None, dec_temperature: Optional[float] = None, share_enc_dec_proj_weights: Optional[bool] = None,
        middle_type: Optional[EncdecMiddleType] = None,
        n_graph_layers: Optional[int] = None, gnn_hidden_dim: Optional[int] = None, gnn_conv_name: Optional[str] = None, gnn_conv_params: Optional[Dict[str, Any]] = None,
        n_emb_attn_layers: Optional[int] = None, emb_mlp_window_size: Optional[int] = None, emb_mlp_n_window_layers: Optional[int] = None, emb_mlp_n_out_layers: Optional[int] = None,
        emb_mlp_act_fn: Optional[str] = None,
        pretrained_model_path: Optional[Path] = None, mask_cfg: Optional[MaskCfg] = None,
        cite_toks_target_weight: Optional[float] = None, cite_toks_target_type: Optional[EncdecCiteToksTargetType] = None, cite_toks_target_scale: Optional[float] = None,
        cite_embs_target_weight: Optional[float] = None, cite_embs_target_type: Optional[EncdecCiteEmbsTargetType] = None, cite_embs_target_scale: Optional[float] = None,
        input_toks_target_weight: Optional[float] = None, input_toks_target_scale: Optional[float] = None, learning_rate: Optional[float] = None,
        optimizer_name: Optional[str] = None, optimizer_params: Optional[Dict[str, Any]] = None,
        lrs_name: Optional[str] = None, lrs_params: Optional[Dict[str, Any]] = None, batch_size: Optional[int] = None,
) -> EncdecGraphBertCfg:
    enc = cfg.enc_bert
    dec = cfg.dec_pyr
    pretrained_model_name = coalesce(pretrained_model_name, enc.pretrained_model_name)
    tokenizer_name = pretrained_model_name
    emb_type = coalesce(emb_type, cfg.enc_bert.emb_type)
    inp_len = coalesce(inp_len, enc.inp_len)
    dec_enhance_type = coalesce(dec_enhance_type, dec.enhance_type)
    dec_n_layers = coalesce(dec_n_layers, dec.n_layers)
    dec_n_similar_layers = coalesce(dec_n_similar_layers, dec.n_similar_layers)
    dec_dropout_rate = coalesce(dec_dropout_rate, dec.dropout_rate)
    dec_temperature = coalesce(dec_temperature, dec.temperature)
    share_enc_dec_proj_weights = coalesce(share_enc_dec_proj_weights, cfg.share_enc_dec_proj_weights)
    middle_type = coalesce(middle_type, cfg.middle_type)
    n_graph_layers = coalesce(n_graph_layers, cfg.emb_graph.n_layers)
    gnn_hidden_dim = coalesce(gnn_hidden_dim, cfg.emb_graph.hidden_dim)
    gnn_conv_name = coalesce(gnn_conv_name, cfg.emb_graph.gnn_conv.cls_name)
    gnn_conv_params = coalesce(gnn_conv_params, cfg.emb_graph.gnn_conv.params)
    n_emb_attn_layers = coalesce(n_emb_attn_layers, cfg.emb_attn.n_layers)
    emb_mlp_window_size = coalesce(emb_mlp_window_size, cfg.emb_mlp.window_size)
    emb_mlp_n_window_layers = coalesce(emb_mlp_n_window_layers, cfg.emb_mlp.n_window_layers)
    emb_mlp_n_out_layers = coalesce(emb_mlp_n_out_layers, cfg.emb_mlp.n_out_layers)
    emb_mlp_act_fn = coalesce(emb_mlp_act_fn, cfg.emb_mlp.act_fn)
    pretrained_model_path = coalesce(pretrained_model_path, cfg.train_cfg.pretrained_model_path)
    cite_toks_target_weight = coalesce(cite_toks_target_weight, cfg.train_cfg.cite_toks_target_weight)
    cite_toks_target_type = coalesce(cite_toks_target_type, cfg.train_cfg.cite_toks_target_type)
    cite_toks_target_scale = coalesce(cite_toks_target_scale, cfg.train_cfg.cite_toks_target_scale)
    cite_embs_target_weight = coalesce(cite_embs_target_weight, cfg.train_cfg.cite_embs_target_weight)
    cite_embs_target_type = coalesce(cite_embs_target_type, cfg.train_cfg.cite_embs_target_type)
    cite_embs_target_scale = coalesce(cite_embs_target_scale, cfg.train_cfg.cite_embs_target_scale)
    input_toks_target_weight = coalesce(input_toks_target_weight, cfg.train_cfg.input_toks_target_weight)
    input_toks_target_scale = coalesce(input_toks_target_scale, cfg.train_cfg.input_toks_target_scale)
    learning_rate = coalesce(learning_rate, cfg.train_cfg.learning_rate)
    if cfg.train_cfg.optimizer is not None:
        optimizer_name = coalesce(optimizer_name, cfg.train_cfg.optimizer.cls_name)
        optimizer_params = {**(cfg.train_cfg.optimizer.params or {}), **(optimizer_params or {})}
    if cfg.train_cfg.learning_rate_scheduler is not None:
        lrs_name = coalesce(lrs_name, cfg.train_cfg.learning_rate_scheduler.cls_name)
        lrs_params = {**(cfg.train_cfg.learning_rate_scheduler.params or {}), **(lrs_params or {})}
    batch_size = coalesce(batch_size, cfg.train_cfg.batch_size)

    return create_encdec_graph_bert_cfg(
        pretrained_model_name=pretrained_model_name, tokenizer_name=tokenizer_name, emb_type=emb_type,
        inp_len=inp_len, dec_enhance_type=dec_enhance_type, dec_n_layers=dec_n_layers, dec_n_similar_layers=dec_n_similar_layers,
        dec_dropout_rate=dec_dropout_rate, dec_temperature=dec_temperature, share_enc_dec_proj_weights=share_enc_dec_proj_weights,
        middle_type=middle_type,
        n_graph_layers=n_graph_layers, gnn_hidden_dim=gnn_hidden_dim, gnn_conv_name=gnn_conv_name, gnn_conv_params=gnn_conv_params,
        n_emb_attn_layers=n_emb_attn_layers, emb_mlp_window_size=emb_mlp_window_size, emb_mlp_n_window_layers=emb_mlp_n_window_layers,
        emb_mlp_n_out_layers=emb_mlp_n_out_layers, emb_mlp_act_fn=emb_mlp_act_fn,
        pretrained_model_path=pretrained_model_path, mask_cfg=mask_cfg,
        cite_toks_target_weight=cite_toks_target_weight, cite_toks_target_type=cite_toks_target_type, cite_toks_target_scale=cite_toks_target_scale,
        cite_embs_target_weight=cite_embs_target_weight, cite_embs_target_type=cite_embs_target_type, cite_embs_target_scale=cite_embs_target_scale,
        input_toks_target_weight=input_toks_target_weight, input_toks_target_scale=input_toks_target_scale, learning_rate=learning_rate,
        optimizer_name=optimizer_name, optimizer_params=optimizer_params, lrs_name=lrs_name, lrs_params=lrs_params,
        batch_size=batch_size,
    )


def copy_override_ranker_hg_cfg(
        cfg: RankerHgCfg, inp_len: int = 0, n_similar_layers: int = 1, reduct_type: HgReductType = HgReductType.Matmul,
        pos_enc_type: PosEncType = PosEncType.Num, dec_mlp_layers: Optional[str] = None, temperature: float = -1, dropout_rate: float = -1,
) -> RankerHgCfg:
    enc, dec = cfg.enc_pyr, cfg.dec_rank
    n_vocab = enc.vocab_encoder.n_vocab
    pad_idx = enc.vocab_encoder.pad_idx
    d_model = enc.d_model
    n_heads = enc.n_heads
    d_inner = enc.d_inner
    step = enc.step

    if 0 < inp_len != enc.inp_len:
        assert inp_len & (inp_len - 1) == 0, f'inp_len = {inp_len} is not power of 2'
    else:
        inp_len = enc.inp_len

    if n_similar_layers != enc.n_similar_layers:
        assert n_similar_layers > 0, f'n_similar_layers = {n_similar_layers}, but must be > 0'

    temperature = temperature if temperature >= 0 else enc.temperature
    dropout_rate = dropout_rate if dropout_rate >=0 else enc.dropout_rate
    dec_mlp_layers = coalesce(dec_mlp_layers, dec.mlp_layers)

    return create_ranker_hg_cfg(
        n_vocab=n_vocab, pad_idx=pad_idx, d_model=d_model, n_heads=n_heads, d_inner=d_inner, inp_len=inp_len, step=step,
        dropout_rate=dropout_rate, n_similar_layers=n_similar_layers, reduct_type=reduct_type,
        pos_enc_type=pos_enc_type, dec_mlp_layers=dec_mlp_layers, temperature=temperature,
    )


def copy_override_ranker_bert_cfg(
        cfg: RankerBertCfg, emb_type: Optional[BertEmbType] = None, inp_len: int = 0, dec_mlp_layers: Optional[str] = None,
) -> RankerBertCfg:
    enc = cfg.enc_bert
    dec = cfg.dec_rank
    emb_type = coalesce(emb_type, enc.emb_type)
    inp_len = inp_len or enc.inp_len
    dec_mlp_layers = coalesce(dec_mlp_layers, dec.mlp_layers)

    return create_ranker_bert_cfg(
        pretrained_model_name=enc.pretrained_model_name, tokenizer_name=enc.tokenizer_name,
        emb_type=emb_type, inp_len=inp_len, dec_mlp_layers=dec_mlp_layers,
    )


def copy_override_encdecrnk_bert_cfg(
        cfg: EncdecBertCfg, emb_type: Optional[BertEmbType] = None, inp_len: int = 0, dec_pyr_enhance_type: Optional[HgEnhanceType] = None,
        dec_pyr_n_layers: int = 0, dec_pyr_n_similar_layers: int = 0, dec_pyr_dropout_rate: Optional[float] = None,
        dec_pyr_temperature: Optional[float] = None, dec_rank_mlp_layers: Optional[str] = None,
) -> EncdecRankBertCfg:
    enc = cfg.enc_bert
    dec = cfg.dec_pyr
    emb_type = coalesce(emb_type, cfg.enc_bert.emb_type)
    inp_len = inp_len or enc.inp_len
    dec_pyr_enhance_type = coalesce(dec_pyr_enhance_type, dec.enhance_type)
    dec_pyr_n_layers = dec_pyr_n_layers or dec.n_layers
    dec_pyr_n_similar_layers = dec_pyr_n_similar_layers or dec.n_similar_layers
    dec_pyr_dropout_rate = coalesce(dec_pyr_dropout_rate, dec.dropout_rate)
    dec_pyr_temperature = coalesce(dec_pyr_temperature, dec.temperature)
    dec_rank_mlp_layers = coalesce(dec_rank_mlp_layers, dec.mlp_layers)

    return create_encdecrnk_bert_cfg(
        pretrained_model_name=enc.pretrained_model_name, tokenizer_name=enc.tokenizer_name, emb_type=emb_type,
        inp_len=inp_len, dec_pyr_enhance_type=dec_pyr_enhance_type, dec_pyr_n_layers=dec_pyr_n_layers, dec_pyr_n_similar_layers=dec_pyr_n_similar_layers,
        dec_pyr_dropout_rate=dec_pyr_dropout_rate, dec_pyr_temperature=dec_pyr_temperature, dec_rank_mlp_layers=dec_rank_mlp_layers,
    )


def copy_override_encmix_bert_cfg(
        cfg: EncmixBertCfg, inp_len: int = 0, out_embs_type: Optional[EncmixOutEmbsType] = None, token_types_for_embs: Optional[bool] = None,
) -> EncmixBertCfg:
    inp_len = inp_len or cfg.inp_len
    out_embs_type = out_embs_type or cfg.out_embs_type
    token_types_for_embs = coalesce(token_types_for_embs, cfg.token_types_for_embs)

    return create_encmix_bert_cfg(
        pretrained_model_name=cfg.pretrained_model_name, tokenizer_name=cfg.tokenizer_name, inp_len=inp_len,
        out_embs_type=out_embs_type, token_types_for_embs=token_types_for_embs,
    )


def copy_override_genmix_bert_cfg(
        cfg: GenmixBertCfg, pretrained_model_name: Optional[str] = None, tokenizer_name: Optional[str] = None, inp_len: int = 0,
        max_inp_chunks: Optional[int] = None, max_out_toks: Optional[int] = None, n_first_embs: int = -1, n_second_embs: int = -1,
        emb_agg_type: Optional[GenmixEmbAggType] = None, emb_exp_type: Optional[GenmixEmbExpType] = None,
) -> GenmixBertCfg:
    pretrained_model_name = coalesce(pretrained_model_name, cfg.pretrained_model_name)
    tokenizer_name = coalesce(tokenizer_name, cfg.tokenizer_name)
    inp_len = inp_len or cfg.inp_len
    max_inp_chunks = coalesce(max_inp_chunks, cfg.max_inp_chunks)
    max_out_toks = coalesce(max_out_toks, cfg.max_out_toks)
    n_first_embs = n_first_embs if n_first_embs >= 0 else cfg.n_first_embs
    n_second_embs = n_second_embs if n_second_embs >= 0 else cfg.n_second_embs
    emb_agg_type = coalesce(emb_agg_type, cfg.emb_agg_type)
    emb_exp_type = coalesce(emb_exp_type, cfg.emb_exp_type)

    return create_genmix_bert_cfg(
        pretrained_model_name=pretrained_model_name, tokenizer_name=tokenizer_name, inp_len=inp_len,
        max_inp_chunks=max_inp_chunks, max_out_toks=max_out_toks, n_first_embs=n_first_embs, n_second_embs=n_second_embs,
        emb_agg_type=emb_agg_type, emb_exp_type=emb_exp_type,
    )


def copy_override_genmixemb_cfg(
        cfg: GenmixembCfg, model_name: str = '', max_inp_toks: Optional[int] = None, max_out_toks: Optional[int] = None,
        toks_agg_type: Optional[TokensAggType] = None, bert_agg_type: Optional[BertAggType] = None, bert_agg_model_name: Optional[str] = None, bert_agg_n_subseq_toks: Optional[int] = None,
        pyr_agg_type: Optional[HgReductType] = None, pyr_agg_step: Optional[int] = None, pyr_agg_n_levels: Optional[int] = None,
        pyr_agg_n_layers_per_level: Optional[int] = None, pyr_share_layer_weights: Optional[bool] = None, cnv_n_levels: Optional[int] = None,
        cnv_n_layers_per_level: Optional[int] = None, cnv_conv_kernel_size: Optional[int] = None, cnv_pool_kernel_size: Optional[int] = None,
        cnv_pool_stride: Optional[int] = None, cnv_share_layer_weights: Optional[bool] = None, train_agg_model: Optional[bool] = None,
        share_agg_enc_token_embeds: Optional[bool] = None, add_token_type_ids: Optional[bool] = None, join_ctx_que_agg: Optional[bool] = None,
        ctx_que_prompt_type: Optional[CtxQuePromptType] = None, dec_expert_type: Optional[DecExpertType] = None, moe_experts_num: Optional[int] = None,
        moe_topk: Optional[int] = None, bert_model_type: Optional[BertModelType] = None, bert_attention_prob_dropout_prob: Optional[float] = None, bert_hidden_dropout_prob: Optional[float] = None,
        gpt2_embd_pdrop: Optional[float] = None, gpt2_attn_pdrop: Optional[float] = None, gpt2_resid_pdrop: Optional[float] = None,
) -> GenmixembCfg:
    model_name = model_name or cfg.model_name
    max_inp_toks = coalesce(max_inp_toks, cfg.max_inp_toks)
    max_out_toks = coalesce(max_out_toks, cfg.max_out_toks)
    toks_agg_type = coalesce(toks_agg_type, cfg.toks_agg_type)
    bert_agg_type = coalesce(bert_agg_type, cfg.bert_agg_type)
    bert_agg_model_name = coalesce(bert_agg_model_name, cfg.bert_agg_model_name)
    bert_agg_n_subseq_toks = coalesce(bert_agg_n_subseq_toks, cfg.bert_agg_n_subseq_toks)
    pyr_agg_n_levels = coalesce(pyr_agg_n_levels, cfg.pyr_agg_n_levels)
    pyr_agg_n_layers_per_level = coalesce(pyr_agg_n_layers_per_level, cfg.pyr_agg_n_layers_per_level)
    train_agg_model = coalesce(train_agg_model, cfg.train_agg_model)
    pyr_agg_type = coalesce(pyr_agg_type, cfg.pyr_agg_type)
    pyr_agg_step = coalesce(pyr_agg_step, cfg.pyr_agg_step)
    pyr_share_layer_weights = coalesce(pyr_share_layer_weights, cfg.pyr_share_layer_weights)
    share_agg_enc_token_embeds = coalesce(share_agg_enc_token_embeds, cfg.share_agg_enc_token_embeds)
    add_token_type_ids = coalesce(add_token_type_ids, cfg.add_token_type_ids)
    join_ctx_que_agg = coalesce(join_ctx_que_agg, cfg.join_ctx_que_agg)
    ctx_que_prompt_type = coalesce(ctx_que_prompt_type, cfg.ctx_que_prompt_type)
    cnv_n_levels = coalesce(cnv_n_levels, cfg.cnv_n_levels)
    cnv_n_layers_per_level = coalesce(cnv_n_layers_per_level, cfg.cnv_n_layers_per_level)
    cnv_conv_kernel_size = coalesce(cnv_conv_kernel_size, cfg.cnv_conv_kernel_size)
    cnv_pool_kernel_size = coalesce(cnv_pool_kernel_size, cfg.cnv_pool_kernel_size)
    cnv_pool_stride = coalesce(cnv_pool_stride, cfg.cnv_pool_stride)
    cnv_share_layer_weights = coalesce(cnv_share_layer_weights, cfg.cnv_share_layer_weights)
    dec_expert_type = coalesce(dec_expert_type, cfg.dec_expert_type)
    moe_experts_num = coalesce(moe_experts_num, cfg.moe_experts_num)
    moe_topk = coalesce(moe_topk, cfg.moe_topk)
    bert_model_type = coalesce(bert_model_type, cfg.bert_model_type)
    bert_attention_prob_dropout_prob = coalesce(bert_attention_prob_dropout_prob, cfg.bert_attention_prob_dropout_prob)
    bert_hidden_dropout_prob = coalesce(bert_hidden_dropout_prob, cfg.bert_hidden_dropout_prob)
    gpt2_embd_pdrop = coalesce(gpt2_embd_pdrop, cfg.gpt2_embd_pdrop)
    gpt2_attn_pdrop = coalesce(gpt2_attn_pdrop, cfg.gpt2_attn_pdrop)
    gpt2_resid_pdrop = coalesce(gpt2_resid_pdrop, cfg.gpt2_resid_pdrop)

    return create_genmixemb_cfg(
        model_name=model_name, max_inp_toks=max_inp_toks, max_out_toks=max_out_toks, toks_agg_type=toks_agg_type,
        bert_agg_type=bert_agg_type, bert_agg_model_name=bert_agg_model_name, bert_agg_n_subseq_toks=bert_agg_n_subseq_toks,
        pyr_agg_type=pyr_agg_type, pyr_agg_step=pyr_agg_step, pyr_agg_n_levels=pyr_agg_n_levels, pyr_agg_n_layers_per_level=pyr_agg_n_layers_per_level,
        pyr_share_layer_weights=pyr_share_layer_weights, cnv_n_levels=cnv_n_levels, cnv_n_layers_per_level=cnv_n_layers_per_level, cnv_conv_kernel_size=cnv_conv_kernel_size,
        cnv_pool_kernel_size=cnv_pool_kernel_size, cnv_pool_stride=cnv_pool_stride, cnv_share_layer_weights=cnv_share_layer_weights,
        train_agg_model=train_agg_model, share_agg_enc_token_embeds=share_agg_enc_token_embeds, add_token_type_ids=add_token_type_ids,
        join_ctx_que_agg=join_ctx_que_agg, ctx_que_prompt_type=ctx_que_prompt_type, dec_expert_type=dec_expert_type, moe_experts_num=moe_experts_num,
        moe_topk=moe_topk, bert_model_type=bert_model_type, bert_attention_prob_dropout_prob=bert_attention_prob_dropout_prob, bert_hidden_dropout_prob=bert_hidden_dropout_prob,
        gpt2_embd_pdrop=gpt2_embd_pdrop, gpt2_attn_pdrop=gpt2_attn_pdrop, gpt2_resid_pdrop=gpt2_resid_pdrop,
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


def gen_prefpostfix_encdec_bert(
        model_cfg: EncdecBertCfg, mask_cfg: Optional[MaskCfg],
        pretrained_model_path: Optional[Path] = None, next_tok_pred: bool = False,
    ) -> tuple[str, str]:
    prefix, postfix_parts = f'encdecbert', []
    enc, dec = model_cfg.enc_bert, model_cfg.dec_pyr

    if pretrained_model_path is not None:
        dname = pretrained_model_path.parent.name
        m = checkpoint_fname_pat.match(dname)
        assert m is not None, f'Cannot parse checkpoint filename "{dname}". Expected format: <prefix>-YYYYMMDD_HHmmSS-<postfix>'
        postfix_parts.append(f'pre_{m.group(1)}{m.group(2)}{m.group(3)}')

    brt_str = enc.pretrained_model_name.replace('-', '')
    tkz_name = enc.tokenizer_name.replace('-', '')
    if brt_str != tkz_name:
        brt_str = f'{brt_str}-{tkz_name}'
    postfix_parts.append(brt_str)

    postfix_parts.append(f'd{enc.d_model}')
    postfix_parts.append(f'emb{enc.emb_type.value.capitalize()}')
    postfix_parts.append(f'inp{dec.inp_len}')
    postfix_parts.append(f'lrs{dec.n_layers}x{dec.n_similar_layers}')
    postfix_parts.append(f'enh{dec.enhance_type.value.capitalize()}')
    postfix_parts.append(f'step{dec.step}')
    postfix_parts.append(f'h{dec.n_heads}')

    if mask_cfg is not None:
        sep_freq, sep_frac = np.round(mask_cfg.sep_freq, 2), np.round(mask_cfg.sep_frac, 2)
        seq_freq, seq_max_frac = np.round(mask_cfg.seq_freq, 2), np.round(mask_cfg.seq_max_frac, 2)
        postfix_parts.append(f'msk_sep_{sep_freq}x{sep_frac}_seq_{seq_freq}x{seq_max_frac}x{mask_cfg.seq_max_len}_last_{mask_cfg.n_last_toks}')

    if next_tok_pred:
        postfix_parts.append('ntp')

    dp_rate = np.round(dec.dropout_rate, 2)
    if dp_rate < 1e-6:
        dp_rate = 0
    postfix_parts.append(f'dp{dp_rate}')

    temp = np.round(dec.temperature, 2)
    postfix_parts.append(f't{temp}')

    postfix = '-'.join(postfix_parts)
    return prefix, postfix


def gen_prefpostfix_encdec_graph_bert(model_cfg: EncdecGraphBertCfg) -> tuple[str, str]:
    prefix, postfix_parts = f'encdecgraphbert', []
    enc, dec, graph, attn, mlp = model_cfg.enc_bert, model_cfg.dec_pyr, model_cfg.emb_graph, model_cfg.emb_attn, model_cfg.emb_mlp
    train = model_cfg.train_cfg

    pretrained_model_path = None
    if train is not None:
        pretrained_model_path, mask_cfg = train.pretrained_model_path, train.mask_cfg

    if pretrained_model_path is not None:
        dname = pretrained_model_path.parent.name
        m = checkpoint_fname_pat.match(dname)
        assert m is not None, f'Cannot parse checkpoint filename "{dname}". Expected format: <prefix>-YYYYMMDD_HHmmSS-<postfix>'
        postfix_parts.append(f'pre_{m.group(1)}{m.group(2)}{m.group(3)}')

    brt_str = enc.pretrained_model_name.replace('-', '')
    tkz_name = enc.tokenizer_name.replace('-', '')
    if brt_str != tkz_name:
        brt_str = f'{brt_str}-{tkz_name}'
    postfix_parts.append(brt_str)

    postfix_parts.append(f'd{enc.d_model}')
    postfix_parts.append(f'emb{enc.emb_type.value.capitalize()}')
    postfix_parts.append(f'inp{dec.inp_len}')
    postfix_parts.append(f'enh{dec.enhance_type.value.capitalize()}')
    postfix_parts.append(f'lrs{dec.n_layers}x{dec.n_similar_layers}')
    postfix_parts.append(f'step{dec.step}')
    postfix_parts.append(f'h{dec.n_heads}')
    postfix_parts.append(bool_param_to_str('sw', model_cfg.share_enc_dec_proj_weights))

    if mask_cfg is not None:
        sep_freq, sep_frac = np.round(mask_cfg.sep_freq, 2), np.round(mask_cfg.sep_frac, 2)
        seq_freq, seq_max_frac = np.round(mask_cfg.seq_freq, 2), np.round(mask_cfg.seq_max_frac, 2)
        postfix_parts.append(f'msk_sep_{sep_freq}x{sep_frac}_seq_{seq_freq}x{seq_max_frac}x{mask_cfg.seq_max_len}_last_{mask_cfg.n_last_toks}')

    dp_rate = np.round(dec.dropout_rate, 2)
    if dp_rate < 1e-6:
        dp_rate = 0
    postfix_parts.append(f'dp{dp_rate}')

    if dec.temperature > 0:
        temp = np.round(dec.temperature, 2)
        postfix_parts.append(f't{temp}')

    if model_cfg.middle_type == EncdecMiddleType.Graph:
        graph_parts = [f'graph_lrs{graph.n_layers}']
        if graph.hidden_dim != enc.d_model:
            graph_parts.append(f'hid{graph.hidden_dim}')
        gnn_str = cls_cfg_to_str(graph.gnn_conv, gnn_conv_param_to_short_str)
        graph_parts.append(gnn_str)
        postfix_parts.append('_'.join(graph_parts))
    elif model_cfg.middle_type == EncdecMiddleType.Attn:
        postfix_parts.append(f'embattn_lrs{attn.n_layers}')
    elif model_cfg.middle_type == EncdecMiddleType.Mlp:
        postfix_parts.append(f'embmlp_win{mlp.window_size}_wlrs{mlp.n_window_layers}_olrs{mlp.n_out_layers}_act{mlp.act_fn}')
    else:
        raise Exception(f'Unsupported middle_type = {model_cfg.middle_type}')

    if train is not None:
        train_parts = ['trn']
        if train.cite_toks_target_weight > 0:
            ctok = np.round(train.cite_toks_target_weight, 2)
            train_parts.append(f'ctok{train.cite_toks_target_type.value.capitalize()}_scl{train.cite_toks_target_scale}_w{ctok}')
        if train.cite_embs_target_weight > 0:
            cemb = np.round(train.cite_embs_target_weight, 2)
            train_parts.append(f'cemb{train.cite_embs_target_type.value.capitalize()}_scl{train.cite_embs_target_scale}_w{cemb}')
        if train.input_toks_target_weight > 0:
            itok = np.round(train.input_toks_target_weight, 2)
            train_parts.append(f'itok_scl{train.input_toks_target_scale}_w{itok}')

        lr = np.round(train.learning_rate, 9)
        train_parts.append(f'lr{lr}')

        # if train.optimizer is not None:
        #     opt_parts = ['opt']
        #     opt_str = cls_cfg_to_str(train.optimizer, optimizer_param_to_short_str)
        #     opt_parts.append(opt_str)
        #     train_parts.append('_'.join(opt_parts))
        
        # if train.learning_rate_scheduler is not None:
        #     lrs_parts = ['lrs']
        #     lrs_str = cls_cfg_to_str(train.learning_rate_scheduler, lrs_param_to_short_str)
        #     lrs_parts.append(lrs_str)
        #     train_parts.append('_'.join(lrs_parts))
        
        train_parts.append(f'bs{train.batch_size}')
        
        postfix_parts.append('_'.join(train_parts))

    postfix = '-'.join(postfix_parts)
    return prefix, postfix


def gen_prefpostfix_ranker_hg(model_cfg: RankerHgCfg) -> tuple[str, str]:
    prefix = f'rankerhg'
    enc = model_cfg.enc_pyr
    dec = model_cfg.dec_rank
    dec_mlp_layers = dec.mlp_layers.replace(',', '_')
    dec_mlp_layers = dec_mlp_layers or 'none'

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


def gen_prefpostfix_ranker_bert(model_cfg: RankerBertCfg) -> tuple[str, str]:
    prefix = f'rankerbert'
    enc = model_cfg.enc_bert
    dec = model_cfg.dec_rank
    dec_mlp_layers = dec.mlp_layers.replace(',', '_')
    dec_mlp_layers = dec_mlp_layers or 'none'

    brt_str = enc.pretrained_model_name.replace('_', '_')
    if enc.tokenizer_name != enc.pretrained_model_name:
        tkz_name = enc.tokenizer_name.replace('-', '_')
        brt_str = f'{brt_str}-{tkz_name}'

    postfix = f'{brt_str}-inp{enc.inp_len}-d{dec.d_model}-emb_{enc.emb_type}-dmlp_{dec_mlp_layers}'
    return prefix, postfix


def gen_prefpostfix_encdecrnk_bert(model_cfg: EncdecRankBertCfg) -> tuple[str, str]:
    prefix = f'encdecrnkbert'
    enc = model_cfg.enc_bert
    dec_pyr = model_cfg.dec_pyr
    dec_rank = model_cfg.dec_rankr

    brt_str = enc.pretrained_model_name.replace('_', '_')
    if enc.tokenizer_name != enc.pretrained_model_name:
        tkz_name = enc.tokenizer_name.replace('-', '_')
        brt_str = f'{brt_str}-{tkz_name}'

    dp_rate = np.round(dec_pyr.dropout_rate, 2)
    if dp_rate < 1e-6:
        dp_rate = 0

    temp = np.round(dec_pyr.temperature, 2)

    dec_mlp_layers = dec_rank.mlp_layers.replace(',', '_')
    dec_mlp_layers = dec_mlp_layers or 'none'

    postfix = (f'{brt_str}-d{enc.d_model}-emb_{enc.emb_type}-inp{dec_pyr.inp_len}-lrs{dec_pyr.n_layers}x{dec_pyr.n_similar_layers}-enh_{dec_pyr.enhance_type.value}-'
               f'step{dec_pyr.step}-h{dec_pyr.n_heads}-dp{dp_rate}-t{temp}-dmlp_{dec_mlp_layers}')

    return prefix, postfix


def gen_prefpostfix_encmix_bert(model_cfg: EncmixBertCfg, train_ds_type: Optional[EncmixTrainDsType] = None, encmix_model_type: Optional[EncmixModelType] = None) -> tuple[str, str]:
    prefix, postfix_parts = f'encmixbert', []
    if encmix_model_type is not None:
        prefix = f'{prefix}{encmix_model_type.value}'

    bert_str = model_cfg.pretrained_model_name.replace('_', '_')
    if model_cfg.tokenizer_name != model_cfg.pretrained_model_name:
        tkz_name = model_cfg.tokenizer_name.replace('-', '_')
        bert_str = f'{bert_str}-{tkz_name}'
    postfix_parts.append(bert_str)

    postfix_parts.append(f'd{model_cfg.d_model}')

    postfix_parts.append(f'inp{model_cfg.inp_len}')

    out_embs_type_str = f'oemb_{model_cfg.out_embs_type.value}'
    postfix_parts.append(out_embs_type_str)

    tte = 't' if model_cfg.token_types_for_embs else 'f'
    postfix_parts.append(f'tte_{tte}')

    if train_ds_type is not None:
        postfix_parts.append(f'ds_{train_ds_type.value}')

    postfix = '-'.join(postfix_parts)
    return prefix, postfix


def gen_prefpostfix_genmix_bert(
        model_cfg: GenmixBertCfg, train_ds_type: GenmixTrainDsType, mask_tgt: bool, max_tgt_len_freq: float,
        max_tgt_len: int, pred_tgt_all: bool,
) -> tuple[str, str]:
    prefix, postfix_parts = f'genmixbert', []

    bert_str = model_cfg.pretrained_model_name.replace('_', '_')
    if model_cfg.tokenizer_name != model_cfg.pretrained_model_name:
        tkz_name = model_cfg.tokenizer_name.replace('-', '_')
        bert_str = f'{bert_str}-{tkz_name}'
    postfix_parts.append(bert_str)

    postfix_parts.append(f'd{model_cfg.d_model}')

    postfix_parts.append(f'inp{model_cfg.inp_len}')

    postfix_parts.append(f'ds{train_ds_type.value.capitalize()}')

    if train_ds_type == GenmixTrainDsType.Wki:
        mask_tgt_str = str(mask_tgt)[0]
        postfix_parts.append(f'msktgt{mask_tgt_str}')
        postfix_parts.append(f'msklf{max_tgt_len_freq}')
        postfix_parts.append(f'mskl{max_tgt_len}')
        pred_tgt_all_str = str(pred_tgt_all)[0]
        postfix_parts.append(f'tgtall{pred_tgt_all_str}')


    if model_cfg.max_inp_chunks > 0:
        postfix_parts.append(f'mxi{model_cfg.max_inp_chunks}')

    if model_cfg.max_out_toks > 0 and train_ds_type != GenmixTrainDsType.Wki:
        postfix_parts.append(f'mxo{model_cfg.max_out_toks}')

    postfix_parts.append(f'nfem{model_cfg.n_first_embs}')
    postfix_parts.append(f'nsem{model_cfg.n_second_embs}')
    postfix_parts.append(f'emag{model_cfg.emb_agg_type.value.capitalize()}')
    postfix_parts.append(f'emex{model_cfg.emb_exp_type.value.capitalize()}')

    postfix = '-'.join(postfix_parts)
    return prefix, postfix


def bool_param_to_str(name: str, val: bool) -> str:
    return f'{name}{str(val)[0]}'


def float_param_to_str(name: str, val: float) -> str:
    val_rnd = np.round(val, 2)
    if val_rnd < 1e-6:
        val_rnd = 0
    return f'{name}{val_rnd}'


checkpoint_fname_pat = re.compile(r'^(\w+)-(\d{8})_(\d{6})-.*$')


def gen_prefpostfix_genmixemb(
        cfg: GenmixembCfg, train_ds_type: GenmixTrainDsType, mask_cfg: Optional[MaskCfg], self_supervise_type: Optional[SelfSuperviseType] = None,
        agg_pretrained_model_path: Optional[Path] = None, gen_pretrained_model_path: Optional[Path] = None,
) -> tuple[str, str]:
    prefix, postfix_parts = f'genmixemb', []

    if agg_pretrained_model_path is not None:
        dname = agg_pretrained_model_path.parent.name
        m = checkpoint_fname_pat.match(dname)
        assert m is not None, f'Cannot parse checkpoint filename "{dname}". Expected format: <prefix>-YYYYMMDD_HHmmSS-<postfix>'
        postfix_parts.append(f'preagg_{m.group(1)}{m.group(2)}{m.group(3)}')

    if gen_pretrained_model_path is not None:
        dname = gen_pretrained_model_path.parent.name
        m = checkpoint_fname_pat.match(dname)
        assert m is not None, f'Cannot parse checkpoint filename "{dname}". Expected format: <prefix>-YYYYMMDD_HHmmSS-<postfix>'
        postfix_parts.append(f'pregen_{m.group(1)}{m.group(2)}{m.group(3)}')

    postfix_parts.append(cfg.model_name.replace('-', ''))

    postfix_parts.append(f'd{cfg.d_model}')

    if cfg.is_bert:
        postfix_parts.append(f'mdl{cfg.bert_model_type.value.capitalize()}')
        if cfg.bert_attention_prob_dropout_prob == cfg.bert_hidden_dropout_prob:
            dp_rate_str = float_param_to_str('dp', cfg.bert_attention_prob_dropout_prob)
            postfix_parts.append(dp_rate_str)
        else:
            dp_attn_str = float_param_to_str('dpa', cfg.bert_attention_prob_dropout_prob)
            dp_hid_str = float_param_to_str('dph', cfg.bert_hidden_dropout_prob)
            postfix_parts.append(dp_attn_str)
            postfix_parts.append(dp_hid_str)
    elif cfg.is_gpt2:
        if cfg.gpt2_embd_pdrop == cfg.gpt2_attn_pdrop == cfg.gpt2_resid_pdrop:
            dp_rate_str = float_param_to_str('dp', cfg.gpt2_embd_pdrop)
            postfix_parts.append(dp_rate_str)
        else:
            dp_embd_str = float_param_to_str('dpe', cfg.gpt2_embd_pdrop)
            dp_attn_str = float_param_to_str('dpa', cfg.gpt2_attn_pdrop)
            dp_resid_str = float_param_to_str('dpr', cfg.gpt2_resid_pdrop)
            postfix_parts.append(dp_embd_str)
            postfix_parts.append(dp_attn_str)
            postfix_parts.append(dp_resid_str)
    else:
        raise Exception(f'Model name {cfg.model_name} is not supported. Supported models are bert-* and gpt2-*')

    postfix_parts.append(f'mxi{cfg.max_inp_toks}')
    postfix_parts.append(f'mxo{cfg.max_out_toks}')

    agg_type_str = f'agg{cfg.toks_agg_type.value.capitalize()}'

    agg_enabled = False
    if cfg.toks_agg_type == TokensAggType.Bert:
        if cfg.bert_agg_n_subseq_toks > 0:
            postfix_parts.append(agg_type_str)
            if cfg.bert_agg_model_name != cfg.model_name:
                agg_bert_str = cfg.bert_agg_model_name.replace('-', '')
                postfix_parts.append(agg_bert_str)
            postfix_parts.append(f'sub{cfg.bert_agg_n_subseq_toks}')
            postfix_parts.append(f'agt{cfg.bert_agg_type.value.capitalize()}')
            agg_enabled = True
    elif cfg.toks_agg_type == TokensAggType.Pyramid:
        if cfg.pyr_agg_step > 0 and cfg.pyr_agg_n_levels > 0:
            postfix_parts.append(agg_type_str)
            postfix_parts.append(f'agt{cfg.pyr_agg_type.value.capitalize()}')
            postfix_parts.append(f'stp{cfg.pyr_agg_step}')
            postfix_parts.append(f'lvl{cfg.pyr_agg_n_levels}')
            postfix_parts.append(f'lrs{cfg.pyr_agg_n_layers_per_level}')
            if cfg.pyr_agg_n_levels > 1:
                postfix_parts.append(bool_param_to_str('shl', cfg.pyr_share_layer_weights))
            agg_enabled = True
    elif cfg.toks_agg_type == TokensAggType.Conv:
        if cfg.cnv_pool_stride > 0 and cfg.cnv_n_levels > 0:
            postfix_parts.append(agg_type_str)
            postfix_parts.append(f'lvl{cfg.cnv_n_levels}')
            postfix_parts.append(f'lrs{cfg.cnv_n_layers_per_level}')
            postfix_parts.append(f'cksz{cfg.cnv_conv_kernel_size}')
            postfix_parts.append(f'pksz{cfg.cnv_pool_kernel_size}')
            postfix_parts.append(f'pst{cfg.cnv_pool_stride}')
            if cfg.cnv_n_levels > 1:
                postfix_parts.append(bool_param_to_str('shl', cfg.cnv_share_layer_weights))
            agg_enabled = True
    else:
        raise Exception(f'Tokens aggregation type {cfg.toks_agg_type} is not supported')

    postfix_parts.append(f'ds{train_ds_type.value.capitalize()}')

    if mask_cfg is not None:
        sep_freq, sep_frac = np.round(mask_cfg.sep_freq, 2), np.round(mask_cfg.sep_frac, 2)
        seq_freq, seq_max_frac = np.round(mask_cfg.seq_freq, 2), np.round(mask_cfg.seq_max_frac, 2)
        postfix_parts.append(f'msk_sep_{sep_freq}x{sep_frac}_seq_{seq_freq}x{seq_max_frac}x{mask_cfg.seq_max_len}_last_{mask_cfg.n_last_toks}')

    if agg_enabled:
        postfix_parts.append(bool_param_to_str('trag', cfg.train_agg_model))

    # Convolutional aggregator always shares token embedding layer weights with the encoder
    if agg_enabled and cfg.toks_agg_type != TokensAggType.Conv:
        postfix_parts.append(bool_param_to_str('shem', cfg.share_agg_enc_token_embeds))

    if train_ds_type == GenmixTrainDsType.Wki:
        assert self_supervise_type is not None
        postfix_parts.append(f'sst{self_supervise_type.value.capitalize()}')
    elif train_ds_type == GenmixTrainDsType.Qna:
        postfix_parts.append(bool_param_to_str('ttid', cfg.add_token_type_ids))
        if agg_enabled:
            # postfix_parts.append(bool_param_to_str('jcq', cfg.join_ctx_que_agg))
            postfix_parts.append(f'cqpr{cfg.ctx_que_prompt_type.value.capitalize()}')
    else:
        raise ValueError(f'Dataset type {train_ds_type} is not supported.')

    if cfg.dec_expert_type == DecExpertType.Non:
        pass
    else:
        postfix_parts.append(f'expr{cfg.dec_expert_type.value.capitalize()}')
        if cfg.dec_expert_type == DecExpertType.Ttid:
            pass
        elif cfg.dec_expert_type == DecExpertType.Moe:
            postfix_parts.append(f'exn{cfg.moe_experts_num}')
            postfix_parts.append(f'exk{cfg.moe_topk}')
        else:
            raise ValueError(f'Expert type {cfg.dec_expert_type} is not supported.')


    postfix = '-'.join(postfix_parts)
    return prefix, postfix

