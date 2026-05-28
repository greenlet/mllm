import json
import os
from datetime import timedelta
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional
import shutil

from datasets import Dataset
from pydantic import BaseModel, Field, validator
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
import functools

import torch
import torch.utils.tensorboard as tb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_optimizer_state_dict,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from transformers import AutoTokenizer

from mllm.config.model import MixedDecoderCfg, MixedDecoderDsType, MixedDecoderType, BertEmbType, \
    DecoderDtype, copy_override_mixed_decoder_cfg, gen_prefpostfix_mixed_decoder, parse_decoder_spec, \
    MixedDecoderTrainCfg
from mllm.exp.args import MIXED_DECODER_MODEL_CFG_FNAME, create_bool_str_field, get_pretrained_model_path, is_arg_true, \
    mask_tokens_ARG
from mllm.model.mixed_decoder import MixedDecoder
from mllm.model.losses import LossesStats
from mllm.train.encdec_graph_bert import MaskedCiteDataset, create_masked_cite_dataloader, load_split_wiki_dataset
from mllm.train.mask_utils import MaskCfg
from mllm.train.next_tok_wiki import NextTokWikiDataset, create_next_tok_dataloader, load_split_wiki_for_next
from mllm.data.qna.dataset import QnaDatasetAgg, load_qna_datasets, create_qna_dataloader
from mllm.train.qna_cite import QnaCiteDataset, create_qna_cite_dataloader, load_split_squadv2
from mllm.train.utils import find_create_train_path, log_weights_grads_stats
from mllm.utils.utils import instantiate_torch_lr_scheduler, instantiate_torch_optimizer, parse_dict_str, rethrow


freeze_encoder_ARG = '--freeze-encoder', 'Freeze encoder weights during training'
use_sep_ARG = '--use-sep', 'Insert SEP/EOS embedding between context embeddings and prompt tokens'
prompt_all_ARG = '--prompt-all', 'Target is whole input chunk (true) or citation only (false)'
decoder_only_ARG = '--decoder-only', 'Train decoder without encoder (raw context tokens fed directly to decoder, encoder is not instantiated).'


class ArgsMixedDecoderTrain(BaseModel):
    data_path: Path = Field(
        ...,
        description='Root data path. Must contain subpath `wikipedia/WIKI_DS_NAME` with Wikipedia dataset.',
        cli=('--data-path',),
    )
    wiki_ds_name: str = Field(
        '20200501.en',
        description='Wikipedia dataset name of the format YYYYMMDD.LANG, for example: 20220301.en',
        cli=('--wiki-ds-name',),
    )
    train_root_path: Path = Field(
        ...,
        description='Path to train root directory. New train subdirectory will be created within each new run.',
        cli=('--train-root-path',),
    )
    train_subdir: str = Field(
        '',
        description='Train subdirectory. Can have values: "last", "<subdirectory-name>". When set to "last", '
            'last subdirectory of TRAIN_ROOT_PATH containing training snapshot will be taken.',
        cli=('--train-subdir',)
    )
    model_cfg_fpath: Path = Field(
        ...,
        description='Path to MixedDecoder model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    bert_model_name: str = Field(
        'bert-base-uncased',
        description='Pretrained BERT model name for encoder.',
        cli=('--bert-model-name',),
    )
    bert_emb_type: BertEmbType = Field(
        BertEmbType.Cls,
        description=f'Bert embedding type. Can have values: {list(x.value for x in BertEmbType)}',
        cli=('--bert-emb-type',),
    )
    inp_len: int = Field(
        ...,
        description='Input tokens number (encoder sequence length).',
        cli=('--inp-len',),
    )
    decoder_type: MixedDecoderType = Field(
        MixedDecoderType.Gpt2,
        description=f'Decoder type. Can have values: {list(x.value for x in MixedDecoderType)}',
        cli=('--decoder-type',),
    )
    decoder_model_name: str = Field(
        'gpt2',
        description='Pretrained decoder model name (e.g. "gpt2", "bert-base-uncased").',
        cli=('--decoder-model-name',),
    )
    decoder_spec: str = Field(
        '',
        description='Compound decoder spec of the form `<family>-<size>[-instruct]-<precision>`, '
            'e.g. `qwen2.5-1.5B-fp32`, `qwen2.5-1.5B-instruct-fp16`, `qwen3-0.6B-fp32`, `gpt2-fp32`. '
            'When set, overrides --decoder-type, --decoder-model-name and selects the AMP precision '
            '(fp32: no autocast; fp16: autocast(fp16) + GradScaler; bf16: autocast(bf16)).',
        cli=('--decoder-spec',),
    )
    max_seq_len: int = Field(
        384,
        description='Maximum combined sequence length (context embeddings + sep + prompt tokens + target tokens).',
        cli=('--max-seq-len',),
    )
    emb_exp_rate: int = Field(
        0,
        description='Embedding expansion rate. If > 0, each CLS embedding is linearly expanded from 1 to emb_exp_rate vectors. '
            'If 1, a linear vector-to-vector transform is performed.',
        cli=('--emb-exp-rate',),
    )
    emb_win_min_size: int = Field(
        0,
        description='Minimum embedding window size. Active when emb_win_min_size <= emb_win_max_size > 0.',
        cli=('--emb-win-min-size',),
    )
    emb_win_max_size: int = Field(
        0,
        description='Maximum embedding window size. Active when emb_win_min_size <= emb_win_max_size > 0.',
        cli=('--emb-win-max-size',),
    )
    train_ds_type: MixedDecoderDsType = Field(
        MixedDecoderDsType.Cite,
        description=f'Training dataset type. Can have values: {list(x.value for x in MixedDecoderDsType)}',
        cli=('--train-ds-type',),
    )
    min_next_toks: int = Field(
        64,
        description='Minimum number of tokens reserved for next-token prediction target (used with train_ds_type=next).',
        cli=('--min-next-toks',),
    )

    freeze_encoder_STR: str = create_bool_str_field(*freeze_encoder_ARG)
    @property
    def freeze_encoder(self) -> bool:
        return is_arg_true(freeze_encoder_ARG[0], self.freeze_encoder_STR)

    use_sep_STR: str = create_bool_str_field(*use_sep_ARG)
    @property
    def use_sep(self) -> bool:
        return is_arg_true(use_sep_ARG[0], self.use_sep_STR)

    prompt_all_STR: str = create_bool_str_field(*prompt_all_ARG)
    @property
    def prompt_all(self) -> bool:
        return is_arg_true(prompt_all_ARG[0], self.prompt_all_STR)

    decoder_only_STR: str = create_bool_str_field(*decoder_only_ARG)
    @property
    def decoder_only(self) -> bool:
        return is_arg_true(decoder_only_ARG[0], self.decoder_only_STR)

    mask_tokens_STR: str = create_bool_str_field(*mask_tokens_ARG)
    @property
    def mask_tokens(self) -> bool:
        return is_arg_true(mask_tokens_ARG[0], self.mask_tokens_STR)

    mask_sep_freq: float = Field(
        0.0,
        description='Sparse mask frequency from 0 to 1.',
        cli=('--mask-sep-freq',),
    )
    mask_sep_frac: float = Field(
        0.0,
        description='Fraction of the input to mask using sparse masking.',
        cli=('--mask-sep-frac',),
    )
    mask_seq_freq: float = Field(
        0.0,
        description='Sequential mask frequency from 0 to 1.',
        cli=('--mask-seq-freq',),
    )
    mask_seq_max_frac: float = Field(
        0.0,
        description='Fraction of the input to calculate maximum length of tokens sequence to mask.',
        cli=('--mask-seq-max-frac',),
    )
    mask_seq_max_len: int = Field(
        0,
        description='Maximum length of tokens sequence to mask.',
        cli=('--mask-seq-max-len',),
    )
    mask_n_last_toks: int = Field(
        0,
        description='Number of last tokens to always mask.',
        cli=('--mask-n-last-toks',),
    )

    docs_batch_size: int = Field(
        3,
        description='Documents batch size.',
        cli=('--docs-batch-size',),
    )
    device: str = Field(
        'cpu',
        description='Device to run training on. Can have values: "cpu", "cuda"',
        cli=('--device',)
    )
    epochs: int = Field(
        None,
        description='Number of training epochs.',
        cli=('--epochs',),
    )
    learning_rate: float = Field(
        0.001,
        description='Initial learning rate of the training process.',
        cli=('--learning-rate',)
    )
    learning_rate_override: float = Field(
        0.0,
        description='If > 0, override the current learning rate by rebuilding optimizer and '
            'scheduler from scratch with this LR (any restored optimizer/scheduler state from '
            'checkpoint is discarded; last_epoch and val_loss_min are still honored). '
            'If <= 0, no override is applied.',
        cli=('--learning-rate-override',),
    )
    optimizer_name: str = Field(
        'AdamW',
        description='Optimizer class name.',
        cli=('--optimizer-name',),
    )
    optimizer_params: dict = Field(
        {},
        description='Optimizer class parameters as a dictionary.',
        cli=('--optimizer-params',),
    )
    @validator('optimizer_params', pre=True)
    def parse_optimizer_params(cls, v):
        return parse_dict_str(v, 'optimizer_params')

    weight_decay_decoder: float = Field(
        0.0,
        description='Weight decay applied to decoder matmul params (biases and norms excluded). '
            '0 disables this knob.',
        cli=('--weight-decay-decoder',),
    )
    weight_decay_other: float = Field(
        0.0,
        description='Weight decay applied to non-decoder matmul params (encoder, emb_exp, enc_proj, '
            'pos_emb, ...; biases and norms excluded). 0 disables this knob.',
        cli=('--weight-decay-other',),
    )
    llrd_decay: float = Field(
        1.0,
        description='Layer-wise LR decay multiplier for decoder layers. 1.0 disables LLRD. '
            'lr_l = base_lr * llrd_decay ** (n_layers - 1 - l).',
        cli=('--llrd-decay',),
    )
    attention_dropout: float = Field(
        0.0,
        description='Qwen self-attention dropout probability. 0 leaves the HF default unchanged. '
            'Qwen-only knob; ignored for GPT-2 / BertDec decoders.',
        cli=('--attention-dropout',),
    )
    label_smoothing: float = Field(
        0.0,
        description='Label smoothing for the cross-entropy loss in MixedDecoder.calc_loss. 0 disables.',
        cli=('--label-smoothing',),
    )
    max_grad_norm: float = Field(
        0.0,
        description='Max gradient norm for clipping. 0 disables gradient clipping. '
            'AMP- and FSDP-aware.',
        cli=('--max-grad-norm',),
    )

    learning_rate_scheduler_name: str = Field(
        'ReduceLROnPlateau',
        description='Learning rate scheduler class name.',
        cli=('--learning-rate-scheduler-name',),
    )
    learning_rate_scheduler_params: dict = Field(
        {},
        description='Learning rate scheduler class parameters as a dictionary.',
        cli=('--learning-rate-scheduler-params',),
    )
    @validator('learning_rate_scheduler_params', pre=True)
    def parse_learning_rate_scheduler_params(cls, v):
        return parse_dict_str(v, 'learning_rate_scheduler_params')

    train_epoch_steps: Optional[int] = Field(
        None,
        description='Number of training steps per epoch.',
        cli=('--train-epoch-steps',),
    )
    val_epoch_steps: Optional[int] = Field(
        None,
        description='Number of validation steps per epoch.',
        cli=('--val-epoch-steps',),
    )
    random_seed: Optional[int] = Field(
        None,
        description='Random seed.',
        cli=('--random-seed',),
    )
    pretrained_encdec_model_path: Optional[Path] = Field(
        None,
        description='Path to EncdecBert model train directory (loads encoder only).',
        cli=('--pretrained-encdec-model-path',),
    )
    pretrained_mixed_decoder_model_path: Optional[Path] = Field(
        None,
        description='Path to MixedDecoder model train directory (loads full model with strict mode).',
        cli=('--pretrained-mixed-decoder-model-path',),
    )
    world_size: int = Field(
        1,
        description='Number of GPU instances to use for distributed training.',
        cli=('--world-size',),
    )
    parallel: str = Field(
        'ddp',
        description='Parallelism mode for the model: "ddp" (replicate full model per rank, default) '
                    'or "fsdp" (FullyShardedDataParallel, shards parameters/gradients/optimizer state '
                    'across ranks). Use "fsdp" to fit Qwen2.5-1.5B + BERT on 32GB GPUs.',
        cli=('--parallel',),
    )
    fsdp_shard: str = Field(
        'full',
        description='FSDP sharding strategy when --parallel fsdp: "full" (FULL_SHARD across all ranks, '
                    'minimum memory) or "hybrid" (HYBRID_SHARD: shard within node, replicate across; '
                    'higher throughput, more memory).',
        cli=('--fsdp-shard',),
    )


_PARALLEL_DDP = 'ddp'
_PARALLEL_FSDP = 'fsdp'


def _resolve_transformer_layer_classes(module: nn.Module) -> set:
    """Collect concrete nn.Module classes that should be wrapped together by FSDP.

    Restricted to the *decoder* subtree: walks ``module.decoder`` (the large causal
    LM — Qwen / GPT-2 / BertGenerationDecoder) and aggregates the names listed in
    ``_no_split_modules`` on any ``PreTrainedModel`` ancestor (e.g.
    ``Qwen2DecoderLayer``), then maps those names back to live class objects.

    The encoder (BERT-base, ~110M) is intentionally **excluded** from sharding:
    its layers are small (~7M params each) and sharding them across N ranks gives
    a terrible comm:compute ratio (an all-gather to do ~1 ms of matmul). Leaving
    the encoder unsharded means FSDP replicates it (same as DDP for that subtree)
    while still sharding the large decoder where the memory savings matter.
    """
    decoder = getattr(module, 'decoder', None)
    search_root = decoder if decoder is not None else module
    wanted_names: set = set()
    for sub in search_root.modules():
        names = getattr(sub, '_no_split_modules', None)
        if names:
            wanted_names.update(names)
    found: set = set()
    for sub in search_root.modules():
        if type(sub).__name__ in wanted_names:
            found.add(type(sub))
    return found


_NO_DECAY_NAME_PATTERNS = ('bias', 'LayerNorm.weight', 'layer_norm.weight', 'norm.weight')


def _is_no_decay_param(name: str, param: torch.Tensor) -> bool:
    """A parameter is excluded from weight decay if it is a bias, a norm weight,
    or any 1-D tensor (catches RMSNorm weights named ``.norm.weight`` in Qwen as
    well as legacy ``LayerNorm.weight``). Matches the HF Trainer / Qwen finetune
    convention.
    """
    if param.ndim == 1:
        return True
    return any(p in name for p in _NO_DECAY_NAME_PATTERNS)


def _get_decoder_layers(decoder: nn.Module, decoder_type: 'MixedDecoderType') -> Optional[list]:
    """Return the ordered list of decoder transformer layers for the given decoder
    type, or None if the structure is not recognised (e.g. decoder=None in
    encoder-only mode). Used for LLRD bucketing.
    """
    if decoder is None:
        return None
    if decoder_type == MixedDecoderType.Qwen:
        return list(decoder.model.layers)
    if decoder_type == MixedDecoderType.Gpt2:
        return list(decoder.transformer.h)
    if decoder_type == MixedDecoderType.BertDec:
        return list(decoder.bert.encoder.layer)
    return None


def build_param_groups(
        model: nn.Module, train_cfg: 'MixedDecoderTrainCfg', decoder_type: 'MixedDecoderType', base_lr: float,
) -> list:
    """Build optimizer parameter groups with optional per-bucket weight decay and
    layer-wise LR decay (LLRD) on the decoder.

    Returns a single legacy group ``[{'params': all_params}]`` when none of
    ``weight_decay_decoder``, ``weight_decay_other``, ``llrd_decay`` deviate from
    their defaults — this preserves optimizer-state shape for resume from
    pre-regularization checkpoints.

    Otherwise emits groups split by:
      - bucket: per-decoder-layer index (when LLRD is on) OR ``decoder_nonlayer``
        (decoder embeddings / final norm / lm_head) OR ``other`` (everything not
        inside the decoder).
      - decay vs. no_decay (biases and norms get weight_decay=0 regardless).

    Each group dict carries ``lr``, ``weight_decay``, and a debug ``name`` field.
    """
    wd_dec = train_cfg.weight_decay_decoder
    wd_oth = train_cfg.weight_decay_other
    llrd = train_cfg.llrd_decay
    legacy = (wd_dec == 0.0 and wd_oth == 0.0 and llrd == 1.0)
    if legacy:
        return [{'params': [p for p in model.parameters() if p.requires_grad]}]

    base = model.module if hasattr(model, 'module') else model
    decoder = getattr(base, 'decoder', None)
    decoder_param_ids: set = set(id(p) for p in decoder.parameters()) if decoder is not None else set()

    layers = _get_decoder_layers(decoder, decoder_type)
    n_layers = len(layers) if layers is not None else 0
    # Map param id -> layer index for params inside any decoder layer.
    layer_idx_by_id: Dict[int, int] = {}
    if layers is not None:
        for li, layer in enumerate(layers):
            for p in layer.parameters():
                layer_idx_by_id[id(p)] = li

    # Bucket key -> (decay_params, nodecay_params, lr, name)
    buckets: Dict[str, dict] = {}

    def _bucket(name: str, lr: float, wd: float) -> dict:
        if name not in buckets:
            buckets[name] = {
                'decay': [], 'nodecay': [], 'lr': lr, 'wd_decay': wd, 'wd_nodecay': 0.0,
            }
        return buckets[name]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_dec = id(p) in decoder_param_ids
        is_nd = _is_no_decay_param(n, p)
        if is_dec and id(p) in layer_idx_by_id:
            li = layer_idx_by_id[id(p)]
            # LLRD: layers near the top (closer to output) keep base_lr; deeper layers shrink.
            lr_l = base_lr * (llrd ** (n_layers - 1 - li))
            bk = _bucket(f'dec_L{li}', lr_l, wd_dec)
        elif is_dec:
            # decoder embeddings, final norm, lm_head — keep base_lr.
            bk = _bucket('dec_nonlayer', base_lr, wd_dec)
        else:
            bk = _bucket('other', base_lr, wd_oth)
        (bk['nodecay'] if is_nd else bk['decay']).append(p)

    groups = []
    for name, bk in buckets.items():
        if bk['decay']:
            groups.append({
                'params': bk['decay'], 'lr': bk['lr'], 'weight_decay': bk['wd_decay'],
                'name': f'{name}_decay',
            })
        if bk['nodecay']:
            groups.append({
                'params': bk['nodecay'], 'lr': bk['lr'], 'weight_decay': bk['wd_nodecay'],
                'name': f'{name}_nodecay',
            })
    return groups


def sanitize_optimizer_state_for_model(optimizer: torch.optim.Optimizer, model: nn.Module) -> int:
    """Clear invalid per-parameter optimizer state entries.

    This guards resume from checkpoints whose optimizer state was saved with a
    different parameter ordering/shapes. AdamW foreach kernels require moment
    tensors to have the same shape as the current parameter gradients.
    """
    bad_count = 0
    for pg in optimizer.param_groups:
        for p in pg.get('params', []):
            pstate = optimizer.state.get(p)
            if not isinstance(pstate, dict) or not pstate:
                continue
            bad = False
            for sk in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                sv = pstate.get(sk)
                if torch.is_tensor(sv) and tuple(sv.shape) != tuple(p.shape):
                    bad = True
                    break
            if bad:
                optimizer.state[p] = {}
                bad_count += 1
    return bad_count


def setup(rank, world_size):
    if world_size <= 1:
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Pin this rank to its GPU BEFORE NCCL init. Without this, NCCL may bind
    # multiple ranks to the same device and fail with
    #   ncclInvalidUsage: Duplicate GPU detected : rank N and rank M both on CUDA device ...
    # which then makes any later collective (broadcast_object_list inside
    # FSDP.optim_state_dict_to_load, etc.) crash.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    backend = 'nccl'
    # Default timeout is 30 min. FSDP's FULL_STATE_DICT save path with
    # rank0_only=True + offload_to_cpu=True can keep rank 0 busy for longer than
    # that on multi-billion-param models (Qwen2.5-1.5B + BERT), during which
    # other ranks block in TCPStore.get() while bringing up a new NCCL
    # communicator for all_gather_object inside FSDP.optim_state_dict. When the
    # store times out we get the observed
    #   "store->get('...') got error: Socket Timeout"
    # crash. Bump to 2h to be safely above the slowest checkpoint we have seen.
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size,
        timeout=timedelta(hours=2),
    )


def cleanup():
    if not dist.is_initialized():
        return
    dist.destroy_process_group()


def train(rank: int, ds_train, ds_val, df_sq, sq_inds_train, sq_inds_val, wiki_ds, wiki_inds_train, wiki_inds_val, qna_agg_train, qna_agg_val, args: ArgsMixedDecoderTrain):
    print(f'Running DDP training on rank {rank}.')
    def log(*msgs: Any, forall: bool = False):
        if rank == 0 or forall:
            print(*msgs)

    setup(rank, args.world_size)

    pretrained_encdec_model_path = get_pretrained_model_path(args.pretrained_encdec_model_path)
    pretrained_mixed_decoder_model_path = get_pretrained_model_path(args.pretrained_mixed_decoder_model_path)

    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    log(f'Using device {device}.')

    mask_cfg = None
    if args.mask_tokens:
        mask_cfg = MaskCfg(
            sep_freq=args.mask_sep_freq, sep_frac=args.mask_sep_frac, seq_freq=args.mask_seq_freq, seq_max_frac=args.mask_seq_max_frac,
            seq_max_len=args.mask_seq_max_len, n_last_toks=args.mask_n_last_toks,
        )

    # Resolve compound spec (if provided) into discrete (decoder_type, decoder_model_name, decoder_dtype).
    if args.decoder_spec:
        decoder_type, decoder_model_name, decoder_dtype = parse_decoder_spec(args.decoder_spec)
        log(f'Parsed decoder_spec={args.decoder_spec!r} -> type={decoder_type.value}, '
            f'model={decoder_model_name}, dtype={decoder_dtype.value}')
    else:
        decoder_type = args.decoder_type
        decoder_model_name = args.decoder_model_name
        decoder_dtype = DecoderDtype.Fp32

    model_cfg = parse_yaml_file_as(MixedDecoderCfg, args.model_cfg_fpath)
    model_cfg = copy_override_mixed_decoder_cfg(
        model_cfg, pretrained_model_name=args.bert_model_name, emb_type=args.bert_emb_type,
        inp_len=args.inp_len, decoder_type=decoder_type, decoder_model_name=decoder_model_name,
        decoder_dtype=decoder_dtype,
        max_seq_len=args.max_seq_len, use_sep=args.use_sep, prompt_all=args.prompt_all, emb_exp_rate=args.emb_exp_rate,
        emb_win_min_size=args.emb_win_min_size, emb_win_max_size=args.emb_win_max_size,
        min_next_toks=args.min_next_toks, train_ds_type=args.train_ds_type,
        decoder_only=args.decoder_only,
        freeze_encoder=args.freeze_encoder,
        pretrained_encdec_model_path=pretrained_encdec_model_path,
        pretrained_mixed_decoder_model_path=pretrained_mixed_decoder_model_path,
        mask_cfg=mask_cfg, learning_rate=args.learning_rate,
        optimizer_name=args.optimizer_name, optimizer_params=args.optimizer_params,
        lrs_name=args.learning_rate_scheduler_name, lrs_params=args.learning_rate_scheduler_params,
        batch_size=args.docs_batch_size,
        weight_decay_decoder=args.weight_decay_decoder,
        weight_decay_other=args.weight_decay_other,
        llrd_decay=args.llrd_decay,
        attention_dropout=args.attention_dropout,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
    )
    if rank == 0:
        pprint(model_cfg.dict())

    prefix, suffix = gen_prefpostfix_mixed_decoder(model_cfg)
    train_path = find_create_train_path(args.train_root_path, prefix, suffix, args.train_subdir, create=(rank == 0))
    log(f'train_path: {train_path}')

    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    # ---- Resume / override decision matrix ------------------------------------
    # Three independent knobs control how a new run relates to an existing
    # train subdir:
    #   1. last.pth presence            -> resume model + (maybe) optimizer/scheduler
    #   2. optimizer_name / params diff -> drop saved optimizer & scheduler state
    #                                      on resume, build fresh from CLI args.
    #   3. --learning-rate-override > 0 -> after the above, REBUILD optimizer +
    #                                      scheduler from scratch with that LR;
    #                                      any restored state is discarded. Epoch
    #                                      counter and val_loss_min are kept.
    # Both (1) and (2) are decided here so the rest of the function can stay
    # simple; (3) is applied below, after the optimizer has been instantiated.
    checkpoint = None
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists(), (
            f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
            f'but {last_checkpoint_path} does not exist.'
        )
    if last_checkpoint_path.exists():
        log(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        log(f'Checkpoint with keys {list(checkpoint.keys())} loaded')
    else:
        if rank == 0:
            to_yaml_file(train_path / MIXED_DECODER_MODEL_CFG_FNAME, model_cfg)

    # optimizer_changed: if the saved checkpoint used a different optimizer class
    # or different optimizer kwargs than what the CLI requests now, drop the
    # saved optimizer + scheduler state on resume and build fresh ones. Use case:
    # switching `Adam` -> `AdamW` and/or changing the learning rate mid-training.
    # `learning_rate` itself is NOT part of this comparison: when the optimizer
    # is unchanged we restore its state (which carries the LR) and the scheduler
    # continues from where it left off. Use --learning-rate-override to force a
    # rebuild without changing the optimizer class.
    optimizer_changed = False
    if checkpoint is not None:
        prev_opt_name = checkpoint.get('optimizer_name')
        prev_opt_params = checkpoint.get('optimizer_params')
        cur_params = dict(args.optimizer_params or {})
        prv_params = dict(prev_opt_params or {})
        if prev_opt_name != args.optimizer_name or prv_params != cur_params:
            optimizer_changed = True
            log(f'Optimizer change detected on resume: '
                f'{prev_opt_name}({prv_params}) -> {args.optimizer_name}({cur_params}). '
                f'Building fresh optimizer; resetting scheduler.')
            if rank == 0:
                # Refresh the persisted YAML so it matches the optimizer
                # actually in use from this point on.
                to_yaml_file(train_path / MIXED_DECODER_MODEL_CFG_FNAME, model_cfg)
    tkz_enc = AutoTokenizer.from_pretrained(args.bert_model_name)
    tkz_dec = AutoTokenizer.from_pretrained(decoder_model_name)
    # We tokenize raw documents in full and chunk/truncate downstream, so HF's
    # "Token indices sequence length is longer than the specified maximum"
    # warning is just noise. Disable model_max_length cap on the tokenizer.
    tkz_enc.model_max_length = int(1e9)
    tkz_dec.model_max_length = int(1e9)
    # GPT-2 has no pad token; use eos as pad/delimiter/end-of-target.
    if tkz_dec.pad_token is None:
        tkz_dec.pad_token = tkz_dec.eos_token
    tkz = tkz_enc  # legacy local alias for downstream references

    log(model_cfg)
    model = MixedDecoder(model_cfg, tkz_enc, tkz_dec)

    parallel_mode = (args.parallel or _PARALLEL_DDP).lower()
    if parallel_mode not in (_PARALLEL_DDP, _PARALLEL_FSDP):
        raise ValueError(f'Unsupported --parallel value: {args.parallel!r}. Expected "ddp" or "fsdp".')
    use_fsdp = (parallel_mode == _PARALLEL_FSDP)
    if use_fsdp:
        if args.world_size <= 1:
            raise ValueError('--parallel fsdp requires --world-size > 1.')
        if device.type != 'cuda':
            raise ValueError('--parallel fsdp requires CUDA devices.')
        if decoder_dtype == DecoderDtype.Fp16:
            raise ValueError(
                'FSDP path does not support fp16 (would need ShardedGradScaler). '
                'Use a bf16 or fp32 decoder_spec, e.g. qwen2.5-1.5B-bf16.'
            )

    if checkpoint is not None and use_fsdp:
        # Reload onto CPU; FSDP places sharded params on the right device during wrap.
        log(f'FSDP: re-mapping checkpoint to CPU before load_pretrained')
        checkpoint_cpu = torch.load(last_checkpoint_path, map_location='cpu')
        # Replace the previously-loaded (device-mapped) tensors so load_pretrained
        # consumes the CPU copy. Keep optimizer/scheduler state for later.
        checkpoint = checkpoint_cpu

    model.load_pretrained(checkpoint)

    if use_fsdp:
        # Do NOT call model.to(device): FSDP will shard and place params via device_id.
        layer_classes = _resolve_transformer_layer_classes(model)
        if not layer_classes:
            raise RuntimeError(
                'FSDP: could not resolve any transformer layer classes from the model '
                '(_no_split_modules empty). Cannot build a sane auto_wrap_policy.'
            )
        log(f'FSDP: auto_wrap_policy will wrap layer classes: '
            f'{sorted(c.__name__ for c in layer_classes)}')

        sharding_strategy = {
            'full': ShardingStrategy.FULL_SHARD,
            'hybrid': ShardingStrategy.HYBRID_SHARD,
        }.get((args.fsdp_shard or 'full').lower())
        if sharding_strategy is None:
            raise ValueError(
                f'Unsupported --fsdp-shard value: {args.fsdp_shard!r}. Expected "full" or "hybrid".'
            )

        # bf16 mixed precision when caller requested it; otherwise keep fp32 (no MP policy).
        mp_policy = None
        if decoder_dtype == DecoderDtype.Bf16:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=layer_classes,
        )
        # Pass a DeviceMesh so FSDP emits DTensor state_dict instead of the
        # legacy/deprecated ShardedTensor. Without this, get_state_dict()
        # crashes inside _get_model_state_dict on tiny/uneven params (e.g.
        # LayerNorm bias of dim 768, or any param whose leading dim < world
        # size, or under HYBRID_SHARD) with:
        #   NotImplementedError: Only single local shard is supported.
        # See pytorch/pytorch#132366 and pytorch/pytorch#141799.
        if sharding_strategy == ShardingStrategy.HYBRID_SHARD:
            # 2D mesh: (replicate_across_nodes, shard_within_node).
            # mp.spawn here is single-node, so the replicate dim is 1; this
            # makes HYBRID effectively equivalent to FULL on a single node
            # but keeps the API consistent if the script is later launched
            # multi-node with the same world_size.
            device_mesh = init_device_mesh(
                'cuda', (1, args.world_size), mesh_dim_names=('replicate', 'shard'),
            )
        else:
            device_mesh = init_device_mesh('cuda', (args.world_size,))
        ddp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_policy,
            device_id=rank,
            use_orig_params=True,
            device_mesh=device_mesh,
            # Comm/compute overlap knobs: prefetch the next layer's all-gather
            # while the current layer computes (forward), and the previous
            # layer's all-gather while the current layer's backward runs.
            # limit_all_gathers caps in-flight all-gathers to avoid memory spikes
            # without serializing them.
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
        log(f'FSDP wrapped: strategy={sharding_strategy.name}, mp_policy={mp_policy}, '
            f'device_mesh={device_mesh}, forward_prefetch=True, '
            f'backward_prefetch=BACKWARD_PRE, limit_all_gathers=True')
    else:
        model.to(device)
        if args.world_size > 1:
            # Gradient checkpointing on the decoder uses use_reentrant=False
            # (see MixedDecoder.__init__), which is DDP-safe. All MixedDecoder
            # parameters participate in the loss in this codebase (decoder + encoder
            # + emb_exp; enc_proj/pos_emb are None when used), so we leave
            # find_unused_parameters=False. Setting it to True with non-reentrant
            # checkpointing adds a per-iteration Int bitmap allreduce that has been
            # observed to mismatch across ranks and tear down the gloo process group.
            ddp_model = DDP(model, device_ids=[rank])
        else:
            ddp_model = model

    params = build_param_groups(ddp_model, model_cfg.train_cfg, decoder_type, base_lr=args.learning_rate)
    # `weight_decay` is now sourced per-group when regularization knobs are enabled,
    # so any `weight_decay` passed via --optimizer-params would double-apply or
    # override the per-bucket settings. Strip + warn.
    opt_params = dict(args.optimizer_params)
    if 'weight_decay' in opt_params and len(params) > 1:
        log(f'WARNING: stripping weight_decay={opt_params["weight_decay"]} from --optimizer-params; '
            f'per-bucket weight_decay (decoder={model_cfg.train_cfg.weight_decay_decoder}, '
            f'other={model_cfg.train_cfg.weight_decay_other}) is used instead.')
        opt_params.pop('weight_decay')
    optimizer = instantiate_torch_optimizer(args.optimizer_name, params, lr=args.learning_rate, **opt_params)
    scheduler = instantiate_torch_lr_scheduler(args.learning_rate_scheduler_name, optimizer, **args.learning_rate_scheduler_params)
    if rank == 0 and len(params) > 1:
        log(f'Optimizer param groups ({len(optimizer.param_groups)}):')
        for g in optimizer.param_groups:
            log(f'  {g.get("name", "?")}: lr={g["lr"]:.3e}, wd={g.get("weight_decay", 0):.4g}, n_params={len(g["params"])}')

    # Mixed-precision setup. For DDP: weights stay fp32, autocast wraps forward,
    # GradScaler is used only with fp16. For FSDP: param/reduce/buffer dtypes are
    # already governed by the FSDP MixedPrecision policy, so autocast and
    # GradScaler are not needed (and disabled here).
    amp_enabled = (
        not use_fsdp
        and decoder_dtype in (DecoderDtype.Fp16, DecoderDtype.Bf16)
        and device.type == 'cuda'
    )
    amp_dtype = {DecoderDtype.Fp16: torch.float16, DecoderDtype.Bf16: torch.bfloat16}.get(decoder_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and decoder_dtype == DecoderDtype.Fp16))
    log(f'AMP: enabled={amp_enabled}, dtype={amp_dtype}, scaler_enabled={scaler.is_enabled()}, '
        f'parallel={parallel_mode}')

    last_epoch, val_loss_min, shuffle = -1, None, False

    # ---- Phase 2: restore optimizer / scheduler / epoch counter from checkpoint.
    # Model weights were already restored above via `model.load_pretrained(checkpoint)`.
    # Here we decide what to do with optimizer + scheduler state:
    #   * optimizer_changed -> keep model + epoch + val_loss_min only;
    #                          optimizer & scheduler stay freshly-built.
    #   * otherwise         -> restore optimizer + scheduler (FSDP path uses
    #                          set_optimizer_state_dict; DDP path uses the plain
    #                          optimizer.load_state_dict).
    if checkpoint is not None:
        if optimizer_changed:
            log('Skipping optimizer/scheduler state restore due to optimizer change.')
        elif use_fsdp:
            # FSDP1 marks the outermost wrapper as _is_root only during the
            # first forward (_lazy_init). set_optimizer_state_dict's verifier
            # walks the module tree, finds auto-wrapped FSDP submodules, and
            # then checks for an FSDP root -- which does not exist yet at this
            # point in training. Force lazy init here so the verifier sees a
            # proper root; otherwise it raises:
            #   "The model has FSDP modules but no FSDP root module exists."
            from torch.distributed.fsdp._runtime_utils import _lazy_init as _fsdp_lazy_init
            _fsdp_lazy_init(ddp_model, ddp_model)
            opt_sd = checkpoint['optimizer']
            # Backward-compat for checkpoints produced with incorrect FQN remap:
            # set_optimizer_state_dict expects every FQN referenced by
            # param_groups to exist in state. Fill any missing entries with
            # empty per-param state so resume can proceed (those params will
            # rebuild optimizer moments from scratch).
            if isinstance(opt_sd, dict):
                st = opt_sd.get('state')
                pgs = opt_sd.get('param_groups')
                if isinstance(st, dict) and isinstance(pgs, list):
                    # Build canonical FQN -> live parameter map (same canonicalization
                    # that DCP/FSDP loader uses internally).
                    from torch.distributed.checkpoint.state_dict import _get_fqns as _dcp_get_fqns
                    fqn_to_param: dict = {}
                    param_id_to_fqn: dict = {}
                    for raw_name, p in ddp_model.named_parameters():
                        _fqns = _dcp_get_fqns(ddp_model, raw_name)
                        if len(_fqns) == 1:
                            fqn = next(iter(_fqns))
                            fqn_to_param[fqn] = p
                            param_id_to_fqn[id(p)] = fqn
                    # Legacy checkpoints can still have optimizer.state keyed by
                    # integer ids. Remap to canonical FQNs using the exact pid
                    # assignment order used by optimizer.state_dict() (walk
                    # optimizer.param_groups in order).
                    if st and all(isinstance(k, int) for k in st.keys()):
                        pid_to_fqn: dict = {}
                        pid = 0
                        for lpg in optimizer.param_groups:
                            for lp in lpg.get('params', []):
                                fqn = param_id_to_fqn.get(id(lp))
                                if fqn is not None:
                                    pid_to_fqn[pid] = fqn
                                pid += 1
                        new_state: dict = {}
                        for pid, pstate in st.items():
                            fqn = pid_to_fqn.get(pid)
                            if fqn is not None:
                                new_state[fqn] = pstate
                        new_pgs: list = []
                        for pg in pgs:
                            pg2 = dict(pg)
                            pg2['params'] = [pid_to_fqn[p] for p in pg.get('params', []) if p in pid_to_fqn]
                            new_pgs.append(pg2)
                        st = new_state
                        pgs = new_pgs
                        opt_sd = {'state': st, 'param_groups': pgs}
                    for pg in pgs:
                        for fqn in pg.get('params', []):
                            if fqn not in st:
                                st[fqn] = {}
                    # Some older checkpoints have wrong-but-present optimizer
                    # entries (state from parameter A stored under parameter B
                    # FQN). Detect this by validating Adam moments against the
                    # live parameter shape; when mismatched, clear state so the
                    # optimizer re-initializes moments on the next step.
                    for fqn, pstate in st.items():
                        p = fqn_to_param.get(fqn)
                        if p is None or not isinstance(pstate, dict):
                            continue
                        bad = False
                        for sk in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                            sv = pstate.get(sk)
                            if torch.is_tensor(sv) and tuple(sv.shape) != tuple(p.shape):
                                bad = True
                                break
                        if bad:
                            st[fqn] = {}
            set_optimizer_state_dict(
                ddp_model,
                optimizer,
                optim_state_dict=opt_sd,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            # `set_optimizer_state_dict` routes through
            # `FSDP.optim_state_dict_to_load` -> `optim.load_state_dict`,
            # which is supposed to copy hyperparameters (lr, betas, ...) from
            # the saved param_groups into the live optimizer. In practice, on
            # the FSDP1 / DTensor path, the live optimizer's param_groups can
            # end up retaining the freshly-instantiated hyperparameters
            # (i.e. `args.learning_rate`) instead of the ones from the
            # checkpoint. Force-copy the saved hyperparams here so resume
            # behaviour matches the DDP branch below.
            saved_pgs = opt_sd.get('param_groups', [])
            for live_pg, saved_pg in zip(optimizer.param_groups, saved_pgs):
                for k, v in saved_pg.items():
                    if k == 'params':
                        continue
                    live_pg[k] = v
            if saved_pgs:
                log(f'FSDP resume: restored optimizer hyperparameters from '
                    f'checkpoint; lr={optimizer.param_groups[0].get("lr")!r}.')
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if not optimizer_changed:
            n_bad_states = sanitize_optimizer_state_for_model(optimizer, ddp_model)
            if n_bad_states:
                log(f'Resume: cleared {n_bad_states} invalid optimizer state entries '
                    f'(shape mismatch vs current parameters).')
        if not optimizer_changed:
            scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        del checkpoint

    # ---- Phase 3: --learning-rate-override (applied AFTER any restore).
    # Rebuild optimizer + scheduler from scratch with the given LR; any restored
    # state is discarded. last_epoch and val_loss_min are kept so the run
    # continues counting epochs and tracking best-val from where it left off.
    if args.learning_rate_override > 0:
        log(f'learning_rate_override={args.learning_rate_override} > 0: rebuilding optimizer '
            f'({args.optimizer_name}) and scheduler ({args.learning_rate_scheduler_name}) from '
            f'scratch; any restored optimizer/scheduler state is discarded.')
        params = build_param_groups(
            ddp_model, model_cfg.train_cfg, decoder_type, base_lr=args.learning_rate_override,
        )
        opt_params = dict(args.optimizer_params)
        if 'weight_decay' in opt_params and len(params) > 1:
            opt_params.pop('weight_decay')
        optimizer = instantiate_torch_optimizer(
            args.optimizer_name, params,
            lr=args.learning_rate_override, **opt_params,
        )
        scheduler = instantiate_torch_lr_scheduler(
            args.learning_rate_scheduler_name, optimizer, **args.learning_rate_scheduler_params,
        )

    n_special_toks = 1000
    if args.train_ds_type == MixedDecoderDsType.Cite:
        ds_train = MaskedCiteDataset(ds_train, tkz_enc, max_seq_len=args.inp_len, n_special_toks=n_special_toks, mask_cfg=mask_cfg, prompt_all=args.prompt_all, device=device, tkz_dec=tkz_dec)
        ds_val = MaskedCiteDataset(ds_val, tkz_enc, max_seq_len=args.inp_len, n_special_toks=n_special_toks, mask_cfg=mask_cfg, prompt_all=args.prompt_all, device=device, tkz_dec=tkz_dec)
        ds_train.shuffle(seed=(args.random_seed or 0) + rank)
        ds_val.shuffle(seed=(args.random_seed or 0) + rank)
        train_batch_it = create_masked_cite_dataloader(ds_train, batch_size=args.docs_batch_size)
        val_batch_it = create_masked_cite_dataloader(ds_val, batch_size=args.docs_batch_size)
    elif args.train_ds_type == MixedDecoderDsType.QnaSquadV2:
        max_chunks = max(args.emb_win_max_size, 1)
        ds_train = QnaCiteDataset(
            df_sq, sq_inds_train, tkz_enc, inp_len=args.inp_len, max_chunks=max_chunks, device=device, tkz_dec=tkz_dec,
        )
        ds_val = QnaCiteDataset(
            df_sq, sq_inds_val, tkz_enc, inp_len=args.inp_len, max_chunks=max_chunks, device=device, tkz_dec=tkz_dec,
        )
        ds_train.shuffle(seed=(args.random_seed or 0) + rank)
        ds_val.shuffle(seed=(args.random_seed or 0) + rank)
        train_batch_it = create_qna_cite_dataloader(ds_train, batch_size=args.docs_batch_size)
        val_batch_it = create_qna_cite_dataloader(ds_val, batch_size=args.docs_batch_size)
    elif args.train_ds_type == MixedDecoderDsType.QnaAll:
        qna_agg_train.device = device
        qna_agg_val.device = device
        max_chunks = max(args.emb_win_max_size, 1)
        qna_agg_train.shuffle(seed=(args.random_seed or 0) + (rank + 10)**2)
        qna_agg_val.shuffle(seed=(args.random_seed or 0) + (rank + 10)**2)
        print(f'qna_agg_train size: {len(qna_agg_train)}. qna_agg_val size: {len(qna_agg_val)}.')
        train_batch_it = create_qna_dataloader(qna_agg_train, batch_size=args.docs_batch_size)
        val_batch_it = create_qna_dataloader(qna_agg_val, batch_size=args.docs_batch_size)
    elif args.train_ds_type == MixedDecoderDsType.QnaAns:
        qna_agg_train.device = device
        qna_agg_val.device = device
        max_chunks = max(args.emb_win_max_size, 1)
        qna_agg_train.shuffle(seed=(args.random_seed or 0) + (rank + 10)**2)
        qna_agg_val.shuffle(seed=(args.random_seed or 0) + (rank + 10)**2)
        print(f'qna_agg_train size: {len(qna_agg_train)}. qna_agg_val size: {len(qna_agg_val)}.')
        train_batch_it = create_qna_dataloader(qna_agg_train, batch_size=args.docs_batch_size)
        val_batch_it = create_qna_dataloader(qna_agg_val, batch_size=args.docs_batch_size)
    elif args.train_ds_type == MixedDecoderDsType.Next:
        ds_train = NextTokWikiDataset(
            wiki_ds, wiki_inds_train, tkz_enc, inp_len=args.inp_len, min_next_toks=args.min_next_toks,
            emb_win_min_size=max(args.emb_win_min_size, 1), emb_win_max_size=max(args.emb_win_max_size, 1),
            device=device, tkz_dec=tkz_dec,
        )
        ds_val = NextTokWikiDataset(
            wiki_ds, wiki_inds_val, tkz_enc, inp_len=args.inp_len, min_next_toks=args.min_next_toks,
            emb_win_min_size=max(args.emb_win_min_size, 1), emb_win_max_size=max(args.emb_win_max_size, 1),
            device=device, tkz_dec=tkz_dec,
        )
        ds_train.shuffle(seed=(args.random_seed or 0) + rank)
        ds_val.shuffle(seed=(args.random_seed or 0) + rank)
        train_batch_it = create_next_tok_dataloader(ds_train, batch_size=args.docs_batch_size)
        val_batch_it = create_next_tok_dataloader(ds_val, batch_size=args.docs_batch_size)
    else:
        raise ValueError(f'Unsupported train_ds_type: {args.train_ds_type}')

    lr = optimizer.param_groups[0]['lr']
    log(f'Scheduler {scheduler.__class__.__name__} lr: {lr:0.10f}.')
    if rank == 0:
        tbsw = tb.SummaryWriter(log_dir=str(train_path))
        log(ddp_model)

        grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
        grad_log_interval = max(grad_log_interval, 1)
        prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
        if prev_train_steps > 0:
            grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1

    if args.world_size > 1:
        dist.barrier()
    for epoch in range(last_epoch + 1, args.epochs):
        ddp_model.train()
        train_losses = LossesStats()
        train_loss = 0.0
        if rank == 0:
            pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        else:
            pbar = range(args.train_epoch_steps)
        for i in pbar:
            batch = next(train_batch_it)

            optimizer.zero_grad()
            if amp_enabled:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    loss_dict, _ = ddp_model(batch)
                loss = loss_dict['loss']
                scaler.scale(loss).backward()
            else:
                loss_dict, _ = ddp_model(batch)
                loss = loss_dict['loss']
                loss.backward()

            if rank == 0:
                if grad_log_ind % grad_log_interval == 0:
                    log_weights_grads_stats(grad_log_step, ddp_model, tbsw)
                    grad_log_step += 1
                grad_log_ind += 1

            # Gradient clipping (AMP- and FSDP-aware). Disabled when max_grad_norm <= 0.
            max_grad_norm = model_cfg.train_cfg.max_grad_norm
            if max_grad_norm > 0:
                if amp_enabled and scaler.is_enabled():
                    # Must unscale before measuring/clipping the unscaled grads.
                    scaler.unscale_(optimizer)
                if use_fsdp:
                    # FSDP's own method correctly all-reduces the norm across shards.
                    ddp_model.clip_grad_norm_(max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)

            if amp_enabled and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            train_loss += loss.item()
            train_losses.update_dict(loss_dict)

            if rank == 0:
                losses_str = train_losses.to_cli_str(aggregate=False)
                pbar.set_postfix_str(f'Train. {losses_str}')

        train_loss /= args.train_epoch_steps
        if rank == 0:
            pbar.close()
            train_losses.log_to_tb('Train', epoch, tbsw)

        ddp_model.eval()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_losses = LossesStats()
        val_loss = 0.0
        if rank == 0:
            pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        else:
            pbar = range(args.val_epoch_steps)
        for i in pbar:
            batch = next(val_batch_it)

            with torch.no_grad():
                if amp_enabled:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        loss_dict, _ = ddp_model(batch)
                else:
                    loss_dict, _ = ddp_model(batch)
            loss = loss_dict['loss']

            val_loss += loss.item()
            val_losses.update_dict(loss_dict)

            if rank == 0:
                losses_str = val_losses.to_cli_str(aggregate=False)
                pbar.set_postfix_str(f'Val. {losses_str}')

        val_loss /= args.val_epoch_steps
        if rank == 0:
            pbar.close()
            val_losses.log_to_tb('Val', epoch, tbsw)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if rank == 0:
            last_lr = scheduler.get_last_lr()[0]
            tbsw.add_scalar(f'{scheduler.__class__.__name__} lr', last_lr, epoch)

            print(f'Train mean loss: {train_loss:.6f}. Val mean loss: {val_loss:.6f}')
            train_losses_str = train_losses.to_cli_str(aggregate=True)
            val_losses_str = val_losses.to_cli_str(aggregate=True)
            print(f'Train mean losses: {train_losses_str}')
            print(f'Val mean losses: {val_losses_str}')
            print(f'Current lr: {last_lr:.10f}.')

            best = False
            if val_loss_min is None or val_loss < val_loss_min:
                val_loss_str = f'{val_loss_min}' if val_loss_min is None else f'{val_loss_min:.6f}'
                print(f'Val min loss change: {val_loss_str} --> {val_loss:.6f}')
                val_loss_min = val_loss
                best = True

        if use_fsdp:
            # Save via the standard FSDP full-state-dict + rank-0 torch.save
            # pattern (this is what HF Trainer and Qwen's finetune.py use).
            #
            # All ranks enter the FSDP.state_dict_type context, then call
            # ddp_model.state_dict() / FSDP.optim_state_dict() — these collectives
            # gather all shards onto rank 0 (offload_to_cpu=True so the gather
            # happens in CPU memory, ~3 GB for Qwen2.5-1.5B+BERT in bf16,
            # avoiding GPU OOM). Rank 0 then writes a single .pth file in
            # the same format as the non-FSDP path; resume goes through the
            # existing `last_checkpoint_path` + `FSDP.optim_state_dict_to_load`
            # branch above.
            #
            # Why not torch.distributed.checkpoint sharded save:
            # On torch 2.3 + FSDP1 the DCP planner serialises a lot of
            # Python work on the coordinator rank for billion-parameter
            # models with many small tensors (LayerNorm/bias), which looks
            # exactly like a hang (one GPU at 100%, others idle for many
            # minutes). Full-state-dict + single-file save sidesteps the
            # planner entirely and finishes in seconds.
            import sys, time
            def _rlog(msg: str):
                print(f'[rank{rank}] {msg}', flush=True)
                sys.stdout.flush()

            dist.barrier()
            t0 = time.time()
            _rlog('ckpt: gathering model state_dict via DTensor.full_tensor()')
            # Per-parameter gather pattern from the PyTorch FSDP tutorial:
            #   https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
            # FSDP was wrapped with use_orig_params=True and a DeviceMesh, so
            # `ddp_model.state_dict()` returns DTensor shards. We iterate the
            # sharded dict and call `.full_tensor()` on each param to all-gather
            # its full tensor across ranks; only rank 0 keeps a CPU copy so peak
            # CPU memory is bounded to a single full param at a time on the
            # other ranks (vs. holding the entire model on every rank with the
            # legacy FULL_STATE_DICT + offload_to_cpu path).
            sharded_msd = ddp_model.state_dict()
            full_msd: dict = {}
            for param_name, sharded_param in sharded_msd.items():
                if hasattr(sharded_param, 'full_tensor'):
                    full_param = sharded_param.full_tensor()
                else:
                    # Non-DTensor entries (e.g. plain buffers already replicated).
                    full_param = sharded_param
                if rank == 0:
                    full_msd[param_name] = full_param.detach().cpu()
                else:
                    del full_param
            del sharded_msd
            _rlog(f'ckpt: model state_dict gather done in {time.time()-t0:.1f}s')

            # Optimizer state: do NOT use FSDP.optim_state_dict(rank0_only=True)
            # nor torch.distributed.checkpoint.state_dict.get_optimizer_state_dict.
            # On torch 2.3 + FSDP1 + use_orig_params=True, both paths route
            # through `gather_object` of pickled per-parameter state on the
            # coordinator rank, which presents as a multi-minute hang with
            # one CPU at 100% and the other ranks blocked in NCCL.
            #
            # Mirror the model gather: walk optimizer.state_dict() and call
            # `.full_tensor()` on each DTensor entry. Non-tensor / non-DTensor
            # entries (e.g. Adam's `step` scalar) are passed through. Param
            # groups are rewritten with FQNs (not int pids) so the resulting
            # dict matches what `set_optimizer_state_dict(full_state_dict=True, ...)`
            # expects on resume; it indexes `state` by FQN. Without this
            # rewrite, loading crashes with
            #   KeyError: 'enc.bert_model.embeddings.word_embeddings.weight'
            # inside `_split_optim_state_dict`.
            t0o = time.time()
            _rlog('ckpt: gathering optim state_dict via per-tensor full_tensor()')
            local_osd = optimizer.state_dict()
            # pid -> canonical FQN. `named_parameters()` on an FSDP1 root
            # with auto_wrap returns names containing `_fsdp_wrapped_module.`;
            # DCP's `_get_fqns` strips those (and DDP/compiler) prefixes the
            # same way the loader does internally.
            from torch.distributed.checkpoint.state_dict import _get_fqns as _dcp_get_fqns
            param_id_to_fqn: dict = {}
            for raw_name, p in ddp_model.named_parameters():
                _fqns = _dcp_get_fqns(ddp_model, raw_name)
                assert len(_fqns) == 1, f'Expected 1 FQN for {raw_name!r}, got {_fqns!r}.'
                param_id_to_fqn[id(p)] = next(iter(_fqns))
            # Optimizer.state_dict() uses integer ids assigned by iterating
            # optimizer.param_groups in order. Rebuild that exact id -> FQN map
            # from optimizer.param_groups (not from named_parameters order),
            # otherwise multi-group optimizers produce mismatched checkpoints.
            pid_to_fqn: dict = {}
            pid = 0
            for pg in optimizer.param_groups:
                for p in pg.get('params', []):
                    fqn = param_id_to_fqn.get(id(p))
                    if fqn is not None:
                        pid_to_fqn[pid] = fqn
                    pid += 1
            full_state: dict = {}
            for pid, pstate in local_osd['state'].items():
                gathered: dict = {}
                for k, v in pstate.items():
                    if hasattr(v, 'full_tensor'):
                        full_v = v.full_tensor()
                        if rank == 0:
                            gathered[k] = full_v.detach().cpu()
                        else:
                            del full_v
                    elif torch.is_tensor(v):
                        if rank == 0:
                            gathered[k] = v.detach().cpu()
                    else:
                        if rank == 0:
                            gathered[k] = v
                if rank == 0:
                    fqn = pid_to_fqn.get(pid, pid)
                    full_state[fqn] = gathered
            if rank == 0:
                remapped_pgs = []
                for pg in local_osd['param_groups']:
                    pg2 = dict(pg)
                    pg2['params'] = [pid_to_fqn.get(p, p) for p in pg.get('params', [])]
                    remapped_pgs.append(pg2)
                full_osd = {'state': full_state, 'param_groups': remapped_pgs}
            else:
                full_osd = {}
            del local_osd
            _rlog(f'ckpt: optim state_dict gather done in {time.time()-t0o:.1f}s')

            if rank == 0:
                t1 = time.time()
                checkpoint = {
                    'model': full_msd,
                    'optimizer': full_osd,
                    'scheduler': scheduler.state_dict(),
                    'last_epoch': epoch,
                    'val_loss_min': val_loss_min,
                    'optimizer_name': args.optimizer_name,
                    'optimizer_params': dict(args.optimizer_params or {}),
                }
                print(f'Saving checkpoint to {last_checkpoint_path}', flush=True)
                torch.save(checkpoint, last_checkpoint_path)
                print(f'ckpt: torch.save done in {time.time()-t1:.1f}s', flush=True)
                if best:
                    print(f'New val loss minimum: {val_loss_min:.6f}. Saving '
                          f'checkpoint to {best_checkpoint_path}', flush=True)
                    shutil.copyfile(last_checkpoint_path, best_checkpoint_path)
            # Free the gathered full state on rank 0 before the next epoch.
            del full_msd, full_osd
            # Keep ranks in lockstep so the next epoch doesn't race ahead of
            # rank 0's write.
            dist.barrier()
        else:
            if rank == 0:
                checkpoint = {
                    'model': ddp_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'last_epoch': epoch,
                    'val_loss_min': val_loss_min,
                    'optimizer_name': args.optimizer_name,
                    'optimizer_params': dict(args.optimizer_params or {}),
                }
                print(f'Saving checkpoint to {last_checkpoint_path}')
                torch.save(checkpoint, last_checkpoint_path)

                if best:
                    print(f'New val loss minimum: {val_loss_min:.6f}. Saving checkpoint to {best_checkpoint_path}')
                    shutil.copyfile(last_checkpoint_path, best_checkpoint_path)

    cleanup()


def main(args: ArgsMixedDecoderTrain) -> int:
    print(args)

    mask_cfg = None
    if args.mask_tokens:
        mask_cfg = MaskCfg(
            sep_freq=args.mask_sep_freq, sep_frac=args.mask_sep_frac, seq_freq=args.mask_seq_freq, seq_max_frac=args.mask_seq_max_frac,
            seq_max_len=args.mask_seq_max_len, n_last_toks=args.mask_n_last_toks,
        )

    tkz_enc = AutoTokenizer.from_pretrained(args.bert_model_name)
    if args.decoder_spec:
        _, decoder_model_name_main, _ = parse_decoder_spec(args.decoder_spec)
    else:
        decoder_model_name_main = args.decoder_model_name
    tkz_dec = AutoTokenizer.from_pretrained(decoder_model_name_main)
    # See note in train(): suppress noisy 'longer than max length' tokenizer warning.
    tkz_enc.model_max_length = int(1e9)
    tkz_dec.model_max_length = int(1e9)
    if tkz_dec.pad_token is None:
        tkz_dec.pad_token = tkz_dec.eos_token

    wiki_ds, wiki_inds_train, wiki_inds_val = None, None, None
    qna_agg_train, qna_agg_val = None, None
    if args.train_ds_type == MixedDecoderDsType.Cite:
        ds_train, ds_val = load_split_wiki_dataset(
            data_path=args.data_path, tkz=tkz_enc, max_seq_len=args.inp_len, val_split_ratio=0.05,
            mask_cfg=mask_cfg, random_seed=args.random_seed,
        )
        df_sq, sq_inds_train, sq_inds_val = None, None, None
    elif args.train_ds_type == MixedDecoderDsType.QnaSquadV2:
        df_sq, sq_inds_train, sq_inds_val = load_split_squadv2(exclude_empty_answers=True)
        ds_train, ds_val = None, None
    elif args.train_ds_type == MixedDecoderDsType.QnaAll:
        ds_train, ds_val = None, None
        df_sq, sq_inds_train, sq_inds_val = None, None, None
        max_chunks = max(args.emb_win_max_size, 1)
        qna_agg_train, qna_agg_val = load_qna_datasets(
            tkz_enc=tkz_enc, tkz_dec=tkz_dec, inp_len=args.inp_len, max_chunks=max_chunks,
            cache_dir=str(args.data_path),
        )
    elif args.train_ds_type == MixedDecoderDsType.QnaAns:
        ds_train, ds_val = None, None
        df_sq, sq_inds_train, sq_inds_val = None, None, None
        max_chunks = max(args.emb_win_max_size, 1)
        qna_agg_train, qna_agg_val = load_qna_datasets(
            tkz_enc=tkz_enc, tkz_dec=tkz_dec, inp_len=args.inp_len, max_chunks=max_chunks,
            cache_dir=str(args.data_path), exclude_noanswer=True,
        )
    elif args.train_ds_type == MixedDecoderDsType.Next:
        wiki_ds, wiki_inds_train, wiki_inds_val = load_split_wiki_for_next(
            data_path=args.data_path, random_seed=args.random_seed,
        )
        ds_train, ds_val = None, None
        df_sq, sq_inds_train, sq_inds_val = None, None, None
    else:
        raise ValueError(f'Unsupported train_ds_type: {args.train_ds_type}')

    mp.spawn(train, args=(
        ds_train, ds_val, df_sq, sq_inds_train, sq_inds_val, wiki_ds, wiki_inds_train, wiki_inds_val, qna_agg_train, qna_agg_val, args,
    ), nprocs=args.world_size, join=True)

    return 0


if __name__ == '__main__':
    run_and_exit(
        ArgsMixedDecoderTrain, main, 'Train MixedDecoder model (Encoder-Decoder with causal attention) multi GPU training.', exception_handler=rethrow,
    )

