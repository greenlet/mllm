import json
import os
from datetime import timedelta
from pathlib import Path
from pprint import pprint
from typing import Any, Optional
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
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from transformers import AutoTokenizer

from mllm.config.model import MixedDecoderCfg, MixedDecoderDsType, MixedDecoderType, BertEmbType, \
    DecoderDtype, copy_override_mixed_decoder_cfg, gen_prefpostfix_mixed_decoder, parse_decoder_spec
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

    Walks ``module`` and aggregates the names listed in ``_no_split_modules`` on any
    ``PreTrainedModel`` ancestor (e.g. ``Qwen2DecoderLayer``, ``BertLayer``), then maps
    those names back to live class objects found in the tree. Robust to import paths.
    """
    wanted_names: set = set()
    for sub in module.modules():
        names = getattr(sub, '_no_split_modules', None)
        if names:
            wanted_names.update(names)
    found: set = set()
    for sub in module.modules():
        if type(sub).__name__ in wanted_names:
            found.add(type(sub))
    return found


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
    )
    if rank == 0:
        pprint(model_cfg.dict())

    prefix, suffix = gen_prefpostfix_mixed_decoder(model_cfg)
    train_path = find_create_train_path(args.train_root_path, prefix, suffix, args.train_subdir, create=(rank == 0))
    log(f'train_path: {train_path}')

    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    # FSDP sharded checkpoint layout (one directory per snapshot containing per-rank
    # shard files written by torch.distributed.checkpoint, plus a small meta pickle
    # written by rank 0 with scalar bookkeeping).
    fsdp_last_dir = train_path / 'last_fsdp'
    fsdp_best_dir = train_path / 'best_fsdp'
    fsdp_last_meta_path = train_path / 'last_fsdp_meta.pth'
    fsdp_best_meta_path = train_path / 'best_fsdp_meta.pth'
    checkpoint = None
    # True iff we have a resumable sharded FSDP checkpoint on disk. When set, we
    # skip the legacy DDP-style full-file load (which is what was hanging on
    # rank 0) and instead load model+optimizer shards via DCP *after* FSDP has
    # wrapped the model.
    resume_from_fsdp_shards = fsdp_last_dir.exists() and fsdp_last_meta_path.exists()
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists() or resume_from_fsdp_shards, \
            (f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
             f'but neither {last_checkpoint_path} nor {fsdp_last_dir} exists.')

    if resume_from_fsdp_shards:
        log(f'Found FSDP sharded checkpoint at {fsdp_last_dir}; will load after FSDP wrap.')
    elif last_checkpoint_path.exists():
        log(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        log(f'Checkpoint with keys {list(checkpoint.keys())} loaded')
    else:
        if rank == 0:
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
        )
        log(f'FSDP wrapped: strategy={sharding_strategy.name}, mp_policy={mp_policy}, '
            f'device_mesh={device_mesh}')
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

    params = ddp_model.parameters()
    optimizer = instantiate_torch_optimizer(args.optimizer_name, params, lr=args.learning_rate, **args.optimizer_params)
    scheduler = instantiate_torch_lr_scheduler(args.learning_rate_scheduler_name, optimizer, **args.learning_rate_scheduler_params)

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

    if resume_from_fsdp_shards:
        # Legacy: a previous run saved sharded DCP shards into `last_fsdp/`.
        # We no longer write that format (see save block below); load the
        # shards with the AppState/Stateful pattern just to migrate forward.
        # New runs will overwrite `last.pth` on the next checkpoint.
        _dcp = dcp
        _sd_opts = StateDictOptions(cpu_offload=True)
        class _AppState(Stateful):
            def __init__(self, m, o):
                self._m, self._o = m, o
            def state_dict(self):
                msd, osd = get_state_dict(self._m, self._o, options=_sd_opts)
                return {'model': msd, 'optim': osd}
            def load_state_dict(self, sd):
                set_state_dict(
                    self._m, self._o,
                    model_state_dict=sd['model'],
                    optim_state_dict=sd['optim'],
                    options=_sd_opts,
                )
        log(f'FSDP: loading legacy sharded checkpoint from {fsdp_last_dir}')
        app = _AppState(ddp_model, optimizer)
        _dcp.load(
            state_dict={'app': app},
            storage_reader=_dcp.FileSystemReader(str(fsdp_last_dir)),
        )
        meta = torch.load(fsdp_last_meta_path, map_location='cpu')
        scheduler.load_state_dict(meta['scheduler'])
        last_epoch = meta['last_epoch']
        val_loss_min = meta['val_loss_min']
        log(f'FSDP: legacy sharded checkpoint loaded; resuming from epoch '
            f'{last_epoch + 1}, val_loss_min={val_loss_min}')
    elif checkpoint is not None:
        if use_fsdp:
            # FSDP1 marks the outermost wrapper as _is_root only during the
            # first forward (_lazy_init). set_optimizer_state_dict's verifier
            # walks the module tree, finds auto-wrapped FSDP submodules, and
            # then checks for an FSDP root -- which does not exist yet at this
            # point in training. Force lazy init here so the verifier sees a
            # proper root; otherwise it raises:
            #   "The model has FSDP modules but no FSDP root module exists."
            # Note: FSDP._lazy_init was moved to torch.distributed.fsdp.
            # _runtime_utils._lazy_init(state, root_module) in PyTorch 2.x.
            from torch.distributed.fsdp._runtime_utils import _lazy_init as _fsdp_lazy_init
            _fsdp_lazy_init(ddp_model, ddp_model)
            # Normalize legacy optimizer state: older runs saved
            # `optimizer.state_dict()` verbatim, which keys `state` by integer
            # param ids and lists those same ids under each `param_groups`
            # entry. `set_optimizer_state_dict(full_state_dict=True, ...)`
            # expects `state` keyed by FQN and `param_groups[i]['params']` to
            # be FQNs as well. Convert in-place; new-format checkpoints
            # (already FQN-keyed) pass through unchanged.
            opt_sd = checkpoint['optimizer']
            opt_state = opt_sd.get('state', {})
            if opt_state and all(isinstance(k, int) for k in opt_state.keys()):
                # Build pid -> canonical FQN map. `named_parameters()` on an
                # FSDP1 root with auto_wrap inserts `_fsdp_wrapped_module.`
                # segments into the names, which do NOT match the FQNs that
                # `set_optimizer_state_dict` looks up. DCP's `_get_fqns`
                # strips FSDP/DDP/compiler prefixes the same way the loader
                # does internally, so use it directly.
                from torch.distributed.checkpoint.state_dict import _get_fqns
                pid_to_fqn: dict = {}
                for i, (raw_name, _) in enumerate(ddp_model.named_parameters()):
                    fqns_set = _get_fqns(ddp_model, raw_name)
                    # With use_orig_params=True, each parameter maps to exactly one FQN.
                    assert len(fqns_set) == 1, (
                        f'Expected 1 FQN for {raw_name!r}, got {fqns_set!r}.'
                    )
                    pid_to_fqn[i] = next(iter(fqns_set))
                new_state = {pid_to_fqn[pid]: st for pid, st in opt_state.items() if pid in pid_to_fqn}
                new_param_groups = []
                for pg in opt_sd.get('param_groups', []):
                    pg2 = dict(pg)
                    pg2['params'] = [pid_to_fqn[p] for p in pg.get('params', []) if p in pid_to_fqn]
                    new_param_groups.append(pg2)
                opt_sd = {'state': new_state, 'param_groups': new_param_groups}
            # Tolerate incomplete optimizer state: with FSDP1 + use_orig_params=True
            # a parameter whose local shard is empty on the saving rank never
            # had Adam state allocated, so its FQN can be missing from
            # `state`. `_split_optim_state_dict` expects every parameter in
            # `param_groups[*]['params']` to have an entry. We CANNOT just
            # insert an empty dict for those entries either: with
            # `full_state_dict=True` the loader allocates a 0-sized exp_avg
            # tensor from `{}` and then `optimizer.step()` crashes with
            #   RuntimeError: The size of tensor a (0) must match the size of
            #   tensor b (N) at non-singleton dimension 0
            # inside `torch._foreach_lerp_`.
            #
            # If we detect this corruption we skip the optimizer-state load
            # entirely. Model weights, scheduler, last_epoch and val_loss_min
            # are still restored from the checkpoint; only Adam's running
            # moments are reset. The next save will write a complete
            # state-dict (see save block: union over ranks), so this branch
            # is only taken once when migrating from a legacy broken
            # checkpoint.
            opt_state = opt_sd.get('state', {})
            opt_pgs = opt_sd.get('param_groups', [])
            referenced_fqns: set = set()
            for pg in opt_pgs:
                referenced_fqns.update(pg.get('params', []))
            missing = [fqn for fqn in referenced_fqns if fqn not in opt_state]
            if missing:
                log(f'FSDP resume: optimizer state is incomplete '
                    f'({len(missing)}/{len(referenced_fqns)} entries missing, '
                    f'e.g. {missing[:3]}). Skipping optimizer-state load; '
                    f'Adam moments will reinitialize. Model weights, scheduler '
                    f'and epoch counter are still restored.')
            else:
                set_optimizer_state_dict(
                    ddp_model,
                    optimizer,
                    optim_state_dict=opt_sd,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        del checkpoint

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
            pid_to_fqn: dict = {}
            for i, (raw_name, _) in enumerate(ddp_model.named_parameters()):
                _fqns = _dcp_get_fqns(ddp_model, raw_name)
                assert len(_fqns) == 1, f'Expected 1 FQN for {raw_name!r}, got {_fqns!r}.'
                pid_to_fqn[i] = next(iter(_fqns))
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

    