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
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)
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
        ddp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_policy,
            device_id=rank,
            use_orig_params=True,
        )
        log(f'FSDP wrapped: strategy={sharding_strategy.name}, mp_policy={mp_policy}')
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
        # Load sharded model + optimizer state via the modern DCP state_dict
        # helpers (torch >= 2.2). These use DTensor under FSDP with
        # use_orig_params=True and avoid the deprecated ShardedTensor path
        # whose rank-0 serialization was the source of the previous hang.
        # All ranks read their own shard files in parallel.
        log(f'FSDP: loading sharded checkpoint from {fsdp_last_dir}')
        msd, osd = get_state_dict(ddp_model, optimizer)
        state = {'model': msd, 'optim': osd}
        dcp.load(
            state_dict=state,
            storage_reader=dcp.FileSystemReader(str(fsdp_last_dir)),
        )
        set_state_dict(
            ddp_model, optimizer,
            model_state_dict=state['model'],
            optim_state_dict=state['optim'],
        )
        meta = torch.load(fsdp_last_meta_path, map_location='cpu')
        scheduler.load_state_dict(meta['scheduler'])
        last_epoch = meta['last_epoch']
        val_loss_min = meta['val_loss_min']
        log(f'FSDP: sharded checkpoint loaded; resuming from epoch {last_epoch + 1}, '
            f'val_loss_min={val_loss_min}')
    elif checkpoint is not None:
        if use_fsdp:
            # Legacy path: resuming an FSDP run from a single-file full-state
            # checkpoint produced before sharded saves were introduced.
            with FSDP.state_dict_type(
                ddp_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=False),
                FullOptimStateDictConfig(rank0_only=False),
            ):
                sharded_optim_state = FSDP.optim_state_dict_to_load(
                    ddp_model, optimizer, checkpoint['optimizer'],
                )
            optimizer.load_state_dict(sharded_optim_state)
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
            # Sharded save via the modern DCP state_dict API. get_state_dict()
            # returns DTensor-based shards (with use_orig_params=True under
            # FSDP) and runs the heavy work in parallel across ranks. The
            # previous SHARDED_STATE_DICT + FSDP.optim_state_dict() path went
            # through the deprecated ShardedTensor flow, which serialized the
            # optimizer-state conversion on rank 0 (GPU 0 at 100%, GPUs 1..N
            # at 0% for hours).
            msd, osd = get_state_dict(ddp_model, optimizer)
            state = {'model': msd, 'optim': osd}
            if rank == 0:
                print(f'Saving FSDP sharded checkpoint to {fsdp_last_dir}')
            dcp.save(
                state_dict=state,
                storage_writer=dcp.FileSystemWriter(str(fsdp_last_dir)),
            )
            if rank == 0:
                torch.save(
                    {
                        'scheduler': scheduler.state_dict(),
                        'last_epoch': epoch,
                        'val_loss_min': val_loss_min,
                    },
                    fsdp_last_meta_path,
                )
                if best:
                    print(f'New val loss minimum: {val_loss_min:.6f}. Copying sharded '
                          f'checkpoint to {fsdp_best_dir}')
                    if fsdp_best_dir.exists():
                        shutil.rmtree(fsdp_best_dir)
                    shutil.copytree(fsdp_last_dir, fsdp_best_dir)
                    shutil.copyfile(fsdp_last_meta_path, fsdp_best_meta_path)
            # Keep ranks in lockstep so the next epoch doesn't race ahead of
            # rank 0's metadata write / best-copy.
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

