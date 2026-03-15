import json
import os
from pathlib import Path
from pprint import pprint
from typing import Any, Optional
import shutil

from datasets import Dataset
from pydantic import BaseModel, Field, validator
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
import torch
import torch.utils.tensorboard as tb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from transformers import AutoTokenizer

from mllm.config.model import MixedDecoderCfg, MixedDecoderDsType, MixedDecoderType, BertEmbType, \
    copy_override_mixed_decoder_cfg, gen_prefpostfix_mixed_decoder
from mllm.exp.args import MIXED_DECODER_MODEL_CFG_FNAME, create_bool_str_field, get_pretrained_model_path, is_arg_true, \
    mask_tokens_ARG
from mllm.model.mixed_decoder import MixedDecoder
from mllm.model.losses import LossesStats
from mllm.train.encdec_graph_bert import MaskedCiteDataset, create_masked_cite_dataloader, load_split_wiki_dataset
from mllm.train.mask_utils import MaskCfg
from mllm.train.next_tok_wiki import NextTokWikiDataset, create_next_tok_dataloader, load_split_wiki_for_next
from mllm.train.qna_cite import QnaCiteDataset, create_qna_cite_dataloader, load_split_squadv2
from mllm.train.utils import find_create_train_path, log_weights_grads_stats
from mllm.utils.utils import instantiate_torch_lr_scheduler, instantiate_torch_optimizer, parse_dict_str, rethrow


freeze_encoder_ARG = '--freeze-encoder', 'Freeze encoder weights during training'
use_sep_ARG = '--use-sep', 'Insert SEP/EOS embedding between context embeddings and prompt tokens'
prompt_all_ARG = '--prompt-all', 'Target is whole input chunk (true) or citation only (false)'


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


def setup(rank, world_size):
    if world_size <= 1:
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = 'nccl'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    if not dist.is_initialized():
        return
    dist.destroy_process_group()


def train(rank: int, ds_train, ds_val, df_sq, sq_inds_train, sq_inds_val, wiki_ds, wiki_inds_train, wiki_inds_val, args: ArgsMixedDecoderTrain):
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

    model_cfg = parse_yaml_file_as(MixedDecoderCfg, args.model_cfg_fpath)
    model_cfg = copy_override_mixed_decoder_cfg(
        model_cfg, pretrained_model_name=args.bert_model_name, emb_type=args.bert_emb_type,
        inp_len=args.inp_len, decoder_type=args.decoder_type, decoder_model_name=args.decoder_model_name,
        max_seq_len=args.max_seq_len, use_sep=args.use_sep, prompt_all=args.prompt_all, emb_exp_rate=args.emb_exp_rate,
        emb_win_min_size=args.emb_win_min_size, emb_win_max_size=args.emb_win_max_size,
        min_next_toks=args.min_next_toks, train_ds_type=args.train_ds_type,
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
    checkpoint = None
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists(),\
            (f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
             f'but file {last_checkpoint_path} does not exits.')

    if last_checkpoint_path.exists():
        log(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        log(f'Checkpoint with keys {list(checkpoint.keys())} loaded')
    else:
        if rank == 0:
            to_yaml_file(train_path / MIXED_DECODER_MODEL_CFG_FNAME, model_cfg)
    tkz = AutoTokenizer.from_pretrained(args.bert_model_name)

    log(model_cfg)
    model = MixedDecoder(model_cfg, tkz)

    model.load_pretrained(checkpoint)

    model.to(device)
    if args.world_size > 1:
        find_unused_parameters = True
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
    else:
        ddp_model = model

    params = ddp_model.parameters()
    optimizer = instantiate_torch_optimizer(args.optimizer_name, params, lr=args.learning_rate, **args.optimizer_params)
    scheduler = instantiate_torch_lr_scheduler(args.learning_rate_scheduler_name, optimizer, **args.learning_rate_scheduler_params)

    last_epoch, val_loss_min, shuffle = -1, None, False
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        del checkpoint

    n_special_toks = 1000
    if args.train_ds_type == MixedDecoderDsType.Cite:
        ds_train = MaskedCiteDataset(ds_train, tkz, max_seq_len=args.inp_len, n_special_toks=n_special_toks, mask_cfg=mask_cfg, prompt_all=args.prompt_all, device=device)
        ds_val = MaskedCiteDataset(ds_val, tkz, max_seq_len=args.inp_len, n_special_toks=n_special_toks, mask_cfg=mask_cfg, prompt_all=args.prompt_all, device=device)
        ds_train.shuffle(seed=(args.random_seed or 0) + rank)
        ds_val.shuffle(seed=(args.random_seed or 0) + rank)
        train_batch_it = create_masked_cite_dataloader(ds_train, batch_size=args.docs_batch_size)
        val_batch_it = create_masked_cite_dataloader(ds_val, batch_size=args.docs_batch_size)
    elif args.train_ds_type == MixedDecoderDsType.Qna:
        max_chunks = max(args.emb_win_max_size, 1)
        ds_train = QnaCiteDataset(
            df_sq, sq_inds_train, tkz, inp_len=args.inp_len, max_chunks=max_chunks, device=device,
        )
        ds_val = QnaCiteDataset(
            df_sq, sq_inds_val, tkz, inp_len=args.inp_len, max_chunks=max_chunks, device=device,
        )
        ds_train.shuffle(seed=(args.random_seed or 0) + rank)
        ds_val.shuffle(seed=(args.random_seed or 0) + rank)
        train_batch_it = create_qna_cite_dataloader(ds_train, batch_size=args.docs_batch_size)
        val_batch_it = create_qna_cite_dataloader(ds_val, batch_size=args.docs_batch_size)
    elif args.train_ds_type == MixedDecoderDsType.Next:
        ds_train = NextTokWikiDataset(
            wiki_ds, wiki_inds_train, tkz, inp_len=args.inp_len, min_next_toks=args.min_next_toks,
            emb_win_min_size=max(args.emb_win_min_size, 1), emb_win_max_size=max(args.emb_win_max_size, 1),
            device=device,
        )
        ds_val = NextTokWikiDataset(
            wiki_ds, wiki_inds_val, tkz, inp_len=args.inp_len, min_next_toks=args.min_next_toks,
            emb_win_min_size=max(args.emb_win_min_size, 1), emb_win_max_size=max(args.emb_win_max_size, 1),
            device=device,
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
            loss_dict, _ = ddp_model(batch)
            loss = loss_dict['loss']
            loss.backward()

            if rank == 0:
                if grad_log_ind % grad_log_interval == 0:
                    log_weights_grads_stats(grad_log_step, ddp_model, tbsw)
                    grad_log_step += 1
                grad_log_ind += 1

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
                if args.world_size > 1:
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

    tkz = AutoTokenizer.from_pretrained(args.bert_model_name)

    wiki_ds, wiki_inds_train, wiki_inds_val = None, None, None
    if args.train_ds_type == MixedDecoderDsType.Cite:
        ds_train, ds_val = load_split_wiki_dataset(
            data_path=args.data_path, tkz=tkz, max_seq_len=args.inp_len, val_split_ratio=0.05,
            mask_cfg=mask_cfg, random_seed=args.random_seed,
        )
        df_sq, sq_inds_train, sq_inds_val = None, None, None
    elif args.train_ds_type == MixedDecoderDsType.Qna:
        df_sq, sq_inds_train, sq_inds_val = load_split_squadv2(exclude_empty_answers=True)
        ds_train, ds_val = None, None
    elif args.train_ds_type == MixedDecoderDsType.Next:
        wiki_ds, wiki_inds_train, wiki_inds_val = load_split_wiki_for_next(
            data_path=args.data_path, random_seed=args.random_seed,
        )
        ds_train, ds_val = None, None
        df_sq, sq_inds_train, sq_inds_val = None, None, None
    else:
        raise ValueError(f'Unsupported train_ds_type: {args.train_ds_type}')

    mp.spawn(train, args=(
        ds_train, ds_val, df_sq, sq_inds_train, sq_inds_val, wiki_ds, wiki_inds_train, wiki_inds_val, args,
    ), nprocs=args.world_size, join=True)

    return 0


if __name__ == '__main__':
    run_and_exit(
        ArgsMixedDecoderTrain, main, 'Train MixedDecoder model (Encoder-Decoder with causal attention) multi GPU training.', exception_handler=rethrow,
    )
