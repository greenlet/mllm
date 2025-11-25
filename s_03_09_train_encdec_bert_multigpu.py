import os
from pathlib import Path
from typing import Any, Optional
import shutil

from datasets import Dataset
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
import torch
import torch.utils.tensorboard as tb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from transformers import AutoTokenizer

from mllm.config.model import HgEnhanceType, EncdecBertCfg, copy_override_encdec_bert_cfg, BertEmbType, \
    gen_prefpostfix_encdec_bert
from mllm.exp.args import ENCDEC_BERT_MODEL_CFG_FNAME, create_bool_str_field, get_pretrained_model_path, is_arg_true, mask_tokens_ARG, next_tok_pred_ARG, masked_loss_for_encoder_ARG
from mllm.model.encdec_ranker_hg import EncdecBert, EncdecBertAgg
from mllm.model.losses import EncdecMaskPadBatchLoss, EncdecPadBatchLoss, EncdecMaskPadItemLoss, LossesStats, accum_losses, log_losses_to_tb, losses_to_str
from mllm.train.mask_utils import MaskCfg
from mllm.train.utils import find_create_train_path, log_weights_grads_stats, get_wiki_ds_batch_iterators2
from mllm.train.encdec_bert import create_dataloader, create_dataloader_iter, load_masked_wiki_dataset


enforce_encoder_mask_understanding_ARG = '--enforce-encoder-mask-understanding', 'Enforce encoder embeddings for both unmasked and masked inputs '
'to be similar'



class ArgsEncdecBertMultigpuTrain(BaseModel):
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
        description='Path to EncdecHg model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    bert_model_name: str = Field(
        'bert-base-uncased',
        description='Pretrained BERT model name. Must be a model from Huggingface models hub (bert-base-*, bert-large-*).',
        cli=('--bert-model-name',),
    )
    bert_emb_type: BertEmbType = Field(
        BertEmbType.Cls,
        description=f'Bert embedding type. Can have values: {list(x.value for x in BertEmbType)}',
        cli=('--bert-emb-type',),
    )
    inp_len: int = Field(
        ...,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    dec_enhance_type: HgEnhanceType = Field(
        HgEnhanceType.Matmul,
        description=f'Decoder layer enhance type. Can have values: {list(x.value for x in HgEnhanceType)}',
        cli=('--dec-enhance-type',),
    )
    dec_n_layers: int = Field(
        0,
        description='Decoder number of layers.',
        cli=('--dec-n-layers',),
    )
    dec_n_similar_layers: int = Field(
        ...,
        description='Number of consecutive similar attention layers for each decoder level.',
        cli=('--dec-n-similar-layers',),
    )

    mask_tokens_STR: str = create_bool_str_field(*mask_tokens_ARG)
    @property
    def mask_tokens(self) -> bool:
        return is_arg_true(mask_tokens_ARG[0], self.mask_tokens_STR)

    mask_sep_freq: float = Field(
        ...,
        description='Sparse mask frequency from 0 to 1. When MASK_SEP_FREQ=0.2 this type of mask will be applied in 20% of cases randomly. '
                    'Must hold: 0 <= MASK_SEP_FREQ and MASK_SEP_FREQ + MASK_SEQ_FREQ <= 1',
        cli=('--mask-sep-freq',),
    )
    mask_sep_frac: float = Field(
        ...,
        description='Fraction of the input to mask using sparse masking.',
        cli=('--mask-sep-frac',),
    )
    mask_seq_freq: float = Field(
        ...,
        description='Sequential mask frequency from 0 to 1. When MASK_SEQ_FREQ=0.2 this type of mask will be applied in 20% of cases randomly. '
                    'Must hold: 0 <= MASK_SEQ_FREQ and MASK_SEP_FREQ + MASK_SEQ_FREQ <= 1',
        cli=('--mask-seq-freq',),
    )
    mask_seq_max_frac: float = Field(
        ...,
        description='Fraction of the input to calculate maximum length of tokens sequence to mask. Resulting value is combined '
                    'with MASK_SEQ_MAX_LEN using min() function.',
        cli=('--mask-seq-max-frac',),
    )
    mask_seq_max_len: int = Field(
        ...,
        description='Maximum length of tokens sequence to mask. Combined with value derived from MASK_SEQ_MAX_FRAC using min() function.',
        cli=('--mask-seq-max-len',),
    )
    mask_n_last_toks: int = Field(
        0,
        description='Number of last tokens to always mask. When 0, no tokens are masked.',
        cli=('--mask-n-last-toks',),
    )

    next_tok_pred_STR: str = create_bool_str_field(*next_tok_pred_ARG)
    @property
    def next_tok_pred(self) -> bool:
        return is_arg_true(next_tok_pred_ARG[0], self.next_tok_pred_STR)

    masked_loss_for_encoder_STR: str = create_bool_str_field(*masked_loss_for_encoder_ARG)
    @property
    def masked_loss_for_encoder(self) -> bool:
        return is_arg_true(masked_loss_for_encoder_ARG[0], self.masked_loss_for_encoder_STR)

    dec_dropout_rate: float = Field(
        0.0,
        required=False,
        description='Decoder dropout rate.',
        cli=('--dec-dropout-rate',),
    )
    docs_batch_size: int = Field(
        3,
        description='Documents batch size. Must be greater or equal than 2.',
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
    pretrained_model_path: Optional[Path] = Field(
        None,
        description='Path to EncdecHg model train directory.',
        cli=('--pretrained-model-path',),
    )

    enforce_encoder_mask_understanding_STR: str = create_bool_str_field(*enforce_encoder_mask_understanding_ARG)
    @property
    def enforce_encoder_mask_understanding(self) -> bool:
        return is_arg_true(enforce_encoder_mask_understanding_ARG[0], self.enforce_encoder_mask_understanding_STR)

    world_size: int = Field(
        1,
        description='Number of GPU instances to use for distributed training.',
        cli=('--world-size',),
    )



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # # such as CUDA, MPS, MTIA, or XPU.
    # acc = torch.accelerator.current_accelerator()
    # backend = torch.distributed.get_default_backend_for_device(acc)
    backend = 'nccl'
    # backend = 'c10d'
    # initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank: int, ds_train: Dataset, ds_val: Dataset, args: ArgsEncdecBertMultigpuTrain):
    print(f'Running DDP training on rank {rank}.')
    def log(*msgs: Any, forall: bool = False):
        if rank == 0 or forall:
            print(*msgs)

    setup(rank, args.world_size)

    pretrained_model_path = get_pretrained_model_path(pretrained_model_path)

    device = torch.device(f'cuda:{rank}')

    model_cfg = parse_yaml_file_as(EncdecBertCfg, args.model_cfg_fpath)
    model_cfg = copy_override_encdec_bert_cfg(
        model_cfg, pretrained_model_name=args.bert_model_name, emb_type=args.bert_emb_type, inp_len=args.inp_len, dec_enhance_type=args.dec_enhance_type,
        dec_n_layers=args.dec_n_layers, dec_n_similar_layers=args.dec_n_similar_layers, dec_dropout_rate=args.dec_dropout_rate,
    )

    mask_cfg = None
    if args.mask_tokens:
        mask_cfg = MaskCfg(
            sep_freq=args.mask_sep_freq, sep_frac=args.mask_sep_frac, seq_freq=args.mask_seq_freq, seq_max_frac=args.mask_seq_max_frac,
            seq_max_len=args.mask_seq_max_len, n_last_toks=args.mask_n_last_toks,
        )
    prefix, suffix = gen_prefpostfix_encdec_bert(
        model_cfg, mask_cfg=mask_cfg, pretrained_model_path=pretrained_model_path, next_tok_pred=args.next_tok_pred
    )
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
        chkpt_model_cfg = parse_yaml_file_as(EncdecBertCfg, train_path / ENCDEC_BERT_MODEL_CFG_FNAME)
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        if rank == 0:
            to_yaml_file(train_path / ENCDEC_BERT_MODEL_CFG_FNAME, model_cfg)

    tkz = AutoTokenizer.from_pretrained(model_cfg.enc_bert.pretrained_model_name)

    log(model_cfg)
    model = EncdecBertAgg(
        model_cfg, tkz, enforce_enc_mask_understanding=args.enforce_encoder_mask_understanding,
        next_tok_pred=args.next_tok_pred, masked_loss_for_encoder=args.masked_loss_for_encoder,
    )

    if checkpoint is None:
        model.load_pretrained(pretrained_model_path)

    last_epoch, val_loss_min, shuffle = -1, None, False
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        del checkpoint
        shuffle = True

    model.to(device)
    # find_unused_parameters = False
    find_unused_parameters = True
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
    params = ddp_model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    train_batch_it = create_dataloader_iter(
        ds_train, batch_size=args.docs_batch_size, num_workers=2, distributed=True, drop_last=False,
    )
    val_batch_it = create_dataloader_iter(
        ds_val, batch_size=args.docs_batch_size, num_workers=2, distributed=True, drop_last=False,
    )

    sched_wait_steps = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6, min_lr=1e-8)
    lr = optimizer.param_groups[0]['lr']
    log(f'Scheduler {scheduler.__class__.__name__} lr: {lr:0.10f}.')
    if rank == 0:
        tbsw = tb.SummaryWriter(log_dir=str(train_path))
        log(ddp_model)

        grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
        prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
        if prev_train_steps > 0:
            grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1
    
    dist.barrier()
    for epoch in range(last_epoch + 1, args.epochs):
        ddp_model.train()
        train_losses = LossesStats()
        train_loss = 0.0
        if rank == 0:
            pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        else:
            pbar = range(args.train_epoch_steps)
        for _ in pbar:
            item = next(train_batch_it)
            # log(type(item))
            # for k, v in item.items():
            #     log(f'  {k}: {v.shape}')
            tokens_inp, tokens_inp_aug = item['input_ids'], item['input_ids_masked']

            optimizer.zero_grad()
            loss_dict = ddp_model(tokens_inp_aug, tokens_inp)
            loss = loss_dict['loss']
            loss.backward()

            if rank == 0:
                # Gradients must be available after loss.backward()
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
        for _ in pbar:
            item = next(val_batch_it)
            tokens_inp, tokens_inp_aug = item['input_ids'], item['input_ids_masked']

            with torch.no_grad():
                loss_dict = ddp_model(tokens_inp_aug, tokens_inp)
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

        if epoch >= sched_wait_steps:
            scheduler.step(val_loss)
        
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
                'last_epoch': epoch,
                'val_loss_min': val_loss_min,
            }
            print(f'Saving checkpoint to {last_checkpoint_path}')
            torch.save(checkpoint, last_checkpoint_path)

            if best:
                print(f'New val loss minimum: {val_loss_min:.6f}. Saving checkpoint to {best_checkpoint_path}')
                shutil.copyfile(last_checkpoint_path, best_checkpoint_path)

    cleanup()


def main(args: ArgsEncdecBertMultigpuTrain) -> int:
    print(args)
   
    mask_cfg = None
    if args.mask_tokens:
        mask_cfg = MaskCfg(
            sep_freq=args.mask_sep_freq, sep_frac=args.mask_sep_frac, seq_freq=args.mask_seq_freq, seq_max_frac=args.mask_seq_max_frac,
            seq_max_len=args.mask_seq_max_len, n_last_toks=args.mask_n_last_toks,
        )

    tkz = AutoTokenizer.from_pretrained(args.bert_model_name)
    ds_train, ds_val = load_masked_wiki_dataset(
        data_path=args.data_path, tkz=tkz, max_seq_len=args.inp_len, val_split_ratio=0.05,
        mask_cfg=mask_cfg, random_seed=args.random_seed,
    )

    mp.spawn(train, args=(
        ds_train, ds_val, args,
    ), nprocs=args.world_size, join=True)


    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(
        ArgsEncdecBertMultigpuTrain, main, 'Train Encoder-Decoder Hourglass model multi GPU training.', exception_handler=rethrow,
    )

