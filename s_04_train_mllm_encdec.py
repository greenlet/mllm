import shutil
from datetime import datetime
import itertools as it
from pathlib import Path
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from torch import nn
import torch.nn.functional as F
from tqdm import trange

from mllm.data.dsfixed import DsLoader
from mllm.utils.utils import DT_PAT_RE, parse_dt_str
from mllm.model.mllm_encdec import MllmEncdec
from mllm.model.mllm_ranker import MllmRanker
from mllm.model.config import CfgMllmRanker, create_mllm_ranker_cfg, create_mllm_encdec_cfg
from mllm.tokenization.chunk_tokenizer import gen_ds_fnames, parse_out_subdir, gen_doc_tokens, split_doc_embs, \
    calc_max_inp_size, gen_all_tokens
from mllm.utils.utils import gen_dt_str
from transformers import GPT2Tokenizer



class ArgsTrain(BaseModel):
    ds_dir_path: Path = Field(
        None,
        required=False,
        description='Dataset directory path. Must contain .csv and .np files with tokenized text.',
        cli=('--ds-dir-path',),
    )
    train_root_path: Path = Field(
        ...,
        required=True,
        description='Path to train root directory. New train subdirectory will be created within each new run.',
        cli=('--train-root-path',),
    )
    train_subdir: str = Field(
        '',
        required=False,
        description='Train subdirectory. Can have values: "last", "<subdirectory-name>". When set to "last", '
            'last subdirectory of TRAIN_ROOT_PATH containing training snapshot will be taken.',
        cli=('--train-subdir',)
    )
    docs_batch_size: Optional[int] = Field(
        3,
        required=False,
        description='Documents batch size. Must be greater or equal than 2.',
        cli=('--docs-batch-size',),
    )
    max_chunks_per_doc: Optional[int] = Field(
        3,
        required=False,
        description='Maximum number of consecutive chunks per document taken in each butch. '
                    'Batch chunk max size will be DOCS_BATCH_SIZE * MAX_CHUNKS_PER_DOC.',
        cli=('--max-chunks-per-doc',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run training on. Can have values: "cpu", "cuda"',
        cli=('--device',)
    )
    epochs: int = Field(
        None,
        required=True,
        description='Number of training epochs.',
        cli=('--epochs',),
    )
    learning_rate: float = Field(
        0.001,
        required=False,
        description='Initial learning rate of the training process.',
        cli=('--learning-rate',)
    )
    train_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of training steps per epoch.',
        cli=('--train-epoch-steps',),
    )
    val_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of validation steps per epoch.',
        cli=('--val-epoch-steps',),
    )


SUBDIR_PAT_STR = re.compile(r'^\w+\-(%s)-.+$' % DT_PAT_RE)
SUBDIR_PAT = re.compile(SUBDIR_PAT_STR)


def gen_train_subdir(ds_dir_path: Path) -> str:
    dt_str = gen_dt_str()
    subdir = f'encdec-{dt_str}-{ds_dir_path.parent.name}-{ds_dir_path.name}'
    return subdir


def encdec_prob_loss(logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> torch.Tensor:
    probs_pred = torch.softmax(logits_pred, dim=-1)
    probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt.to(torch.int64).unsqueeze(-1))
    loss = -torch.mean(torch.log(probs_gt))
    return loss



def concat_tokens(*chunks: torch.Tensor, shuffle: bool = True) ->torch.Tensor:
    if shuffle:
        chunks = list(chunks)
        np.random.shuffle(chunks)
    return torch.concat(chunks, dim=0)


# chunks: input token chunks of the shape [n_docs x n_tokens_per_doc]
def remove_tokens(chunks: torch.Tensor, pad_tok: int, rem_ratio: float = 0.1) -> torch.Tensor:
    p = rem_ratio
    mask = torch.distributions.Bernoulli(probs=p).sample(chunks.size()).to(chunks.device)
    res = chunks.clone()
    res[mask.bool()] = pad_tok
    return res


def find_last_train_subdir(train_root_path: Path) -> Optional[Path]:
    dt_last: Optional[datetime] = None
    subdir_last: Optional[str] = None
    for subpath in train_root_path.iterdir():
        if not subpath.is_dir():
            continue
        m = SUBDIR_PAT.match(subpath.name)
        dt_cur = parse_dt_str(m.group(1))
        if dt_cur is None:
            continue
        if dt_last is None or dt_cur > dt_last:
            dt_last = dt_cur
            subdir_last = subpath.name
    if subdir_last is not None:
        return train_root_path / subdir_last


def main(args: ArgsTrain) -> int:
    print(args)

    device = torch.device(args.device)

    checkpoint = None
    if args.train_subdir == 'last':
        train_path = find_last_train_subdir(args.train_root_path)
        if train_path is None:
            print(f'Cannot find last subdirectory of the format `{SUBDIR_PAT_STR}` in {args.train_root_path}')
            sys.exit(1)
    elif args.train_subdir:
        train_path = args.train_root_path / args.train_subdir
        assert train_path.exists(), f'Directory {train_path} does not exist'
    else:
        train_subdir = gen_train_subdir(args.ds_dir_path)
        train_path = args.train_root_path / train_subdir
        train_path.mkdir(parents=True, exist_ok=True)
    print(f'train_path: {train_path}')

    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    checkpoint = None
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists(),\
            (f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
             f'but file {last_checkpoint_path} does not exits.')
        print(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        print(f'Checkpoint with keys {list(checkpoint.keys())} loaded')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=100000)
    tok_dict = gen_all_tokens(tokenizer)
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind
    ds_loader = DsLoader(
        ds_dir_path=args.ds_dir_path, docs_batch_size=args.docs_batch_size, max_chunks_per_doc=args.max_chunks_per_doc,
        pad_tok=pad_tok, qbeg_tok=qbeg_tok, qend_tok=qend_tok, device=device
    )

    inp_len = ds_loader.emb_chunk_size if ds_loader.fixed_size else calc_max_inp_size(ds_loader.emb_chunk_size)
    print(f'Creating model with vocab size = {len(tokenizer)}')

    torch.autograd.set_detect_anomaly(True)

    model_cfg = create_mllm_encdec_cfg(
        n_vocab=len(tokenizer), d_word_wec=256, inp_len=inp_len,
        enc_n_layers=1, dec_n_layers=1,
        n_heads=8, d_model=256, d_inner=1024,
        pad_idx=pad_tok, dropout_rate=0.1, enc_with_emb_mat=True,
    )
    input_zeros_ratio = 0.2
    print(model_cfg)
    model = MllmEncdec(model_cfg).to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    last_epoch, val_loss_min = -1, None
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        ds_loader.shuffle(train=True)
        ds_loader.shuffle(train=False)

    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    calc_batches = lambda n_docs: n_docs // args.docs_batch_size + (n_docs % args.docs_batch_size > 1)
    n_batches_train = calc_batches(ds_loader.n_docs_train)
    n_batches_val = calc_batches(ds_loader.n_docs_val)
    loss_fn = encdec_prob_loss
    # loss_fn = nn.CrossEntropyLoss()
    graph_written = True
    i_train, i_val = 0, 0
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss = 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = ds_loader.get_batch(i_train, train=True)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()

            chunks = concat_tokens(docs_chunks, target_chunks)
            chunks_inp = remove_tokens(chunks, pad_tok, input_zeros_ratio)

            optimizer.zero_grad()

            out_logits = model(chunks_inp)
            if not graph_written:
                tbsw.add_graph(model, docs_chunks, verbose=True, use_strict_trace=False)
                graph_written = True

            loss = loss_fn(out_logits, chunks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            i_train += 1
            if i_train == n_batches_train:
                ds_loader.shuffle(train=True)
                i_train %= n_batches_train

            # if i_train == 2:
            #     import sys
            #     sys.exit()

            pbar.set_postfix_str(f'Train. loss: {loss.item():.6f}')
        pbar.close()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)

        model.eval()
        val_loss = 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = ds_loader.get_batch(i_val, train=False)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()
            
            chunks = concat_tokens(docs_chunks, target_chunks)
            out_logits = model(chunks)

            loss = loss_fn(out_logits, chunks)
            val_loss += loss.item()

            i_val += 1
            if i_val == n_batches_val:
                ds_loader.shuffle(train=False)
                i_val %= n_batches_val

            pbar.set_postfix_str(f'Val. loss: {loss.item():.6f}')
        pbar.close()
        val_loss /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)

        print(f'Train loss: {train_loss:.6f}. Val loss: {val_loss:.6f}')
        best = False
        if val_loss_min is None or val_loss < val_loss_min:
            val_loss_str = f'{val_loss_min}' if val_loss_min is None else f'{val_loss_min:.6f}'
            print(f'Val min loss change: {val_loss_str} --> {val_loss:.6f}')
            val_loss_min = val_loss
            best = True

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_epoch': epoch,
            'val_loss_min': val_loss_min,
        }
        print(f'Saving checkpoint to {last_checkpoint_path}')
        torch.save(checkpoint, last_checkpoint_path)

        if best:
            print(f'New val loss minimum: {val_loss_min:.6f}. Saving checkpoint to {best_checkpoint_path}')
            shutil.copyfile(last_checkpoint_path, best_checkpoint_path)

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsTrain, main, 'Train Mllm model.', exception_handler=rethrow)


