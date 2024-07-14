import shutil
from datetime import datetime
import itertools as it
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from torch import nn
from tqdm import trange

from mllm.data.dsfixed import DsLoader
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


def gen_train_subdir(ds_dir_path: Path) -> str:
    dt_str = gen_dt_str()
    subdir = f'encdec-{dt_str}-{ds_dir_path.parent.name}-{ds_dir_path.name}'
    return subdir


def encdec_prob_loss(logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> torch.Tensor:
    probs_pred = torch.softmax(logits_pred, dim=-1)
    probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt.to(torch.int64).unsqueeze(-1))
    loss = -torch.mean(torch.log(probs_gt))
    return loss


def main(args: ArgsTrain) -> int:
    print(args)

    device = torch.device(args.device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=100000)
    tok_dict = gen_all_tokens(tokenizer)
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind
    ds_loader = DsLoader(
        ds_dir_path=args.ds_dir_path, docs_batch_size=args.docs_batch_size, max_chunks_per_doc=args.max_chunks_per_doc,
        pad_tok=pad_tok, qbeg_tok=qbeg_tok, qend_tok=qend_tok, device=device
    )

    train_subdir = gen_train_subdir(args.ds_dir_path)
    train_path = args.train_root_path / train_subdir
    train_path.mkdir(parents=True, exist_ok=True)
    inp_len = ds_loader.emb_chunk_size if ds_loader.fixed_size else calc_max_inp_size(ds_loader.emb_chunk_size)
    print(f'Creating model with vocab size = {len(tokenizer)}')

    torch.autograd.set_detect_anomaly(True)

    model_cfg = create_mllm_encdec_cfg(
        n_vocab=len(tokenizer), d_word_wec=256, inp_len=inp_len,
        enc_n_layers=1, dec_n_layers=1,
        n_heads=8, d_model=256, d_inner=1024,
        pad_idx=pad_tok, dropout_rate=0.0, enc_with_emb_mat=True,
    )
    print(model_cfg)
    model = MllmEncdec(model_cfg).to(device)
    params = model.parameters()
    # params = [p for n, p in model.named_parameters()]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)
    tbsw = tb.SummaryWriter(log_dir=str(train_path))
    val_loss_min = None
    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'

    calc_batches = lambda n_docs: n_docs // args.docs_batch_size + (n_docs % args.docs_batch_size > 1)
    n_batches_train = calc_batches(ds_loader.n_docs_train)
    n_batches_val = calc_batches(ds_loader.n_docs_val)
    loss_fn = encdec_prob_loss
    # loss_fn = nn.CrossEntropyLoss()
    graph_written = True
    for epoch in range(args.epochs):
        model.train()
        # model.eval()
        train_loss = 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for i in pbar:
            i_batch = i % n_batches_train
            if i > 0 and i_batch == 0:
                ds_loader.shuffle(train=True)
            batch = ds_loader.get_batch(i_batch, train=True)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()

            optimizer.zero_grad()

            out_logits = model(docs_chunks)
            if not graph_written:
                tbsw.add_graph(model, docs_chunks, verbose=True, use_strict_trace=False)
                graph_written = True

            loss = loss_fn(out_logits, docs_chunks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # if i == 2:
            #     import sys
            #     sys.exit()

            pbar.set_postfix_str(f'Train. loss: {loss.item():.6f}')
        pbar.close()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)

        model.eval()
        val_loss = 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for i in pbar:
            i_batch = i % n_batches_val
            if i > 0 and i_batch == 0:
                ds_loader.shuffle(train=False)
            batch = ds_loader.get_batch(i_batch, train=False)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()
            out_logits = model(docs_chunks)

            loss = loss_fn(out_logits, docs_chunks)
            val_loss += loss.item()

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


