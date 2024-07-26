import itertools
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.data.dsfixed import DsLoader
from mllm.model.mllm_ranker import MllmRanker
from mllm.model.mllm_encdec import MllmEncdec
from mllm.model.config import create_mllm_ranker_cfg, create_mllm_encdec_cfg
from mllm.tokenization.chunk_tokenizer import calc_max_inp_size, gen_all_tokens
from mllm.train.args import ArgsTrain
from mllm.train.utils import gen_train_subdir, find_create_train_path
from mllm.utils.utils import gen_dt_str
from transformers import GPT2Tokenizer, PreTrainedTokenizer


def rank_prob_loss(prob_pred: torch.Tensor, mask_gt: torch.Tensor, tgt_weight: float = 0.5) -> torch.Tensor:
    # prob_pred = prob_pred.squeeze(-1)
    # mask_gt = mask_gt.unsqueeze(0)
    prob_pred = prob_pred.squeeze()
    # prob_tgt, prob_nontgt = prob_pred[mask_gt], prob_pred[~mask_gt]
    prob_tgt = torch.masked_select(prob_pred, mask_gt)
    prob_nontgt = torch.masked_select(prob_pred, ~mask_gt)

    # prob_tgt, prob_nontgt = prob_tgt**2, prob_nontgt**2
    loss_tgt = 1 - torch.mean(prob_tgt)
    loss_nontgt = torch.mean(prob_nontgt)

    # print(f'loss_tgt = {loss_tgt}. loss_nontgt = {loss_nontgt}')
    # loss = tgt_weight * loss_tgt + (1 - tgt_weight) * loss_nontgt
    # loss = tgt_weight * loss_tgt + (1 - tgt_weight) * loss_nontgt
    loss = loss_tgt + loss_nontgt
    # loss = loss_tgt + loss_nontgt
    # print(loss_tgt.item(), loss_nontgt.item())
    return loss


class RankProbLoss(nn.Module):
    def __init__(self, target_weight: float = 0.5):
        super().__init__()
        self.target_weight = target_weight

    def forward(self, prob_pred: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prob_pred = prob_pred.squeeze()
        prob_tgt = torch.masked_select(prob_pred, mask_gt)
        prob_nontgt = torch.masked_select(prob_pred, ~mask_gt)
        loss_tgt = -torch.mean(torch.log(prob_tgt))
        loss_nontgt = -torch.mean(torch.log(1 - prob_nontgt))
        loss = self.target_weight * loss_tgt + (1 - self.target_weight) * loss_nontgt
        return loss, loss_tgt, loss_nontgt


class TokenAugmenter:
    tokenizer: PreTrainedTokenizer
    act_prob: float
    min_tokens: int
    max_tokens: int

    def __init__(self, tokenizer: PreTrainedTokenizer, act_prob: float = 0.8, min_tokens: int = 4, max_tokens: int = 50, seed: int = 11):
        self.tokenizer = tokenizer
        self.act_prob = act_prob
        assert min_tokens <= max_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        np.random.seed(seed)

    def __call__(self, tokens: list[list[int]]) -> list[list[int]]:
        n_tokens = sum(len(t) for t in tokens)
        p = np.random.uniform()
        if p > self.act_prob or n_tokens <= self.min_tokens:
            return tokens
        max_tokens = min(self.min_tokens, n_tokens)
        n_res = np.random.randint(self.min_tokens, max_tokens + 1)
        i1 = np.random.randint(n_tokens)
        i2 = i1 + n_res
        n_beg = 0
        if i2 > n_tokens:
            n_beg = i2 - n_tokens
            i2 = n_tokens
        n_end = i2 - i1

        res = []
        for t in tokens:
            if n_beg > 0:
                n1 = len(t)
                t = t[n_beg:]
                n2 = len(t)
                n_beg -= n1 - n2
            elif n_end > 0:
                n1 = len(t)
                t = t[:-n_end]
                n2 = len(t)
                n_end -= n1 - n2
            res.append([*t])

        return res


def main(args: ArgsTrain) -> int:
    print(args)

    device = torch.device(args.device)

    train_path = find_create_train_path(
        args.train_root_path, 'ranker', f'{args.ds_dir_path.parent.name}-{args.ds_dir_path.name}', args.train_subdir)
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

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=10000)
    tok_dict = gen_all_tokens(tokenizer)
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind
    # n_total = 1000
    n_total = 0
    ds_loader = DsLoader(
        ds_dir_path=args.ds_dir_path, docs_batch_size=args.docs_batch_size, max_chunks_per_doc=args.max_chunks_per_doc,
        pad_tok=pad_tok, qbeg_tok=qbeg_tok, qend_tok=qend_tok, device=device, n_total=n_total,
    )

    inp_len = ds_loader.emb_chunk_size if ds_loader.fixed_size else calc_max_inp_size(ds_loader.emb_chunk_size)
    print(f'Creating model with vocab size = {len(tokenizer)}')

    torch.autograd.set_detect_anomaly(True)

    model_cfg = create_mllm_ranker_cfg(
        n_vocab=len(tokenizer), inp_len=inp_len, d_word_wec=256,
        n_levels=1, enc_n_layers=1, dec_n_layers=1,
        n_heads=8, d_k=32, d_v=32, d_model=256, d_inner=1024,
        pad_idx=pad_tok, dropout_rate=0.1, enc_with_emb_mat=True,
    )
    print(model_cfg)
    model = MllmRanker(model_cfg).to(device)

    if args.pretrained_model_path is not None and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path)
        model_encdec_cfg = create_mllm_encdec_cfg(
            n_vocab=len(tokenizer), d_word_wec=256, inp_len=inp_len,
            enc_n_layers=1, dec_n_layers=1,
            n_heads=8, d_model=256, d_inner=1024,
            pad_idx=pad_tok, dropout_rate=0.1, enc_with_emb_mat=True,
        )
        model_encdec = MllmEncdec(model_encdec_cfg).to(device)
        model_encdec.load_state_dict(pretrained_checkpoint['model'])
        print(f'Load model weights for vocab_encoder:', list(model_encdec.vocab_encoder.state_dict().keys()))
        model.vocab_encoder.load_state_dict(model_encdec.vocab_encoder.state_dict())
        print(f'Load model weights for encoder:', list(model_encdec.encoder.state_dict().keys()))
        model.encoders[0].load_state_dict(model_encdec.encoder.state_dict())

    params = model.parameters()
    # params = [p for n, p in model.named_parameters()]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)

    last_epoch, val_loss_min = -1, None
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        ds_loader.shuffle(train=True)
        ds_loader.shuffle(train=False)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, threshold=1e-4, min_lr=1e-7)
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    calc_batches = lambda n_docs: n_docs // args.docs_batch_size + (n_docs % args.docs_batch_size > 1)
    n_batches_train = calc_batches(ds_loader.n_docs_train)
    n_batches_val = calc_batches(ds_loader.n_docs_val)
    # loss_fn = rank_prob_loss
    loss_fn = RankProbLoss()
    token_augmenter = TokenAugmenter(tokenizer=tokenizer)
    # token_augmenter = None
    graph_written = True
    i_train, i_val = 0, 0
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss, train_loss_tgt, train_loss_nontgt = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = ds_loader.get_batch(i_train, train=True, target_augmenter=token_augmenter)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()

            optimizer.zero_grad()

            out_dec_rank = model(target_chunks, docs_chunks)
            if not graph_written:
                tbsw.add_graph(model, [target_chunks, docs_chunks], verbose=True, use_strict_trace=False)
                graph_written = True

            loss, loss_tgt, loss_nontgt = loss_fn(out_dec_rank, target_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_tgt += loss_tgt.item()
            train_loss_nontgt += loss_nontgt.item()

            i_train += 1
            if i_train == n_batches_train:
                ds_loader.shuffle(train=True)
                i_train %= n_batches_train

            # if i == 2:
            #     import sys
            #     sys.exit()

            pbar.set_postfix_str(f'Train. loss: {loss.item():.6f}. loss_tgt: {loss_tgt.item():.6f}. loss_nontgt: {loss_nontgt.item():.6f}')
        pbar.close()
        train_loss /= args.train_epoch_steps
        train_loss_tgt /= args.train_epoch_steps
        train_loss_nontgt /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)
        tbsw.add_scalar('LossTgt/Train', train_loss_tgt, epoch)
        tbsw.add_scalar('LossNontgt/Train', train_loss_nontgt, epoch)

        model.eval()
        val_loss, val_loss_tgt, val_loss_nontgt = 0, 0, 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = ds_loader.get_batch(i_val, train=False, target_augmenter=token_augmenter)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()
            out_dec_rank = model(target_chunks, docs_chunks)

            loss, loss_tgt, loss_nontgt = loss_fn(out_dec_rank, target_mask)
            val_loss += loss.item()
            val_loss_tgt += loss_tgt.item()
            val_loss_nontgt += loss_nontgt.item()

            i_val += 1
            if i_val == n_batches_val:
                ds_loader.shuffle(train=False)
                i_val %= n_batches_val

            pbar.set_postfix_str(f'Val. loss: {loss.item():.6f}. loss_tgt: {loss_tgt.item():.6f}. loss_nontgt: {loss_nontgt.item():.6f}')
        pbar.close()
        val_loss /= args.val_epoch_steps
        val_loss_tgt /= args.val_epoch_steps
        val_loss_nontgt /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)
        tbsw.add_scalar('LossTgt/Val', val_loss_tgt, epoch)
        tbsw.add_scalar('LossNontgt/Val', val_loss_nontgt, epoch)

        scheduler.step(val_loss)
        last_lr = scheduler.get_last_lr()[0]
        tbsw.add_scalar(f'{scheduler.__class__.__name__} lr', last_lr, epoch)

        print(f'Train loss: {train_loss:.6f}, loss_tgt: {train_loss_tgt:.6f}, loss_nontgt: {train_loss_nontgt:.6f}')
        print(f'Val loss:   {val_loss:.6f}, loss_tgt: {val_loss_tgt:.6f}, loss_nontgt: {val_loss_nontgt:.6f}')
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


