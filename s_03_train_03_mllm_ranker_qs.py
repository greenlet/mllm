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

from mllm.config.exp_config import ExpConfig
from mllm.data.dsmsmarco import MsmDsLoader
from mllm.data.dswiki import WikiDsLoader
from mllm.model.mllm_ranker import MllmRanker
from mllm.model.mllm_encdec import MllmEncdec
from mllm.model.config import create_mllm_ranker_cfg, create_mllm_encdec_cfg
from mllm.tokenization.chunk_tokenizer import calc_max_inp_size, gen_all_tokens, ChunkTokenizer
from mllm.train.args import ArgsTrain
from mllm.train.utils import gen_train_subdir, find_create_train_path
from mllm.utils.utils import gen_dt_str
from transformers import GPT2Tokenizer, PreTrainedTokenizer


class ArgsTrainRankerQs(ArgsTrain):
    embs_chunk_size: Optional[int] = Field(
        100,
        required=False,
        description='Number of tokens in chunk converted to a single embedding vector.',
        cli=('--embs-chunk-size',),
    )


class RankProbLoss(nn.Module):
    def __init__(self, target_weight: float = 0.5):
        super().__init__()
        self.target_weight = target_weight
        self.register_buffer('prob_cap', torch.scalar_tensor(1e-6))

    def forward(self, prob_pred: list[torch.Tensor], mask_gt: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_tgt = torch.scalar_tensor(0, dtype=torch.float32, device=prob_pred[0].device)
        loss_nontgt = torch.scalar_tensor(0, dtype=torch.float32, device=prob_pred[0].device)
        n_batch = len(prob_pred)
        for i in range(n_batch):
            prob_tgt = torch.masked_select(prob_pred[i], mask_gt[i])
            prob_nontgt = 1 - torch.masked_select(prob_pred[i], ~mask_gt[i])
            prob_tgt = torch.maximum(prob_tgt, self.prob_cap)
            prob_nontgt = torch.maximum(prob_nontgt, self.prob_cap)
            loss_tgt += -torch.mean(torch.log(prob_tgt))
            loss_nontgt += -torch.mean(torch.log(prob_nontgt))

        loss_tgt /= n_batch
        loss_nontgt /= n_batch
        loss = self.target_weight * loss_tgt + (1 - self.target_weight) * loss_nontgt
        return loss, loss_tgt, loss_nontgt

    def forward_1(self, prob_pred: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prob_pred = prob_pred.squeeze()
        prob_tgt = torch.masked_select(prob_pred, mask_gt)
        prob_nontgt = 1 - torch.masked_select(prob_pred, ~mask_gt)
        prob_tgt = torch.maximum(prob_tgt, self.prob_cap)
        prob_nontgt = torch.maximum(prob_nontgt, self.prob_cap)
        loss_tgt = -torch.mean(torch.log(prob_tgt))
        loss_nontgt = -torch.mean(torch.log(prob_nontgt))
        loss = self.target_weight * loss_tgt + (1 - self.target_weight) * loss_nontgt
        return loss, loss_tgt, loss_nontgt


def main(args: ArgsTrainRankerQs) -> int:
    print(args)

    exp_cfg = ExpConfig()

    device = torch.device(args.device)

    train_path = find_create_train_path(
        args.train_root_path, 'ranker', args.ds_dir_path.name, args.train_subdir)
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
    ch_tkz = ChunkTokenizer(tok_dict, tokenizer, n_emb_tokens=args.embs_chunk_size, fixed_size=True)
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind
    ds_loader = MsmDsLoader(
        ds_path=args.ds_dir_path, emb_chunk_size=args.embs_chunk_size, docs_batch_size=args.docs_batch_size,
        max_chunks_per_doc=args.max_chunks_per_doc, pad_tok=pad_tok, qbeg_tok=qbeg_tok, qend_tok=qend_tok, ch_tkz=ch_tkz,
        device=device,
    )

    print(f'Creating model with vocab size = {len(tokenizer)}')

    torch.autograd.set_detect_anomaly(True)

    model_cfg = create_mllm_ranker_cfg(
        n_vocab=len(tokenizer), inp_len=args.embs_chunk_size, d_word_wec=256,
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
            n_vocab=len(tokenizer), d_word_wec=256, inp_len=args.embs_chunk_size,
            enc_n_layers=1, dec_n_layers=1,
            n_heads=8, d_model=256, d_inner=1024,
            pad_idx=pad_tok, dropout_rate=0.1, enc_with_emb_mat=True,
        )
        model_encdec = MllmEncdec(model_encdec_cfg).to(device)
        model_encdec.load_state_dict(pretrained_checkpoint['model'], strict=False)
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
    n_batches_train = calc_batches(ds_loader.n_qs_train)
    n_batches_val = calc_batches(ds_loader.n_qs_val)
    loss_fn = RankProbLoss()
    i_train, i_val = 0, 0
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss, train_loss_tgt, train_loss_nontgt = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = ds_loader.get_batch(i_train, train=True)
            docs_chunks, qs_chunks = batch.gen_tensors()

            optimizer.zero_grad()
            out_rank, target_mask = model.run_qs(docs_chunks, qs_chunks, batch.docs_off_len, batch.qs_off_len)
            loss, loss_tgt, loss_nontgt = loss_fn(out_rank, target_mask)
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
            batch = ds_loader.get_batch(i_val, train=False)
            docs_chunks, qs_chunks = batch.gen_tensors()

            out_rank, target_mask = model.run_qs(docs_chunks, qs_chunks, batch.docs_off_len, batch.qs_off_len)
            loss, loss_tgt, loss_nontgt = loss_fn(out_rank, target_mask)

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
    run_and_exit(ArgsTrainRankerQs, main, 'Train Mllm model.', exception_handler=rethrow)


