import shutil
from typing import Union

import numpy as np
from pydantic_cli import run_and_exit
import torch

import torch.utils.tensorboard as tb
from pydantic_yaml import parse_yaml_file_as
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from tqdm import trange
from transformers import GPT2Tokenizer

from mllm.data.wiki.dswiki import WikiDsLoader
from mllm.exp.args import ArgsTokensChunksTrain, TOKENIZER_CFG_FNAME, ENCDEC_MODEL_CFG_FNAME
from mllm.train.utils import find_create_train_path
from mllm.model.mllm_encdec import MllmEncdecLevel
from mllm.config.model import create_mllm_encdec_cfg, TokenizerCfg, MllmEncdecCfg, gen_prefpostfix
from mllm.tokenization.chunk_tokenizer import calc_max_inp_size, gen_all_tokens, tokenizer_from_config


def encdec_prob_loss_softmax(logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> torch.Tensor:
    tokens_gt = tokens_gt.to(torch.int64).unsqueeze(-1)
    probs_pred = torch.softmax(logits_pred, dim=-1)
    probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt)
    loss = -torch.mean(torch.log(probs_gt))
    return loss


def encdec_prob_loss_sigmoid(logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = logits_pred.device
    tokens_gt = tokens_gt.to(torch.int64).unsqueeze(-1)
    probs_pred = torch.sigmoid(logits_pred)
    prob_cap = torch.tensor(1e-6, dtype=torch.float32, device=device)
    probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt)
    probs_gt = torch.maximum(probs_gt, prob_cap)
    loss_gt = -torch.mean(torch.log(probs_gt))
    loss_nongt = torch.tensor(0, dtype=torch.float32, device=device)
    for i in range(probs_pred.shape[0]):
        mask = torch.full((logits_pred.shape[-2], logits_pred.shape[-1],), True, device=device)
        mask = mask.scatter(1, tokens_gt[i], 0)
        probs_nongt = 1 - probs_pred[i][mask]
        probs_nongt = torch.maximum(probs_nongt, prob_cap)
        loss_nongt += -torch.mean(torch.log(probs_nongt))
    loss_nongt = loss_nongt / logits_pred.shape[0]
    loss = loss_gt + loss_nongt
    return loss_gt, loss_nongt, loss


class EncdecProbLossSigmoid(nn.Module):
    def __init__(self, seq_len: int, n_tokens: int, device: torch.device):
        super().__init__()
        mask = torch.full((seq_len, n_tokens), True, device=device)
        self.register_buffer('mask', mask)

    def forward(self, logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> torch.Tensor:
        tokens_gt = tokens_gt.to(torch.int64).unsqueeze(-1)
        probs_pred = torch.sigmoid(logits_pred)
        probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt)
        loss_gt = -torch.mean(torch.log(probs_gt))
        loss_nongt = 0
        for i in range(probs_pred.shape[0]):
            self.mask.scatter_(1, tokens_gt[i], 0)
            loss_nongt += -torch.mean(torch.log(1 - probs_pred[i][self.mask]))
            self.mask.scatter_(1, tokens_gt[i], 1)
        loss_nongt = loss_nongt / logits_pred.shape[0]
        loss = loss_gt + loss_nongt
        return loss


def concat_tokens(*chunks: torch.Tensor, shuffle: bool = True) ->torch.Tensor:
    if shuffle:
        chunks = list(chunks)
        np.random.shuffle(chunks)
    return torch.concat(chunks, dim=0)


# chunks: input token chunks of the shape [n_docs x n_tokens_per_doc]
def remove_tokens(chunks: torch.Tensor, mask_tok: int, rem_ratio: float = 0.15, rem_conseq_ratio: float = 0.2) -> torch.Tensor:
    res = chunks.clone()
    if torch.rand((1,)) > 1 / 2:
        p = rem_ratio
        mask = torch.distributions.Bernoulli(probs=p).sample(chunks.size()).to(chunks.device)
        res[mask.bool()] = mask_tok
    else:
        n = chunks.shape[-1]
        n_rem = int(n * rem_conseq_ratio)
        i = np.random.randint(n - n_rem + 1)
        res[:, i:i + n_rem] = mask_tok
    return res


def main(args: ArgsTokensChunksTrain) -> int:
    print(args)

    device = torch.device(args.device)

    tkz_cfg = parse_yaml_file_as(TokenizerCfg, args.tokenizer_cfg_fpath)
    model_cfg = parse_yaml_file_as(MllmEncdecCfg, args.model_cfg_fpath)
    enc_cfg, dec_cfg = model_cfg.encoders[args.model_level], model_cfg.decoders[args.model_level]
    enc_cfg.n_layers = args.n_enc_layers if args.n_enc_layers > 0 else enc_cfg.n_layers
    dec_cfg.n_layers = args.n_dec_layers if args.n_dec_layers > 0 else dec_cfg.n_layers

    prefix, suffix = gen_prefpostfix(model_cfg, args.model_level)
    suffix = f'{args.ds_dir_path.parent.name}-{args.ds_dir_path.name}-{suffix}'
    train_path = find_create_train_path(args.train_root_path, prefix, suffix, args.train_subdir)
    print(f'train_path: {train_path}')

    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    checkpoint = None
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists(),\
            (f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
             f'but file {last_checkpoint_path} does not exits.')

    if last_checkpoint_path.exists():
        print(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        print(f'Checkpoint with keys {list(checkpoint.keys())} loaded')
        chkpt_tkz_cfg = parse_yaml_file_as(TokenizerCfg, train_path / TOKENIZER_CFG_FNAME)
        chkpt_model_cfg = parse_yaml_file_as(MllmEncdecCfg, train_path / ENCDEC_MODEL_CFG_FNAME)
        assert tkz_cfg == chkpt_tkz_cfg, f'{args.tokenizer_cfg_fpath} != {chkpt_tkz_cfg}'
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        shutil.copy(args.tokenizer_cfg_fpath, train_path / TOKENIZER_CFG_FNAME)
        shutil.copy(args.model_cfg_fpath, train_path / ENCDEC_MODEL_CFG_FNAME)

    tokenizer = tokenizer_from_config(tkz_cfg)

    tok_dict = tkz_cfg.custom_tokens
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind
    mask_tok = tok_dict['mask'].ind
    ds_loader = WikiDsLoader(
        ds_path=args.ds_dir_path, docs_batch_size=args.docs_batch_size, max_chunks_per_doc=args.max_chunks_per_doc,
        pad_tok=pad_tok, qbeg_tok=qbeg_tok, qend_tok=qend_tok, device=device, n_total=0,
    )

    inp_len = ds_loader.emb_chunk_size if ds_loader.fixed_size else calc_max_inp_size(ds_loader.emb_chunk_size)
    print(f'Creating model with vocab size = {len(tokenizer)}')

    torch.autograd.set_detect_anomaly(True)

    input_zeros_ratio = 0.3
    print(model_cfg)
    model = MllmEncdecLevel(model_cfg, args.model_level).to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    last_epoch, val_loss_min = -1, None
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        ds_loader.shuffle(train=True)
        ds_loader.shuffle(train=False)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, threshold=1e-6, min_lr=1e-7)
    print(f'Scheduler {scheduler.__class__.__name__} lr: {scheduler.get_last_lr()[0]:0.10f}.')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    calc_batches = lambda n_docs: n_docs // args.docs_batch_size + (n_docs % args.docs_batch_size > 1)
    n_batches_train = calc_batches(ds_loader.n_docs_train)
    n_batches_val = calc_batches(ds_loader.n_docs_val)
    loss_fn = encdec_prob_loss_softmax
    # loss_fn = encdec_prob_loss_sigmoid
    # loss_fn = EncdecProbLossSigmoid(seq_len=inp_len, n_tokens=len(tokenizer), device=device)
    # loss_fn = nn.CrossEntropyLoss()
    graph_written = True
    i_train, i_val = 0, 0
    loss_gt, loss_nongt = None, None
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss, train_loss_gt, train_loss_nongt = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = ds_loader.get_batch(i_train, train=True)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()

            chunks = concat_tokens(docs_chunks, target_chunks)
            chunks_inp = remove_tokens(chunks, mask_tok, input_zeros_ratio)

            optimizer.zero_grad()

            out_logits = model(chunks_inp)
            if not graph_written:
                tbsw.add_graph(model, docs_chunks, verbose=True, use_strict_trace=False)
                graph_written = True

            loss = loss_fn(out_logits, chunks)
            if type(loss) == tuple:
                loss_gt, loss_nongt, loss = loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if loss_gt is not None:
                train_loss_gt += loss_gt.item()
                train_loss_nongt += loss_nongt.item()

            i_train += 1
            if i_train == n_batches_train:
                ds_loader.shuffle(train=True)
                i_train %= n_batches_train

            # if i_train == 2:
            #     import sys
            #     sys.exit()

            s = f'Train. loss: {loss.item():.6f}'
            if loss_gt is not None:
                s += f'. loss_gt: {loss_gt.item():.6f}. loss_nongt: {loss_nongt.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)
        if loss_gt is not None:
            train_loss_gt /= args.train_epoch_steps
            train_loss_nongt /= args.train_epoch_steps
            tbsw.add_scalar('LossGt/Train', train_loss_gt, epoch)
            tbsw.add_scalar('LossNongt/Train', train_loss_nongt, epoch)

        model.eval()
        val_loss, val_loss_gt, val_loss_nongt = 0, 0, 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = ds_loader.get_batch(i_val, train=False)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()
            
            chunks = concat_tokens(docs_chunks, target_chunks)
            out_logits = model(chunks)

            loss = loss_fn(out_logits, chunks)
            if type(loss) == tuple:
                loss_gt, loss_nongt, loss = loss
            val_loss += loss.item()
            if loss_gt is not None:
                val_loss_gt += loss_gt.item()
                val_loss_nongt += loss_nongt.item()

            i_val += 1
            if i_val == n_batches_val:
                ds_loader.shuffle(train=False)
                i_val %= n_batches_val

            s = f'Val. loss: {loss.item():.6f}'
            if loss_gt is not None:
                s += f'. loss_gt: {loss_gt.item():.6f}. loss_nongt: {loss_nongt.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        val_loss /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)
        if loss_gt is not None:
            val_loss_gt /= args.val_epoch_steps
            val_loss_nongt /= args.val_epoch_steps
            tbsw.add_scalar('LossGt/Val', val_loss_gt, epoch)
            tbsw.add_scalar('LossNongt/Val', val_loss_nongt, epoch)

        scheduler.step(val_loss)
        last_lr = scheduler.get_last_lr()[0]
        tbsw.add_scalar(f'{scheduler.__class__.__name__} lr', last_lr, epoch)

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
    run_and_exit(ArgsTokensChunksTrain, main, 'Train Mllm model.', exception_handler=rethrow)


