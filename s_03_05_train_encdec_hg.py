import shutil
from pathlib import Path
from typing import Union, Optional

import numpy as np
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch

import torch.utils.tensorboard as tb
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from tqdm import trange
from transformers import GPT2Tokenizer

from mllm.data.wiki.dswiki import WikiDsLoader
from mllm.exp.args import TOKENIZER_CFG_FNAME, ENCDEC_MODEL_CFG_FNAME
from mllm.model.encdec_hg import EncdecHg
from mllm.train.utils import find_create_train_path
from mllm.model.mllm_encdec import MllmEncdecLevel, encdec_embs_loss_cos
from mllm.config.model import create_mllm_encdec_cfg, TokenizerCfg, MllmEncdecCfg, gen_prefpostfix_level, EncdecHgCfg, \
    create_encdec_hg_cfg, copy_override_encdec_hg_cfg, gen_prefpostfix_hg
from mllm.tokenization.chunk_tokenizer import calc_max_inp_size, gen_all_tokens, tokenizer_from_config


class ArgsEncdecHgTrain(BaseModel):
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
    tokenizer_cfg_fpath: Path = Field(
        ...,
        required=True,
        description='Path to tokenizer config Yaml file.',
        cli=('--tokenizer-cfg-fpath',),
    )
    model_cfg_fpath: Path = Field(
        ...,
        required=True,
        description='Path to ranker model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    inp_len: int = Field(
        ...,
        required=True,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    docs_batch_size: int = Field(
        3,
        required=False,
        description='Documents batch size. Must be greater or equal than 2.',
        cli=('--docs-batch-size',),
    )
    max_chunks_per_doc: int = Field(
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
    pretrained_model_path: Optional[Path] = Field(
        None,
        required=False,
        description='Path to pretrained model weights.',
        cli=('--pretrained-model-path',),
    )
    emb_chunk_size: Optional[int] = Field(
        100,
        required=False,
        description='Number of tokens in chunk converted to a single embedding vector.',
        cli=('--embs-chunk-size',),
    )



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
def remove_tokens(chunks: torch.Tensor, mask_tok: int, rem_ratio: float = 0.15, rem_conseq_ratio: float = 0.3) -> torch.Tensor:
    res = chunks.clone()
    rv = np.random.rand()
    if rv < 1 / 3:
        p = rem_ratio
        mask = torch.distributions.Bernoulli(probs=p).sample(chunks.size()).to(chunks.device)
        res[mask.bool()] = mask_tok
    elif rv < 2 / 3:
        n = chunks.shape[-1]
        n_rem = int(n * rem_conseq_ratio)
        n_rem = np.random.randint(1, n_rem)
        i = np.random.randint(n - n_rem + 1)
        res[:, i:i + n_rem] = mask_tok
    return res


def main(args: ArgsEncdecHgTrain) -> int:
    print(args)

    device = torch.device(args.device)

    tkz_cfg = parse_yaml_file_as(TokenizerCfg, args.tokenizer_cfg_fpath)
    model_cfg = parse_yaml_file_as(EncdecHgCfg, args.model_cfg_fpath)
    model_cfg = copy_override_encdec_hg_cfg(model_cfg, inp_len=args.inp_len)

    prefix, suffix = gen_prefpostfix_hg(model_cfg)
    suffix = f'{args.ds_dir_path.parent.name}-{args.ds_dir_path.name}-{suffix}'
    train_path = find_create_train_path(args.train_root_path, prefix, suffix, args.train_subdir)
    print(f'train_path: {train_path}')

    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    checkpoint = None
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists(),\
            (f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
             f'but file {last_checkpoint_path} does not exits.')
###
    if last_checkpoint_path.exists():
        print(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        print(f'Checkpoint with keys {list(checkpoint.keys())} loaded')
        chkpt_tkz_cfg = parse_yaml_file_as(TokenizerCfg, train_path / TOKENIZER_CFG_FNAME)
        chkpt_model_cfg = parse_yaml_file_as(MllmEncdecCfg, train_path / ENCDEC_MODEL_CFG_FNAME)
        assert tkz_cfg == chkpt_tkz_cfg, f'{args.tokenizer_cfg_fpath} != {chkpt_tkz_cfg}'
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / TOKENIZER_CFG_FNAME, tkz_cfg)
        to_yaml_file(train_path / ENCDEC_MODEL_CFG_FNAME, model_cfg)

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
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        ds_loader.shuffle(train=True)
        ds_loader.shuffle(train=False)
    elif args.pretrained_model_path and args.pretrained_model_path.name:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path)
        model_encdec_cfg_fpath = args.pretrained_model_path / ENCDEC_MODEL_CFG_FNAME
        model_encdec_cfg = parse_yaml_file_as(MllmEncdecCfg, model_encdec_cfg_fpath)
        model_encdec = MllmEncdecLevel(model_encdec_cfg, args.model_level).to(device)
        model_encdec.load_state_dict(pretrained_checkpoint['model'], strict=False)
        # if args.model_level == 0:
        #     assert model.vocab_encoder is not None and model_encdec.vocab_encoder is not None
        #     print(f'Load model weights for vocab_encoder:', list(model_encdec.vocab_encoder.state_dict().keys()))
        #     model.vocab_encoder.load_state_dict(model_encdec.vocab_encoder.state_dict())
        # print(f'Load model weights for encoder:', list(model_encdec.encoder.state_dict().keys()))
        # model.encoder.load_state_dict(model_encdec.encoder.state_dict())
        print(f'Load model weights:', list(model_encdec.state_dict().keys()))
        model.load_state_dict(model_encdec.state_dict(), strict=False)

    sched_wait_steps = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6, min_lr=1e-7)
    print(f'Scheduler {scheduler.__class__.__name__} lr: {scheduler.get_last_lr()[0]:0.10f}.')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    calc_batches = lambda n_docs: n_docs // args.docs_batch_size + (n_docs % args.docs_batch_size > 1)
    n_batches_train = calc_batches(ds_loader.n_docs_train)
    n_batches_val = calc_batches(ds_loader.n_docs_val)
    if model_cfg.with_vocab_decoder:
        # loss_fn = encdec_prob_loss_sigmoid
        # loss_fn = EncdecProbLossSigmoid(seq_len=inp_len, n_tokens=len(tokenizer), device=device)
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = encdec_prob_loss_softmax
    else:
        loss_fn = encdec_embs_loss_cos
    
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
            inp_tok_embs = None
            if not model_cfg.with_vocab_decoder and model.level == 0:
                inp_tok_embs = model.run_vocab_encoder(chunks)

            if not graph_written:
                tbsw.add_graph(model, docs_chunks, verbose=True, use_strict_trace=False)
                graph_written = True

            if model_cfg.with_vocab_decoder or model.level != 0:
                loss = loss_fn(out_logits, chunks)
            else:
                loss = loss_fn(out_logits, inp_tok_embs)
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
            inp_tok_embs = None
            if not model_cfg.with_vocab_decoder and model.level == 0:
                inp_tok_embs = model.run_vocab_encoder(chunks)

            if model_cfg.with_vocab_decoder or model.level != 0:
                loss = loss_fn(out_logits, chunks)
            else:
                loss = loss_fn(out_logits, inp_tok_embs)
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

        if epoch >= sched_wait_steps:
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


