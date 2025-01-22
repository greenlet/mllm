import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.tensorboard as tb
from datasets import load_dataset
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.config.model import TokenizerCfg, EncdecHgCfg, copy_override_encdec_hg_cfg, gen_prefpostfix_encdec_hg, HgReductType, \
    HgEnhanceType, PosEncType
from mllm.exp.args import TOKENIZER_CFG_FNAME, ENCDEC_HG_MODEL_CFG_FNAME
from mllm.model.encdec_ranker_hg import EncdecHg
from mllm.model.losses import encdec_prob_loss_softmax
from mllm.tokenization.chunk_tokenizer import tokenizer_from_config
from mllm.train.utils import find_create_train_path, log_weights_grads_stats, remove_tokens


class ArgsEncdecHgTrain(BaseModel):
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
    tokenizer_cfg_fpath: Path = Field(
        ...,
        description='Path to tokenizer config Yaml file.',
        cli=('--tokenizer-cfg-fpath',),
    )
    model_cfg_fpath: Path = Field(
        ...,
        description='Path to EncdecHg model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    inp_len: int = Field(
        ...,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    n_similar_layers: int = Field(
        ...,
        description='Number of consecutive similar attention layers for each level dedicated of increasing/decreasing input size.',
        cli=('--n-similar-layers',),
    )
    reduct_type: HgReductType = Field(
        HgReductType.Matmul,
        description=f'Encoder layer reduct type. Can have values: {list(x.value for x in HgReductType)}',
        cli=('--reduct-type',),
    )
    enhance_type: HgEnhanceType = Field(
        HgEnhanceType.Matmul,
        description=f'Decoder layer enhance type. Can have values: {list(x.value for x in HgEnhanceType)}',
        cli=('--enhance-type',),
    )
    pos_enc_type: PosEncType = Field(
        PosEncType.Num,
        description=
        f'Positional encoder type. Can have values: {list(x.value for x in PosEncType)}. {PosEncType.Num} - '
        f'trigonometric numerical values generated. {PosEncType.Emb} - learned embeddings.',
        cli=('--pos-enc-type',),
    )
    dropout_rate: float = Field(
        0.0,
        required=False,
        description='Dropout rate for all layers (vocab encoder, encoder, decoder).',
        cli=('--dropout-rate',),
    )
    dec_n_layers: int = Field(
        0,
        description='Decoder number of layers.',
        cli=('--dec-n-layers',),
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


class RankMaskPadLoss(nn.Module):
    pad_tok: int
    pad_weight: float
    nonpad_weight: float
    # prob_cap: float

    def __init__(self, pad_tok: int, pad_weight: float = 0.01, prob_cap: float = 1e-6):
        super().__init__()
        pad_weight = min(max(pad_weight, 0), 1)
        assert 0 <= prob_cap <= 1, f'prob_cap (={prob_cap}) must pertain to [0, 1] interval'
        self.pad_tok = pad_tok
        self.pad_weight = pad_weight
        self.nonpad_weight = 1 - pad_weight
        self.register_buffer('prob_cap', torch.scalar_tensor(prob_cap))

    # logits_pred: (batch_size, inp_len, vocab_size)
    # tokens_gt: (batch_size, inp_len)
    def forward(self, logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> torch.Tensor:
        # tokens_gt: (batch_size, inp_len, 1)
        tokens_gt = tokens_gt.to(torch.int64).unsqueeze(-1)
        # mask_pad: (batch_size, inp_len, 1)
        mask_pad = tokens_gt == self.pad_tok
        # mask_npad: (batch_size, inp_len, 1)
        mask_npad = ~mask_pad

        # probs_pred: (batch_size, inp_len, vocab_size)
        probs_pred = torch.softmax(logits_pred, dim=-1)
        # probs_gt: (batch_size, inp_len, 1)
        probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt)
        # probs_gt = torch.maximum(probs_gt, self.prob_cap)

        # probs_gt_pad: (n_pad_tokens, )
        # probs_gt_npad: (n_nonpad_tokens, )
        # n_pad_tokens + n_nonpad_tokens = batch_size * inp_len
        probs_gt_pad, probs_gt_npad = probs_gt[mask_pad], probs_gt[mask_npad]

        # loss_pad: (1,)
        # loss_npad: (1,)
        loss_pad = torch.zeros((1,), dtype=torch.float32, device=probs_gt.device)
        loss_npad = loss_pad
        if probs_gt_pad.size()[0] > 0:
            loss_pad = -torch.mean(torch.log(probs_gt_pad))
        if probs_gt_npad.size()[0] > 0:
            loss_npad = -torch.mean(torch.log(probs_gt_npad))
        # loss: (1,)
        loss = loss_npad * self.nonpad_weight + loss_pad * self.pad_weight
        return loss


def mask_random_tokens(chunks: torch.Tensor, mask_tok: int, rem_ratio: float = 0.15, rem_conseq_ratio: float = 0.3) -> torch.Tensor:
    res = chunks.clone()
    rv = np.random.rand()
    if rv < 1 / 5:
        p = rem_ratio
        mask = torch.distributions.Bernoulli(probs=p).sample(chunks.size()).to(chunks.device)
        res[mask.bool()] = mask_tok
    elif rv < 2 / 5:
        n = chunks.shape[-1]
        n_rem = int(n * rem_conseq_ratio)
        n_rem = np.random.randint(1, n_rem)
        i = np.random.randint(n - n_rem + 1)
        res[:, i:i + n_rem] = mask_tok
    return res


NEWLINE_PAT = re.compile(r'[\n\r]+', re.M)
STR_DELIM_PAT = re.compile(r'\s+')


def mask_random_words(
        s: str, mask_tok_str: str, rem_freq: float = 0.33, rem_prob: float = 0.15,
        rem_conseq_freq: float = 0.33, rem_conseq_prob: float = 0.2, rem_conseq_max_len: int = 20,
        rem_conseq_max_times: int = 5,
        ) -> Optional[str]:
    rv = np.random.rand()
    if rv < 1 - (rem_freq + rem_conseq_freq):
        return
    lines = NEWLINE_PAT.split(s)
    res = []
    n_total = 0
    for line in lines:
        if not line:
            continue
        words = STR_DELIM_PAT.split(line)
        words = filter(None, words)
        words = list(words)
        if not words:
            continue
        res.append(words)
        n_total += len(words)

    if n_total < 5:
        return

    if rv < 1 - rem_conseq_freq:
        mask = np.random.rand(n_total) <= rem_prob
    else:
        rem_conseq_times = np.random.randint(1, rem_conseq_max_times + 1)
        rem_interval = n_total // rem_conseq_times
        off = 0
        mask = np.full(n_total, False, dtype=bool)
        while off < n_total:
            n_rem = int(n_total * rem_conseq_prob)
            n_rem = np.random.randint(2, max(n_rem, 2) + 1)
            n_rem = min(n_rem, rem_conseq_max_len)
            i = np.random.randint(off, off + rem_interval)
            i1 = max(i - n_rem // 2, 0)
            i2 = min(i1 + n_rem, n_total - 1)
            if i1 < i2:
                mask[i1:i2] = True
            off = max(off + rem_interval, i2 + int(n_rem * 1.5))

    im = 0
    for words in res:
        for iw in range(len(words)):
            if mask[im]:
                words[iw] = mask_tok_str
            im += 1

    return '\n'.join([' '.join(words) for words in res])


def main(args: ArgsEncdecHgTrain) -> int:
    print(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    device = torch.device(args.device)

    tkz_cfg = parse_yaml_file_as(TokenizerCfg, args.tokenizer_cfg_fpath)
    model_cfg = parse_yaml_file_as(EncdecHgCfg, args.model_cfg_fpath)
    model_cfg = copy_override_encdec_hg_cfg(
        model_cfg, inp_len=args.inp_len, n_similar_layers=args.n_similar_layers, reduct_type=args.reduct_type,
        enhance_type=args.enhance_type, pos_enc_type=args.pos_enc_type, dropout_rate=args.dropout_rate,
        dec_n_layers=args.dec_n_layers,
    )

    prefix, suffix = gen_prefpostfix_encdec_hg(model_cfg)
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
        chkpt_model_cfg = parse_yaml_file_as(EncdecHgCfg, train_path / ENCDEC_HG_MODEL_CFG_FNAME)
        assert tkz_cfg == chkpt_tkz_cfg, f'{args.tokenizer_cfg_fpath} != {chkpt_tkz_cfg}'
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / TOKENIZER_CFG_FNAME, tkz_cfg)
        to_yaml_file(train_path / ENCDEC_HG_MODEL_CFG_FNAME, model_cfg)

    tkz = tokenizer_from_config(tkz_cfg)
    tok_dict = tkz_cfg.custom_tokens
    pad_tok, mask_tok = tok_dict['pad'].ind, tok_dict['mask'].ind
    print(f'Loading Wikipedia dataset: {args.wiki_ds_name}')
    wiki_ds_subdir = 'wikipedia'
    dss = load_dataset(wiki_ds_subdir, args.wiki_ds_name, beam_runner='DirectRunner', cache_dir=str(args.data_path))
    ds = dss['train']
    n_docs = len(ds)
    print(f'Wikipedia {args.wiki_ds_name} docs: {n_docs}')

    doc_inds = list(range(n_docs))
    val_ratio = 0.05
    n_docs_val = int(n_docs * val_ratio)
    n_docs_train = n_docs - n_docs_val
    doc_inds_train, doc_inds_val = doc_inds[:n_docs_train], doc_inds[n_docs_train:]

    input_zeros_ratio = 0.3
    print(model_cfg)
    model = EncdecHg(model_cfg).to(device)

    if args.pretrained_model_path and (args.pretrained_model_path / 'best.pth').exists() and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path)
        model_encdec_cfg_fpath = args.pretrained_model_path / ENCDEC_HG_MODEL_CFG_FNAME
        model_encdec_cfg = parse_yaml_file_as(EncdecHgCfg, model_encdec_cfg_fpath)
        model_cfg.enc_pyr.temperature = model_encdec_cfg.enc_pyr.temperature
        model_encdec = EncdecHg(model_encdec_cfg).to(device)
        model_encdec.load_state_dict(pretrained_checkpoint['model'], strict=False)
        print(f'Load model weights for enc_pyr:', list(model_encdec.enc_pyr.state_dict().keys()))
        model.load_state_dict(model_encdec.state_dict(), strict=False)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    last_epoch, val_loss_min = -1, None
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        np.random.shuffle(doc_inds_train)
        np.random.shuffle(doc_inds_val)

    sched_wait_steps = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6, min_lr=1e-7)
    print(f'Scheduler {scheduler.__class__.__name__} lr: {scheduler.get_last_lr()[0]:0.10f}.')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    def get_batch_tokens(doc_inds: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        docs_toks = np.full((len(doc_inds), args.inp_len), pad_tok)
        docs_toks_aug = np.full((len(doc_inds), args.inp_len), pad_tok)
        for i, doc_ind in enumerate(doc_inds):
            doc = ds[doc_ind]
            title, text = doc['title'], doc['text']
            if np.random.rand() < 1 / 4:
                doc_txt: str = title
            else:
                doc_txt: str = text
            # doc_txt = f'{title} {text}'
            # doc_txt = text
            doc_toks = tkz(doc_txt)['input_ids']
            n_toks = len(doc_toks)
            if n_toks > args.inp_len:
                i_off = np.random.randint(n_toks - args.inp_len + 1)
                doc_toks = doc_toks[i_off:i_off + args.inp_len]
            docs_toks[i, :len(doc_toks)] = doc_toks

            doc_txt_aug = mask_random_words(doc_txt, tok_dict['mask'].repr)
            if doc_txt_aug is None:
                doc_toks_aug = doc_toks
            else:
                doc_toks_aug = tkz(doc_txt_aug)['input_ids']
                n_toks_aug = len(doc_toks_aug)
                if n_toks_aug > args.inp_len:
                    i_off = np.random.randint(n_toks_aug - args.inp_len + 1)
                    doc_toks_aug = doc_toks_aug[i_off:i_off + args.inp_len]
            docs_toks_aug[i, :len(doc_toks_aug)] = doc_toks_aug

        docs_toks_t = torch.from_numpy(docs_toks).to(device)
        docs_toks_aug_t = torch.from_numpy(docs_toks_aug).to(device)
        return docs_toks_t, docs_toks_aug_t

    def get_batch(inds: list[int], i_batch: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        i1 = i_batch * args.docs_batch_size
        i2 = i1 + args.docs_batch_size
        batch_inds = inds[i1:i2]
        rest_batch_size = args.docs_batch_size - len(batch_inds)
        if rest_batch_size > 0:
            batch_inds = batch_inds + inds[:rest_batch_size * args.docs_batch_size]
        if i2 >= len(batch_inds):
            i_batch = 0
            np.random.shuffle(inds)
        batch_toks, batch_toks_aug = get_batch_tokens(batch_inds)
        return batch_toks, batch_toks_aug, i_batch

    # loss_fn = encdec_prob_loss_sigmoid
    # loss_fn = EncdecProbLossSigmoid(seq_len=inp_len, n_tokens=len(tokenizer), device=device)
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = encdec_prob_loss_softmax
    loss_fn = RankMaskPadLoss(pad_tok=pad_tok, pad_weight=0.1)

    print(model)

    i_train, i_val = 0, 0
    loss_gt, loss_nongt = None, None
    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
    if prev_train_steps > 0:
        grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss, train_loss_gt, train_loss_nongt = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            tokens_inp, tokens_inp_aug, i_train = get_batch(doc_inds_train, i_train)
            # tokens_inp_aug = mask_random_tokens(tokens_inp, mask_tok, input_zeros_ratio)
            # tokens_inp_aug = tokens_inp

            optimizer.zero_grad()

            out_logits = model(tokens_inp_aug)
            loss = loss_fn(out_logits, tokens_inp)
            if type(loss) == tuple:
                loss_gt, loss_nongt, loss = loss

            loss.backward()
            # Gradients must be available after loss.backward()
            if grad_log_ind % grad_log_interval == 0:
                log_weights_grads_stats(grad_log_step, model, tbsw)
                grad_log_step += 1
            grad_log_ind += 1

            optimizer.step()
            train_loss += loss.item()
            if loss_gt is not None:
                train_loss_gt += loss_gt.item()
                train_loss_nongt += loss_nongt.item()

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
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        val_loss, val_loss_gt, val_loss_nongt = 0, 0, 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            tokens_inp, _, i_val = get_batch(doc_inds_train, i_val)

            out_logits = model(tokens_inp)
            loss = loss_fn(out_logits, tokens_inp)
            if type(loss) == tuple:
                loss_gt, loss_nongt, loss = loss

            val_loss += loss.item()
            if loss_gt is not None:
                val_loss_gt += loss_gt.item()
                val_loss_nongt += loss_nongt.item()

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
    run_and_exit(ArgsEncdecHgTrain, main, 'Train Encoder-Decoder Hourglass model.', exception_handler=rethrow)

