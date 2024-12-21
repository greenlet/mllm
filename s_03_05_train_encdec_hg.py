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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.config.model import TokenizerCfg, EncdecHgCfg, copy_override_encdec_hg_cfg, gen_prefpostfix_encdec_hg, HgReductType, \
    HgEnhanceType, PosEncType
from mllm.exp.args import TOKENIZER_CFG_FNAME, ENCDEC_HG_MODEL_CFG_FNAME
from mllm.model.encdec_ranker_hg import EncdecHg
from mllm.model.losses import encdec_prob_loss_softmax
from mllm.tokenization.chunk_tokenizer import tokenizer_from_config
from mllm.train.utils import find_create_train_path, log_weights_grads_stats


class ArgsEncdecHgTrain(BaseModel):
    data_path: Path = Field(
        ...,
        required=True,
        description='Root data path. Must contain subpath `wikipedia/WIKI_DS_NAME` with Wikipedia dataset.',
        cli=('--data-path',),
    )
    wiki_ds_name: str = Field(
        '20200501.en',
        required=False,
        description='Wikipedia dataset name of the format YYYYMMDD.LANG, for example: 20220301.en',
        cli=('--wiki-ds-name',),
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
        description='Path to EncdecHg model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    inp_len: int = Field(
        ...,
        required=True,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    n_similar_layers: int = Field(
        ...,
        required=True,
        description='Number of consecutive similar attention layers for each level dedicated of increasing/decreasing input size.',
        cli=('--n-similar-layers',),
    )
    reduct_type: HgReductType = Field(
        HgReductType.Matmul,
        required=False,
        description=f'Encoder layer reduct type. Can have values: {list(x.value for x in HgReductType)}',
        cli=('--reduct-type',),
    )
    enhance_type: HgEnhanceType = Field(
        HgEnhanceType.Matmul,
        required=False,
        description=f'Decoder layer enhance type. Can have values: {list(x.value for x in HgEnhanceType)}',
        cli=('--enhance-type',),
    )
    pos_enc_type: PosEncType = Field(
        PosEncType.Num,
        required=False,
        description=
        f'Positional encoder type. Can have values: {list(x.value for x in PosEncType)}. {PosEncType.Num} - '
        f'trigonometric numerical values generated. {PosEncType.Emb} - learned embeddings.',
        cli=('--pos-enc-type',),
    )
    docs_batch_size: int = Field(
        3,
        required=False,
        description='Documents batch size. Must be greater or equal than 2.',
        cli=('--docs-batch-size',),
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



def main(args: ArgsEncdecHgTrain) -> int:
    print(args)

    device = torch.device(args.device)

    tkz_cfg = parse_yaml_file_as(TokenizerCfg, args.tokenizer_cfg_fpath)
    model_cfg = parse_yaml_file_as(EncdecHgCfg, args.model_cfg_fpath)
    model_cfg = copy_override_encdec_hg_cfg(
        model_cfg, inp_len=args.inp_len, n_similar_layers=args.n_similar_layers, reduct_type=args.reduct_type,
        enhance_type=args.enhance_type, pos_enc_type=args.pos_enc_type,
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
    ds = load_dataset(wiki_ds_subdir, args.wiki_ds_name, beam_runner='DirectRunner', cache_dir=str(args.data_path))
    ds = ds['train']
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

    def get_batch_tokens(doc_inds: list[int]) -> torch.Tensor:
        docs_toks = np.full((len(doc_inds), args.inp_len), pad_tok)
        for i, doc_ind in enumerate(doc_inds):
            doc = ds[doc_ind]
            title, text = doc['title'], doc['text']
            doc_txt = f'{title} {text}'
            doc_toks = tkz(doc_txt)['input_ids']
            n_toks = len(doc_toks)
            if n_toks > args.inp_len:
                i_off = np.random.randint(n_toks - args.inp_len + 1)
                doc_toks = doc_toks[i_off:i_off + args.inp_len]
            docs_toks[i, :len(doc_toks)] = doc_toks
        docs_toks_t = torch.from_numpy(docs_toks).to(device)
        return docs_toks_t

    def get_batch(inds: list[int], i_batch: int) -> tuple[torch.Tensor, int]:
        i1 = i_batch * args.docs_batch_size
        i2 = i1 + args.docs_batch_size
        batch_inds = inds[i1:i2]
        rest_batch_size = args.docs_batch_size - len(batch_inds)
        if rest_batch_size > 0:
            batch_inds = batch_inds + inds[:rest_batch_size * args.docs_batch_size]
        if i2 >= len(batch_inds):
            i_batch = 0
            np.random.shuffle(inds)
        batch_toks = get_batch_tokens(batch_inds)
        return batch_toks, i_batch

    # loss_fn = encdec_prob_loss_sigmoid
    # loss_fn = EncdecProbLossSigmoid(seq_len=inp_len, n_tokens=len(tokenizer), device=device)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = encdec_prob_loss_softmax

    print(model)

    i_train, i_val = 0, 0
    loss_gt, loss_nongt = None, None
    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss, train_loss_gt, train_loss_nongt = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            tokens_inp, i_train = get_batch(doc_inds_train, i_train)
            # tokens_inp = remove_tokens(tokens_inp, mask_tok, input_zeros_ratio)

            optimizer.zero_grad()

            out_logits = model(tokens_inp)
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
        val_loss, val_loss_gt, val_loss_nongt = 0, 0, 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            tokens_inp, i_val = get_batch(doc_inds_train, i_val)

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

