import itertools
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.tensorboard as tb
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.config.model import GenmixBertCfg, copy_override_genmix_bert_cfg, gen_prefpostfix_genmix_bert
from mllm.exp.args import ENCDEC_BERT_MODEL_CFG_FNAME, ENCMIX_BERT_MODEL_CFG_FNAME
from mllm.model.encmix import EncmixBert, EncmixBertSep
from mllm.train.utils import find_create_train_path, log_weights_grads_stats, get_wiki_ds_batch_iterators, QnaQuesInp, \
    get_squadv2_df, split_df, gen_loss, get_squadv2_batch_iterator, get_squadv2_tensor_iterators
from transformers import AutoTokenizer


class ArgsGenmixBertTrain(BaseModel):
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
    inp_len: int = Field(
        ...,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    batch_size: int = Field(
        3,
        description='Documents batch size. Must be greater or equal than 2.',
        cli=('--batch-size',),
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


def main(args: ArgsGenmixBertTrain) -> int:
    print(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    device = torch.device(args.device)

    model_cfg = parse_yaml_file_as(GenmixBertCfg, args.model_cfg_fpath)
    model_cfg = copy_override_genmix_bert_cfg(
        model_cfg, inp_len=args.inp_len,
    )

    prefix, suffix = gen_prefpostfix_genmix_bert(model_cfg)
    train_path = find_create_train_path(args.train_root_path, prefix, suffix, args.train_subdir)
    print(f'train_path: {train_path}')
#
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
        chkpt_model_cfg = parse_yaml_file_as(EncmixBertCfg, train_path / ENCMIX_BERT_MODEL_CFG_FNAME)
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / ENCMIX_BERT_MODEL_CFG_FNAME, model_cfg)

    tkz = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name)

    print(model_cfg)
    if args.encmix_model_type == EncmixModelType.One:
        model = EncmixBert(model_cfg, tkz=tkz, device=device)
    elif args.encmix_model_type == EncmixModelType.Sep:
        model = EncmixBertSep(model_cfg, tkz=tkz, device=device)
    else:
        raise Exception(f'model type {args.encmix_model_type} is not supported.')

    if args.pretrained_model_path and (args.pretrained_model_path / 'best.pth').exists() and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path, map_location=device)
        # model.load_state_dict(pretrained_checkpoint['model'], strict=False)
        if isinstance(model, EncmixBert):
            print(list(pretrained_checkpoint['model'].keys()))
            model.load_state_dict(pretrained_checkpoint['model'], strict=True)
        else:
            print(list(pretrained_checkpoint['model'].keys()))
            prefix = 'enc_bert.bert_model.'
            prefix_len = len(prefix)
            model_chkpt = {}
            for k, v in pretrained_checkpoint['model'].items():
                if k.startswith(prefix):
                    k = k[prefix_len:]
                if k.startswith('dec_pyr.'):
                    continue
                model_chkpt[k] = v
            model.enc_model.load_state_dict(model_chkpt, strict=True)
            del pretrained_checkpoint
            del model_chkpt

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    last_epoch, val_loss_min, shuffle = -1, None, False
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        shuffle = True
        # for t in itertools.chain(checkpoint['model'].values(), checkpoint['optimizer'].values()):
        #     t.to('cpu')
        del checkpoint

    val_ratio = 0.05
    if args.train_ds_type == EncmixTrainDsType.Msk:
        train_batch_it, val_batch_it = get_wiki_ds_batch_iterators(
            wiki_ds_name=args.wiki_ds_name, data_path=args.data_path, inp_len=args.inp_len, docs_batch_size=args.batch_size,
            tkz=tkz, device=device, shuffle=shuffle, val_ratio=val_ratio,
        )
    else:
        train_batch_it, val_batch_it = get_squadv2_tensor_iterators(
            inp_len=args.inp_len, batch_size=args.batch_size, ques_inp=QnaQuesInp.Enc, exclude_empty_answers=True,
            tkz=tkz, device=device, val_ratio=val_ratio,
        )

    sched_wait_steps = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=13, threshold=1e-6, min_lr=1e-8)
    # lr = scheduler.get_last_lr()[0]
    lr = optimizer.param_groups[0]['lr']
    print(f'Scheduler {scheduler.__class__.__name__} lr: {lr:0.10f}.')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    loss_fn = gen_loss

    print(model)

    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
    if prev_train_steps > 0:
        grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss = 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            docs_toks_aug, plain_toks, docs_toks_tgt, _ = next(train_batch_it)

            optimizer.zero_grad()
            out_logits = model.run_chunks_plain_seq(
                chunk_toks=docs_toks_aug, plain_toks=plain_toks, target_toks=docs_toks_tgt,
            )
            loss = loss_fn(out_logits, docs_toks_tgt)

            loss.backward()
            # Gradients must be available after loss.backward()
            if grad_log_ind % grad_log_interval == 0:
                log_weights_grads_stats(grad_log_step, model, tbsw)
                grad_log_step += 1
            grad_log_ind += 1

            optimizer.step()
            train_loss += loss.item()

            # if i_train == 2:
            #     import sys
            #     sys.exit()

            s = f'Train. loss: {loss.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)

        model.eval()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_loss = 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            docs_toks_aug, plain_toks, docs_toks_tgt, _ = next(val_batch_it)

            with torch.no_grad():
                out_logits = model.run_chunks_plain_seq(
                    chunk_toks=docs_toks_aug, plain_toks=plain_toks, target_toks=docs_toks_tgt,
                )
                loss = loss_fn(out_logits, docs_toks_tgt)
            val_loss += loss.item()

            s = f'Val. loss: {loss.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        val_loss /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)

        if epoch >= sched_wait_steps:
            scheduler.step(val_loss)
        # last_lr = scheduler.get_last_lr()[0]
        last_lr = optimizer.param_groups[0]['lr']
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
    run_and_exit(ArgsGenmixBertTrain, main, 'Train EncmixBert model to predict masked input.', exception_handler=rethrow)

