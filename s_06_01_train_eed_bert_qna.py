import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard as tb
from datasets import load_dataset
from pydantic import Field, BaseModel
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from transformers import BertGenerationEncoder, BertGenerationDecoder, BertTokenizer

from mllm.config.model import EncdecBertCfg
from mllm.exp.args import ENCDEC_BERT_MODEL_CFG_FNAME
from mllm.model.embgen_bert import EncoderEmbDecoderModel
from mllm.model.encdec_ranker_hg import EncdecBert
from mllm.train.embgen_bert import QnaBatch, get_sq_batch_iterator, run_eed_model_on_batch, get_sq_df, split_df, QuesInp
from mllm.train.utils import find_create_train_path, log_weights_grads_stats
from mllm.utils.utils import reraise


class ArgsTrainEedBertQna(BaseModel):
    ds_dir_paths: list[Path] = Field(
        [],
        description='Qrels datasets directory paths. Supported datasets: Msmarco, Fever.'
                    'Naming convention: directory name must contain the name of dataset: msmarco, fever. Unknown datasets '
                    'will cause an error and exit.',
        cli=('--ds-dir-paths',),
    )
    train_root_path: Path = Field(
        ...,
        description='Path to train root directory. New train subdirectory will be created within each new run.',
        cli=('--train-root-path',),
    )
    pretrained_model_path: Optional[Path] = Field(
        None,
        description='Path to EncdecBert model train directory.',
        cli=('--pretrained-model-path',),
    )
    train_subdir: str = Field(
        '',
        description='Train subdirectory. Can have values: "last", "<subdirectory-name>". When set to "last", '
            'last subdirectory of TRAIN_ROOT_PATH containing training snapshot will be taken.',
        cli=('--train-subdir',)
    )
    inp_len: int = Field(
        ...,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    batch_size: int = Field(
        3,
        description='Question-answer batch size.',
        cli=('--batch-size',),
    )
    ques_inp: QuesInp = Field(
        ...,
        description=f'Question input type: {list(qi for qi in QuesInp)}.'
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


def main(args: ArgsTrainEedBertQna) -> int:
    print(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    device = torch.device(args.device)

    # model_name = 'google-bert/bert-base-uncased'
    model_name = 'bert-base-uncased'

    tkz = BertTokenizer.from_pretrained(model_name)
    print(tkz)
    enc_model: BertGenerationEncoder = BertGenerationEncoder.from_pretrained(model_name, bos_token_id=101, eos_token_id=102)
    # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
    dec_model: BertGenerationDecoder = BertGenerationDecoder.from_pretrained(
        model_name, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
    )
    model = EncoderEmbDecoderModel(encoder=enc_model, decoder=dec_model).to(device)

    val_ratio = 0.05
    df_sq = get_sq_df(exclude_empty_answers=True)
    df_sq_t, df_sq_v = split_df(df_sq, val_ratio=val_ratio)
    print(f'n_total = {len(df_sq)}. n_train = {len(df_sq_t)}. n_val = {len(df_sq_v)}')

    prefix = 'eedbert'
    mname = model_name.replace('-', '_')
    suffix = f'{mname}-d{enc_model.config.hidden_size}'
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

    if (args.pretrained_model_path / 'best.pth').exists() and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path)
        model_encdec_cfg_fpath = args.pretrained_model_path / ENCDEC_BERT_MODEL_CFG_FNAME
        model_encdec_cfg = parse_yaml_file_as(EncdecBertCfg, model_encdec_cfg_fpath)
        model_encdec = EncdecBert(model_encdec_cfg).to(device)
        model_encdec.load_state_dict(pretrained_checkpoint['model'], strict=False)
        state_dict = {k.replace('bert_model.', ''): v for k, v in model_encdec.enc_bert.state_dict().items()}
        print(f'Load model weights for encoder:', list(state_dict.keys()))
        # print(f'Current keys:', list(model.encoder.state_dict().keys()))
        # strict = True
        strict = False
        model.encoder.load_state_dict(state_dict, strict=strict)

    params = model.parameters()
    # params = [p for n, p in model.named_parameters()]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6, min_lr=1e-8)
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    last_epoch, val_loss_min = -1, None
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        np.random.seed(int(time.time() * 1000) % 10_000_000)

    train_batch_it = get_sq_batch_iterator(df_sq=df_sq_t, tkz=tkz, batch_size=args.batch_size, inp_len=args.inp_len, device=device, ques_inp=args.ques_inp)
    val_batch_it = get_sq_batch_iterator(df_sq=df_sq_v, tkz=tkz, batch_size=args.batch_size, inp_len=args.inp_len, device=device, ques_inp=args.ques_inp)

    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
    if prev_train_steps > 0:
        grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss = 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch: QnaBatch = next(train_batch_it)
            batch.gen_tensors()

            optimizer.zero_grad()
            loss = run_eed_model_on_batch(model, batch)
            loss.backward()
            optimizer.step()

            # Gradients must be available after loss.backward()
            if grad_log_ind % grad_log_interval == 0:
                log_weights_grads_stats(grad_log_step, model, tbsw)
                grad_log_step += 1
            grad_log_ind += 1

            train_loss += loss.item()
            s = f'Train. loss: {loss.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        model.eval()
        val_loss = 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch: QnaBatch = next(val_batch_it)
            loss = run_eed_model_on_batch(model, batch)
            if torch.isnan(loss):
                print(f'Loss is nan!!!')
                import sys
                sys.exit(0)

            val_loss += loss.item()
            s = f'Val. loss: {loss.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        val_loss /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)

        scheduler.step(val_loss)
        last_lr = scheduler.get_last_lr()[0]
        tbsw.add_scalar(f'{scheduler.__class__.__name__} lr', last_lr, epoch)

        print(f'Train loss: {train_loss:.6f}')
        print(f'Val loss:   {val_loss:.6f}')
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

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsTrainEedBertQna, main, 'Train EncoderEmbDecoder model on Qna dataset.', exception_handler=reraise)


