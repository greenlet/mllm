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
from transformers import BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, BertConfig, BertGenerationConfig

from mllm.config.model import EncdecBertCfg
from mllm.data.utils import HfDsIterator
from mllm.exp.args import ENCDEC_BERT_MODEL_CFG_FNAME, is_arg_true, ARG_TRUE_VALUES_STR, ARG_FALSE_VALUES_STR
from mllm.model.embgen_bert import EncoderEmbDecoderModel, EncEmbExpansionType, EncoderEmbDecoderConfig
from mllm.model.encdec_ranker_hg import EncdecBert
from mllm.train.embgen_bert import run_eed_model_on_batch, get_eed_bert_model, run_eed_model_on_masked_input
from mllm.train.utils import find_create_train_path, log_weights_grads_stats, get_wiki_ds_batch_iterators, QnaQuesInp, \
    QnaBatch, get_squadv2_df, split_df, get_squadv2_batch_iterator
from mllm.utils.utils import reraise


class ArgsTrainEedBertQna(BaseModel):
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
    in_empty_ans: str = Field(
        'true',
        required=False,
        description='Boolean flag determining whether include empty answers in dataset or no. ' \
            f'EMPTY_ANS can take value from {ARG_TRUE_VALUES_STR} to be True or {ARG_FALSE_VALUES_STR} to be False.',
        cli=('--in-empty-ans',),
    )
    @property
    def in_empty_ans_bool(self) -> bool:
        return is_arg_true('--in-empty-ans', self.in_empty_ans)

    ques_inp: QnaQuesInp = Field(
        ...,
        description=f'Question input type: {list(qi for qi in QnaQuesInp)}.',
        cli = ('--ques-inp',)
    )
    enc_emb_exp_type: EncEmbExpansionType = Field(
        ...,
        description=f'Encoder embedding expansion type: {list(et for et in EncEmbExpansionType)}.',
        cli = ('--enc-emb-exp-type',)
    )
    enc_emb_exp_bias: str = Field(
        'false',
        description=f'Encoder embedding bias presence for Matrix expansion. True values: {ARG_TRUE_VALUES_STR}. False values: {ARG_FALSE_VALUES_STR}',
        cli = ('--enc-emb-exp-bias',)
    )
    @property
    def enc_emb_exp_bias_bool(self) -> bool:
        return is_arg_true('--enc-emb-exp-bias', self.enc_emb_exp_bias)

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


def gen_prefpostfix_embgen_bert_txt(args: ArgsTrainEedBertQna, bert_cfg: EncoderEmbDecoderConfig) -> tuple[str, str]:
    prefix = 'eedbert_txt'

    # Model name
    model_name = bert_cfg.encoder.name_or_path.replace('-', '_')
    postfix_parts = [model_name]

    # Model hidden size
    postfix_parts.append(f'd{bert_cfg.encoder.hidden_size}')

    # Train with or wihout empty answers
    emp_ans = str(args.in_empty_ans_bool)[0].lower()
    postfix_parts.append(f'emp_{emp_ans}')

    exp_str = f'exp_{bert_cfg.enc_emb_exp_type}'
    if bert_cfg.enc_emb_exp_bias:
        exp_str = f'{exp_str}_b'
    postfix_parts.append(exp_str)

    postfix_parts.append(f'bt_{bert_cfg.enc_inp_batch_size}')

    # If initializing weights from checkpoint, add its name and datetime
    if args.pretrained_model_path is not None and (args.pretrained_model_path / 'best.pth').exists():
        ch_parts = args.pretrained_model_path.name.split('-')[:2]
        ch_name = '_'.join(ch_parts)
    else:
        ch_name = 'none'
    postfix_parts.append(f'chkpt_{ch_name}')

    # Each part consists of [0-9a-z] and '_' symbols. Parts are divided by '-'
    postfix = '-'.join(postfix_parts)

    return prefix, postfix


def main(args: ArgsTrainEedBertQna) -> int:
    print(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    device = torch.device(args.device)

    tkz, model = get_eed_bert_model(
        inp_len=args.inp_len, ques_inp=args.ques_inp, enc_emb_exp_type=args.enc_emb_exp_type, enc_emb_exp_bias=args.enc_emb_exp_bias_bool,
        batch_size=args.batch_size, device=device,
    )

    prefix, postfix = gen_prefpostfix_embgen_bert_txt(args, model.config)
    train_path = find_create_train_path(args.train_root_path, prefix, postfix, args.train_subdir)
    print(f'train_path: {train_path}')

    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    checkpoint = None
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists(),\
            (f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
             f'but file {last_checkpoint_path} does not exits.')

    shuffle = False
    if last_checkpoint_path.exists():
        print(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        print(f'Checkpoint with keys {list(checkpoint.keys())} loaded')
        shuffle = True

    if args.pretrained_model_path is not None and (args.pretrained_model_path / 'best.pth').exists() and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path, map_location=device)
        # model_encdec_cfg_fpath = args.pretrained_model_path / ENCDEC_BERT_MODEL_CFG_FNAME
        # model_encdec_cfg = parse_yaml_file_as(EncdecBertCfg, model_encdec_cfg_fpath)
        # model_encdec = EncdecBert(model_encdec_cfg).to(device)
        # model_encdec.load_state_dict(pretrained_checkpoint['model'], strict=False)
        # state_dict = {k.replace('bert_model.', ''): v for k, v in model_encdec.enc_bert.state_dict().items()}

        state_dict = {k.replace('bert_model.', ''): v for k, v in pretrained_checkpoint['model'].items()}
        print(f'Load model weights for encoder:', list(state_dict.keys()))
        # print(f'Current keys:', list(model.encoder.state_dict().keys()))
        # strict = True
        strict = False
        model.encoder.load_state_dict(state_dict, strict=strict)
        for t in state_dict.values():
            t.to('cpu')
        del state_dict
        del pretrained_checkpoint

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

    train_batch_it, val_batch_it = get_wiki_ds_batch_iterators(
        wiki_ds_name=args.wiki_ds_name, data_path=args.data_path, inp_len=args.inp_len, docs_batch_size=args.batch_size,
        tkz=tkz, device=device, shuffle=shuffle,
    )

    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
    if prev_train_steps > 0:
        grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss = 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            docs_toks_aug, docs_toks_tgt = next(train_batch_it)

            optimizer.zero_grad()
            loss = run_eed_model_on_masked_input(model, tkz, docs_toks_aug, docs_toks_tgt)
            loss.backward()
            optimizer.step()

            # Gradients must be available after loss.backward()
            if grad_log_ind % grad_log_interval == 0:
                log_weights_grads_stats(grad_log_step, model, tbsw)
                grad_log_step += 1
            grad_log_ind += 1

            if torch.isnan(loss):
                print(f'Loss is nan!!!')
                import sys
                sys.exit(0)

            train_loss += loss.item()
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
            docs_toks_aug, docs_toks_tgt = next(val_batch_it)
            with torch.no_grad():
                loss = run_eed_model_on_masked_input(model, tkz, docs_toks_aug, docs_toks_tgt)
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


