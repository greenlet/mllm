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
from transformers import AutoTokenizer

from mllm.config.model import HgEnhanceType, EncdecBertCfg, copy_override_encdec_bert_cfg, BertEmbType, \
    gen_prefpostfix_encdec_bert
from mllm.exp.args import ENCDEC_BERT_MODEL_CFG_FNAME, create_bool_str_field, is_arg_true, mask_tokens_ARG, next_tok_pred_ARG
from mllm.model.encdec_ranker_hg import EncdecBert, EncdecBertAgg
from mllm.model.losses import EncdecMaskPadBatchLoss, EncdecPadBatchLoss, EncdecMaskPadItemLoss, accum_losses, log_losses_to_tb, losses_to_str
from mllm.train.mask_utils import MaskCfg
from mllm.train.utils import find_create_train_path, log_weights_grads_stats, get_wiki_ds_batch_iterators2


enforce_encoder_mask_understanding_ARG = '--enforce-encoder-mask-understanding', 'Enforce encoder embeddings for both unmasked and masked inputs '
'to be similar'



class ArgsEncdecBertTrain(BaseModel):
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
    bert_model_name: str = Field(
        'bert-base-uncased',
        description='Pretrained BERT model name. Must be a model from Huggingface models hub (bert-base-*, bert-large-*).',
        cli=('--bert-model-name',),
    )
    bert_emb_type: BertEmbType = Field(
        BertEmbType.Cls,
        description=f'Bert embedding type. Can have values: {list(x.value for x in BertEmbType)}',
        cli=('--bert-emb-type',),
    )
    inp_len: int = Field(
        ...,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    dec_enhance_type: HgEnhanceType = Field(
        HgEnhanceType.Matmul,
        description=f'Decoder layer enhance type. Can have values: {list(x.value for x in HgEnhanceType)}',
        cli=('--dec-enhance-type',),
    )
    dec_n_layers: int = Field(
        0,
        description='Decoder number of layers.',
        cli=('--dec-n-layers',),
    )
    dec_n_similar_layers: int = Field(
        ...,
        description='Number of consecutive similar attention layers for each decoder level.',
        cli=('--dec-n-similar-layers',),
    )

    mask_tokens_STR: str = create_bool_str_field(*mask_tokens_ARG)
    @property
    def mask_tokens(self) -> bool:
        return is_arg_true(mask_tokens_ARG[0], self.mask_tokens_STR)

    mask_sep_freq: float = Field(
        ...,
        description='Sparse mask frequency from 0 to 1. When MASK_SEP_FREQ=0.2 this type of mask will be applied in 20% of cases randomly. '
                    'Must hold: 0 <= MASK_SEP_FREQ and MASK_SEP_FREQ + MASK_SEQ_FREQ <= 1',
        cli=('--mask-sep-freq',),
    )
    mask_sep_frac: float = Field(
        ...,
        description='Fraction of the input to mask using sparse masking.',
        cli=('--mask-sep-frac',),
    )
    mask_seq_freq: float = Field(
        ...,
        description='Sequential mask frequency from 0 to 1. When MASK_SEQ_FREQ=0.2 this type of mask will be applied in 20% of cases randomly. '
                    'Must hold: 0 <= MASK_SEQ_FREQ and MASK_SEP_FREQ + MASK_SEQ_FREQ <= 1',
        cli=('--mask-seq-freq',),
    )
    mask_seq_max_frac: float = Field(
        ...,
        description='Fraction of the input to calculate maximum length of tokens sequence to mask. Resulting value is combined '
                    'with MASK_SEQ_MAX_LEN using min() function.',
        cli=('--mask-seq-max-frac',),
    )
    mask_seq_max_len: int = Field(
        ...,
        description='Maximum length of tokens sequence to mask. Combined with value derived from MASK_SEQ_MAX_FRAC using min() function.',
        cli=('--mask-seq-max-len',),
    )

    next_tok_pred_STR: str = create_bool_str_field(*next_tok_pred_ARG)
    @property
    def next_tok_pred(self) -> bool:
        return is_arg_true(next_tok_pred_ARG[0], self.next_tok_pred_STR)

    dec_dropout_rate: float = Field(
        0.0,
        required=False,
        description='Decoder dropout rate.',
        cli=('--dec-dropout-rate',),
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

    enforce_encoder_mask_understanding_STR: str = create_bool_str_field(*enforce_encoder_mask_understanding_ARG)
    @property
    def enforce_encoder_mask_understanding(self) -> bool:
        return is_arg_true(enforce_encoder_mask_understanding_ARG[0], self.enforce_encoder_mask_understanding_STR)

def main(args: ArgsEncdecBertTrain) -> int:
    print(args)
    if args.pretrained_model_path and args.pretrained_model_path.name:
        pretrained_model_path = args.pretrained_model_path
        if not pretrained_model_path.is_file():
            pretrained_model_path /= 'best.pth'
    else:
        pretrained_model_path = None

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    device = torch.device(args.device)

    model_cfg = parse_yaml_file_as(EncdecBertCfg, args.model_cfg_fpath)
    model_cfg = copy_override_encdec_bert_cfg(
        model_cfg, pretrained_model_name=args.bert_model_name, emb_type=args.bert_emb_type, inp_len=args.inp_len, dec_enhance_type=args.dec_enhance_type,
        dec_n_layers=args.dec_n_layers, dec_n_similar_layers=args.dec_n_similar_layers, dec_dropout_rate=args.dec_dropout_rate,
    )

    mask_cfg = None
    if args.mask_tokens:
        mask_cfg = MaskCfg(
            sep_freq=args.mask_sep_freq, sep_frac=args.mask_sep_frac, seq_freq=args.mask_seq_freq, seq_max_frac=args.mask_seq_max_frac,
            seq_max_len=args.mask_seq_max_len,
        )
    prefix, suffix = gen_prefpostfix_encdec_bert(
        model_cfg, mask_cfg=mask_cfg, pretrained_model_path=pretrained_model_path,
    )
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
        chkpt_model_cfg = parse_yaml_file_as(EncdecBertCfg, train_path / ENCDEC_BERT_MODEL_CFG_FNAME)
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / ENCDEC_BERT_MODEL_CFG_FNAME, model_cfg)

    tkz = AutoTokenizer.from_pretrained(model_cfg.enc_bert.pretrained_model_name)

    print(model_cfg)
    model = EncdecBertAgg(
        model_cfg, tkz, enforce_enc_mask_understanding=args.enforce_encoder_mask_understanding,
        next_tok_pred=args.next_tok_pred,
    )
    model.to(device)

    if checkpoint is None:
        model.load_pretrained(pretrained_model_path)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    last_epoch, val_loss_min, shuffle = -1, None, False
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        del checkpoint
        shuffle = True

    train_batch_it, val_batch_it = get_wiki_ds_batch_iterators2(
        wiki_ds_name=args.wiki_ds_name, data_path=args.data_path, inp_len=args.inp_len, docs_batch_size=args.docs_batch_size,
        tkz=tkz, mask_cfg=mask_cfg, device=device, shuffle=shuffle,
    )

    sched_wait_steps = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6, min_lr=1e-8)
    lr = optimizer.param_groups[0]['lr']
    print(f'Scheduler {scheduler.__class__.__name__} lr: {lr:0.10f}.')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    print(model)

    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
    if prev_train_steps > 0:
        grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_losses = {}
        train_loss = 0.0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            tokens_inp, tokens_inp_aug, _ = next(train_batch_it)

            optimizer.zero_grad()
            loss_dict = model(tokens_inp_aug, tokens_inp)
            loss = loss_dict['loss']
            loss.backward()

            # Gradients must be available after loss.backward()
            if grad_log_ind % grad_log_interval == 0:
                log_weights_grads_stats(grad_log_step, model, tbsw)
                grad_log_step += 1
            grad_log_ind += 1

            optimizer.step()
            train_loss += loss.item()
            accum_losses(loss_dict, train_losses)

            # if i_train == 2:
            #     import sys
            #     sys.exit()

            loss_str = losses_to_str(train_losses)
            pbar.set_postfix_str(f'Train. {loss_str}')
        pbar.close()
        train_loss /= args.train_epoch_steps
        log_losses_to_tb('Train', epoch, train_losses, tbsw)

        model.eval()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_losses = {}
        val_loss = 0.0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            tokens_inp, tokens_inp_aug, _ = next(val_batch_it)

            with torch.no_grad():
                loss_dict = model(tokens_inp_aug, tokens_inp)
            loss = loss_dict['loss']

            val_loss += loss.item()
            val_losses = accum_losses(loss_dict, val_losses)

            s = losses_to_str(val_losses)
            pbar.set_postfix_str(s)
        pbar.close()
        val_loss /= args.val_epoch_steps
        log_losses_to_tb('Val', epoch, val_losses, tbsw)

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
    run_and_exit(ArgsEncdecBertTrain, main, 'Train Encoder-Decoder Hourglass model.', exception_handler=rethrow)

