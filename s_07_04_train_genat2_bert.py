import shutil
import sys
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
import torch
import torch.utils.tensorboard as tb
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.config.model import GenmixTrainDsType
from mllm.exp.args import is_arg_true, create_bool_str_field
from mllm.model.genat2_bert import Genat2Model
from mllm.config.configuration_bert_at2_generation import Genat2Cfg, gen_prefpostfix_genat2
from mllm.train.utils import find_create_train_path, log_weights_grads_stats, get_squadv2_txt_iterators, \
    get_billsum_txt_iterators, SumTuple, QnaTuple
from mllm.utils.utils import rethrow

encoder_enc_at2_enabled_ARG = '--encoder-enc-at2-enabled', 'Enables SelfAttention2 in Encoder'
decoder_enc_at2_enabled_ARG = '--decoder-enc-at2-enabled', 'Enables SelfAttention2 of encoder embeddings in Decoder'
decoder_dec_at2_enabled_ARG = '--decoder-dec-at2-enabled', 'Enables SelfAttention2 of decoder embeddings in Decoder'
decoder_last_dec_to_all_enc_at2_enabled_ARG = '--decoder-last-dec-to-all-enc-at2-enabled', 'Enables CrossAttention2 of last decoder embedding to encoder embeddings in Decoder'


class ArgsGenat2BertTrain(BaseModel):
    data_path: Path = Field(
        ...,
        description='Root data path. Must contain subpath `wikipedia/WIKI_DS_NAME` with Wikipedia dataset.',
        cli=('--data-path',),
    )
    train_root_path: Path = Field(
        ...,
        description='Path to train root directory. New train subdirectory will be created within each new run.',
        cli=('--train-root-path',),
    )
    pretrained_model_path: Optional[Path] = Field(
        None,
        description='Path to EncdecHg model train directory.',
        cli=('--pretrained-model-path',),
    )
    train_subdir: str = Field(
        '',
        description='Train subdirectory. Can have values: "last", "<subdirectory-name>". When set to "last", '
            'last subdirectory of TRAIN_ROOT_PATH containing training snapshot will be taken.',
        cli=('--train-subdir',)
    )
    train_ds_type: GenmixTrainDsType = Field(
        GenmixTrainDsType.Qna,
        description=f'Train dataset type, one of: {[t.value for t in GenmixTrainDsType]}',
        cli=('--train-ds-type',),
    )
    model_cfg_fpath: Path = Field(
        ...,
        description='Path to EncdecHg model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    bert_model_name: str = Field(
        'bert-base-uncased',
        description='Bert pretrained model name (bert-base-uncased, bert-large-uncased).',
        cli=('--bert-model-name',),
    )
    inp_len: int = Field(
        ...,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    max_inp_chunks: int = Field(
        ...,
        description='Maximum input chunks. Model input will have dimensions [n, INP_LEN] where n <= MAX_INP_CHUNKS.',
        cli=('--max-inp-chunks',),
    )
    max_out_toks: int = Field(
        ...,
        description='Maximum output tokens to use in training.',
        cli=('--max-out-toks',),
    )

    encoder_enc_at2_enabled_str: str = create_bool_str_field(*encoder_enc_at2_enabled_ARG)
    @property
    def encoder_enc_at2_enabled(self) -> bool:
        return is_arg_true(encoder_enc_at2_enabled_ARG[0], self.encoder_enc_at2_enabled_str)
    
    decoder_enc_at2_enabled_str: str = create_bool_str_field(*decoder_enc_at2_enabled_ARG)
    @property
    def decoder_enc_at2_enabled(self) -> bool:
        return is_arg_true(decoder_enc_at2_enabled_ARG[0], self.decoder_enc_at2_enabled_str)

    decoder_dec_at2_enabled_str: str = create_bool_str_field(*decoder_dec_at2_enabled_ARG)
    @property
    def decoder_dec_at2_enabled(self) -> bool:
        return is_arg_true(decoder_dec_at2_enabled_ARG[0], self.decoder_dec_at2_enabled_str)

    decoder_last_dec_to_all_enc_at2_enabled_str: str = create_bool_str_field(*decoder_last_dec_to_all_enc_at2_enabled_ARG)
    @property
    def decoder_last_dec_to_all_enc_at2_enabled(self) -> bool:
        return is_arg_true(decoder_last_dec_to_all_enc_at2_enabled_ARG[0], self.decoder_last_dec_to_all_enc_at2_enabled_str)

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


def main(args: ArgsGenat2BertTrain) -> int:
    print(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    device = torch.device(args.device)

    model_cfg = Genat2Cfg.copy_override(
        args.model_cfg_fpath, inp_len=args.inp_len, max_inp_chunks=args.max_inp_chunks, max_out_toks=args.max_out_toks,
        encoder_enc_at2_enabled=args.encoder_enc_at2_enabled, decoder_enc_at2_enabled=args.decoder_enc_at2_enabled,
        decoder_dec_at2_enabled=args.decoder_dec_at2_enabled, decoder_last_dec_to_all_enc_at2_enabled=args.decoder_last_dec_to_all_enc_at2_enabled,
    )

    prefix, suffix = gen_prefpostfix_genat2(model_cfg, train_ds_type=args.train_ds_type)
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
        chkpt_model_cfg = Genat2Cfg.load_from_yaml(train_path)
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        model_cfg.save_to_yaml(train_path)

    pprint(model_cfg)
    model = Genat2Model(model_cfg, device=device)

    if args.pretrained_model_path and (args.pretrained_model_path / 'best.pth').exists() and checkpoint is None:
        model.load_weights_from_pretrained_encoder(args.pretrained_model_path)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    last_epoch, val_loss_min, shuffle = -1, None, False
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        del checkpoint

    val_ratio = 0.05
    if args.train_ds_type == GenmixTrainDsType.Qna:
        train_it, val_it = get_squadv2_txt_iterators(exclude_empty_answers=True, val_ratio=val_ratio)
    elif args.train_ds_type == GenmixTrainDsType.Sum:
        train_it, val_it = get_billsum_txt_iterators(val_ratio=val_ratio)
    else:
        raise Exception(f'Dataset type {args.train_ds_type} is not supported.')

    sched_wait_steps = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6, min_lr=1e-8)
    # lr = scheduler.get_last_lr()[0]
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
        train_loss = 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            item = next(train_it)

            optimizer.zero_grad()
            if args.train_ds_type == GenmixTrainDsType.Qna:
                item: QnaTuple = item
                loss = model.run_on_qna_txt(
                    context=item.context, question=item.question, answer=item.answer,
                )
            elif args.train_ds_type == GenmixTrainDsType.Sum:
                item: SumTuple = item
                loss = model.run_on_sum_txt(text=item.text, summary=item.summary, title=item.title)
            else:
                raise
            if loss.isnan():
                print('Loss is NaN!!!')
                sys.exit(0)

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
            item = next(val_it)

            with torch.no_grad():
                if args.train_ds_type == GenmixTrainDsType.Qna:
                    item: QnaTuple = item
                    loss = model.run_on_qna_txt(
                        context=item.context, question=item.question, answer=item.answer,
                    )
                elif args.train_ds_type == GenmixTrainDsType.Sum:
                    item: SumTuple = item
                    loss = model.run_on_sum_txt(title=item.title, text=item.text, summary=item.summary)
                else:
                    raise
                if loss.isnan():
                    print('Loss is NaN!!!')
                    sys.exit(0)

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
    run_and_exit(
        ArgsGenat2BertTrain, main, 'Train Genat2Bert model on summary and qna datasets.',
        exception_handler=rethrow,
    )

