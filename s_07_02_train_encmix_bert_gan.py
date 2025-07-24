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

from mllm.config.model import EncdecBertCfg, EncmixBertCfg, copy_override_encmix_bert_cfg, gen_prefpostfix_encmix_bert, \
    EncmixOutEmbsType, EncmixTrainDsType
from mllm.exp.args import ENCDEC_BERT_MODEL_CFG_FNAME, ENCMIX_BERT_MODEL_CFG_FNAME
from mllm.model.encmix import EncmixBert, qna_gan_loss, EncmixBertGan
from mllm.train.utils import find_create_train_path, log_weights_grads_stats, get_wiki_ds_batch_iterators, QnaQuesInp, \
    gen_loss, get_squadv2_batch_iterator, get_squadv2_tensor_iterators, \
    get_squadv2_txt_iterators
from mllm.data.utils import get_squadv2_df, split_df
from transformers import AutoTokenizer


class ArgsEncmixBertGanTrain(BaseModel):
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
    train_ds_type: EncmixTrainDsType = Field(
        EncmixTrainDsType.Msk,
        description=f'Train dataset type, one of: {[tds.value for tds in EncmixTrainDsType]}',
        cli=('--train-ds-type',),
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
    out_embs_type: EncmixOutEmbsType = Field(
        EncmixOutEmbsType.Non,
        description=f'Out embeddings type. Possible values: {[t.value for t in EncmixOutEmbsType]}.',
        cli=('--out-embs-type',),
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


def main(args: ArgsEncmixBertGanTrain) -> int:
    print(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    device = torch.device(args.device)

    model_cfg = parse_yaml_file_as(EncmixBertCfg, args.model_cfg_fpath)
    model_cfg = copy_override_encmix_bert_cfg(
        model_cfg, inp_len=args.inp_len, out_embs_type=args.out_embs_type,
    )

    prefix, suffix = gen_prefpostfix_encmix_bert(model_cfg, train_ds_type=args.train_ds_type)
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
        chkpt_model_cfg = parse_yaml_file_as(EncmixBertCfg, train_path / ENCMIX_BERT_MODEL_CFG_FNAME)
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / ENCMIX_BERT_MODEL_CFG_FNAME, model_cfg)

    tkz = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name)

    print(model_cfg)
    model = EncmixBertGan(model_cfg, tkz=tkz, device=device)

    print(args.pretrained_model_path and (args.pretrained_model_path / 'best.pth').exists(), checkpoint)
    if args.pretrained_model_path and (args.pretrained_model_path / 'best.pth').exists() and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path)
        model.gen_model.load_state_dict(pretrained_checkpoint['model'], strict=False)
        model.dis_model.load_state_dict(pretrained_checkpoint['model'], strict=False)

    betas = (0.5, 0.999)
    opt_gen = torch.optim.Adam(model.get_gen_parameters(), lr=args.learning_rate, betas=betas)
    opt_dis = torch.optim.Adam(model.get_dis_parameters(), lr=args.learning_rate, betas=betas)

    last_epoch, val_loss_min, shuffle = -1, None, False
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        opt_gen.load_state_dict(checkpoint['optimizer_generator'])
        opt_dis.load_state_dict(checkpoint['optimizer_discriminator'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        del checkpoint

    val_ratio = 0.05
    if args.train_ds_type == EncmixTrainDsType.Qna:
        train_it, val_it = get_squadv2_txt_iterators(exclude_empty_answers=True, val_ratio=val_ratio)
    else:
        raise Exception(f'Train dataset type {args.train_ds_type} is not supported.')

    sched_wait_steps = 0
    sched_params = dict(mode='min', factor=0.5, patience=10, threshold=1e-6, min_lr=1e-8)
    sched_gen = ReduceLROnPlateau(opt_gen, **sched_params)
    sched_dis = ReduceLROnPlateau(opt_gen, **sched_params)
    lr_gen = opt_gen.param_groups[0]['lr']
    lr_dis = opt_dis.param_groups[0]['lr']
    print(f'Generator scheduler {sched_gen.__class__.__name__} lr: {lr_gen:0.10f}.')
    print(f'Discriminator scheduler {sched_dis.__class__.__name__} lr: {lr_dis:0.10f}.')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    # print(model)

    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
    if prev_train_steps > 0:
        grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss, train_loss_gen, train_loss_dis = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            item = next(train_it)

            # opt_gen.zero_grad()
            # opt_dis.zero_grad()
            # loss, loss_gen, loss_dis = model.run_qna_gan(context=item.context, question=item.question, answer=item.answer)
            # loss_gen.backward(retain_graph=True)
            # opt_gen.step()
            # loss_dis.backward()
            # opt_dis.step()

            opt_gen.zero_grad()
            opt_dis.zero_grad()
            loss_gen = model.run_qna_gan(context=item.context, question=item.question, answer=item.answer, is_gen=True)
            loss_gen.backward()

            # Gradients must be available after loss.backward()
            if grad_log_ind % grad_log_interval == 0:
                log_weights_grads_stats(grad_log_step, model, tbsw)
                grad_log_step += 1
            grad_log_ind += 1

            opt_gen.step()

            opt_gen.zero_grad()
            opt_dis.zero_grad()
            loss_dis = model.run_qna_gan(context=item.context, question=item.question, answer=item.answer, is_gen=False)
            loss_dis.backward()
            opt_dis.step()

            loss = (loss_gen + loss_dis) / 2
            train_loss += (loss_gen.item() + loss_dis.item()) / 2
            train_loss_gen += loss_gen.item()
            train_loss_dis += loss_dis.item()

            # if i_train == 2:
            #     import sys
            #     sys.exit()

            s = f'Train. loss: {loss.item():.6f}. loss_gen: {loss_gen.item():.6f}. loss_dis: {loss_dis.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        train_loss /= args.train_epoch_steps
        train_loss_gen /= args.train_epoch_steps
        train_loss_dis /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)
        tbsw.add_scalar('LossGen/Train', train_loss_gen, epoch)
        tbsw.add_scalar('LossDis/Train', train_loss_dis, epoch)

        model.eval()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_loss, val_loss_gen, val_loss_dis = 0, 0, 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            item = next(val_it)

            with torch.no_grad():
                loss_gen = model.run_qna_gan(context=item.context, question=item.question, answer=item.answer, is_gen=True)
                loss_dis = model.run_qna_gan(context=item.context, question=item.question, answer=item.answer, is_gen=False)
                loss = (loss_gen + loss_dis) / 2
            val_loss += loss.item()
            val_loss_gen += loss_gen.item()
            val_loss_dis += loss_dis.item()

            s = f'Val. loss: {loss.item():.6f}. loss_gen: {loss_gen.item():.6f}. loss_dis: {loss_dis.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        val_loss /= args.val_epoch_steps
        val_loss_gen /= args.val_epoch_steps
        val_loss_dis /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)
        tbsw.add_scalar('LossGen/Val', val_loss_gen, epoch)
        tbsw.add_scalar('LossDis/Val', val_loss_dis, epoch)

        if epoch >= sched_wait_steps:
            sched_gen.step(val_loss)
            sched_dis.step(val_loss)
        last_lr_gen = opt_gen.param_groups[0]['lr']
        last_lr_dis = opt_dis.param_groups[0]['lr']
        tbsw.add_scalar(f'Generator {sched_gen.__class__.__name__} lr', last_lr_gen, epoch)
        tbsw.add_scalar(f'Discriminator {sched_gen.__class__.__name__} lr', last_lr_dis, epoch)

        print(f'Train loss: {train_loss:.6f}. Val loss: {val_loss:.6f}')
        best = False
        if val_loss_min is None or val_loss < val_loss_min:
            val_loss_str = f'{val_loss_min}' if val_loss_min is None else f'{val_loss_min:.6f}'
            print(f'Val min loss change: {val_loss_str} --> {val_loss:.6f}')
            val_loss_min = val_loss
            best = True

        checkpoint = {
            'model': model.state_dict(),
            'optimizer_generator': opt_gen.state_dict(),
            'optimizer_discriminator': opt_dis.state_dict(),
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
    run_and_exit(ArgsEncmixBertGanTrain, main, 'Train EncmixBert model with adversarial counterpart.', exception_handler=rethrow)

