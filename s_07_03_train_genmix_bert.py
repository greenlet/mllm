import shutil
import sys
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

from mllm.config.model import GenmixBertCfg, copy_override_genmix_bert_cfg, gen_prefpostfix_genmix_bert, \
    GenmixTrainDsType, GenmixEmbAggType, GenmixEmbExpType
from mllm.exp.args import GENMIX_BERT_MODEL_CFG_FNAME, create_bool_str_field, is_arg_true
from mllm.model.genmix import GenmixBert
from mllm.train.utils import find_create_train_path, log_weights_grads_stats, get_squadv2_txt_iterators, \
    get_billsum_txt_iterators, SumTuple, QnaTuple
from mllm.data.wiki.itwiki import WikiItem, get_wiki_iterators

mask_tgt_ARG = '--mask-tgt', 'Masks textual target'
pred_tgt_all_ARG = '--pred-tgt-all', 'Predict all target tokens at once'


class ArgsGenmixBertTrain(BaseModel):
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

    mask_tgt_str: str = create_bool_str_field(*mask_tgt_ARG)
    @property
    def mask_tgt(self) -> bool:
        return is_arg_true(mask_tgt_ARG[0], self.mask_tgt_str)

    max_tgt_len_freq: float = Field(
        ...,
        description='Max target words ratio to the total number of words. When MAX_TGT_LEN > 0, the minimum of the resulting values will be taken.',
        cli=('--max-tgt-len-freq',),
    )
    max_tgt_len: int = Field(
        ...,
        description='Max target words number. When MAX_TGT_LEN_FREQ > 0, the minimum of the resulting values will be taken.',
        cli=('--max-tgt-len',),
    )

    pred_tgt_all_str: str = create_bool_str_field(*pred_tgt_all_ARG)
    @property
    def pred_tgt_all(self) -> bool:
        return is_arg_true(pred_tgt_all_ARG[0], self.pred_tgt_all_str)

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
    n_first_embs: int = Field(
        ...,
        description='Number of the first embeddings to be extracted from embedding generating model. If N_FIRST_EMBS > INP_LEN '
            'then all INP_LEN embeddings will be passed to Generating model.',
        cli=('--n-first-embs',),
    )
    n_second_embs: int = Field(
        ...,
        description='Number of embeddings that will be created from N_FIRST_EMBS and serve as an input to Generator.',
        cli=('--n-second-embs',),
    )
    emb_agg_type: GenmixEmbAggType = Field(
        GenmixEmbAggType.Fst,
        description=f'Aggregation method for N_FIRST_EMBS: {[t.value for t in GenmixEmbAggType]}',
        cli=('--emb-agg-type',),
    )
    emb_exp_type: GenmixEmbExpType = Field(
        GenmixEmbExpType.Non,
        description=f'Embeddings expansion type from N_FIRST_EMBS to N_SECOND_EMBS: {[t.value for t in GenmixEmbExpType]}',
        cli=('--emb-exp-type',),
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
    mask_tgt, pred_tgt_all = args.mask_tgt, args.pred_tgt_all

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    device = torch.device(args.device)

    model_cfg = parse_yaml_file_as(GenmixBertCfg, args.model_cfg_fpath)
    model_cfg = copy_override_genmix_bert_cfg(
        model_cfg, pretrained_model_name=args.bert_model_name, inp_len=args.inp_len, max_inp_chunks=args.max_inp_chunks,
        max_out_toks=args.max_out_toks, n_first_embs=args.n_first_embs, n_second_embs=args.n_second_embs,
        emb_agg_type=args.emb_agg_type, emb_exp_type=args.emb_exp_type,
    )

    prefix, suffix = gen_prefpostfix_genmix_bert(
        model_cfg, train_ds_type=args.train_ds_type, mask_tgt=mask_tgt, max_tgt_len_freq=args.max_tgt_len_freq,
        max_tgt_len=args.max_tgt_len, pred_tgt_all=pred_tgt_all,
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
        chkpt_model_cfg = parse_yaml_file_as(GenmixBertCfg, train_path / GENMIX_BERT_MODEL_CFG_FNAME)
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / GENMIX_BERT_MODEL_CFG_FNAME, model_cfg)

    print(model_cfg)
    model = GenmixBert(model_cfg, device=device)

    if args.pretrained_model_path and (args.pretrained_model_path / 'best.pth').exists() and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path, map_location=device)
        # model.load_state_dict(pretrained_checkpoint['model'], strict=False)
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
        model.enc.load_state_dict(model_chkpt, strict=True)
        del pretrained_checkpoint
        del model_chkpt

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    last_epoch, val_loss_min, shuffle = -1, None, False
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        shuffle = True
        del checkpoint

    val_ratio = 0.05
    if args.train_ds_type == GenmixTrainDsType.Qna:
        train_it, val_it = get_squadv2_txt_iterators(exclude_empty_answers=True, val_ratio=val_ratio)
    elif args.train_ds_type == GenmixTrainDsType.Sum:
        train_it, val_it = get_billsum_txt_iterators(val_ratio=val_ratio)
    elif args.train_ds_type == GenmixTrainDsType.Wki:
        train_it, val_it = get_wiki_iterators(
            data_path=args.data_path, val_ratio=val_ratio, shuffle=False,
        )
    else:
        raise Exception(f'Dataset type {args.train_ds_type} is not supported.')

    sched_wait_steps = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12, threshold=1e-6, min_lr=1e-8)
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
            elif args.train_ds_type == GenmixTrainDsType.Wki:
                item: WikiItem = item
                loss = model.run_on_wiki_txt_all(
                    title=item.title, text=item.text, mask_tgt=mask_tgt, max_tgt_len_freq=args.max_tgt_len_freq,
                    max_tgt_len=args.max_tgt_len, pred_tgt_all=pred_tgt_all,
                )
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
                    loss = model.run_on_sum_txt(text=item.text, summary=item.summary, title=item.title)
                elif args.train_ds_type == GenmixTrainDsType.Wki:
                    item: WikiItem = item
                    loss = model.run_on_wiki_txt_all(
                        title=item.title, text=item.text, mask_tgt=mask_tgt, max_tgt_len_freq=args.max_tgt_len_freq,
                        max_tgt_len=args.max_tgt_len, pred_tgt_all=pred_tgt_all,
                    )
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
    def rethrow(e):
        raise e
    run_and_exit(ArgsGenmixBertTrain, main, 'Train EncmixBert model to predict masked input.', exception_handler=rethrow)

