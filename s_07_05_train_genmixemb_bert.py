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

from mllm.config.model import GenmixTrainDsType, TokensAggType, GenmixembBertCfg, copy_override_genmixemb_bert_cfg, \
    gen_prefpostfix_genmixemb_bert, HgReductType, BertAggType, CtxQuePromptType
from mllm.data.itsquadv2 import get_squadv2_batch_iterators_v2, QnaBatchV2
from mllm.exp.args import GENMIXEMB_BERT_MODEL_CFG_FNAME, create_bool_str_field, is_arg_true
from mllm.model.genmixemb_bert import GenmixembBert
from mllm.train.mask_utils import MaskCfg
from mllm.train.utils import find_create_train_path, log_weights_grads_stats, SumTuple, QnaTuple
from mllm.data.wiki.itwiki import WikiItem, get_wiki_batch_iterators, WikiBatch
from mllm.utils.utils import rethrow

train_agg_model_ARG = '--train-agg-model', 'Train aggregation model'
pred_next_sent_ARG = '--pred-next-sent', 'Predict next sentence'
mask_tokens_ARG = '--mask-tokens', 'Mask input tokens'
share_agg_enc_token_embeds_ARG = '--share-agg-enc-token-embeds', 'Share token embeddings between aggregator model and encoder'
add_token_type_ids_ARG = '--add-token-type-ids', 'Add token type ids to input tokens'
join_ctx_que_agg_ARG = '--join-ctx-que-agg', 'Join context and question for aggregation'
pyr_share_layer_weights_ARG = '--pyr-share-layer-weights', 'Share Pyramid layers weights between levels'


class ArgsGenmixembBertTrain(BaseModel):
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
    max_inp_toks: int = Field(
        ...,
        description='Maximum number of tokens in input to aggregate using TOKS_AGG_TYPE model.',
        cli=('--max-inp-toks',),
    )
    max_out_toks: int = Field(
        ...,
        description='Maximum number of predicted tokens.',
        cli=('--max-out-toks',),
    )
    toks_agg_type: TokensAggType = Field(
        TokensAggType.Bert,
        description=f'Aggregation method for sequence of tokens. Values {[t.value for t in TokensAggType]}',
        cli=('--toks-agg-type',),
    )
    bert_agg_type: BertAggType = Field(
        BertAggType.Sep,
        description=f'Bert aggregation type. Values: {[t.value for t in BertAggType]}',
        cli=('--bert-agg-type',),
    )
    bert_agg_n_subseq_toks: int = Field(
        ...,
        description=f'Number of sequential tokens to aggregate for Bert.',
        cli=('--bert-agg-n-subseq-toks',),
    )
    pyr_agg_type: HgReductType = Field(
        HgReductType.Decim,
        description=f'Pyramid aggregation type. Values: {[t.value for t in HgReductType]}',
        cli=('--pyr-agg-type',),
    )
    pyr_agg_step: int = Field(
        ...,
        description=
        f'Aggregation step for each Pyramid level. One level will reduce the number of tokens by factor PYR_AGG_STEP. '
        f'k levels will reduce by factor PYR_AGG_STEP^k',
        cli=('--pyr-agg-step',),
    )
    pyr_agg_n_levels: int = Field(
        ...,
        description=f'Number of hierarchical levels of aggregation for TOKS_AGG_TYPE={TokensAggType.Pyramid}.',
        cli=('--pyr-agg-n-levels',),
    )
    pyr_agg_n_layers_per_level: int = Field(
        ...,
        description=f'Number of self attention layers per level of aggregation for TOKS_AGG_TYPE={TokensAggType.Pyramid}.',
        cli=('--pyr-agg-n-layers-per-level',),
    )
    pyr_share_layer_weights_STR: str = create_bool_str_field(*pyr_share_layer_weights_ARG)
    @property
    def pyr_share_layer_weights(self) -> bool:
        return is_arg_true(pyr_share_layer_weights_ARG[0], self.pyr_share_layer_weights_STR)

    train_agg_model_STR: str = create_bool_str_field(*train_agg_model_ARG)
    @property
    def train_agg_model(self) -> bool:
        return is_arg_true(train_agg_model_ARG[0], self.train_agg_model_STR)

    pred_next_sent_STR: str = create_bool_str_field(*pred_next_sent_ARG)
    @property
    def pred_next_sent(self) -> bool:
        return is_arg_true(pred_next_sent_ARG[0], self.pred_next_sent_STR)

    share_agg_enc_token_embs_STR: str = create_bool_str_field(*share_agg_enc_token_embeds_ARG)
    @property
    def share_agg_enc_token_embs(self) -> bool:
        return is_arg_true(share_agg_enc_token_embeds_ARG[0], self.share_agg_enc_token_embs_STR)

    add_token_type_ids_STR: str = create_bool_str_field(*add_token_type_ids_ARG)
    @property
    def add_token_type_ids(self) -> bool:
        return is_arg_true(add_token_type_ids_ARG[0], self.add_token_type_ids_STR)

    join_ctx_que_agg_STR: str = create_bool_str_field(*join_ctx_que_agg_ARG)
    @property
    def join_ctx_que_agg(self) -> bool:
        return is_arg_true(join_ctx_que_agg_ARG[0], self.join_ctx_que_agg_STR)

    ctx_que_prompt_type: CtxQuePromptType = Field(
        CtxQuePromptType.Tok,
        description=f'Context-Question prompt type. Values: {[t.value for t in CtxQuePromptType]}',
        cli=('--ctx-que-prompt-type',),
    )

    n_toks_min: int = Field(
        ...,
        description='Minimum number of tokens in text data to include it into training. Texts with less number of tokens will be skipped.',
        cli=('--n-toks-min',),
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


def main(args: ArgsGenmixembBertTrain) -> int:
    print(args)
    pretrained_model_path = args.pretrained_model_path if args.pretrained_model_path and args.pretrained_model_path.name else None

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    device = torch.device(args.device)

    model_cfg = parse_yaml_file_as(GenmixembBertCfg, args.model_cfg_fpath)
    model_cfg = copy_override_genmixemb_bert_cfg(
        model_cfg, bert_model_name=args.bert_model_name, max_inp_toks=args.max_inp_toks, max_out_toks=args.max_out_toks, toks_agg_type=args.toks_agg_type,
        bert_agg_type=args.bert_agg_type, bert_agg_n_subseq_toks=args.bert_agg_n_subseq_toks, pyr_agg_type=args.pyr_agg_type, pyr_agg_step=args.pyr_agg_step,
        pyr_agg_n_levels=args.pyr_agg_n_levels, pyr_agg_n_layers_per_level=args.pyr_agg_n_layers_per_level, pyr_share_layer_weights=args.pyr_share_layer_weights,
        train_agg_model=args.train_agg_model, share_agg_enc_token_embeds=args.share_agg_enc_token_embs, add_token_type_ids=args.add_token_type_ids,
        join_ctx_que_agg=args.join_ctx_que_agg, ctx_que_prompt_type=args.ctx_que_prompt_type,
    )

    mask_cfg = None
    if args.mask_tokens:
        mask_cfg = MaskCfg(
            sep_freq=args.mask_sep_freq, sep_frac=args.mask_sep_frac, seq_freq=args.mask_seq_freq, seq_max_frac=args.mask_seq_max_frac,
            seq_max_len=args.mask_seq_max_len,
        )
    prefix, suffix = gen_prefpostfix_genmixemb_bert(
        model_cfg, train_ds_type=args.train_ds_type, mask_cfg=mask_cfg, pred_next_sent=args.pred_next_sent, pretrained_model_path=pretrained_model_path,
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
        chkpt_model_cfg = parse_yaml_file_as(GenmixembBertCfg, train_path / GENMIXEMB_BERT_MODEL_CFG_FNAME)
        assert model_cfg == chkpt_model_cfg, f'{model_cfg} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / GENMIXEMB_BERT_MODEL_CFG_FNAME, model_cfg)

    print(model_cfg)
    model = GenmixembBert(model_cfg, device=device)

    if pretrained_model_path and checkpoint is None:
        dname = pretrained_model_path.parent.name
        print(f'Loading checkpoint with pretrained model from {args.pretrained_model_path}')
        pretrained_checkpoint = torch.load(args.pretrained_model_path, map_location=device)
        # model.load_state_dict(pretrained_checkpoint['model'], strict=False)
        print(list(pretrained_checkpoint['model'].keys()))
        if dname.startswith('encdecbert-'):
            prefix = 'enc_bert.bert_model.'
            prefix_len = len(prefix)
            model_chkpt = {}
            for k, v in pretrained_checkpoint['model'].items():
                if k.startswith(prefix):
                    k = k[prefix_len:]
                if k.startswith('dec_pyr.'):
                    continue
                model_chkpt[k] = v
            model.agg.load_state_dict(model_chkpt, strict=True)
            del model_chkpt
        elif dname.startswith('genmixemb-'):
            # strict = True
            strict = False
            model.load_state_dict(pretrained_checkpoint['model'], strict=strict)
        else:
            raise Exception(f'Model checkpoint {args.pretrained_model_path.name} is not supported')
        del pretrained_checkpoint

    if model_cfg.train_agg_model:
        params = model.parameters()
    else:
        params = model.gen.parameters()
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
    if args.train_ds_type == GenmixTrainDsType.Wki:
        train_it, val_it = get_wiki_batch_iterators(
            data_path=args.data_path, tkz=model.tkz, batch_size=args.batch_size, val_ratio=val_ratio, shuffle=False, rand_seed=args.random_seed,
            n_toks_min=args.n_toks_min, n_toks_max=args.max_inp_toks, mask_cfg=mask_cfg, device=device, pred_next_sent=args.pred_next_sent,
            n_toks_pred_max=args.max_out_toks,
        )
    elif args.train_ds_type == GenmixTrainDsType.Qna:
        train_it, val_it = get_squadv2_batch_iterators_v2(
            batch_size=args.batch_size, exclude_empty_answers=True, tkz=model.tkz, max_inp_len=args.max_inp_toks, max_out_len=args.max_out_toks,
            device=device,
        )
    else:
        raise Exception(f'Dataset type {args.train_ds_type} is not supported.')

    sched_wait_steps = 0
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12, threshold=1e-6, min_lr=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6, min_lr=1e-8)
    # lr = scheduler.get_last_lr()[0]
    lr = optimizer.param_groups[0]['lr']
    print(f'Scheduler {scheduler.__class__.__name__} lr: {lr:0.10f}.')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    print(model)

    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
    if prev_train_steps > 0:
        grad_log_step = (prev_train_steps - 1) // grad_log_interval + 1
        grad_log_ind = prev_train_steps
    for epoch in range(last_epoch + 1, args.epochs):
        if model_cfg.train_agg_model:
            model.train()
        else:
            model.agg.eval()
            model.gen.train()
        train_loss = 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            item = next(train_it)

            optimizer.zero_grad()
            if args.train_ds_type == GenmixTrainDsType.Wki:
                batch: WikiBatch = item
                loss = model.run_on_wiki(batch=batch)
            elif args.train_ds_type == GenmixTrainDsType.Qna:
                batch: QnaBatchV2 = item
                loss = model.run_on_qna(batch=batch)
                # loss = model.run_on_qna_v2(batch=batch)
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
                if args.train_ds_type == GenmixTrainDsType.Wki:
                    batch: WikiBatch = item
                    loss = model.run_on_wiki(batch=batch)
                elif args.train_ds_type == GenmixTrainDsType.Qna:
                    batch: QnaBatchV2 = item
                    loss = model.run_on_qna(batch=batch)
                    # loss = model.run_on_qna_v2(batch=batch)
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
        ArgsGenmixembBertTrain, main, 'Train GenmixembBert model to predict masked input.',
        exception_handler=rethrow,
    )

