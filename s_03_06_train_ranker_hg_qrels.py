import re
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.tensorboard as tb
from pydantic import Field, BaseModel
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.config.model import TokenizerCfg, EncdecHgCfg, HgReductType, HgEnhanceType, PosEncType, RankerHgCfg, \
    gen_prefpostfix_ranker_hg, copy_override_ranker_hg_cfg
from mllm.data.dsqrels import QrelsPlainBatch
from mllm.data.utils import load_qrels_datasets
from mllm.exp.args import TOKENIZER_CFG_FNAME, ARG_TRUE_VALUES_STR, ARG_FALSE_VALUES_STR, is_arg_true, \
    ENCDEC_HG_MODEL_CFG_FNAME, RANKER_HG_MODEL_CFG_FNAME
from mllm.model.encdec_ranker_hg import EncdecHg, RankerHg
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer, tokenizer_from_config
from mllm.train.utils import find_create_train_path, calc_print_batches, log_weights_grads_stats


class ArgsRankerHgQrelsTrain(BaseModel):
    ds_dir_paths: list[Path] = Field(
        [],
        required=True,
        description='Qrels datasets directory paths. Supported datasets: Msmarco, Fever.'
                    'Naming convention: directory name must contain the name of dataset: msmarco, fever. Unknown datasets '
                    'will cause an error and exit.',
        cli=('--ds-dir-paths',),
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
        description='Path to RankerHg model config Yaml file.',
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
    dec_dropout_rate: float = Field(
        -1,
        required=False,
        description=f'Decoder dropout rate. If not set the value from encoder config will be used.',
        cli=('--dec-dropout-rate',),
    )

    dec_with_bias: str = Field(
        'false',
        required=False,
        description='Boolean flag determining whether decoder linear layer should have bias. ' \
            f'DEC_WITH_BIAS can take value from {ARG_TRUE_VALUES_STR} to be True or {ARG_FALSE_VALUES_STR} to be False. (default: true)',
        cli=('--dec-with-bias',),
    )
    @property
    def dec_with_bias_bool(self) -> bool:
        return is_arg_true('--dec-with-bias', self.dec_with_bias)

    dec_mlp_sizes: str = Field(
        '',
        required=False,
        description=f'Consecutive MLP sizes transforming initial embedding to a relevance vector. The size can be -1 which '
                    f'means the same size as initial embedding. When not set or empty, no MLP layers will be created. Default: empty.',
        cli=('--dec-mlp-sizes',)
    )
    @property
    def dec_mlp_sizes_list(self) -> list[int]:
        dec_mlp_sizes = re.compile(r'^[([\s]*|[)\]\s]*$').sub('', self.dec_mlp_sizes)
        parts = re.compile(r'[\s+,]+').split(dec_mlp_sizes)
        return [int(p) for p in parts if p]

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
    pretrained_model_path: Optional[Path] = Field(
        None,
        required=False,
        description='Path to EncdecHg model train directory.',
        cli=('--pretrained-model-path',),
    )
    train_dec_only: str = Field(
        'true',
        required=False,
        description='Boolean flag determining whether decoder\' weights will be trained or the full model. ' \
            f'TRAIN_DEC_ONLY can take value from {ARG_TRUE_VALUES_STR} to be True or {ARG_FALSE_VALUES_STR} to be False. (default: true)',
        cli=('--train-dec-only',),
    )
    @property
    def train_dec_only_bool(self) -> bool:
        return is_arg_true('--train-dec-only', self.train_dec_only)


# prob_pred: (n_docs, n_qs)
# mask_gt: (n_docs, n_qs)
def ranker_cos_loss(cos_pred: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_gt = mask_gt.to(torch.bool)
    n_docs = len(cos_pred)
    loss_tgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
    loss_nontgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
    prob_cap = torch.tensor(1e-6, dtype=torch.float32, device=cos_pred.device)
    for i in range(n_docs):
        probs_tgt = torch.masked_select(cos_pred[i], mask_gt[i])
        probs_nontgt = -torch.masked_select(cos_pred[i], ~mask_gt[i])
        probs_tgt = (probs_tgt + 1) / 2
        probs_nontgt = (probs_nontgt + 1) / 2
        probs_tgt = torch.maximum(probs_tgt, prob_cap)
        probs_nontgt = torch.maximum(probs_nontgt, prob_cap)
        lt, lnt = -torch.mean(torch.log(probs_tgt)), -torch.mean(torch.log(probs_nontgt))
        loss_tgt = loss_tgt + lt
        loss_nontgt = loss_nontgt + lnt
    loss_tgt = loss_tgt / n_docs
    loss_nontgt = loss_nontgt / n_docs
    loss = (loss_tgt + loss_nontgt) / 2
    if torch.isnan(loss).any():
        print('!!!', torch.isnan(cos_pred).any())
        print(mask_gt)
        import sys
        sys.exit(0)
    return loss, loss_tgt, loss_nontgt


class RankerCosEmbLoss(nn.Module):
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.register_buffer('margin', torch.scalar_tensor(margin))
        self.register_buffer('zero', torch.scalar_tensor(0.0))

    # prob_pred: (n_docs, n_qs)
    # mask_gt: (n_docs, n_qs)
    def forward(self, cos_pred: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_gt = mask_gt.to(torch.bool)
        n_docs = len(cos_pred)
        loss_tgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
        loss_nontgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
        for i in range(n_docs):
            probs_tgt = 1 - torch.masked_select(cos_pred[i], mask_gt[i])
            probs_nontgt = torch.masked_select(cos_pred[i], ~mask_gt[i])
            probs_nontgt = torch.maximum(probs_nontgt - self.margin, self.zero)
            lt, lnt = torch.mean(probs_tgt), torch.mean(probs_nontgt)
            loss_tgt = loss_tgt + lt
            loss_nontgt = loss_nontgt + lnt
        loss_tgt = loss_tgt / n_docs
        loss_nontgt = loss_nontgt / n_docs
        loss = (loss_tgt + loss_nontgt) / 2
        if torch.isnan(loss).any():
            print('!!!', torch.isnan(cos_pred).any())
            print(mask_gt)
            import sys
            sys.exit(0)
        return loss, loss_tgt, loss_nontgt


def main(args: ArgsRankerHgQrelsTrain) -> int:
    print(args)
    print(f'dec_mlp_sizes_list: {args.dec_mlp_sizes_list}')

    assert args.ds_dir_paths, '--ds-dir-paths is expected to list at least one Qrels datsaset'

    device = torch.device(args.device)

    tkz_cfg = parse_yaml_file_as(TokenizerCfg, args.tokenizer_cfg_fpath)
    model_cfg = parse_yaml_file_as(RankerHgCfg, args.model_cfg_fpath)
    model_cfg = copy_override_ranker_hg_cfg(
        model_cfg, inp_len=args.inp_len, n_similar_layers=args.n_similar_layers, reduct_type=args.reduct_type,
        pos_enc_type=args.pos_enc_type, dec_with_bias=args.dec_with_bias_bool, dec_mlp_sizes=args.dec_mlp_sizes_list,
    )
    print(model_cfg)

    prefix, suffix = gen_prefpostfix_ranker_hg(model_cfg)
    ds_names = '-'.join([dpath.name for dpath in args.ds_dir_paths])
    deconly_str = 't' if args.train_dec_only_bool else 'f'
    suffix = f'{ds_names}-{suffix}-tdec_{deconly_str}'
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
        chkpt_model_cfg = parse_yaml_file_as(RankerHgCfg, train_path / RANKER_HG_MODEL_CFG_FNAME)
        assert tkz_cfg == chkpt_tkz_cfg, f'{args.tokenizer_cfg_fpath} != {chkpt_tkz_cfg}'
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / TOKENIZER_CFG_FNAME, tkz_cfg)
        to_yaml_file(train_path / RANKER_HG_MODEL_CFG_FNAME, model_cfg)

    tokenizer = tokenizer_from_config(tkz_cfg)
    tok_dict = tkz_cfg.custom_tokens
    ch_tkz = ChunkTokenizer(tok_dict, tokenizer, n_emb_tokens=args.inp_len, fixed_size=True)
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind
    ds = load_qrels_datasets(args.ds_dir_paths, ch_tkz, args.inp_len, device)
    print(ds)

    print(f'Creating model with vocab size = {len(tokenizer)}')

    # torch.autograd.set_detect_anomaly(True)

    model = RankerHg(model_cfg).to(device)

    if args.pretrained_model_path and (args.pretrained_model_path / 'best.pth').exists() and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path)
        model_encdec_cfg_fpath = args.pretrained_model_path / ENCDEC_HG_MODEL_CFG_FNAME
        model_encdec_cfg = parse_yaml_file_as(EncdecHgCfg, model_encdec_cfg_fpath)
        model_encdec = EncdecHg(model_encdec_cfg).to(device)
        model_encdec.load_state_dict(pretrained_checkpoint['model'], strict=False)
        print(f'Load model weights for enc_pyr:', list(model_encdec.enc_pyr.state_dict().keys()))
        model.enc_pyr.load_state_dict(model_encdec.enc_pyr.state_dict())

    if args.train_dec_only_bool:
        params = model.dec_rank.parameters()
    else:
        params = model.parameters()
    # params = [p for n, p in model.named_parameters()]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, threshold=1e-6, min_lr=1e-7)
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    ds_view = ds.get_view_plain_qids(batch_size=args.docs_batch_size)
    ds_view.shuffle()
    view_train, view_val = ds_view.split((-1, 0.05))

    last_epoch, val_loss_min = -1, None
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        np.random.seed(int(time.time() * 1000) % 10_000_000)
        view_train.shuffle()
        view_val.shuffle()

    n_batches_train, n_batches_val = calc_print_batches(view_train, view_val, args.docs_batch_size, 'Queries')
    # loss_fn = RankProbLoss()
    # loss_fn = ranker_prob_loss_softmax
    # loss_fn = ranker_cos_loss
    loss_fn = RankerCosEmbLoss()
    n_epochs = args.epochs - (last_epoch + 1)
    train_batch_it = view_train.get_batch_iterator(
        n_batches=n_epochs * n_batches_train,
        drop_last=False,
        shuffle_between_loops=True,
    )
    val_batch_it = view_val.get_batch_iterator(
        n_batches=n_epochs * n_batches_val,
        drop_last=False,
        shuffle_between_loops=True,
    )
    model.eval()
    loss_tgt, loss_nontgt = None, None
    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    for epoch in range(last_epoch + 1, args.epochs):
        if args.train_dec_only_bool:
            for params in model.enc_pyr.parameters():
                params.requires_grad = False
            model.dec_rank.train()
        else:
            model.train()
        train_loss, train_loss_tgt, train_loss_nontgt = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch: QrelsPlainBatch = next(train_batch_it)
            qs_toks, qs_masks, docs_toks, docs_masks, qrels_masks = batch.gen_tensors()

            optimizer.zero_grad()
            out_rank = model(docs_toks, qs_toks)
            loss = loss_fn(out_rank, qrels_masks)
            if type(loss) == tuple:
                loss, loss_tgt, loss_nontgt = loss

            loss.backward()
            # Gradients must be available after loss.backward()
            if grad_log_ind % grad_log_interval == 0:
                log_weights_grads_stats(grad_log_step, model, tbsw)
                grad_log_step += 1
            grad_log_ind += 1

            optimizer.step()

            train_loss += loss.item()
            if loss_tgt is not None:
                train_loss_tgt += loss_tgt.item()
                train_loss_nontgt += loss_nontgt.item()

            s = f'Train. loss: {loss.item():.6f}'
            if loss_tgt is not None:
                s = f'{s}. loss_tgt: {loss_tgt.item():.6f}. loss_nontgt: {loss_nontgt.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)
        if loss_tgt is not None:
            train_loss_tgt /= args.train_epoch_steps
            train_loss_nontgt /= args.train_epoch_steps
            tbsw.add_scalar('LossTgt/Train', train_loss_tgt, epoch)
            tbsw.add_scalar('LossNontgt/Train', train_loss_nontgt, epoch)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        model.eval()
        val_loss, val_loss_tgt, val_loss_nontgt = 0, 0, 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch: QrelsPlainBatch = next(val_batch_it)
            qs_toks, qs_masks, docs_toks, docs_masks, qrels_masks = batch.gen_tensors()

            out_rank = model(docs_toks, qs_toks)
            loss = loss_fn(out_rank, qrels_masks)
            if type(loss) == tuple:
                loss, loss_tgt, loss_nontgt = loss

            val_loss += loss.item()
            if loss_tgt is not None:
                val_loss_tgt += loss_tgt.item()
                val_loss_nontgt += loss_nontgt.item()

            s = f'Val. loss: {loss.item():.6f}'
            if loss_tgt is not None:
                s = f'{s}. loss_tgt: {loss_tgt.item():.6f}. loss_nontgt: {loss_nontgt.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        val_loss /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)
        if loss_tgt is not None:
            val_loss_tgt /= args.val_epoch_steps
            val_loss_nontgt /= args.val_epoch_steps
            tbsw.add_scalar('LossTgt/Val', val_loss_tgt, epoch)
            tbsw.add_scalar('LossNontgt/Val', val_loss_nontgt, epoch)

        scheduler.step(val_loss)
        last_lr = scheduler.get_last_lr()[0]
        tbsw.add_scalar(f'{scheduler.__class__.__name__} lr', last_lr, epoch)

        print(f'Train loss: {train_loss:.6f}, loss_tgt: {train_loss_tgt:.6f}, loss_nontgt: {train_loss_nontgt:.6f}')
        print(f'Val loss:   {val_loss:.6f}, loss_tgt: {val_loss_tgt:.6f}, loss_nontgt: {val_loss_nontgt:.6f}')
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

    ds.close()
    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsRankerHgQrelsTrain, main, 'Train Mllm Ranking model.', exception_handler=rethrow)


