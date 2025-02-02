import shutil
import time
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import torch
import torch.utils.tensorboard as tb
from pydantic import Field, BaseModel
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from transformers import AutoTokenizer

from mllm.config.model import TokenizerCfg, EncdecHgCfg, HgReductType, HgEnhanceType, PosEncType, RankerHgCfg, \
    gen_prefpostfix_ranker_hg, copy_override_ranker_hg_cfg, BertEmbType, RankerBertCfg, copy_override_ranker_bert_cfg, \
    gen_prefpostfix_ranker_bert
from mllm.data.common import DsView
from mllm.data.dsqrels import QrelsPlainBatch, DsQrels
from mllm.data.utils import load_qrels_datasets
from mllm.exp.args import TOKENIZER_CFG_FNAME, ARG_TRUE_VALUES_STR, ARG_FALSE_VALUES_STR, is_arg_true, \
    ENCDEC_HG_MODEL_CFG_FNAME, RANKER_HG_MODEL_CFG_FNAME, RANKER_BERT_MODEL_CFG_FNAME
from mllm.model.encdec_ranker_hg import EncdecHg, RankerHg, RankerBert
from mllm.model.losses import RankerCosEmbLoss
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer, tokenizer_from_config, gen_all_tokens
from mllm.train.utils import find_create_train_path, log_weights_grads_stats
from mllm.utils.utils import reraise


class ArgsRankerBertQrelsTrain(BaseModel):
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
    train_subdir: str = Field(
        '',
        description='Train subdirectory. Can have values: "last", "<subdirectory-name>". When set to "last", '
            'last subdirectory of TRAIN_ROOT_PATH containing training snapshot will be taken.',
        cli=('--train-subdir',)
    )
    model_cfg_fpath: Path = Field(
        ...,
        description='Path to RankerBert model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    inp_len: int = Field(
        ...,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    bert_emb_type: BertEmbType = Field(
        BertEmbType.Cls,
        description=f'Bert embedding type. Can have values: {list(x.value for x in BertEmbType)}',
        cli=('--bert-emb-type',),
    )
    dec_mlp_layers: str = Field(
        '',
        description=f'Consecutive dense layers\' sizes and activation functions delimited with a comma transforming initial embedding to a relevance vector. '
                    f'Examples: "512,relu,1024" - no activation in the end, "512,tanh,512,tanh" - both layers have activations. Adding suffix "b" to a layer '
                    f'dimension will add bias to that layer. Absense of suffix means no bias. Example: "1024b",tanh,512b - both layers have biases. '
                    f'Adding "norm" to the list will add layer normalization.',
        cli=('--dec-mlp-layers',)
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
    pretrained_model_path: Optional[Path] = Field(
        None,
        description='Path to EncdecBert model train directory.',
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

    random_seed: Optional[int] = Field(
        None,
        description='Random seed.',
        cli=('--random-seed',),
    )


BatchView = DsView[DsQrels, QrelsPlainBatch]
BatchIt = Generator[QrelsPlainBatch, None, None]

def agg_batch_it(iterators: list[BatchIt], n_batches: np.ndarray) -> BatchIt:
    inds = np.zeros_like(n_batches)
    i, n = 0, len(iterators)
    while True:
        it = iterators[i]
        if i == 0:
            yield next(it)
            inds[i] += 1
            i = (i + 1) % n
        elif (inds[i] + 1) / n_batches[i] <= inds[0] / n_batches[0]:
            yield next(it)
            inds[i] += 1
        else:
            i = (i + 1) % n


def get_batch_iterators(views_train: list[BatchView], views_val: list[BatchView], n_epochs: int, batch_size: int) \
        -> tuple[BatchIt, BatchIt]:
    calc_batches_num = lambda n_qs: n_qs // batch_size + min(n_qs % batch_size, 1)
    zeros = lambda: np.zeros(len(views_train), dtype=int)
    n_qs_train, n_qs_val = sum(len(v) for v in views_train), sum(len(v) for v in views_val)
    n_batches_train, n_batches_val = calc_batches_num(n_qs_train), calc_batches_num(n_qs_val)
    # Sort views by size in increasing order
    pairs = list(zip(views_train, views_val))
    pairs = sorted(pairs, key=lambda pair: len(pair[0]))
    nbt, nbv = zeros(), zeros()
    train_batch_its, val_batch_its = [], []
    for i, (view_train, view_val) in enumerate(pairs):
        nbt[i], nbv[i] = calc_batches_num(len(view_train)), calc_batches_num(len(view_val))
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
        train_batch_its.append(train_batch_it)
        val_batch_its.append(val_batch_it)

    return agg_batch_it(train_batch_its, nbt), agg_batch_it(val_batch_its, nbt)


def main(args: ArgsRankerBertQrelsTrain) -> int:
    print(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    assert args.ds_dir_paths, '--ds-dir-paths is expected to list at least one Qrels datsaset'

    device = torch.device(args.device)

    model_cfg = parse_yaml_file_as(RankerBertCfg, args.model_cfg_fpath)
    model_cfg = copy_override_ranker_bert_cfg(
        model_cfg, emb_type=args.bert_emb_type, inp_len=args.inp_len, dec_mlp_layers=args.dec_mlp_layers,
    )
    print(model_cfg)

    prefix, suffix = gen_prefpostfix_ranker_bert(model_cfg)
    ds_names = '-'.join([dpath.name for dpath in args.ds_dir_paths])
    deconly_str = 't' if args.train_dec_only_bool else 'f'
    suffix = f'{ds_names}-{suffix}-tdo_{deconly_str}'
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
        chkpt_model_cfg = parse_yaml_file_as(RankerBertCfg, train_path / RANKER_BERT_MODEL_CFG_FNAME)
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / RANKER_BERT_MODEL_CFG_FNAME, model_cfg)

    tkz = AutoTokenizer.from_pretrained(model_cfg.enc_bert.pretrained_model_name)
    print(tkz)
    custom_tokens = gen_all_tokens()
    ch_tkz = ChunkTokenizer(custom_tokens, tkz, n_emb_tokens=args.inp_len, fixed_size=True)
    dss = load_qrels_datasets(args.ds_dir_paths, ch_tkz, args.inp_len, device, join=False)
    for ds in dss:
        print(ds)

    # torch.autograd.set_detect_anomaly(True)

    model = RankerBert(model_cfg).to(device)

    if args.pretrained_model_path and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path)
        model_encdec_cfg_fpath = args.pretrained_model_path / ENCDEC_HG_MODEL_CFG_FNAME
        model_encdec_cfg = parse_yaml_file_as(EncdecHgCfg, model_encdec_cfg_fpath)
        model_cfg.enc_pyr.temperature = model_encdec_cfg.enc_pyr.temperature
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

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6, min_lr=1e-8)
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    views_train, views_val = [], []
    for ds in dss:
        ds_view = ds.get_view_plain_qids(batch_size=args.docs_batch_size)
        ds_view.shuffle(seed=777)
        view_train, view_val = ds_view.split((-1, 0.05))
        view_train.shuffle()
        view_val.shuffle()
        views_train.append(view_train)
        views_val.append(view_val)

    last_epoch, val_loss_min = -1, None
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        np.random.seed(int(time.time() * 1000) % 10_000_000)

    # loss_fn = RankProbLoss()
    # loss_fn = ranker_prob_loss_softmax
    # loss_fn = ranker_cos_loss
    loss_fn = RankerCosEmbLoss()
    n_epochs = args.epochs - (last_epoch + 1)

    train_batch_it, val_batch_it = get_batch_iterators(views_train, views_val, n_epochs, n_epochs)

    model.eval()
    assert args.train_epoch_steps is not None
    loss_tgt, loss_nontgt = None, None
    grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
    prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
    if prev_train_steps > 0:
        grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1
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

    [ds.close() for ds in dss]
    return 0


if __name__ == '__main__':
    run_and_exit(ArgsRankerBertQrelsTrain, main, 'Train RankerBert model.', exception_handler=reraise)


