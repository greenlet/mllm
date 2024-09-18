import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.config.model import MllmEncdecCfg
from mllm.data.dsqrels_embs import DsQrelsEmbs, QrelsDocsEmbsBatch
from mllm.model.mllm_encdec import MllmEncdecLevel
from mllm.train.utils import find_create_train_path, calc_print_batches


class ArgsTrainEncdecEmbs(BaseModel):
    ds_dir_path: Path = Field(
        None,
        required=False,
        description='Embeddings dataset path. Must contain docs_embs.npy, docs_ids.tsv, qs_embs.npy, qs_ids.tsv files with'
                    'Embeddings generated from previous step and doc/query ids corresponding to embeddings.',
        cli=('--ds-dir-path',),
    )
    ds_dir_paths: list[Path] = Field(
        [],
        required=True,
        description='Qrels datasets directory paths. Supported datasets: Msmarco, Fever.'
                    'Naming convention: directory name must contain the name of dataset: msmarco, fever. Unknown datasets '
                    'will cause an error and exit.',
        cli=('--ds-dir-paths',),
    )
    model_cfg_fpath: Path = Field(
        ...,
        required=True,
        description='Path to ranker model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    model_level: int = Field(
        ...,
        required=True,
        description='Model level. 0 - start from tokens and produce embeddins_0. k - start from embeddings from level k - 1 '
                    'and produce embeddings_k.',
        cli=('--model-level',),
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
    batch_size: int = Field(
        3,
        required=False,
        description='Embeddings batch size. Must be greater or equal than 2.',
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


# chunks: input token chunks of the shape [n_batch, chunk_size, emb_size]
def zero_random_embs(embs: torch.Tensor, rem_ratio: float = 0.1) -> torch.Tensor:
    p = rem_ratio
    mask = torch.distributions.Bernoulli(probs=p).sample(embs.shape[:2]).to(embs.device)
    res = embs.clone()
    res[mask.bool()] = 0
    return res


# embs_pred [n_batch, seq_len, emb_size] float32 - embeddings sequence predicted by model
# embs_gt [n_batch, seq_len, emb_size] float32 - ground truth embeddings
# mask_zeros [n_batch, seq_len, 1] bool - mask which set to True for input embeddings set to 0 and False for
# input embeddings remained intact
def encdec_embs_loss_cos_masked(embs_pred: torch.Tensor, embs_gt: torch.Tensor, mask_zeros: torch.Tensor) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [n_batch, seq_len, 1]
    cos_dist = F.cosine_similarity(embs_pred, embs_gt, dim=-1).unsqueeze(-1)
    cos_dist = torch.abs(cos_dist)
    loss_zeros = torch.mean(cos_dist[mask_zeros])
    loss_nonzeros = torch.mean(cos_dist[~mask_zeros])
    loss = loss_zeros + loss_nonzeros
    return loss_zeros, loss_nonzeros, loss


# embs_pred [n_batch, seq_len, emb_size] float32 - embeddings sequence predicted by model
# embs_gt [n_batch, seq_len, emb_size] float32 - ground truth embeddings
def encdec_embs_loss_cos(embs_pred: torch.Tensor, embs_gt: torch.Tensor) -> torch.Tensor:
    # [n_batch, seq_len]
    cos_dist = F.cosine_similarity(embs_pred, embs_gt, dim=-1)
    cos_dist = torch.abs(cos_dist)
    # []
    loss = torch.mean(cos_dist)
    return loss


# embs_pred [n_batch, seq_len, emb_size] float32 - embeddings sequence predicted by model
# embs_gt [n_batch, seq_len, emb_size] float32 - ground truth embeddings
def encdec_embs_loss_mse(embs_pred: torch.Tensor, embs_gt: torch.Tensor) -> torch.Tensor:
    # [n_batch, seq_len]
    diff = embs_pred - embs_gt
    # []
    loss = torch.mean(diff**2)
    return loss


def main(args: ArgsTrainEncdecEmbs) -> int:
    print(args)

    device = torch.device(args.device)

    ds_names = '-'.join([dpath.name for dpath in args.ds_dir_paths])
    train_path = find_create_train_path(
        args.train_root_path, f'encdec-l{args.model_level}', ds_names, args.train_subdir)
    print(f'train_path: {train_path}')

    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    checkpoint = None
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists(),\
            (f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
             f'but file {last_checkpoint_path} does not exits.')
        print(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        print(f'Checkpoint with keys {list(checkpoint.keys())} loaded')

    model_cfg = parse_yaml_file_as(MllmEncdecCfg, args.model_cfg_fpath)
    print(model_cfg)
    model = MllmEncdecLevel(model_cfg, args.model_level).to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    enc_cfg = model_cfg.encoders[args.model_level]
    ds = DsQrelsEmbs(
        ds_dir_path=args.ds_dir_path, chunk_size=enc_cfg.inp_len, emb_size=enc_cfg.d_model, emb_dtype=np.float32, device=device
    )
    ds_view = ds.get_docs_embs_view(args.batch_size)
    np.random.seed(12)
    ds_view.shuffle()
    view_train, view_val = ds_view.split((-1, 0.05))

    n_batches_train, n_batches_val = calc_print_batches(view_train, view_val, ds_view.batch_size, 'Embeddings')

    last_epoch, val_loss_min = -1, None
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        np.random.seed(int(time.time() * 1000))
        view_train.shuffle()
        view_val.shuffle()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, threshold=1e-6, min_lr=1e-7)
    print(f'Scheduler {scheduler.__class__.__name__} lr: {scheduler.get_last_lr()[0]:0.10f}.')
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    loss_fn = encdec_embs_loss_cos
    # loss_fn = encdec_embs_loss_mse

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
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        train_loss = 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch: QrelsDocsEmbsBatch = next(train_batch_it)
            embs = batch.get_tensor()

            optimizer.zero_grad()
            out = model(embs)
            loss = loss_fn(out, embs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            s = f'Train. loss: {loss.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)

        model.eval()
        val_loss = 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch: QrelsDocsEmbsBatch = next(val_batch_it)
            embs = batch.get_tensor()

            out = model(embs)
            loss = loss_fn(out, embs)

            val_loss += loss.item()
            s = f'Val. loss: {loss.item():.6f}'
            pbar.set_postfix_str(s)
        pbar.close()
        val_loss /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)

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
    run_and_exit(ArgsTrainEncdecEmbs, main, 'Train Mllm model.', exception_handler=rethrow)


