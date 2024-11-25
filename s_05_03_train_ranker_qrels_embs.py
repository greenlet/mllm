import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.tensorboard as tb
from pydantic import Field, BaseModel
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.config.model import MllmRankerCfg, MllmEncdecCfg, gen_prefpostfix_level
from mllm.data.dsqrels_embs import DsQrelsEmbs, QrelsEmbsBatch
from mllm.exp.args import RANKER_MODEL_CFG_FNAME, ENCDEC_MODEL_CFG_FNAME
from mllm.model.mllm_encdec import MllmEncdecLevel
from mllm.model.mllm_ranker import RankProbLoss, MllmRankerLevel
from mllm.train.utils import find_create_train_path, calc_print_batches, get_dt_from_subdir


class ArgsQrelsEmbsTrain(BaseModel):
    embs_ds_dir_path: Path = Field(
        None,
        required=False,
        description='0 level embeddings dataset path. Must contain docs_embs.npy, docs_ids.tsv, qs_embs.npy, qs_ids.tsv files with'
                    'Embeddings generated from previous step and doc/query ids corresponding to embeddings.',
        cli=('--embs-ds-dir-path',),
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
    ranker_model_cfg_fpath: Optional[Path] = Field(
        None,
        required=False,
        description='Path to Ranker model config Yaml file.',
        cli=('--ranker-model-cfg-fpath',),
    )
    model_level: int = Field(
        ...,
        required=True,
        description='Model level. 0 - start from tokens and produce embeddins_0. k - start from embeddings from level k - 1 '
                    'and produce embeddings_k.',
        cli=('--model-level',),
    )
    chunk_size: int = Field(
        100,
        required=False,
        description='Number of embedding in a chunk.',
        cli=('--chunk-size',),
    )
    chunks_batch_size: int = Field(
        3,
        required=False,
        description='Embeddings chunks batch size. Must be greater or equal than 2. Each chunk will contain a number'
                    'of embeddings defined in ranker model config',
        cli=('--chunks-batch-size',),
    )
    max_docs_embs: int = Field(
        0,
        required=False,
        description='Limit for a number of embeddings per document. MAX_DOCS_EMBS <= 0 means there is no limit.',
        cli=('--max-docs-embs',),
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
    encdec_pretrained_model_path: Optional[Path] = Field(
        None,
        required=False,
        description='Path to pretrained Encdec model weights.',
        cli=('--encdec-pretrained-model-path',),
    )


def main(args: ArgsQrelsEmbsTrain) -> int:
    print(args)

    device = torch.device(args.device)

    ranker_model_cfg = parse_yaml_file_as(MllmRankerCfg, args.ranker_model_cfg_fpath)
    print(ranker_model_cfg)

    prefix, postfix = gen_prefpostfix_level(ranker_model_cfg, args.model_level)
    if args.encdec_pretrained_model_path:
        encdec_subdir_dt = get_dt_from_subdir(args.encdec_pretrained_model_path.name)
        if encdec_subdir_dt:
            postfix = f'{postfix}-encdec-{encdec_subdir_dt}'
    train_path = find_create_train_path(
        args.train_root_path, prefix, postfix, args.train_subdir)
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
    else:
        shutil.copy(args.ranker_model_cfg_fpath, train_path / RANKER_MODEL_CFG_FNAME)

    model_ranker = MllmRankerLevel(ranker_model_cfg, args.model_level).to(device)
    params = model_ranker.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    enc_cfg = model_ranker.cfg_enc

    ds = DsQrelsEmbs(
        ds_dir_path=args.embs_ds_dir_path, chunk_size=enc_cfg.inp_len, emb_size=enc_cfg.d_model, emb_dtype=np.float32,
        doc_id_driven=True, max_docs_embs=args.max_docs_embs, device=device,
    )

    if args.encdec_pretrained_model_path is not None and checkpoint is None:
        encdec_model_cfg_fpath = args.encdec_pretrained_model_path / ENCDEC_MODEL_CFG_FNAME
        encdec_model_cfg = parse_yaml_file_as(MllmEncdecCfg, encdec_model_cfg_fpath)
        encdec_model_path = args.encdec_pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {encdec_model_path}')
        pretrained_checkpoint = torch.load(encdec_model_path, map_location=device)
        model_encdec = MllmEncdecLevel(encdec_model_cfg, args.model_level).to(device)
        model_encdec.load_state_dict(pretrained_checkpoint['model'], strict=True)
        if model_ranker.vocab_encoder is not None:
            print(f'Load model weights for vocab_encoder:', list(model_encdec.vocab_encoder.state_dict().keys()))
            model_ranker.vocab_encoder.load_state_dict(model_encdec.vocab_encoder.state_dict())
        print(f'Load model weights for encoder:', list(model_encdec.encoder.state_dict().keys()))
        model_ranker.encoder.load_state_dict(model_encdec.encoder.state_dict())

    last_epoch, val_loss_min = -1, None
    if checkpoint:
        model_ranker.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, threshold=1e-4, min_lr=1e-7)
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    n_docs_per_chunk = args.chunk_size // args.max_docs_embs
    view = ds.get_embs_view(batch_size=args.chunks_batch_size * n_docs_per_chunk, with_queries=True)
    view.shuffle()
    view_train, view_val = view.split((-1, 0.05))
    n_batches_train, n_batches_val = calc_print_batches(view_train, view_val, view.batch_size, 'EmbsChunks')

    loss_fn = RankProbLoss()
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
    model_ranker.eval()
    for epoch in range(last_epoch + 1, args.epochs):
        model_ranker.decoder.train()
        # model_ranker.train()
        train_loss, train_loss_tgt, train_loss_nontgt = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            # t1 = time.time()
            batch: QrelsEmbsBatch = next(train_batch_it)
            # print(f'next_batch: {time.time() - t1:.3f}')
            # t1 = time.time()
            docs_embs = batch.get_docs_embs_tensor()
            qs_embs, qs_masks = batch.get_qs_tensors()
            # print(f'get_tensors: {time.time() - t1:.3f}')

            optimizer.zero_grad()
            # t1 = time.time()
            out_rank = model_ranker.run_qs_embs(docs_embs, qs_embs, batch.qs_ind_len)
            # print(f'run_qs_embs: {time.time() - t1:.3f}')

            loss, loss_tgt, loss_nontgt = loss_fn(out_rank, qs_masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_tgt += loss_tgt.item()
            train_loss_nontgt += loss_nontgt.item()

            # if i == 2:
            #     import sys
            #     sys.exit()

            pbar.set_postfix_str(f'Train. loss: {loss.item():.6f}. loss_tgt: {loss_tgt.item():.6f}. loss_nontgt: {loss_nontgt.item():.6f}')
        pbar.close()
        train_loss /= args.train_epoch_steps
        train_loss_tgt /= args.train_epoch_steps
        train_loss_nontgt /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)
        tbsw.add_scalar('LossTgt/Train', train_loss_tgt, epoch)
        tbsw.add_scalar('LossNontgt/Train', train_loss_nontgt, epoch)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        model_ranker.eval()
        val_loss, val_loss_tgt, val_loss_nontgt = 0, 0, 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch: QrelsEmbsBatch = next(val_batch_it)
            docs_embs = batch.get_docs_embs_tensor()
            qs_embs, qs_masks = batch.get_qs_tensors()

            out_rank = model_ranker.run_qs_embs(docs_embs, qs_embs, batch.qs_ind_len)
            loss, loss_tgt, loss_nontgt = loss_fn(out_rank, qs_masks)

            val_loss += loss.item()
            val_loss_tgt += loss_tgt.item()
            val_loss_nontgt += loss_nontgt.item()

            pbar.set_postfix_str(f'Val. loss: {loss.item():.6f}. loss_tgt: {loss_tgt.item():.6f}. loss_nontgt: {loss_nontgt.item():.6f}')
        pbar.close()
        val_loss /= args.val_epoch_steps
        val_loss_tgt /= args.val_epoch_steps
        val_loss_nontgt /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)
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
            'model': model_ranker.state_dict(),
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
    run_and_exit(ArgsQrelsEmbsTrain, main, 'Train Mllm Ranking model for embeddings.', exception_handler=rethrow)


