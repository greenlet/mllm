import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from pydantic import Field
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.config.model import TokenizerCfg, MllmEncdecCfg, MllmRankerCfg
from mllm.data.utils import load_qrels_datasets
from mllm.exp.args import ArgsTokensChunksTrain, TOKENIZER_CFG_FNAME, ENCDEC_MODEL_CFG_FNAME, RANKER_MODEL_CFG_FNAME
from mllm.model.mllm_encdec import MllmEncdecLevel
from mllm.model.mllm_ranker import MllmRankerLevel, RankProbLoss
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer, tokenizer_from_config
from mllm.train.utils import find_create_train_path, calc_print_batches


class ArgsQrelsTrain(ArgsTokensChunksTrain):
    ds_dir_paths: list[Path] = Field(
        [],
        required=True,
        description='Qrels datasets directory paths. Supported datasets: Msmarco, Fever.'
                    'Naming convention: directory name must contain the name of dataset: msmarco, fever. Unknown datasets '
                    'will cause an error and exit.',
        cli=('--ds-dir-paths',),
    )


def main(args: ArgsQrelsTrain) -> int:
    print(args)

    assert args.ds_dir_paths, '--ds-dir-paths is expected to list at least one Qrels datsaset'

    device = torch.device(args.device)

    ds_names = '-'.join([dpath.name for dpath in args.ds_dir_paths])
    train_path = find_create_train_path(
        args.train_root_path, 'ranker', ds_names, args.train_subdir)
    print(f'train_path: {train_path}')

    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    checkpoint = None
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists(),\
            (f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
             f'but file {last_checkpoint_path} does not exits.')

    tokenizer_cfg_fpath = args.tokenizer_cfg_fpath
    model_cfg_fpath = args.model_cfg_fpath
    if last_checkpoint_path.exists():
        print(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        print(f'Checkpoint with keys {list(checkpoint.keys())} loaded')
        tokenizer_cfg_fpath = train_path / TOKENIZER_CFG_FNAME
        model_cfg_fpath = train_path / RANKER_MODEL_CFG_FNAME
    else:
        shutil.copy(args.tokenizer_cfg_fpath, train_path / TOKENIZER_CFG_FNAME)
        shutil.copy(args.model_cfg_fpath, train_path / RANKER_MODEL_CFG_FNAME)

    print(f'Loading tokenizer config from {tokenizer_cfg_fpath}')
    tkz_cfg = parse_yaml_file_as(TokenizerCfg, tokenizer_cfg_fpath)
    tokenizer = tokenizer_from_config(tkz_cfg)

    tok_dict = tkz_cfg.custom_tokens
    ch_tkz = ChunkTokenizer(tok_dict, tokenizer, n_emb_tokens=args.emb_chunk_size, fixed_size=True)
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind

    ds = load_qrels_datasets(args.ds_dir_paths, ch_tkz, args.emb_chunk_size, device)
    print(ds)

    print(f'Creating model with vocab size = {len(tokenizer)}')

    torch.autograd.set_detect_anomaly(True)

    model_cfg = parse_yaml_file_as(MllmRankerCfg, model_cfg_fpath)
    print(model_cfg)
    model = MllmRankerLevel(model_cfg, args.model_level).to(device)

    if args.pretrained_model_path is not None and checkpoint is None:
        pretrained_model_path = args.pretrained_model_path / 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path)
        model_encdec_cfg_fpath = args.pretrained_model_path / ENCDEC_MODEL_CFG_FNAME
        model_encdec_cfg = parse_yaml_file_as(MllmEncdecCfg, model_encdec_cfg_fpath)
        model_encdec = MllmEncdecLevel(model_encdec_cfg, args.model_level).to(device)
        model_encdec.load_state_dict(pretrained_checkpoint['model'], strict=False)
        print(f'Load model weights for vocab_encoder:', list(model_encdec.vocab_encoder.state_dict().keys()))
        model.vocab_encoder.load_state_dict(model_encdec.vocab_encoder.state_dict())
        print(f'Load model weights for encoder:', list(model_encdec.encoder.state_dict().keys()))
        model.encoder.load_state_dict(model_encdec.encoder.state_dict())

    params = model.parameters()
    # params = [p for n, p in model.named_parameters()]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, threshold=1e-6, min_lr=1e-7)
    tbsw = tb.SummaryWriter(log_dir=str(train_path))

    ds_view = ds.get_view(batch_size=args.docs_batch_size)
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
    model.eval()
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()
        # model.decoder.train()
        # model.vocab_encoder.train()
        train_loss, train_loss_tgt, train_loss_nontgt = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = next(train_batch_it)
            docs_chunks, qs_chunks = batch.gen_tensors()

            optimizer.zero_grad()
            out_rank, target_mask = model.run_qs(docs_chunks, qs_chunks, batch.docs_off_len, batch.qs_off_len)
            loss, loss_tgt, loss_nontgt = loss_fn(out_rank, target_mask)
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

        model.eval()
        val_loss, val_loss_tgt, val_loss_nontgt = 0, 0, 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = next(val_batch_it)
            docs_chunks, qs_chunks = batch.gen_tensors()

            out_rank, target_mask = model.run_qs(docs_chunks, qs_chunks, batch.docs_off_len, batch.qs_off_len)
            loss, loss_tgt, loss_nontgt = loss_fn(out_rank, target_mask)

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
    run_and_exit(ArgsQrelsTrain, main, 'Train Mllm Ranking model.', exception_handler=rethrow)


