import shutil
import time
from email.policy import strict
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
import torch.utils.tensorboard as tb
from pydantic import Field, BaseModel
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.config.model import TokenizerCfg, MllmEncdecCfg, MllmRankerCfg, gen_prefpostfix_level, \
    copy_override_encdec_hg_cfg, EncdecHgCfg, HgReductType, HgEnhanceType, PosEncType, RankerHgCfg, \
    gen_prefpostfix_ranker_hg, copy_override_ranker_hg_cfg, DecRankType
from mllm.data.utils import load_qrels_datasets
from mllm.exp.args import ArgsTokensChunksTrain, TOKENIZER_CFG_FNAME, ENCDEC_MODEL_CFG_FNAME, RANKER_MODEL_CFG_FNAME
from mllm.model.encdec_ranker_hg import EncdecHg
from mllm.model.mllm_encdec import MllmEncdecLevel
from mllm.model.mllm_ranker import MllmRankerLevel
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer, tokenizer_from_config
from mllm.train.utils import find_create_train_path, calc_print_batches


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
    dec_type: DecRankType = Field(
        DecRankType.Simple,
        required=False,
        description=
        f'Hourglass ranker decoder type: {list(x.value for x in DecRankType)}. {DecRankType.Simple} - '
        f'Simple linear transform of encoder embeddings. {DecRankType.Trans} - Transformer decoder ranker.',
        cli=('--dec-type',),
    )
    dec_n_layers: int = Field(
        -1,
        required=False,
        description=f'Decoder number of layers for DEC_TYPE={DecRankType.Trans}. If not set the value from encoder config will be used.',
        cli=('--dec-n-layers',),
    )
    dec_dropout_rate: float = Field(
        -1,
        required=False,
        description=f'Decoder dropout rate. If not set the value from encoder config will be used.',
        cli=('--dec-dropout-rate',),
    )
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


def main(args: ArgsRankerHgQrelsTrain) -> int:
    print(args)

    assert args.ds_dir_paths, '--ds-dir-paths is expected to list at least one Qrels datsaset'

    device = torch.device(args.device)
    import sys
    sys.exit(0)

    tkz_cfg = parse_yaml_file_as(TokenizerCfg, args.tokenizer_cfg_fpath)
    model_cfg = parse_yaml_file_as(RankerHgCfg, args.model_cfg_fpath)
    model_cfg = copy_override_ranker_hg_cfg(
        model_cfg, inp_len=args.inp_len, n_similar_layers=args.n_similar_layers, reduct_type=args.reduct_type,
        pos_enc_type=args.pos_enc_type, dec_type=args.de
    )
    print(model_cfg)

    prefix, suffix = gen_prefpostfix_level(model_cfg, args.model_level)
    ds_names = '-'.join([dpath.name for dpath in args.ds_dir_paths])
    suffix = f'{ds_names}-{suffix}'
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
        chkpt_model_cfg = parse_yaml_file_as(MllmEncdecCfg, train_path / ENCDEC_MODEL_CFG_FNAME)
        assert tkz_cfg == chkpt_tkz_cfg, f'{args.tokenizer_cfg_fpath} != {chkpt_tkz_cfg}'
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        to_yaml_file(train_path / TOKENIZER_CFG_FNAME, tkz_cfg)
        to_yaml_file(train_path / RANKER_MODEL_CFG_FNAME, model_cfg)

    tokenizer = tokenizer_from_config(tkz_cfg)

    tok_dict = tkz_cfg.custom_tokens
    ch_tkz = ChunkTokenizer(tok_dict, tokenizer, n_emb_tokens=args.emb_chunk_size, fixed_size=True)
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind

    ds = load_qrels_datasets(args.ds_dir_paths, ch_tkz, args.emb_chunk_size, device)
    print(ds)

    print(f'Creating model with vocab size = {len(tokenizer)}')

    torch.autograd.set_detect_anomaly(True)

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
        print(f'Load model weights for decoder:', list(model_encdec.decoder.state_dict().keys()))
        model.decoder.load_state_dict(model_encdec.decoder.state_dict(), strict=False)

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
    # loss_fn = ranker_prob_loss_softmax
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
    for epoch in range(last_epoch + 1, args.epochs):
        # model.train()
        model.decoder.train()
        train_loss, train_loss_tgt, train_loss_nontgt = 0, 0, 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for _ in pbar:
            batch = next(train_batch_it)
            docs_chunks, qs_chunks = batch.gen_tensors()

            optimizer.zero_grad()
            out_rank, target_mask = model.run_qs(docs_chunks, qs_chunks, batch.docs_off_len, batch.qs_off_len)
            loss = loss_fn(out_rank, target_mask)
            if type(loss) == tuple:
                loss, loss_tgt, loss_nontgt = loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if loss_tgt is not None:
                train_loss_tgt += loss_tgt.item()
                train_loss_nontgt += loss_nontgt.item()

            # if i == 2:
            #     import sys
            #     sys.exit()

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
            batch = next(val_batch_it)
            docs_chunks, qs_chunks = batch.gen_tensors()

            out_rank, target_mask = model.run_qs(docs_chunks, qs_chunks, batch.docs_off_len, batch.qs_off_len)
            loss = loss_fn(out_rank, target_mask)
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


