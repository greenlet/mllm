import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from torch import nn
from tqdm import trange

from mllm.data.dsfixed import DsLoader
from mllm.model.mllm_ranker import MllmRanker
from mllm.model.mllm_encdec import MllmEncdec
from mllm.model.config import create_mllm_ranker_cfg, create_mllm_encdec_cfg
from mllm.tokenization.chunk_tokenizer import calc_max_inp_size, gen_all_tokens
from mllm.utils.utils import gen_dt_str
from transformers import GPT2Tokenizer


class ArgsTrain(BaseModel):
    ds_dir_path: Path = Field(
        None,
        required=False,
        description='Dataset directory path. Must contain .csv and .np files with tokenized text.',
        cli=('--ds-dir-path',),
    )
    train_root_path: Path = Field(
        ...,
        required=True,
        description='Path to train root directory. New train subdirectory will be created within each new run.',
        cli=('--train-root-path',),
    )
    docs_batch_size: Optional[int] = Field(
        3,
        required=False,
        description='Documents batch size. Must be greater or equal than 2.',
        cli=('--docs-batch-size',),
    )
    max_chunks_per_doc: Optional[int] = Field(
        3,
        required=False,
        description='Maximum number of consecutive chunks per document taken in each butch. '
                    'Batch chunk max size will be DOCS_BATCH_SIZE * MAX_CHUNKS_PER_DOC.',
        cli=('--max-chunks-per-doc',),
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
        description='Path to pretrained model weigths. Encoder weights will be utilized from it.',
        cli=('--pretrained-model-path',),
    )
    

def gen_train_subdir(ds_dir_path: Path) -> str:
    dt_str = gen_dt_str()
    subdir = f'ranker-{dt_str}-{ds_dir_path.parent.name}-{ds_dir_path.name}'
    return subdir


def rank_prob_loss(prob_pred: torch.Tensor, mask_gt: torch.Tensor, tgt_weight: float = 0.5) -> torch.Tensor:
    # prob_pred = prob_pred.squeeze(-1)
    # mask_gt = mask_gt.unsqueeze(0)
    prob_pred = prob_pred.squeeze()
    # prob_tgt, prob_nontgt = prob_pred[mask_gt], prob_pred[~mask_gt]
    prob_tgt = torch.masked_select(prob_pred, mask_gt)
    prob_nontgt = torch.masked_select(prob_pred, ~mask_gt)

    # prob_tgt, prob_nontgt = prob_tgt**2, prob_nontgt**2
    loss_tgt = 1 - torch.mean(prob_tgt)
    loss_nontgt = torch.mean(prob_nontgt)

    # print(f'loss_tgt = {loss_tgt}. loss_nontgt = {loss_nontgt}')
    # loss = tgt_weight * loss_tgt + (1 - tgt_weight) * loss_nontgt
    # loss = tgt_weight * loss_tgt + (1 - tgt_weight) * loss_nontgt
    loss = loss_tgt + loss_nontgt
    # loss = loss_tgt + loss_nontgt
    # print(loss_tgt.item(), loss_nontgt.item())
    return loss


class RankProbLoss(nn.Module):
    def __init__(self, target_weight: float = 0.5):
        super().__init__()
        self.target_weight = target_weight

    def forward(self, prob_pred: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        prob_pred = prob_pred.squeeze()
        prob_tgt = torch.masked_select(prob_pred, mask_gt)
        prob_nontgt = torch.masked_select(prob_pred, ~mask_gt)
        # loss_tgt = 1 - torch.mean(prob_tgt)
        # loss_nontgt = torch.mean(prob_nontgt)
        loss_tgt = -torch.mean(torch.log(prob_tgt))
        loss_nontgt = -torch.mean(torch.log(1 - prob_nontgt))

        # mask_gt = mask_gt.to(torch.float32)
        # prob_tgt = mask_gt * prob_pred
        # prob_nontgt = (1 - mask_gt) * prob_pred
        # n_tgt = mask_gt.sum()
        # n_nontgt = len(mask_gt) - n_tgt
        # loss_tgt = 1 - prob_tgt.sum() / n_tgt
        # loss_nontgt = prob_nontgt.sum() / n_nontgt

        loss = self.target_weight * loss_tgt + (1 - self.target_weight) * loss_nontgt
        return loss


def main(args: ArgsTrain) -> int:
    print(args)

    device = torch.device(args.device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=10000)
    tok_dict = gen_all_tokens(tokenizer)
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind
    n_total = 1000
    ds_loader = DsLoader(
        ds_dir_path=args.ds_dir_path, docs_batch_size=args.docs_batch_size, max_chunks_per_doc=args.max_chunks_per_doc,
        pad_tok=pad_tok, qbeg_tok=qbeg_tok, qend_tok=qend_tok, device=device, n_total=n_total,
    )

    train_subdir = gen_train_subdir(args.ds_dir_path)
    train_path = args.train_root_path / train_subdir
    train_path.mkdir(parents=True, exist_ok=True)
    inp_len = ds_loader.emb_chunk_size if ds_loader.fixed_size else calc_max_inp_size(ds_loader.emb_chunk_size)
    print(f'Creating model with vocab size = {len(tokenizer)}')

    torch.autograd.set_detect_anomaly(True)

    model_cfg = create_mllm_ranker_cfg(
        n_vocab=len(tokenizer), inp_len=inp_len, d_word_wec=256,
        n_levels=1, enc_n_layers=1, dec_n_layers=1,
        n_heads=8, d_k=32, d_v=32, d_model=256, d_inner=1024,
        pad_idx=pad_tok, dropout_rate=0.1, enc_with_emb_mat=True,
    )
    print(model_cfg)
    model = MllmRanker(model_cfg).to(device)

    if args.pretrained_model_path is not None:
        print(f'Loading checkpoint with pretrained model from {args.pretrained_model_path}')
        checkpoint = torch.load(args.pretrained_model_path)
        model_encdec_cfg = create_mllm_encdec_cfg(
            n_vocab=len(tokenizer), d_word_wec=256, inp_len=inp_len,
            enc_n_layers=1, dec_n_layers=1,
            n_heads=8, d_model=256, d_inner=1024,
            pad_idx=pad_tok, dropout_rate=0.1, enc_with_emb_mat=True,
        )
        model_encdec = MllmEncdec(model_encdec_cfg).to(device)
        print(f'Load model weights for vocab_encoder:', list(model_encdec.vocab_encoder.state_dict().keys()))
        model.vocab_encoder.load_state_dict(model_encdec.vocab_encoder.state_dict())
        print(f'Load model weights for encoder:', list(model_encdec.encoder.state_dict().keys()))
        model.encoders[0].load_state_dict(model_encdec.encoder.state_dict())

    params = model.parameters()
    # params = [p for n, p in model.named_parameters()]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)
    tbsw = tb.SummaryWriter(log_dir=str(train_path))
    val_loss_min = None
    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'

    calc_batches = lambda n_docs: n_docs // args.docs_batch_size + (n_docs % args.docs_batch_size > 1)
    n_batches_train = calc_batches(ds_loader.n_docs_train)
    n_batches_val = calc_batches(ds_loader.n_docs_val)
    # loss_fn = rank_prob_loss
    loss_fn = RankProbLoss()
    graph_written = True
    for epoch in range(args.epochs):
        model.train()
        # model.eval()
        train_loss = 0
        pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for i in pbar:
            i_batch = i % n_batches_train
            if i > 0 and i_batch == 0:
                ds_loader.shuffle(train=True)
            batch = ds_loader.get_batch(i_batch, train=True)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()

            optimizer.zero_grad()

            out_dec_rank = model(target_chunks, docs_chunks)
            if not graph_written:
                tbsw.add_graph(model, [target_chunks, docs_chunks], verbose=True, use_strict_trace=False)
                graph_written = True

            loss = loss_fn(out_dec_rank, target_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # if i == 2:
            #     import sys
            #     sys.exit()

            pbar.set_postfix_str(f'Train. loss: {loss.item():.6f}')
        pbar.close()
        train_loss /= args.train_epoch_steps
        tbsw.add_scalar('Loss/Train', train_loss, epoch)

        model.eval()
        val_loss = 0
        pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        for i in pbar:
            i_batch = i % n_batches_val
            if i > 0 and i_batch == 0:
                ds_loader.shuffle(train=False)
            batch = ds_loader.get_batch(i_batch, train=False)
            docs_chunks, target_chunks, target_mask = batch.gen_tensors()
            out_dec_rank = model(target_chunks, docs_chunks)

            loss = loss_fn(out_dec_rank, target_mask)
            val_loss += loss.item()

            pbar.set_postfix_str(f'Val. loss: {loss.item():.6f}')
        pbar.close()
        val_loss /= args.val_epoch_steps
        tbsw.add_scalar('Loss/Val', val_loss, epoch)

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
    run_and_exit(ArgsTrain, main, 'Train Mllm model.', exception_handler=rethrow)


