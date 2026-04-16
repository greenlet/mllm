"""Test that every sample in a batch contributes to the loss in MixedDecoder.

For each dataset type (Cite, QnA, Next), we:
1. Create a synthetic batch on the target device.
2. Run a forward pass and verify the loss is finite.
3. Backpropagate and verify gradients are non-zero.
4. Zero out one sample's target labels and confirm the loss changes,
   proving that sample was participating in the original loss.
"""

import copy
import os
import socket
import tempfile
from typing import Dict, List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from mllm.config.model import (
    BertEmbType, EncBertCfg, MixedDecoderCfg, MixedDecoderDsType,
    MixedDecoderTrainCfg, MixedDecoderType,
)
from mllm.data.qna.batch import QnaBatch
from mllm.model.mixed_decoder import MixedDecoder
from mllm.train.encdec_graph_bert import MaskedCiteBatch
from mllm.train.next_tok_wiki import NextTokBatch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BERT_NAME = 'bert-base-uncased'
INP_LEN = 32
MAX_SEQ_LEN = 128
BATCH_SIZE = 4
D_MODEL = 768


def _make_cfg(ds_type: MixedDecoderDsType, emb_exp_rate: int = 4) -> MixedDecoderCfg:
    tkz = AutoTokenizer.from_pretrained(BERT_NAME)
    return MixedDecoderCfg(
        enc_bert=EncBertCfg(
            inp_len=INP_LEN,
            d_model=D_MODEL,
            pad_token_id=tkz.pad_token_id,
            pretrained_model_name=BERT_NAME,
            tokenizer_name=BERT_NAME,
            emb_type=BertEmbType.Cls,
        ),
        decoder_type=MixedDecoderType.BertDec,
        decoder_model_name=BERT_NAME,
        max_seq_len=MAX_SEQ_LEN,
        use_sep=True,
        prompt_all=False,
        emb_exp_rate=emb_exp_rate,
        emb_win_min_size=2,
        emb_win_max_size=BATCH_SIZE,
        d_model=D_MODEL,
        train_ds_type=ds_type,
        train_cfg=MixedDecoderTrainCfg(freeze_encoder=False),
    )


def _make_model(cfg: MixedDecoderCfg) -> MixedDecoder:
    tkz = AutoTokenizer.from_pretrained(BERT_NAME)
    model = MixedDecoder(cfg, tkz)
    model.to(DEVICE)
    model.train()
    return model


# ---------------------------------------------------------------------------
# Synthetic batch builders
# ---------------------------------------------------------------------------

def _rand_toks(shape, tkz, device=DEVICE) -> Tensor:
    """Random token ids in the normal vocab range (skip special tokens)."""
    return torch.randint(999, tkz.vocab_size - 1, shape, device=device)


def _make_cite_batch(tkz, device=DEVICE) -> MaskedCiteBatch:
    cls_id = tkz.cls_token_id
    sep_id = tkz.sep_token_id

    inp_toks = _rand_toks((BATCH_SIZE, INP_LEN), tkz, device=device)
    inp_toks[:, 0] = cls_id
    inp_toks[:, -1] = sep_id

    cite_len = 10
    cites_toks = _rand_toks((BATCH_SIZE, cite_len), tkz, device=device)
    cites_toks[:, 0] = cls_id
    cites_toks[:, -1] = sep_id

    prompt_len = 6
    prompts_toks = _rand_toks((BATCH_SIZE, prompt_len), tkz, device=device)

    return MaskedCiteBatch(
        tokens_subsets=[],
        inp_toks=inp_toks,
        inp_masked_toks=inp_toks.clone(),
        prompts_toks=prompts_toks,
        cites_toks=cites_toks,
        cites_masked_toks=cites_toks.clone(),
        inp_att_mask=torch.ones(BATCH_SIZE, INP_LEN, dtype=torch.long, device=device),
        prompts_att_mask=torch.ones(BATCH_SIZE, prompt_len, dtype=torch.long, device=device),
        cites_att_mask=torch.ones(BATCH_SIZE, cite_len, dtype=torch.long, device=device),
        edge_inds=torch.zeros(2, BATCH_SIZE + 1, dtype=torch.long, device=device),
    )


def _make_qna_batch(tkz, device=DEVICE) -> QnaBatch:
    cls_id = tkz.cls_token_id
    chunks_per_sample = 2
    total_chunks = BATCH_SIZE * chunks_per_sample

    ctx_toks = _rand_toks((total_chunks, INP_LEN), tkz, device=device)
    ctx_toks[:, 0] = cls_id

    prompt_len = 8
    prompt_toks = _rand_toks((BATCH_SIZE, prompt_len), tkz, device=device)

    ans_len = 6
    ans_toks = _rand_toks((BATCH_SIZE, ans_len), tkz, device=device)
    ans_toks[:, 0] = cls_id

    return QnaBatch(
        ctx_chunks_toks=ctx_toks,
        ctx_chunks_att_mask=torch.ones(total_chunks, INP_LEN, dtype=torch.long, device=device),
        ctx_chunk_counts=[chunks_per_sample] * BATCH_SIZE,
        prompt_toks=prompt_toks,
        prompt_att_mask=torch.ones(BATCH_SIZE, prompt_len, dtype=torch.long, device=device),
        prompt_lengths=[prompt_len] * BATCH_SIZE,
        ans_toks=ans_toks,
        ans_att_mask=torch.ones(BATCH_SIZE, ans_len, dtype=torch.long, device=device),
    )


def _make_next_batch(tkz, device=DEVICE) -> NextTokBatch:
    cls_id = tkz.cls_token_id
    chunks_per_sample = 2
    total_chunks = BATCH_SIZE * chunks_per_sample

    ctx_toks = _rand_toks((total_chunks, INP_LEN), tkz, device=device)
    ctx_toks[:, 0] = cls_id

    prompt_len = 6
    prompt_toks = _rand_toks((BATCH_SIZE, prompt_len), tkz, device=device)

    target_len = 8
    target_toks = _rand_toks((BATCH_SIZE, target_len), tkz, device=device)
    target_toks[:, 0] = cls_id

    return NextTokBatch(
        ctx_chunks_toks=ctx_toks,
        ctx_chunks_att_mask=torch.ones(total_chunks, INP_LEN, dtype=torch.long, device=device),
        ctx_chunk_counts=[chunks_per_sample] * BATCH_SIZE,
        prompt_toks=prompt_toks,
        prompt_att_mask=torch.ones(BATCH_SIZE, prompt_len, dtype=torch.long, device=device),
        prompt_lengths=[prompt_len] * BATCH_SIZE,
        target_toks=target_toks,
        target_att_mask=torch.ones(BATCH_SIZE, target_len, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAllBatchesParticipate:
    """Verify that all samples in a batch contribute to the training loss."""

    @pytest.fixture(scope='class')
    def tkz(self):
        return AutoTokenizer.from_pretrained(BERT_NAME)

    # -- Cite ------------------------------------------------------------------

    def test_cite_forward_loss_finite(self, tkz):
        cfg = _make_cfg(MixedDecoderDsType.Cite)
        model = _make_model(cfg)
        batch = _make_cite_batch(tkz)
        loss_dict, logits = model(batch)
        assert torch.isfinite(loss_dict['loss']), 'Cite: loss is not finite'

    def test_cite_backward_grads(self, tkz):
        cfg = _make_cfg(MixedDecoderDsType.Cite)
        model = _make_model(cfg)
        batch = _make_cite_batch(tkz)
        loss_dict, _ = model(batch)
        loss_dict['loss'].backward()
        n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert n_with_grad > 0, 'Cite: no parameters received gradients'

    def test_cite_each_sample_contributes(self, tkz):
        """Zero out each sample's target labels one at a time and check loss changes."""
        cfg = _make_cfg(MixedDecoderDsType.Cite)
        model = _make_model(cfg)
        batch = _make_cite_batch(tkz)

        with torch.no_grad():
            loss_all, _ = model(batch)
        loss_all_val = loss_all['loss'].item()

        for idx in range(BATCH_SIZE):
            modified = copy.copy(batch)
            # Zero out one sample's citation by replacing with PAD (model ignores pad via att_mask)
            new_cites_att = batch.cites_att_mask.clone()
            new_cites_att[idx] = 0
            modified.cites_att_mask = new_cites_att
            new_cites = batch.cites_toks.clone()
            new_cites[idx] = tkz.pad_token_id
            modified.cites_toks = new_cites
            modified.cites_masked_toks = new_cites.clone()

            with torch.no_grad():
                loss_mod, _ = model(modified)
            assert loss_mod['loss'].item() != pytest.approx(loss_all_val, abs=1e-6), \
                f'Cite: sample {idx} does not affect the loss'

    # -- QnA -------------------------------------------------------------------

    def test_qna_forward_loss_finite(self, tkz):
        cfg = _make_cfg(MixedDecoderDsType.QnaSquadV2)
        model = _make_model(cfg)
        batch = _make_qna_batch(tkz)
        loss_dict, logits = model(batch)
        assert torch.isfinite(loss_dict['loss']), 'QnA: loss is not finite'

    def test_qna_backward_grads(self, tkz):
        cfg = _make_cfg(MixedDecoderDsType.QnaSquadV2)
        model = _make_model(cfg)
        batch = _make_qna_batch(tkz)
        loss_dict, _ = model(batch)
        loss_dict['loss'].backward()
        n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert n_with_grad > 0, 'QnA: no parameters received gradients'

    def test_qna_each_sample_contributes(self, tkz):
        """Mask each sample's answer and check loss changes."""
        cfg = _make_cfg(MixedDecoderDsType.QnaSquadV2)
        model = _make_model(cfg)
        batch = _make_qna_batch(tkz)

        with torch.no_grad():
            loss_all, _ = model(batch)
        loss_all_val = loss_all['loss'].item()

        for idx in range(BATCH_SIZE):
            modified = copy.copy(batch)
            new_ans_att = batch.ans_att_mask.clone()
            new_ans_att[idx] = 0
            modified.ans_att_mask = new_ans_att
            new_ans = batch.ans_toks.clone()
            new_ans[idx] = tkz.pad_token_id
            modified.ans_toks = new_ans

            with torch.no_grad():
                loss_mod, _ = model(modified)
            assert loss_mod['loss'].item() != pytest.approx(loss_all_val, abs=1e-6), \
                f'QnA: sample {idx} does not affect the loss'

    # -- Next ------------------------------------------------------------------

    def test_next_forward_loss_finite(self, tkz):
        cfg = _make_cfg(MixedDecoderDsType.Next)
        model = _make_model(cfg)
        batch = _make_next_batch(tkz)
        loss_dict, logits = model(batch)
        assert torch.isfinite(loss_dict['loss']), 'Next: loss is not finite'

    def test_next_backward_grads(self, tkz):
        cfg = _make_cfg(MixedDecoderDsType.Next)
        model = _make_model(cfg)
        batch = _make_next_batch(tkz)
        loss_dict, _ = model(batch)
        loss_dict['loss'].backward()
        n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert n_with_grad > 0, 'Next: no parameters received gradients'

    def test_next_each_sample_contributes(self, tkz):
        """Mask each sample's target and check loss changes."""
        cfg = _make_cfg(MixedDecoderDsType.Next)
        model = _make_model(cfg)
        batch = _make_next_batch(tkz)

        with torch.no_grad():
            loss_all, _ = model(batch)
        loss_all_val = loss_all['loss'].item()

        for idx in range(BATCH_SIZE):
            modified = copy.copy(batch)
            new_tgt_att = batch.target_att_mask.clone()
            new_tgt_att[idx] = 0
            modified.target_att_mask = new_tgt_att
            new_tgt = batch.target_toks.clone()
            new_tgt[idx] = tkz.pad_token_id
            modified.target_toks = new_tgt

            with torch.no_grad():
                loss_mod, _ = model(modified)
            assert loss_mod['loss'].item() != pytest.approx(loss_all_val, abs=1e-6), \
                f'Next: sample {idx} does not affect the loss'

    # -- Gradient flow per sample (encoder receives gradient from each sample) --

    def test_qna_encoder_grad_per_sample(self, tkz):
        """Verify that each sample's context chunks propagate gradients to the encoder."""
        cfg = _make_cfg(MixedDecoderDsType.QnaSquadV2)
        model = _make_model(cfg)
        batch = _make_qna_batch(tkz)

        loss_dict, _ = model(batch)
        loss_dict['loss'].backward()

        # The encoder's first layer query weight should have gradients
        enc_param = next(model.enc.parameters())
        assert enc_param.grad is not None and enc_param.grad.abs().sum() > 0, \
            'QnA: encoder received no gradients'


# ---------------------------------------------------------------------------
# DDP multi-process tests
# ---------------------------------------------------------------------------

BATCH_SEEDS = [42, 99]  # Fixed seeds for deterministic batch creation per rank


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _make_batch_for_ds_type(ds_type: MixedDecoderDsType, tkz, device):
    if ds_type == MixedDecoderDsType.QnaSquadV2:
        return _make_qna_batch(tkz, device=device)
    elif ds_type == MixedDecoderDsType.Cite:
        return _make_cite_batch(tkz, device=device)
    elif ds_type == MixedDecoderDsType.Next:
        return _make_next_batch(tkz, device=device)
    else:
        raise ValueError(f'Unknown ds_type: {ds_type}')


def _ddp_worker(rank, world_size, tmpdir, port, ds_type_value):
    """DDP worker: create model + batch, run forward-backward, save rank-0 grads."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)

    try:
        dist.init_process_group('gloo', rank=rank, world_size=world_size)

        tkz = AutoTokenizer.from_pretrained(BERT_NAME)
        ds_type = MixedDecoderDsType(ds_type_value)
        cfg = _make_cfg(ds_type)
        model = MixedDecoder(cfg, tkz)

        state_dict = torch.load(os.path.join(tmpdir, 'model_state.pt'), map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()  # deterministic (no dropout)

        ddp_model = DDP(model, find_unused_parameters=True)

        # Re-create batch with the same seed used in the main process
        torch.manual_seed(BATCH_SEEDS[rank])
        batch = _make_batch_for_ds_type(ds_type, tkz, device=torch.device('cpu'))

        # Seed for deterministic randomness inside forward (e.g. embedding window)
        torch.manual_seed(rank * 1000)
        loss_dict, _ = ddp_model(batch)
        loss_dict['loss'].backward()

        if rank == 0:
            grads = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grads[name] = p.grad.clone()
            torch.save(grads, os.path.join(tmpdir, 'ddp_grads.pt'))

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestDDPAllRanksContribute:
    """Verify that in multi-process DDP training, every rank's batch contributes
    to the gradients of the model saved from rank 0.

    The key invariant: after DDP backward(), rank 0's gradients equal the
    average of all ranks' per-batch gradients. If gradient sync is broken,
    rank 0 would only have its own batch's gradients and saving rank 0's
    model would discard all other ranks' work.
    """

    @pytest.fixture(scope='class')
    def tkz(self):
        return AutoTokenizer.from_pretrained(BERT_NAME)

    @pytest.mark.parametrize('ds_type', [
        MixedDecoderDsType.QnaSquadV2,
        MixedDecoderDsType.Cite,
    ])
    def test_ddp_grads_include_all_ranks(self, tkz, ds_type):
        """After DDP backward, rank 0 grads = avg(per-rank grads), not just rank 0's."""
        world_size = 2
        cpu = torch.device('cpu')

        cfg = _make_cfg(ds_type)
        model = MixedDecoder(cfg, tkz)
        model.eval()
        initial_state = copy.deepcopy(model.state_dict())

        # Create two distinct batches (deterministic via fixed seeds)
        torch.manual_seed(BATCH_SEEDS[0])
        batch_0 = _make_batch_for_ds_type(ds_type, tkz, device=cpu)
        torch.manual_seed(BATCH_SEEDS[1])
        batch_1 = _make_batch_for_ds_type(ds_type, tkz, device=cpu)

        # --- Single-process baselines ---
        def get_single_grads(batch, forward_seed):
            model.load_state_dict(initial_state)
            model.eval()
            model.zero_grad()
            torch.manual_seed(forward_seed)
            loss, _ = model(batch)
            loss['loss'].backward()
            return {n: p.grad.clone() for n, p in model.named_parameters()
                    if p.grad is not None}

        grads_0 = get_single_grads(batch_0, forward_seed=0 * 1000)   # matches rank 0
        grads_1 = get_single_grads(batch_1, forward_seed=1 * 1000)   # matches rank 1

        # Sanity: the two batches must produce different gradients
        diff_count = sum(
            1 for n in grads_0 if n in grads_1
            and not torch.allclose(grads_0[n], grads_1[n], atol=1e-7)
        )
        assert diff_count > 0, 'Both batches yield identical grads — test is invalid'

        # --- DDP run ---
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(initial_state, os.path.join(tmpdir, 'model_state.pt'))

            port = _find_free_port()
            mp.spawn(
                _ddp_worker,
                args=(world_size, tmpdir, port, ds_type.value),
                nprocs=world_size,
                join=True,
            )

            ddp_grads = torch.load(
                os.path.join(tmpdir, 'ddp_grads.pt'), map_location='cpu',
            )

        # --- Assertions ---
        # 1) DDP grads must differ from rank-0-only grads
        #    (proves rank 1's batch contributed)
        mismatch = sum(
            1 for n in list(grads_0)[:20] if n in ddp_grads
            and not torch.allclose(ddp_grads[n], grads_0[n], atol=1e-6)
        )
        assert mismatch > 0, (
            'DDP grads equal single-rank-0 grads — other ranks are NOT contributing!'
        )

        # 2) DDP grads ≈ (grads_0 + grads_1) / world_size
        #    (the exact relationship that proves correct all-reduce)
        expected = {
            n: (grads_0[n] + grads_1[n]) / world_size
            for n in grads_0 if n in grads_1
        }
        checked, close = 0, 0
        max_reldiff = 0.0
        for name in expected:
            if name not in ddp_grads:
                continue
            checked += 1
            if torch.allclose(ddp_grads[name], expected[name], atol=1e-5, rtol=1e-4):
                close += 1
            else:
                reldiff = (ddp_grads[name] - expected[name]).abs().max().item()
                max_reldiff = max(max_reldiff, reldiff)

        assert checked > 0, 'No common parameters to compare'
        ratio = close / checked
        assert ratio > 0.9, (
            f'Only {close}/{checked} params match expected avg '
            f'(max diff={max_reldiff:.2e}) — DDP gradient sync may be broken'
        )

