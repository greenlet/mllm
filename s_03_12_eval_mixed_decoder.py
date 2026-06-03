"""Per-dataset / per-split QnA evaluation for the MixedDecoder model.

Usage (system python3, no conda):
    cd /scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd
    PYTHONPATH=. python3 s_03_12_eval_mixed_decoder.py

What this script does
---------------------
1. **Dataset analysis** – prints train/val sizes, answerable-item counts
   (loaded from the noanswer cache so no re-filtering is needed), and
   basic answer-length statistics per dataset.

2. **Model evaluation** – loads the best checkpoint from the last QnA
   training run onto a *single* GPU (cuda:0) as bfloat16, then evaluates
   on a fixed number of random batches from each dataset / split and
   reports mean loss ± std and perplexity.

Cache directory
---------------
Identical to training: cache_dir = data_path, so the noanswer-filtered
index arrays are read from data/qna_noanswer_cache/ without any re-scan.
"""

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths (relative to repo root – run from the repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()
DATA_PATH = REPO_ROOT / 'data'
TRAIN_ROOT = DATA_PATH / 'train_mllm_encdec_bert'


def _resolve_latest_qnaanscite_run_dir(train_root: Path) -> Path:
    """Return the latest mixeddecoder run dir trained on dsQnaanscite."""
    candidates = sorted(
        [
            p for p in train_root.glob('mixeddecoder-*')
            if p.is_dir() and 'dsQnaanscite' in p.name
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f'No mixeddecoder run directory with dsQnaanscite found under {train_root}'
        )
    return candidates[0]


RUN_DIR = _resolve_latest_qnaanscite_run_dir(TRAIN_ROOT)
BEST_CKPT = RUN_DIR / 'best.pth'
MODEL_CFG_YAML = RUN_DIR / 'mixed_decoder_model_cfg.yaml'
CACHE_DIR = DATA_PATH  # cache subdir = DATA_PATH/qna_noanswer_cache/

# Evaluation hyper-params
BATCH_SIZE = 10        # items per forward pass (small enough for single V100)
N_EVAL_BATCHES = 80    # batches per dataset/split -> 80*10 = 800 items per split
RANDOM_SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Keep emb_win at the training value for deterministic context windows
EMB_WIN_MAX = 6
EMB_WIN_MIN = 2

# Suppress HF tokenizer length warnings
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ---------------------------------------------------------------------------
# PYTHONPATH
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports (project-internal)
# ---------------------------------------------------------------------------
import yaml
from transformers import AutoTokenizer

from mllm.config.model import MixedDecoderCfg
from mllm.data.qna.dataset import (
    QNA_DATASETS_DEFAULT,
    QnaDatasetType,
    load_qna_datasets,
)
from mllm.model.mixed_decoder import MixedDecoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(v: float, digits: int = 4) -> str:
    return f'{v:.{digits}f}'


def print_sep(char: str = '-', width: int = 72):
    print(char * width)


def load_model() -> MixedDecoder:
    """Instantiate and load the best checkpoint onto DEVICE as bf16."""
    print(f'\nLoading model config from {MODEL_CFG_YAML}')
    with open(MODEL_CFG_YAML) as f:
        cfg_dict = yaml.safe_load(f)
    model_cfg = MixedDecoderCfg(**cfg_dict)

    tkz_enc = AutoTokenizer.from_pretrained(model_cfg.enc_bert.pretrained_model_name)
    tkz_dec = AutoTokenizer.from_pretrained(model_cfg.decoder_model_name)
    tkz_enc.model_max_length = int(1e9)
    tkz_dec.model_max_length = int(1e9)
    if tkz_dec.pad_token is None:
        tkz_dec.pad_token = tkz_dec.eos_token

    print(f'Building MixedDecoder …')
    model = MixedDecoder(model_cfg, tkz_enc, tkz_dec)

    print(f'Loading checkpoint from {BEST_CKPT} …')
    ckpt = torch.load(BEST_CKPT, map_location='cpu')
    print(f'  checkpoint keys: {list(ckpt.keys())}')
    print(f'  last_epoch={ckpt.get("last_epoch")}, val_loss_min={ckpt.get("val_loss_min")}')
    model.load_pretrained(ckpt)
    del ckpt

    # Cast entire model to bf16 (matches FSDP training dtype) and move to device
    model = model.to(dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    print(f'Model on {DEVICE} as {next(model.parameters()).dtype}')
    return model, tkz_enc, tkz_dec


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _cache_npy_size(ds_cls_name: str, total_size: int) -> Optional[int]:
    """Return the number of answerable items from the noanswer cache, or None."""
    cache_path = CACHE_DIR / 'qna_noanswer_cache' / f'{ds_cls_name}_n{total_size}.npy'
    if cache_path.exists():
        arr = np.load(cache_path)
        return len(arr)
    return None


def analyze_datasets(tkz_enc, tkz_dec) -> None:
    """Print dataset sizes, answerable counts and answer-length stats."""
    print_sep('=')
    print('SECTION 1 — DATASET ANALYSIS (sizes + answer-length statistics)')
    print_sep('=')

    # Load all datasets so we can inspect them
    INP_LEN = 128
    MAX_CHUNKS = EMB_WIN_MAX

    print('\nLoading all QnA datasets (will use noanswer cache for filtered datasets) …\n')
    qna_train, qna_val = load_qna_datasets(
        tkz_enc=tkz_enc,
        tkz_dec=tkz_dec,
        inp_len=INP_LEN,
        max_chunks=MAX_CHUNKS,
        cache_dir=CACHE_DIR,
        sources=QNA_DATASETS_DEFAULT,
        exclude_noanswer=True,
    )

    ds_names = [ds_type.value for ds_type in QNA_DATASETS_DEFAULT]
    train_dss = qna_train.datasets
    val_dss = qna_val.datasets

    rng = np.random.default_rng(RANDOM_SEED)

    rows: List[dict] = []
    for name, trn, vld in zip(ds_names, train_dss, val_dss):
        n_trn = len(trn)
        n_vld = len(vld)

        # Sample up to 500 items per split to compute answer-length stats.
        # Use sub_ds.inds[i] as the row index so we sample from the FILTERED
        # (answerable) set — this is the CORRECT path used by QnaBaseDataset.get_batch
        # but NOT the QnaDatasetAgg path (which uses position indices 0..N-1 directly).
        sample_n = 500

        def _ans_stats(ds, n_sample: int) -> Tuple[float, float, float, float]:
            idxs = rng.choice(len(ds.inds), size=min(n_sample, len(ds.inds)), replace=False)
            lengths = []
            ans_count = 0
            for i in idxs:
                idx = int(ds.inds[i])  # actual HF row index from filtered set
                try:
                    _, _, answers, is_ans = ds._get_item(idx)
                    ans_toks = ds.tokenize_answer(answers[-1])
                    lengths.append(len(ans_toks))
                    if is_ans:
                        ans_count += 1
                except Exception:
                    pass
            if not lengths:
                return 0.0, 0.0, 0.0, 0.0
            a = np.array(lengths, dtype=float)
            return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

        # Also sample via AGGREGATOR path (position 0..N-1) to check answerability leakage
        def _agg_answerability(ds, n_sample: int) -> float:
            """Fraction of answerable items when accessing via position indices (aggregator path)."""
            idxs = rng.choice(len(ds.inds), size=min(n_sample, len(ds.inds)), replace=False)
            answerable = 0
            for pos in idxs:
                try:
                    _, _, answers, is_ans = ds._get_item(int(pos))  # position index, not ds.inds[pos]
                    if is_ans:
                        answerable += 1
                except Exception:
                    pass
            return answerable / len(idxs) if len(idxs) > 0 else 0.0

        trn_mean, trn_std, trn_min, trn_max = _ans_stats(trn, sample_n)
        vld_mean, vld_std, vld_min, vld_max = _ans_stats(vld, sample_n)

        # Answerability via aggregator path (position-based access)
        trn_ans_rate = _agg_answerability(trn, min(sample_n, n_trn))
        vld_ans_rate = _agg_answerability(vld, min(sample_n, n_vld))

        rows.append(dict(
            name=name,
            n_trn=n_trn, n_vld=n_vld,
            trn_ans_mean=trn_mean, trn_ans_std=trn_std,
            trn_ans_min=trn_min, trn_ans_max=trn_max,
            vld_ans_mean=vld_mean, vld_ans_std=vld_std,
            vld_ans_min=vld_min, vld_ans_max=vld_max,
            trn_ans_rate=trn_ans_rate, vld_ans_rate=vld_ans_rate,
        ))

    # Print table
    hdr = f"{'Dataset':<22} {'TrnN':>7} {'ValN':>7}  {'AnsLen(trn)mean±std':>22}  {'AnsLen(val)mean±std':>22}  {'AggAnsRate(trn/val)':>20}"
    print(hdr)
    print_sep('-')
    total_trn = total_vld = 0
    for r in rows:
        total_trn += r['n_trn']
        total_vld += r['n_vld']
        trn_str = f"{r['trn_ans_mean']:.1f}±{r['trn_ans_std']:.1f} [{r['trn_ans_min']:.0f},{r['trn_ans_max']:.0f}]"
        vld_str = f"{r['vld_ans_mean']:.1f}±{r['vld_ans_std']:.1f} [{r['vld_ans_min']:.0f},{r['vld_ans_max']:.0f}]"
        rate_str = f"{r['trn_ans_rate']:.2f}/{r['vld_ans_rate']:.2f}"
        print(f"{r['name']:<22} {r['n_trn']:>7} {r['n_vld']:>7}  {trn_str:>22}  {vld_str:>22}  {rate_str:>20}")
    print_sep('-')
    print(f"{'TOTAL':<22} {total_trn:>7} {total_vld:>7}")
    print()
    print('AggAnsRate: fraction of answerable items when accessed via position-index path')
    print('(the aggregator path used during training). <1.0 means unanswerable items leak in.')
    print()
    return qna_train, qna_val


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dataset(
    model: MixedDecoder,
    ds_agg,
    ds_name: str,
    split: str,
    n_batches: int = N_EVAL_BATCHES,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run the model on `n_batches` random batches and collect per-batch losses."""
    rng = np.random.default_rng(seed)
    inds = ds_agg.inds.copy()
    rng.shuffle(inds)

    losses: List[float] = []
    n_total = len(inds)
    if n_total == 0:
        return dict(ds_name=ds_name, split=split, n_items=0, n_batches=0,
                    mean_loss=float('nan'), std_loss=float('nan'), perplexity=float('nan'))

    # Move dataset device pointer to eval device
    for ds in ds_agg.datasets:
        ds.device = DEVICE
    ds_agg.device = DEVICE

    offset = 0
    for b in range(n_batches):
        end = min(offset + batch_size, n_total)
        batch_inds = inds[offset:end].tolist()
        if len(batch_inds) < batch_size:
            # wrap around
            need = batch_size - len(batch_inds)
            batch_inds += inds[:need].tolist()
        offset = end % n_total

        try:
            batch = ds_agg.get_batch(batch_inds)
            # Move batch tensors to device
            batch.ctx_chunks_toks = batch.ctx_chunks_toks.to(DEVICE)
            batch.ctx_chunks_att_mask = batch.ctx_chunks_att_mask.to(DEVICE)
            batch.prompt_toks = batch.prompt_toks.to(DEVICE)
            batch.prompt_att_mask = batch.prompt_att_mask.to(DEVICE)
            batch.ans_toks = batch.ans_toks.to(DEVICE)
            batch.ans_att_mask = batch.ans_att_mask.to(DEVICE)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss_dict, _ = model(batch)
            loss_val = loss_dict['loss'].item()
            if math.isfinite(loss_val):
                losses.append(loss_val)
        except Exception as e:
            print(f'  [WARN] batch {b} failed: {e}')
            continue

    if not losses:
        return dict(ds_name=ds_name, split=split, n_items=n_total, n_batches=0,
                    mean_loss=float('nan'), std_loss=float('nan'), perplexity=float('nan'))

    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))
    perplexity = math.exp(mean_loss) if mean_loss < 50 else float('inf')
    return dict(
        ds_name=ds_name, split=split, n_items=n_total,
        n_batches=len(losses),
        mean_loss=mean_loss, std_loss=std_loss, perplexity=perplexity,
    )


def run_model_eval(model: MixedDecoder, qna_train, qna_val) -> None:
    print_sep('=')
    print('SECTION 2 — MODEL EVALUATION (per-dataset / per-split loss)')
    print_sep('=')
    print(f'  device={DEVICE}, batch_size={BATCH_SIZE}, n_eval_batches={N_EVAL_BATCHES}')
    print(f'  checkpoint: {BEST_CKPT.name}')
    print()
    print('  NOTE: QnaDatasetAgg maps position indices [0..N] directly to HF row indices,')
    print('  meaning the noanswer-filtered sub_ds.inds is bypassed in the aggregator path.')
    print('  This eval replicates that same behavior so loss values are comparable to training.')
    print()

    ds_names = [ds_type.value for ds_type in QNA_DATASETS_DEFAULT]
    results: List[dict] = []

    from mllm.data.qna.dataset import QnaDatasetAgg as _Agg

    for ds_idx, name in enumerate(ds_names):
        for split_label, agg in [('train', qna_train), ('val', qna_val)]:
            sub_ds = agg.datasets[ds_idx]
            # Wrap single sub-dataset in a fresh aggregator — replicates the same
            # position-index lookup behavior as the multi-dataset training aggregator.
            single_agg = _Agg([sub_ds], device=DEVICE)

            print(f'  Evaluating {name:22s} / {split_label} ({len(single_agg):,} items) …', flush=True)
            result = evaluate_dataset(
                model, single_agg,
                ds_name=name, split=split_label,
                seed=RANDOM_SEED + ds_idx * 17 + (0 if split_label == 'train' else 1000),
            )
            results.append(result)
            print(f'    loss={result["mean_loss"]:.4f}±{result["std_loss"]:.4f}  '
                  f'ppl={result["perplexity"]:.2f}  '
                  f'({result["n_batches"]} batches)', flush=True)

    # Print summary table
    print()
    print_sep('=')
    print('SUMMARY TABLE')
    print_sep('=')
    hdr = (f"{'Dataset':<22} {'Split':<6} {'N':>8}  "
           f"{'Loss mean':>10}  {'Loss std':>9}  {'PPL':>8}")
    print(hdr)
    print_sep('-')

    trn_losses, val_losses = [], []
    for r in results:
        tag = r['split']
        print(f"{r['ds_name']:<22} {tag:<6} {r['n_items']:>8,}  "
              f"{r['mean_loss']:>10.4f}  {r['std_loss']:>9.4f}  {r['perplexity']:>8.2f}")
        if tag == 'train':
            trn_losses.append(r['mean_loss'])
        else:
            val_losses.append(r['mean_loss'])

    print_sep('-')
    if trn_losses and val_losses:
        print(f"{'OVERALL mean':22}  {'train':6} loss={np.mean(trn_losses):.4f}  "
              f"val loss={np.mean(val_losses):.4f}  "
              f"gap={np.mean(val_losses)-np.mean(trn_losses):.4f}")

    # Also highlight biggest train/val gaps per dataset
    print()
    print('Per-dataset train/val loss gap (positive = val worse):')
    print_sep('-')
    for i in range(0, len(results), 2):
        r_trn = results[i]
        r_vld = results[i + 1]
        gap = r_vld['mean_loss'] - r_trn['mean_loss']
        direction = '▲ val worse' if gap > 0.05 else ('▼ val better' if gap < -0.05 else '≈ similar')
        print(f"  {r_trn['ds_name']:<22}  gap={gap:+.4f}  {direction}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_sep('=')
    print('QnA Per-Dataset Evaluation  —  MixedDecoder (Qwen2.5-1.5B + BERT)')
    print_sep('=')
    print(f'DEVICE: {DEVICE}')
    print(f'Run dir: {RUN_DIR.name}')
    print(f'Checkpoint: {BEST_CKPT}')
    print(f'Cache dir: {CACHE_DIR}')
    print()

    model, tkz_enc, tkz_dec = load_model()

    # Section 1: dataset analysis (also loads the datasets)
    qna_train, qna_val = analyze_datasets(tkz_enc, tkz_dec)

    # Section 2: model evaluation
    run_model_eval(model, qna_train, qna_val)

    print()
    print_sep('=')
    print('Done.')


if __name__ == '__main__':
    main()
