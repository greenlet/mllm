"""Controlled next-token perplexity comparison for MixedDecoder.

Baseline  vs  Raw Context (fine-tuned)  vs  Soft Context (fine-tuned)
----------------------------------------------------------------------
Three ways to predict the next K tokens after the SAME preceding text:

  1. **Baseline** (no MixedDecoder training at all): the stock pretrained
     decoder (``Qwen/Qwen2.5-1.5B``, never fine-tuned) reading the raw context
     tokens directly. Built as a ``decoder_only=True`` MixedDecoder wrapper
     whose checkpoint load step is skipped, so only the HF pretrained weights
     from ``AutoModelForCausalLM.from_pretrained`` are used.

  2. **Raw context** (fine-tuned, decoder-only): the N context tokens are fed
     directly to the decoder as ordinary token embeddings.  (``decoder_only = True``)

  3. **Soft context** (fine-tuned, encoder ON): N context tokens are packed into
     ``ceil(N / (inp_len - 2))`` soft tokens by the BERT encoder (one CLS
     embedding per chunk, optionally x``emb_exp_rate`` expanded), which the
     decoder consumes in place of the raw context.  (``decoder_only = False``)

All three are scored on IDENTICAL (N context tokens, K target tokens) samples
so the only differences are (a) whether the decoder was fine-tuned at all and
(b) whether the context reaches it raw or compressed into soft tokens.

Validation data is drawn from the SAME multi-source corpora used to train the
two pinned runs (``next_sources`` in their saved configs — the latest
``s_03_11_train_mixed_decoder.sh`` setting: pg19, bookcorpusopen, arxiv,
govreport, gutenberg). Since those corpora (tens of GB total) are not cached
locally, each source is streamed from the Hub and a small, deterministic,
order-based subset of documents is materialized in memory as a PROXY
validation split — NOT the exact random 95/5 held-out indices used at train
time (see ``load_streamed_source_for_next_eval``).

Usage (run from the repo root)::

    PYTHONPATH=. python3 s_03_13_eval_next_tok_ppl.py

Requirements
------------
Two trained checkpoints (see the two training commands in
``s_03_13_eval_next_tok_ppl.md``):
  * a soft-context run  (``dsNext`` + ``embEnc*`` in the run-dir name),
  * a raw-context run   (``dsNext`` + ``deco``     in the run-dir name),
trained with the SAME decoder and the SAME ``next_sources``. The baseline needs
no checkpoint — it reuses the raw run's architecture config.

Set ``SOFT_RUN_DIR`` / ``RAW_RUN_DIR`` below to pin specific runs, or leave them
as ``None`` to auto-resolve the latest matching run under ``TRAIN_ROOT``.
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

# Pin specific run directories here, or leave None to auto-resolve the latest
# matching run under TRAIN_ROOT.
SOFT_RUN_DIR: Optional[Path] = None   # soft-context: encoder ON  (embEnc* + dsNext)
RAW_RUN_DIR: Optional[Path] = None    # raw-context : decoder-only (deco  + dsNext)

# ---------------------------------------------------------------------------
# Evaluation hyper-params
# ---------------------------------------------------------------------------
# Context window sizes (number of chunks) to sweep in the pooled multi-source
# comparison. Each chunk holds ``inp_len - 2`` content tokens, so
# N_context_tokens = win_size * (inp_len - 2). 10 is the actual training window
# (``emb_win_min_size == emb_win_max_size == 10``, i.e. ``ewn10x10``) for both
# pinned runs; the rest sweep the compression/context trade-off around it.
WIN_SIZES: List[int] = [2, 4, 8, 10]
TRAIN_WIN_SIZE = 10              # matches ewn10x10 in both pinned run dir names
FIXED_TARGET_TOKS = 64           # K: decoder target tokens per sample (matches mnt64)
# Three 1.5B-decoder models are resident on one GPU at once; batch_size=8 hit
# OOM at win_size>=8 (raw-context attention is O(seq^2), seq up to ~1324 at
# win=10), silently dropping batches. 4 stays comfortably within 32GB.
BATCH_SIZE = 4                   # items per forward pass
N_EVAL_BATCHES_SWEEP = 15        # batches per window size in the pooled sweep
N_EVAL_BATCHES_SOURCE = 12       # batches per source in the per-source breakdown
RANDOM_SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Large budget so the raw (uncompressed) context always fits; only used for the
# length assertion / target truncation (Qwen uses RoPE, so pos_emb is None).
EVAL_MAX_SEQ_LEN = 8192

# Streamed validation-subset materialization (see load_streamed_source_for_next_eval).
N_STREAM_DOCS = 120       # documents materialized per source
MIN_DOC_CHARS = 8000      # cheap pre-filter so docs can fill the largest window under test

# Suppress HF tokenizer length warnings
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ---------------------------------------------------------------------------
# PYTHONPATH + project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset, load_dataset
from pydantic_yaml import parse_yaml_file_as
from transformers import AutoTokenizer

from mllm.config.model import MixedDecoderCfg
from mllm.exp.args import MIXED_DECODER_MODEL_CFG_FNAME
from mllm.model.mixed_decoder import MixedDecoder
from mllm.train.next_tok_wiki import (
    SOURCE_REGISTRY, StackedNextTokDataset, build_stacked_next_tok_datasets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_sep(char: str = '-', width: int = 78):
    print(char * width)


def _resolve_latest_next_run(train_root: Path, marker: str) -> Path:
    """Return the latest mixeddecoder run dir trained on dsNext containing ``marker``.

    ``marker`` is ``'embEnc'`` for the soft-context (encoder) run or ``'deco'``
    for the raw-context (decoder-only) run.  Only run dirs that actually contain
    a ``best.pth`` and the model-config YAML are considered.

    ``marker`` is matched against dash-delimited NAME TOKENS (skipping the fixed
    ``mixeddecoder-<timestamp>`` prefix), not a raw substring search: every run
    name starts with the literal word "mixed**deco**der", so a plain substring
    check for ``'deco'`` would false-positive-match every single run.
    """
    candidates = sorted(
        [
            p for p in train_root.glob('mixeddecoder-*')
            if p.is_dir() and 'dsNext' in p.name
            and any(tok.startswith(marker) for tok in p.name.split('-')[2:])
            and (p / 'best.pth').exists()
            and (p / MIXED_DECODER_MODEL_CFG_FNAME).exists()
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f'No dsNext mixeddecoder run with marker {marker!r} (and best.pth + '
            f'{MIXED_DECODER_MODEL_CFG_FNAME}) found under {train_root}'
        )
    return candidates[0]


def load_cfg(run_dir: Path) -> MixedDecoderCfg:
    return parse_yaml_file_as(MixedDecoderCfg, run_dir / MIXED_DECODER_MODEL_CFG_FNAME)


def build_tokenizers(cfg: MixedDecoderCfg) -> Tuple[AutoTokenizer, AutoTokenizer]:
    tkz_enc = AutoTokenizer.from_pretrained(cfg.enc_bert.pretrained_model_name)
    tkz_dec = AutoTokenizer.from_pretrained(cfg.decoder_model_name)
    tkz_enc.model_max_length = int(1e9)
    tkz_dec.model_max_length = int(1e9)
    if tkz_dec.pad_token is None:
        tkz_dec.pad_token = tkz_dec.eos_token
    return tkz_enc, tkz_dec


def load_model(run_dir: Path, cfg: MixedDecoderCfg, tkz_enc, tkz_dec) -> MixedDecoder:
    """Instantiate a MixedDecoder from ``run_dir`` and load its best checkpoint.

    The (encoder, decoder) tokenizers are passed in so both the soft-context and
    raw-context models share the exact same decoder vocabulary, keeping their
    target token ids — and therefore their perplexities — directly comparable.
    """
    best_ckpt = run_dir / 'best.pth'
    print(f'  Building MixedDecoder (decoder_only={cfg.decoder_only}) from {run_dir.name} …')
    model = MixedDecoder(cfg, tkz_enc, tkz_dec)

    print(f'  Loading checkpoint {best_ckpt} …')
    ckpt = torch.load(best_ckpt, map_location='cpu')
    print(f'    last_epoch={ckpt.get("last_epoch")}, val_loss_min={ckpt.get("val_loss_min")}')
    model.load_pretrained(ckpt)
    del ckpt

    # Raise the sequence budget so the uncompressed context always fits. This
    # only affects the length assertion / target truncation; it does not touch a
    # learned positional table for Qwen (pos_emb is None under RoPE).
    if model.pos_emb is None:
        model.cfg.max_seq_len = EVAL_MAX_SEQ_LEN

    model = model.to(dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    return model


def load_baseline_model(run_dir: Path, tkz_enc, tkz_dec) -> MixedDecoder:
    """Build the "no MixedDecoder training at all" baseline: a fresh, never
    fine-tuned decoder-only wrapper around the stock pretrained decoder.

    Reuses the RAW run's config purely for architecture knobs (decoder_only=True,
    max_seq_len, use_sep, prompt_first, decoder_model_name) via a fresh parse of
    its YAML (independent object, so mutating ``max_seq_len`` below cannot leak
    into the fine-tuned raw model). ``MixedDecoder.__init__`` already loads the
    plain HF pretrained decoder weights via ``AutoModelForCausalLM.from_pretrained``;
    we deliberately skip ``load_pretrained`` so no fine-tuned checkpoint is applied.
    """
    cfg = load_cfg(run_dir)
    assert cfg.decoder_only, 'Baseline requires a decoder_only=True config (pure Qwen, no BERT encoder).'
    print(f'  Building BASELINE MixedDecoder (decoder_only=True, pretrained-only, '
          f'architecture borrowed from {run_dir.name}) …')
    model = MixedDecoder(cfg, tkz_enc, tkz_dec)

    if model.pos_emb is None:
        model.cfg.max_seq_len = EVAL_MAX_SEQ_LEN

    model = model.to(dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    return model


def _n_soft_tokens(cfg: MixedDecoderCfg, win_size: int) -> int:
    """Number of soft tokens the decoder sees for a ``win_size``-chunk context."""
    if cfg.use_interactive_extractor:
        return win_size * max(cfg.ie_exp_rate, 1)
    if cfg.emb_exp_rate > 0:
        return win_size * cfg.emb_exp_rate
    return win_size


def load_streamed_source_for_next_eval(
        source: str, n_docs: int = N_STREAM_DOCS, min_chars: int = MIN_DOC_CHARS,
) -> Tuple[Dataset, np.ndarray, np.ndarray, str]:
    """Materialize a small, offline-friendly validation-like subset of ``source``.

    The training pipeline (``load_split_source_for_next``) downloads and locally
    splits the ENTIRE corpus (several GB per source); that is impractical here
    since these corpora are not pre-cached on this machine. Instead we stream the
    source's designated split and keep the first ``n_docs`` documents long enough
    to fill the largest context window under test.

    This is a PROXY validation subset (first-N-long-enough, not the exact random
    95/5 held-out split used at train time) — reported as such in the write-up.
    Returns a tuple shaped like ``load_split_source_for_next``'s
    ``(ds, inds_train, inds_val, text_field)`` so it drops straight into
    ``build_stacked_next_tok_datasets``.
    """
    spec = SOURCE_REGISTRY[source]
    stream = load_dataset(
        spec.hf_id, name=spec.hf_config, split=spec.split,
        streaming=True, trust_remote_code=True,
    )
    rows = []
    for row in stream:
        text = row.get(spec.text_field) or ''
        if len(text) >= min_chars:
            rows.append({spec.text_field: text})
            if len(rows) >= n_docs:
                break
    if not rows:
        raise RuntimeError(f'No documents >= {min_chars} chars found while streaming {source!r}.')
    ds = Dataset.from_list(rows)
    print(f'    Streamed {source!r} ({spec.hf_id}): materialized {len(ds)} docs (proxy val subset).')
    return ds, np.array([], dtype=np.int64), np.arange(len(ds)), spec.text_field


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _agg(losses: List[float]) -> Tuple[float, float, float]:
    if not losses:
        return float('nan'), float('nan'), float('nan')
    mean = float(np.mean(losses))
    std = float(np.std(losses))
    ppl = math.exp(mean) if mean < 50 else float('inf')
    return mean, std, ppl


@torch.no_grad()
def score_dataset(
        baseline_model: MixedDecoder, raw_model: MixedDecoder, soft_model: MixedDecoder,
        ds, n_batches: int, batch_size: int, label: str = '',
) -> Dict[str, dict]:
    """Score all three models on IDENTICAL, deterministic samples from ``ds``.

    A single ``ds.get_batch`` call produces each batch; the SAME batch is fed to
    the baseline, raw-context and soft-context models so their losses are
    directly comparable (only the model differs, not the data).
    """
    losses: Dict[str, List[float]] = {'baseline': [], 'raw': [], 'soft': []}
    for b in range(n_batches):
        batch = ds.get_batch(batch_size)  # already on DEVICE, deterministic
        try:
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                baseline_loss_dict, _ = baseline_model(batch)
                raw_loss_dict, _ = raw_model(batch)
                soft_loss_dict, _ = soft_model(batch)
            vals = {
                'baseline': baseline_loss_dict['loss'].item(),
                'raw': raw_loss_dict['loss'].item(),
                'soft': soft_loss_dict['loss'].item(),
            }
            if all(math.isfinite(v) for v in vals.values()):
                for k, v in vals.items():
                    losses[k].append(v)
        except torch.cuda.OutOfMemoryError as e:
            print(f'    [WARN] {label} batch {b} failed (OOM): {e}')
            torch.cuda.empty_cache()
            continue
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f'    [WARN] {label} batch {b} failed: {e}')
            continue

    out = {}
    for k, arr in losses.items():
        mean, std, ppl = _agg(arr)
        out[k] = dict(mean=mean, std=std, ppl=ppl, n_batches=len(arr))
    return out


def build_pooled_dataset(
        sources_data: dict, sources: List[str], win_size: int,
        tkz_enc, tkz_dec, inp_len: int,
) -> StackedNextTokDataset:
    """Build a deterministic, fixed-(N,K) dataset pooled across ``sources``."""
    return build_stacked_next_tok_datasets(
        sources=sources, sources_data=sources_data, split='val',
        tkz_enc=tkz_enc, inp_len=inp_len, min_next_toks=FIXED_TARGET_TOKS,
        emb_win_min_size=win_size, emb_win_max_size=win_size,
        max_target_toks=FIXED_TARGET_TOKS, device=DEVICE, tkz_dec=tkz_dec,
        fixed_win_size=win_size, fixed_target_toks=FIXED_TARGET_TOKS,
        deterministic=True, prompt='', seed=RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_sep('=')
    print('Next-Token Perplexity — Baseline  vs  Raw Context  vs  Soft Context')
    print_sep('=')
    print(f'DEVICE: {DEVICE}')

    soft_run = SOFT_RUN_DIR or _resolve_latest_next_run(TRAIN_ROOT, 'embEnc')
    raw_run = RAW_RUN_DIR or _resolve_latest_next_run(TRAIN_ROOT, 'deco')
    print(f'SOFT run (encoder, fine-tuned)     : {soft_run.name}')
    print(f'RAW  run (decoder-only, fine-tuned): {raw_run.name}')
    print(f'BASELINE                           : pretrained {{decoder_model_name}}, no fine-tuning')
    print()

    soft_cfg = load_cfg(soft_run)
    raw_cfg = load_cfg(raw_run)

    # Both models must share the decoder vocabulary for comparable perplexities.
    assert soft_cfg.decoder_model_name == raw_cfg.decoder_model_name, (
        f'Decoder mismatch: soft={soft_cfg.decoder_model_name!r} vs '
        f'raw={raw_cfg.decoder_model_name!r}. Perplexities would not be comparable.'
    )
    assert soft_cfg.enc_bert.inp_len == raw_cfg.enc_bert.inp_len, (
        f'inp_len mismatch: soft={soft_cfg.enc_bert.inp_len} vs raw={raw_cfg.enc_bert.inp_len}.'
    )
    assert not soft_cfg.decoder_only, 'SOFT run must be an encoder (decoder_only=False) run.'
    assert raw_cfg.decoder_only, 'RAW run must be a decoder-only (decoder_only=True) run.'
    assert soft_cfg.next_sources == raw_cfg.next_sources, (
        f'next_sources mismatch: soft={soft_cfg.next_sources!r} vs raw={raw_cfg.next_sources!r}.'
    )

    inp_len = soft_cfg.enc_bert.inp_len
    sources = list(soft_cfg.next_sources)
    print(f'Shared decoder : {soft_cfg.decoder_model_name}')
    print(f'inp_len        : {inp_len}  (chunk content = {inp_len - 2} tokens)')
    print(f'Fixed target K : {FIXED_TARGET_TOKS} tokens')
    print(f'Window sizes   : {WIN_SIZES}  (training window = {TRAIN_WIN_SIZE})')
    print(f'next_sources   : {sources}')
    print()

    # Shared tokenizers (built from the soft config; asserted decoder-compatible).
    tkz_enc, tkz_dec = build_tokenizers(soft_cfg)

    # Load all three models (share the exact same tokenizers).
    print('Loading models …')
    baseline_model = load_baseline_model(raw_run, tkz_enc, tkz_dec)
    raw_model = load_model(raw_run, raw_cfg, tkz_enc, tkz_dec)
    soft_model = load_model(soft_run, soft_cfg, tkz_enc, tkz_dec)
    print()

    # Stream a small, deterministic proxy validation subset per source (see
    # load_streamed_source_for_next_eval — corpora are not pre-cached locally).
    print_sep('=')
    print(f'STREAMING VALIDATION SUBSETS ({N_STREAM_DOCS} docs/source, >= {MIN_DOC_CHARS} chars)')
    print_sep('=')
    sources_data = {}
    for src in sources:
        sources_data[src] = load_streamed_source_for_next_eval(src)
    print()

    # ------------------------------------------------------------------
    # Part A — pooled multi-source window-size sweep
    # ------------------------------------------------------------------
    print_sep('=')
    print('PART A — POOLED WINDOW-SIZE SWEEP (all sources combined)')
    print_sep('=')
    sweep_results: List[dict] = []
    for win_size in WIN_SIZES:
        print(f'  win_size={win_size}  (N_ctx={win_size * (inp_len - 2)} tokens) …', flush=True)
        ds = build_pooled_dataset(sources_data, sources, win_size, tkz_enc, tkz_dec, inp_len)
        scores = score_dataset(
            baseline_model, raw_model, soft_model, ds,
            n_batches=N_EVAL_BATCHES_SWEEP, batch_size=BATCH_SIZE, label=f'win={win_size}',
        )
        n_ctx_toks = win_size * (inp_len - 2)
        n_soft = _n_soft_tokens(soft_cfg, win_size)
        ratio = n_ctx_toks / n_soft if n_soft > 0 else float('nan')
        sweep_results.append(dict(win_size=win_size, n_ctx_toks=n_ctx_toks, n_soft=n_soft, ratio=ratio, **scores))
        b, r, s = scores['baseline'], scores['raw'], scores['soft']
        print(f"    baseline: loss={b['mean']:.4f} ppl={b['ppl']:.3f}   "
              f"raw: loss={r['mean']:.4f} ppl={r['ppl']:.3f}   "
              f"soft: loss={s['mean']:.4f} ppl={s['ppl']:.3f}   "
              f"({scores['soft']['n_batches']} batches)", flush=True)

    print()
    print_sep('-')
    print('SUMMARY  (Part A — pooled sweep)')
    print_sep('-')
    hdr = (f"{'win':>4} {'N_ctx':>7} {'N_soft':>7} {'ratio':>7}  "
           f"{'base loss':>10} {'base ppl':>9}  {'raw loss':>10} {'raw ppl':>9}  "
           f"{'soft loss':>10} {'soft ppl':>9}  {'Δraw-base':>10} {'Δsoft-raw':>10}")
    print(hdr)
    print_sep('-')
    for r in sweep_results:
        b, rw, s = r['baseline'], r['raw'], r['soft']
        print(f"{r['win_size']:>4} {r['n_ctx_toks']:>7} {r['n_soft']:>7} {r['ratio']:>6.1f}x  "
              f"{b['mean']:>10.4f} {b['ppl']:>9.3f}  "
              f"{rw['mean']:>10.4f} {rw['ppl']:>9.3f}  "
              f"{s['mean']:>10.4f} {s['ppl']:>9.3f}  "
              f"{rw['ppl'] - b['ppl']:>+10.3f} {s['ppl'] - rw['ppl']:>+10.3f}")
    print_sep('-')
    print('base = pretrained-only baseline (no fine-tuning) | raw = fine-tuned decoder-only | soft = fine-tuned encoder+decoder')
    print('Δraw-base  = raw ppl - baseline ppl   (negative => fine-tuning helped)')
    print('Δsoft-raw  = soft ppl - raw ppl        (positive => soft-token compression costs perplexity)')
    print()

    # ------------------------------------------------------------------
    # Part B — per-source breakdown at the training window size
    # ------------------------------------------------------------------
    print_sep('=')
    print(f'PART B — PER-SOURCE BREAKDOWN (win_size={TRAIN_WIN_SIZE}, matches ewn{TRAIN_WIN_SIZE}x{TRAIN_WIN_SIZE} training config)')
    print_sep('=')
    source_results: List[dict] = []
    for src in sources:
        print(f'  source={src} …', flush=True)
        ds = build_pooled_dataset(sources_data, [src], TRAIN_WIN_SIZE, tkz_enc, tkz_dec, inp_len)
        scores = score_dataset(
            baseline_model, raw_model, soft_model, ds,
            n_batches=N_EVAL_BATCHES_SOURCE, batch_size=BATCH_SIZE, label=src,
        )
        source_results.append(dict(source=src, **scores))
        b, r, s = scores['baseline'], scores['raw'], scores['soft']
        print(f"    baseline: loss={b['mean']:.4f} ppl={b['ppl']:.3f}   "
              f"raw: loss={r['mean']:.4f} ppl={r['ppl']:.3f}   "
              f"soft: loss={s['mean']:.4f} ppl={s['ppl']:.3f}   "
              f"({scores['soft']['n_batches']} batches)", flush=True)

    print()
    print_sep('-')
    print('SUMMARY  (Part B — per-source breakdown)')
    print_sep('-')
    hdr = (f"{'source':<16} {'base loss':>10} {'base ppl':>9}  {'raw loss':>10} {'raw ppl':>9}  "
           f"{'soft loss':>10} {'soft ppl':>9}  {'Δraw-base':>10} {'Δsoft-raw':>10}")
    print(hdr)
    print_sep('-')
    for r in source_results:
        b, rw, s = r['baseline'], r['raw'], r['soft']
        print(f"{r['source']:<16} {b['mean']:>10.4f} {b['ppl']:>9.3f}  "
              f"{rw['mean']:>10.4f} {rw['ppl']:>9.3f}  "
              f"{s['mean']:>10.4f} {s['ppl']:>9.3f}  "
              f"{rw['ppl'] - b['ppl']:>+10.3f} {s['ppl'] - rw['ppl']:>+10.3f}")
    print_sep('-')
    print()
    print('Done.')


if __name__ == '__main__':
    main()

