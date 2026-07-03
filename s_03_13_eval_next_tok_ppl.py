"""Controlled next-token perplexity comparison for MixedDecoder.

Soft Context + Target  vs  Context + Target
-------------------------------------------
Two ways to condition the SAME causal decoder on the SAME preceding text before
predicting the next K tokens:

  1. **Soft context** (encoder ON): N context tokens are packed into
     ``ceil(N / (inp_len - 2))`` soft tokens by the BERT encoder (one CLS
     embedding per chunk, optionally x``emb_exp_rate`` expanded), which the
     decoder consumes in place of the raw context.  (``decoder_only = False``)

  2. **Raw context** (decoder-only): the N context tokens are fed directly to
     the decoder as ordinary token embeddings.  (``decoder_only = True``)

Both regimes are scored on IDENTICAL (N context tokens, K target tokens) samples
drawn deterministically from the Wikipedia validation split, so the only
difference is compression.  The gap in perplexity is the price of compressing
the context into soft tokens.

Usage (run from the repo root)::

    PYTHONPATH=. python3 s_03_13_eval_next_tok_ppl.py

Requirements
------------
Two trained checkpoints (see the two training commands in
``s_03_13_eval_next_tok_ppl.md``):
  * a soft-context run  (``dsNext`` + ``embEnc*`` in the run-dir name),
  * a raw-context run   (``dsNext`` + ``deco``     in the run-dir name),
ideally trained with the SAME decoder and the SAME ``--next-fixed-win-size`` /
``--next-fixed-target-toks``.

Set ``SOFT_RUN_DIR`` / ``RAW_RUN_DIR`` below to pin specific runs, or leave them
as ``None`` to auto-resolve the latest matching run under ``TRAIN_ROOT``.
"""

import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
# Context window sizes (number of chunks) to sweep. Each chunk holds
# ``inp_len - 2`` content tokens, so N_context_tokens = win_size * (inp_len - 2)
# and the compression ratio is (inp_len - 2) : 1 per chunk (before emb_exp).
WIN_SIZES: List[int] = [1, 2, 4, 8]
FIXED_TARGET_TOKS = 64          # K: decoder target tokens predicted per sample
BATCH_SIZE = 8                  # items per forward pass
N_EVAL_BATCHES = 50             # batches per window size -> 50*8 = 400 items
RANDOM_SEED = 42
VAL_RATIO = 0.05
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Large budget so the raw (uncompressed) context always fits; only used for the
# length assertion / target truncation (Qwen uses RoPE, so pos_emb is None).
EVAL_MAX_SEQ_LEN = 8192

# Suppress HF tokenizer length warnings
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# ---------------------------------------------------------------------------
# PYTHONPATH + project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT))

from pydantic_yaml import parse_yaml_file_as
from transformers import AutoTokenizer

from mllm.config.model import MixedDecoderCfg
from mllm.exp.args import MIXED_DECODER_MODEL_CFG_FNAME
from mllm.model.mixed_decoder import MixedDecoder
from mllm.train.next_tok_wiki import NextTokWikiDataset, load_split_wiki_for_next


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
    """
    candidates = sorted(
        [
            p for p in train_root.glob('mixeddecoder-*')
            if p.is_dir() and 'dsNext' in p.name and marker in p.name
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


def _n_soft_tokens(cfg: MixedDecoderCfg, win_size: int) -> int:
    """Number of soft tokens the decoder sees for a ``win_size``-chunk context."""
    if cfg.use_interactive_extractor:
        return win_size * max(cfg.ie_exp_rate, 1)
    if cfg.emb_exp_rate > 0:
        return win_size * cfg.emb_exp_rate
    return win_size


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_window(
        soft_model: MixedDecoder, raw_model: MixedDecoder,
        wiki_ds, wiki_inds_val: np.ndarray, tkz_enc, tkz_dec,
        inp_len: int, win_size: int,
) -> dict:
    """Score both models on identical, deterministic (N, K) samples.

    A single deterministic ``NextTokWikiDataset`` produces the batches; the SAME
    batch is fed to both models so soft-context and raw-context perplexities are
    computed over exactly the same context text and target tokens.
    """
    ds = NextTokWikiDataset(
        wiki_ds, wiki_inds_val, tkz_enc,
        inp_len=inp_len, min_next_toks=FIXED_TARGET_TOKS,
        emb_win_min_size=win_size, emb_win_max_size=win_size,
        max_target_toks=FIXED_TARGET_TOKS, device=DEVICE, tkz_dec=tkz_dec,
        fixed_win_size=win_size, fixed_target_toks=FIXED_TARGET_TOKS,
        deterministic=True,
    )

    soft_losses: List[float] = []
    raw_losses: List[float] = []
    for b in range(N_EVAL_BATCHES):
        batch = ds.get_batch(BATCH_SIZE)  # already on DEVICE, deterministic
        try:
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                soft_loss_dict, _ = soft_model(batch)
                raw_loss_dict, _ = raw_model(batch)
            sl = soft_loss_dict['loss'].item()
            rl = raw_loss_dict['loss'].item()
            if math.isfinite(sl) and math.isfinite(rl):
                soft_losses.append(sl)
                raw_losses.append(rl)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f'    [WARN] win={win_size} batch {b} failed: {e}')
            continue

    def _agg(losses: List[float]) -> Tuple[float, float, float]:
        if not losses:
            return float('nan'), float('nan'), float('nan')
        mean = float(np.mean(losses))
        std = float(np.std(losses))
        ppl = math.exp(mean) if mean < 50 else float('inf')
        return mean, std, ppl

    soft_mean, soft_std, soft_ppl = _agg(soft_losses)
    raw_mean, raw_std, raw_ppl = _agg(raw_losses)

    n_ctx_toks = win_size * (inp_len - 2)
    n_soft = _n_soft_tokens(soft_model.cfg, win_size)
    ratio = n_ctx_toks / n_soft if n_soft > 0 else float('nan')

    return dict(
        win_size=win_size, n_ctx_toks=n_ctx_toks, n_soft=n_soft, ratio=ratio,
        n_batches=len(soft_losses),
        soft_mean=soft_mean, soft_std=soft_std, soft_ppl=soft_ppl,
        raw_mean=raw_mean, raw_std=raw_std, raw_ppl=raw_ppl,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_sep('=')
    print('Next-Token Perplexity — Soft Context + Target  vs  Context + Target')
    print_sep('=')
    print(f'DEVICE: {DEVICE}')

    soft_run = SOFT_RUN_DIR or _resolve_latest_next_run(TRAIN_ROOT, 'embEnc')
    raw_run = RAW_RUN_DIR or _resolve_latest_next_run(TRAIN_ROOT, 'deco')
    print(f'SOFT run (encoder)     : {soft_run.name}')
    print(f'RAW  run (decoder-only): {raw_run.name}')
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

    inp_len = soft_cfg.enc_bert.inp_len
    print(f'Shared decoder : {soft_cfg.decoder_model_name}')
    print(f'inp_len        : {inp_len}  (chunk content = {inp_len - 2} tokens)')
    print(f'Fixed target K : {FIXED_TARGET_TOKS} tokens')
    print(f'Window sizes   : {WIN_SIZES}')
    print()

    # Shared tokenizers (built from the soft config; asserted decoder-compatible).
    tkz_enc, tkz_dec = build_tokenizers(soft_cfg)

    # Load both models (share the exact same tokenizers).
    print('Loading models …')
    soft_model = load_model(soft_run, soft_cfg, tkz_enc, tkz_dec)
    raw_model = load_model(raw_run, raw_cfg, tkz_enc, tkz_dec)
    print()

    # Wikipedia validation split (deterministic).
    wiki_ds, _wiki_inds_train, wiki_inds_val = load_split_wiki_for_next(
        DATA_PATH, val_ratio=VAL_RATIO, random_seed=RANDOM_SEED,
    )

    print_sep('=')
    print('EVALUATION')
    print_sep('=')
    results: List[dict] = []
    for win_size in WIN_SIZES:
        print(f'  win_size={win_size}  (N_ctx={win_size * (inp_len - 2)} tokens) …', flush=True)
        r = score_window(
            soft_model, raw_model, wiki_ds, wiki_inds_val, tkz_enc, tkz_dec,
            inp_len=inp_len, win_size=win_size,
        )
        results.append(r)
        print(f'    soft: loss={r["soft_mean"]:.4f}  ppl={r["soft_ppl"]:.3f}   '
              f'raw: loss={r["raw_mean"]:.4f}  ppl={r["raw_ppl"]:.3f}   '
              f'Δppl={r["soft_ppl"] - r["raw_ppl"]:+.3f}  ({r["n_batches"]} batches)',
              flush=True)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print_sep('=')
    print('SUMMARY  (Soft Context vs Raw Context, next-token perplexity)')
    print_sep('=')
    hdr = (f"{'win':>4} {'N_ctx':>7} {'N_soft':>7} {'ratio':>7}  "
           f"{'soft loss':>10} {'soft ppl':>9}  {'raw loss':>10} {'raw ppl':>9}  "
           f"{'Δppl':>8} {'Δloss':>8}")
    print(hdr)
    print_sep('-')
    for r in results:
        print(f"{r['win_size']:>4} {r['n_ctx_toks']:>7} {r['n_soft']:>7} {r['ratio']:>6.1f}x  "
              f"{r['soft_mean']:>10.4f} {r['soft_ppl']:>9.3f}  "
              f"{r['raw_mean']:>10.4f} {r['raw_ppl']:>9.3f}  "
              f"{r['soft_ppl'] - r['raw_ppl']:>+8.3f} {r['soft_mean'] - r['raw_mean']:>+8.4f}")
    print_sep('-')
    print('N_ctx  = raw context tokens fed to the decoder-only model')
    print('N_soft = soft tokens the encoder model feeds the decoder')
    print('ratio  = N_ctx / N_soft (context compression factor)')
    print('Δ      = soft - raw  (positive => compression costs perplexity)')
    print()
    print('Done.')


if __name__ == '__main__':
    main()
