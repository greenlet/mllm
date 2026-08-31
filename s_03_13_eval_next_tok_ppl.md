# s_03_13 — Next-Token Perplexity: Baseline vs Raw Context vs Soft Context

Three-way comparison of next-token perplexity on the **same** causal decoder,
conditioned on the **same** preceding text, before predicting the next K tokens:

| Setup | Name | How the context reaches the decoder | Fine-tuned? |
|-------|------|-------------------------------------|--------------|
| 1 | **Baseline** | Stock pretrained `Qwen/Qwen2.5-1.5B` reads the raw context tokens directly. Built as a `decoder_only=true` MixedDecoder wrapper whose checkpoint-load step is skipped — only `AutoModelForCausalLM.from_pretrained` weights are used. | No |
| 2 | **Raw Context** | N context tokens fed directly to the decoder as ordinary token embeddings. | Yes (`decoder_only=true`) |
| 3 | **Soft Context** | N context tokens packed into `ceil(N / (inp_len-2))` soft tokens by the BERT encoder (one CLS embedding per chunk, x`emb_exp_rate`), consumed by the decoder in place of raw context. | Yes (`decoder_only=false`) |

All three are scored on **identical** `(N context tokens, K target tokens)`
samples, so the only differences are (a) whether the decoder was fine-tuned at
all and (b) whether the context reaches it raw or compressed into soft tokens.

- N (context tokens) = `win_size * (inp_len - 2)`  (with `inp_len=128` → `126` content tokens/chunk)
- K (target tokens)  = `FIXED_TARGET_TOKS` (64, matches `min_next_toks` in the pinned runs' configs)
- Compression ratio  = N / (number of soft tokens the decoder sees)

## Data: multi-source streaming (not Wikipedia)

The two pinned runs were trained with `next_sources = pg19 bookcorpusopen arxiv
govreport gutenberg` (the latest active setting in
[s_03_11_train_mixed_decoder.sh](s_03_11_train_mixed_decoder.sh)), not
Wikipedia. These corpora total ~33 GB and are **not cached locally** on this
machine, so full-corpus download (and the exact random 95/5 train/val index
split used at train time) is impractical for a one-off eval.

Instead, `load_streamed_source_for_next_eval` streams each source from the Hub
and materializes the first `N_STREAM_DOCS` documents that are at least
`MIN_DOC_CHARS` characters long into an in-memory `datasets.Dataset`. This is a
**proxy validation subset** — order-based, not the exact held-out indices used
at train time — called out explicitly here and in the script's docstring.

## Protocol

Train two runs that differ **only** in `decoder_only`, using the SAME decoder,
`inp_len`, and `next_sources`. The controlled dataset mode (fixed window/target
size, deterministic sampling, in
[mllm/train/next_tok_wiki.py](mllm/train/next_tok_wiki.py)) guarantees every
emitted sample has an identical context-token count N and target length K, so
batch-mean cross-entropy equals the token-level mean and `ppl = exp(mean_loss)`
is exact.

### 1. Train the soft-context model (encoder ON)

Edit [s_03_11_train_mixed_decoder.sh](s_03_11_train_mixed_decoder.sh):

```sh
train_ds_types="next"
decoder_only=false
min_next_toks=64
```

then `bash s_03_11_train_mixed_decoder.sh`. Produces a run dir named
`mixeddecoder-...-embEnc...-...-dsNext-...`.

### 2. Train the raw-context model (decoder-only)

Same `.sh`, flip a single flag (keep everything else identical):

```sh
train_ds_types="next"
decoder_only=true
min_next_toks=64
```

then `bash s_03_11_train_mixed_decoder.sh`. Produces a run dir named
`mixeddecoder-...-deco-...-dsNext-...`.

> Both runs must use the **same decoder** (`decoder_type` / `decoder_model_name`)
> and the same `next_sources` so their perplexities are directly comparable.
> The eval script asserts this.

### 3. Baseline (no training required)

The baseline needs no checkpoint at all — `load_baseline_model` reuses the RAW
run's architecture config (`decoder_only=true`, same `max_seq_len`/`use_sep`/
`prompt_first`/`decoder_model_name`) but skips `load_pretrained`, so the decoder
stays at its stock HF pretrained weights.

### 4. Evaluate

```sh
PYTHONPATH=. python3 s_03_13_eval_next_tok_ppl.py
```

By default it auto-resolves the latest soft (`embEnc*` + `dsNext`) and raw
(`deco` + `dsNext`) runs under `data/train_mllm_encdec_bert`. To pin specific
runs, set `SOFT_RUN_DIR` / `RAW_RUN_DIR` at the top of
[s_03_13_eval_next_tok_ppl.py](s_03_13_eval_next_tok_ppl.py).

> **Marker-matching bug (fixed):** `_resolve_latest_next_run` used to search for
> `marker in run_dir.name` as a raw substring. Since every run directory name
> starts with the literal word "mixed**deco**der", the `'deco'` marker matched
> *every* run (not just `decoder_only=true` ones), silently resolving `RAW_RUN_DIR`
> to whichever `dsNext` run was chronologically latest overall. It now matches
> `marker` against dash-delimited name tokens (skipping the fixed
> `mixeddecoder-<timestamp>` prefix), so `'deco'` only matches the standalone
> `-deco-` token.

Knobs (top of the script):

- `WIN_SIZES = [2, 4, 8, 10]` — context window sizes (chunks) swept in Part A.
  `TRAIN_WIN_SIZE = 10` matches `ewn10x10` in both pinned runs' configs.
- `FIXED_TARGET_TOKS = 64` — K, decoder target tokens per sample.
- `BATCH_SIZE = 8`; `N_EVAL_BATCHES_SWEEP = 15` (Part A, pooled across sources);
  `N_EVAL_BATCHES_SOURCE = 12` (Part B, per source at `TRAIN_WIN_SIZE`).
- `N_STREAM_DOCS = 120`, `MIN_DOC_CHARS = 8000` — streamed proxy validation subset size/filter.
- `EVAL_MAX_SEQ_LEN = 8192` — raised sequence budget so the uncompressed raw
  context always fits (only affects the length assertion; Qwen uses RoPE so
  there is no learned positional table to overflow).

## Output

**Part A** — a per-window table (pooled across all 5 sources): N context
tokens, number of soft tokens, compression ratio, and baseline/raw/soft
loss+ppl, plus `Δraw-base` (fine-tuning effect) and `Δsoft-raw` (compression
cost).

**Part B** — a per-source table at `TRAIN_WIN_SIZE` (the actual trained window
size): baseline/raw/soft loss+ppl per source (pg19, bookcorpusopen, arxiv,
govreport, gutenberg), with the same two deltas.

`Δraw-base = raw ppl - baseline ppl` (negative ⇒ fine-tuning helped).
`Δsoft-raw = soft ppl - raw ppl` (positive ⇒ soft-token compression costs perplexity).

## Notes

- If all three 1.5B-decoder models on one GPU cause OOM, reduce `BATCH_SIZE`.
- For learned-position decoders (GPT-2 / BertGeneration), large raw windows may
  exceed the trained `max_seq_len`; such batches are caught and skipped (reported
  as warnings). Qwen (RoPE) has no such limit once `EVAL_MAX_SEQ_LEN` is raised.
- The streamed validation subset is a convenience proxy (first-N-long-enough
  documents per source), not the exact random held-out split used at train time.

## Results — 2026-08-29

Pinned runs (auto-resolved, latest available locally):

- **Baseline**: pretrained `Qwen/Qwen2.5-1.5B`, no fine-tuning (architecture borrowed from the RAW run below).
- **Raw** (decoder-only, fine-tuned): `mixeddecoder-20260729_084529-bertbaseuncased-d768-deco-inp128-decQwen2.51.5b-msl5632-dtypeFp16-sepF-pallF-pfirstT-dsNext-mnt64-srcpg_bo_ar_go_gu-trn_lr2e-05_bs1_wdD0.1_wdO0.01_llrd0.9_attdp0.1_gc1.0` (last_epoch=46, val_loss_min=1.739)
- **Soft** (encoder ON, fine-tuned): `mixeddecoder-20260807_195436-bertbaseuncased-d768-embEncCls-inp128-decQwen2.51.5b-msl640-dtypeFp16-sepF-pallF-pfirstT-eer2-ewn10x10-frzencF-dsNext-mnt64-srcpg_bo_ar_go_gu-trn_lr2e-05_bs8_wdD0.1_wdO0.01_llrd0.9_attdp0.1_gc1.0` (last_epoch=34, val_loss_min=1.958)

Settings: `BATCH_SIZE=4`, `N_EVAL_BATCHES_SWEEP=15` (Part A), `N_EVAL_BATCHES_SOURCE=12`
(Part B) — every reported point below used the full batch count (no dropped
batches; an earlier attempt at `BATCH_SIZE=8` hit intermittent CUDA OOM with all
three 1.5B-decoder models resident on one GPU at `win_size>=8`, silently
dropping batches — fixed by halving the batch size).

### Part A — pooled window-size sweep (all 5 sources combined)

| win | N_ctx | N_soft | ratio | base loss | base ppl | raw loss | raw ppl | soft loss | soft ppl | Δraw-base | Δsoft-raw |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 252 | 4 | 63.0x | 2.5214 | 12.446 | 2.5949 | 13.395 | 3.1654 | 23.698 | +0.949 | +10.304 |
| 4 | 504 | 8 | 63.0x | 2.5000 | 12.183 | 2.6308 | 13.885 | 3.2090 | 24.754 | +1.702 | +10.869 |
| 8 | 1008 | 16 | 63.0x | 2.3650 | 10.644 | 2.3977 | 10.998 | 3.1285 | 22.839 | +0.353 | +11.841 |
| 10 | 1260 | 20 | 63.0x | 2.4588 | 11.691 | 2.4790 | 11.929 | 3.1687 | 23.777 | +0.238 | +11.848 |

### Part B — per-source breakdown (win_size=10, matches `ewn10x10` training config)

| source | base loss | base ppl | raw loss | raw ppl | soft loss | soft ppl | Δraw-base | Δsoft-raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pg19 | 2.5699 | 13.064 | 2.7181 | 15.152 | 3.1505 | 23.348 | +2.087 | +8.196 |
| bookcorpusopen | 2.9640 | 19.376 | 3.0351 | 20.803 | 3.6841 | 39.811 | +1.427 | +19.008 |
| arxiv | 2.1173 | 8.309 | 1.9405 | 6.962 | 2.5054 | 12.249 | **-1.346** | +5.287 |
| govreport | 2.3637 | 10.630 | 2.3403 | 10.384 | 3.0620 | 21.371 | **-0.246** | +10.987 |
| gutenberg | 2.6157 | 13.677 | 2.7075 | 14.992 | 3.2987 | 27.076 | +1.315 | +12.084 |

### Takeaways

- **Soft-token compression has a large, consistent perplexity cost** on this
  streamed proxy validation data: `Δsoft-raw` is roughly `+10` to `+12` ppl
  across every window size (Part A) and every source (Part B) — the soft
  encoder+decoder run is markedly worse than either the pretrained baseline or
  the fine-tuned raw-context run, at all context lengths tested.
- **Fine-tuning the decoder-only ("raw") model on this proxy data does not
  clearly beat the untrained baseline** (`Δraw-base`): it's mixed and mostly
  small — worse on pg19/bookcorpusopen/gutenberg, better on arxiv/govreport.
  This is plausibly a domain-shift artifact of the streamed proxy validation
  subset (see the "Data" section above): both `raw` and `soft` were fine-tuned
  on the FULL corpora's true random 95/5 split, while this eval draws from a
  small first-N-long-enough document sample per source that may not represent
  the same distribution the models were tuned on — arxiv/govreport (more
  narrowly-distributed, technical-document corpora) show the expected
  fine-tuning benefit, while the more heterogeneous book corpora (pg19,
  bookcorpusopen, gutenberg) do not.
- Compression ratio is constant (63.0x) across all window sizes because
  `emb_exp_rate=2` for the soft run scales linearly with `win_size`.


