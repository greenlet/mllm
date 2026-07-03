# s_03_13 — Next-Token Perplexity: Soft Context vs Raw Context

Controlled comparison of two ways to condition the **same** causal decoder on the
**same** preceding text before predicting the next K tokens:

| Setup | Name | How the context reaches the decoder | Flag |
|-------|------|-------------------------------------|------|
| 1 | **Soft Context + Target** | N context tokens are packed into `ceil(N / (inp_len-2))` soft tokens by the BERT encoder (one CLS embedding per chunk, optionally x`emb_exp_rate`), which the decoder consumes in place of the raw context. | `decoder_only=false` |
| 2 | **Context + Target** | The N context tokens are fed directly to the decoder as ordinary token embeddings. | `decoder_only=true` |

Both are scored on **identical** `(N context tokens, K target tokens)` samples
drawn deterministically from the Wikipedia validation split, so the only
difference is context compression. The perplexity gap is the price of squeezing
the context into soft tokens.

- N (context tokens) = `next_fixed_win_size * (inp_len - 2)`  (with `inp_len=128` → `126` content tokens/chunk)
- K (target tokens)  = `next_fixed_target_toks`
- Compression ratio  = N / (number of soft tokens the decoder sees)

## Protocol

Train two runs that differ **only** in `decoder_only`, using the SAME
`--next-fixed-win-size`, `--next-fixed-target-toks`, decoder, and `inp_len`.
The controlled dataset mode (in [mllm/train/next_tok_wiki.py](mllm/train/next_tok_wiki.py))
guarantees every emitted sample has an identical context-token count N and target
length K, so batch-mean cross-entropy equals the token-level mean and
`ppl = exp(mean_loss)` is exact.

### 1. Train the soft-context model (encoder ON)

Edit [s_03_11_train_mixed_decoder.sh](s_03_11_train_mixed_decoder.sh):

```sh
train_ds_types="next"
decoder_only=false
min_next_toks=64
next_fixed_win_size=8        # N = 8 * 126 = 1008 context tokens
next_fixed_target_toks=64    # K = 64 target tokens
```

then

```sh
bash s_03_11_train_mixed_decoder.sh
```

Produces a run dir named `mixeddecoder-...-embEnc...-...-dsNext-...`.

### 2. Train the raw-context model (decoder-only)

Same `.sh`, flip a single flag (keep everything else identical):

```sh
train_ds_types="next"
decoder_only=true
min_next_toks=64
next_fixed_win_size=8        # same N
next_fixed_target_toks=64    # same K
```

then

```sh
bash s_03_11_train_mixed_decoder.sh
```

Produces a run dir named `mixeddecoder-...-deco-...-dsNext-...`.

> Both runs must use the **same decoder** (`decoder_type` / `decoder_model_name`)
> so their target vocabularies — and therefore their perplexities — are directly
> comparable. The eval script asserts this.

### 3. Evaluate perplexity across a compression sweep

```sh
PYTHONPATH=. python3 s_03_13_eval_next_tok_ppl.py
```

By default it auto-resolves the latest soft (`embEnc` + `dsNext`) and raw
(`deco` + `dsNext`) runs under `data/train_mllm_encdec_bert`. To pin specific
runs, set `SOFT_RUN_DIR` / `RAW_RUN_DIR` at the top of
[s_03_13_eval_next_tok_ppl.py](s_03_13_eval_next_tok_ppl.py).

Sweep / budget knobs (top of the script):

- `WIN_SIZES = [1, 2, 4, 8]` — context window sizes (chunks) to evaluate.
- `FIXED_TARGET_TOKS = 64` — K, decoder target tokens per sample.
- `BATCH_SIZE = 8`, `N_EVAL_BATCHES = 50` — ≈400 items per window size.
- `EVAL_MAX_SEQ_LEN = 8192` — raised sequence budget so the uncompressed raw
  context always fits (only affects the length assertion; Qwen uses RoPE so there
  is no learned positional table to overflow).

## Output

A per-window table reporting, for each window size: N context tokens, number of
soft tokens, compression ratio, soft-context loss/ppl, raw-context loss/ppl, and
Δppl (soft − raw). A positive Δ means compression costs perplexity.

```
 win   N_ctx  N_soft   ratio   soft loss  soft ppl    raw loss   raw ppl      Δppl    Δloss
   1     126       1  126.0x      ...        ...         ...        ...         ...      ...
   2     252       2  126.0x      ...        ...         ...        ...         ...      ...
   4     504       4  126.0x      ...        ...         ...        ...         ...      ...
   8    1008       8  126.0x      ...        ...         ...        ...         ...      ...
```

## Notes

- If both 1.5B models on one GPU cause OOM, reduce `BATCH_SIZE`, or run the two
  models on separate GPUs.
- For learned-position decoders (GPT-2 / BertGeneration), large raw windows may
  exceed the trained `max_seq_len`; such batches are caught and skipped (reported
  as warnings). Qwen (RoPE) has no such limit once `EVAL_MAX_SEQ_LEN` is raised.
