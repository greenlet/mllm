# KVQuant — Hooper et al., 2024

> **arXiv:** 2401.18079v6 · **Venue:** NeurIPS 2024 · **Affiliation:** UC Berkeley

## TL;DR
KVQuant enables **sub-4-bit** KV-cache quantization with <0.1 perplexity degradation at 3 bits, via
four techniques: **per-channel** key quantization, **pre-RoPE** key quantization, **non-uniform**
sensitivity-weighted datatypes, and **per-vector dense-and-sparse** outlier isolation. It keeps the
context **verbatim** (unlike eviction) and pushes single-GPU context length to millions of tokens.

## Problem & motivation
For very long contexts the KV cache dominates memory. Quantization compresses it, but naive schemes
fail below 4 bits because Key activations have **per-channel outliers** and RoPE spreads distortion
across positions. KVQuant targets accurate sub-4-bit KV without dropping any tokens.

## Key idea
Quantize where the distribution is friendliest and isolate the outliers:
- **Per-Channel Key Quantization** — quantize keys along the channel dimension to match their
  outlier structure.
- **Pre-RoPE Key Quantization** — quantize keys **before** applying rotary embeddings to avoid RoPE
  smearing quantization error.
- **Non-Uniform Quantization** — per-layer, sensitivity-weighted non-uniform datatypes.
- **Per-Vector Dense-and-Sparse** — split each vector into a dense low-bit part plus a few sparse
  outliers stored separately.

## How it works
- Calibrate per-channel/per-layer quantization datatypes offline; store keys pre-RoPE and apply RoPE
  after dequantization.
- Isolate a small fraction of outliers per vector into a sparse side channel; keep the rest dense at
  ~2–4 bits.
- Custom **CUDA kernels** implement the dense-and-sparse format for real speedups.

## Training / data
Post-training quantization with calibration (Wikitext-2, C4). No fine-tuning of model weights.

## Results
| Metric | Result | Notes |
|---|---|---|
| Perplexity degradation @ 3-bit | **< 0.1** | Wikitext-2 & C4 (per abstract) |
| Max context, LLaMA-7B, single A100-80GB | up to **1M** tokens | per abstract |
| Max context, 8-GPU system | up to **10M** tokens | per abstract |
| Kernel speedup | up to **~1.7×** vs fp16 matvec | LLaMA-7B (per abstract) |

## Limitations & follow-ups
- **Uniform, lossy** compression: reduces *bits per entry*, not the *number* of entries — orthogonal
  to and stackable with eviction ([SnapKV](kvcache_2024_snapkv.md), [KVzip](kvcache_2025_kvzip.md)).
- Requires calibration and custom kernels for the dense-and-sparse layout.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2401.18079) · [html](https://arxiv.org/html/2401.18079v6) · [pdf](https://arxiv.org/pdf/2401.18079)
- **Code:** https://github.com/SqueezeAILab/KVQuant
- **Hugging Face:** —
- **Related:** [KV-cache compression thread](../context/kv_cache/kv_cache.md)
