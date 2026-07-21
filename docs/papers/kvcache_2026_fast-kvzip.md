# Fast KVzip — Kim et al., 2026

> **arXiv:** 2601.17668v2 · **Venue:** preprint · **Affiliation:** NAVER AI Lab · Seoul National University

## TL;DR
Fast KVzip is a **gating-based** KV-cache eviction method for **frozen-weight** LLMs: lightweight
sink-attention gating modules identify and retain the critical KV pairs, achieving high compression
ratios at **negligible computational cost**. The gates are trained with **forward passes only** (no
backpropagation) against a task-agnostic reconstruction objective.

## Problem & motivation
Existing KV compression trades off performance loss against compute overhead — reconstruction-based
scoring ([KVzip](kvcache_2025_kvzip.md)) is accurate but adds an expensive compression pass. Fast
KVzip keeps the query-agnostic, reconstruction-driven quality while making the eviction decision
**cheap** and streamable across prefill and decode.

## Key idea
Attach small **sink-attention gating** modules that score KV pairs for retention, trained without
backprop via forward-pass signals against a reconstruction target:

$$
g = \sigma\!\big(\text{SinkAttnGate}(K,V)\big), \qquad \text{retain KV where } g > \tau
$$

## How it works
- Insert lightweight gating modules that produce per-KV retention scores; the base LLM weights stay
  **frozen**.
- Train the gates with a **task-agnostic reconstruction objective** using only forward passes
  (avoiding expensive backpropagation), yielding strong task generalization.
- Integrate the gate into **both prefill and decoding**, evicting up to ~70% of the cache online.

## Training / data
Gate-only training via forward passes; base model frozen. Task-agnostic reconstruction objective.

## Results
| Metric | Result | Notes |
|---|---|---|
| KV cache evicted | up to **70%** | near-lossless (per abstract) |
| Model families | Qwen2.5-1M, Qwen3, Gemma3 | per abstract |
| Task coverage | long-context, code, math | consistent across tasks (per abstract) |

## Limitations & follow-ups
- Still **post-prefill** eviction; no reduction of prefill TTFT itself.
- Adds (small) gating parameters vs fully training-free scorers like
  [Expected Attention](kvcache_2025_expected-attention.md).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2601.17668) · [pdf](https://arxiv.org/pdf/2601.17668)
- **Code:** per paper ("Source code: this https URL")
- **Hugging Face:** —
- **Related:** [KVzip](kvcache_2025_kvzip.md) · [KV-cache compression thread](../context/kv_cache/kv_cache.md)
