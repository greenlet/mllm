# KVzip — Kim et al., 2025

> **arXiv:** 2505.23416v2 · **Venue:** NeurIPS 2025 (Oral) · **Affiliation:** Seoul National University · NAVER AI Lab

## TL;DR
KVzip is a **query-agnostic** KV-cache eviction method: it scores each cached KV pair by how well
the underlying LLM can **reconstruct the original context** from the cache, then evicts the least
important pairs. Because the score does not depend on any particular query, one compressed cache is
**reusable across many future queries**.

## Problem & motivation
Query-aware eviction ([SnapKV](kvcache_2024_snapkv.md)) tunes the retained cache to the current
query, so it degrades badly when the same cache must serve *different* later queries — a common
multi-turn / agentic pattern (the paper shows degradation even at a 90% budget under multi-query).
KVzip aims for a **single** compressed cache that works for any subsequent query.

## Key idea
Importance = contribution to **context reconstruction**. Run the LLM to reconstruct the original
context from the cached KV; KV pairs whose removal least harms reconstruction are redundant and
evicted:

$$
\text{importance}(k,v) \propto \Delta\, \mathcal{L}_{\text{reconstruct}}(\text{context}\mid \text{cache}\setminus\{(k,v)\})
$$

## How it works
- Use forward passes of the frozen LLM to measure each KV pair's role in reconstructing the context.
- Rank and **evict** low-importance pairs down to a target budget; the surviving cache is
  query-agnostic and reused across queries/turns.
- No weight updates; wraps standard decoders; compatible with FlashAttention decoding.

## Training / data
Training-free (inference-time scoring via the base model's own forward pass).

## Results
| Metric | Result | Notes |
|---|---|---|
| KV cache size | **3–4×** reduction | negligible loss (per abstract) |
| FlashAttention decode latency | **~2×** lower | per abstract |
| Tasks | QA, retrieval, reasoning, code | LLaMA3.1, Qwen2.5, Gemma3 |
| Context length | up to **170K** tokens | per abstract |
| vs query-aware methods | outperforms | those degrade even at 90% budget (multi-query) |

## Limitations & follow-ups
- Still compresses **after a full prefill** — no TTFT savings from the prefill itself.
- Reconstruction scoring adds a compression-time pass. Successor: [Fast KVzip](kvcache_2026_fast-kvzip.md)
  (gated eviction, cheaper). Latent-space alternative: [Attention Matching](kvcache_2026_attention-matching.md).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2505.23416) · [pdf](https://arxiv.org/pdf/2505.23416)
- **Code:** https://github.com/snu-mllab/KVzip
- **Hugging Face:** —
- **Related:** [KV-cache compression thread](../context/kv_cache/kv_cache.md)
