# Fast KV Compaction via Attention Matching — Zweiger et al., 2026

> **arXiv:** 2602.16284v2 · **Venue:** preprint · **Affiliation:** MIT

## TL;DR
Attention Matching **compacts** the KV cache in **latent space** by constructing a small set of keys
and values that **reproduce the original attention outputs** and preserve attention mass at a
per-KV-head level. Unlike token-space summarization (lossy) or Cartridges (accurate but slow
end-to-end training), the objective decomposes into subproblems — several with **closed-form**
solutions — giving up to **50× compaction in seconds**.

## Problem & motivation
Deployed long-context systems compact the cache via **summarization** in token space, which is
lossy. [Cartridges](kvcache_2025_cartridges.md) showed compact **latent** KV caches can match
full-context quality, but require slow, expensive end-to-end optimization. This work seeks the same
latent-space fidelity **fast**.

## Key idea
Find compact $\tilde K,\tilde V$ that make attention over the compressed cache match attention over
the full cache, per head:

$$
\min_{\tilde K,\tilde V}\; \big\| \mathrm{Attn}(Q, K, V) - \mathrm{Attn}(Q, \tilde K, \tilde V) \big\|
\quad\text{(+ attention-mass preservation)}
$$

## How it works
- Formulate compaction as **attention output matching** at the per-KV-head level, preserving both
  outputs and total attention mass.
- **Decompose** the objective into simpler subproblems; solve some in **closed form**, others cheaply.
- Yields a family of methods trading compaction time vs quality, applied without slow end-to-end SGD.

## Training / data
No end-to-end gradient training of a cache; solves a matching objective per head (some closed-form).

## Results
| Metric | Result | Notes |
|---|---|---|
| Compaction ratio | up to **50×** | little quality loss (per abstract) |
| Compaction time | **seconds** on some datasets | vs slow Cartridges optimization |
| Frontier | pushes compaction-time vs quality Pareto | per abstract |

## Limitations & follow-ups
- Matches attention for observed $Q$; generalization to arbitrary future queries depends on the
  query set used.
- Latent-space cousin of eviction methods ([KVzip](kvcache_2025_kvzip.md)); still post-prefill.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2602.16284) · [html](https://arxiv.org/html/2602.16284v2) · [pdf](https://arxiv.org/pdf/2602.16284)
- **Code:** —
- **Hugging Face:** —
- **Related:** [Cartridges](kvcache_2025_cartridges.md) · [KV-cache compression thread](../context/kv_cache/kv_cache.md)
