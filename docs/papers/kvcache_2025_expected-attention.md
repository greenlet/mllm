# Expected Attention — Devoto et al., 2025

> **arXiv:** 2510.00636v1 · **Venue:** preprint · **Affiliation:** Sapienza University of Rome · NVIDIA

## TL;DR
Expected Attention is a **training-free** KV-cache compression method that scores each KV pair by
**predicting how future queries will attend to it**, computing an expected attention score in
**closed form** from the distributional properties of LLM activations. It works in both prefill and
decode and ships with **KVPress**, a library of 20+ KV-compression techniques.

## Problem & motivation
Attention-score-based pruning is attractive but faces two practical blocks: (1) attention scores
from **future** tokens are unavailable at compression time, and (2) FlashAttention **never
materializes** the full attention matrix, so past scores are inaccessible. Expected Attention
sidesteps both by *estimating* future attention analytically rather than reading it.

## Key idea
Model the distribution of future query vectors and compute each KV pair's **expected** attention in
closed form; rank and prune the lowest-scoring pairs with minimal impact on the residual stream:

$$
\mathbb{E}_{q\sim p(q)}\big[\mathrm{softmax}(q\,k_j^\top)\big] \;\to\; \text{importance}(k_j,v_j)
$$

## How it works
- Estimate the query distribution $p(q)$ from activation statistics; derive a closed-form expected
  attention score per KV pair.
- Rank KV pairs by expected attention and prune to budget; designed so pruning perturbs the residual
  stream minimally.
- Operates **seamlessly across prefilling and decoding**; fully training-free.

## Training / data
None — training-free; uses only activation-distribution statistics at inference.

## Results
| Metric | Result | Notes |
|---|---|---|
| vs SOTA baselines | consistently outperforms | prefill **and** decode (per abstract) |
| Tooling | **KVPress** library, 20+ methods | released (per abstract) |

*(Abstract reports relative gains rather than absolute benchmark numbers.)*

## Limitations & follow-ups
- Closed-form estimate relies on distributional assumptions about query activations.
- Query-agnostic like [KVzip](kvcache_2025_kvzip.md); still a **post-prefill** operation.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2510.00636) · [html](https://arxiv.org/html/2510.00636v1) · [pdf](https://arxiv.org/pdf/2510.00636)
- **Code:** KVPress — https://github.com/NVIDIA/kvpress
- **Hugging Face:** —
- **Related:** [KV-cache compression thread](../context/kv_cache/kv_cache.md)
