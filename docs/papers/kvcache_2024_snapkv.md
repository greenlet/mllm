# SnapKV — Li et al., 2024

> **arXiv:** 2404.14469v2 · **Venue:** preprint (code widely adopted) · **Affiliation:** UIUC · Cohere · Princeton

## TL;DR
SnapKV is a **fine-tuning-free** KV-cache compression method: for each attention head it selects
a small cluster of the prompt's important KV positions using the attention pattern observed in a
short window at the **end** of the prompt, then discards the rest before generation. It cuts memory
and speeds decoding with negligible accuracy loss on long-context tasks.

## Problem & motivation
The KV cache grows linearly with input length, dominating memory and latency for long prompts.
Prior eviction heuristics either need training or degrade quality. SnapKV exploits an empirical
regularity: **each head consistently attends to the same prompt positions during generation**, and
that pattern is already visible from the last few prompt tokens — so the important positions can be
identified *before* generation starts.

## Key idea
Use an **observation window** (the final $L_{obs}$ prompt tokens) to vote on which earlier prompt
KV positions matter, then keep only the top-scoring, spatially **clustered** positions per head:

$$
s_j = \sum_{q \in \text{obs window}} \mathrm{softmax}(q\,k_j^\top), \qquad
\text{keep } \mathrm{TopK}_j\big(\mathrm{pool}(s_j)\big)
$$

## How it works
- Compute attention from the observation-window queries to all prompt keys; aggregate into a
  per-position importance score per head.
- **Pool/cluster** the scores (1-D pooling) so retained tokens form contiguous, information-complete
  spans rather than isolated tokens.
- Retain the top-$k$ clustered prompt KV entries **plus** the full observation window; evict the rest.
- Query-aware: the retained set is tuned to the current prompt/query. Fine-tuning-free; works as a
  drop-in with minor changes to a HuggingFace decoder.

## Training / data
None — inference-time only, no parameter updates.

## Results
| Metric | Result | Notes |
|---|---|---|
| Generation speed | **3.6×** vs baseline @ 16K input | per abstract |
| Memory efficiency | **8.2×** vs baseline @ 16K input | per abstract |
| Long-context quality | comparable across **16** datasets | per abstract |
| Max context (single A100-80GB) | up to **380K** tokens | negligible NIAH drop (per abstract) |

## Limitations & follow-ups
- **Query-dependent**: the retained cache is tuned to the current prompt, so it does not reuse
  cleanly across different future queries — the gap query-agnostic methods target
  ([KVzip](kvcache_2025_kvzip.md), [Expected Attention](kvcache_2025_expected-attention.md)).
- **Per-head, non-uniform** eviction complicates paged-attention engines (uniform block layout).
- Successors: [Fast KVzip](kvcache_2026_fast-kvzip.md), [Attention Matching](kvcache_2026_attention-matching.md).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2404.14469) · [html](https://arxiv.org/html/2404.14469v2) · [pdf](https://arxiv.org/pdf/2404.14469)
- **Code:** https://github.com/FasterDecoding/SnapKV
- **Hugging Face:** —
- **Related:** [KV-cache compression thread](../context/kv_cache/kv_cache.md)
