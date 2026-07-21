# Cartridges — Eyuboglu et al., 2025

> **arXiv:** 2506.06266v3 · **Venue:** preprint · **Affiliation:** Stanford University · University at Buffalo

## TL;DR
A **Cartridge** is a small KV cache **trained offline** on a fixed corpus and loaded at inference to
answer queries about that corpus — amortizing prefill across all queries. Naive next-token training
underperforms in-context learning (ICL); the paper's **self-study** recipe (synthetic conversations
+ context distillation) closes the gap, matching ICL with **38.6× less memory** and **26.4× higher
throughput**.

## Problem & motivation
Grounding answers in a large corpus (codebase, legal docs, chat history) by stuffing it into the
context window is accurate but expensive: KV memory scales with input length and is re-paid every
query. Training a **reusable, compact** cache per corpus amortizes that cost — but only if it can
match ICL quality, which naive training does not.

## Key idea
**Self-study**: rather than train the cache with next-token prediction on the raw corpus, generate
**synthetic conversations** about the corpus and train the Cartridge with a **context-distillation**
objective so it reproduces the behavior of the full-context model:

$$
\min_{\text{Cartridge}}\; \mathrm{KL}\big(p_{\text{full-context}}(\cdot\mid x) \,\|\, p_{\text{Cartridge}}(\cdot\mid x)\big)
$$

## How it works
- For a fixed corpus, synthesize diverse Q/A-style conversations grounded in it.
- Train a small KV cache (the Cartridge) via **context distillation** against the full-context
  model's outputs.
- At inference, **load** the Cartridge (no corpus in context) and decode; cost is amortized across
  every query referencing that corpus. Cartridges are **composable** at inference without retraining.

## Training / data
Offline per-corpus training; synthetic self-study conversations; context-distillation objective.

## Results
| Metric | Result | Notes |
|---|---|---|
| Memory vs ICL | **38.6×** less | matches ICL quality (per abstract) |
| Throughput vs ICL | **26.4×** higher | per abstract |
| Effective context (MTOB) | **128k → 484k** | per abstract |
| Composition | multiple Cartridges without retraining | per abstract |

## Limitations & follow-ups
- **Per-corpus offline training** cost (amortized only if the corpus is queried many times).
- Query-agnostic but **corpus-specific**. Fast latent-space alternative:
  [Attention Matching](kvcache_2026_attention-matching.md).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2506.06266) · [pdf](https://arxiv.org/pdf/2506.06266)
- **Code:** https://github.com/HazyResearch/cartridges
- **Hugging Face:** —
- **Related:** [KV-cache compression thread](../context/kv_cache/kv_cache.md) · [Soft-token compression thread](../context/soft_token/soft_token.md)
