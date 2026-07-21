# Thread: Hard-token / prompt compression

The line of work that shortens a long prompt by operating on **discrete tokens** — *deleting*
low-information tokens or *paraphrasing/summarizing* spans into fewer real tokens — so the
compressed prompt is still **plain text** a frozen LM can read unchanged. Unlike
[soft-token compression](../soft_token/soft_token.md) (replace tokens with continuous
embeddings) or [KV-cache compression](../kv_cache/kv_cache.md) (shrink the cache *after*
prefill), hard-token methods keep the interface **fully text-in / text-out**: no encoder, no
adapter, no model surgery — which is why they are the easiest to deploy and the most
intrinsically **lossy** on exact detail.

## Evolution

| Paper | Year | Core contribution | Compression unit | Query-aware? |
|---|---|---|---|---|
| [Selective Context][SelectiveCtx] | 2023 | Score tokens/phrases by **self-information** from a small LM and drop the least informative; first clean "prune by information content". | token / phrase | no |
| [LLMLingua][LLMLingua] | 2023 | **Coarse-to-fine**: budget controller + iterative token-level perplexity pruning with a small LM, plus a distribution-alignment step; up to ~20× with little loss. | token (budgeted) | no |
| [LongLLMLingua][LongLLMLingua] | 2024 | Adds **question-aware** coarse+fine scoring and **document reordering** to fight the [lost-in-the-middle][LostMiddle] effect for long-context RAG. | doc + token | **yes** |
| [NL-Prompt][NLPrompt] | 2024 | Learn to compress a prompt **into natural language** (a summary/abstractive form), not just delete tokens — trains the compressor explicitly. | abstractive rewrite | optional |
| [CompAct][CompAct] | 2024 | **Actively** compress retrieved documents with an LM that iteratively keeps only query-relevant content across chunks; RAG-focused. | document (iterative) | **yes** |

## The shared recipe

Every member maps text → shorter text; the design space is the *unit*, the *scorer*, and
whether the query is in the loop:

```
long prompt ──► [Scorer] ──► [Select / Rewrite] ──► shorter TEXT ──► [frozen LM]
                   │                │
        self-information       delete tokens (LLMLingua, Selective Context)
        perplexity (small LM)  paraphrase/summarize (NL-Prompt, CompAct)
        query relevance        reorder docs (LongLLMLingua)
```

- **Extractive vs abstractive** — *delete* tokens/sentences ([Selective Context][SelectiveCtx],
  [LLMLingua][LLMLingua]) keeps surviving text verbatim; *rewrite* into fewer tokens
  ([NL-Prompt][NLPrompt], [CompAct][CompAct]) is denser but can hallucinate.
- **Scorer** — a **small LM** provides the information/perplexity signal
  ([LLMLingua][LLMLingua], [Selective Context][SelectiveCtx]); [CompAct][CompAct] uses a full
  LM as the compressor itself.
- **Query-aware vs query-agnostic** — [LongLLMLingua][LongLLMLingua] / [CompAct][CompAct]
  condition on the question (better for RAG, but the compressed prompt **doesn't reuse** across
  queries); [LLMLingua][LLMLingua] / [Selective Context][SelectiveCtx] are query-agnostic and
  reusable.
- **Budget control** — an explicit token budget / target ratio ([LLMLingua][LLMLingua]'s budget
  controller) versus an implicit relevance threshold.

## Why this thread matters for the repo

- It is the **third baseline family** (alongside [KV-cache](../kv_cache/kv_cache.md)) that
  [LCLM](../ctx_compression.md) and [MixedDecoder](../../mixed_decoder/mixed_decoder.md) are
  contrasted with — and the clearest illustration of the **fidelity ceiling** that motivates
  soft tokens: once a token is deleted or paraphrased, exact lexical/structural detail is
  **unrecoverable**, so hard-token methods can't match soft tokens on exact-recall tasks
  ([RULER][RULER] needle-in-a-haystack, [GSM8K][GSM8K]).
- It is the **cheapest to adopt** — pure text-in/text-out, no training of the target model, no
  engine changes — so it sets the "free baseline" bar any soft-token gain must clear.
- Its **query-aware** variants ([LongLLMLingua][LongLLMLingua], [CompAct][CompAct]) mirror the
  same query-dependent-vs-reusable tension seen in [KV-cache](../kv_cache/kv_cache.md)
  ([SnapKV][SnapKV] vs [KVzip][KVzip]): conditioning on the query helps accuracy but kills
  cross-turn reuse.

## Relation to the neighboring threads

- **Soft-token compression** ([thread](../soft_token/soft_token.md)) — compresses into
  *continuous* embeddings that can carry sub-token information; hard tokens are bounded by the
  vocabulary and by what survives deletion. The two are **stackable** (prune text, then encode).
- **KV-cache compression** ([thread](../kv_cache/kv_cache.md)) — acts on cached K/V *after*
  prefill; hard-token methods shorten the text *before* prefill, so they also cut the prefill
  cost that KV methods cannot. Complementary.
- **Agentic memory** ([Recursive Language Models][RLM], [MemGPT][MemGPT]) — abstractive
  summarization ([CompAct][CompAct], [NL-Prompt][NLPrompt]) is exactly the "summarize old
  context" primitive these frameworks use for long-term memory, with expand-on-demand to
  recover dropped detail.

## Open follow-ups for this thread

- **Fidelity vs ratio floor** — a controlled measurement of where extractive/abstractive
  pruning breaks on exact-recall tasks, versus soft tokens at the same ratio. TODO recap.
- **Stacking hard + soft** — prune obvious filler with [LLMLingua][LLMLingua], then compress the
  remainder into latents; does the composition beat either alone on the Pareto frontier?
- **Query-agnostic abstractive compression** — can a [CompAct][CompAct]-style rewrite be made
  reusable across queries without the [LongLLMLingua][LongLLMLingua] per-query recompute?
- **Compressor distillation** — replace the small-LM perplexity scorer with a cheaper learned
  saliency model at fixed quality.

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — names hard-token
  compression as intrinsically lossy and positions soft tokens against it.
- [Soft-token / encoder–decoder context compression](../soft_token/soft_token.md) — the
  continuous-embedding counterpart and the repo's own direction.
- [KV-cache compression](../kv_cache/kv_cache.md) — the *compress-after-prefill* baseline family.
- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) — the repo's soft-token compressor.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[LLMLingua]: https://arxiv.org/abs/2310.05736 "LLMLingua (Jiang et al. 2023)"
[LongLLMLingua]: https://arxiv.org/abs/2310.06839 "LongLLMLingua (Jiang et al. 2024)"
[SelectiveCtx]: https://arxiv.org/abs/2310.06201 "Selective Context (Li et al. 2023)"
[NLPrompt]: https://arxiv.org/abs/2402.18700 "Learning to Compress Prompt in Natural Language (Chuang et al. 2024)"
[CompAct]: https://arxiv.org/abs/2407.09014 "CompAct (Yoon et al. 2024)"
[SnapKV]: ../../papers/kvcache_2024_snapkv.md "SnapKV (Li et al. 2024)"
[KVzip]: ../../papers/kvcache_2025_kvzip.md "KVzip (Kim et al. 2025)"
[LostMiddle]: https://arxiv.org/abs/2307.03172 "Lost in the Middle (Liu et al. 2024)"
[RULER]: https://arxiv.org/abs/2404.06654 "RULER (Hsieh et al. 2024)"
[GSM8K]: https://arxiv.org/abs/2110.14168 "GSM8K (Cobbe et al. 2021)"
[RLM]: https://arxiv.org/abs/2512.24601 "Recursive Language Models (Zhang et al. 2025)"
[MemGPT]: https://arxiv.org/abs/2310.08560 "MemGPT (Packer et al. 2023)"
