# Thread: KV-cache compression

The line of work that shrinks the **[KV cache][KVCache]** — the per-token key/value tensors a
decoder caches during autoregressive generation — *after* the context has been read, so that
long-context decoding stays within a memory budget. Unlike
[soft-token compression](../soft_token/soft_token.md) (replace tokens with continuous
embeddings *before* decoding) or [hard-token compression][LLMLingua] (delete/paraphrase real
tokens), KV-cache methods keep the input **verbatim** and instead **evict, merge, quantize, or
re-derive** cache entries. They are the primary **baselines** [LCLM](../ctx_compression.md)
measures against, and the family whose structural costs (full prefill, per-head eviction,
engine incompatibility) motivate the soft-token route.

## Evolution

| Paper | Year | Core contribution | Compression knob | Query-dependent? |
|---|---|---|---|---|
| [KVQuant][KVCache] | 2024 | **Quantize** keys/values to ~2–4 bits with per-channel/pre-RoPE schemes; enables 10M-token context on limited memory. | bits per entry | no (lossy but uniform) |
| [SnapKV][SnapKV] | 2024 | Use an **observation window** of recent queries to score which prompt KV entries matter, then **evict** the rest before generation. | # retained tokens/head | **yes** (query-aware) |
| [Cartridges][Cartridges] | 2025 | **Self-study**: train a small, reusable KV "cartridge" offline for a fixed corpus, amortizing prefill across many queries. | trained cartridge size | no (corpus-specific) |
| [Exploiting Sparsity][SparseLC] | 2025 | Systems/algorithm co-design exploiting **attention sparsity** for long-context inference. | sparse block budget | partial |
| [Expected Attention][ExpAttn] | 2025 | Estimate each entry's importance from the **expected attention of future queries**, giving a query-agnostic eviction score. | # retained tokens | no (marginalizes queries) |
| [KVzip][KVzip] | 2025 | **Query-agnostic** compression: score entries by how well the cache can **reconstruct the context**, evicting redundant KV so one compressed cache serves *any* future query. | reconstruction budget | no (reusable) |
| [Fast KVzip][FastKVzip] | 2026 | Adds a **gated eviction** mechanism for cheaper, faster KVzip-style compaction. | gate threshold | no |
| [Attention Matching][AM] | 2026 | **Fast KV compaction** by matching the compressed cache's attention output to the full cache — a distillation-style objective, engine-friendlier. | compaction ratio | no |

## The shared recipe

Every member operates on the **already-computed** KV tensors; the design space *is* which
operation and which importance signal:

```
full context ──► [Prefill] ──► KV cache  K,V ∈ ℝ^{L×H×d}
                                   │
                    ┌──────────────┼──────────────┬───────────────┐
                    ▼              ▼              ▼               ▼
                 Evict          Merge          Quantize        Re-derive
              (SnapKV,        (compaction     (KVQuant)      (Cartridges:
               KVzip,          via AM)                        train offline)
               ExpAttn)
                    │
        importance signal: recent-query attention (SnapKV) │ reconstruction (KVzip)
                            expected future attention (ExpAttn) │ attention match (AM)
```

- **When you compress** — *after* prefill. This is the defining trait: the model must read the
  **full context at full length** before any savings materialize, so **time-to-first-token
  (TTFT) is dominated by prefill** and is largely *independent* of the compression ratio.
- **What you compress** — evict whole entries ([SnapKV][SnapKV], [KVzip][KVzip],
  [Expected Attention][ExpAttn]), merge/compact them ([Attention Matching][AM]), or quantize
  them in place ([KVQuant][KVCache]).
- **Importance signal** — recent-query attention ([SnapKV][SnapKV]), context-reconstruction
  ([KVzip][KVzip]), expected attention of *future* queries ([Expected Attention][ExpAttn]), or
  matching the full cache's attention output ([Attention Matching][AM]).
- **Query-dependent vs reusable** — [SnapKV][SnapKV]-style caches are tuned to the *current*
  query and don't reuse across turns; **query-agnostic** methods ([KVzip][KVzip],
  [Expected Attention][ExpAttn]) build one compressed cache that serves any subsequent query
  — the trait that makes them competitive for multi-turn / agentic reuse.

## Why this thread matters for the repo

- It is the **baseline family** the whole [ctx_compression](../ctx_compression.md) /
  [MixedDecoder](../../mixed_decoder/mixed_decoder.md) direction is measured against. LCLM's
  headline result is that soft tokens land on a **better accuracy/TTFT/memory Pareto frontier**
  than these methods — so understanding *why* KV-cache compression is bounded is the argument
  for soft tokens.
- It exposes three structural limits soft-token compression is designed to avoid:
  1. **Prefill is unavoidable** — savings come only *after* reading the full context, so KV
     methods appear as near-**vertical lines** on the time/quality plot (cost ≈ constant in
     the ratio).
  2. **Per-head, non-uniform eviction** ([SnapKV][SnapKV]) is **incompatible with
     [paged-attention][PagedAttn] engines** ([vLLM][vLLM] / [SGLang][SGLang]), which assume
     uniform page layout — a deployment blocker soft tokens sidestep entirely.
  3. **Query-dependent caches don't reuse** across turns, unlike a precomputed soft-token
     memory or a query-agnostic ([KVzip][KVzip]) cache.
- It defines the **efficiency axes** ([TTFT][PagedAttn] and peak GPU memory) that MixedDecoder's
  evaluation currently skips.

## Relation to the neighboring threads

- **Soft-token compression** ([soft-token thread](../soft_token/soft_token.md)) — compresses
  *before* decoding into continuous embeddings; **parallelizable** and **engine-native**, where
  KV methods compress *after* a full prefill. The two are the central contrast in
  [LCLM](../ctx_compression.md).
- **Hard-token compression** ([LLMLingua][LLMLingua]) — deletes/paraphrases the *input* text;
  KV methods keep text verbatim and act on the cache instead. Complementary and stackable.
- **Architectural KV reduction** ([MLA][MLA] / DeepSeek, [GQA-style] grouping) — reduces the
  KV footprint *by design* at pretraining time; KV-cache **compression** is a *post-hoc* layer
  on top of any such architecture. [KimiLinear][KimiLinear] and [Mamba][Mamba]-family models
  attack the same memory wall by replacing softmax attention outright.
- **Agentic memory** ([Recursive Language Models][RLM], [MemGPT][MemGPT]) — a compressed cache
  ([Cartridges][Cartridges]) is one substrate for reusable long-term memory; expand-on-demand
  tools can recover exact detail the compression dropped.

## Open follow-ups for this thread

- **Reconciling eviction with paged attention** — can non-uniform per-head eviction
  ([SnapKV][SnapKV]) be reshaped into page-aligned budgets so [vLLM][vLLM] / [SGLang][SGLang]
  keep it? TODO recap.
- **Query-agnostic reuse** — how far can [KVzip][KVzip] / [Expected Attention][ExpAttn]
  push a single compressed cache across many turns before accuracy degrades?
- **Compression vs quantization stacking** — [KVQuant][KVCache] (bits) × eviction (entries) ×
  compaction ([AM][AM]) — where is the combined Pareto point?
- **KV compression vs soft tokens on the *same* budget** — a controlled head-to-head at fixed
  memory, the comparison [LCLM](../ctx_compression.md) runs and MixedDecoder should reproduce.

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — the paper that
  benchmarks against this thread and argues soft tokens dominate its Pareto frontier.
- [Soft-token / encoder–decoder context compression](../soft_token/soft_token.md) — the
  *compress-before-decode* counterpart and the repo's own direction.
- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) — the repo's soft-token compressor; these
  methods are its efficiency baselines.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[SnapKV]: https://arxiv.org/abs/2404.14469 "SnapKV (Li et al. 2024)"
[KVzip]: https://arxiv.org/abs/2505.23416 "KVzip (Kim et al. 2025)"
[FastKVzip]: https://arxiv.org/abs/2601.17668 "Fast KVzip (Kim et al. 2026)"
[ExpAttn]: https://arxiv.org/abs/2510.00636 "Expected Attention (Devoto et al. 2025)"
[AM]: https://arxiv.org/abs/2602.16284 "Fast KV Compaction via Attention Matching (Zweiger et al. 2026)"
[KVCache]: https://arxiv.org/abs/2401.18079 "KVQuant (Hooper et al. 2024)"
[Cartridges]: https://arxiv.org/abs/2506.06266 "Cartridges (Eyuboglu et al. 2025)"
[SparseLC]: https://arxiv.org/abs/2502.06766 "Exploiting Sparsity for Long-Context Inference (Synk et al. 2025)"
[LLMLingua]: https://arxiv.org/abs/2310.05736 "LLMLingua (Jiang et al. 2023)"
[PagedAttn]: https://arxiv.org/abs/2309.06180 "PagedAttention / vLLM (Kwon et al. 2023)"
[vLLM]: https://arxiv.org/abs/2309.06180 "PagedAttention / vLLM (Kwon et al. 2023)"
[SGLang]: https://arxiv.org/abs/2312.07104 "SGLang (Zheng et al. 2024)"
[MLA]: https://arxiv.org/abs/2405.04434 "DeepSeek-V2 / MLA (Liu et al. 2024)"
[KimiLinear]: https://arxiv.org/abs/2510.26692 "Kimi Linear (2025)"
[Mamba]: https://arxiv.org/abs/2312.00752 "Mamba (Gu & Dao 2024)"
[RLM]: https://arxiv.org/abs/2512.24601 "Recursive Language Models (Zhang et al. 2025)"
[MemGPT]: https://arxiv.org/abs/2310.08560 "MemGPT (Packer et al. 2023)"
