# Thread: Inference engines & systems (why soft tokens deploy unchanged)

[LCLM](../ctx_compression.md)'s central *systems* claim is deceptively small: its compressed
context is a sequence of **ordinary continuous embeddings**, so it flows through today's
serving stacks with **no kernel or scheduler changes**. That single property is what separates
soft-token compression from [KV-cache eviction](../kv_cache/kv_cache.md), whose per-head,
query-dependent caches fight the very abstractions these engines are built on. This thread
collects the systems those claims are measured against — the paged-KV allocator ([vLLM][vLLM]),
the structured-execution runtime ([SGLang][SGLang]), and the reference training/inference
library ([HF Transformers][HFTransformers]).

> **The through-line.** A soft token occupies the same slot as a token embedding, so
> **[PagedAttention][PagedAttn]'s block allocator, prefix caching, and continuous batching all
> apply verbatim.** KV-cache compression breaks at least one of these (uniform per-token blocks,
> reusable prefixes) — which is why LCLM can claim engine compatibility and its baselines cannot.

## The landscape

| System | Year | What it provides | Why it matters to LCLM |
|---|---|---|---|
| [PagedAttention / vLLM][vLLM] | 2023 | KV cache stored in **fixed-size blocks** with an OS-style page table; near-zero fragmentation, **prefix sharing**, continuous batching. | LCLM's soft tokens are embeddings, so they page and batch **exactly like real tokens** — no non-uniform per-head layout. The efficiency numbers (TTFT, peak memory) are reported on this class of engine. |
| [SGLang][SGLang] | 2024 | **Structured LM program** runtime: RadixAttention prefix reuse across calls, and a front-end for multi-call / tool workflows. | The natural host for LCLM's **`EXPAND(i)`** agentic loop (skim compressed corpus → expand a chunk); prefix reuse across turns is where compressed context that *reuses across turns* pays off vs. query-dependent KV caches. |
| [HF Transformers][HFTransformers] | 2020 | The reference model/attention/generation implementation and checkpoint hub. | Where the encoder ([Qwen3-Embedding][Qwen3Emb]) + adapter + decoder ([Qwen3][Qwen3]) are implemented and released; soft-token ingestion is just an `inputs_embeds` path, not a custom op. |

## Why the "unchanged engine" property is the whole point

- **Uniform layout.** Paged engines assume every position contributes the **same-shaped** KV
  block. Soft tokens preserve this (they are embeddings that produce standard KV on prefill);
  [SnapKV][SnapKV]/[KVzip][KVzip]-style **non-uniform per-head eviction** does not, so it needs
  bespoke kernels the mainstream engines don't ship.
- **Prefix reuse across turns.** Query-dependent KV compression can't be shared between
  requests with different queries; a **query-agnostic** compressed context can, matching
  vLLM/SGLang prefix caching.
- **Compress *before* prefill.** KV methods compress *after* a full prefill (cost ≈ vertical
  line, ratio-independent). LCLM shortens the sequence *before* the decoder prefill, so the
  engine simply sees a shorter input — higher ratio directly cuts compute and memory.
- **Deployment path is `inputs_embeds`.** No decoding change: the adapter output is fed in the
  embedding slot, exactly like a VLM feeds image tokens ([multimodal thread](../multimodal/multimodal.md)).

## Why this thread matters for the repo

- It is the **portability argument** for the repo's own
  [MixedDecoder](../../mixed_decoder/mixed_decoder.md): whatever bridge it adopts, keeping the
  decoder's input a plain embedding sequence means it can be served on vLLM/SGLang **without
  writing kernels** — a hard constraint that rules out clever-but-incompatible tricks.
- It defines *how* the efficiency half of the [benchmark bar](../benchmarks/benchmarks.md) is
  measured: **TTFT and peak memory on a paged engine**, not FLOP counts in isolation.
- It points at the runtime (**SGLang**) where an expand-on-demand / agentic-memory design would
  actually live, linking the architecture to a concrete serving story.

## Relation to the neighboring threads

- **KV-cache compression** ([thread](../kv_cache/kv_cache.md)) — the family whose engine
  *incompatibility* is the foil for this thread; its costs show up as ratio-independent
  vertical lines precisely because of the prefill-then-evict shape.
- **Soft-token compression** ([thread](../soft_token/soft_token.md)) — the methods that inherit
  the engine-compatibility property by construction.
- **Long-context benchmarks** ([thread](../benchmarks/benchmarks.md)) — supplies the accuracy axis
  paired with the TTFT/memory axis these engines produce.
- **Agentic memory & `EXPAND(i)`** — the SGLang-hosted multi-call pattern that turns compressed
  latents (global visibility) + on-demand expansion (exact detail) into a served workflow.

## Open follow-ups for this thread

- **Actually serve MixedDecoder on vLLM.** Wire the compressed embeddings through the
  `inputs_embeds` path and reproduce TTFT/memory curves against a full-KV baseline. TODO experiment.
- **SGLang `EXPAND(i)` prototype.** Implement the skim-then-expand loop as an SGLang program and
  measure prefix-reuse savings across turns. TODO experiment.
- **Paper recaps to add.** vLLM/PagedAttention, SGLang, and HF Transformers are arXiv/ACL links
  only. TODO recaps.

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — §4 reports the
  TTFT/memory numbers on these engines; §7.8 is the reference list this thread expands.
- [KV-cache compression](../kv_cache/kv_cache.md) — the engine-incompatible foil.
- [Soft-token / encoder–decoder context compression](../soft_token/soft_token.md) — the
  engine-compatible method family.
- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) — the repo's compressor whose deployment
  path this thread constrains.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[PagedAttn]: https://arxiv.org/abs/2309.06180 "PagedAttention / vLLM (Kwon et al. 2023)"
[vLLM]: https://arxiv.org/abs/2309.06180 "vLLM (Kwon et al. 2023)"
[SGLang]: https://arxiv.org/abs/2312.07104 "SGLang (Zheng et al. 2024)"
[HFTransformers]: https://aclanthology.org/2020.emnlp-demos.6 "HuggingFace Transformers (Wolf et al. 2020)"
[SnapKV]: https://arxiv.org/abs/2404.14469 "SnapKV (Li et al. 2024)"
[KVzip]: https://arxiv.org/abs/2505.23416 "KVzip (Kim et al. 2025)"
[Qwen3]: https://arxiv.org/abs/2505.09388 "Qwen3 Technical Report (2025)"
[Qwen3Emb]: https://arxiv.org/abs/2506.05176 "Qwen3 Embedding (2025)"
