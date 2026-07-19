# Thread: Agentic memory & frameworks (the outer loop)

Everything else in the [LCLM](../ctx_compression.md) references works *inside* the context
window — compress the tokens, shrink the KV, cheapen the attention. This thread is about the
**outer loop**: systems that decide *what enters the window at all*, paging information between a
bounded working context and an unbounded external store, or recursively decomposing a giant
input into sub-queries. LCLM meets this world through one primitive — the **`EXPAND(i)`** tool —
and the authors explicitly flag **composing LCLMs with [Recursive Language Models][RLM]** as
future work. This thread places that composition on the map.

> **The through-line.** Compression gives the agent **global visibility** (skim a whole corpus
> as cheap latents); agentic memory gives it **selective exact recall** (page in / expand only
> what matters). LCLM's `EXPAND(i)` is the hinge between the two: latents for breadth, raw
> re-tokenization for depth, chosen per step by an outer controller.

## The landscape

| Framework | Year | Mechanism | What it manages | Relation to LCLM |
|---|---|---|---|---|
| [MemGPT][MemGPT] | 2023 | **LLM as an OS**: a fixed "main context" plus paged "external context"; the model issues function calls to **page memory in/out**. | Virtual-memory-style paging of context | The paging controller LCLM's `EXPAND(i)` fits into: compressed latents are the *main context*; expansion pages in the exact page. |
| [MemoryBank][MemoryBank] | 2023 | Retrieval-based **long-term memory** with a Ebbinghaus-style **forgetting curve**; store, retrieve, and decay past interactions. | Persistent, decaying episodic memory | Suggests *what to keep compressed vs. drop*: a density/recency signal for adaptive-ratio compression. |
| [A-Mem][AMem] | 2025 | **Agentic memory**: self-organizing notes that the agent links, updates, and re-indexes as it works (Zettelkasten-style). | Structured, evolving working memory | Compressed latents could be the **note payloads**; the agent links/expands them instead of storing raw text. |
| [Recursive Language Models][RLM] | 2025 | **Recursively** decompose a huge context: the model calls itself on sub-spans and aggregates, so effective context is unbounded. | Divide-and-conquer over giant inputs | The composition the LCLM authors flag: RLM chooses *which spans*; LCLM makes each span **cheap to hold** as latents — breadth × depth without OOM. |

## How LCLM plugs into the outer loop

LCLM §4.3 already prototypes the hinge:

```
  corpus ──► split into 512-tok chunks ──► compress each ──► id-tagged latent memory
                                                    │
              agent skims ALL latents (global view) │
                                                    ▼
                        EXPAND(i)  ──► returns chunk i's raw text (exact detail)
```

- **Latents = cheap global memory.** The agent reads the whole compressed corpus at
  $16\times$ cost, then spends exact-token budget only on the chunk(s) it expands — on RULER
  needle tasks this recovers **+17–20 avg** exact-match over the raw compressor.
- **`EXPAND(i)` = the page-in op.** It is MemGPT's "page from external context" specialized to
  "re-tokenize this compressed span," and RLM's "recurse into this sub-span" specialized to
  "materialize this sub-span."
- **The controller is the open design.** *When* is exact re-tokenization worth the latency?
  MemoryBank's decay, A-Mem's linking, and RLM's recursion are candidate policies for that
  decision.

## Why this thread matters for the repo

- It is where the repo's [MixedDecoder](../../mixed_decoder/mixed_decoder.md) becomes a
  **memory substrate**, not just a compressor: its chunk latents are exactly the id-tagged
  payloads an outer agent would skim-then-expand.
- It reframes the compressor's job: not "reconstruct everything through the bottleneck" but
  "provide **queryable global visibility** with an escape hatch to exact detail" — which is a
  much more achievable target than lossless high-ratio compression.
- It supplies the natural home for [generated-state compression](../soft_token/soft_token.md)
  (long CoT, tool observations): the agent's working history *is* agentic memory, and
  compressing it is the same primitive applied to the outer loop's own transcript.

## Relation to the neighboring threads

- **Soft-token compression** ([thread](../soft_token/soft_token.md)) — supplies the latents that
  become memory payloads; the `EXPAND(i)`/REFRAG expand-on-demand op is the shared hinge.
- **Inference engines & systems** ([thread](../systems/systems.md)) — [SGLang][SGLang]'s
  structured multi-call runtime with prefix reuse is where this skim-then-expand loop actually
  executes.
- **Long-context benchmarks** ([thread](../benchmarks/benchmarks.md)) — RULER needle tasks are
  the setting where the agentic loop's exact-recall gain is measured.
- **Efficient long-sequence modeling** ([thread](../long_seq/long_seq.md)) — the *architectural*
  route to unbounded context; agentic memory is the *orchestration* route. Both compose with
  input compression.

## Open follow-ups for this thread

- **Expand-policy learning.** Compare a fixed heuristic vs. a learned/RL controller (REFRAG-style)
  for *when* to call `EXPAND(i)`; measure the latency/accuracy trade. TODO experiment.
- **Compress the working history.** Apply the compressor to the agent's own CoT + tool
  observations (generated-state compression), not just the static input. TODO experiment.
- **LCLM × RLM composition.** Prototype recursive decomposition where each recursion level holds
  its span as latents; check whether effective context scales without OOM. TODO experiment.
- **Paper recaps to add.** None of the entries here has a local recap yet (Recursive Language
  Models, MemGPT, A-Mem, MemoryBank). TODO recaps.

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — §4.3 (agentic latent
  context) is the prototype this thread generalizes; §7.10 is the reference list it expands.
- [Soft-token / encoder–decoder context compression](../soft_token/soft_token.md) — the latents
  that serve as memory payloads.
- [Inference engines & systems](../systems/systems.md) — the SGLang runtime hosting the loop.
- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) — the repo's compressor as a memory substrate.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[RLM]: https://arxiv.org/abs/2512.24601 "Recursive Language Models (Zhang et al. 2025)"
[MemGPT]: https://arxiv.org/abs/2310.08560 "MemGPT (Packer et al. 2023)"
[AMem]: https://arxiv.org/abs/2502.12110 "A-Mem (Xu et al. 2025)"
[MemoryBank]: https://arxiv.org/abs/2305.10250 "MemoryBank (Zhong et al. 2023)"
[SGLang]: https://arxiv.org/abs/2312.07104 "SGLang (Zheng et al. 2024)"
