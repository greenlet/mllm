# Thread: Multimodal soft-token bridges

How vision/audio encoders feed a frozen-or-tuned language model through *continuous
soft tokens* — the closest analogy to MixedDecoder's "do with text what models do with
images." The throughline is a small **bridge** that turns encoder features into a handful
of LLM-space vectors; the design choices in that bridge (how many tokens, learned queries
vs. raw features, frozen vs. trainable LLM, prefix vs. per-layer injection) are exactly the
choices MixedDecoder gets wrong. See the framing axes — **capacity / addressability /
forcing function** — in [mixed_decoder.md §3.0](../mixed_decoder.md).

## Evolution

| Paper | Year | Core contribution | Key lesson for the bridge | Relevance to MixedDecoder |
|---|---|---|---|---|
| [Perceiver / Perceiver IO](https://arxiv.org/abs/2103.03206) | 2021 | A fixed set of **latent vectors cross-attends** to an arbitrarily large input array, decoupling compute from input size. | *Learned latent queries as an information bottleneck.* | The component MixedDecoder is missing between BERT and Qwen — a real cross-attention resampler instead of a linear blow-up of one `[CLS]`. **(addressability + capacity)** |
| [Flamingo](../../papers/multimodal_2022_flamingo-perceiver-resampler.md) | 2022 | Frozen vision encoder → **Perceiver Resampler** → fixed *K* latents → frozen LM via **gated cross-attention** at every layer (zero-init `tanh` gate). | Bridge emits *many* tokens; **LM frozen**; conditioning enters at *every* layer; zero-init gating stabilizes cold-start. | Four direct fixes: more tokens, freeze the decoder, layer-wise (not prefix-only) injection, stable cold-start. **(all three axes)** |
| [BLIP-2 / Q-Former](../../papers/multimodal_2023_blip2-qformer.md) | 2023 | **K = 32 learned queries** cross-attend to frozen image features → 32 soft tokens to a frozen LLM; **two-stage** curriculum (representation learning, then LLM bridging). | Multiple *independent* queries specialize; separate *extract* from *express*; masks control the objective. | The single most relevant paper: replaces single-`[CLS]` + ×4 linear with K addressable slots, and its two-stage curriculum is the cure for MixedDecoder's Phase-3 stagnation. **(addressability + forcing)** |
| [LLaVA](https://arxiv.org/abs/2304.08485) | 2023 | Keep **all** patch tokens (no bottleneck) + a simple linear projector; fine-tune the LLM on visual-instruction data. | A linear/MLP projector works **only at low compression** — because it pays full token cost. | MixedDecoder uses a LLaVA-grade projector at a Q-Former-grade compression ratio — the worst of both. **(capacity)** |
| [LLaVA-1.5](https://arxiv.org/abs/2310.03744) | 2023 | 2-layer GELU MLP projector + better data; strong results without a resampler. | At low compression, projector *quality* matters more than a learned-query bottleneck. | Confirms: if you insist on heavy compression you need Q-Former machinery, not a bigger projector. **(capacity)** |
| [InstructBLIP](https://arxiv.org/abs/2305.06500) | 2023 | **Instruction-aware Q-Former**: the queries also attend to the instruction text, so the extracted tokens depend on the task. | Make the bridge *query-conditioned* on the actual question. | Directly attacks MixedDecoder's "reconstruction geometry ≠ queryable geometry": let the question steer what the slots extract. **(addressability + forcing)** |
| [Honeybee](https://arxiv.org/abs/2312.06742) | 2023 | **Locality-enhanced projector** that preserves spatial/positional locality while abstracting. | Preserving *input locality* measurably helps fine-grained tasks (OCR, counting). | Text analog: preserve *token-order locality* → slot-per-region pooling should improve recall of specific names/numbers vs. a single global `[CLS]`. **(capacity + addressability)** |

## Why this thread matters for MixedDecoder

- The user's mental model — "do with text what models do with images" — *is* the VLM bridge
  problem, and the literature is unambiguous about what makes it work.
- Today's MixedDecoder bridge is a single `[CLS]` per 128-token chunk blown up by a linear
  `emb_exp` into 4 rank-≤1 copies of the same vector. That adds **no information** (a
  deterministic linear map cannot raise capacity) and produces a *reconstruction*-organized
  geometry, not a *query*-addressable one.
- Every successful bridge above does the opposite on at least one axis MixedDecoder fails:
  emits **K independent slots** (Perceiver, BLIP-2), **freezes the LLM** so the bridge must
  do the representation work (Flamingo, BLIP-2), injects context **at every layer** (Flamingo),
  or makes extraction **query-conditioned** (InstructBLIP).
- LLaVA/LLaVA-1.5 are the instructive counter-example: a plain projector works *only because
  it does not compress*. MixedDecoder pairs a LLaVA-grade projector with a Q-Former-grade
  compression ratio — the worst of both worlds.

> **Takeaway:** replace single-`[CLS]` + linear expansion with a *real resampler* (K learned
> queries cross-attending to the encoder's full 128 hidden states), freeze/LoRA the decoder,
> and consider layer-wise gated cross-attention instead of prefix-only conditioning. See the
> concrete experiments in [mixed_decoder.md §4.1–§4.2 and §4.6](../mixed_decoder.md).

## Lineage note

Flamingo's Perceiver Resampler → BLIP-2's Q-Former is one of the cleanest design-evolution
stories in VLMs; BLIP-2 frames the Q-Former as "similar to the Perceiver Resampler in
Flamingo" but adds the stage-1 representation-learning curriculum that turns the bridge from
*expressive* into *extractive*. That added stage is precisely the forcing function MixedDecoder's
task-CE-only training lacks.

## See also

- [Text context compression into embeddings](../compression/compression.md) — the *text* analog
  of this bridge (ICAE / Gisting / xRAG); same lessons at 4–16× instead of vision's token-grid.
- [Retrieval & late interaction](../retrieval/retrieval.md) — ColBERT's per-token vectors are the
  retrieval-side argument for *many addressable slots* over one gist.
- [Soft-prompt & prefix conditioning](../softprompt/softprompt.md) — the mechanism (continuous
  vectors in the embedding slot) MixedDecoder and every bridge here share.
- [mixed_decoder.md §3.1](../mixed_decoder.md) — full prose analysis this thread condenses.
- [mllm/model/mixed_decoder.py](../../../mllm/model/mixed_decoder.py) — the `run_enc` / `emb_exp`
  bottleneck this thread argues to replace.

## Open follow-ups for this thread

- **Q-Former-for-text head** — implement K learned queries over BERT's full 128 hidden states;
  sweep `K ∈ {4, 8, 16, 32}` against the linear-expansion baseline. TODO experiment ([§4.1](../mixed_decoder.md)).
- **Gated cross-attention (Flamingo/CEPE-style)** — re-read the compressed memory at every Qwen
  layer with zero-init gates, vs. prefix-only. TODO experiment ([§4.6](../mixed_decoder.md)).
- **Instruction-aware bridge (InstructBLIP)** — feed the question into the resampler so slots are
  query-conditioned. TODO experiment.
- **Honeybee locality variant** — slot-per-region pooling over strided spans; measure name/number
  recall. TODO experiment.
- **Paper recaps to add:** Perceiver / Perceiver IO, LLaVA / LLaVA-1.5, InstructBLIP, Honeybee
  (only Flamingo and BLIP-2 have local recaps so far).
