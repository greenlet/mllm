# Thread: Multimodal / VLM alignment (the staged-recipe analogy)

Why does [LCLM](../ctx_compression.md) train in **four progressively-unfrozen stages**
(adapter warmup → encoder → end-to-end → SFT) instead of end-to-end from scratch? Because the
architecture *is* a vision-language model with the vision tower swapped for a **text
encoder**: a frozen-ish backbone ([Qwen3-Embedding-0.6B][Qwen3Emb]) produces features, a small
**MLP projector** ([LLaVA][LLaVA]-style) maps them into the decoder's embedding space, and a
causal decoder ([Qwen3-4B][Qwen3]) consumes the projected vectors as soft tokens. Every design
knob — projector shape, what to freeze when, how many tokens per input region — was settled
first by the VLM literature. This thread is the **alignment-recipe** half of that literature;
the **bridge-capacity** half (Perceiver / Q-Former / resamplers) lives in the sibling
[Multimodal soft-token bridges](../../mixed_decoder/multimodal/multimodal.md) thread.

> **The mapping in one line.** *image patches → visual tokens → projector → LLM* is
> structurally identical to *text spans → latent tokens → adapter → decoder*. The
> compression ratio $N$ plays the role of ViT's patch size; the staged unfreeze plays the role
> of LLaVA's pretrain-then-instruction-tune schedule.

## Evolution

| Paper | Year | Core contribution | Key lesson for the recipe | Relevance to LCLM |
|---|---|---|---|---|
| [ViT][ViT] | 2020 | An image is **16×16 patches** linearly embedded and fed to a plain Transformer — no convolutions. | *Chunk-then-embed a high-resolution input into a short token sequence.* | The direct analog of LCLM's "block of $N$ tokens → one latent": patch size ⇔ compression ratio $N$; fewer/larger patches trade fidelity for sequence length. |
| [CPVT][CPVT] | 2021 | **Conditional positional encodings** generated from local neighborhoods, so a ViT generalizes across input resolutions without retraining position tables. | *Make positional information depend on content/locality, not a fixed grid.* | Justifies LCLM's per-window ($W=1024$) causal encoding + [RoPE][RoPE]/[YaRN][YaRN] on the decoder side: the compressor must stay resolution-agnostic as context length varies 4K→1M. |
| [LLaVA][LLaVA] ([recap](../../papers/multimodal_2023_llava.md)) | 2023 | Frozen ViT + **linear projector** + LLM, trained in **two stages**: (1) projector-only feature alignment on caption data, then (2) end-to-end visual-instruction tuning. | *Warm up the projector against a frozen backbone before unfreezing anything.* | This is **exactly LCLM Stage 0 → Stage 3**. LCLM cites LLaVA as the template for its progressive unfreeze; training the full model from step 0 degrades it (large early gradients). |
| [LLaVA-1.5][LLaVA] ([recap](../../papers/multimodal_2023_llava-1.5.md)) | 2023 | Swap the linear projector for a **2-layer GELU MLP**, add better data; strong results with no resampler. | *A 2-layer GELU MLP projector is a sufficient bridge — projector quality > exotic connector at moderate compression.* | LCLM's adapter is precisely this: RMSNorm → Linear → GELU → Linear, per-latent and independent (no cross-token mixing). The VLM ablation is why the authors did **not** reach for an attention connector. |
| [Cambrian-1][Cambrian] | 2024 | Vision-centric study of **connector design, data curricula, and unfreezing schedules** across many vision encoders; introduces the Spatial Vision Aggregator. | *The staged schedule and data mix — not just the connector — determine final quality; unfreeze in the right order with the right LR.* | Validates LCLM's stage boundaries and **small-LR decoder unfreeze** in Stage 2; its "which encoder init?" finding echoes LCLM's *embedding-init beats LM-init* result. |
| [Qwen3-VL][Qwen3VL] | 2025 | Production VLM on the same [Qwen3][Qwen3] backbone: native dynamic resolution, MLP patch-merge into visual tokens, [M-RoPE][MRoPE] over the LLM. | *A modern, deployed instance of the whole recipe on the identical decoder family.* | The reference implementation LCLM can literally reuse — same decoder, same soft-token ingestion path, same inference engines ([vLLM][vLLM]/[SGLang][SGLang]). |

## How the VLM recipe maps onto LCLM's four stages

LCLM §3.2 is a text-encoder VLM alignment schedule; each stage has a vision precedent:

```
              VISION-LANGUAGE                         LCLM (text context compression)
 Stage 0  projector warmup (ViT+LLM frozen) ──►  adapter warmup (enc+dec frozen)
 Stage 1  align encoder features             ──►  encoder training (dec frozen)
 Stage 2  end-to-end, small LR (LLaVA-2)     ──►  end-to-end continual pretrain, small LR
 Stage 3  visual-instruction tuning          ──►  SFT on compressed prompts, higher dec LR
```

- **Freeze order.** Both start by adapting *only the bridge* against frozen encoder+decoder so
  early gradients stay smooth; the decoder is not yet used to embedding-model outputs.
- **Small-LR unfreeze.** LLaVA/Cambrian unfreeze the LLM gently to avoid wrecking the
  pretrained weights; LCLM unfreezes the decoder at a **small LR** to align it to compressed
  context *without catastrophic forgetting* (see the [continual-training thread][CatForgetLLM]).
- **Projector = adapter.** The VLM verdict (LLaVA-1.5 vs. resampler ablations) that a 2-layer
  GELU MLP suffices at moderate compression is why LCLM's adapter is an MLP, not an
  attention connector — it beats the attention variant at lower compute.
- **Patchify = compression ratio.** ViT's patch size sets how many visual tokens an image
  becomes; LCLM's $N\in\{4,8,16\}$ sets how many latents a text span becomes. Both are the
  single knob trading sequence length against fidelity.

## Why this thread matters for the repo

- It is the **justification** for LCLM's headline training decision. "Train end-to-end from
  scratch" fails; the fix is borrowed wholesale from VLM alignment, so understanding *why the
  VLM schedule works* explains *why LCLM's schedule works*.
- It tells the repo's own [MixedDecoder](../../mixed_decoder/mixed_decoder.md) which half of the
  VLM literature it is *missing*: MixedDecoder copied the LLaVA **projector** (a cheap MLP) but
  skipped the LLaVA **schedule** (staged, forcing-function-first alignment) — and paired the
  cheap projector with a Q-Former-grade compression ratio.
- It anchors the compression ratio in a well-understood vision knob (patch size), which makes
  the 4×/8×/16× operating points and their fidelity trade-offs intuitive.

## Relation to the neighboring threads

- **Multimodal soft-token bridges** ([thread](../../mixed_decoder/multimodal/multimodal.md):
  Perceiver · Flamingo · BLIP-2/Q-Former · InstructBLIP · Honeybee) — the **capacity /
  addressability** half. That thread argues *how many* soft tokens and *how* to extract them;
  this thread argues *when to unfreeze what*. Together they cover the two halves of VLM design.
- **Soft-token compression** ([thread](../soft_token/soft_token.md): Gist · ICAE · CEPE · xRAG ·
  REFRAG · E2LLM · LCLM) — the *text* instantiation. Its recurring objective lesson
  (reconstruction alone collapses to copying; NTP alone loses exact-string fidelity; the
  mixture generalizes) is the text analog of visual-instruction data replacing caption-only
  pretraining.
- **Backbone components** ([thread](../backbone/backbone.md): Qwen3 · Qwen3-Embedding · RoPE ·
  YaRN · RMSNorm · GELU) — supplies the concrete encoder/decoder and the ViT-side positional
  machinery ([CPVT][CPVT] → [M-RoPE][MRoPE]) this thread references.
- **KV-cache compression** ([thread](../kv_cache/kv_cache.md)) and **hard-token compression**
  ([thread](../hard_token/hard_token.md)) — the non-soft-token alternatives LCLM is measured
  against; unrelated to the alignment recipe but complete the §7 map.

## Open follow-ups for this thread

- **Adaptive "patch size."** ViT variants use multi-scale patches; the text analog is
  **information-density-dependent compression ratio** (LCLM's flagged future work). TODO recap.
- **Instruction-aware alignment.** InstructBLIP conditions the bridge on the question; the
  Stage-3 analog is query-conditioned compression of the long-context segment. Cross-links to
  the [bridges thread](../../mixed_decoder/multimodal/multimodal.md) open items.
- **Paper recaps to add.** Local recaps exist for [LLaVA](../../papers/multimodal_2023_llava.md)
  and [LLaVA-1.5](../../papers/multimodal_2023_llava-1.5.md); **ViT, CPVT, Cambrian-1, and
  Qwen3-VL** are still arXiv/GitHub links only. TODO recaps.

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — §3.2 is the staged
  recipe this thread explains; §7.6 is the reference list this thread expands.
- [Multimodal soft-token bridges](../../mixed_decoder/multimodal/multimodal.md) — the
  bridge-capacity companion (Perceiver / Flamingo / Q-Former).
- [Soft-token / encoder–decoder context compression](../soft_token/soft_token.md) — the text
  family LCLM belongs to.
- [Qwen overview](../../qwen/overview.md) — the [Qwen3][Qwen3] / [Qwen3-VL][Qwen3VL] /
  [Qwen3-Embedding][Qwen3Emb] backbones and their multimodal building blocks.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[LLaVA]: https://arxiv.org/abs/2304.08485 "LLaVA — Visual Instruction Tuning (Liu et al. 2023)"
[Cambrian]: https://arxiv.org/abs/2406.16860 "Cambrian-1 (Tong et al. 2024)"
[ViT]: https://arxiv.org/abs/2010.11929 "An Image Is Worth 16×16 Words — ViT (Dosovitskiy et al. 2020)"
[CPVT]: https://arxiv.org/abs/2102.10882 "Conditional Positional Encodings for Vision Transformers (Chu et al. 2021)"
[Qwen3VL]: https://github.com/QwenLM/Qwen3-VL "Qwen3-VL (2025)"
[Qwen3]: https://arxiv.org/abs/2505.09388 "Qwen3 Technical Report (2025)"
[Qwen3Emb]: https://arxiv.org/abs/2506.05176 "Qwen3 Embedding (2025)"
[RoPE]: https://arxiv.org/abs/2104.09864 "RoFormer / RoPE (Su et al. 2021)"
[YaRN]: https://arxiv.org/abs/2309.00071 "YaRN (Peng et al. 2023)"
[MRoPE]: https://arxiv.org/abs/2409.12191 "Qwen2-VL / M-RoPE (2024)"
[vLLM]: https://arxiv.org/abs/2309.06180 "PagedAttention / vLLM (Kwon et al. 2023)"
[SGLang]: https://arxiv.org/abs/2312.07104 "SGLang (Zheng et al. 2024)"
[CatForgetLLM]: https://arxiv.org/abs/2308.08747 "Catastrophic Forgetting during Continual Fine-tuning (Luo et al. 2025)"
