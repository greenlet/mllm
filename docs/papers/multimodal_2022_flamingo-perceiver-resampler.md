# Flamingo: a Visual Language Model for Few-Shot Learning — Alayrac et al., 2022

> **arXiv:** 2204.14198v2 · **Venue:** NeurIPS 2022 · **Affiliation:** DeepMind

> **Scope of this recap.** This recap is focused on the two architectural primitives that the rest of the multimodal-LM literature inherited from Flamingo — the **Perceiver Resampler** and the **gated cross-attention dense (GATED XATTN-DENSE)** layers — rather than on the few-shot benchmark sweep that gives the paper its title. The benchmarks are summarized briefly at the end.

## TL;DR
Flamingo bridges a frozen vision encoder (NFNet-F6) and a frozen language model (Chinchilla 1.4B / 7B / 70B) with two small trainable modules: (1) a **Perceiver Resampler** that compresses any number of image/video patch features into a fixed bank of $R{=}64$ latent vectors via cross-attention from learned queries, and (2) **gated cross-attention layers** interleaved into the frozen LM that let text tokens attend to those latents, with a $\tanh(\alpha)$ gate initialized at $\alpha{=}0$ so the LM is identical to its frozen base at the start of training.

## Problem & motivation
A pretrained vision encoder emits a variable number of feature tokens per input:
- one image with $S$ patches ⇒ $S$ tokens,
- a video with $T$ frames ⇒ $T\cdot S$ tokens,
- an interleaved web page with several images ⇒ a per-image variable count.

A frozen LM expects a fixed-budget interface. Two naive fixes fail:
- **Flatten every patch as a soft prompt.** Sequence length explodes (videos especially), KV-cache cost is prohibitive, and the LM never sees the same "shape" twice.
- **Mean-pool everything to one vector.** Throws away the spatial / temporal structure that VQA, OCR, and captioning rely on.

The Perceiver Resampler resolves this by *summarizing*, not *flattening*: $T\cdot S$ visual tokens are compressed to exactly $R$ latent tokens, where $R$ is a small fixed budget (64 in the paper) chosen independently of the input.

## Key idea
**Perceiver Resampler.** A small Transformer carries a learned bank of $R$ latent query vectors. Each block performs cross-attention from these queries to the (variable-length) sequence of vision features, optionally followed by self-attention among the queries and an FFN. After a few blocks, the output is exactly $R$ vectors, regardless of how many visual features came in. This is the Perceiver IO (Jaegle et al. 2021) construction adapted to the "variable visual input → fixed prefix for an LM" use case.

**Gated cross-attention dense layers (GATED XATTN-DENSE).** New cross-attention + FFN sub-layers are *inserted* into the frozen LM (between existing Transformer blocks, periodically). They let language tokens attend to the $R$ resampler outputs. The contribution is multiplied by a learnable scalar $\tanh(\alpha)$ with $\alpha$ initialized to $0$:

$$
h \leftarrow h + \tanh(\alpha_{\text{xattn}})\cdot \mathrm{XATTN}(h,\,V_{\text{vis}}) + \tanh(\alpha_{\text{ff}})\cdot \mathrm{FFN}\!\big(h + \tanh(\alpha_{\text{xattn}})\cdot \mathrm{XATTN}(h,V_{\text{vis}})\big).
$$

At initialization $\tanh(0)=0$, so the inserted layers are pass-through and the model is bit-identical to frozen Chinchilla. Training learns $\alpha$ away from zero, smoothly mixing in the visual signal without ever destabilizing the LM.

## How it works

### Perceiver Resampler (the visual bottleneck)
- **Inputs.** $X_f \in \mathbb{R}^{T\cdot S \times d_v}$ — concatenation of patch features from the frozen NFNet-F6 vision encoder over $T$ frames; each frame contributes $S$ spatial tokens. Learned per-frame *temporal* embeddings are added to $X_f$ to mark which frame each token came from. Single images are the $T{=}1$ case.
- **Latents.** $L \in \mathbb{R}^{R\times d}$, $R=64$ learned vectors.
- **Block (×6, per §3.2 / appendix).** Cross-attention with $Q=L$, $K=V=[\,X_f \,\Vert\, L\,]$ (concatenation along the token axis, so latents can attend both to vision features and to themselves), followed by self-attention over $L$ and an FFN. Outputs replace $L$ for the next block.
- **Output.** $L \in \mathbb{R}^{R\times d}$ — exactly $R$ vectors, the LM-facing "visual tokens."

### Gated cross-attention dense layers (the LM-side hookup)
- **Placement.** Inserted into the frozen LM with a periodicity that depends on the LM size: every layer for Flamingo-3B (Chinchilla-1.4B), every 4th for Flamingo-9B, every 7th for Flamingo-80B (per §3.3, Table 4).
- **Per-layer op.** Cross-attention from LM hidden states to the resampler latents, then a Dense (FFN) sub-layer, both gated by independent $\tanh(\alpha)$ scalars, both initialized at $\alpha=0$.
- **Causal masking over interleaved sequences.** For training on interleaved image+text web pages, each text token is restricted to attend to the visual features of the *most recent preceding image*, not all images on the page — this prevents the model from "cheating" by attending to a future image when generating earlier text.

### What is trained vs frozen
| Module | Status | Approx. params |
|---|---|---|
| Vision encoder (NFNet-F6) | frozen | ~435M |
| Language model (Chinchilla 1.4B / 7B / 70B) | frozen | 1.4B – 70B |
| Perceiver Resampler | **trained** | ~194M |
| Gated cross-attention layers | **trained** | varies with insertion frequency |

The total *trainable* parameter count for Flamingo-80B is ~10B — large in absolute terms but a small fraction of the model.

## Training / data
- **Datasets (per §2.4 / §A.3).**
  - **M3W** — Multimodal Massive Web, internal: web pages with interleaved text and images; key to the in-context, few-shot capability.
  - **ALIGN** — 1.8B noisy alt-text/image pairs.
  - **LTIP** — Long Text & Image Pairs, ~312M pairs.
  - **VTP** — Video & Text Pairs, ~27M clips.
- **Objective.** A single per-sample next-token cross-entropy on the text, with the visual prefix produced by the resampler. Datasets are mixed with per-dataset weights $\lambda_m$ and the per-batch loss is a weighted sum (per §A.2).
- **Optimizer.** AdamW; the 80B model is trained on TPUv4 pods.

## Results
| Benchmark | Setting | Flamingo-80B | Prior best (fine-tuned) | Notes |
|---|---|---|---|---|
| VQAv2 (test-dev) | 32-shot | 67.6 | 81.3 (fine-tuned SimVLM) | per Table 1; *zero-shot* not strictly comparable to fine-tuned SOTA |
| OK-VQA | 32-shot | 57.8 | 54.4 | per Table 1; SOTA at publication for few-shot |
| COCO captioning (CIDEr) | 32-shot | 113.8 | 143.3 (fine-tuned) | per Table 1 |
| MSRVTT-QA | 32-shot | 31.0 | n/a | per Table 1 |

> Headline framing from the abstract: a single Flamingo model sets state of the art on **16** benchmarks in the few-shot regime, outperforming methods fine-tuned on *thousands of times* more task-specific data.

## Limitations & follow-ups
- **In-context learning quality** depends heavily on the M3W interleaved corpus; reproductions without similar data have struggled to match few-shot scores.
- **Frozen vision encoder** caps OCR / fine-grained perception; later systems retrain or swap it.
- **Inherited LM hallucinations / biases** — Flamingo doesn't fix anything the frozen LM gets wrong.

Successor designs that reuse or replace the resampler:
- **BLIP-2 / Q-Former** (Li et al., 2023, arXiv:2301.12597) — replaces the Perceiver Resampler with a **Q-Former** trained in two stages (representation learning + generative bridging) that achieves better quality with ~54× fewer trainable parameters. See the local recap: [multimodal_2023_blip2-qformer.md](multimodal_2023_blip2-qformer.md).
- **OpenFlamingo** (Awadalla et al., 2023, arXiv:2308.01390) — open reproduction with LAION-2B-en + MMC4 instead of M3W.
- **IDEFICS** (HuggingFace, 2023) — open reimplementation of the Flamingo architecture on Llama backbones.
- **Qwen-VL / Qwen2-VL** — drop the gated-xattn machinery, project all merged patch tokens directly into the LM token stream via placeholder IDs (see [docs/qwen/overview.md §3.2](../qwen/overview.md#L94)).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2204.14198) · [pdf](https://arxiv.org/pdf/2204.14198)
- **Code:** —  (DeepMind did not release the original implementation)
- **Open reproduction:** [open-flamingo/open-flamingo](https://github.com/mlfoundations/open_flamingo)
- **HuggingFace (Flamingo-style):** [HuggingFaceM4/idefics-80b](https://huggingface.co/HuggingFaceM4/idefics-80b)
- **DeepMind blog post:** <https://deepmind.google/discover/blog/tackling-multiple-tasks-with-a-single-visual-language-model/>
- **OpenReview:** <https://openreview.net/forum?id=EbMuimAbPbs>
- **BibTeX:** see [the NeurIPS 2022 proceedings entry](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)
- **Related / successor papers:** [Perceiver IO (Jaegle et al. 2021, arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — the resampler's intellectual parent; [BLIP-2 / Q-Former (local recap)](multimodal_2023_blip2-qformer.md); [LLaVA (Liu et al. 2023, arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — opposite design choice (concat-all-patch-tokens, no resampler, no gated cross-attn).
