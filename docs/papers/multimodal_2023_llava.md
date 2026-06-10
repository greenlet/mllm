# Visual Instruction Tuning (LLaVA) — Liu et al., 2023

> **arXiv:** 2304.08485v2 · **Venue:** NeurIPS 2023 (Oral) · **Affiliation:** UW–Madison, Microsoft Research, Columbia

## TL;DR
LLaVA is the first attempt to use **language-only GPT-4** to generate multimodal
instruction-following data, then trains an end-to-end assistant that connects a **frozen CLIP
ViT-L/14** vision encoder to a **Vicuna** LLM through a single **trainable linear projection** — all
patch tokens are kept (no learned-query bottleneck). With 595K alignment pairs + 158K GPT-4-generated
instructions it reaches **90.92%** on ScienceQA (92.53% ensembled with GPT-4).

## Problem & motivation
Instruction tuning made text LLMs follow open-ended commands, but multimodal instruction data was
scarce and vision models lacked conversational interactivity. LLaVA's bet: bootstrap the missing data
with GPT-4 and bridge vision↔language with the simplest possible connector, end-to-end.

## Key idea
**Data:** feed GPT-4 only the *textual* representation of an image — its captions and object bounding
boxes — and prompt it to produce three response styles: multi-turn **conversation**, **detailed
description**, and **complex reasoning** (158K samples total). **Model:** a frozen CLIP encoder
produces patch features $Z_v$; a learned matrix $W$ projects them into the LLM word-embedding space,

$$
H_v = W \cdot Z_v,
$$

and the projected visual tokens are concatenated with the text tokens for the LLM.

## How it works
- **Architecture.** CLIP ViT-L/14 (frozen) → linear projector $W$ (trainable) → Vicuna decoder.
  Visual tokens are placed inline with the instruction; the autoregressive LM loss is computed only
  on the answer tokens.
- **Stage 1 — feature alignment** (1 epoch, 595K filtered CC3M pairs): freeze the encoder *and* the
  LLM, train **only** $W$ so the projected visual tokens land in the LLM embedding space. LR 2e-3,
  batch 128.
- **Stage 2 — end-to-end instruction tuning** (3 epochs, 158K data): train $W$ **and** Vicuna; the
  encoder stays frozen. LR 2e-5, batch 32.
- **Compute.** ~4 h pretraining + ~10 h fine-tuning on 8×A100 (per §5 / appendix).

## Training / data
- 595K image–text pairs (CC3M filtered for noun-phrase coverage) for alignment.
- 158K GPT-4-generated samples = 58K conversation + 23K detailed description + 77K complex reasoning,
  produced from COCO captions + boxes (image itself never shown to GPT-4).
- Objective: standard causal LM cross-entropy on the assistant turn.

## Results
| Benchmark | Setting | LLaVA | Comparator | Notes |
|---|---|---:|---|---|
| ScienceQA | fine-tuned | 90.92 | MM-CoT 91.68 (prior SOTA) | per §6 / Table 7 |
| ScienceQA | LLaVA + GPT-4 (judge ensemble) | **92.53** | — | new SOTA (per abstract) |
| LLaVA-Bench (COCO) | relative to text GPT-4 | 85.1 | — | overall score (per abstract / Table 4) |
| LLaVA-Bench (in-the-wild) | vs BLIP-2 | 67.3 | 38.1 (BLIP-2) | +29% (per Table 5) |

## Limitations & follow-ups
- A **"bag-of-patches"** failure mode: can confuse distinct objects/regions, and hallucinate during
  long detailed descriptions.
- Keeps **all** patch tokens, so cost scales with image resolution; weak short-answer VQA in v1.
- Limited multilingual ability and no multi-image reasoning.
- Direct successor [LLaVA-1.5](multimodal_2023_llava-1.5.md) fixes the projector (MLP), adds VQA data
  and response-format prompts, and raises resolution.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2304.08485) · [html](https://arxiv.org/html/2304.08485v2) · [pdf](https://arxiv.org/pdf/2304.08485)
- **Code:** [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- **Hugging Face:** [liuhaotian/LLaVA-7b-delta-v0](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0) · dataset [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- **Project page:** <https://llava-vl.github.io>
- **Papers-with-Code:** <https://paperswithcode.com/paper/visual-instruction-tuning>
- **BibTeX:** see the arXiv "Export BibTeX" link on the abs page.
- **Related / successor papers:** [LLaVA-1.5 (local recap)](multimodal_2023_llava-1.5.md); [BLIP-2 (local recap)](multimodal_2023_blip2-qformer.md); [InstructBLIP (local recap)](multimodal_2023_instructblip.md).
