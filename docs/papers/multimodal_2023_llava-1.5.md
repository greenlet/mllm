# Improved Baselines with Visual Instruction Tuning (LLaVA-1.5) — Liu et al., 2023

> **arXiv:** 2310.03744v2 · **Venue:** CVPR 2024 (Highlight) · **Affiliation:** UW–Madison, Microsoft Research

## TL;DR
LLaVA-1.5 shows the fully-connected LLaVA connector is "surprisingly powerful and data-efficient":
three simple changes — swap the linear projector for a **2-layer MLP**, raise the vision encoder to
**CLIP ViT-L/14-336px**, and add **academic-task VQA data with response-format prompts** — reach SOTA
on **11 benchmarks** using only ~1.2M public samples, training in **~1 day on a single 8×A100 node**.

## Problem & motivation
LLaVA excelled at free-form chat but lagged on short-answer VQA, while InstructBLIP used 129M
pre-training pairs to LLaVA's 600K — leaving it unclear which design choices actually matter.
LLaVA-1.5 is a controlled study isolating cheap, reproducible improvements over the LLaVA baseline.

## Key idea
Keep LLaVA's simplicity (frozen CLIP encoder, keep all patches, two-stage training) but fix three
bottlenecks:
1. **MLP connector** — replace the single linear map with a 2-layer GELU MLP.
2. **Response-format prompts** — append e.g. *"Answer the question using a single word or phrase."* so
   the model controls output length per instruction instead of overfitting to short answers.
3. **Academic VQA mixture + higher resolution** — add VQAv2/GQA/OKVQA/OCR/region datasets and move to
   336px input.

## How it works
- **Architecture.** CLIP ViT-L/14-**336px** (frozen) → 2-layer MLP projector (trainable) → Vicuna-1.5
  7B/13B. 336px yields 576 visual tokens.
- **Two-stage training** (unchanged shape from LLaVA): Stage 1 pretrains the projector (encoder + LLM
  frozen), Stage 2 fine-tunes projector + LLM end-to-end.
- **LLaVA-1.5-HD** variant: split a high-res image into 224px sub-images encoded independently, then
  merge features + a downsampled global view, reaching up to 448px without position-embedding
  interpolation.
- **Hyperparameters (per Table 9):** pretrain batch 256, LR 1e-3, 1 epoch; instruction-tune batch
  128, LR 2e-5, 1 epoch; AdamW, cosine schedule, DeepSpeed ZeRO-2/3.

## Training / data
- **~665K** instruction mixture: LLaVA-158K + ShareGPT-40K + VQAv2/GQA/OKVQA/A-OKVQA + OCRVQA/TextCaps
  + RefCOCO/VG, each wrapped with a response-format prompt (abstract cites ~1.2M total public data
  across both stages).
- **Compute:** ~26 h total on 8×A100 (≈1 day; per §3 / abstract).

## Results
| Benchmark | LLaVA-1.5-7B | LLaVA-1.5-13B | Comparator | Notes |
|---|---:|---:|---|---|
| MME (perception) | 1510.7 | 1531.3 | InstructBLIP-13B 1212.8 | per Table 4 |
| MMBench (en) | 64.3 | 67.7 | — | per Table 4 |
| SEED-Bench | 58.6 | 61.6 | — | per Table 4 |
| MM-Vet | 31.1 | 36.1 | InstructBLIP-13B 25.6 | per Table 4 |
| POPE (random) | 87.3 | 87.1 | — | hallucination eval (per Table 4) |
| — | — | — | — | **SOTA on 11/12 benchmarks** with 1.2M public data (per abstract) |

## Limitations & follow-ups
- Keeps **all** patch tokens, so it is less token-efficient than resampler/Q-Former designs (the
  trade-off [Honeybee](multimodal_2023_honeybee.md) targets).
- No multi-image reasoning; hallucination persists and is tied to resolution/capacity mismatch.
- Weak on specialized domains; not for safety-critical use.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2310.03744) · [html](https://arxiv.org/html/2310.03744v2) · [pdf](https://arxiv.org/pdf/2310.03744)
- **Code:** [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- **Hugging Face:** [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) · [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b)
- **Project page:** <https://llava-vl.github.io>
- **Papers-with-Code:** <https://paperswithcode.com/paper/improved-baselines-with-visual-instruction>
- **BibTeX:** see the arXiv "Export BibTeX" link on the abs page.
- **Related / successor papers:** [LLaVA (local recap)](multimodal_2023_llava.md); [InstructBLIP (local recap)](multimodal_2023_instructblip.md); [Honeybee (local recap)](multimodal_2023_honeybee.md).
