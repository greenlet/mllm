# InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning — Dai et al., 2023

> **arXiv:** 2305.06500v2 · **Venue:** NeurIPS 2023 · **Affiliation:** Salesforce Research, HKUST, NTU Singapore

## TL;DR
InstructBLIP performs systematic **vision-language instruction tuning** on top of frozen
[BLIP-2](multimodal_2023_blip2-qformer.md), adding an **instruction-aware Q-Former**: the instruction
text is fed into the Q-Former alongside its $K{=}32$ learned queries, so the extracted visual features
are *tailored to the task*. Trained on 13 held-in datasets (of 26 total), it sets SOTA zero-shot
results on 13 held-out datasets and beats much larger Flamingo models.

## Problem & motivation
BLIP-2's Q-Former extracts the **same** visual features regardless of the question, which is
suboptimal when the same image must support captioning, counting, OCR, and reasoning. And
vision-language *instruction* tuning (vs. plain multitask training) was under-explored: the paper
shows multitask training without instructions fails to generalize to unseen tasks.

## Key idea
Make feature extraction **instruction-conditioned**. The instruction tokens are concatenated with the
$K{=}32$ query embeddings and interact with them through the Q-Former's **self-attention**, so the
queries then cross-attend to the frozen image features in a task-aware way and emit $K$ soft tokens to
the frozen LLM. Only the Q-Former is trained; image encoder and LLM stay frozen.

## How it works
- **Architecture.** ViT-g/14 EVA-CLIP (frozen) → instruction-aware Q-Former (trainable, ~188M) →
  linear projection → frozen LLM (FlanT5-XL/XXL or Vicuna-7B/13B). Image resolution 224×224.
- **Instruction path.** Instruction text → Q-Former self-attention (shared with the queries) →
  queries become task-conditioned → cross-attend to image patches → 32 output vectors as a soft
  prompt for the LLM.
- **Balanced multi-dataset sampling.** To avoid over/under-fitting datasets of very different sizes,
  sample dataset $d$ with probability proportional to the square root of its size:

  $$
  p_d = \frac{\sqrt{S_d}}{\sum_{i=1}^{D}\sqrt{S_i}},
  $$

  with manual tweaks (e.g. down-weight multiple-choice A-OKVQA). 10–15 instruction templates per task.
- **Recipe (per §2.6):** AdamW, LR warmup $10^{-8}\!\to\!10^{-5}$ over 1k steps then cosine decay; up
  to 60k steps; ~1.5 days on 16×A100-40GB.

## Training / data
26 public datasets across 11 task families, split into **held-in** (training: COCO Caption,
Web CapFilt, TextCaps, VQAv2, OKVQA, A-OKVQA, OCR-VQA, LLaVA-Instruct-150K, …) and **held-out**
(zero-shot eval: NoCaps, Flickr30K, GQA, VSR, IconQA, TextVQA, VizWiz, Visual Dialog, ScienceQA,
MSVD-QA, MSRVTT-QA, iVQA, HatefulMemes). Four task categories are held out entirely. Objective:
standard LM generation loss.

## Results
| Benchmark (zero-shot) | InstructBLIP (FlanT5-XL) | BLIP-2 (FlanT5-XL) | Notes |
|---|---:|---:|---|
| NoCaps (CIDEr) | 119.9 | 104.5 | per Table 1 |
| GQA (acc) | 48.4 | 44.0 | per Table 1 |
| ScienceQA-IMG (acc) | 70.4 | 54.9 | per Table 1 |
| TextVQA (acc) | 46.6 | 43.1 | per Table 1 |
| MSRVTT-QA (acc) | 25.0 | 16.2 | zero-shot video, no video training (per Table 1) |
| ScienceQA-IMG (fine-tuned) | 90.7 | — | SOTA on fine-tuning (per abstract / Table 3) |

InstructBLIP reports ~15% average relative gain over BLIP-2 (FlanT5-XL) and outperforms Flamingo-80B
on all shared evaluation datasets despite far fewer parameters (per §3). Ablating the instruction-aware
Q-Former costs ~4.3 points on held-in average (per Table 2).

## Limitations & follow-ups
- Inherits frozen-LLM failure modes (hallucination, ungrounded generation, bias).
- 224×224 resolution limits fine OCR/small-object reading.
- Builds directly on BLIP-2; contrasts with [LLaVA-1.5](multimodal_2023_llava-1.5.md), which keeps all
  patch tokens (no learned-query bottleneck) and competes via better data + MLP projector.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2305.06500) · [html](https://arxiv.org/html/2305.06500v2) · [pdf](https://arxiv.org/pdf/2305.06500)
- **Code:** [salesforce/LAVIS — `projects/instructblip`](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)
- **Hugging Face:** [Salesforce/instructblip-vicuna-7b](https://huggingface.co/Salesforce/instructblip-vicuna-7b) · [Salesforce/instructblip-flan-t5-xl](https://huggingface.co/Salesforce/instructblip-flan-t5-xl)
- **Papers-with-Code:** <https://paperswithcode.com/paper/instructblip-towards-general-purpose-vision>
- **BibTeX:** see the arXiv "Export BibTeX" link on the abs page.
- **Related / successor papers:** [BLIP-2 / Q-Former (local recap)](multimodal_2023_blip2-qformer.md); [Flamingo (local recap)](multimodal_2022_flamingo-perceiver-resampler.md); [LLaVA-1.5 (local recap)](multimodal_2023_llava-1.5.md).
