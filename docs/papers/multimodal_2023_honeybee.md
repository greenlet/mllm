# Honeybee: Locality-enhanced Projector for Multimodal LLM — Cha et al., 2023

> **arXiv:** 2312.06742v2 · **Venue:** CVPR 2024 · **Affiliation:** Kakao Brain

## TL;DR
Honeybee identifies two properties a vision→LLM **projector** must satisfy — **flexibility** in the
number of visual tokens (for efficiency) and **preservation of local context** (for spatial
understanding) — and shows linear projectors give locality-without-flexibility while resampler/Q-Former
abstractors give flexibility-without-locality. Its **locality-enhanced projectors** (C-Abstractor,
D-Abstractor) achieve both and set SOTA across MME, MMBench, SEED-Bench, and LLaVA-Bench.

## Problem & motivation
A linear projector is a 1-to-1 map: it preserves every patch's local context but forces the visual
token count to equal the patch count, so cost scales with resolution. Resampler/Q-Former abstractors
reduce the token count freely but summarize from a few salient regions, **losing fine-grained spatial
detail** — which hurts localization and counting. Honeybee wants a projector that is *both* flexible
and locality-preserving, plus a "hidden recipe" for combining many instruction datasets.

## Key idea
Inject **locality-aware operations** into the abstraction step:
- **C-Abstractor** (convolutional): $L$ ResNet bottleneck blocks → adaptive average pool to a chosen
  grid → $L$ more ResNet blocks. Convolution's locality bias is retained while the pool sets the
  output token count $M$.
- **D-Abstractor** (deformable attention): learnable queries gather features by sampling around
  reference points spread uniformly over the feature map,

  $$
  z^{l+1} = \sum_{k=1}^{K} A_k^{l}\, X_{\text{feat}}\!\big(p + \Delta o_k^{l}\big),
  $$

  with $p$ reference points and $\Delta o_k$ learned offsets — flexible token count with preserved
  locality.

## How it works
- **Architecture.** CLIP ViT-L/14 (frozen) → C-/D-Abstractor projector (trained) → Vicuna-1.5 7B/13B
  (jointly trained with the projector). Configs: 7B at 224px → 144 tokens; 13B at 336px → 256 tokens.
- **Two-stage training.** Stage 1 (pretrain projector only, encoder+LLM frozen) on BlipCapFilt+COYO;
  Stage 2 (instruction tuning, projector + LLM) on a multifaceted dataset mix.
- **"Hidden recipe" findings.** Per-dataset hand-tuned balancing beats uniform; **fine-grained
  templates** (one per dataset) beat coarse per-task templates; extra template *diversity* gives
  little gain; multi-turn merging with de-duplication prevents shortcut learning; the second-last CLIP
  layer beats the last.
- **Hyperparameters.** Pretrain batch 256, LR 3e-4; instruction-tune batch 128, LR 2e-5; AdamW
  ($\beta_2{=}0.98$), 8×A100-80GB, DeepSpeed ZeRO-2, FlashAttention-2.

## Training / data
Stage-1 alignment on web image–text (BlipCapFilt + COYO); Stage-2 on a mix spanning open/multiple-choice
VQA, captioning, referring-expression comprehension, and instruction-following datasets, combined per
the recipe above.

## Results
| Benchmark | Honeybee-7B | Honeybee-13B | Notes |
|---|---:|---:|---|
| MMBench (en) | 70.1 | 73.2 | per Table 6 |
| MME (perception) | 1584.2 | 1629.3 | per Table 6 |
| SEED-Bench | 64.5 | 68.2 | per Table 6 |
| LLaVA-Bench | 67.1 | 75.7 | per Table 6 |

Projector comparison at matched setting (6 spatial-understanding tasks, per Table): C-Abstractor at
144 tokens reaches ~70.6 avg vs. linear-256 ~70.0 and resampler-144 ~64.2, at resampler-level speed —
i.e. it matches a linear projector's *locality* while keeping an abstractor's *efficiency*. Accuracy
rises monotonically with more visual tokens (per Table 7).

## Limitations & follow-ups
- C-Abstractor's convolution imposes a **strict locality inductive bias**; D-Abstractor trades some of
  that for flexibility.
- At 7B, object-hallucination (POPE) slightly trails LLaVA-1.5; competitive at 13B.
- Strong dependence on careful recipe tuning (sampling ratios, ~10k instruction steps as a sweet spot,
  CLIP feature-layer choice).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2312.06742) · [html](https://arxiv.org/html/2312.06742v2) · [pdf](https://arxiv.org/pdf/2312.06742)
- **Code:** [kakaobrain/honeybee](https://github.com/kakaobrain/honeybee)
- **Hugging Face:** — (checkpoints released via the GitHub repo)
- **Papers-with-Code:** <https://paperswithcode.com/paper/honeybee-locality-enhanced-projector-for>
- **BibTeX:** see the arXiv "Export BibTeX" link on the abs page.
- **Related / successor papers:** [BLIP-2 / Q-Former (local recap)](multimodal_2023_blip2-qformer.md); [LLaVA-1.5 (local recap)](multimodal_2023_llava-1.5.md); [Flamingo / Perceiver Resampler (local recap)](multimodal_2022_flamingo-perceiver-resampler.md).
