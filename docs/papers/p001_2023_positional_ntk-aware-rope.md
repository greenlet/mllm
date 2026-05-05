# NTK-aware Scaled RoPE — bloc97, 2023

> **Origin:** Reddit `r/LocalLLaMA` post (Jun 2023) · **Formal treatment:** [YaRN, §3.1 + App. A.2](p002_2023_positional_yarn-context-extension.md) · **Author:** community handle `bloc97`

## TL;DR
A one-line modification to RoPE that extends a model's usable context window **without any fine-tuning**: instead of *stretching* positions like Position Interpolation does (which destroys the high-frequency channels), scale RoPE's base $b$ from $10000$ to $b' = b\cdot s^{d/(d-2)}$. Low-frequency dimensions then get interpolated, high-frequency dimensions stay (almost) untouched, and most pretrained RoPE models survive a 2–8× context extension out-of-the-box.

## Problem & motivation
**Position Interpolation (PI, [Chen et al. 2306.15595](https://arxiv.org/abs/2306.15595))** extends a model's context length $L \to L'=sL$ by feeding the model the *fractional* positions $m/s$. This compresses every RoPE frequency uniformly:

$$
\theta_i^{\text{PI}} = \theta_i,\quad m \mapsto m/s.
$$

The problem (formalized in YaRN §3.1 / App. A.2): RoPE's high-frequency channels are exactly what carries fine-grained relative-position information (think NTK / Fourier-features intuition — high-frequency components are what let the model resolve nearby tokens). PI scales them down equally, blurring locality. Fine-tuning has trouble recovering this and PI typically caps out at $s\!\approx\!8$.

## Key idea
**Don't shrink positions; shrink frequencies *non-uniformly* by changing the base.** Replace

$$
\theta_i = b^{-2(i-1)/d}\quad\text{with}\quad \theta_i' = (b')^{-2(i-1)/d},\qquad b' = b\cdot s^{\,d/(d-2)}.
$$

Derivation (YaRN App. A.2): pick $b'$ so that the **lowest-frequency** dimension ($i=d/2$) sees its wavelength stretched by exactly $s$ (matching PI), while leaving the highest-frequency dimension's wavelength essentially unchanged. Setting $\lambda'_{d/2}/\lambda_{d/2}=s$ and solving gives the formula above.

Effect across the spectrum:
- low-freq dims (large $\lambda$): wavelength grows by $\approx s$ — interpolation, like PI;
- high-freq dims (small $\lambda$): wavelength barely changes — extrapolation preserved;
- mid-freq dims: smooth transition.

## How it works
**Implementation diff** in any RoPE codebase is literally one line:

```python
# original (b = 10000, d = head_dim)
inv_freq = 1.0 / (b ** (torch.arange(0, d, 2).float() / d))

# NTK-aware, scale s = L_new / L_train  (e.g. s = 4 for 2k -> 8k)
b_new   = b * (s ** (d / (d - 2)))
inv_freq = 1.0 / (b_new ** (torch.arange(0, d, 2).float() / d))
```

**Hyperparameters.** Just $s$. Practical values for LLaMA-class 2k models: $s\!\in\!\{2,4,8,16\}$ giving 4k–32k contexts.

**Variants quickly developed by the same community.**
- **Dynamic NTK** — recompute $b'$ at every forward pass using $s=\max(1,\,L_{\text{cur}}/L_{\text{train}})$, so the model is identical at short contexts and only "stretches" as the sequence grows. Used in Code Llama (100k) and Qwen.
- **NTK-by-parts** ([bloc97 / scaled-rope PR #1](https://github.com/jquesnelle/scaled-rope/pull/1)) — instead of a global $b'$, apply a piecewise interpolation per dimension, keeping high-freq dims completely untouched. This is the basis of YaRN.

## Training / data
None — this is a **training-free** inference-time tweak. (Optional fine-tuning, e.g. ~400 PG-19 steps as in YaRN, recovers small gaps but is not required.)

## Results
Per YaRN's evaluation on LLaMA-7B / Proof-pile, sliding-window perplexity:

| Method | Tuned? | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 |
|---|---|---|---|---|---|---|
| RoPE (no extension) | – | 4.05 | — | — | — | — |
| PI ($s=8$) | ✗ | 4.36 | 3.90 | — | — | — |
| **NTK-aware** ($s=16$) | ✗ | **4.08** | (degrades) | — | — | — |
| **NTK-aware** ($s=16$) | ✓ (400 steps) | 4.39 | 3.92 | 3.73 | 3.21 | **8.49** |

(per YaRN Table 5; "—" means the method was not evaluated at that length.)

**Take-away.** Zero-shot, NTK-aware beats PI in the regime "near the trained context"; once you fine-tune, however, the extrapolation on high-freq dims becomes a liability and perplexity blows up at very long contexts (32k entry above) — the gap that NTK-by-parts and YaRN close.

## Limitations & follow-ups
- **Fine-tuning paradox.** The mechanism that makes NTK-aware great zero-shot (preserving high-freq dims) makes it hard to fine-tune cleanly: those dims get pushed beyond the values seen in pre-training.
- **No principled tuning knob.** $b'$ is fixed by a closed-form derivation; you can't trade off interpolation/extrapolation per dimension.
- **Superseded by [YaRN](p002_2023_positional_yarn-context-extension.md)**, which adds (i) a per-dimension ramp ("NTK-by-parts") and (ii) attention-temperature scaling to fix the entropy shift at long contexts.

## Links
- **Original Reddit post:** [r/LocalLLaMA — "NTK-Aware Scaled RoPE allows LLaMA models to have extended context …"](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
- **Formal write-up:** YaRN paper (Peng et al. 2023) — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071), §3.1 "Loss of high frequency information" + App. A.2
- **Community blog:** [EleutherAI — *YaRN: long context for LLaMA*](https://blog.eleuther.ai/yarn/) (covers the NTK-aware lineage)
- **Code:** [`jquesnelle/scaled-rope`](https://github.com/jquesnelle/scaled-rope) (NTK-by-parts PR #1) · [`jquesnelle/yarn`](https://github.com/jquesnelle/yarn) (YaRN reference impl)
- **Hugging Face:** Built into `transformers` LLaMA modelling under `rope_scaling={"type":"dynamic","factor": s}` (see [`modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py))
- **Used in:** Code Llama (100k), Qwen-7B/14B (Dynamic NTK), many community LLaMA fine-tunes
- **Related / successor papers:** [Position Interpolation (Chen et al.)](https://arxiv.org/abs/2306.15595) · [YaRN](p002_2023_positional_yarn-context-extension.md) · [Dual Chunk Attention](p003_2024_positional_dca-dual-chunk-attention.md) · NTK theory: [Tancik et al. 2020 — Fourier Features](https://arxiv.org/abs/2006.10739)
- **BibTeX:** none (community post); cite the Reddit URL plus YaRN App. A.2 for the formal version
