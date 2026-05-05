# RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al., 2021

> **arXiv:** 2104.09864v5 · **Venue:** preprint (cs.CL) · **Affiliation:** Zhuiyi Technology

## TL;DR
RoPE encodes absolute position by **multiplying** queries and keys with a position-dependent rotation matrix, so the inner product $\langle R_m q, R_n k\rangle$ depends only on the **relative** offset $n-m$. It costs no extra parameters, generalizes to unseen lengths, exhibits a natural decay of attention with distance, and is the first relative-position scheme compatible with linear / efficient attention.

## Problem & motivation
Prior positional encodings either (i) add a fixed/learned vector to token embeddings (Vaswani et al. sinusoidal, learned absolute) — which the attention layer treats as just another feature and which does not generalize across lengths, or (ii) inject a learned bias into the attention matrix (Shaw, T5-RPE, ALiBi) — which encodes relative position cleanly but requires materializing the full $N\times N$ attention map and is therefore incompatible with linear-attention variants such as Performer. RoFormer's goal: a single mechanism that is (a) relative-aware, (b) length-flexible, (c) compatible with linear attention.

## Key idea
Treat each consecutive pair of channels in $q,k\in\mathbb{R}^d$ as a complex number and rotate it by an angle proportional to position $m$:

$$
R_{\Theta,m}^{d}=\operatorname{blkdiag}\!\Big(\!\big[\begin{smallmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{smallmatrix}\big]\!\Big)_{i=1}^{d/2},\qquad \theta_i = 10000^{-2(i-1)/d}.
$$

The crucial property — the inner product is shift-equivariant:

$$
\big\langle R_{\Theta,m}^{d}\,q,\; R_{\Theta,n}^{d}\,k\big\rangle = \big\langle q,\; R_{\Theta,n-m}^{d}\,k\big\rangle .
$$

So absolute positions enter the computation, but only the **difference** $n-m$ survives in the attention score.

## How it works
- **Where it applies.** Only to $q$ and $k$ (not $v$), at every layer, per head.
- **Efficient form.** Build sin/cos vectors for the chosen positions; the rotation is then an elementwise multiply-add, no $d\times d$ matmul:
  ```python
  # x: [..., d]  with d even
  x1, x2 = x[..., 0::2], x[..., 1::2]                 # split into pairs
  x_rot  = torch.stack([-x2, x1], dim=-1).flatten(-2) # rotate 90°
  return x * cos + x_rot * sin
  ```
- **Hyperparameters.** Base $b=10000$ (controls the wavelength range $2\pi b^{2i/d}$); pair count $d/2$.
- **Length flexibility.** No max length is baked in: just pick $m$ from `arange(L)`.
- **Long-term decay.** The Toeplitz-like sum of rotated dot products decays in expectation as $|n-m|$ grows — i.e. distant tokens contribute less, no manual masking required.
- **Linear-attention compatibility.** Because the rotation is applied to $q,k$ *before* the attention kernel, RoPE composes with $\phi(q)\phi(k)^\top$ feature-map style attention, which earlier relative-position schemes did not.

## Training / data
The original Zhuiyi training: a Chinese RoFormer in the BERT family (`L-12_H-768_A-12` and a 6-layer variant), trained as a masked LM and evaluated on Chinese long-text benchmarks (notably **CAIL2019-SCM**) plus GLUE for English (per §4 of the arXiv abstract; specific compute / token counts not given in the paper). The exact pre-training corpus and budget are not fully reported; the paper is more about the encoding than a single SOTA run.

## Results
Third-party validations (cited in §5 / blog companion) at modest scale show RoPE consistently below sinusoidal and T5-RPE in pre-training perplexity:

| Setup (per source §5) | Encoding | Val. loss | Val. ppl |
|---|---|---|---|
| GPT-NeoX 125M, OWT2 | Learned absolute | 2.809 | 16.59 |
| GPT-NeoX 125M, OWT2 | T5-RPE           | 2.801 | 16.46 |
| GPT-NeoX 125M, OWT2 | **RoPE**         | **2.759** | **15.78** |
| Mesh-TF JAX 1.4B, The Pile | Learned absolute | 2.240 | 9.393 |
| Mesh-TF JAX 1.4B, The Pile | T5-RPE           | 2.223 | 9.234 |
| Mesh-TF JAX 1.4B, The Pile | **RoPE**         | **2.173** | **8.784** |

(Numbers as reproduced in the EleutherAI write-up cited from the paper; CAIL2019-SCM exact scores not extracted from the arXiv abstract.)

## Limitations & follow-ups
- **No native length extrapolation.** Inference at $m \gg L_{\text{train}}$ visits unseen rotation angles and degrades sharply — fixed by [Position Interpolation](https://arxiv.org/abs/2306.15595), [NTK-aware RoPE](p001_2023_positional_ntk-aware-rope.md), [YaRN](p002_2023_positional_yarn-context-extension.md), and training-free [DCA](p003_2024_positional_dca-dual-chunk-attention.md).
- **Per-layer cost.** Naïve implementation is ~4–5× a single bias-add; with kernel fusion this falls to a 1–3 % overhead in large models.
- **Variants in later models.** RoPE has been extended to multimodal axes (M-RoPE in Qwen2-VL; TMRoPE in Qwen2.5-Omni), to 2-D / 3-D images, and combined with QK-Norm in modern stacks.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2104.09864) · [pdf](https://arxiv.org/pdf/2104.09864) · (HTML version unavailable at time of recap)
- **Code:** [ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer) · PyTorch port: [JunnYu/RoFormer_pytorch](https://github.com/JunnYu/RoFormer_pytorch)
- **Hugging Face:** [transformers RoFormer doc](https://huggingface.co/docs/transformers/model_doc/roformer) · models: search `roformer` (e.g. `junnyu/roformer_chinese_base`)
- **Project page:** —
- **Blog posts:** [Su Jianlin — kexue.fm 8265 (Chinese)](https://kexue.fm/archives/8265) · [EleutherAI — Rotary Embeddings: A Comprehensive Explanation](https://blog.eleuther.ai/rotary-embeddings/)
- **Talks / videos:** —
- **OpenReview / venue page:** — (preprint)
- **Papers-with-Code:** [paperswithcode.com/paper/roformer-enhanced-transformer-with-rotary](https://paperswithcode.com/paper/roformer-enhanced-transformer-with-rotary)
- **BibTeX:** available from the arXiv abs page
- **Related / successor papers:** Position Interpolation (Chen et al. 2306.15595) · [NTK-aware RoPE](p001_2023_positional_ntk-aware-rope.md) · [YaRN](p002_2023_positional_yarn-context-extension.md) · [DCA](p003_2024_positional_dca-dual-chunk-attention.md) · M-RoPE (in the Qwen2-VL paper, arXiv:2409.12191)
