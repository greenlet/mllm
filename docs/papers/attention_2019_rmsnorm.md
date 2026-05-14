# Root Mean Square Layer Normalization — Zhang & Sennrich, 2019

> **arXiv:** 1910.07467v1 · **Venue:** NeurIPS 2019 · **Affiliation:** University of Edinburgh

## TL;DR
Argues that **only the re-scaling invariance of LayerNorm matters**, not its mean re-centering. Drops the mean-subtraction and bias terms, normalizing instead by the root-mean-square of the activations. Result: a drop-in replacement that matches LayerNorm quality across MT, LM and image tasks while running **7–64 % faster**.

## Problem & motivation
[LayerNorm](https://arxiv.org/abs/1607.06450) computes a per-token mean *and* variance, plus a gain and a bias. On RNNs (Sennrich's NMT setting) and small Transformers this normalization is a real fraction of step time. The paper hypothesizes that the re-*centering* component is the dispensable one — only re-*scaling* truly stabilizes activations across depth.

## Key idea
Replace LN with **RMSNorm**, a normalization that uses only the second moment:

$$\operatorname{RMS}(x) = \sqrt{\tfrac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon},\qquad \operatorname{RMSNorm}(x) = \frac{x}{\operatorname{RMS}(x)}\odot g.$$

No mean subtraction, no learnable bias — just a per-feature gain $g\in\mathbb{R}^d$. Preserves re-scaling invariance ($x\to \alpha x$ leaves the output unchanged) and gives the implicit-LR-adaptation property that LN was credited with, at a fraction of the cost.

## How it works
- **Forward op count:** one mul, one mean, one rsqrt, one mul-by-gain — vs LN's two means, two muls, one add, one rsqrt.
- **Backward:** likewise simpler; no mean-of-grad term.
- **pRMSNorm variant:** estimate the RMS from the first $p\%$ of the dimensions only, trading a small accuracy loss for further speed-up (useful on very wide layers).
- **Initialization:** $g \leftarrow \mathbf{1}$, $\varepsilon\!\sim\!10^{-6}$ (LLaMA / Qwen) or $10^{-8}$ (paper default).

## Training / data
Tested as a drop-in for LN with **identical hyperparameters** in:
- Transformer NMT on WMT'14 EN-DE / WMT'16 EN-RO,
- RNN-LM on Wikitext-103 and a Chinese corpus,
- Convolutional / capsule classifiers on CIFAR-10 and IWSLT.

No tuning required for the swap.

## Results
| Task | Model | Baseline (LN) | RMSNorm | Speed-up | Source |
|---|---|---|---|---|---|
| WMT'14 EN-DE | Transformer-base | 27.27 BLEU | 27.20 | **+11 %** wall-clock | Tables 2, 8 |
| WMT'16 EN-RO | Transformer-base | 22.79 BLEU | 22.96 | +14 % | Tables 2, 8 |
| Wikitext-103 LM | LSTM | 102.0 PPL | 101.3 | +32 % | Table 4 |
| IWSLT'14 DE-EN | RNNSearch | 31.99 BLEU | 32.45 | **+64 %** | Tables 1, 7 |
| CIFAR-10 | ConvNet | 91.5 % | 91.6 % | +7 %  | Table 6 |

Across the board, RMSNorm **matches or exceeds** LN within noise while always being faster.

## Limitations & follow-ups
- Without re-centering, RMSNorm is more sensitive to *biased* activations — early layers in a tied-embedding LM may need a slightly larger $\varepsilon$.
- pRMSNorm's quality drops once $p<25\%$ on Transformer encoders.
- Now the **default normalization in essentially every open LLM**: T5 (variant), LLaMA-1/2/3, Mistral, Gemma, DeepSeek and the entire Qwen line (Qwen 1 → Qwen 3) place RMSNorm in pre-norm position around every sub-layer.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/1910.07467) · [html](https://arxiv.org/html/1910.07467v1) · [pdf](https://arxiv.org/pdf/1910.07467)
- **Code:** [github.com/bzhangGo/rmsnorm](https://github.com/bzhangGo/rmsnorm)
- **OpenReview:** [NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
- **Related / successor papers:** [Transformer](attention_2017_transformer.md), [SwiGLU](attention_2020_swiglu.md), [QK-Norm](attention_2020_qk-norm.md)
