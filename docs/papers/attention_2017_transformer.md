# Attention Is All You Need — Vaswani et al., 2017

> **arXiv:** 1706.03762v7 · **Venue:** NeurIPS 2017 · **Affiliation:** Google Brain / Google Research / U. Toronto

## TL;DR
Introduces the **Transformer**, a sequence-transduction architecture built entirely on multi-head self-attention with no recurrence or convolution. On WMT'14 EN-DE / EN-FR translation it sets a new SOTA (28.4 / 41.8 BLEU) while training in a fraction of the wall-clock time of the strongest RNN/CNN baselines (3.5 days on 8 P100 GPUs for the Big variant).

## Problem & motivation
- **RNNs** factor computation along the sequence dimension, preventing intra-example parallelism and bottlenecking on memory for long sequences.
- **CNN sequence models** (ConvS2S, ByteNet) connect distant positions only through $O(n/k)$ stacked layers, making long-range learning hard.
- Attention had been used as a *complement* to RNNs ([Bahdanau et al. 2014](https://arxiv.org/abs/1409.0473)) but never as the sole computational primitive of a sequence model.

## Key idea
Replace recurrence with **scaled dot-product attention** plus **multi-head projections**. Self-attention gives every position $O(1)$ path length to every other, exposing maximum parallelism on TPUs/GPUs.

$$\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

$$\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O$$

with $\text{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ and per-head dimension $d_k = d_v = d_\text{model}/h$.

Positions are injected via fixed sinusoidal embeddings (chosen for length extrapolation):

$$PE_{(\text{pos},2i)} = \sin\!\left(\tfrac{\text{pos}}{10000^{2i/d_\text{model}}}\right),\quad PE_{(\text{pos},2i+1)} = \cos\!\left(\tfrac{\text{pos}}{10000^{2i/d_\text{model}}}\right).$$

## How it works
- **Encoder–decoder** with $N=6$ identical blocks each side; each block = sub-layer + residual + **post-LN** ([Ba et al. 2016](https://arxiv.org/abs/1607.06450)).
- **Encoder block:** self-attention → position-wise FFN. **Decoder block:** masked self-attention → cross-attention over encoder output → FFN.
- **FFN:** $\operatorname{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$ with $d_{ff}=2048$ (base) / $4096$ (big).
- **Regularization:** dropout 0.1 (0.3 for big EN-DE) on every sub-layer output, on embeddings + PE; **label smoothing** $\varepsilon_{ls}=0.1$.

| Variant | $N$ | $d_\text{model}$ | $d_{ff}$ | $h$ | $d_k$ | Params |
|---|---|---|---|---|---|---|
| Base | 6 | 512  | 2048 | 8  | 64 | 65 M  |
| Big  | 6 | 1024 | 4096 | 16 | 64 | 213 M |

![Transformer architecture (Figure 1)](_assets/attention_2017_transformer/architecture.png)

## Training / data
- **Data:** WMT'14 EN-DE (4.5 M pairs, 37 K BPE) and WMT'14 EN-FR (36 M pairs, 32 K word-piece).
- **Batches:** ~25 K source + 25 K target tokens per batch.
- **Optimizer:** Adam ($\beta_1=0.9$, $\beta_2=0.98$, $\varepsilon=10^{-9}$) with the now-canonical inverse-sqrt schedule and 4 000 warm-up steps:
  $$\eta_t = d_\text{model}^{-1/2}\cdot\min\!\left(t^{-1/2},\, t\cdot \text{warmup}^{-3/2}\right).$$
- **Hardware:** 8× P100. Base = 100 K steps (~12 h). Big = 300 K steps (~3.5 d).

## Results
| Benchmark | Transformer-Big | Prior best (single) | Notes |
|---|---|---|---|
| WMT'14 EN-DE BLEU | **28.4** | 25.16 (ConvS2S) | per Table 2 |
| WMT'14 EN-FR BLEU | **41.8** | 40.46 (GNMT)    | per Table 2 |
| Training cost (FLOPs) EN-FR | 2.3 × 10¹⁹ | 1.4 × 10²⁰ (GNMT+RL) | ~6× cheaper, per Table 2 |

Also evaluated on English constituency parsing (Table 4): 91.3–92.7 F1, competitive with the best RNN models trained with much more data.

## Limitations & follow-ups
- Sinusoidal PE saturates beyond train length — replaced by **relative** ([Shaw et al. 2018](https://arxiv.org/abs/1803.02155)) and **rotary** PE ([RoPE, 2021](positional_2021_rope-roformer.md)) in modern LLMs.
- **Post-LN** is unstable at depth — **Pre-LN** ([Xiong et al. 2020](https://arxiv.org/abs/2002.04745)) is now standard.
- Quadratic memory in sequence length — addressed by IO-aware kernels ([FlashAttention-2](attention_2023_flash-attention-2.md)) and by KV-cache compression ([GQA](attention_2023_gqa.md)).
- ReLU FFN superseded by GLU variants ([SwiGLU](attention_2020_swiglu.md)); LayerNorm by [RMSNorm](attention_2019_rmsnorm.md); separate embeddings by [tied embeddings](attention_2016_tied-embeddings.md).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/1706.03762) · [html](https://arxiv.org/html/1706.03762v7) · [pdf](https://arxiv.org/pdf/1706.03762)
- **Code (reference):** [tensor2tensor](https://github.com/tensorflow/tensor2tensor) · [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- **BibTeX:** [NeurIPS proceedings page](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
- **Related / successor papers:** [RoPE](positional_2021_rope-roformer.md), [Tied embeddings](attention_2016_tied-embeddings.md), [RMSNorm](attention_2019_rmsnorm.md), [SwiGLU](attention_2020_swiglu.md), [QK-Norm](attention_2020_qk-norm.md), [GQA](attention_2023_gqa.md), [FlashAttention-2](attention_2023_flash-attention-2.md)
