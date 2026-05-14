# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al., 2023

> **arXiv:** 2305.13245v3 · **Venue:** EMNLP 2023 · **Affiliation:** Google Research

## TL;DR
Decoder inference is bottlenecked by **KV-cache memory bandwidth**, not FLOPs. **Multi-Query Attention** ([Shazeer 2019](https://arxiv.org/abs/1911.02150)) shrinks the KV cache by sharing one K/V head across all query heads, but degrades quality and trains unstably. **Grouped-Query Attention** interpolates: $G$ K/V heads, each shared by $H/G$ query heads. With **uptraining** — converting an existing MHA checkpoint by mean-pooling K/V and continuing pre-training for ~5 % of the original budget — GQA-8 reaches near-MHA quality at near-MQA inference speed.

## Problem & motivation
At each autoregressive step, decoding a Transformer requires loading the full KV cache for every layer, every head and every past token. With $H$ K/V heads, head-dim $d$, $L$ layers and context length $T$:

$$\text{KV bytes/token} = 2\cdot L\cdot H\cdot d\cdot \text{dtype}.$$

For a 70 B model at 32 K context this is tens of gigabytes streamed *per token* — far exceeding compute, so HBM bandwidth dominates latency. **MQA** ($H_\text{kv}=1$) cuts this by $H\times$, but training from scratch is brittle, fine-tuning a pre-trained MHA model into MQA is non-trivial, and quality drops (especially on long-input tasks).

## Key idea
**GQA-$G$**: split the $H$ query heads into $G$ groups; each group attends through **one** shared K/V head. Special cases:
- $G=H$ recovers **MHA**.
- $G=1$ recovers **MQA**.
- $1<G<H$ is the new regime, with KV cache shrunk by $H/G$.

For **uptraining** an existing MHA model into GQA:

1. For each new K/V head, **mean-pool** the K (resp. V) projection matrices of the $H/G$ MHA heads it replaces.
2. Continue pre-training on the original recipe for $\alpha \approx 5\%$ of the original number of steps.
3. Fine-tune normally for downstream tasks.

Mean-pooling beats both random init and "pick one head" by a large margin (Figure 4 of the paper).

![MHA → GQA → MQA conversion (Figure 1 / 2)](_assets/attention_2023_gqa/architecture.png)

## How it works
- **Memory:** KV cache scales with $H_\text{kv}=G$ instead of $H$ — a $H/G$× reduction. With Qwen2-7B's $H=28$, $H_\text{kv}=4$ that is a **7× cache reduction** vs. MHA at the same head dim.
- **Compute:** Identical FLOPs to MHA; only the K/V projections shrink.
- **Implementation:** during attention, `repeat_interleave(K, H/G, dim=heads)` (and same for V) before the dot product, or — in modern kernels (FlashAttention-2, xformers, TE) — pass `num_kv_heads=G` directly so the kernel broadcasts internally without materializing copies.
- **Stability:** GQA trains stably from scratch *and* from MHA checkpoints; MQA frequently needs run-averaging to converge.

## Training / data
- Base model: T5.1.1-XXL (11 B encoder-decoder).
- Uptraining corpus: same as original T5 pre-training (C4) at the same recipe.
- Uptraining budget: $\alpha = 0.05$ (i.e. 5 % of pre-training steps) is the sweet spot per the ablation in Figure 5; gains saturate beyond $\alpha=0.10$.
- Fine-tuned and evaluated on summarization (CNN/DM, XSum, arXiv, PubMed, MediaSum, MultiNews) and translation (WMT'14 EN-DE).

## Results
Per Table 1 of the paper (T5-XXL, time per inference sample in seconds, ROUGE / BLEU):

| Model    | Time/sample | CNN/DM R1 | arXiv R1 | PubMed R1 | MediaSum R1 | MultiNews R1 | WMT BLEU |
|---|---|---|---|---|---|---|---|
| MHA-Large | 0.37 s | 46.0 | 42.9 | 44.6 | 46.2 | 35.5 | 46.6 |
| MHA-XXL   | 1.51 s | 47.2 | 43.8 | 45.6 | 47.5 | 36.4 | 46.9 |
| MQA-XXL   | **0.24 s** | 46.6 | 43.0 | 45.0 | 46.9 | 36.1 | 46.5 |
| **GQA-8-XXL** | **0.28 s** | **47.1** | **43.5** | **45.4** | **47.7** | **36.3** | **47.2** |

GQA-8 sits within noise of MHA-XXL on every task while running ~5.4× faster (close to MQA), and dominates MQA on every metric.

## Limitations & follow-ups
- The choice of $G$ is empirical; $G=8$ is the "free lunch" for XXL, but optimal $G$ scales with $H$ (rule of thumb in modern LLMs: $G = H/8$ or $H_\text{kv}\!\in\!\{4,8\}$).
- Uptraining requires the original pre-training data and recipe.
- Successor: **MLA** (Multi-Head Latent Attention, [DeepSeek-V2 2024](https://arxiv.org/abs/2405.04434)) compresses the KV cache further by projecting to a low-rank latent before caching; orthogonal to GQA.
- **Adopted by:** LLaMA-2-70B, LLaMA-3-all, Mistral, Mixtral, Gemma, Falcon-180B, **every Qwen2/2.5/3 model** (e.g. Qwen2-7B uses 28 Q heads, 4 KV heads; Qwen3-32B uses 64 Q heads, 8 KV heads).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2305.13245) · [html](https://arxiv.org/html/2305.13245v3) · [pdf](https://arxiv.org/pdf/2305.13245)
- **ACL Anthology:** [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.298/)
- **MQA predecessor:** [Shazeer 2019, *Fast Transformer Decoding*](https://arxiv.org/abs/1911.02150)
- **Successor:** [DeepSeek-V2 / MLA (2024)](https://arxiv.org/abs/2405.04434)
- **Related / successor papers:** [Transformer](attention_2017_transformer.md), [FlashAttention-2](attention_2023_flash-attention-2.md), [RoPE](positional_2021_rope-roformer.md)
