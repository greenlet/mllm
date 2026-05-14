# Query-Key Normalization for Transformers — Henry et al., 2020

> **arXiv:** 2010.04245v1 · **Venue:** Findings of EMNLP 2020 · **Affiliation:** Cyndx

## TL;DR
L2-normalize the query and key vectors **per head** before the attention dot product, then rescale by a learned temperature $g$. This bounds the pre-softmax logits to $[-1,1]\cdot g$ regardless of activation magnitude, eliminating softmax saturation that destabilizes training — especially in low-resource MT and very wide models. Reported gain: +0.93 BLEU on average across five low-resource translation pairs; later adopted by **ViT-22B**, **Chameleon**, and **Qwen3** to stabilize multi-billion-parameter training.

## Problem & motivation
Scaled dot-product attention divides by $\sqrt{d_k}$ to keep logits in a "reasonable" regime — but only at *initialization*. As training proceeds (and especially when $\lVert q\rVert$ and $\lVert k\rVert$ drift), the inner product $q\!\cdot\! k$ can grow without bound. Saturated softmax cells receive vanishing gradients, training stalls or diverges, and the failure mode is more frequent on small data and on very large/deep models.

## Key idea
Normalize $q$ and $k$ to the unit sphere along the head dimension and replace the constant $1/\sqrt{d_k}$ scale with a **learnable** scalar $g$ (per layer or per head):

$$\operatorname{QK\text{-}Norm}(Q,K,V) = \operatorname{softmax}\!\left( g \cdot \frac{Q}{\lVert Q\rVert_2}\cdot\frac{K^\top}{\lVert K\rVert_2}\right) V.$$

The dot product of two unit vectors lies in $[-1,1]$, so logits live in $[-g, g]$ and saturation is governed by a single trainable knob.

## How it works
- $\lVert Q\rVert_2$ / $\lVert K\rVert_2$ are computed **per head, per token** (along $d_k$); cost is negligible.
- $g$ is a learnable parameter; the paper recommends initializing $g \approx \log(\text{seq\_len})$ so that even uniform attention can produce sharp distributions when needed.
- Drop the $1/\sqrt{d_k}$ factor — it is absorbed into $g$.
- Implementation is one extra `F.normalize(q, dim=-1)` and one for $k$.

## Training / data
- Five low-resource WMT/IWSLT pairs (TED Talks corpus): **EN↔VI, EN↔HA, EN↔CHV, EN↔KK, EN↔SI**.
- Standard Transformer-base, only the attention op changed.

## Results
| Pair | Baseline BLEU | + QK-Norm | Δ | Source |
|---|---|---|---|---|
| EN→VI | 31.4 | **32.4** | +1.0 | per Table 2 |
| EN→HA | 16.0 | **17.5** | +1.5 | per Table 2 |
| EN→CHV | 13.5 | **14.0** | +0.5 | per Table 2 |
| EN→KK | 8.5  | **8.9**  | +0.4 | per Table 2 |
| EN→SI | 7.4  | **8.0**  | +0.6 | per Table 2 |
| **average over 5 pairs** | — | — | **+0.93 BLEU** | per §4 |

Beyond BLEU, the paper visualizes that attention entropy stays well-behaved through training instead of collapsing onto single tokens.

## Limitations & follow-ups
- Adds two L2 norms per attention call. With FlashAttention-2 the cost is hidden (it can be fused into the K/V load).
- Some implementations (e.g. **Qwen3**, **Chameleon**) prefer a per-head **RMSNorm** (with learned gain) on $q$ and $k$ instead of literal L2-normalization — same effect, drop-in compatible with [RMSNorm](attention_2019_rmsnorm.md) kernels.
- ViT-22B ([Dehghani et al. 2023](https://arxiv.org/abs/2302.05442)) reports that QK-Norm was the single change that made training a 22 B-parameter ViT possible.
- In LLMs, QK-Norm is now a default at frontier scale (Qwen3, Chameleon, Stable Diffusion 3).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2010.04245) · [html](https://arxiv.org/html/2010.04245v1) · [pdf](https://arxiv.org/pdf/2010.04245)
- **ACL Anthology:** [Findings of EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.379/)
- **Code:** [github.com/CyndxAI/QKNorm](https://github.com/CyndxAI/QKNorm)
- **Related / successor papers:** [Transformer](attention_2017_transformer.md), [RMSNorm](attention_2019_rmsnorm.md), [GQA](attention_2023_gqa.md), [FlashAttention-2](attention_2023_flash-attention-2.md)
