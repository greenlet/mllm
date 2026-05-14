# GLU Variants Improve Transformer — Shazeer, 2020

> **arXiv:** 2002.05202v1 · **Venue:** preprint (never formally published) · **Affiliation:** Google

## TL;DR
Replaces the Transformer's two-layer ReLU/GELU FFN with **gated** variants built from [GLU (Dauphin et al. 2017)](https://arxiv.org/abs/1612.08083). The Swish-gated version, **SwiGLU**, gives the cleanest improvements on T5 pre-training and downstream tasks at matched parameter count, and has since become the default FFN in every open frontier LLM (PaLM, LLaMA, Mistral, Qwen, Gemma).

## Problem & motivation
The original FFN — $\operatorname{FFN}(x)=\max(0, xW_1)W_2$ — is the simplest possible MLP. Many activations have been tried (GELU, Swish, ELU…) with mixed results. GLU instead uses two parallel projections, one of which **gates** the other, and had shown gains in NLP convolutions but had not been systematically evaluated inside Transformers.

## Key idea
Take the GLU recipe — $\operatorname{GLU}(x; W, V) = \sigma(xW)\odot xV$ — and instantiate it with several activations on the gate, dropping all biases:

$$\begin{aligned}
\operatorname{ReGLU}(x)  &= \operatorname{ReLU}(xW)\odot xV,\\
\operatorname{GEGLU}(x) &= \operatorname{GELU}(xW)\odot xV,\\
\operatorname{SwiGLU}(x) &= \operatorname{Swish}_1(xW)\odot xV,\\
\operatorname{Bilinear}(x) &= (xW)\odot xV.
\end{aligned}$$

Plugged into the FFN as $\operatorname{FFN}_{\text{GLU}}(x) = \big(\operatorname{*GLU}(x)\big)W_2$, this introduces a **third** weight matrix. To keep the parameter count equal to the standard FFN, $d_{ff}$ is reduced from $4d_\text{model}$ to $\tfrac{8}{3}d_\text{model}$ (rounded to a multiple of e.g. 128).

Swish is $\operatorname{Swish}_\beta(x) = x \cdot \sigma(\beta x)$; SwiGLU uses $\beta = 1$.

## How it works
- Each FFN layer now has weights $W_1, V_1 \in \mathbb{R}^{d\times d_{ff}}$ and $W_2\in\mathbb{R}^{d_{ff}\times d}$ — three matmuls instead of two.
- For Qwen-style implementations the two input projections are conventionally named `gate_proj` ($W_1$) and `up_proj` ($V_1$), with `down_proj` = $W_2$:

  ```python
  def swiglu_ffn(x):
      return down_proj(F.silu(gate_proj(x)) * up_proj(x))
  ```
  (`F.silu` is PyTorch's name for $\operatorname{Swish}_1$.)

- Bias terms are removed throughout, matching the LLaMA/Qwen convention.

## Training / data
- T5-Base recipe: encoder–decoder Transformer, **C4** span-corruption pre-training for 524 K steps at batch 65 K tokens.
- **Identical** optimizer, schedule and dropout to the baseline; *only* the FFN swap differs.
- Fine-tuning evaluation on **GLUE**, **SuperGLUE** and **SQuAD v1.1 / v2.0**.

## Results
Pre-training perplexity on the held-out C4 split (per Table 1 of the paper):

| FFN | Log-perplexity ↓ |
|---|---|
| ReLU (baseline) | 1.997 |
| GELU            | 1.983 |
| Swish           | 1.994 |
| Bilinear (no activation) | 1.960 |
| ReGLU           | 1.953 |
| **GEGLU**       | **1.942** |
| **SwiGLU**      | **1.944** |

Downstream fine-tuning (per Tables 2–5 of the paper, average over GLUE / SuperGLUE / SQuAD): **GEGLU** and **SwiGLU** are the only variants that **strictly dominate** the ReLU baseline on every task within noise; e.g. on SuperGLUE the average rises from 71.7 (ReLU) to **73.2** (SwiGLU).

The paper famously offers no theoretical explanation, attributing the result "as all else, to divine benevolence."

## Limitations & follow-ups
- 50 % more parameters per FFN matrix (3 matmuls vs 2) — offset by reducing $d_{ff}$ to $\tfrac{8}{3}d_\text{model}$.
- No ablation of $\beta$ in $\operatorname{Swish}_\beta$ — later work (PaLM) uses learnable $\beta$ but reports no improvement.
- Adopted by **PaLM**, **LLaMA-1/2/3**, **Mistral**, **Gemma**, **DeepSeek**, and the **entire Qwen line** (Qwen 1 → Qwen 3, including all MoE experts and VL/Audio projector MLPs).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2002.05202) · [html](https://arxiv.org/html/2002.05202v1) · [pdf](https://arxiv.org/pdf/2002.05202)
- **GLU origin:** [Dauphin et al. 2017](https://arxiv.org/abs/1612.08083)
- **Swish:** [Ramachandran et al. 2017](https://arxiv.org/abs/1710.05941)
- **Related / successor papers:** [Transformer](attention_2017_transformer.md), [RMSNorm](attention_2019_rmsnorm.md)
