# Thread: Attention & normalization backbone

The pieces of the modern decoder-only Transformer block that **every** Qwen model — text, VL, audio, omni, embedding, MoE — shares with its [positional](../positional/positional.md) stack: how attention is computed, normalized, projected, and gated.

## Evolution

| Paper | Year | Core contribution | Key knob | Used in Qwen |
|---|---|---|---|---|
| [Transformer / *Attention Is All You Need*](../../papers/attention_2017_transformer.md) | 2017 | Multi-head **scaled dot-product** self-attention; replaces RNN/CNN sequence models. | $h$, $d_\text{model}$, $d_{ff}\!=\!4d$ | Architectural backbone of every Qwen model. |
| [Tied embeddings (Press & Wolf)](../../papers/attention_2016_tied-embeddings.md) | 2016 | Tie input embedding $U$ with output projection $V$ — same matrix plays both roles. | weight-tied vs untied | **Tied** in small Qwen2/2.5/3 (≤ 1.5 B); **untied** in Qwen2/2.5/3 ≥ 7 B (param saving negligible at $d_\text{model}\!\geq\!4096$). |
| [RMSNorm](../../papers/attention_2019_rmsnorm.md) | 2019 | Drop the mean re-centering of LayerNorm; normalize by $\sqrt{\overline{x^2}}$ only. | $\varepsilon$, gain $g$ | Default normalization in **every** Qwen model, in pre-norm position around every sub-layer. |
| [SwiGLU](../../papers/attention_2020_swiglu.md) | 2020 | Replace the FFN's ReLU with a **gated** GLU using Swish: $(\operatorname{Swish}(xW_g)\odot xW_u)W_d$. | $d_{ff}\!\approx\!\tfrac{8}{3}d_\text{model}$ | FFN of every Qwen text model and every MoE expert (`gate_proj` / `up_proj` / `down_proj`). |
| [QK-Norm](../../papers/attention_2020_qk-norm.md) | 2020 | L2-normalize $q,k$ per head before the dot product; learnable temperature $g$. | init $g\!\approx\!\log T$ | **Qwen3** (added on top of GQA + RoPE) for stable training of 30 B-A3B / 235 B-A22B and bias-free linears. |
| [GQA](../../papers/attention_2023_gqa.md) | 2023 | Group $H$ query heads into $G$ groups, each sharing one K/V head — interpolates MHA ↔ MQA. Uptraining converts MHA checkpoints in 5 % of pre-training. | $H_\text{kv}\!=\!G$ | Every Qwen2/2.5/3 dense and MoE model (e.g. Qwen2-7B: $H\!=\!28$, $H_\text{kv}\!=\!4$; Qwen3-32B: $H\!=\!64$, $H_\text{kv}\!=\!8$). |
| [FlashAttention-2](../../papers/attention_2023_flash-attention-2.md) | 2023 | IO-aware exact attention kernel with sequence-parallelism and warp-level K/V splitting; ~73 % of A100 peak. | $B_r$, $B_c$ block sizes | Default training & inference kernel for every Qwen2/2.5/3 model in HF Transformers, vLLM, SGLang. |

## Why this thread matters for Qwen

- The **block recipe** "**RMSNorm → MHA/GQA(+RoPE[+QK-Norm]) → residual → RMSNorm → SwiGLU FFN → residual**" is invariant from Qwen 1.0 (2023) through Qwen 3 (2025); only $G$ (KV heads), $\varepsilon$, $d_{ff}$ ratio, and the addition of QK-Norm in Qwen3 change.
- **GQA** is the single architectural change that makes 32 K → 128 K → 1 M context economically serveable (the [positional](../positional/positional.md) stack handles the *positional* extrapolation; GQA handles the *KV-cache* cost).
- **FlashAttention-2** lets the 128 K and 1 M context regimes — already enabled by [YaRN](../../papers/positional_2023_yarn-context-extension.md) + [DCA](../../papers/positional_2024_dca-dual-chunk-attention.md) — actually run in finite memory and time. The Qwen2.5-1M tech report cites it as the kernel of choice for both training and inference.
- **QK-Norm** is the stabilizer that allowed Qwen3 to drop linear biases and train the 235 B-A22B model end-to-end without loss spikes (analogous to the role it played in [ViT-22B](https://arxiv.org/abs/2302.05442) and Chameleon).
- **SwiGLU + RMSNorm + tied/untied embeddings** are unchanged since Qwen 1.0 — they are inherited LLaMA/PaLM-style defaults that Qwen never revisited because the costs are tiny and the gains compound across every layer.

## Multimodal reuse

The same attention/norm stack runs inside the multimodal towers:
- **Qwen2-VL / 2.5-VL** ViT uses RMSNorm + SwiGLU + GQA + FlashAttention-2 (with **2-D RoPE** instead of 1-D).
- **Qwen2.5-Omni "Thinker"** runs the same block on text + visual + audio tokens; the "Talker" codec decoder reuses the FFN/norm pieces but replaces attention with causal codec attention.
- **Qwen3-Embedding** is a Qwen3 backbone with last-token pooling — the attention/norm stack is identical to the chat models.

## See also

- [Positional encoding & long-context scaling](../positional/positional.md) — RoPE / NTK / YaRN / DCA, composable with everything above.
- [Sparse Mixture-of-Experts](../moe/moe.md) — replaces the SwiGLU FFN with a router + many SwiGLU experts; everything else in this thread is unchanged.
- [DeepSeek-V2 / MLA (2024)](https://arxiv.org/abs/2405.04434) — KV-cache compression beyond GQA; not yet adopted by Qwen.

## Open follow-ups for this thread

- **FlashAttention-3** ([Shah et al. 2024](https://arxiv.org/abs/2407.08608)) — Hopper-specific WGMMA + FP8 kernel; ~75 % of H100 peak. TODO recap.
- **Pre-LN vs Sandwich-LN vs DeepNorm** — alternative residual recipes for very deep stacks; not currently used by Qwen but relevant for any deeper variant.
- **Multi-head Latent Attention (MLA)** — DeepSeek-V2's low-rank KV compression. TODO recap; orthogonal to GQA.
- **Differential Transformer** ([Ye et al. 2024](https://arxiv.org/abs/2410.05258)) — two attention maps subtracted; claims better long-context retrieval.
