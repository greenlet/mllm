# Thread: Positional encoding & long-context scaling for RoPE LLMs

How Qwen got from a 2k–8k pre-training window to **1M-token inference contexts**, and which paper contributed which trick.

## Evolution

| Paper | Year | Core idea | Key hyperparam | Used in Qwen |
|---|---|---|---|---|
| [RoPE / RoFormer](../../papers/positional_2021_rope-roformer.md) | 2021 | Encode position by **rotating** $q,k$ pairs by $m\theta_i$; relative position falls out of the inner product. | base $b=10000$ | All Qwen text models (Qwen1, 1.5, 2, 2.5, 3) — backbone positional encoding |
| [NTK-aware RoPE](../../papers/positional_2023_ntk-aware-rope.md) | 2023 | Don't stretch positions; **rescale the base** $b\!\to\!b\cdot s^{d/(d-2)}$ so high-freq dims survive. | scale $s$ | Qwen-7B / 14B (Dynamic NTK at inference for ad-hoc context extension) |
| [YaRN](../../papers/positional_2023_yarn-context-extension.md) | 2023 | Per-dimension **piecewise** interpolation (NTK-by-parts) + softmax temperature $1/t = 0.1\ln s + 1$. ~10× cheaper than PI. | $\alpha=1, \beta=32$, $s$ | Qwen2 / 2.5 (32k → 128k); core long-context recipe |
| [DCA](../../papers/positional_2024_dca-dual-chunk-attention.md) | 2024 | **Training-free** chunked attention: intra/inter/successive-chunk position remappings keep all relative distances ≤ $c-1$. | chunk $s\!\approx\!\tfrac34 c$, window $w=c-s$ | Qwen2.5 long-context (128k); stacked with YaRN to 1M in **Qwen2.5-1M / Qwen2.5-Turbo** |

## Why this thread matters for Qwen

- **Qwen 1 / 1.5 / 2 / 2.5 / 3** all use **RoPE** as the per-head positional mechanism on $q$ and $k$. Qwen3 additionally adds **QK-Norm** on top.
- **Qwen-7B** introduced **Dynamic NTK** (NTK-aware with $s$ recomputed per forward) to allow zero-shot extension to ~32k beyond the 8k pre-training window.
- **Qwen2 / Qwen2.5** standardized the recipe **YaRN + DCA** for long-context inference at 128k. The Qwen2.5 tech report cites both explicitly.
- **Qwen2.5-1M / Qwen2.5-Turbo** (arXiv 2501.15383) push to **1 000 000** tokens by combining YaRN (frequency-side fix) with DCA (chunked attention pattern), plus a brief long-context fine-tune of ≈ 1B tokens — the paper attributes the fact that this is feasible with so little training to YaRN's data-efficiency and DCA's training-freeness.
- **Qwen2-VL / Qwen2.5-Omni** generalize RoPE to multimodal axes (M-RoPE / TMRoPE), so the rotational mechanism from this thread also underlies the visual / temporal positional encoding for vision and audio (those papers are in the `qwen-papers` thread).

## See also
- `qwen-papers/` thread (when added) — Qwen2-VL's M-RoPE and Qwen2.5-Omni's TMRoPE.
- `attention/` thread (when added) — GQA / FlashAttention-2 / QK-Norm, which are independent but composable with everything above.

## Open follow-ups for this thread
- **Position Interpolation** ([Chen et al. 2306.15595](https://arxiv.org/abs/2306.15595)) — historical baseline; TODO recap.
- **StreamingLLM** ([Xiao et al. 2309.17453](https://arxiv.org/abs/2309.17453)) — alternative streaming approach; orthogonal to DCA.
- **LongLora** ([Chen et al. 2309.12307](https://arxiv.org/abs/2309.12307)) — sparse-attention fine-tune competitor.
