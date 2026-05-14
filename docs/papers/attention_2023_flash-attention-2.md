# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning — Dao, 2023

> **arXiv:** 2307.08691v1 · **Venue:** ICLR 2024 (workshop track) · **Affiliation:** Princeton / Together AI

## TL;DR
FlashAttention-1 ([Dao et al. 2022](https://arxiv.org/abs/2205.14135)) made attention **IO-aware** and exact, slashing memory from $O(N^2)$ to $O(N)$. **FlashAttention-2** triples its arithmetic intensity by (1) cutting non-matmul ops, (2) parallelizing across the **sequence** dimension, and (3) repartitioning work so that warps split K/V instead of Q. Result: ~**73 % of A100 peak FLOPs/s** in the forward pass, **~2× faster than FA-1**, and **~9× faster than vanilla PyTorch attention** — enabling 72 % MFU end-to-end GPT training.

## Problem & motivation
[FlashAttention-1](https://arxiv.org/abs/2205.14135) already exploits SRAM tiling and recomputation, but on A100 it only reaches **25–40 % of peak**. Three causes:

1. **Non-matmul FLOPs dominate.** Each tile scales the running output by `exp(m_old − m_new)`; doing this every iteration consumes a large fraction of the budget on hardware whose matmul throughput vastly exceeds vector throughput.
2. **No parallelism over sequence length.** FA-1 schedules one block per (batch, head) — fine when batch×heads is large, but for **long sequences with small batch** (typical LLM inference and prefill, training of >32 K contexts) GPUs are underused.
3. **Warp partitioning is wasteful.** FA-1 splits Q across warps within a thread block; each warp then needs all of K and V, forcing inter-warp shared-memory traffic in the backward pass.

## Key idea
Three orthogonal kernel-level rewrites:

1. **Defer the rescale.** Maintain unnormalized `O` and the running denominator `ℓ` through the whole loop; divide by `ℓ` *once*, at the end. Eliminates one full division per tile and several exponentials.
2. **Parallelize across sequence.** Add a third grid dimension over rows of $Q$ so that long sequences spread across SMs even at batch = 1.
3. **Split K/V across warps, share Q.** All warps in a block see the same Q tile from registers; each warp owns a strip of K/V, computes its partial output, then a single shared-memory reduction merges them. No cross-warp comm during the inner loop.

## How it works
The exact tiled **online-softmax** recurrence (per row of $Q$, iterating over $K,V$ blocks $j = 1\dots N/B_c$) is:

$$
\begin{aligned}
S_j &= \tfrac{1}{\sqrt{d}}\, Q\, K_j^\top,\\
m_j &= \max\!\left(m_{j-1},\;\max_\text{row} S_j\right),\\
\tilde P_j &= \exp(S_j - m_j),\\
\ell_j &= e^{\,m_{j-1}-m_j}\,\ell_{j-1} + \sum_\text{row}\tilde P_j,\\
O_j &= \operatorname{diag}(e^{\,m_{j-1}-m_j})\, O_{j-1} + \tilde P_j\, V_j.
\end{aligned}
$$

After the last block, $O \leftarrow \operatorname{diag}(\ell)^{-1} O$. FA-2 fuses this final normalization, runs the inner loop with K/V split across warps, and parallelizes the $Q$-row dimension across CTAs.

For the **backward** pass, the same partitioning swaps roles: $dQ$ is parallelized across warps (since each $dK_j, dV_j$ needs the full row of $Q$), and recomputation of $S, P$ from saved $(O, \ell, m)$ avoids storing the $N\!\times\!N$ matrix.

**Block sizes:** $B_r = B_c = 128$ for $d_\text{head} \le 128$; $B_r = 64, B_c = 64$ for $d_\text{head} = 256$. Supports FP16 and BF16, head dims up to 256, causal masking, paged-KV, ALiBi and (in v2.1+) GQA / MQA broadcast.

## Training / data
Pure systems paper — no model training. Benchmarks on A100-80GB, GPT-style configs (head dim 64–256, batch 1–8, sequence 512–16 K), and end-to-end GPT-3-style runs.

## Results
Per Tables/Figures in the paper:

| Setting | Throughput | vs FA-1 | Notes |
|---|---|---|---|
| Forward, A100, fp16, head-dim 128 | **230 TFLOPs/s** (~73 % peak) | **2.0×** | per Fig. 4 |
| Backward, A100, fp16 | ~165 TFLOPs/s (~63 % peak) | 1.7× | per Fig. 5 |
| Combined (fwd+bwd) | ~190 TFLOPs/s | 2.0× | per §3 |
| End-to-end GPT-3-2.7B training | **225 TFLOPs/s/GPU** (72 % MFU) | n/a | per §4 |
| Long-context (16 K, batch 1) fwd | up to 9× | n/a | vs PyTorch `nn.MultiheadAttention` |

Causal-masked variants (the LLM regime) gain disproportionately because masked tiles are entirely skipped under the new scheduling.

## Limitations & follow-ups
- **FP8 / Hopper-specific paths** are not covered — addressed by **FlashAttention-3** ([Shah et al. 2024](https://arxiv.org/abs/2407.08608)), which reaches ~75 % of H100 peak with WGMMA and FP8.
- **Variable-length / packed batches** require the `varlen_func` API; rectangular batches still pad.
- **Triton vs CUDA**: the official kernel is hand-tuned CUDA; reasonable Triton ports exist (xformers, [`flash-attn-jax`](https://github.com/nshepperd/flash_attn_jax)).
- **Adopted by:** LLaMA-2/3, Mistral, Mixtral, Falcon, DeepSeek, Gemma, **all Qwen2/2.5/3** training and inference paths (via `attn_implementation="flash_attention_2"` in HF Transformers, vLLM, SGLang).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2307.08691) · [html](https://arxiv.org/html/2307.08691v1) · [pdf](https://arxiv.org/pdf/2307.08691)
- **Code:** [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **Predecessor (FA-1, NeurIPS 2022):** [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **Successor (FA-3, 2024):** [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- **Related / successor papers:** [Transformer](attention_2017_transformer.md), [GQA](attention_2023_gqa.md), [RoPE](positional_2021_rope-roformer.md)
