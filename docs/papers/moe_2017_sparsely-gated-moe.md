# Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer — Shazeer et al., 2017

> **arXiv:** 1701.06538v1 · **Venue:** ICLR 2017 · **Affiliation:** Google Brain

## TL;DR
Introduces the **Sparsely-Gated Mixture-of-Experts (MoE) layer** — a stack of up to thousands of feed-forward "expert" sub-networks selected per-example by a learned **noisy top-k gating network** — and the auxiliary losses needed to keep gating from collapsing. Demonstrates LSTM language and translation models with **up to 137 B parameters** trained on 32–64 GPUs at FLOPs comparable to a much smaller dense baseline, beating prior SOTA on the 1B Word Benchmark and WMT'14 En→Fr / En→De at lower compute.

## Problem & motivation
Conditional computation — *use only a fraction of the model per example* — had been proposed for years (Bengio et al. 2013/2015, Davis & Arel 2013, Eigen et al. 2013) as the way to scale capacity without scaling FLOPs, but every prior implementation failed in practice on real hardware because of:

1. **GPU branch divergence** — modern accelerators want dense matmuls, not per-example control flow.
2. **Network-bandwidth bottleneck** — distributing experts across devices required all-to-all token shuffles whose cost dominated.
3. **Batch-size shrinkage** — when only $k$ of $N$ experts fire on a token, each expert sees only $\tfrac{kB}{N}$ tokens per batch — too few for efficient batched matmul.
4. **Load imbalance & collapse** — the gate naturally converges to picking the same few "favorite" experts.

This paper resolves all four and ships the first working *sparse* MoE at scale.

## Key idea
Insert an MoE layer between LSTM stacks: it has $N$ expert FFNs $E_1,\dots,E_N$ and a trainable gating network $G$ that outputs a sparse $N$-vector $G(x)$. The layer's output is

$$
y \;=\; \sum_{i=1}^{N} G(x)_i\, E_i(x),\qquad G(x)_i = 0 \text{ for all but } k \text{ indices}.
$$

**Noisy top-k gating** (eq. 3–5 in the paper):

$$
H(x)_i \;=\; (x \cdot W_g)_i \;+\; \mathcal{N}(0,1)\cdot \text{softplus}\big((x\cdot W_{\text{noise}})_i\big),
$$

$$
G(x) \;=\; \mathrm{softmax}\big(\mathrm{KeepTopK}(H(x), k)\big),
$$

where `KeepTopK` zeros out everything except the $k$ largest entries (sets them to $-\infty$ before softmax). The Gaussian noise term is *learned per-component* and is essential for load balancing — it makes the top-k decision stochastic enough for under-used experts to still occasionally fire.

## How it works
- **Conv-style application.** The MoE layer is applied "convolutionally" between LSTM time-steps, so a batch of $B$ sequences of length $T$ produces $B\cdot T$ MoE invocations — each of which routes to its own $k$ experts.
- **Expert parallelism (model parallelism).** Each expert lives on (potentially) a different GPU. After gating, tokens are *all-to-all*-shuffled to the GPU(s) hosting their chosen experts, computed locally, and shuffled back. Because each expert sees $\tfrac{kBT}{N}$ tokens per step, increasing $B$ recovers efficient batched matmul on each expert.
- **$k$ is small** — the paper uses $k=4$ for $N=4{,}096$ and $k=2{-}4$ throughout. Ratio $k/N \ll 1$ is what gives sparsity its compute savings.
- **Hierarchical MoE** (§B) — for $N$ in the tens of thousands, two-level gating: a primary gate picks among groups of experts, a secondary gate picks within. Used for the 131k-expert variant.

**Auxiliary losses (the part everyone copies):**

1. **Importance loss** — $\mathrm{Importance}(X)_i = \sum_{x\in X} G(x)_i$ is each expert's gate-mass over the batch. Penalize the squared coefficient of variation $\mathrm{CV}(\mathrm{Importance})^2$ with weight $w_{\text{importance}}$ to push *gate output* to be uniform across experts.
2. **Load loss** — Importance ≠ actual sample count when noise dominates. The paper defines a *smooth* estimator $\mathrm{Load}(X)_i$ of the expected number of tokens routed to expert $i$ (using the noise distribution to compute $P(\text{expert }i\text{ in top-}k)$ analytically) and penalizes $\mathrm{CV}(\mathrm{Load})^2$ with weight $w_{\text{load}}$.

Both weights are set to $10^{-2}$. Without them, gating collapses onto a few experts within the first few thousand steps.

## Training / data

**Language modeling** — 1B Word Benchmark (Chelba et al. 2013) and a 100B-word Google News dataset. Backbone: stacked LSTM (Zaremba et al. 2014); the MoE layer replaces the FFN between layers. Configurations sweep $N \in \{4, 32, 256, 1024, 4096, 16384, 65536, 131072\}$ keeping per-token compute fixed.

**Machine translation** — GNMT (Wu et al. 2016) with an MoE layer added between encoder and decoder. WMT'14 En→Fr and En→De; multilingual setup (Johnson et al. 2017) — 12 language pairs, single shared model.

**Hardware** — up to 32–64 K40 / K80 GPUs; training time on the order of a few days per run.

## Results

**Language modeling, 1B Word Benchmark (per Table 1 of the paper).** Test perplexity vs. ops/timestep:

| Model | Params (excl. embed) | Ops/timestep | Test PPL |
|---|---|---|---|
| Best previous (Józefowicz et al. 2016, 2-layer LSTM-8192-1024) | 151 M | 151 M | 30.0 |
| **MoE 4 B (32 experts, k=4)** | 4.4 B | 8.9 M | **28.0** |
| **MoE 4 B (256 experts, k=4)** | 4.4 B | 8.9 M | **24.1** |
| **MoE 137 B (65 536 hierarchical experts)** | 137 B | 33 M | **28.0** |

Note: the 137 B model uses ~5× the per-token compute of the dense 151 M baseline, but holds **>900× more parameters**.

**Machine translation, WMT'14 En→Fr (per Table 2).** Single-model BLEU:

| Model | BLEU |
|---|---|
| GNMT (Wu et al. 2016) | 38.95 |
| GNMT-Mixture-of-Experts | **40.56** |
| GNMT+RL ensemble | 41.16 |

**Multilingual NMT (per Table 3).** Single MoE model on 12 language pairs beats the previous multilingual GNMT (Johnson et al. 2017) on **11 of 12 pairs**, *and* beats per-pair specialist GNMT models on a majority of pairs — the first time a multilingual model surpassed bilingual specialists at this scale.

## Limitations & follow-ups
- **LSTM backbone.** The MoE layer is inserted between LSTM stacks. Adapting it to Transformers (where MoE replaces FFN sub-layers) is the contribution of [GShard](positional_2020_gshard.md) and [Switch Transformer](moe_2021_switch-transformer.md). *(local recap links)*
- **Noisy-top-k routing instabilities** — addressed by Switch Transformer (top-1 + simpler load loss) and ST-MoE (bf16 router, expert dropout).
- **Load loss and importance loss are two separate terms** — Switch Transformer collapses them into one.
- **Capacity factor / token-drop semantics** are not yet defined here — added by GShard.
- **Expert specialization** is observed qualitatively but not exploited architecturally — this is the angle of [DeepSeek-MoE](moe_2024_deepseek-moe.md).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/1701.06538) · [pdf](https://arxiv.org/pdf/1701.06538)
- **Code:** —
- **OpenReview:** <https://openreview.net/forum?id=B1ckMDqlg>
- **BibTeX:** [DBLP](https://dblp.uni-trier.de/rec/bibtex/journals/corr/ShazeerMMDLHD17)
- **Related / successor papers:** [GShard (Lepikhin et al. 2020)](moe_2020_gshard.md) · [Switch Transformer (Fedus et al. 2021)](moe_2021_switch-transformer.md) · [DeepSeek-MoE (Dai et al. 2024)](moe_2024_deepseek-moe.md) · ST-MoE (Zoph et al. 2022, arXiv:2202.08906) · Expert-Choice Routing (Zhou et al. 2022, arXiv:2202.09368)
