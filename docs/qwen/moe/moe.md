# Thread: Sparse Mixture-of-Experts in Qwen

How Qwen scaled past dense‑72B parameter budgets without scaling FLOPs proportionally — by adopting **sparsely‑gated Mixture‑of‑Experts (MoE)** layers and inheriting a decade of routing tricks (top‑k gating, capacity factors, load‑balancing losses, expert / shared‑expert specialization).

## Evolution

| Paper | Year | Core idea | Key hyperparam | Used in Qwen |
|---|---|---|---|---|
| [Sparsely-Gated MoE](../../papers/moe_2017_sparsely-gated-moe.md) — Shazeer et al. | 2017 | Insert **MoE layer** between LSTM stacks: a softmax gate picks **top‑k of N** experts per token; only those experts run. Adds **noisy top‑k gating** + an **importance / load auxiliary loss** to keep gating from collapsing onto a few experts. Demonstrates **137 B‑param** language model trained at the FLOPs of a much smaller dense one. | $N{=}512{-}131{,}072$, $k{=}2{-}4$, $\alpha_{\text{aux}}{=}10^{-2}$ | Conceptual ancestor — top‑k gating + aux load loss inherited by every Qwen MoE. |
| [GShard](../../papers/moe_2020_gshard.md) — Lepikhin et al. | 2020 | First **MoE Transformer**: every other FFN replaced by an MoE FFN with **top‑2** routing. Introduces **expert parallelism via SPMD** sharding (`MoE` axis), **capacity factor** $C$ (each expert sees ≤ $C\,\frac{T}{N}$ tokens; overflow tokens skip via residual), and **random secondary routing** so the 2nd expert is sampled with prob $\propto g_2$. Trains a **600 B**-param multilingual MT model on 2048 TPU‑v3 cores. | top‑2, capacity $C{=}1.0{-}2.0$ | Capacity factor + token‑drop semantics + expert‑parallel sharding all reused by Qwen MoE training. |
| [Switch Transformer](../../papers/moe_2021_switch-transformer.md) — Fedus, Zoph, Shazeer | 2021 | Drops $k$ to **top‑1** ("switch" routing): one expert per token → 7× speedup, simpler. Replaces the dual importance/load aux loss with a single **load‑balancing loss** $\alpha N\sum_i f_i\,P_i$ ($f_i$ = fraction of tokens routed to expert $i$, $P_i$ = mean gate prob). Adds **selective fp32 router** + **expert dropout** for stable bf16 training. Trains **1.6 T**‑param Switch‑C. | top‑1, $\alpha{=}10^{-2}$, capacity $1.25$ | Single‑term load‑balance loss and fp32 router are the de‑facto recipe in Qwen MoE; Qwen3 MoE keeps top‑8 (à la DeepSeek) but the loss form is the Switch one. |
| [DeepSeek‑MoE](../../papers/moe_2024_deepseek-moe.md) — Dai et al. | 2024 | Two architectural changes for **expert specialization**: (1) **fine‑grained experts** — split each expert FFN into $m$ smaller ones (so $N{\to}mN$), then activate $mK$ instead of $K$; combinatorial diversity goes from $\binom{N}{K}$ to $\binom{mN}{mK}$. (2) **Shared experts** — isolate $K_s$ "always‑on" experts that absorb common knowledge, leaving routed experts free to specialize. **DeepSeek‑MoE‑16B** matches LLaMA‑2‑7B at ≈ 40 % of the compute. | $m{=}4$, $K{=}2$, $K_s{=}2$ (16B variant) | **Direct architectural template** for Qwen2‑57B‑A14B and the Qwen3 MoEs (128 fine‑grained experts, top‑8 routing). |

## Why this thread matters for Qwen

- **Qwen1.5‑MoE‑A2.7B** (Mar 2024) was Alibaba's first open MoE — 14 B total / 2.7 B active, **top‑4 of 60** routed experts plus **4 shared experts**, upcycled from the dense Qwen1.5‑1.8B (à la *upcycling* — Komatsuzaki et al. 2022). Performance roughly matched dense Qwen1.5‑7B at ~⅓ the active FLOPs.
- **Qwen2‑57B‑A14B** (Jun 2024) scaled to **64 routed experts top‑8 + 8 shared experts**, fine‑grained‑style à la DeepSeek‑MoE; 57 B total params, ~14 B active per token.
- **Qwen3‑30B‑A3B** and **Qwen3‑235B‑A22B** (Apr 2025) standardise on **128 routed experts, top‑8** (`128 / 8` in the Qwen3 spec table); ~3.7 % of experts active per token. The Qwen3 tech report drops the shared‑expert slot used in Qwen2‑MoE — every active expert is routed — and uses a Switch‑style auxiliary load‑balancing loss with a fp32 router.
- All Qwen MoE checkpoints inherit the **expert‑parallel SPMD layout** popularised by GShard, with **capacity factor ~1.25** and overflow tokens skipping via the residual stream.
- Naming convention `Total‑A{Active}` (e.g. **235B‑A22B**) reports total weights vs. tokens × per‑token active weights — the metric that matters for inference cost. Qwen3‑235B‑A22B activates ≈ 9.4 % of weights per token; Qwen3‑30B‑A3B activates ≈ 10 %.

## How the four ideas compose in Qwen MoE

1. **Per‑token routing** — Shazeer's top‑$k$ noisy gating, with $k{=}8$ in Qwen3 MoE, $k{=}4$ in Qwen1.5‑MoE.
2. **Token‑side balance** — Switch‑style auxiliary load‑balancing loss $\alpha N\sum_i f_i P_i$, applied per‑layer; Qwen3 keeps $\alpha\!\approx\!10^{-2}$.
3. **Expert parallelism** — GShard SPMD: experts of one MoE layer are split across devices on a dedicated `MoE` mesh axis; tokens are dispatched via all‑to‑all and combined symmetrically.
4. **Fine‑grained experts** — DeepSeek‑MoE's split: instead of $N{=}16$ wide experts top‑2, use $N{=}128$ narrow experts top‑8 — same $K\cdot d_{\text{ff,expert}}$ but $\binom{128}{8}{\gg}\binom{16}{2}$ possible combinations, encouraging specialization.

## Qwen MoE configurations at a glance

| Model | Total / active | Routed experts | Top‑k | Shared experts | FFN per expert | Layers |
|---|---|---|---|---|---|---|
| Qwen1.5‑MoE‑A2.7B | 14 B / 2.7 B | 60 | 4 | 4 | 1408 | 24 |
| Qwen2‑57B‑A14B | 57 B / 14 B | 64 | 8 | 8 | 2560 | 28 |
| Qwen3‑30B‑A3B | 30 B / 3 B | 128 | 8 | — | 768 | 48 |
| Qwen3‑235B‑A22B | 235 B / 22 B | 128 | 8 | — | 1536 | 94 |

## See also

- [`positional/`](../positional/positional.md) — RoPE / YaRN / DCA, the long‑context machinery these MoE models use orthogonally.
- `attention/` thread (when added) — GQA, FlashAttention‑2, QK‑Norm: the *dense* parts of every MoE block.
- `qwen-papers/` thread (when added) — the Qwen2 / Qwen3 tech reports themselves, which describe the per‑model MoE choices in detail.

## Open follow-ups for this thread

- **Mixtral 8×7B / 8×22B** ([Jiang et al. 2401.04088](https://arxiv.org/abs/2401.04088)) — contemporary 8‑expert‑top‑2 MoE; useful contrast with the fine‑grained 128‑expert‑top‑8 design adopted by Qwen3.
- **MoE upcycling** ([Komatsuzaki et al. 2212.05055](https://arxiv.org/abs/2212.05055)) — initialise MoE experts from a dense checkpoint; the recipe behind Qwen1.5‑MoE‑A2.7B.
- **Expert‑Choice Routing** ([Zhou et al. 2202.09368](https://arxiv.org/abs/2202.09368)) — invert the gating direction so each expert picks its top tokens; eliminates token‑drop without a capacity factor. Not (yet) used in Qwen.
- **Auxiliary‑loss‑free balancing** ([Wang et al. 2408.15664](https://arxiv.org/abs/2408.15664), DeepSeek‑V3) — replaces the load‑balance loss with bias‑adjusted routing; reportedly improves quality. Possible direction for Qwen4‑MoE.
- Per‑paper recaps for the four foundational papers above are available under [`docs/papers/`](../../papers/) (`moe_2017_sparsely-gated-moe.md`, `moe_2020_gshard.md`, `moe_2021_switch-transformer.md`, `moe_2024_deepseek-moe.md`).
