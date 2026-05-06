# GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding — Lepikhin et al., 2020

> **arXiv:** 2006.16668v1 · **Venue:** ICLR 2021 · **Affiliation:** Google

## TL;DR
GShard is **(a) a SPMD compiler / annotation API on top of XLA** that auto-shards giant models across thousands of accelerators, and **(b) the first MoE-Transformer recipe** to scale conditional computation past **600 B parameters**. A 600 B-param **multilingual MoE Transformer** trained on 2 048 TPU v3 cores in **4 days** delivers far better quality on translation from **100 languages → English** than dense Transformer baselines that took *order-of-magnitude more compute*.

## Problem & motivation
After [Shazeer et al. 2017](moe_2017_sparsely-gated-moe.md) showed that sparsely-gated MoE could push LSTMs to 137 B parameters, two things were still missing for the Transformer era:

1. **An MoE-Transformer recipe.** Where exactly should expert layers go? How many experts per layer? How is routing implemented when the FFN is replaced by an expert-bank?
2. **A scalable sharding story.** Hand-written model-parallel code (Mesh-TensorFlow, Megatron-LM) was author-time intensive and couplings between data-, model-, and expert-parallel axes broke down at 1000+ devices.

GShard delivers both: a tiny `replicate / split / shard` annotation API the user attaches to a few key tensors, and an MoE-Transformer that becomes the de-facto template for nearly every large MoE since.

## Key idea
**MoE Transformer** — replace **every other** FFN sub-layer in a Transformer encoder/decoder with an MoE layer of $E$ experts. Each token independently picks its **top-2** experts via softmax gating; the layer's output is the gate-weighted sum of the two experts' FFN outputs.

$$
y_t \;=\; g_{i_1,t}\,\mathrm{FFN}_{i_1}(x_t) \;+\; g_{i_2,t}\,\mathrm{FFN}_{i_2}(x_t),\qquad (i_1, i_2) = \mathrm{top}_2(\mathrm{softmax}(W_g x_t)).
$$

Three new mechanisms make this work at 600 B parameters:

- **Capacity factor $C$** — each expert is statically allocated $\big\lceil C\cdot \tfrac{T}{E}\big\rceil$ token slots per batch (with $T$ the total token count and $E$ the number of experts). $C\!\in\![1.0, 2.0]$ in the paper. **Tokens that overflow** their chosen expert's capacity are *dropped* (i.e., they skip the expert and pass through the residual unchanged). This makes the per-expert workload deterministic and therefore SPMD-shardable.
- **Random routing for the second expert** — given the top-2 gate values $g_1 \geq g_2$, the second expert is *kept with probability* $\propto g_2 / g_1$ (or simply with prob $g_2$). If skipped, only the first expert fires. This saves bandwidth on weak second choices.
- **Auxiliary load-balancing loss** $\mathcal{L}_{\text{aux}} = E\cdot\sum_{i=1}^{E} f_i\cdot \overline{P}_i$ where $f_i$ is the fraction of tokens routed to expert $i$ and $\overline{P}_i$ is the mean gate prob for expert $i$. Inherited (in slightly different form) from Shazeer et al. 2017.

**Sharding annotations** — three primitives the user adds to the model:
- `replicate(t)` — same on every device.
- `split(t, dim, mesh_axis)` — sharded along one tensor dimension across one mesh axis.
- `shard(t, mesh)` — fully partitioned per a user-given device mesh.

The XLA compiler then propagates the annotations through the graph (using the SPMD partitioner), inserting all-reduce / all-to-all collectives where needed. For MoE specifically: the `MoE` axis of the device mesh holds one expert per device, and the all-to-all dispatch/combine pattern is generated automatically.

## How it works
- **Architecture.** 36-layer Transformer encoder + decoder, $d_{\text{model}}=1024$, $d_{\text{ff}}=8192$, 16 attention heads. MoE layers replace every second FFN. $E\!\in\!\{128, 512, 2048\}$ experts; per-expert FFN is identical in shape to the dense FFN.
- **Largest model.** $E=2048$ experts × 36 MoE layers ⇒ **~600 B params** (≈ 50 % of which are sparsely activated FFN weights). Activated params per token ≈ 1.5 B (two experts × FFN).
- **Hardware.** 2 048 TPU v3 cores; the `MoE` mesh axis holds 2 048 experts (one per core when $E=2048$); a separate data-parallel axis handles batch.
- **Training time.** 4 days for the 600 B model; smaller variants in 24–36 hours.

## Training / data
- **Task.** Multilingual machine translation from **100 languages → English**, single shared model, 25 B sentence-pair training corpus mined from the web.
- **Loss.** Standard cross-entropy + load-balance auxiliary with weight $0.01$.
- **Optimizer.** Adafactor.
- **Capacity factor.** $C=2.0$ during training, $C=1.0$ at inference (the paper notes inference quality is robust down to $C=1$).

## Results

**Translation BLEU (avg over 100 language pairs)** — sourced from Figure 4 / Table in the paper:

| Model | Params | TPU·core·days | $\Delta$ BLEU vs. bilingual baseline |
|---|---|---|---|
| Bilingual dense Transformer (per pair) | ~400 M each | — | 0 (reference) |
| Multilingual dense Transformer (T = 96 layers, 2.3 B params) | 2.3 B | ~6× MoE-600B | +5.8 |
| **MoE Transformer, $E=128$, 12 layers** | 12.9 B | 22 | +6.1 |
| **MoE Transformer, $E=512$, 36 layers** | 150 B | 178 | +12.0 |
| **MoE Transformer, $E=2048$, 36 layers** | **600 B** | **22** *(parallel on 2048 cores)* = 4 days | **+13.5** |

Headline: the 600 B MoE achieves **+13.5 BLEU avg** over per-pair bilingual baselines at **~6× lower wall-clock** than the dense 96-layer multilingual baseline.

**Per-language analysis (Figure 6).** Improvements are most pronounced for low-resource languages (>+15 BLEU on the bottom-quartile pairs) — multitask transfer benefits dominate the parameter budget when capacity is no longer the bottleneck.

## Limitations & follow-ups
- **Top-2 gating + capacity factor is wasteful** — the second expert is often weak; this motivates [Switch Transformer](moe_2021_switch-transformer.md)'s top-1.
- **Token dropping is bad for inference quality** at $C\!=\!1$ — addressed by Expert-Choice Routing (Zhou et al. 2022) which inverts the gating direction.
- **No expert specialization analysis** — the paper measures aggregate quality but not what each expert learns. [DeepSeek-MoE](moe_2024_deepseek-moe.md) takes this on directly.
- **Compiler complexity.** GShard's SPMD partitioner became GSPMD (Xu et al. 2021) and now ships as PartIR / PJRT in JAX/TF, but reproducing the recipe outside Google requires equivalent infrastructure (DeepSpeed-MoE, Tutel, Megatron-MoE).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2006.16668) · [pdf](https://arxiv.org/pdf/2006.16668)
- **OpenReview:** <https://openreview.net/forum?id=qrwe7XHTmYb>
- **Code:** Not publicly released; community ports include DeepSpeed-MoE, FastMoE, Tutel.
- **BibTeX:** [DBLP](https://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-2006-16668)
- **Related / successor papers:** [Sparsely-Gated MoE (Shazeer et al. 2017)](moe_2017_sparsely-gated-moe.md) · [Switch Transformer (Fedus et al. 2021)](moe_2021_switch-transformer.md) · [DeepSeek-MoE (Dai et al. 2024)](moe_2024_deepseek-moe.md) · GSPMD (Xu et al. 2021, arXiv:2105.04663) · ST-MoE (Zoph et al. 2022, arXiv:2202.08906) · GLaM (Du et al. 2022, arXiv:2112.06905)
