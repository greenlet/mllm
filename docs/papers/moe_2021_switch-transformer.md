# Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity — Fedus, Zoph, Shazeer, 2021

> **arXiv:** 2101.03961v3 · **Venue:** JMLR 2022 · **Affiliation:** Google Brain

## TL;DR
Switch Transformer simplifies MoE routing to **top-1** ("switch" routing) — each token goes to a single expert — and shows that this drop-in simplification gives **up to 7× pre-training speed-up** vs. T5-Base/Large at fixed compute, scales to a **1.6 T-parameter** Switch-C, trains stably in **bfloat16** with a single load-balancing loss, and even improves **multilingual mT5** across all 101 languages.

## Problem & motivation
By 2020, MoE in Transformers was working ([GShard](moe_2020_gshard.md)) but inherited three problems from Shazeer et al. 2017:

1. **Top-2 gating is complex** — needs a "second-best" path with random sampling, an extra all-to-all, and an extra FFN compute.
2. **Two auxiliary losses** (importance + load) are awkward to tune.
3. **Training instability in bfloat16.** Sparse models historically required fp32 throughout — too expensive to scale.

Switch Transformer asks: *what's the simplest possible MoE that still works?* Answer: $k=1$ + one balance loss + a few precision and stability tricks.

## Key idea

**Top-1 ("switch") routing.** For token $x$, the gate $h(x) = W_r x \in \mathbb{R}^N$ produces logits over $N$ experts. Pick the single argmax expert $i^* = \arg\max_i h(x)_i$ and use $g = \mathrm{softmax}(h(x))_{i^*}$ as the gate value:

$$
y \;=\; g \cdot E_{i^*}(x).
$$

Counter to received wisdom (Shazeer et al. argued $k\!\geq\!2$ was needed for the gate to receive useful gradient), top-1 trains *fine* — and is faster, simpler, and uses half the bandwidth.

**Single load-balancing loss** (eq. 4):

$$
\mathcal{L}_{\text{aux}} \;=\; \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i,
$$

where
- $f_i = \tfrac{1}{T}\sum_{t=1}^{T}\mathbb{1}[i^*_t = i]$ — fraction of tokens routed to expert $i$,
- $P_i = \tfrac{1}{T}\sum_{t=1}^{T}\mathrm{softmax}(h(x_t))_i$ — mean gate probability for expert $i$.

Under uniform routing both $f_i$ and $P_i$ equal $1/N$, so $\mathcal{L}_{\text{aux}}=\alpha$. Penalizing the dot product $\sum f_i P_i$ pushes both quantities to be uniform. Default $\alpha = 10^{-2}$.

**Capacity factor $C$.** Each expert is given a fixed buffer of $\big\lceil C\cdot T/N\big\rceil$ slots; tokens beyond that *overflow* and skip the expert via residual. The paper shows top-1 lets you run smaller $C$ than top-2 at the same quality (default $C=1.25$ for training, $C=2.0$ optional).

## How it works

**Stability and precision tricks (the practical contributions):**

- **Selective fp32 router.** Cast only the router logits + softmax to fp32 inside an otherwise bf16 model. Removes the previously-mandatory full-fp32 training. Cuts memory and bandwidth by ~half.
- **Smaller expert-init.** Initialize gate weights with the standard Transformer init scaled down by 10× — prevents early router collapse.
- **Expert dropout.** Apply higher dropout (0.4) inside expert FFN layers than in the rest of the network (0.1) during fine-tuning to mitigate overfitting in down-stream SFT.
- **Pre-LN + RMSNorm** as in T5; no other architectural changes.

**Distillation back to dense (§7).** A trained sparse Switch model is distilled into a dense student of the same size as the dense baseline. Switch-Base 7.4 B → dense 223 M retains **30 %** of the quality gain over the dense T5-Base trained from scratch — a route to ship sparse-trained quality on dense-only inference hardware.

**Configurations.** Three production models:

| Model | $N$ experts | Total params | FLOPs / token | vs. T5 dense |
|---|---|---|---|---|
| Switch-Base | 128 | 7.4 B | matches T5-Base | 7.5× speedup to T5-Base PPL (per §5.2) |
| Switch-Large | 128 | 26.3 B | matches T5-Large | 2.5× speedup vs. T5-Large |
| Switch-XXL | 64 | 395 B | matches T5-XXL | 4× speedup vs. T5-XXL |
| **Switch-C** | **2 048** | **1.571 T** | matches T5-Base | trains stably in bf16 |

## Training / data
- **Pre-training.** Span-corruption objective on the **C4** corpus (Raffel et al. 2020), same as T5.
- **Hardware.** Up to 1 024 TPU v3 cores; Switch-C trains on 2 048 TPU v3.
- **Optimizer.** Adafactor; LR schedule matches T5.
- **Aux-loss weight.** $\alpha = 10^{-2}$.
- **Capacity factor.** $C = 1.25$ during pre-training.

## Results

**Pre-training speedup at fixed compute (per §5.2, Figure 4).** Steps to reach T5-Base / T5-Large quality on C4 negative log-perplexity:

| Comparison | Speedup |
|---|---|
| Switch-Base vs. T5-Base | **7.5×** |
| Switch-Large vs. T5-Large | **2.5×** |
| Switch-XXL vs. T5-XXL | **4×** (per §5) |

**Multilingual pre-training (per §6, Figure 6).** Switch-Base on **mC4 / 101 languages** beats dense mT5-Base on **all 101 languages**, with average **5×** speedup to a fixed quality threshold. Strongest gains on the lowest-resource languages.

**Downstream fine-tuning (per Table 6).** Switch-Base fine-tuned on SuperGLUE matches T5-Base (avg. 76.4 vs. 73.7 — Switch wins despite identical FLOPs/token), and Switch-Large beats T5-Large (avg. 87.5 vs. 87.0).

**Switch-C 1.6 T (per §5.6).** Trains stably in bf16, achieves the same C4 quality as T5-XXL in **4× fewer steps**.

**Distillation (per Table 9).** Sparse → dense distillation recovers ~30 % of the quality gap on Switch-Base → 223 M dense.

## Limitations & follow-ups
- **Token dropping at $C\!\approx\!1$** still costs a small but measurable amount of quality. Addressed by Expert-Choice Routing (Zhou et al. 2022, arXiv:2202.09368) which makes experts pick their top-$k$ tokens (no overflow possible).
- **Top-1 vs. fine-grained top-many.** [DeepSeek-MoE](moe_2024_deepseek-moe.md) shows that splitting experts more finely and routing top-many gives more combinatorial diversity at fixed FLOPs — Switch's top-1 leaves combinatorial richness on the table.
- **Auxiliary loss tuning is fragile.** Auxiliary-loss-free balancing (Wang et al. 2024, arXiv:2408.15664), used in DeepSeek-V3, replaces $\mathcal{L}_{\text{aux}}$ with bias-adjusted routing.
- **Dense quality at ≥1 T params is still better per FLOP** — Chinchilla-optimal dense LLMs (Hoffmann et al. 2022) showed that the headline "1.6 T" is partly a marketing artifact when activated params is what matters at inference.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2101.03961) · [pdf](https://arxiv.org/pdf/2101.03961)
- **JMLR:** <https://jmlr.org/papers/v23/21-0998.html>
- **Code:** T5X (<https://github.com/google-research/t5x>), Mesh-TensorFlow Switch (<https://github.com/tensorflow/mesh>)
- **BibTeX:** [DBLP](https://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-2101-03961)
- **Related / successor papers:** [Sparsely-Gated MoE (Shazeer et al. 2017)](moe_2017_sparsely-gated-moe.md) · [GShard (Lepikhin et al. 2020)](moe_2020_gshard.md) · [DeepSeek-MoE (Dai et al. 2024)](moe_2024_deepseek-moe.md) · ST-MoE (Zoph et al. 2022, arXiv:2202.08906) · Expert-Choice Routing (Zhou et al. 2022, arXiv:2202.09368) · GLaM (Du et al. 2022, arXiv:2112.06905) · Mixtral 8×7B (Jiang et al. 2024, arXiv:2401.04088)
