# DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models — Dai et al., 2024

> **arXiv:** 2401.06066v1 · **Venue:** Preprint (DeepSeek-AI tech report) · **Affiliation:** DeepSeek-AI · Peking University · Tsinghua IIIS · Nanjing University

## TL;DR
DeepSeek-MoE introduces two complementary architectural changes — **fine-grained expert segmentation** (split each FFN expert into $m$ smaller ones, activate $m\cdot K$ instead of $K$) and **shared-expert isolation** (always-on experts that absorb common knowledge) — that together push expert specialization much further than GShard / Switch top-1/2. **DeepSeek-MoE 16B** matches **LLaMA2 7B** on most benchmarks at **~40 % of the FLOPs** (per Table 4); a preliminary **DeepSeek-MoE 145B** matches **DeepSeek 67B** at **28.5 %** of the FLOPs (per Table 6). The architecture is the direct template later adopted by **Qwen2-57B-A14B** and **Qwen3** MoEs.

## Problem & motivation
Standard MoE — [GShard](moe_2020_gshard.md) (top-2) and [Switch Transformer](moe_2021_switch-transformer.md) (top-1) — typically uses 8–64 wide experts. The paper identifies two ways this hurts **expert specialization**:

1. **Knowledge hybridity.** With only 8–16 experts, each expert receives a wide variety of tokens and ends up cramming heterogeneous knowledge into one FFN — hard to retrieve cleanly.
2. **Knowledge redundancy.** Different experts independently relearn the same "common" facts (function words, generic syntax) — wasted parameters.

DeepSeek-MoE's thesis: more, smaller experts (combinatorially richer routing) **plus** carve-outs for shared knowledge (less redundancy) ⇒ each routed expert gets to specialize.

## Key idea

**(1) Fine-grained expert segmentation.** Replace $N$ experts of FFN width $d_{\text{ff}}$ activating $K$, with $mN$ experts of width $d_{\text{ff}}/m$ activating $mK$. The total expert parameter count and the per-token compute are identical, but the number of possible expert-combinations explodes (per §3.1):

$$
\binom{N}{K}\;\to\;\binom{mN}{mK}.
$$

Concrete example from the paper: $N=16,\ K=2 \Rightarrow \binom{16}{2}=120$ combinations; with $m=4$, $\binom{64}{8}=4{,}426{,}165{,}368$ combinations.

**(2) Shared-expert isolation.** Out of $mN$ fine-grained experts, designate $K_s$ as **shared** — every token is routed to all $K_s$ deterministically (no gate). The top-$(mK-K_s)$ routed experts are picked by the gate from the remaining $mN-K_s$. The MoE-layer output (per eq. 9) becomes

$$
h_t = \underbrace{\sum_{i=1}^{K_s} \mathrm{FFN}_i(u_t)}_{\text{shared}} \;+\; \underbrace{\sum_{i=K_s+1}^{mN} g_{i,t}\,\mathrm{FFN}_i(u_t)}_{\text{routed top-}(mK-K_s)} \;+\; u_t.
$$

The shared experts soak up "common" knowledge, freeing routed experts to specialize. Empirically (per §4.5), disabling the shared expert in DeepSeek-MoE 2B raises Pile loss from **1.808 → 2.414** — they're irreplaceable.

**Load balancing.** Two complementary auxiliary losses:

- **Expert-level balance** (eq. 12): $\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i$ over the $N'=mN-K_s$ routed experts. Default $\alpha_1\!=\!0.01$ (2B) / $\!0.001$ (16B) / $\!0.003$ (145B).
- **Device-level balance** (eq. 15): groups experts into $D$ device-shards and balances their *aggregate* loads, allowing intra-shard imbalance. Default $\alpha_2\!=\!0.05$.

Splitting the loss this way prevents over-constraining intra-device balance from compromising quality.

## How it works

| Variant | Layers | $d_{\text{model}}$ | Heads | Shared / routed experts | Activated | Expert width (× FFN) | Total params | Active params | LR |
|---|---|---|---|---|---|---|---|---|---|
| Validation | 9 | 1280 | 10 | **1 + 63** (top-7 routed) | 1 + 7 | **0.25** | 2.0 B | 0.3 B | 1.08e-3 |
| **DeepSeek-MoE 16B** | 28 | 2048 | 16 | **2 + 64** (top-6 routed) | 2 + 6 | **0.25** | 16.4 B | 2.8 B | 4.2e-4 |
| **DeepSeek-MoE 145B** | 62 | 4096 | 32 | **4 + 128** (top-12 routed) | 4 + 12 | **0.125** | 144.6 B | 22.2 B | 3.0e-4 |

(Per Appendix A, Table 7.) All FFNs except the first layer are replaced by MoE; the first layer stays dense because its routing converges much more slowly. Sequence length 4 K, AdamW with $\beta_1\!=\!0.9$, $\beta_2\!=\!0.95$, weight decay 0.1, gradient clip 1.0. 16B trained on **2 T tokens**; 145B trained on **245 B tokens** (preliminary). Pipeline parallelism with all experts of one layer on the same device for 16B; expert+data parallelism with 4 expert shards for 145B.

## Training / data
- **Corpus.** DeepSeek-AI's bilingual (en/zh-heavy) web/code/math corpus.
- **Tokenizer.** Byte-level BPE; vocab 8 K (validation), 100 K (16B / 145B).
- **No dropout** during pre-training (sufficient data).
- **No token drop** in pre-training (16B pipeline-parallel, 145B uses device-level balance loss instead).

## Results

**Validation (2B-class, 100 B tokens; per Table 1).** Same total / activated params, same training data:

| Metric | Dense 0.2B | Hash Layer 2B | Switch 2B | GShard 2B | **DeepSeek-MoE 2B** |
|---|---|---|---|---|---|
| Pile loss | 2.060 | 1.932 | 1.881 | 1.867 | **1.808** |
| HellaSwag (0-shot acc) | 38.8 | 46.2 | 49.1 | 50.5 | **54.8** |
| ARC-challenge (0-shot acc) | 26.0 | 28.2 | 30.2 | 31.6 | **34.3** |
| TriviaQA (5-shot EM) | 4.9 | 6.5 | 8.9 | 10.2 | **16.6** |
| HumanEval (Pass@1) | 0.0 | 1.2 | 2.4 | 3.7 | **4.9** |

DeepSeek-MoE 2B matches **GShard×1.5** (i.e. GShard with 1.5× expert params and FLOPs; per Table 2) on Pile loss — confirming the architecture is ~1.5× more parameter-efficient than the GShard baseline at this scale. It also approaches the upper bound set by **Dense×16** (a dense model with 16× the FFN params, used as the strict MoE upper bound; per Table 2).

**16B vs. dense (per Tables 3 & 4; both trained on 2 T tokens).**

| Metric | DeepSeek 7B (Dense) | LLaMA2 7B | **DeepSeek-MoE 16B** |
|---|---|---|---|
| Total / activated params | 6.9 B / 6.9 B | 6.7 B / 6.7 B | 16.4 B / **2.8 B** |
| FLOPs / 4 K tokens | 183.5 T | 187.9 T | **74.4 T** (≈ 40 %) |
| Pile (BPB) | 0.75 | 0.76 | **0.74** |
| HellaSwag (0-shot) | 75.4 | 75.6 | **77.1** |
| HumanEval (Pass@1) | 26.2 | 14.6 | **26.8** |
| MBPP (Pass@1) | 39.0 | 21.8 | **39.2** |
| GSM8K (8-shot EM) | 17.4 | 15.5 | **18.8** |
| MMLU (5-shot acc) | 48.2 | 45.8 | 45.0 |
| CHID (Acc) | 89.3 | 37.9 | 89.4 |

DeepSeek-MoE 16B matches DeepSeek 7B (and beats LLaMA2 7B on the majority of tasks) at ~40 % of the FLOPs, and runs on a single 40 GB GPU.

**145B preliminary (245 B tokens; per Table 6).**

| Metric | DeepSeek 67B (Dense) | GShard 137B | **DeepSeek-MoE 145B** | DeepSeek-MoE 142B (½-act.) |
|---|---|---|---|---|
| Total / activated params | 67.4 B / 67.4 B | 136.5 B / 21.6 B | 144.6 B / **22.2 B** | 142.3 B / **12.2 B** |
| FLOPs / 4 K tokens | 2057.5 T | 572.7 T | 585.6 T (≈ **28.5 %**) | 374.6 T (≈ **18.2 %**) |
| Pile loss | 1.905 | 1.961 | **1.876** | 1.888 |
| HellaSwag | 74.8 | 72.0 | **75.8** | 74.9 |
| TriviaQA EM | 57.2 | 52.5 | **61.1** | 59.8 |

DeepSeek-MoE 145B matches dense DeepSeek 67B at ~28.5 % of compute; the half-activated 142B variant matches it at ~18.2 % — empirical evidence that fine-grained + shared experts is a real Pareto improvement over GShard.

**Specialization analysis (§4.5).** Disabling top routed experts hurts DeepSeek-MoE more than GShard×1.5 (Figure 4) — i.e., its experts are *less redundant*. Activating only **4 of 63** routed experts in DeepSeek-MoE 2B already matches GShard's Pile loss (Figure 5).

## Limitations & follow-ups
- **Multiple-choice tasks** (MMLU, CEval) under-perform a same-tokens dense baseline — the paper attributes this to small attention parameter count (~0.5 B in 16B vs. 2.5 B in DeepSeek 7B). Successor models (DeepSeek-V2/V3) add Multi-Head Latent Attention to fix this.
- **Aux losses still tuned by hand.** [Auxiliary-loss-free balancing](https://arxiv.org/abs/2408.15664) (Wang et al. 2024, deployed in DeepSeek-V3) replaces $\mathcal{L}_{\text{ExpBal}}$ with router-bias adjustment.
- **Routing collapse risk** when scaling expert count further; DeepSeek-V3 introduces *node-limited routing* and *complementary sequence-wise* losses.
- **Distillation back to dense** is not studied here (Switch had it).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2401.06066) · [html](https://arxiv.org/html/2401.06066v1) · [pdf](https://arxiv.org/pdf/2401.06066)
- **Code:** <https://github.com/deepseek-ai/DeepSeek-MoE>
- **Hugging Face:** <https://huggingface.co/deepseek-ai/deepseek-moe-16b-base>
- **BibTeX:** see arXiv abs page
- **Related / successor papers:** [Sparsely-Gated MoE (Shazeer et al. 2017)](moe_2017_sparsely-gated-moe.md) · [GShard (Lepikhin et al. 2020)](moe_2020_gshard.md) · [Switch Transformer (Fedus et al. 2021)](moe_2021_switch-transformer.md) · DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) · Aux-loss-free balancing (Wang et al. 2024, arXiv:2408.15664) · Mixtral 8×7B (Jiang et al. 2024, arXiv:2401.04088)
