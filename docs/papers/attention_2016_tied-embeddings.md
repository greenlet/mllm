# Using the Output Embedding to Improve Language Models — Press & Wolf, 2016

> **arXiv:** 1608.05859v3 · **Venue:** EACL 2017 (short paper) · **Affiliation:** Tel Aviv University

## TL;DR
Shows that **tying** the input embedding $U$ with the output projection $V$ (so a single matrix $S$ plays both roles) cuts $\sim$25–50 % of an LM's parameters *and* lowers perplexity, by improving regularization and gradient flow to every embedding row at every step. For NMT they extend this to **three-way weight tying** of source-input, target-input, and target-output embeddings.

## Problem & motivation
Standard neural LMs and seq2seq models learn $U\in\mathbb{R}^{C\times d}$ (input lookup) and $V\in\mathbb{R}^{C\times d}$ (output projection) **independently**, even though both ought to encode similar word semantics. The pair costs $2Cd$ parameters, dominating model size for large vocabularies. Empirically, the *output* embedding alone is a stronger word-similarity model than the *input* one in RNN LMs — suggesting $U$ is undertrained.

## Key idea
Set $U = V = S$. The model's softmax becomes:

$$p_t(w \mid h_t) = \frac{\exp(S_w^\top h_t)}{\sum_{x=1}^{C}\exp(S_x^\top h_t)}.$$

For NMT add **three-way tying** (TWWT): source-encoder input, decoder input, and decoder output share one matrix — viable because BPE vocabularies overlap 80–90 % between related languages.

When the model lacks dropout, add a **projection regularizer** $h_3 = V P h_2$ with penalty $\lambda \lVert P\rVert_F^2$ ($\lambda = 0.15$) to dampen the larger effective LR on the shared matrix.

## How it works
Per-step gradient analysis explains the win:

- **Untied $U$**: only the row $U_{i_t}$ (current input word) gets a gradient.
- **Untied $V$**: *every* row gets a gradient through the softmax denominator.
- **Tied $S$**: each row receives both signals at every step, so updates resemble the (better) output-side updates while training every row continuously.

Practically, tying is a one-line change (`decoder.weight = encoder.weight`) compatible with any embedding-LSTM or Transformer.

## Training / data
- **PTB** + **Wikitext-2** for LM perplexity (LSTM small/large + Recurrent Highway Network).
- **WMT'15 EN→FR / EN→DE** for NMT (1000-d encoder, 500-d embeddings, BPE 89.5 K, Adadelta, 300 K updates).
- **Optional projection reg.**: $\lambda=0.15$.

## Results
| Setting | Params | Test PPL / BLEU | Source |
|---|---|---|---|
| PTB Large LSTM, baseline | 66 M | 78.4 PPL | Table 5 |
| PTB Large LSTM, **WT** | 51 M (−23 %) | **74.3 PPL** | Table 5 |
| PTB RHN, baseline | 32 M | 68.5 PPL | Table 6 |
| PTB RHN, **WT** | 24 M (−25 %) | **66.0 PPL** | Table 6 |
| PTB Small LSTM (no dropout) + **PR + WT** | 2.69 M | **100.9 PPL** (vs 114.5) | Table 5 |
| EN→FR NMT, baseline | 168 M | 29.49 BLEU | Table 8 |
| EN→FR NMT, **TWWT** | 80 M (−52 %) | 29.43 BLEU | Table 8 |
| EN→DE NMT, **TWWT** | 79 M (−52 %) | 21.02 BLEU (vs 20.96) | Table 8 |

Take-away: across LM and NMT, weight tying matches or improves quality at roughly half the embedding budget.

## Limitations & follow-ups
- The argument hinges on $U$ and $V$ living in the same space; for vastly larger vocabularies or factored softmaxes the analysis weakens.
- Concurrently derived from a knowledge-distillation perspective by [Inan et al. 2016](https://arxiv.org/abs/1611.01462).
- Modern LLMs (Qwen2/3 ≥ 7B, GPT-style ≥ 7B) **untie** because at $d_\text{model}\!\geq\!4096$ the parameter saving is a small fraction of total params and an independent output head measurably improves loss; tying remains the default for **small** models (Qwen2/2.5/3 ≤ 1.5 B, Phi, Gemma-2B).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/1608.05859) · [html](https://arxiv.org/html/1608.05859v3) · [pdf](https://arxiv.org/pdf/1608.05859)
- **ACL Anthology:** [E17-2025](https://aclanthology.org/E17-2025/)
- **Concurrent work:** [Inan, Khosravi, Socher — *Tying Word Vectors and Word Classifiers* (2016)](https://arxiv.org/abs/1611.01462)
- **Related / successor papers:** [Transformer](attention_2017_transformer.md)
