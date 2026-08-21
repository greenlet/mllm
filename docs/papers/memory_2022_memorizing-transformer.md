# Memorizing Transformers — Wu, Rabe, Hutchins & Szegedy, 2022

> **arXiv:** 2203.08913v1 · **Venue:** ICLR 2022 (spotlight) · **Affiliation:** Google Research

## TL;DR
Give a decoder-only Transformer a **large non-differentiable external memory** of past (key, value)
pairs, and let **one** attention layer near the top of the stack do an approximate **$k$-nearest-neighbor
(kNN) lookup** into it. Because gradients are *not* backpropagated into the memory, the (key, value)
records computed on earlier steps can be reused as-is, so memory scales to **65K–262K tokens** on a single
device at almost no extra step-time. The kNN result is blended with ordinary local attention through a
**learned per-head gate**. Adding an 8K-token memory beats growing a vanilla Transformer by **5×** in
parameters, gains hold from 200M→8B params, and — critically — a *pretrained* model can be **finetuned**
to use memory in ~4% of pretraining steps.

## Problem & motivation
Transformer quality on long documents (books, code repos, math papers, formal proofs) is throttled by the
short attention context. The important long-range references — a character introduced 20 chapters ago, a
function defined in another file, a lemma proven earlier — lie far outside the window. Prior long-range
methods either **summarize/average** distant tokens (pooling, Compressive Transformer) or **approximate**
the softmax (Performer, Linformer), both of which blur exact content. Two observations motivate a
different route:

1. Attention over far-away tokens is really a form of **information retrieval** — a fact stored as a
   (key, value) pair can be *looked up* later instead of slowly baked into weights.
2. If the external memory is **non-differentiable**, you never have to recompute all its keys/values on
   every step, so it can be enormous and cheap.

## Key idea
Keep, per head, a FIFO cache of the last $M$ (key, value) pairs seen in the current document (the
**external memory**). At the kNN-augmented layer, the *same* queries attend to both the **local context**
(standard dense self-attention) and the **top-$k$ retrieved memories**:

- kNN retrieval returns, for each query token, its top-$k$ nearest keys in memory; attention is the usual
  softmax over dot products with those retrieved keys, weighting the retrieved values → $V_m$.
- Local self-attention produces $V_c$.
- The two are combined by a **learned gate** (per-head scalar bias, position-independent):

$$
g=\sigma(b_g),\qquad V_a = V_m\odot g + V_c\odot(1-g).
$$

$\sigma$ is the sigmoid, $\odot$ element-wise, $b_g$ a learned per-head parameter that lets each head
choose local vs. long-range. In practice most heads learn to attend **almost exclusively to memory**.
Unlike dense attention, the retrieved (key, value) set is **different for every query**.

Two design points that make it work at scale:

- **No gradient into memory.** Keys/values are functions of the model parameters, but the stored records
  are treated as constants; this is what lets old records be reused across steps.
- **Distributional shift / staleness.** As parameters drift, old keys become "stale". They mitigate this
  by **query/key normalization** (so old and new keys share magnitude), and for very large memories they
  **pretrain small (8K) then finetune large (131K/262K)**.

Positioning: local context uses **T5 relative position bias**; retrieved memories get **no position
bias** (relative position is meaningless at long range, and T5 buckets all far tokens together anyway).

## How it works
```mermaid
flowchart TB
  subgraph Doc["Long document, fed sequentially (no shuffling)"]
    direction LR
    S1["subseq t-1 (512 tok)"] --> S2["subseq t (512 tok)"] --> S3["subseq t+1"]
  end
  S2 --> EMB["token embeddings"]
  EMB --> L1["dense self-attn layers (+ XL cache)"]
  L1 --> KNN["kNN-augmented layer (e.g. layer 9)"]
  KNN -->|query| MEM["external memory: last M (k,v) pairs (FIFO, non-diff)"]
  MEM -->|top-k retrieved (k,v)| KNN
  KNN -->|"gate g=σ(b_g): V_a = g·V_m + (1-g)·V_c"| L2["remaining layers"]
  L2 --> OUT["next-token prediction (causal)"]
  KNN -. append current (k,v) after step .-> MEM
  classDef m fill:#eef,stroke:#88a;
  class MEM m;
```

Mechanics per training step: process one 512-token subsequence *in document order* (not shuffled, like
Transformer-XL); an **XL-style cache** of the previous step's (k,v) provides a 512-token local window;
after the step, append the current (k,v) to external memory (dropping oldest if full). Each document in a
batch owns a **separate memory**, cleared at document boundaries.

- **kNN layer placement:** middle of the stack is best. In a 12-layer model, layer index 3/6/9/12 gives
  ppl 2.40 / 2.36 / 2.37 / 2.43 — layers too close to input or output help less. Multiple kNN layers give
  no further gain.
- **Neighbors $k$:** $k=32$ already matches $k=128$ or $256$ (2.38 vs 2.37). Approximate kNN (~90% recall)
  is as good as exact — the model is robust to retrieval quality.
- **Cost:** for a ~200M model, step time 0.2s (no mem) → 0.25s (8K) → 0.6s (65K) on TPUv3.

## Training / data
12-layer decoder-only Transformer, $d=1024$, 8 heads of dim 128, FFN 4096, SentencePiece vocab 32K,
$k=32$, kNN at layer 9. Adafactor, lr 1.0, 1000-step linear warmup then rsqrt decay, 500K steps (100K for
the small Isabelle set), 32 TPU cores, JAX/Flax. Batch size adjusted to keep $2^{17}$ tokens/batch across
context lengths. Five long-document datasets: **arXiv Math** (LaTeX source), **GitHub** (whole repo
concatenated per doc), **Isabelle** (formal proofs, 684 theories), **C4(4K+)** (web docs ≥4096 tokens),
**PG-19** (books). Scaling runs go to **1B** (8 layers, 32 heads, $d$=2048, $d_{ff}$=16384) and **8B**
(16 layers, 64 heads, $d$=4096, $d_{ff}$=32768).

## Results
Average token-level perplexity (bits-per-token for code/proofs); lower is better.

| Setting (context / memory / XL-cache) | arXiv | PG19 | C4(4K+) | GitHub | Isabelle | Source |
|---|---:|---:|---:|---:|---:|---|
| 512 / — / — (vanilla) | 3.29 | 13.71 | 17.20 | 3.05 | 3.09 | Table 4 |
| 2048 / — / — (long context) | 2.69 | 12.37 | 14.81 | 2.22 | 2.39 | Table 4 |
| 512 / — / 512 (Transformer-XL) | 2.67 | 12.34 | 15.38 | 2.26 | 2.46 | Table 4 |
| 512 / 8192 / 512 (**+memory**) | 2.37 | 11.93 | 14.04 | 2.03 | 2.08 | Table 4 |
| 512 / 65K / 512 | 2.31 | 11.62 | 14.04 | 1.87 | 2.06 | Table 4 |
| 2048 / 65K / 2048 (**best**) | **2.26** | **11.37** | **13.64** | **1.80** | **1.99** | Table 4 |

- **A small memory ≈ a much bigger model.** An 8K-token Memorizing Transformer matches a vanilla model
  with **5× more parameters** (Fig. 1), and the benefit persists from 200M→8B.
- **Bigger memory keeps helping** up to a point of diminishing returns; on arXiv, finetuning to 262K
  improves ppl monotonically (2.37→2.26 at ctx 512; 2.33→2.21 at ctx 2048, Table 5). 262K exceeds almost
  all arXiv doc lengths, so no gain is expected beyond.
- **Cheap retrofit.** Finetuning a *non-memory* pretrained 1B model closes **85%** of the gap to a
  from-scratch Memorizing Transformer within 20K steps (~4% of pretraining), and the full gap by 100K.
- **It really retrieves definitions.** On Isabelle, when predicting a lemma name, the model looks up the
  lemma's *body* in **8/10** manually checked cases; on arXiv/GitHub it retrieves citation keys and
  function/variable names — the gains are sparse and concentrated on rare tokens (proper names,
  references). Robustness: Transformer-XL 2.67±0.01 vs Memorizing 2.37±0.005 over 3 seeds.

## Limitations & follow-ups
- **Staleness** of stored keys can destabilize training with very large memories from scratch (hence the
  small→large finetuning recipe and query/key norm).
- The gate $g$ here is **content-independent** (a learned per-head scalar); a content-dependent gate is a
  trivial but unexplored extension.
- Retrieval is **exact-value lookup**, not summarization — good for precise facts (citations, function
  bodies) but complementary to recurrence, which captures diffuse style. Block-Recurrent Transformers
  ([memory_2022_block-recurrent-transformer.md](memory_2022_block-recurrent-transformer.md)) find that
  a recurrent cell reaches near-identical PG19/arXiv perplexity while training ~2× faster.
- Ethics: memory can be **cleared** of sensitive/copyrighted content per-document, unlike facts baked into
  weights.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2203.08913) · [html](https://arxiv.org/html/2203.08913v1) · [pdf](https://arxiv.org/pdf/2203.08913)
- **Code:** part of Google Research **Meliad** — <https://github.com/google-research/meliad>
- **BibTeX:**
  ```bibtex
  @inproceedings{wu2022memorizing,
    title     = {Memorizing Transformers},
    author    = {Wu, Yuhuai and Rabe, Markus N. and Hutchins, DeLesley and Szegedy, Christian},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year      = {2022},
    url       = {https://arxiv.org/abs/2203.08913}
  }
  ```
- **Related papers:** [Compressive Transformer](longseq_2019_compressive-transformer.md) ·
  [Transformer-XL](longseq_2019_transformer-xl.md) ·
  [Recurrent Memory Transformer](memory_2022_recurrent-memory-transformer.md) ·
  [Block-Recurrent Transformer](memory_2022_block-recurrent-transformer.md)
- **In-repo:** [§6.5 in mixed_decoder](../mixed_decoder/mixed_decoder.md) ·
  retrieval-augmented siblings: [DPR](retrieval_2020_dpr.md), [ColBERTv2](retrieval_2021_colbertv2.md)
