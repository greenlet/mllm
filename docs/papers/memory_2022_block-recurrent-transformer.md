# Block-Recurrent Transformers — Hutchins, Schlag, Wu, Dyer & Neyshabur, 2022

> **arXiv:** 2203.07852v3 · **Venue:** NeurIPS 2022 · **Affiliation:** Google Research (Blueshift) & IDSIA

## TL;DR
Take an **ordinary Transformer layer and run it recurrently along the sequence**, on **blocks of tokens**
instead of one token at a time. The recurrent cell is "merely a transformer layer": it does self-attention
+ cross-attention in two directions — a **vertical** direction (tokens attend to states, the usual layer
stack) and a **horizontal** direction (a block of **state vectors** attends to the tokens and to itself),
with the residual connections in the horizontal direction replaced by **LSTM-style gates**. Because it
operates on $S=W=512$ state vectors in parallel, the recurrent state is orders of magnitude larger than an
LSTM's, and there are far fewer recurrent steps, so information/gradients travel over **60K+ tokens**. Cost
in FLOPs and parameters equals adding **one extra layer**, yet it beats a Transformer-XL baseline by a wide
margin **while running ~2× faster**, and sets a new SOTA word-level perplexity on PG19 (**26.5**).

## Problem & motivation
RNNs process tokens sequentially (slow, and one state vector must compress the whole past); Transformers
process in parallel and attend directly to the past, but at **quadratic** cost and with **no memory** of
tokens outside the window. The paper wants the best of both: a fixed-size recurrent state that
**summarizes** everything seen so far (RNN-like, linear complexity) but with the **capacity and
trainability** of attention. Three RNN limitations to fix: small state, sequential slowness, and the
forget-gate vanishing gradient — all addressed by making recurrence operate on **blocks** of tokens and
**blocks** of states.

## Key idea
The model is built on **sliding-window attention** over a long segment ($N=4096$), diced into blocks of
$W=512$; each block attends to itself and the previous block ($W\times 2W$ tiles), so attention is
**linear in $N$**. A Transformer-XL-style non-differentiable cache carries keys/values across segments.

A **recurrent cell** replaces one such layer. It takes $W$ token embeddings + $S$ "current state" vectors
and emits $W$ output embeddings + $S$ "next state" vectors:

- **Vertical direction** (tokens → outputs): an ordinary layer that does self-attention over tokens **and**
  cross-attention to the states, **in parallel**, concatenated then projected.
- **Horizontal direction** (states → next states): self-attention over the state vectors + cross-attention
  to the tokens, with **gates** instead of residual adds.

Keys/values are **shared** between directions: $(K_e,V_e)$ from tokens, $(K_s,V_s)$ from states; four
query sets $Q^v_e, Q^v_s, Q^h_s, Q^h_e$. Self- and cross-attention are done **in parallel** specifically so
the horizontal path has a single gate (doing them sequentially would need a worse-performing third gate).

**Gates.** Fixed gate (a learned convex combination, ≈ exponential moving average over blocks):

$$
z_t = W_z h_t + b_z,\quad g=\sigma(b_g),\quad c_{t+1}=c_t\odot g + z_t\odot(1-g).
$$

LSTM gate (input/forget, input-dependent, strictly more expressive):

$$
z_t=\tanh(W_z h_t + b_z),\quad i_t=\sigma(W_i h_t + b_i - 1),\quad f_t=\sigma(W_f h_t + b_f + 1),\quad
c_{t+1}=c_t\odot f_t + z_t\odot i_t.
$$

The $-1/+1$ biases initialize the gate to "remember". **State IDs** (learned position-embedding-like
vectors) are added to state vectors so the shared weights don't collapse all states to the same value.
Position: T5 relative bias on vertical self-attention; **no** position bias on state↔token cross-attention;
query/key normalization for stability.

Surprising empirical winner: **`fixed:skip`** — the *fixed* (non-input-dependent) gate with the MLP+gate
removed (skip config). It's the fastest and best in 3/4 datasets, suggesting the recurrent layer is mostly
doing an exponential-moving-average summary (name lookups), not complex reasoning.

## How it works
```mermaid
flowchart LR
  subgraph Cell["Block-recurrent cell (one layer)"]
    direction TB
    Tok["W token embeddings"] -->|vertical: self-attn + cross-attn to states (parallel)| Out["W output embeddings"]
    St["S current state vectors (+ state IDs)"] -->|horizontal: self-attn + cross-attn to tokens| Gate["LSTM/fixed gate (replaces residual)"]
    Tok --- St
    Gate --> NSt["S next state vectors"]
  end
  Prev["states from previous block"] --> St
  NSt -->|for-loop over blocks in segment| NextBlk["next block's current states"]
  NSt -.->|cached across training steps| Cache["state + kv cache (TBTT)"]
  classDef s fill:#eef,stroke:#88a;
  class St,NSt,Prev s;
```

Blocks in a segment are processed by a simple **for-loop**, "next states" feeding the next block's
"current states". Only **one** recurrent layer is used (layer 10 of a 12-layer model, all layers with an
XL cache); the last block's states + kv are cached as the initial state for the next training step
(truncated BPTT over book-length docs). Adding recurrence to one layer costs ≈ one extra plain layer
(13-layer Transformer-XL is the fair baseline). Theoretical receptive field becomes **infinite**;
effective context is empirically 60K+ tokens.

## Training / data
Auto-regressive LM on **PG19** (books), **arXiv** (math LaTeX), **GitHub** (repos concatenated), reported
in **bits-per-token** (log₂ ppl). Segments $N=4096$, window $W=512$, states $S=512$. Adafactor, inverse-
sqrt (later cosine) decay, lr 1.0 applied ≈0.03, dropout 0.05, 500K steps on 32 V4 TPUs (~48h). Scaling
study spans **40M→1.3B** params (24-layer recurrent for the two largest, recurrence at layers 10 & 20).
Gate init is delicate (Adafactor-aware: biases ~ N(0, 0.1), $\pm1$ on input/forget) to avoid the failure
mode where the model **ignores the recurrent state** (a non-recoverable local optimum).

## Results
Average **bits-per-token** (log₂ perplexity), lower is better. Recurrent models `Rec:gate:config` cost the
same as `Slide:13L`.

| Model | PG19 (tokens) | arXiv | GitHub | rel. step time | Source |
|---|---:|---:|---:|---:|---|
| XL:512 (Transformer-XL) | 3.62 | 1.45 | 1.21 | 0.88 | Table 1 |
| XL:2048 | 3.58 | 1.31 | 1.01 | 2.11 | Table 1 |
| Slide:13L (fair baseline) | 3.58 | 1.42 | 1.17 | 1.00 | Table 1 |
| **Rec:fixed:skip** (best) | **3.53** | 1.26–1.31 | ~1.01 | ~1.00 | Table 1 |

- **Recurrence beats width and window.** `Rec:fixed:skip` outperforms the 13-layer baseline by a wide
  margin and beats **XL:2048** (which runs **>2× slower**). Adding a plain 13th layer helps far less than
  adding recurrence.
- **Scaling (PG19 bits/token):** recurrence wins at every size 40M→1.3B; at large scale adding recurrence
  is worth **> doubling parameters** (e.g. 1.3B: 3.22 recurrent vs 3.31 for 13-layer XL).
- **New SOTA on PG19 word-level ppl:** 24-layer 1.3B model = **26.50** (vs Compressive Transformer 33.6,
  Routing Transformer 33.2, Perceiver AR 28.9), at 3.22 bits/token.
- **vs Memorizing Transformer** (fair, identical config, 64K memory): Block-Recurrent nearly matches on
  arXiv, matches on PG19, and trains **~2× faster** — the two use long-range capacity the same way
  (mostly proper-name lookups), but recurrence *summarizes* while kNN memory does *precise lookups*.
- **Qualitative:** in **17/20** random PG19 cases the recurrent model's biggest wins over Transformer-XL
  are **proper names** absent from the 512-token window; it remembers a book's title/author across
  **60K+** tokens (lost when the state is cleared per segment).

Ablations: one recurrent layer suffices (two adjacent layers don't help); state count best at ~1024
(worse at 2048); the **LSTM gate lags the fixed gate**; block-feedback (all layers cross-attend states)
helps but costs +35–40% step time.

## Limitations & follow-ups
- Best config is a **simple EMA** (`fixed:skip`) with the MLP removed — the recurrent layer isn't doing
  much "reasoning"; the authors note fully exploiting recurrence for knowledge extraction needs further
  advances.
- **Gate initialization is fragile** (a real failure mode where recurrence is ignored irrecoverably).
- Evaluated only on LM perplexity; downstream long-context tasks (summarization, QA over books, code
  completion) are left to future work.
- Sits between exact-memory and recurrence: precise-lookup counterpart is
  [Memorizing Transformers](memory_2022_memorizing-transformer.md); a lighter input/output-only recurrence
  is [Recurrent Memory Transformer](memory_2022_recurrent-memory-transformer.md); ancestor compression
  approach is the [Compressive Transformer](longseq_2019_compressive-transformer.md).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2203.07852) · [html](https://arxiv.org/html/2203.07852v3) · [pdf](https://arxiv.org/pdf/2203.07852)
- **Code:** Google Research **Meliad** — <https://github.com/google-research/meliad>
- **BibTeX:**
  ```bibtex
  @inproceedings{hutchins2022blockrecurrent,
    title     = {Block-Recurrent Transformers},
    author    = {Hutchins, DeLesley and Schlag, Imanol and Wu, Yuhuai and Dyer, Ethan and Neyshabur, Behnam},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2022},
    url       = {https://arxiv.org/abs/2203.07852}
  }
  ```
- **Related papers:** [Transformer-XL](longseq_2019_transformer-xl.md) ·
  [Compressive Transformer](longseq_2019_compressive-transformer.md) ·
  [Memorizing Transformer](memory_2022_memorizing-transformer.md) ·
  [Recurrent Memory Transformer](memory_2022_recurrent-memory-transformer.md) ·
  [Linear Attention](longseq_2020_linear-attention.md)
- **In-repo:** [§6.5 in mixed_decoder](../mixed_decoder/mixed_decoder.md)
