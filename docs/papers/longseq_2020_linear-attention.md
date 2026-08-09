# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention — Katharopoulos et al., 2020

> **arXiv:** 2006.16236v3 · **Venue:** ICML 2020 · **Affiliation:** Idiap Research Institute · EPFL

## TL;DR
Standard self-attention costs $O(N^2)$ in time and memory because it materializes the full softmax
attention matrix. Replace the softmax similarity with a **kernel feature map** $\phi(\cdot)$ and use the
**associativity of matrix products** to compute attention in $O(N)$. As a bonus, causal (autoregressive)
linear attention becomes a **recurrent** network with a constant-size state — so a Transformer *is* an RNN
at inference, giving **up to 4000× faster** generation. Achieves competitive accuracy on image generation
(MNIST, CIFAR-10) and speech recognition while being dramatically faster on long sequences.

## Problem & motivation
Self-attention over $N$ tokens computes an $N\times N$ matrix
$\mathrm{softmax}(QK^\top/\sqrt{d})V$ — $O(N^2)$ time and $O(N^2)$ memory. For high-resolution images,
long audio, or long documents this is prohibitive. Worse, **autoregressive** Transformers recompute
attention over the whole prefix at every step, making generation of an $N$-token sequence $O(N^2)$ and
memory-heavy. The paper seeks attention that is **linear in sequence length** for both training and
generation, without approximating with sparsity or hashing.

![Figure 1a: forward+backward time (ms) vs sequence length (log-log). Softmax attention (red, dashed) is quadratic and out-of-memory past 2^12; linear attention (black ×) scales linearly and runs to 2^16; LSH variants (blue) sit in between.](_assets/longseq_2020_linear-attention/complexity-time.png)

## Key idea
### Attention as a similarity-weighted average
Write a generalized attention with an arbitrary similarity function $\mathrm{sim}(\cdot,\cdot)\ge 0$:

$$
V_i' = \frac{\sum_{j=1}^{N}\mathrm{sim}(Q_i,K_j)\,V_j}{\sum_{j=1}^{N}\mathrm{sim}(Q_i,K_j)}.
$$

Softmax attention is the special case $\mathrm{sim}(q,k)=\exp(q^\top k/\sqrt{d})$. Symbols: $Q_i,K_j\in\mathbb{R}^d$
are query/key rows, $V_j\in\mathbb{R}^m$ value rows, $N$ sequence length.

### Kernelize and reassociate
Require $\mathrm{sim}$ to be a **kernel** with a feature map $\phi$:
$\mathrm{sim}(q,k)=\phi(q)^\top\phi(k)$ with $\phi(x)\in\mathbb{R}_{+}^{r}$. Substituting and pulling
$\phi(Q_i)$ out of the sums (linearity):

$$
V_i' = \frac{\phi(Q_i)^\top\sum_{j=1}^{N}\phi(K_j)V_j^\top}{\phi(Q_i)^\top\sum_{j=1}^{N}\phi(K_j)}.
\tag{Eq. 5}
$$

The key trick: $\sum_j\phi(K_j)V_j^\top\in\mathbb{R}^{r\times m}$ and $\sum_j\phi(K_j)\in\mathbb{R}^{r}$ are
computed **once**, then reused for every query. Cost drops from $O(N^2\max(d,m))$ to $O(N\,r\,m)$ —
**linear in $N$**. The chosen feature map (positive, cheap, no approximation):

$$
\phi(x) = \mathrm{elu}(x)+1,
\tag{Eq. 7}
$$

so $\phi(x)>0$ everywhere ($\mathrm{elu}(x)=x$ for $x>0$, $\alpha(e^x-1)$ for $x\le 0$); here $r=d$.

![Figure 1b: peak GPU memory (MB) vs sequence length. Legend: linear(ours)=black, softmax=red, lsh-1/4/8=blue. Softmax memory blows up quadratically and OOMs at 2^12; linear attention grows linearly and fits 2^16 tokens.](_assets/longseq_2020_linear-attention/complexity-memory.png)

## How it works
### Causal masking ⇒ a recurrent network
For autoregressive models the sums run only over $j\le i$. Define the running state as a prefix sum:

$$
S_i=\sum_{j\le i}\phi(K_j)V_j^\top\in\mathbb{R}^{r\times m},\qquad
Z_i=\sum_{j\le i}\phi(K_j)\in\mathbb{R}^{r}.
$$

Both admit an $O(1)$ update, turning attention into an RNN with a **fixed-size state** $(S_i,Z_i)$
independent of sequence length:

$$
S_i = S_{i-1}+\phi(K_i)V_i^\top,\qquad
Z_i = Z_{i-1}+\phi(K_i),\qquad
V_i' = \frac{\phi(Q_i)^\top S_i}{\phi(Q_i)^\top Z_i}.
$$

This is the paper's namesake result — **"Transformers are RNNs."** Training is still parallel (linear
attention over the whole sequence with a causal cumulative sum), but **generation** runs step-by-step with
$O(1)$ memory and $O(1)$ time per token, i.e. $O(N)$ total instead of $O(N^2)$. Gradients of the cumulative
sums are computed in linear time and constant memory (Appendix; avoids storing all $S_i$).

```mermaid
flowchart LR
  subgraph train["Training (parallel, O(N))"]
    K1["φ(K_j)V_jᵀ"] --> CS["causal cumulative sums S_i, Z_i"]
    Q1["φ(Q_i)"] --> OUT1["V_i' = φ(Q_i)ᵀS_i / φ(Q_i)ᵀZ_i"]
    CS --> OUT1
  end
  subgraph gen["Generation (recurrent, O(1)/step)"]
    ST["state (S_{i-1}, Z_{i-1})"] -->|+ φ(K_i)V_iᵀ, + φ(K_i)| ST2["(S_i, Z_i)"]
    Qi["φ(Q_i)"] --> OUTi["V_i'"]
    ST2 --> OUTi
    ST2 --> ST
  end
```

## Training / data
Standard task losses (cross-entropy / bits-per-dim / CTC-style). Feature map $\phi=\mathrm{elu}+1$; no
extra hyperparameters vs softmax attention. Evaluated on autoregressive image generation (per-pixel),
sequence classification, and speech recognition; compared head-to-head against full softmax attention and
the **Reformer** LSH attention baselines (lsh-1/4/8 hashing rounds).

## Results
| Task | Metric | Linear | Softmax | Speed | Source |
|---|---|---:|---:|---|---|
| MNIST gen | bits/dim | **0.644** | 0.621 | 142.8 img/s (**317×** softmax) | §Table 1 |
| CIFAR-10 gen | bits/dim | **3.40** | 3.47 | 17.85 img/s (**4462×** softmax) | §Table 2 |
| Speech (WSJ) | PER | **8.08** | 5.12 (full) | **3×** faster/epoch, ~half memory | §Table 3 |

- On MNIST generation, linear attention **matches** softmax quality (0.644 vs 0.621 bits/dim) while
  generating images **317×** faster; on CIFAR-10 the gap stays small (3.40 vs 3.47) at **4462×** speed.
- The lsh-$k$ (Reformer) baselines are faster than softmax but slower and less accurate than linear
  attention, and their cost grows with the number of hashing rounds.
- Autoregressive generation throughput scales **flat** with prefix length for linear attention vs steep
  quadratic growth for softmax — the source of the "up to 4000×" headline for long sequences.

## Limitations & follow-ups
- The $\mathrm{elu}+1$ feature map is a **finite, non-negative** approximation of softmax similarity;
  expressivity can lag full attention on some tasks (e.g. the WSJ PER gap), motivating richer maps such as
  **Performer** (FAVOR+ random features) and **cosFormer**.
- The linear recurrent state $S_i\in\mathbb{R}^{r\times m}$ is a fixed-capacity summary — related in spirit
  to the state-space view of [S4](longseq_2021_s4.md) and modern gated linear attention / **Mamba**.
- Complementary to memory-recurrence methods ([Transformer-XL](longseq_2019_transformer-xl.md),
  [Compressive Transformer](longseq_2019_compressive-transformer.md)): those keep full attention but cache
  past states; linear attention changes the attention operator itself. The constant-state generation view
  also underlies KV-cache-free decoding ideas relevant to inference engines in
  [systems](../context/systems.md).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2006.16236) · [html (ar5iv)](https://ar5iv.labs.arxiv.org/html/2006.16236) · [pdf](https://arxiv.org/pdf/2006.16236)
- **Code:** <https://github.com/idiap/fast-transformers> · project page <https://linear-transformers.com>
- **BibTeX:**
  ```bibtex
  @inproceedings{katharopoulos2020transformers,
    title={Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention},
    author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois},
    booktitle={Proceedings of the 37th International Conference on Machine Learning (ICML)},
    year={2020}
  }
  ```
- **Related papers:** [Transformer-XL](longseq_2019_transformer-xl.md) · [Compressive Transformer](longseq_2019_compressive-transformer.md) · [S4](longseq_2021_s4.md)
