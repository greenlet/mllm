# Landmark Attention: Random-Access Infinite Context Length for Transformers — Mohtashami & Jaggi, 2023

> **arXiv:** 2305.16300v2 · **Venue:** NeurIPS 2023 · **Affiliation:** EPFL

## TL;DR
Insert one **landmark token** after every block of $\ell_{block}$ tokens and train attention so that the
landmark's key becomes a **representative** of its block: a high attention score to the landmark gates
access to the whole block. At inference, each query first attends to the (cheap) landmarks, retrieves the
top-$k$ blocks, and runs standard attention only over those — **retrieval happens through attention
itself**, not a separate retriever, so the model keeps full **random access** to any past token. This lets
a model trained at length 512 run at **arbitrary** inference length, cuts compute/memory by ~the block
size (50× with $\ell_{block}=50$), and — fine-tuning **LLaMA 7B** — extends usable context past **32K
tokens** with **98%** passkey-retrieval accuracy. It is the origin of the **passkey / pass-phrase
retrieval** diagnostic now standard for long-context evaluation.

## Problem & motivation
Attention's quadratic cost caps context length. Prior fixes each sacrifice something:
- **Recurrent memory** (Transformer-XL, RMT, Compressive) compresses the past → loses the ability to
  attend to *specific* distant tokens (random access).
- **Retrieval augmentation** (REALM, RAG) bolts on a *separate* retriever trained with hand-crafted
  score-reduction rules, not compatible with the model's own attention, and awkward to update on fresh
  long inputs.
- **kNN-augmented** (Memorizing Transformer, kNN-LM) interpolate a kNN prediction with local attention via
  a *tuned, input-independent* weight — the mixture ignores whether memory actually holds relevant info.

Goal: emulate full attention over an arbitrarily long context while (a) keeping random access, (b) letting
**attention itself** decide what to retrieve, and (c) decoupling train length from inference length.

## Key idea
Since attention weights sum to 1, only a few keys carry large weight even in a huge context — so retrieve
just those. Assign each block a **representative vector** such that a high score to *any* token in the
block implies a high score to the representative. That representative is the **key of a special landmark
token** appended after the block, learned end-to-end.

Training uses a **Grouped Softmax**: normal softmax applied separately within groups. For a vector
$\mathbf v$ and group index $\mathbf g$,

$$
\sigma_G(\mathbf v,\mathbf g)_x = \frac{e^{\mathbf v_x}}{\sum_{y:\,\mathbf g_y=\mathbf g_x} e^{\mathbf v_y}} .
$$

Each block's *normal* tokens form their own group; the landmarks of *other* blocks are placed in the
current token's group; the current block's own landmark is ignored (landmarks are only used by tokens in
other blocks). Final attention weight to a token in another block is its within-block softmax **multiplied
by** that block's landmark softmax:

$$
\text{AttWeight}_{i,j}=
\begin{cases}
0 & p_j=j\ (\text{$j$ is a landmark})\\
S_{i,j} & \text{same block as }i\\
S_{i,j}\cdot S_{i,p_j} & \text{other block (gated by its landmark)}
\end{cases}
$$

where $S=\text{GroupedSoftmax}(QK^\top/\sqrt{d},\,\mathbf G)$ and $p_j$ is the index of token $j$'s
landmark. Weights still sum to 1; because same-block tokens and their landmark share a softmax group, the
model must **trade off** local attention vs. retrieving another block — forcing it to retrieve only
relevant blocks.

**Inference:** feed the input in chunks of length $\ell_{local}$, keep a KV cache of past blocks; each
token scores the cached landmarks, picks the top-$k$ blocks, and runs attention over just those + local
context. Non-retrieved blocks' KV can be **offloaded to CPU/disk** (only landmarks stay in GPU memory) →
random-access "infinite" context. Position for retrieved far tokens uses **stingy position mapping** (a
pre-allocated prefix segment; works with RoPE by adding positions at retrieval time), since Transformers
can't extrapolate raw positions.

## How it works
```mermaid
flowchart TB
  subgraph Cache["KV cache of past blocks (landmarks in GPU, blocks offloadable)"]
    B1["block 1 [50 tok] + landmark_1"]
    B2["block 2 + landmark_2"]
    Bn["block m + landmark_m"]
  end
  Q["current chunk tokens (ℓ_local=250)"] -->|score vs landmarks| L["landmark scores"]
  L -->|top-k blocks (k=2..5)| SEL["retrieve blocks"]
  SEL --> ATT["Grouped-Softmax attention over retrieved blocks + local context"]
  Q --> ATT
  ATT --> OUT["output / next token"]
  classDef m fill:#eef,stroke:#88a;
  class Cache,SEL m;
```

Cost: finding blocks scales linearly but only **1 per $\ell_{block}+1$ tokens**; attention over the
$k$ retrieved blocks is **constant** regardless of total length → ~$\ell_{block}\times$ fewer ops and
memory. Retrieval granularity can be relaxed (same blocks across heads/tokens) for throughput at a small
perplexity cost. Combines naturally with FlashAttention and FAISS; an optional **Context-Miss Token (CMT)**
adds a hierarchical "do I even need to retrieve?" gate (drops ~50% of retrievals with minor ppl impact).

## Training / data
GPT-2-style 12-layer decoder (8 heads×128, $d$=1024, FFN 4096), GPT-2 tokenizer, AdamW ($\beta$=0.9/0.95,
wd 0.001), base lr 2e-3, 2% warmup + cosine to 4e-4, bf16, effective batch 128, 240K steps at
$\ell_{seq}=512$, on ≤4×A100. LM datasets: **PG-19** (3.7B tokens, books) and **arXiv math** (5.6B). For
the LLM demo, **LLaMA 7B** is fine-tuned 15K steps at context 512 on a RedPajama subset, then evaluated on
**passkey retrieval**: a random pass key (1–50000) is hidden at a random position inside long filler text
("The grass is green. The sky is blue…"), and the model must reproduce it.

## Results
Language-model perplexity (landmark model attends far fewer tokens than its effective context):

| Setting (eval len / blocks / k / attn size) | PG19 | arXiv | Source |
|---|---:|---:|---|
| Baseline, ctx 512 | 16.12 | 4.01 | Table 1 |
| Ours, eval 250, 10 blocks, k=2 (attn 360) | 16.23 | 4.01 | Table 1 |
| Ours, eval 250, 40 blocks, k=4 (attn 460) | 14.92 | 3.35 | Table 1 |
| Transformer-XL, ctx 2048, XL-cache 256 | 14.72 | — | Table 1 |
| Ours, eval 4096, k=4 (attn 470) | **14.72** | **3.18** | Table 1 |

- **Matches Transformer-XL** (trained at length 2048) while retrieving far fewer tokens, and — unlike XL —
  keeps exact random access + interpretable retrieval (you can see which blocks were used).
- **Runs beyond training length:** trained at 512, improves monotonically as eval length grows to 4096.
- **Passkey (LLaMA 7B):** base LLaMA fails once context exceeds ~2048; the landmark-fine-tuned model
  retrieves the pass key with high accuracy at all lengths, reaching **98%** at **32,070 tokens** with
  CPU-offloaded KV — i.e. GPT-4-class context length via a light fine-tune.
- Retrieval-flexibility ablation: same blocks across heads costs only ~0.23 ppl; CMT drops ~50% of
  retrieval calls for a minor ppl increase (16.28→16.43 at 57% drop rate).

## Limitations & follow-ups
- **Positional extrapolation is unsolved** — the stingy-mapping workaround means far tokens are selected by
  *semantics*, not exact position; a proper extrapolatable encoding would help (Appendix E prototypes
  positional-jump augmentation).
- Retrieval overhead (two matmuls, CPU↔GPU traffic when offloading) needs reduced flexibility / FAISS to
  stay fast at extreme lengths.
- Focused on **causal LM**; masked-LM adaptation is only sketched.
- Complementary to summarizing memory ([Compressive Transformer](longseq_2019_compressive-transformer.md),
  [Recurrent Memory Transformer](memory_2022_recurrent-memory-transformer.md)) and to exact kNN lookup
  ([Memorizing Transformers](memory_2022_memorizing-transformer.md)); the **passkey** test it introduced is
  a staple of long-context benchmarks like [RULER](benchmark_2024_ruler.md).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/2305.16300) · [html](https://arxiv.org/html/2305.16300v2) · [pdf](https://arxiv.org/pdf/2305.16300)
- **Code:** <https://github.com/epfml/landmark-attention>
- **BibTeX:**
  ```bibtex
  @inproceedings{mohtashami2023landmark,
    title     = {Random-Access Infinite Context Length for Transformers},
    author    = {Mohtashami, Amirkeivan and Jaggi, Martin},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2023},
    url       = {https://arxiv.org/abs/2305.16300}
  }
  ```
- **Related papers:** [Memorizing Transformer](memory_2022_memorizing-transformer.md) ·
  [Recurrent Memory Transformer](memory_2022_recurrent-memory-transformer.md) ·
  [Compressive Transformer](longseq_2019_compressive-transformer.md) ·
  [RULER](benchmark_2024_ruler.md)
- **In-repo:** [§6.7 in mixed_decoder](../mixed_decoder/mixed_decoder.md)
