# End-to-End Context Compression at Scale (LCLMs)

> An encoder ([Qwen3-Embedding-0.6B][Qwen3Emb]) maps each contiguous block of *N* input
> tokens to a single **latent (soft) token**; an MLP **adapter** projects those latents
> into the decoder's embedding space; a causal decoder ([Qwen3-4B-Instruct-2507][Qwen3])
> consumes them *in place of* the original context tokens and decodes normally. Trained
> end-to-end on **>350B tokens** across compression ratios **1:4, 1:8, 1:16**, this family
> — **Latent Context Language Models (LCLMs)** — sets a new accuracy/latency/memory
> Pareto frontier for long-context inference.

**Paper.** [End-to-End Context Compression at Scale][paper] (Li, McLeish, Chen et al., arXiv:2606.09659, Jun 2026).
**Code/models.** [github.com/LeonLixyz/LCLM][code] · [huggingface.co/latent-context][hf].

---

## 1. The problem

Long-context LLM inference is **memory-bound**: the [KV cache][KVCache] grows linearly with
context length, and models also [struggle to use information][LostMiddle] spread across long
inputs. Two existing remedies fall short:

| Family | Idea | Limitation |
|---|---|---|
| **KV-cache compression** ([SnapKV][SnapKV], [KVzip][KVzip], [Expected Attention][ExpAttn], [Attention Matching][AM]) | Evict / compact cache entries after a full prefill | Still requires full-context **prefill** before compressing; query-dependent caches don't reuse across turns; non-uniform per-head eviction is **incompatible with [paged-attention][PagedAttn] engines** ([vLLM][vLLM], [SGLang][SGLang]) |
| **Hard-token compression** ([LLMLingua][LLMLingua], summarization) | Delete / paraphrase input tokens | Intrinsically **lossy** — exact lexical/structural detail is unrecoverable |
| **Soft-token compression** ([ICAE][ICAE], [Gisting][Gist], [AutoCompressor][AutoComp], [CEPE][CEPE], [REFRAG][REFRAG]) | Encode tokens → short sequence of continuous embeddings | Historically **either degrades the base model substantially or needs task-specific training**; not competitive on dense long-context tasks like [RULER][RULER] |

LCLMs revisit the **encoder–decoder soft-token** route and close the gap with KV-cache
methods, while keeping three structural advantages: compression is **parallelizable**, uses
**standard decoding / inference engines**, and extends the decoder **beyond its native
context window**.

---

## 2. Architecture

### 2.1 High-level data flow

```
raw context tokens  x_1:T
  │  split into I = ⌈T/W⌉ encoder windows of size W = 1024
  ▼
window w_i  ──►  Encoder (Qwen3-Embedding-0.6B, causal mask)  ──►  hidden states h_1:|w_i|
  │
  │  Pooling: partition into blocks of N, aggregate → M_i = ⌈|w_i|/N⌉ latent tokens
  ▼
latent sequence  z_1:M  ∈ ℝ^{M × d_enc}
  │  Adapter a(·): RMSNorm → Linear → GELU → Linear   (d_enc → d_dec)
  ▼
soft tokens  s_1:M  ∈ ℝ^{M × d_dec}
  ▼
decoder input  =  [ uncompressed_toks , s_1:M (memory) , uncompressed_toks , … ]
  ▼
Decoder (Qwen3-4B-Instruct-2507, d_dec, causal, standard KV cache)
  ▼
next-token CE loss on UNCOMPRESSED tokens only
```

For a sequence $x_{1:T}$ and compression ratio $N$, the encoder maps each block of $N$ tokens
to **one** latent token. With window size $W$, the input is split into $I=\lceil T/W\rceil$
windows; each window yields $M_i=\lceil |w_i|/N\rceil$ latents, concatenated into the full
compressed sequence $z_{1:M}$, $M=\lceil T/N\rceil$.

### 2.2 Components and dimensions

| Component | Spec | Notes |
|---|---|---|
| **Encoder** | `Qwen3-Embedding-0.6B`, **causal** attention | Initialized from an *embedding* model, which beats LM init (§5); window-batched for parallel compression. |
| **Encoder window** | $W = 1024$ tokens | Larger $W$ lets the encoder contextualize more local content; $W=1024$ ≫ the $W=N$ used by prior work ([REFRAG][REFRAG], [E2LLM][E2LLM]). No boundary overlap by default. |
| **Compression ratio** | $N \in \{4, 8, 16\}$ | $16\times$ / $8\times$ remove **93.75% / 87.5%** of input tokens. |
| **Pooling** | **Mean** (default at $16\times$) or **Concat** | Mean = average hidden states per block; Concat = stack the $N$ hidden states into one $N\!\cdot\!d_{enc}$ vector then project. Both beat [EOS/CLS][CLSpool] token pooling. |
| **Adapter** | 2-layer **MLP**, [GELU][GELU] + [RMSNorm][RMSNorm] pre-norm ([LLaVA][LLaVA]-style) | Projects $d_{enc}\to d_{dec}$ **per latent token, independently** (no cross-latent mixing). MLP beats an attention-based adapter while using less compute. |
| **Decoder** | `Qwen3-4B-Instruct-2507`, causal, **standard KV cache** | Fully fine-tuned (small LR); consumes soft tokens exactly like ordinary token embeddings — fully compatible with [vLLM][vLLM] / [SGLang][SGLang]. |
| **Loss** | next-token CE on **uncompressed** tokens only | Because loss skips compressed spans, reported pre-training loss is much lower than standard NTP loss. |

### 2.3 Key design choices (from the architecture search)

A from-scratch sweep (both encoder and decoder = randomly-initialized `Qwen3-0.6B`, 38B
tokens, $N=16$) isolates each axis; findings then validated at scale:

1. **Pooling — mean ≈ concat ≫ token-based.** Mean pooling edges out concat at high ratios
   ($16\times$); concat wins at low ratios ($4\times$); the gap narrows with context length.
   Both clearly beat appending learned `EOS`/`CLS` pooling tokens.
2. **Encoding granularity — bigger window helps.** $W: N \to 256$ gives a large gain;
   $256 \to 1024$ a smaller one. Default $W=1024$. **Boundary overlap** (each window sees
   neighbor tokens) does *not* improve loss and adds compute → not used.
3. **Encoder mask — causal beats bidirectional**, consistently lower pre-training loss
   (despite [prefix-LM][PrefixLM] intuition favoring bidirectional).
4. **Adapter — MLP beats attention adapter** (lower loss, less compute), contrary to some
   prior work.
5. **Encoder init — embedding model beats LM.** `Qwen3-Embedding-0.6B` (itself initialized
   from the LLM) gives a better encoder representation; gap shrinks with longer training.
6. **Scaling — decoder ≫ encoder.** Growing the **decoder** drops pre-training loss far more
   than growing the encoder; downstream results are mixed (the 8B hybrid-thinking decoder
   underperforms the 4B-instruct one, likely a data/recipe mismatch).

---

## 3. Training recipe

### 3.1 Data — three interleaved types

- **Continual pre-training** (interleaved compressed / uncompressed blocks). Each sequence is
  split into segments; alternating spans are wrapped in `<|memory_start|>…<|memory_end|>` and
  treated as compressed; **NTP loss is computed only on uncompressed tokens**. Distributing
  compressed spans *throughout* the sequence (not just a first-half/second-half split as in
  prior work) teaches the decoder to condition on latent context at **multiple positions**.
  Sourced largely from NVIDIA Nemotron mixes (CC text, code, reasoning) + OLMo-3 long-context.
- **SFT** with compressed prompts + long documents: reasoning, long-context instruction
  following, and general/multi-turn chat. Outdated completions are **relabeled** with
  `Qwen3-30B-A3B` / `Qwen3-235B-A22B` (Apache-2.0).
- **Auxiliary reconstruction** — compress a document, ask the decoder to **repeat** it.
  Spans code/text/long-docs/math/LaTeX, with a bank of ~100 prompt templates per source to
  avoid prompt overfitting. Encourages preservation of **fine-grained detail** (exact
  retrieval), included in both pre-training and SFT.

> **Why mix NTP + reconstruction?** Reconstruction-only collapses to a copy task that
> *generalizes poorly* (a glorified prefix-tuning). NTP-only enables downstream reasoning but
> yields representations that **lack exact-string fidelity**. The mixture gives task-aligned
> signal *plus* an information-preservation prior that accelerates early training.

### 3.2 Stages — progressively unfreeze (VLM-style alignment)

| Stage | Name | Trainable | Rationale |
|---|---|---|---|
| **0** | Adapter warmup | adapter only (enc + dec frozen) | Keeps gradients smooth; the decoder isn't yet used to embedding-model outputs. |
| **1** | Encoder training | encoder (dec frozen) | Adapt the encoder representation. |
| **2** | End-to-end continual pre-training | decoder unfrozen, **small LR** | Aligns the decoder to compressed context without catastrophic forgetting. |
| **3** | SFT | full model, higher decoder LR | Instruction following / reasoning on compressed inputs. |

Training the full model end-to-end *from the start* underperforms — large encoder+decoder
gradients early on degrade the model. The staged recipe (analogous to [LLaVA][LLaVA] /
[Cambrian][Cambrian] VLM pipelines) is the fix. Fully-frozen-decoder and [LoRA][LoRA]-only
variants substantially underperform full-parameter training. Sequences are **packed** with
block-diagonal attention (reset at example boundaries) via [varlen][FlashAttn] kernels.

---

## 4. Results

Benchmarks: [RULER][RULER] (4K–16K), [LongBench][LongBench], [LongHealth][LongHealth], and
[GSM8K][GSM8K] (whole-prompt compression). Instructions stay **uncompressed**; only the
long-context segment is compressed. Efficiency measured as **TTFT** (time-to-first-token) and
**peak GPU memory** on a single H200.

### 4.1 A new Pareto frontier

- **Time/quality.** LCLMs compress *much faster* than KV-cache baselines at equal-or-higher
  accuracy. KV-cache methods appear as near-**vertical lines** — their cost is dominated by
  the full-context prefill and is largely *independent* of compression ratio; LCLMs instead
  reduce sequence length **before** decoder prefill, so higher ratios directly cut compute
  and memory.
- **Scaling with context length (4K→1M).** LCLMs have the lowest TTFT and, at long contexts,
  substantially lower peak memory. Peak memory stays nearly **flat** for the $16\times$ model
  from 128K→512K (encoder activations dominate), rising only once decoder prefill over the
  compressed latents takes over. Most KV-cache baselines OOM at 512K/1M on 141GB.

### 4.2 Accuracy (summary, [Qwen3-4B-Instruct][Qwen3] decoder, RULER avg / LongHealth / GSM8K)

| Setting | RULER 4K | LongHealth5 | GSM8K |
|---|---:|---:|---:|
| **Full KV (uncompressed)** | 94.4 | 75.8 | 93.3 |
| **LCLM $4\times$** (concat) | 92.3 | 82.5 | 89.9 |
| **LCLM $8\times$** (concat) | 87.2 | 79.5 | 87.4 |
| **LCLM $16\times$** (mean) | 75.1 | 67.5 | 81.1 |
| Best KV-cache baseline @ $16\times$ | 62.7 ([KVzip][KVzip]) | 76.3 ([SnapKV-QA][SnapKV]) | 31.4 ([AM][AM]) |

On **GSM8K** (short, information-dense, whole-prompt compression) LCLMs are *far* ahead of
KV-cache baselines at high ratios — evidence they aren't specialized only for long-context QA.

### 4.3 Agentic latent context (expand-on-demand)

Segment the input into 512-token chunks, compress each, give it an integer id, and expose one
**`EXPAND(i)`** tool that returns a chunk's raw text. The agent **skims the whole compressed
corpus**, then expands only the relevant chunk(s). On RULER needle-in-a-haystack tasks at
$16\times$, this lifts exact-match accuracy markedly over the raw LCLM (e.g. **+17–20 avg**),
sometimes matching the uncompressed model — compressed latents give **global visibility**;
expansion supplies **exact fine-grained** detail.

---

## 5. Relation to this repo's work

This paper is the **scaled, well-resourced realization** of the exact direction explored in
[MixedDecoder](../mixed_decoder/mixed_decoder.md) — and it confirms several of that doc's
diagnoses about why a naive text→embedding bridge fails:

| Axis | MixedDecoder (current) | LCLM (this paper) | Lesson |
|---|---|---|---|
| **Slots per chunk** | single `[CLS]` × linear ×4 expansion (rank-≤1 copies) | $\lceil \text{block}/N\rceil$ **distinct** latents; mean/concat pooling | More, *independent* latents ≫ one gist blown up linearly (the **capacity / addressability** fix). |
| **Compression ratio** | ~128× | **4×–16×** | The prior literature's working regime; 128× is past the capacity wall for exact recall. |
| **Decoder** | fully trainable from start | staged unfreeze, **small LR**, init-from-pretrained | Avoids the "memorize question→answer in decoder weights" cheating path; staged training prevents degradation. |
| **Objective** | task CE only | **NTP + reconstruction** interleaved | Reconstruction supplies the fine-grained-fidelity / forcing signal that bare CE lacks. |
| **Encoder mask / init** | BERT (bidirectional) | causal, **embedding-model** init | Empirically lower loss; queryable geometry. |
| **Eval** | QnA loss gap, anecdotal | RULER/LongBench/LongHealth/GSM8K + TTFT/memory | The ratio/accuracy *and* efficiency curves MixedDecoder is missing. |

**Concrete transfers** for MixedDecoder: (i) emit several latents per chunk via mean/concat
pooling instead of one `[CLS]` + linear expansion; (ii) move to a **4×–16×** operating point;
(iii) adopt the **adapter-warmup → encoder → end-to-end → SFT** staged recipe; (iv) add
**reconstruction** as a co-objective with next-token prediction; (v) interleave
compressed/uncompressed spans *throughout* the sequence; (vi) consider an **expand-on-demand**
tool for exact retrieval rather than forcing all detail through the bottleneck.

---

## 6. Limitations & open directions

- **Decoder scaling is mixed** — an 8B decoder lowers pre-training loss but not downstream
  accuracy in this recipe (data tuned for the 4B-instruct decoder; 8B is hybrid-thinking).
- **High-ratio fidelity** still trails full KV on the hardest RULER multi-hop tasks at $16\times$.
- **Future work the authors flag:** multi-granularity / **adaptive** compression by
  information density or perplexity; compressing not just static input but the model's
  **generated state** (long CoT, tool observations, agent working history); composing LCLMs
  with [Recursive Language Models][RLM] and other agentic memory frameworks.

---

<!-- Reference links -->
[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (arXiv:2606.09659)"
[code]: https://github.com/LeonLixyz/LCLM "LCLM code"
[hf]: https://huggingface.co/latent-context "LCLM models on Hugging Face"
[Qwen3]: https://arxiv.org/abs/2505.09388 "Qwen3 Technical Report"
[Qwen3Emb]: https://arxiv.org/abs/2506.05176 "Qwen3 Embedding"
[KVCache]: https://arxiv.org/abs/2401.18079 "KVQuant — KV cache scaling / memory bottleneck"
[LostMiddle]: https://arxiv.org/abs/2307.03172 "Lost in the Middle (Liu et al. 2024)"
[SnapKV]: https://arxiv.org/abs/2404.14469 "SnapKV (Li et al. 2024)"
[KVzip]: https://arxiv.org/abs/2505.23416 "KVzip (Kim et al. 2025)"
[ExpAttn]: https://arxiv.org/abs/2510.00636 "Expected Attention (Devoto et al. 2025)"
[AM]: https://arxiv.org/abs/2602.16284 "Fast KV Compaction via Attention Matching (Zweiger et al. 2026)"
[PagedAttn]: https://arxiv.org/abs/2309.06180 "PagedAttention / vLLM (Kwon et al. 2023)"
[vLLM]: https://arxiv.org/abs/2309.06180 "vLLM"
[SGLang]: https://arxiv.org/abs/2312.07104 "SGLang"
[LLMLingua]: https://arxiv.org/abs/2310.05736 "LLMLingua (Jiang et al. 2023)"
[ICAE]: https://arxiv.org/abs/2307.06945 "In-Context Autoencoder (Ge et al. 2023)"
[Gist]: https://arxiv.org/abs/2304.08467 "Learning to Compress Prompts with Gist Tokens (Mu et al. 2023)"
[AutoComp]: https://arxiv.org/abs/2305.14788 "Adapting LMs to Compress Contexts / AutoCompressor (Chevalier et al. 2023)"
[CEPE]: https://aclanthology.org/2024.acl-long.142 "CEPE — Parallel Context Encoding (Yen et al. 2024)"
[REFRAG]: https://arxiv.org/abs/2509.01092 "REFRAG (Lin et al. 2025)"
[E2LLM]: https://aclanthology.org/2025.emnlp-main "E2LLM (Liao et al. 2025)"
[RULER]: https://arxiv.org/abs/2404.06654 "RULER (Hsieh et al. 2024)"
[LongBench]: https://aclanthology.org/2024.acl-long.172 "LongBench (Bai et al. 2024)"
[LongHealth]: https://arxiv.org/abs/2401.14490 "LongHealth (Adams et al. 2025)"
[GSM8K]: https://arxiv.org/abs/2110.14168 "GSM8K (Cobbe et al. 2021)"
[GELU]: https://arxiv.org/abs/1606.08415 "GELU (Hendrycks & Gimpel 2016)"
[RMSNorm]: https://arxiv.org/abs/1910.07467 "RMSNorm (Zhang & Sennrich 2019)"
[LLaVA]: https://arxiv.org/abs/2304.08485 "LLaVA — Visual Instruction Tuning (Liu et al. 2023)"
[Cambrian]: https://arxiv.org/abs/2406.16860 "Cambrian-1 (Tong et al. 2024)"
[LoRA]: https://arxiv.org/abs/2106.09685 "LoRA (Hu et al. 2021)"
[FlashAttn]: https://arxiv.org/abs/2307.08691 "FlashAttention-2 / varlen kernels (Dao 2023)"
[PrefixLM]: https://arxiv.org/abs/1910.10683 "T5 / prefix-LM (Raffel et al. 2020)"
[CLSpool]: https://arxiv.org/abs/2305.14788 "EOS/CLS token pooling"
[RLM]: https://arxiv.org/abs/2512.24601 "Recursive Language Models (Zhang et al. 2025)"
