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

## 7. References

**In-repo threads:** [MixedDecoder](../mixed_decoder/mixed_decoder.md) · [Qwen overview](../qwen/overview.md) · [Multimodal / VLM alignment](multimodal/multimodal.md) · [Multimodal soft-token bridges](../mixed_decoder/multimodal/multimodal.md) · [Soft-token compression](soft_token/soft_token.md) · [KV-cache compression](kv_cache/kv_cache.md) · [Hard-token compression](hard_token/hard_token.md) · [Backbone components](backbone/backbone.md) · [Long-context benchmarks](benchmarks/benchmarks.md) · [Inference engines & systems](systems/systems.md) · [Efficient long-sequence modeling](long_seq/long_seq.md)

### 7.1 This paper — LCLM
- [Li et al., *End-to-End Context Compression at Scale (LCLM)* (2026)][paper] — [code][code] · [models][hf]

### 7.2 KV-cache compression (baselines)

**Thread:** [KV-cache compression](kv_cache/kv_cache.md) (KVQuant · SnapKV · Cartridges · Expected Attention · KVzip · Fast KVzip · Attention Matching)

- [Li et al., *SnapKV: LLM Knows What You Are Looking For Before Generation* (2024)][SnapKV]
- [Kim et al., *KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction* (2025)][KVzip]
- [Kim et al., *Fast KVzip: Efficient LLM Inference with Gated KV Eviction* (2026)][FastKVzip]
- [Devoto et al., *Expected Attention: KV Cache Compression from Future Queries* (2025)][ExpAttn]
- [Zweiger et al., *Fast KV Compaction via Attention Matching* (2026)][AM]
- [Hooper et al., *KVQuant: 10M-Context Inference via KV Cache Quantization* (2024)][KVCache]
- [Eyuboglu et al., *Cartridges: Long-Context Representations via Self-Study* (2025)][Cartridges]
- [Synk et al., *Exploiting Sparsity for Long-Context Inference* (2025)][SparseLC]

### 7.3 Soft-token / encoder–decoder context compression

**Thread:** [Soft-token / encoder–decoder context compression](soft_token/soft_token.md) (Gist · ICAE · AutoCompressor · CEPE · xRAG · REFRAG · E2LLM · LCLM)

- [Mu et al., *Learning to Compress Prompts with Gist Tokens* (2023)][Gist]
- [Ge et al., *In-context Autoencoder for Context Compression (ICAE)* (2023)][ICAE]
- [Chevalier et al., *Adapting LMs to Compress Contexts (AutoCompressor)* (2023)][AutoComp]
- [Cheng et al., *xRAG: Extreme Context Compression for RAG with One Token* (2024)][xRAG]
- [Yen et al., *Long-Context LM with Parallel Context Encoding (CEPE)* (2024)][CEPE]
- [Lin et al., *REFRAG: Rethinking RAG-based Decoding* (2025)][REFRAG]
- [Liao et al., *E2LLM: Encoder Elongated Large Language Models* (2025)][E2LLM]
- [Dai et al., *Pretraining Context Compressor with Embedding-Based Memory* (2025)][PCC]
- [Li et al., *500xCompressor: Generalized Prompt Compression* (2025)][C500x]
- [Tang et al., *GMSA: Group Merging & Layer Semantic Alignment* (2025)][GMSA]
- [Feldman & Artzi, *Simple Context Compression: Mean-Pooling & Multi-Ratio Training* (2025)][SimpleCC]
- [Pilchen et al., *ARC-Encoder: Compressed Text Representations for LLMs* (2025)][ARCEnc]
- [Tan et al., *LLoCO: Learning Long Contexts Offline* (2024)][LLoCO]
- [Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts* (2021)][PrefixTuning]

### 7.4 Hard-token / prompt compression

**Thread:** [Hard-token / prompt compression](hard_token/hard_token.md) (Selective Context · LLMLingua · LongLLMLingua · NL-Prompt · CompAct)

- [Jiang et al., *LLMLingua: Compressing Prompts for Accelerated Inference* (2023)][LLMLingua]
- [Jiang et al., *LongLLMLingua: Prompt Compression for Long-Context Scenarios* (2024)][LongLLMLingua]
- [Li et al., *Compressing Context to Enhance Inference Efficiency (Selective Context)* (2023)][SelectiveCtx]
- [Chuang et al., *Learning to Compress Prompt in Natural Language Formats* (2024)][NLPrompt]
- [Yoon et al., *CompAct: Compressing Retrieved Documents Actively* (2024)][CompAct]

### 7.5 Backbone models & components

**Thread:** [Backbone models & components](backbone/backbone.md) (Qwen3 · Qwen3-Embedding · prefix-LM · RoPE · YaRN · RMSNorm · GELU · FlashAttention-2 · FlexAttention)

- [Yang et al., *Qwen3 Technical Report* (2025)][Qwen3]
- [Zhang et al., *Qwen3 Embedding* (2025)][Qwen3Emb]
- [Raffel et al., *T5 — Unified Text-to-Text Transformer (prefix-LM)* (2020)][PrefixLM]
- [Peng et al., *YaRN: Efficient Context Window Extension* (2023)][YaRN]
- [Su et al., *RoFormer: Rotary Position Embedding (RoPE)* (2021)][RoPE]
- [Zhang & Sennrich, *Root Mean Square Layer Normalization (RMSNorm)* (2019)][RMSNorm]
- [Hendrycks & Gimpel, *Gaussian Error Linear Units (GELU)* (2016)][GELU]
- [Dao, *FlashAttention-2* (2023)][FlashAttn]
- [Dong et al., *FlexAttention* (2024)][FlexAttn]

### 7.6 Multimodal / VLM alignment (analogy for the staged recipe)

**Thread:** [Multimodal / VLM alignment](multimodal/multimodal.md) (ViT · CPVT · LLaVA · LLaVA-1.5 · Cambrian-1 · Qwen3-VL) — the staged-recipe analogy behind §3.2's progressive unfreeze.

- [Liu et al., *Visual Instruction Tuning (LLaVA)* (2023)][LLaVA]
- [Tong et al., *Cambrian-1: Vision-Centric Multimodal LLMs* (2024)][Cambrian]
- [Dosovitskiy et al., *An Image Is Worth 16×16 Words (ViT)* (2020)][ViT]
- [Chu et al., *Conditional Positional Encodings for Vision Transformers* (2021)][CPVT]
- [Qwen Team, *Qwen3-VL* (2025)][Qwen3VL]

### 7.7 Long-context benchmarks & datasets

**Thread:** [Long-context benchmarks & datasets](benchmarks/benchmarks.md) (RULER · LongBench · LongHealth · GSM8K · Lost-in-the-Middle · Effective-Context · Train-LC) — the evaluation bar behind §4's results.

- [Hsieh et al., *RULER: Real Context Size of Long-Context LMs* (2024)][RULER]
- [Bai et al., *LongBench: Bilingual Multitask Long-Context Benchmark* (2024)][LongBench]
- [Adams et al., *LongHealth: QA over Long Clinical Documents* (2025)][LongHealth]
- [Cobbe et al., *Training Verifiers to Solve Math Word Problems (GSM8K)* (2021)][GSM8K]
- [Liu et al., *Lost in the Middle: How LMs Use Long Contexts* (2024)][LostMiddle]
- [An et al., *Why Does the Effective Context Length of LLMs Fall Short?* (2025)][EffCtx]
- [Gao et al., *How to Train Long-Context Language Models (Effectively)* (2025)][TrainLC]

### 7.8 Inference engines & systems

**Thread:** [Inference engines & systems](systems/systems.md) (vLLM / PagedAttention · SGLang · HF Transformers) — why soft tokens deploy on today's serving stacks unchanged.

- [Kwon et al., *PagedAttention / vLLM* (2023)][PagedAttn]
- [Zheng et al., *SGLang: Structured LM Program Execution* (2024)][SGLang]
- [Wolf et al., *HuggingFace Transformers* (2020)][HFTransformers]

### 7.9 Efficient long-sequence modeling

**Thread:** [Efficient long-sequence modeling](long_seq/long_seq.md) (Transformer-XL · Compressive-T · Linear Attention · S4 · Mamba · Kimi Linear · MLA · DeepSeek-V4) — the orthogonal, composable axis to input compression.

- [Dai et al., *Transformer-XL* (2019)][TransfoXL]
- [Rae et al., *Compressive Transformers* (2019)][CompTransf]
- [Katharopoulos et al., *Transformers are RNNs (Linear Attention)* (2020)][LinAttn]
- [Gu et al., *Structured State Spaces (S4)* (2021)][S4]
- [Gu & Dao, *Mamba: Selective State Spaces* (2024)][Mamba]
- [Kimi Team, *Kimi Linear: Efficient Attention Architecture* (2025)][KimiLinear]
- [Liu et al., *DeepSeek-V2 / Multi-head Latent Attention (MLA)* (2024)][MLA]
- [DeepSeek-AI, *DeepSeek-V4: Million-Token Context* (2026)][DSV4]

### 7.10 Agentic memory & frameworks
- [Zhang et al., *Recursive Language Models* (2025)][RLM]
- [Packer et al., *MemGPT: LLMs as Operating Systems* (2023)][MemGPT]
- [Xu et al., *A-Mem: Agentic Memory for LLM Agents* (2025)][AMem]
- [Zhong et al., *MemoryBank: Long-Term Memory for LLMs* (2023)][MemoryBank]

### 7.11 Continual training / catastrophic forgetting
- [Luo et al., *Catastrophic Forgetting during Continual Fine-tuning* (2025)][CatForgetLLM]
- [Li et al., *Revisiting Catastrophic Forgetting in LLM Tuning* (2024)][RevisitCF]

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[code]: https://github.com/LeonLixyz/LCLM "LCLM code"
[hf]: https://huggingface.co/latent-context "LCLM models on Hugging Face"

[SnapKV]: https://arxiv.org/abs/2404.14469 "SnapKV (Li et al. 2024)"
[KVzip]: https://arxiv.org/abs/2505.23416 "KVzip (Kim et al. 2025)"
[FastKVzip]: https://arxiv.org/abs/2601.17668 "Fast KVzip (Kim et al. 2026)"
[ExpAttn]: https://arxiv.org/abs/2510.00636 "Expected Attention (Devoto et al. 2025)"
[AM]: https://arxiv.org/abs/2602.16284 "Fast KV Compaction via Attention Matching (Zweiger et al. 2026)"
[KVCache]: https://arxiv.org/abs/2401.18079 "KVQuant (Hooper et al. 2024)"
[Cartridges]: https://arxiv.org/abs/2506.06266 "Cartridges (Eyuboglu et al. 2025)"
[SparseLC]: https://arxiv.org/abs/2502.06766 "Exploiting Sparsity for Long-Context Inference (Synk et al. 2025)"

[Gist]: https://arxiv.org/abs/2304.08467 "Gist Tokens (Mu et al. 2023)"
[ICAE]: https://arxiv.org/abs/2307.06945 "In-Context Autoencoder (Ge et al. 2023)"
[AutoComp]: https://arxiv.org/abs/2305.14788 "AutoCompressor (Chevalier et al. 2023)"
[xRAG]: https://arxiv.org/abs/2405.13792 "xRAG (Cheng et al. 2024)"
[CEPE]: https://arxiv.org/abs/2402.16617 "CEPE — Parallel Context Encoding (Yen et al. 2024)"
[REFRAG]: https://arxiv.org/abs/2509.01092 "REFRAG (Lin et al. 2025)"
[E2LLM]: https://arxiv.org/abs/2409.06679 "E2LLM (Liao et al. 2025)"
[PCC]: https://aclanthology.org/events/acl-2025/ "Pretraining Context Compressor with Embedding-Based Memory (Dai et al., ACL 2025)"
[C500x]: https://arxiv.org/abs/2408.03094 "500xCompressor (Li et al. 2025)"
[GMSA]: https://arxiv.org/abs/2505.12215 "GMSA (Tang et al. 2025)"
[SimpleCC]: https://arxiv.org/abs/2510.20797 "Simple Context Compression (Feldman & Artzi 2025)"
[ARCEnc]: https://arxiv.org/abs/2510.20535 "ARC-Encoder (Pilchen et al. 2025)"
[LLoCO]: https://arxiv.org/abs/2404.07979 "LLoCO (Tan et al. 2024)"
[PrefixTuning]: https://arxiv.org/abs/2101.00190 "Prefix-Tuning (Li & Liang 2021)"

[LLMLingua]: https://arxiv.org/abs/2310.05736 "LLMLingua (Jiang et al. 2023)"
[LongLLMLingua]: https://arxiv.org/abs/2310.06839 "LongLLMLingua (Jiang et al. 2024)"
[SelectiveCtx]: https://arxiv.org/abs/2310.06201 "Selective Context (Li et al. 2023)"
[NLPrompt]: https://arxiv.org/abs/2402.18700 "Learning to Compress Prompt in Natural Language (Chuang et al. 2024)"
[CompAct]: https://arxiv.org/abs/2407.09014 "CompAct (Yoon et al. 2024)"

[Qwen3]: https://arxiv.org/abs/2505.09388 "Qwen3 Technical Report (2025)"
[Qwen3Emb]: https://arxiv.org/abs/2506.05176 "Qwen3 Embedding (2025)"
[PrefixLM]: https://arxiv.org/abs/1910.10683 "T5 / prefix-LM (Raffel et al. 2020)"
[YaRN]: https://arxiv.org/abs/2309.00071 "YaRN (Peng et al. 2023)"
[RoPE]: https://arxiv.org/abs/2104.09864 "RoFormer / RoPE (Su et al. 2021)"
[RMSNorm]: https://arxiv.org/abs/1910.07467 "RMSNorm (Zhang & Sennrich 2019)"
[GELU]: https://arxiv.org/abs/1606.08415 "GELU (Hendrycks & Gimpel 2016)"
[FlashAttn]: https://arxiv.org/abs/2307.08691 "FlashAttention-2 (Dao 2023)"
[FlexAttn]: https://arxiv.org/abs/2412.05496 "FlexAttention (Dong et al. 2024)"

[LLaVA]: https://arxiv.org/abs/2304.08485 "LLaVA (Liu et al. 2023)"
[Cambrian]: https://arxiv.org/abs/2406.16860 "Cambrian-1 (Tong et al. 2024)"
[ViT]: https://arxiv.org/abs/2010.11929 "ViT (Dosovitskiy et al. 2020)"
[CPVT]: https://arxiv.org/abs/2102.10882 "Conditional Positional Encodings for ViT (Chu et al. 2021)"
[Qwen3VL]: https://github.com/QwenLM/Qwen3-VL "Qwen3-VL (2025)"

[RULER]: https://arxiv.org/abs/2404.06654 "RULER (Hsieh et al. 2024)"
[LongBench]: https://arxiv.org/abs/2308.14508 "LongBench (Bai et al. 2024)"
[LongHealth]: https://arxiv.org/abs/2401.14490 "LongHealth (Adams et al. 2025)"
[GSM8K]: https://arxiv.org/abs/2110.14168 "GSM8K (Cobbe et al. 2021)"
[LostMiddle]: https://arxiv.org/abs/2307.03172 "Lost in the Middle (Liu et al. 2024)"
[EffCtx]: https://arxiv.org/abs/2410.18745 "Why Does the Effective Context Length Fall Short? (An et al. 2025)"
[TrainLC]: https://arxiv.org/abs/2410.02660 "How to Train Long-Context LMs Effectively (Gao et al. 2025)"

[PagedAttn]: https://arxiv.org/abs/2309.06180 "PagedAttention / vLLM (Kwon et al. 2023)"
[vLLM]: https://arxiv.org/abs/2309.06180 "vLLM (Kwon et al. 2023)"
[SGLang]: https://arxiv.org/abs/2312.07104 "SGLang (Zheng et al. 2024)"
[HFTransformers]: https://aclanthology.org/2020.emnlp-demos.6 "HuggingFace Transformers (Wolf et al. 2020)"

[TransfoXL]: https://arxiv.org/abs/1901.02860 "Transformer-XL (Dai et al. 2019)"
[CompTransf]: https://arxiv.org/abs/1911.05507 "Compressive Transformers (Rae et al. 2019)"
[LinAttn]: https://arxiv.org/abs/2006.16236 "Linear Transformers (Katharopoulos et al. 2020)"
[S4]: https://arxiv.org/abs/2111.00396 "S4 (Gu et al. 2021)"
[Mamba]: https://arxiv.org/abs/2312.00752 "Mamba (Gu & Dao 2024)"
[KimiLinear]: https://arxiv.org/abs/2510.26692 "Kimi Linear (2025)"
[MLA]: https://arxiv.org/abs/2405.04434 "DeepSeek-V2 / MLA (Liu et al. 2024)"
[DSV4]: https://github.com/deepseek-ai "DeepSeek-V4 (2026)"

[RLM]: https://arxiv.org/abs/2512.24601 "Recursive Language Models (Zhang et al. 2025)"
[MemGPT]: https://arxiv.org/abs/2310.08560 "MemGPT (Packer et al. 2023)"
[AMem]: https://arxiv.org/abs/2502.12110 "A-Mem (Xu et al. 2025)"
[MemoryBank]: https://arxiv.org/abs/2305.10250 "MemoryBank (Zhong et al. 2023)"

[CatForgetLLM]: https://arxiv.org/abs/2308.08747 "Catastrophic Forgetting during Continual Fine-tuning (Luo et al. 2025)"
[RevisitCF]: https://arxiv.org/abs/2406.04836 "Revisiting Catastrophic Forgetting in LLM Tuning (Li et al. 2024)"
[LoRA]: https://arxiv.org/abs/2106.09685 "LoRA (Hu et al. 2021)"
[CLSpool]: https://arxiv.org/abs/2305.14788 "EOS/CLS token pooling"
