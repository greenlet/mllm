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

**In-repo threads:** [MixedDecoder](../mixed_decoder/mixed_decoder.md) · [Qwen overview](../qwen/overview.md) · [Multimodal soft-token bridges](../mixed_decoder/multimodal/multimodal.md)

### 7.1 This paper — LCLM
- Li, McLeish, Chen, Kalra, Chen, Gazizov, Morisetty, Kailkhura, Menon, Liu, Bartoldson, Goldstein, Lotfi, Goldblum, Izmailov. *End-to-End Context Compression at Scale.* arXiv:2606.09659, 2026. ([paper][paper] · [code][code] · [models][hf])

### 7.2 KV-cache compression (baselines)
- Li et al. *SnapKV: LLM Knows What You Are Looking For Before Generation.* NeurIPS 2024. arXiv:2404.14469. ([SnapKV][SnapKV])
- Kim et al. *KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction.* arXiv:2505.23416, 2025. ([KVzip][KVzip])
- Kim et al. *Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction.* arXiv:2601.17668, 2026.
- Devoto et al. *Expected Attention: KV Cache Compression by Estimating Attention from Future Queries.* arXiv:2510.00636, 2025. ([Expected Attention][ExpAttn])
- Zweiger et al. *Fast KV Compaction via Attention Matching.* arXiv:2602.16284, 2026. ([Attention Matching][AM])
- Hooper et al. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* NeurIPS 2024. arXiv:2401.18079. ([KV cache][KVCache])
- Eyuboglu et al. *Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study.* arXiv:2506.06266, 2025.
- Synk et al. *Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs.* arXiv:2502.06766, 2025.

### 7.3 Soft-token / encoder–decoder context compression
- Ge et al. *In-context Autoencoder for Context Compression in a Large Language Model (ICAE).* arXiv:2307.06945, 2023. ([ICAE][ICAE])
- Chevalier et al. *Adapting Language Models to Compress Contexts (AutoCompressor).* arXiv:2305.14788, 2023. ([AutoCompressor][AutoComp])
- Mu, Li, Goodman. *Learning to Compress Prompts with Gist Tokens.* NeurIPS 2023. arXiv:2304.08467. ([Gisting][Gist])
- Cheng et al. *xRAG: Extreme Context Compression for RAG with One Token.* NeurIPS 2024.
- Yen et al. *Long-Context Language Modeling with Parallel Context Encoding (CEPE).* ACL 2024. ([CEPE][CEPE])
- Lin et al. *REFRAG: Rethinking RAG-based Decoding.* arXiv:2509.01092, 2025. ([REFRAG][REFRAG])
- Liao et al. *E2LLM: Encoder Elongated Large Language Models for Long-Context Understanding and Reasoning.* EMNLP 2025. ([E2LLM][E2LLM])
- Dai et al. *Pretraining Context Compressor for LLMs with Embedding-Based Memory.* ACL 2025.
- Li et al. *500xCompressor: Generalized Prompt Compression for LLMs.* ACL 2025.
- Tang et al. *GMSA: Enhancing Context Compression via Group Merging and Layer Semantic Alignment.* arXiv:2505.12215, 2025.
- Feldman & Artzi. *Simple Context Compression: Mean-Pooling and Multi-Ratio Training.* arXiv:2510.20797, 2025.
- Pilchen et al. *ARC-Encoder: Learning Compressed Text Representations for Large Language Models.* arXiv:2510.20535, 2025.
- Tan et al. *LLoCO: Learning Long Contexts Offline.* EMNLP 2024.
- Li & Liang. *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* arXiv:2101.00190, 2021. ([prefix-LM][PrefixLM])

### 7.4 Hard-token / prompt compression
- Jiang et al. *LLMLingua: Compressing Prompts for Accelerated Inference of LLMs.* arXiv:2310.05736, 2023. ([LLMLingua][LLMLingua])
- Jiang et al. *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.* ACL 2024.
- Li et al. *Compressing Context to Enhance Inference Efficiency of LLMs (Selective Context).* EMNLP 2023.
- Chuang et al. *Learning to Compress Prompt in Natural Language Formats.* arXiv:2402.18700, 2024.
- Yoon et al. *CompAct: Compressing Retrieved Documents Actively for Question Answering.* arXiv:2407.09014, 2024.

### 7.5 Backbone models & components
- Yang et al. *Qwen3 Technical Report.* arXiv:2505.09388, 2025. ([Qwen3][Qwen3])
- Zhang et al. *Qwen3 Embedding: Advancing Text Embedding and Reranking through Foundation Models.* arXiv:2506.05176, 2025. ([Qwen3-Embedding][Qwen3Emb])
- Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5 / prefix-LM).* JMLR 2020. arXiv:1910.10683.
- Peng et al. *YaRN: Efficient Context Window Extension of Large Language Models.* arXiv:2309.00071, 2023.
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing 2024. arXiv:2104.09864.
- Zhang & Sennrich. *Root Mean Square Layer Normalization (RMSNorm).* NeurIPS 2019. ([RMSNorm][RMSNorm])
- Hendrycks & Gimpel. *Gaussian Error Linear Units (GELU).* arXiv:1606.08415, 2016. ([GELU][GELU])
- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691, 2023. ([varlen kernels][FlashAttn])
- Dong et al. *FlexAttention: A Programming Model for Generating Optimized Attention Kernels.* arXiv:2412.05496, 2024.

### 7.6 Multimodal / VLM alignment (analogy for the staged recipe)
- Liu et al. *Visual Instruction Tuning (LLaVA).* NeurIPS 2023. arXiv:2304.08485. ([LLaVA][LLaVA])
- Tong et al. *Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs.* NeurIPS 2024. ([Cambrian][Cambrian])
- Dosovitskiy et al. *An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT).* arXiv:2010.11929, 2020.
- Chu et al. *Conditional Positional Encodings for Vision Transformers.* arXiv:2102.10882, 2021.
- Qwen Team. *Qwen3-VL.* Technical report, 2025.

### 7.7 Long-context benchmarks & datasets
- Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* arXiv:2404.06654, 2024. ([RULER][RULER])
- Bai et al. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.* ACL 2024. ([LongBench][LongBench])
- Adams et al. *LongHealth: A Question Answering Benchmark with Long Clinical Documents.* J. Healthcare Informatics Research 2025. ([LongHealth][LongHealth])
- Cobbe et al. *Training Verifiers to Solve Math Word Problems (GSM8K).* arXiv:2110.14168, 2021. ([GSM8K][GSM8K])
- Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. ([Lost in the Middle][LostMiddle])
- An et al. *Why Does the Effective Context Length of LLMs Fall Short?* ICLR 2025.
- Gao et al. *How to Train Long-Context Language Models (Effectively).* ACL 2025.

### 7.8 Inference engines & systems
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).* SOSP 2023. arXiv:2309.06180. ([PagedAttention][PagedAttn] · [vLLM][vLLM])
- Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS 2024. ([SGLang][SGLang])
- Wolf et al. *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020 (demos).

### 7.9 Efficient long-sequence modeling
- Dai et al. *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.* ACL 2019.
- Rae et al. *Compressive Transformers for Long-Range Sequence Modelling.* arXiv:1911.05507, 2019.
- Katharopoulos et al. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020.
- Gu et al. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* arXiv:2111.00396, 2021.
- Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024.
- Team et al. *Kimi Linear: An Expressive, Efficient Attention Architecture.* arXiv:2510.26692, 2025.
- Liu et al. *DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model (Multi-head Latent Attention).* arXiv:2405.04434, 2024.
- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.* 2026.

### 7.10 Agentic memory & frameworks
- Zhang, Kraska, Khattab. *Recursive Language Models.* arXiv:2512.24601, 2025. ([RLM][RLM])
- Packer et al. *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560, 2023.
- Xu et al. *A-Mem: Agentic Memory for LLM Agents.* arXiv:2502.12110, 2025.
- Zhong et al. *MemoryBank: Enhancing Large Language Models with Long-Term Memory.* arXiv:2305.10250, 2023.

### 7.11 Continual training / catastrophic forgetting
- Luo et al. *An Empirical Study of Catastrophic Forgetting in LLMs during Continual Fine-tuning.* IEEE TASLP 2025.
- Li et al. *Revisiting Catastrophic Forgetting in Large Language Model Tuning.* EMNLP Findings 2024.

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
