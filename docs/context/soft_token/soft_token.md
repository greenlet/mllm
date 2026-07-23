# Thread: Soft-token / encoder–decoder context compression

The line of work that replaces a long span of **discrete context tokens** with a short
sequence of **continuous embeddings** ("soft tokens" / "latent tokens" / "memory slots")
fed straight into a frozen-ish decoder — the family the repo's own [MixedDecoder](../../mixed_decoder/mixed_decoder.md)
belongs to, and the one [LCLM](../ctx_compression.md) scales to >350B tokens. Unlike
[hard-token compression][LLMLingua] (delete/paraphrase real tokens) or
[KV-cache compression][SnapKV] (evict entries *after* a full prefill), soft-token methods
compress **before** decoding, are **parallelizable**, and reuse **standard inference
engines** ([vLLM][vLLM] / [SGLang][SGLang]).

## Evolution

| Paper | Year | Core contribution | Compression knob | Trained how |
|---|---|---|---|---|
| [Prefix-Tuning][PrefixTuning] | 2021 | Prepend a few learned continuous vectors to a **frozen** LM; the ancestor of "soft context". | fixed # prefix vectors (task-specific) | tune prefix only |
| [Gisting][Gist] | 2023 | Learn to compress a *prompt* into a handful of **gist tokens** via a modified attention mask; single-model, no extra encoder. | 1–10 gist tokens per prompt | instruction-tune the LM itself |
| [ICAE][ICAE] | 2023 | **In-context autoencoder**: a LoRA'd encoder emits memory slots; the frozen decoder reconstructs + answers. First clean encoder→decoder autoencoder framing. | 4× (fixed) | reconstruction + LM, encoder LoRA |
| [AutoCompressor][AutoComp] | 2023 | **Recursively** summarize long context into soft "summary vectors" carried across segments; unbounded context via accumulation. | summary vectors / segment | continued pretraining |
| [CEPE][CEPE] | 2024 | Small **parallel encoder** + cross-attention adapters injected into a frozen decoder; encode chunks in parallel, decode long. | chunked parallel encode | train encoder + cross-attn |
| [xRAG][xRAG] | 2024 | Compress a **whole retrieved document to a single token** via a projector on a frozen retriever embedding; RAG-specific extreme ratio. | 1 token / document | train projector only |
| [E2LLM][E2LLM] | 2025 | **Encoder-elongated** LLM: chunk → encoder embeddings → adapter → decoder, with an "understand + reconstruct" dual objective. | per-chunk soft tokens | dual objective |
| [REFRAG][REFRAG] | 2025 | Encode chunks to embeddings and let an RL policy decide **which chunks to expand** back to tokens; decode over the mixed sequence. TTFT ↓. | selective expand | RL policy + LM |
| [LCLM][paper] | 2026 | **End-to-end** encoder ([Qwen3-Embedding-0.6B][Qwen3Emb]) + MLP adapter + decoder ([Qwen3-4B][Qwen3]); mean/concat pooling, W=1024 window, 4×/8×/16×, >350B tokens; matches KV-cache baselines. | 4×/8×/16× (mean/concat) | 4-stage, NTP + reconstruction |

## The shared recipe

Every member instantiates the same four boxes; the design space *is* how each box is filled:

```
long context ──► [Encoder] ──► [Pool] ──► [Adapter] ──► soft tokens ──► [Decoder]
                    │             │           │                            │
        LM vs embed init   token/mean/concat  MLP vs attn         frozen / LoRA / full-FT
        causal vs bidir    fixed vs adaptive  (dim d_enc→d_dec)   std KV cache
```

- **Encoder** — a separate model (CEPE, E2LLM, LCLM) or the decoder itself with a special
  mask (Gisting, ICAE). LCLM's finding: **embedding-model init beats LM init**, and a
  **causal** encoder mask beats bidirectional.
- **Pooling / ratio** — how many soft tokens per span. Early work fixes one ratio;
  [Simple Context Compression][SimpleCC] and LCLM show **mean/concat pooling with multi-ratio
  training** beats learned `EOS`/`CLS` tokens.
- **Adapter** — a small **MLP** ([LLaVA][LLaVA]-style, GELU + [RMSNorm][RMSNorm]) reliably
  beats an attention-based connector at lower compute.
- **Decoder** — kept compatible with ordinary token embeddings so **paged-attention engines
  work unchanged**; fine-tuned with a small LR rather than frozen.
- **Objective** — the recurring lesson (ICAE → E2LLM → LCLM): **reconstruction alone
  collapses to copying**; **NTP alone loses exact-string fidelity**; the **mixture** is what
  generalizes.

## Why this thread matters for the repo

- It is the **direct super-set of [MixedDecoder](../../mixed_decoder/mixed_decoder.md)**: MixedDecoder emits one `[CLS]`
  latent per chunk + a linear expansion; this thread says emit **several latents via
  mean/concat pooling**, move to a **4×–16×** operating point, add **reconstruction** as a
  co-objective, and **interleave** compressed/uncompressed spans throughout the sequence.
- It defines the **evaluation bar** MixedDecoder currently skips: [RULER][RULER] /
  [LongBench][LongBench] / [LongHealth][LongHealth] / [GSM8K][GSM8K] accuracy **plus**
  TTFT/memory curves versus [KV-cache compression][KVCache] baselines.
- It is the **inference-engine-friendly** branch of long-context: soft tokens are consumed
  like normal embeddings, so [vLLM][vLLM] / [SGLang][SGLang] need no per-head eviction logic
  (the incompatibility that limits [SnapKV][SnapKV]-style methods).

## Relation to the neighboring threads

- **Hard-token compression** ([thread](../hard_token/hard_token.md): [LLMLingua][LLMLingua], [Selective Context][SelectiveCtx],
  [CompAct][CompAct]) — deletes/paraphrases *real* tokens; complementary but intrinsically
  lossy on exact detail.
- **KV-cache compression** ([thread](../kv_cache/kv_cache.md): [SnapKV][SnapKV], [KVzip][KVzip], [Expected Attention][ExpAttn]) —
  compresses *after* a full prefill; soft tokens compress *before* it and reuse across turns.
- **Multimodal soft-token bridges** ([multimodal thread](../../mixed_decoder/multimodal/multimodal.md)) — Q-Former / Perceiver /
  [LLaVA][LLaVA] project *image* patches into decoder soft tokens; text context compression
  is the **same architecture with a text encoder**, which is why the [LLaVA][LLaVA]/
  [Cambrian][Cambrian] staged-alignment recipe transfers directly.
- **Agentic memory** ([Recursive Language Models][RLM], [MemGPT][MemGPT]) — soft-token
  memory can be composed with expand-on-demand tools (REFRAG, LCLM's `EXPAND(i)`) for exact
  retrieval instead of forcing all detail through the bottleneck.

## Open follow-ups for this thread

- **Adaptive / multi-granularity ratios** — choose the ratio per span by information density
  or perplexity rather than a fixed 4×/8×/16× ([Simple Context Compression][SimpleCC] is a
  first step). TODO recap.
- **Compressing the *generated* state** — long CoT, tool observations, agent working history —
  not just static input context.
- **Expand-on-demand as a first-class op** — [REFRAG][REFRAG]'s RL expansion vs LCLM's
  agentic `EXPAND(i)` tool; when is exact re-tokenization worth the latency?
- **Distillation of the encoder** — can a smaller/embedding encoder be distilled from a
  larger LM encoder without the LCLM "decoder ≫ encoder" scaling asymmetry?

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — the scaled, end-to-end instance of this thread.
- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) — the repo's own soft-token compressor and the concrete transfer target.
- [Qwen overview](../../qwen/overview.md) — the [Qwen3][Qwen3] / [Qwen3-Embedding][Qwen3Emb] backbones used as encoder & decoder.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[Qwen3]: https://arxiv.org/abs/2505.09388 "Qwen3 Technical Report (2025)"
[Qwen3Emb]: https://arxiv.org/abs/2506.05176 "Qwen3 Embedding (2025)"
[PrefixTuning]: ../../papers/softtoken_2021_prefix-tuning.md "Prefix-Tuning (Li & Liang 2021)"
[Gist]: ../../papers/softtoken_2023_gisting.md "Gist Tokens (Mu et al. 2023)"
[ICAE]: ../../papers/softtoken_2023_icae.md "In-Context Autoencoder (Ge et al. 2023)"
[AutoComp]: ../../papers/softtoken_2023_autocompressor.md "AutoCompressor (Chevalier et al. 2023)"
[CEPE]: https://arxiv.org/abs/2402.16617 "CEPE — Parallel Context Encoding (Yen et al. 2024)"
[xRAG]: https://arxiv.org/abs/2405.13792 "xRAG (Cheng et al. 2024)"
[E2LLM]: https://arxiv.org/abs/2409.06679 "E2LLM (Liao et al. 2025)"
[REFRAG]: https://arxiv.org/abs/2509.01092 "REFRAG (Lin et al. 2025)"
[SimpleCC]: https://arxiv.org/abs/2510.20797 "Simple Context Compression (Feldman & Artzi 2025)"
[LLMLingua]: https://arxiv.org/abs/2310.05736 "LLMLingua (Jiang et al. 2023)"
[SelectiveCtx]: https://arxiv.org/abs/2310.06201 "Selective Context (Li et al. 2023)"
[CompAct]: https://arxiv.org/abs/2407.09014 "CompAct (Yoon et al. 2024)"
[SnapKV]: ../../papers/kvcache_2024_snapkv.md "SnapKV (Li et al. 2024)"
[KVzip]: ../../papers/kvcache_2025_kvzip.md "KVzip (Kim et al. 2025)"
[ExpAttn]: ../../papers/kvcache_2025_expected-attention.md "Expected Attention (Devoto et al. 2025)"
[KVCache]: ../../papers/kvcache_2024_kvquant.md "KVQuant (Hooper et al. 2024)"
[vLLM]: https://arxiv.org/abs/2309.06180 "PagedAttention / vLLM (Kwon et al. 2023)"
[SGLang]: https://arxiv.org/abs/2312.07104 "SGLang (Zheng et al. 2024)"
[LLaVA]: https://arxiv.org/abs/2304.08485 "LLaVA (Liu et al. 2023)"
[Cambrian]: https://arxiv.org/abs/2406.16860 "Cambrian-1 (Tong et al. 2024)"
[RMSNorm]: https://arxiv.org/abs/1910.07467 "RMSNorm (Zhang & Sennrich 2019)"
[RULER]: https://arxiv.org/abs/2404.06654 "RULER (Hsieh et al. 2024)"
[LongBench]: https://arxiv.org/abs/2308.14508 "LongBench (Bai et al. 2024)"
[LongHealth]: https://arxiv.org/abs/2401.14490 "LongHealth (Adams et al. 2025)"
[GSM8K]: https://arxiv.org/abs/2110.14168 "GSM8K (Cobbe et al. 2021)"
[RLM]: https://arxiv.org/abs/2512.24601 "Recursive Language Models (Zhang et al. 2025)"
[MemGPT]: https://arxiv.org/abs/2310.08560 "MemGPT (Packer et al. 2023)"
