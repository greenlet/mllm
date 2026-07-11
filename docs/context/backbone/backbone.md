# Thread: Backbone models & components

The building blocks the [LCLM](../ctx_compression.md) stack (and the repo's own
[MixedDecoder](../../mixed_decoder/mixed_decoder.md)) is assembled from: the **encoder** and
**decoder** backbones, the **positional / normalization / activation** primitives inside every
transformer block, and the **attention kernels** that make long-sequence training feasible.
Unlike the compression *methods* threads ([soft-token](../soft_token/soft_token.md) ·
[KV-cache](../kv_cache/kv_cache.md) · [hard-token](../hard_token/hard_token.md)), this thread
is the **component inventory** — the pieces those methods plug together.

## The stack at a glance

```
                 ┌─────────────────────── Decoder: Qwen3-4B-Instruct ───────────────────────┐
 soft tokens ──► │  [RoPE] pos  →  attention ([FlashAttention-2] kernel)  →  [RMSNorm] pre-norm │ ──► logits
                 │                                                          →  MLP ([GELU]/SwiGLU) │
                 └────────────────────────────────────────────────────────────────────────────┘
       ▲
   [Adapter MLP]  (d_enc → d_dec, RMSNorm + GELU)
       ▲
                 ┌────────────── Encoder: Qwen3-Embedding-0.6B (causal) ──────────────┐
 raw tokens ───► │  [RoPE]  →  attention  →  [RMSNorm]  →  MLP  →  pooled latents      │
                 └──────────────────────────────────────────────────────────────────┘
   context-window extension: [YaRN] / RoPE scaling · packing: [FlexAttention] / varlen
```

## The components

| Component | Role in the stack | Why this one |
|---|---|---|
| [Qwen3][Qwen3] | **Decoder** backbone (`Qwen3-4B-Instruct-2507`) | Strong open instruct model; fully fine-tuned to consume soft tokens like ordinary embeddings. |
| [Qwen3-Embedding][Qwen3Emb] | **Encoder** backbone (`Qwen3-Embedding-0.6B`) | Embedding-model init **beats LM init** for the encoder (LCLM §5); queryable geometry. |
| [T5 / prefix-LM][PrefixLM] | Conceptual ancestor of encoder→decoder + **prefix** conditioning | The bidirectional-prefix intuition LCLM tests — and finds a **causal** encoder mask beats. |
| [RoPE][RoPE] | **Rotary** positional encoding in every block | Relative positions via rotation; the substrate context-window extension acts on. |
| [YaRN][YaRN] | **Context-window extension** by RoPE frequency scaling | Stretches the decoder beyond its native window to cover long compressed sequences. |
| [RMSNorm][RMSNorm] | **Pre-norm** in blocks and in the adapter | Cheaper than LayerNorm; the [LLaVA][LLaVA]-style adapter uses RMSNorm pre-norm. |
| [GELU][GELU] | Adapter / MLP **activation** | Smooth nonlinearity in the 2-layer projection adapter (and MLP blocks). |
| [FlashAttention-2][FlashAttn] | **Attention kernel** | IO-aware exact attention; makes full-length prefill / long-context training tractable. |
| [FlexAttention][FlexAttn] | **Programmable masking** kernel | Block-diagonal / varlen masks to **pack** examples with per-example attention reset. |

## How the pieces bind to the compression design choices

- **Encoder init & mask** — the choice of [Qwen3-Embedding][Qwen3Emb] (not a plain
  [Qwen3][Qwen3] LM) and a **causal** rather than bidirectional [prefix-LM][PrefixLM] mask are
  the two encoder-side findings that most move pre-training loss (LCLM §2.3).
- **Adapter primitives** — [RMSNorm][RMSNorm] + [GELU][GELU] in a 2-layer MLP is the
  [LLaVA][LLaVA]-style connector that beats an attention adapter at less compute; it projects
  $d_{enc}\to d_{dec}$ per latent, no cross-latent mixing.
- **Positions & window** — [RoPE][RoPE] + [YaRN][YaRN] scaling is what lets the decoder's native
  window cover the (already shortened) compressed sequence, extending effective reach.
- **Kernels & packing** — [FlashAttention-2][FlashAttn] plus [FlexAttention][FlexAttn]/varlen
  block-diagonal masking implement the **packed** training (attention reset at example
  boundaries) the recipe relies on.

## Why this thread matters for the repo

- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) currently uses a **BERT (bidirectional)**
  encoder; this thread records the LCLM evidence that an **embedding-model, causal** encoder
  ([Qwen3-Embedding][Qwen3Emb]) is the better default, and lists the exact primitives
  ([RMSNorm][RMSNorm]/[GELU][GELU] adapter, [RoPE][RoPE]/[YaRN][YaRN] positions,
  [FlashAttention-2][FlashAttn]/[FlexAttention][FlexAttn] packing) to match.
- It is the **shared vocabulary** for the other threads: every
  [soft-token](../soft_token/soft_token.md) method fills the same encoder/adapter/decoder boxes
  with choices drawn from this inventory.
- It connects to the [Qwen overview](../../qwen/overview.md) for the backbone details reused as
  encoder and decoder.

## Open follow-ups for this thread

- **Encoder swap** — quantify the MixedDecoder gain from BERT →
  [Qwen3-Embedding][Qwen3Emb]-init causal encoder at matched compute. TODO recap.
- **Window extension for compressed sequences** — is [YaRN][YaRN] even needed once the sequence
  is 4×–16× shorter, or does native RoPE suffice?
- **Adapter ablation** — reconfirm [RMSNorm][RMSNorm]+[GELU][GELU] MLP vs attention adapter on
  the repo's data.
- **Kernel choice** — [FlashAttention-2][FlashAttn] vs [FlexAttention][FlexAttn] for the
  block-diagonal packed masks at the repo's sequence lengths.

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — the stack these
  components assemble into.
- [Qwen overview](../../qwen/overview.md) — the [Qwen3][Qwen3] / [Qwen3-Embedding][Qwen3Emb]
  backbones in depth.
- [Soft-token compression](../soft_token/soft_token.md) — the method thread that fills these
  boxes; [KV-cache](../kv_cache/kv_cache.md) · [Hard-token](../hard_token/hard_token.md) — the
  baseline threads.
- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) — the repo's compressor and the transfer
  target for these component choices.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
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
