# Thread: Efficient long-sequence modeling (the orthogonal axis)

[LCLM](../ctx_compression.md) makes long context cheap by **shortening the input** — compress
$N$ tokens into one latent, then run a *standard* quadratic Transformer over the shorter
sequence. This thread is the **other** way to make long context cheap: change the **sequence
model itself** — recurrent memory, sub-quadratic attention, state-space models, or a cheaper KV
representation — so cost grows sub-linearly with length while keeping every token. The two axes
are **orthogonal and composable**: you can run a soft-token compressor *on top of* a linear-attention
or MLA decoder. This thread exists to place LCLM on that map and flag the composition opportunities.

> **The through-line.** *Compress the sequence* (LCLM, this repo) vs. *compress the computation*
> (linear attention, SSMs, MLA) vs. *compress the memory of the past* (Transformer-XL,
> Compressive Transformers). LCLM sits in the first bucket but borrows its core intuition —
> "summarize old context into a few vectors" — straight from the third.

## The landscape

| Model | Year | Mechanism | Family | Relation to LCLM |
|---|---|---|---|---|
| [Transformer-XL][TransfoXL] | 2019 | **Segment-level recurrence**: cache the previous segment's hidden states as extended context; relative positions. | Recurrent memory | The ancestor of "carry a compressed past." LCLM makes that carried state *explicitly* short (latents) instead of a rolling activation cache. |
| [Compressive Transformer][CompTransf] | 2019 | Adds a **second memory** that *compresses* old activations (pooling/conv) into fewer slots before eviction. | Recurrent memory | The **closest conceptual ancestor**: compress old context into a few vectors. LCLM's difference is *end-to-end learned* compression of the *input* with a task objective, not a fixed pooling of activations. |
| [Linear Attention][LinAttn] | 2020 | Kernel-feature trick makes attention **associative** → O(N) with an RNN-style recurrent state. | Sub-quadratic attention | Cuts the decoder's per-token cost; **composes** with LCLM (shorter sequence *and* cheaper attention). |
| [S4][S4] | 2021 | **Structured state-space** sequence operator; long-range mixing via a learned linear SSM, sub-quadratic. | State-space | Alternative decoder backbone; orthogonal to input compression. |
| [Mamba][Mamba] | 2024 | **Selective** SSM with input-dependent gating; matches Transformer quality at O(N) inference. | State-space | A candidate *compression-friendly* decoder: LCLM latents fed to a Mamba decoder would stack both savings. |
| [Kimi Linear][KimiLinear] | 2025 | Hybrid **linear-attention** architecture tuned to beat full attention at long context in practice. | Sub-quadratic attention | Production evidence the sub-quadratic axis is real; a natural decoder to test LCLM latents on. |
| [MLA (DeepSeek-V2)][MLA] | 2024 | **Multi-head Latent Attention**: compress KV into a low-rank **latent** cache, decompress per head. | Cheaper KV | The *KV-side* analog of LCLM's *input-side* latent: both shrink what the decoder must hold. Composable — MLA shrinks per-token KV, LCLM shrinks the token count. |
| [DeepSeek-V4][DSV4] | 2026 | Million-token context via MLA + systems co-design. | Cheaper KV | The scaled deployment of the MLA line; a target decoder family for compressed-context serving. |

## Three ways to buy long context (and where LCLM sits)

```
  compress the MEMORY of the past      compress the COMPUTATION            compress the SEQUENCE
  (keep tokens, shrink the cache)       (keep tokens, cheaper mixing)       (fewer tokens, standard model)
  Transformer-XL, Compressive-T,        Linear Attn, S4, Mamba,             LCLM  ◄── this repo
  MLA / DeepSeek-V2/V4                   Kimi Linear
        │                                      │                                   │
        └──────────────── all three COMPOSE; LCLM is orthogonal to the other two ──┘
```

- **LCLM keeps a vanilla quadratic Transformer** and wins by making $N_{\text{tokens}}$ smaller
  *before* prefill — which is why it stays [engine-compatible](../systems/systems.md).
- **The other two axes change the model**, so they need custom kernels/backbones but keep every
  token. Their savings **multiply** with LCLM's rather than competing.
- **Compressive Transformer is the idea LCLM industrializes**: "summarize old context into a few
  vectors," but end-to-end learned with NTP + reconstruction instead of fixed pooling.

## Why this thread matters for the repo

- It clarifies that the repo's [MixedDecoder](../../mixed_decoder/mixed_decoder.md) is on the
  **sequence-compression** axis; efficiency claims should be stated as *complementary to* (not
  competing with) linear-attention / SSM / MLA decoders.
- It surfaces the highest-leverage **composition experiment**: feed compressed latents to a
  **Mamba / MLA** decoder and measure whether the savings stack without accuracy loss.
- It gives the honest framing for benchmarks: on the [long-context bar](../benchmarks/benchmarks.md),
  LCLM should be compared to *full-attention* decoders at equal accuracy, and separately noted
  as composable with sub-quadratic ones.

## Relation to the neighboring threads

- **KV-cache compression** ([thread](../kv_cache/kv_cache.md)) — MLA is the *architectural*
  cousin of post-hoc KV eviction: it builds the low-rank latent cache into the model instead of
  pruning after prefill.
- **Soft-token compression** ([thread](../soft_token/soft_token.md)) — LCLM's own family; this
  thread is the set of decoder backbones those soft tokens could run on.
- **Inference engines & systems** ([thread](../systems/systems.md)) — why LCLM's standard-Transformer
  choice matters: linear/SSM decoders trade engine ubiquity for asymptotics.
- **Backbone components** ([thread](../backbone/backbone.md)) — the attention/RoPE stack these
  models replace or extend.

## Open follow-ups for this thread

- **Latents on a sub-quadratic decoder.** Swap the [Qwen3][Qwen3] decoder for a Mamba/MLA-style
  backbone and check that input compression + cheap attention savings **compose**. TODO experiment.
- **Compressive-Transformer baseline.** Compare LCLM's learned latents against fixed-pooling
  memory compression at matched slot count. TODO experiment.
- **Paper recaps to add.** None of the entries here has a local recap yet (Transformer-XL,
  Compressive Transformer, Linear Attention, S4, Mamba, Kimi Linear, MLA, DeepSeek-V4). TODO recaps.

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — the sequence-compression
  instance this thread situates; §7.9 is the reference list it expands.
- [Soft-token / encoder–decoder context compression](../soft_token/soft_token.md) — LCLM's family.
- [KV-cache compression](../kv_cache/kv_cache.md) — post-hoc cousin of MLA's built-in latent KV.
- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) — the repo's sequence-compression compressor.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[TransfoXL]: https://arxiv.org/abs/1901.02860 "Transformer-XL (Dai et al. 2019)"
[CompTransf]: https://arxiv.org/abs/1911.05507 "Compressive Transformers (Rae et al. 2019)"
[LinAttn]: https://arxiv.org/abs/2006.16236 "Linear Transformers (Katharopoulos et al. 2020)"
[S4]: https://arxiv.org/abs/2111.00396 "S4 (Gu et al. 2021)"
[Mamba]: https://arxiv.org/abs/2312.00752 "Mamba (Gu & Dao 2024)"
[KimiLinear]: https://arxiv.org/abs/2510.26692 "Kimi Linear (2025)"
[MLA]: https://arxiv.org/abs/2405.04434 "DeepSeek-V2 / MLA (Liu et al. 2024)"
[DSV4]: https://github.com/deepseek-ai "DeepSeek-V4 (2026)"
[Qwen3]: https://arxiv.org/abs/2505.09388 "Qwen3 Technical Report (2025)"
