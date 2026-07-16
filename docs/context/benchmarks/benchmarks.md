# Thread: Long-context benchmarks & datasets (the evaluation bar)

What does it *mean* for [LCLM](../ctx_compression.md) to "match KV-cache baselines"? It means
holding accuracy on a specific battery of long-context tasks while cutting time-to-first-token
and peak memory. This thread collects the **motivation** papers (why long context is hard to
use at all), the **benchmarks** LCLM reports on, and the **training-methodology** references
that shape how a long-context model is built and measured. Together they define the bar every
soft-token compressor — including the repo's own
[MixedDecoder](../../mixed_decoder/mixed_decoder.md) — must clear.

> **The through-line.** Long inputs don't automatically help: models
> [lose information in the middle][LostMiddle] and their [effective context falls short][EffCtx]
> of the advertised window. So compression must be judged not by *reconstruction* but by
> *task accuracy at length* — which is exactly what [RULER][RULER], [LongBench][LongBench],
> [LongHealth][LongHealth], and [GSM8K][GSM8K] measure, and what
> [How to Train Long-Context LMs][TrainLC] tells you how to optimize.

## The landscape

| Reference | Year | Role | What it measures / argues | How LCLM uses it |
|---|---|---|---|---|
| [Lost in the Middle][LostMiddle] | 2024 | Motivation | Accuracy is **U-shaped** in the position of the relevant fact: models use the *start* and *end* of a long context far better than the *middle*. | Justifies **interleaving** compressed/uncompressed spans *throughout* the sequence (§3.1) so the decoder learns to condition on latent context at many positions, not just a prefix. |
| [Why Does Effective Context Fall Short?][EffCtx] | 2025 | Motivation | The **usable** context length is much smaller than the trained window; positional under-training leaves the tail of the window weak. | Frames compression as an *effective-length multiplier*: fewer, denser tokens keep the useful signal inside the well-trained region. |
| [RULER][RULER] | 2024 | Benchmark (primary) | Synthetic **needle-in-a-haystack, multi-hop tracing, aggregation** tasks at controlled lengths (4K–128K); the paper shows real usable length ≪ claimed. | The **headline metric** (RULER 4K–16K avg). High-ratio ($16\times$) fidelity on multi-hop RULER is where LCLM still trails full KV — the hardest sub-tasks. |
| [LongBench][LongBench] | 2024 | Benchmark | **Bilingual, multitask** real-world long-context suite (QA, summarization, code, few-shot). | Breadth check that compression generalizes beyond synthetic probes. |
| [LongHealth][LongHealth] | 2025 | Benchmark (domain) | Multiple-choice **QA over long clinical documents** (LongHealth5 = 5-doc setting). | A dense, domain-specific recall test; LCLM is competitive-to-better here even at $8\times$. |
| [GSM8K][GSM8K] | 2021 | Benchmark (dense/short) | Grade-school **math word problems** — short, information-dense; used as **whole-prompt** compression. | Evidence LCLM isn't specialized only for long QA: at high ratios it far outperforms KV-cache baselines on this short, dense task. |
| [How to Train Long-Context LMs][TrainLC] | 2025 | Methodology | Data mix, continued-pretraining schedule, and evaluation practice for **effective** (not just nominal) long context. | Informs LCLM's continual-pretraining data (long-doc sources) and the choice to measure *effective* accuracy, not perplexity alone. |

## How the metrics wire into LCLM's results

- **Instructions stay uncompressed; only the long segment is compressed.** Every benchmark is
  run this way so the score isolates *context-compression* fidelity, not instruction-following
  degradation.
- **Accuracy vs. efficiency, jointly.** Each benchmark accuracy is paired with **TTFT** and
  **peak GPU memory** (single H200) at matched compression ratio — the Pareto frontier claim
  is meaningless without both axes.
- **Ratio-stratified reporting.** Results are broken out at $4\times/8\times/16\times$ because
  the benchmarks degrade at *different rates*: dense/short (GSM8K) tolerates high ratios far
  better than multi-hop RULER.
- **The motivation papers set the design, not just the pitch.** *Lost in the Middle* →
  interleave compressed spans everywhere; *Effective Context* → treat compression as extending
  usable length rather than raw window.

## Why this thread matters for the repo

- It is the **evaluation bar [MixedDecoder](../../mixed_decoder/mixed_decoder.md) currently
  skips.** MixedDecoder reports reconstruction / task-CE on its own data; to be comparable it
  must report **RULER / LongBench / LongHealth / GSM8K accuracy plus TTFT & memory curves**
  versus the [KV-cache](../kv_cache/kv_cache.md) baselines.
- It explains **why reconstruction alone is the wrong objective**: none of these benchmarks
  reward copying — they reward *retrieval and reasoning at length*, which is why the
  [soft-token thread](../soft_token/soft_token.md)'s NTP+reconstruction mixture matters.
- It supplies the **length ladder** (4K → 1M) any long-context claim in the repo should be
  plotted against, and the **position-sensitivity** caveat (*Lost in the Middle*) that any
  interleaving strategy must be tested for.

## Relation to the neighboring threads

- **Soft-token compression** ([thread](../soft_token/soft_token.md)) — the methods being
  scored; this thread is the scoreboard.
- **KV-cache compression** ([thread](../kv_cache/kv_cache.md)) — the baselines LCLM is plotted
  against; they appear as near-vertical lines on the TTFT axis because their cost is the full
  prefill, largely independent of ratio.
- **Backbone components** ([thread](../backbone/backbone.md): [RoPE][RoPE] · [YaRN][YaRN]) —
  the positional machinery whose *under-training* is exactly what [Effective Context][EffCtx]
  diagnoses; length-extension quality is measured on the benchmarks here.
- **Agentic memory & expand-on-demand** — RULER needle tasks are where LCLM's `EXPAND(i)` tool
  recovers exact-match accuracy at $16\times$; the benchmark is what makes that gain visible.

## Open follow-ups for this thread

- **Position-sweep on interleaving.** Directly re-run a *Lost in the Middle* style sweep on
  LCLM: vary where the compressed vs. uncompressed spans sit and measure the U-curve under
  compression. TODO experiment.
- **Effective-length curve.** Plot accuracy vs. *effective* (not nominal) length at each ratio
  to quantify the [EffCtx][EffCtx] multiplier compression buys. TODO experiment.
- **Add a generation-side benchmark.** All of these score *reading* a long context; a companion
  measuring compression of the model's **generated** state (long CoT) is missing. TODO.
- **Paper recaps to add.** None of the entries here has a local recap yet
  (RULER, LongBench, LongHealth, GSM8K, Lost-in-the-Middle, Effective-Context, Train-LC). TODO recaps.

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — §4 (Results) is the
  scoreboard this thread annotates; §7.7 is the reference list it expands.
- [Soft-token / encoder–decoder context compression](../soft_token/soft_token.md) — the methods
  under test.
- [KV-cache compression](../kv_cache/kv_cache.md) — the efficiency baselines.
- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) — the repo's compressor that must adopt
  these benchmarks to become comparable.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[RULER]: https://arxiv.org/abs/2404.06654 "RULER (Hsieh et al. 2024)"
[LongBench]: https://arxiv.org/abs/2308.14508 "LongBench (Bai et al. 2024)"
[LongHealth]: https://arxiv.org/abs/2401.14490 "LongHealth (Adams et al. 2025)"
[GSM8K]: https://arxiv.org/abs/2110.14168 "GSM8K (Cobbe et al. 2021)"
[LostMiddle]: https://arxiv.org/abs/2307.03172 "Lost in the Middle (Liu et al. 2024)"
[EffCtx]: https://arxiv.org/abs/2410.18745 "Why Does the Effective Context Length Fall Short? (An et al. 2025)"
[TrainLC]: https://arxiv.org/abs/2410.02660 "How to Train Long-Context LMs Effectively (Gao et al. 2025)"
[RoPE]: https://arxiv.org/abs/2104.09864 "RoFormer / RoPE (Su et al. 2021)"
[YaRN]: https://arxiv.org/abs/2309.00071 "YaRN (Peng et al. 2023)"
