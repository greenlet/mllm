# Thread: Continual training & catastrophic forgetting (the fine-tuning risk)

[LCLM](../ctx_compression.md) **fully fine-tunes its decoder** ([Qwen3-4B][Qwen3]) to consume
compressed context — and full fine-tuning of a pretrained LLM is exactly the setting where
**catastrophic forgetting** bites: the model learns the new "read latents" skill while quietly
losing its original reasoning/instruction-following ability. LCLM's *entire staged recipe*
(§3.2) is a forgetting-mitigation strategy in disguise. This thread collects the references that
diagnose the problem and justify the recipe's countermeasures.

> **The tension.** LCLM ablations show **frozen-decoder** and **[LoRA][LoRA]-only** variants
> *substantially underperform* full-parameter training — so the decoder *must* be fully tuned.
> But full tuning is what forgets. The resolution isn't "freeze more"; it's **adapter warmup +
> small-LR staged unfreeze + a continual-pretraining data mix** that keeps the original
> distribution in the loss.

## The landscape

| Reference | Year | What it establishes | How it shapes LCLM's recipe |
|---|---|---|---|
| [Catastrophic Forgetting during Continual Fine-tuning][CatForgetLLM] | 2025 | Forgetting **worsens with model scale** during continual instruction tuning; general capabilities (reasoning, in-context learning, factual recall) degrade as new skills are added. | Motivates the **small decoder LR** in Stage 2 and the choice *not* to train end-to-end from step 0 (large early gradients are exactly what erase pretrained ability). |
| [Revisiting Catastrophic Forgetting in LLM Tuning][RevisitCF] | 2024 | Systematizes *when* forgetting happens in LLM fine-tuning and *which* mitigations (data replay, LR control, staging) actually help. | Justifies the **continual-pretraining data mix** (interleaved compressed/uncompressed general text) as implicit replay, and **progressive unfreezing** as LR/staging control. |
| [LoRA][LoRA] | 2021 | Parameter-efficient tuning: freeze base weights, learn low-rank deltas — a standard forgetting *avoidance* tool because the base is untouched. | The tempting-but-insufficient shortcut: LCLM **tried it and found it underperforms** full-FT for this task. Forgetting must be managed *within* full-FT, not sidestepped by LoRA. |

## Why the staged recipe *is* the mitigation

LCLM §3.2's four stages map one-to-one onto known forgetting controls:

```
 Stage 0  adapter warmup (enc+dec frozen)   ──►  no decoder gradients yet → zero forgetting while the bridge learns
 Stage 1  encoder training (dec frozen)     ──►  decoder still protected; representation adapts first
 Stage 2  end-to-end, SMALL decoder LR      ──►  gentle updates = the LR-control mitigation from [RevisitCF]
 Stage 3  SFT, higher decoder LR            ──►  task alignment only after the model already tolerates latents
```

- **Warm up the bridge before touching the backbone.** Adapter-only Stage 0 means the decoder
  never sees large, misaligned gradients from a cold projector — the single biggest source of
  early degradation (the same reason [VLM alignment](../multimodal/multimodal.md) freezes the
  LLM first).
- **Small LR is the load-bearing knob.** Stage 2 unfreezes the decoder at a *small* LR precisely
  because [scale amplifies forgetting][CatForgetLLM]; a 4B decoder tuned aggressively would lose
  its GSM8K/reasoning ability even as RULER improves.
- **Data mix = implicit replay.** Interleaving compressed *and* uncompressed general text keeps
  the original next-token distribution in the objective, the replay-style mitigation
  [RevisitCF][RevisitCF] identifies — without a separate rehearsal buffer.
- **Full-FT is non-negotiable here.** Because LoRA-only/frozen underperform, the recipe can't
  hide behind parameter freezing; it must *manage* forgetting through order, LR, and data.

## Why this thread matters for the repo

- It is the **risk register** for the repo's [MixedDecoder](../../mixed_decoder/mixed_decoder.md):
  the moment it unfreezes Qwen to accept compressed embeddings, it inherits this forgetting risk
  and must adopt the same guards (adapter warmup, small-LR staging, mixed data) rather than a
  single-phase full-FT.
- It explains a *negative* result cleanly: MixedDecoder's task-CE-only, single-stage training is
  both a [forcing-function](../multimodal/multimodal.md) problem *and* a forgetting problem —
  aggressive full-FT on a narrow objective is the worst case for both.
- It provides the **evaluation caveat** for the [benchmark bar](../benchmarks/benchmarks.md):
  always report *retained* general capability (e.g. GSM8K, base instruction-following), not just
  the new long-context score, so forgetting is visible.

## Relation to the neighboring threads

- **Multimodal / VLM alignment** ([thread](../multimodal/multimodal.md)) — the staged-unfreeze
  recipe this thread reads as forgetting mitigation; VLM alignment freezes-then-thaws for the
  same reason.
- **Soft-token compression** ([thread](../soft_token/soft_token.md)) — the NTP+reconstruction
  objective mix doubles as replay (general text stays in the loss).
- **Backbone components** ([thread](../backbone/backbone.md)) — the [Qwen3][Qwen3] decoder whose
  pretrained capability is the thing at risk.

## Open follow-ups for this thread

- **Measure the forgetting curve.** Track base-capability benchmarks (GSM8K, MMLU-style,
  instruction-following) *across* Stages 0→3 to quantify what each stage costs/retains. TODO experiment.
- **Replay-ratio sweep.** Vary the uncompressed:compressed mix in continual pretraining and plot
  retained vs. new capability. TODO experiment.
- **LoRA-plus-warmup revisit.** Re-test whether LoRA becomes competitive *after* a full-FT warmup
  (rank/target-module sweep), reconciling LCLM's negative LoRA result with its convenience. TODO experiment.
- **Paper recaps to add.** Neither forgetting paper has a local recap yet
  (Continual-FT Forgetting, Revisiting-CF); [LoRA][LoRA] is covered in the backbone thread. TODO recaps.

## See also

- [LCLM — End-to-End Context Compression at Scale](../ctx_compression.md) — §3.2 is the staged
  recipe this thread reads as forgetting mitigation; §7.11 is the reference list it expands.
- [Multimodal / VLM alignment](../multimodal/multimodal.md) — the freeze-then-thaw schedule LCLM borrows.
- [Soft-token / encoder–decoder context compression](../soft_token/soft_token.md) — the objective
  mix that doubles as replay.
- [MixedDecoder](../../mixed_decoder/mixed_decoder.md) — the repo's compressor that inherits this risk on unfreeze.

---

<!-- Link reference definitions (invisible in rendered output) -->

[paper]: https://arxiv.org/abs/2606.09659 "End-to-End Context Compression at Scale (2026)"
[CatForgetLLM]: https://arxiv.org/abs/2308.08747 "Catastrophic Forgetting during Continual Fine-tuning (Luo et al. 2025)"
[RevisitCF]: https://arxiv.org/abs/2406.04836 "Revisiting Catastrophic Forgetting in LLM Tuning (Li et al. 2024)"
[LoRA]: https://arxiv.org/abs/2106.09685 "LoRA (Hu et al. 2021)"
[Qwen3]: https://arxiv.org/abs/2505.09388 "Qwen3 Technical Report (2025)"
