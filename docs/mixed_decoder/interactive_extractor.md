# Interactive Extractor — query-conditioned iterative memory retrieval

> A design plan for replacing MixedDecoder's static linear prefix bottleneck with an
> **active, iterative *Expand → Visit → Pool* loop**: compressed chunk memory is expanded,
> queried by the prompt, and pooled back down — repeatedly — so the representation aligns
> its geometry with the question *before* the causal decoder generates the answer.
>
> Companion to [mixed_decoder.md](mixed_decoder.md). That doc diagnoses *why* the current
> design fails (capacity / addressability / forcing-function); this doc specifies *one*
> concrete fix and how to train, feed, and test it.

---

## 1. Problem

MixedDecoder today (see [mixed_decoder.md](mixed_decoder.md) §1 and
[mllm/model/mixed_decoder.py](../../mllm/model/mixed_decoder.py)):

- Each 128-token chunk → one BERT `[CLS]` vector `e ∈ ℝ^768` (`run_enc`, single vector).
- `emb_exp = Linear(768, emb_exp_rate·d_dec)` expands that one vector into
  `emb_exp_rate` decoder-space vectors (rate 4 in current runs). This is a **deterministic
  linear map of a single vector** → rank-≤1, **adds no information** and is **query-blind**.
- `build_decoder_input` lays out `[ctx_embs, SEP, prompt, target]` as **one causal stream**.
  Consequence: the context embeddings are produced *before* the prompt is seen and can
  **never re-read the question**; only the target can look back at them.

Three structural gaps follow (mapped to the axes in [mixed_decoder.md](mixed_decoder.md) §3.0):

| Gap | Axis | Symptom |
|---|---|---|
| One `[CLS]` + linear expansion | capacity + addressability | names/numbers/dates dropped (Phase 3 Apono/Hypoon fails) |
| Memory never sees the prompt | addressability | extraction is query-blind; the gist is organized for reconstruction, not retrieval |
| Static prefix, fully-trainable decoder | forcing function | cheapest gradient route = memorize Q→A in Qwen weights; val stagnates while train improves |

**Goal of this design:** give the bridge a mechanism to do *two things in parallel* — (a)
**expand** the gist back toward word-level capacity, and (b) **filter** it down to only the
content relevant to the prompt — by letting the memory *query the prompt* iteratively.

---

## 2. The technique — iterative `Expand → Visit → Pool`

Let the encoder emit a memory matrix for a context window of `N` chunks. The prompt provides
a conditioning matrix `Q ∈ ℝ^{L_q × d_dec}` (the prompt token embeddings already available in
`build_decoder_input` via `self.word_embeddings(prompt_toks)`).

One extraction layer `t`:

```
        Memory M_{t-1}        (N · K_{t-1} vectors, ℝ^{d_dec})
              │
   1. EXPAND  │  sequence-length expansion: 1 slot → s slots per group
              ▼
        Expanded E_t          (N · K_{t-1} · s vectors)
              │
   2. VISIT   │  cross-attention: E_t = queries, prompt Q = keys/values
              │  (memory reads "what is the question looking for")
              ▼
        Conditioned H_t       (same length, relevance-tagged)
              │
   3. POOL    │  query-driven downsampling (Perceiver/slot-attention/learned-query)
              ▼
        Filtered M_t          (N · K_t vectors; K_t ≤ K_{t-1}·s)
```

Run `L` loops, then prepend the final `M_L` to the causal decoder exactly where `ctx_embs`
go today. The user's worked example — `128 → 1 → expand to 2 → visit → pool to 1 → expand …`
then a fixed token window relevant to the prompt is extracted and decoded causally — is the
`s=2`, `K_t=K_{t-1}` special case of this loop.

### 2.1 The three actions, concretely

**Expand (reconstruct structural capacity).** A linear map can't add Shannon entropy, but it
*can* add **slots** for downstream attention to write routing keys into. Use a
**sequence-length** expansion, not a channel blow-up of one vector:
`Linear(d_dec, s·d_dec)` reshaped to `(N·K·s, d_dec)` — `s` *distinct* token slots per group.
Better: expand from the encoder's **full 128 hidden states** (not just `[CLS]`) so the new
slots carry genuinely different content (see §3, Option C).

**Visit (let memory see the prompt).** Standard causal masking forbids this; the extraction
phase must be **non-causal**. Cross-attention with memory as query, prompt as key/value: each
memory slot updates by reading the prompt, so a slot holding *"Nelly Armweak"* that matches
*"Who walked on the Hypoon?"* spikes its relevance. This is the **addressability** fix — the
component [mixed_decoder.md](mixed_decoder.md) §3.1/§3.3 (Perceiver/Q-Former/ColBERT) says is missing.

**Pool (filter to a query-conditioned gist).** Reduce length back to the sequence budget with
a **query-aware** pooler, not mean/max (which are query-blind): learned latent queries
(Perceiver Resampler) or slot attention. Slots that matched the prompt survive; irrelevant
ones are suppressed. This realizes "reduce the content to the gist related to the query."

---

## 3. Implementation options

Three wirings, ordered by how much they touch Qwen. All produce soft tokens consumed by the
existing `build_decoder_input` path.

### Option A — In-decoder bidirectional bridge then causal generation

Interleave extraction with the first decoder layers by editing the attention mask:

- **Layers 1–k (bidirectional "thinking"):** context slots ↔ prompt tokens attend both ways.
- **Bottleneck layer:** a Perceiver/top-k pooling drops context length to the budget.
- **Layers k+1…N (causal):** strictly causal; compressed query-informed context is frozen and
  Qwen generates autoregressively.

Pros: layer-wise conditioning (Flamingo / P-Tuning-v2 evidence, [mixed_decoder.md](mixed_decoder.md) §3.1/§3.6).
Cons: invasive; needs a custom attention mask inside Qwen layers; harder to isolate with LoRA;
gradient-checkpointing + `use_reentrant=False` interactions to verify.

### Option B — Front-end pre-processor (recommended first) *(Qwen pristine)*

Leave Qwen entirely untouched (frozen or LoRA). Build a standalone `IterativeExtractor`
module that runs the loop, then prepend its output where `ctx_embs` currently go.

```python
class IterativeExtractor(nn.Module):
    def __init__(self, d_model=768, d_dec=1536, num_loops=2, slots_per_chunk=4, n_heads=8):
        super().__init__()
        self.slots = slots_per_chunk
        # Expansion: encoder CLS (or pooled hidden states) -> slots decoder-space vectors
        self.expand = nn.Linear(d_model, slots_per_chunk * d_dec, bias=False)
        # VISIT: memory queries the prompt (non-causal cross-attention), one per loop
        self.cross_attns = nn.ModuleList(
            nn.MultiheadAttention(d_dec, n_heads, batch_first=True) for _ in range(num_loops)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_dec) for _ in range(num_loops))
        # POOL: query-aware reduction (start simple: per-group MLP; upgrade to slot-attn)
        self.poolers = nn.ModuleList(
            nn.Linear(slots_per_chunk * d_dec, slots_per_chunk * d_dec) for _ in range(num_loops)
        )

    def forward(self, chunk_embs, prompt_tok_embs, prompt_pad_mask=None):
        # chunk_embs: [B, N, d_model]; prompt_tok_embs: [B, L_q, d_dec]
        B, N, _ = chunk_embs.shape
        M = self.expand(chunk_embs).view(B, N * self.slots, -1)        # EXPAND
        for attn, norm, pool in zip(self.cross_attns, self.norms, self.poolers):
            visited, _ = attn(query=M, key=prompt_tok_embs, value=prompt_tok_embs,
                              key_padding_mask=prompt_pad_mask)         # VISIT (non-causal)
            M = norm(M + visited)
            M = pool(M.view(B, N, -1)).view(B, N * self.slots, -1)      # POOL (keep locality)
        return M                                                       # -> prepend as ctx_embs
```

Wire-in point: in `run_on_qna` / `run_on_text_citation`, replace the `emb_exp(ctx_embs)` step
with `IterativeExtractor(ctx_embs, prompt_embs)` and feed the result to `build_decoder_input`
(prompt is then also still appended in the causal stream, or optionally dropped from it once
the memory has already absorbed it — ablate both).

Pros: Qwen weights pristine → clean LoRA isolation, cheap to ablate, easy to checkpoint
separately, matches the literature consensus (freeze the LLM, [mixed_decoder.md](mixed_decoder.md) §3.2).
Cons: prefix-only conditioning (no per-layer re-reading) — acceptable as a first cut.

### Option C — Resampler over full encoder states (capacity upgrade, combine with B)

Expand from the encoder's **128 token hidden states**, not the single `[CLS]`, using K learned
queries (Q-Former). This is the real capacity fix from [mixed_decoder.md](mixed_decoder.md) §4.1; the iterative
prompt-visiting loop of Option B then sits *on top* of genuinely independent slots. Requires
`run_enc` to return `out_enc_last_hidden_state` (already computed) instead of only `[:, 0]`.

**Recommendation:** start with **Option B** (front-end, frozen/LoRA Qwen) for a clean,
isolable experiment; fold in **Option C**'s full-hidden-state expansion once the loop helps;
keep **Option A** as a later experiment if prefix-only saturates.

---

## 4. Training — make the loop unavoidable

A beautiful loop still collapses to identity if the decoder can cheat. Couple it to the
forcing functions from [mixed_decoder.md](mixed_decoder.md) §4.2–§4.3:

1. **Freeze or LoRA the decoder.** No `freeze_decoder` flag exists today (only `frzenc*` on
   the encoder). Add one, or attach LoRA to Qwen and train only `{extractor, LoRA}`. This
   removes the memorize-in-weights route that causes train↓/val→.
2. **Multi-objective loss** (weighted sum, [mixed_decoder.md](mixed_decoder.md) §4.3):
   $$\mathcal{L} = \lambda_\text{recon}\mathcal{L}_\text{AE} + \lambda_\text{cont}\mathcal{L}_\text{LM} + \lambda_\text{KL}\mathcal{L}_\text{distill} + \lambda_\text{task}\mathcal{L}_\text{extract}$$
   - $\mathcal{L}_\text{AE}$ — reconstruct the chunk from its memory (Phase-1 already converges;
     keep it **on** as a capacity guarantee).
   - $\mathcal{L}_\text{distill}$ — per-token KL vs. a **frozen full-context Qwen teacher** on
     context-grounded continuations (dense signal; the student can only match it by preserving
     the facts). The user's "Idea 1".
   - $\mathcal{L}_\text{extract}$ — the structured-extraction / cite / QnA CE.
3. **Curriculum (BLIP-2 style):** Stage 1 = representation (AE + continuation + distill, *no*
   QnA) → Stage 2 = extraction/QnA bridging. Separates *what to extract* from *how to express*.
4. **Visibility ablation:** train with the prompt present **only** in the VISIT step (drop it
   from the causal stream) for part of the schedule, so the only way the answer can be produced
   is through the pooled memory — a direct forcing function for the loop itself.

### 4.1 Diagnosing whether the loop works

Log and visualize the **VISIT cross-attention maps**. On a synthetic key→value context with
the prompt `Key: X`, attention mass should concentrate on the slot(s) holding `X`. If it stays
diffuse, the loop is not extracting and the pooler is averaging — a concrete, early failure
signal independent of downstream loss.

---

## 5. Data — structured extraction to "grind" the context

> **User's question: can automated JSON / XML / XPath / SQL extraction examples serve as
> initial training on par with the Cite dataset, to enforce context grinding?**
>
> **Yes — this is one of the strongest available forcing functions, for four reasons:**
>
> 1. **Unguessable & fully grounded.** The answer (a field value, an attribute, a cell) lives
>    *only* in the provided record. Qwen's parametric priors cannot supply a random `"id":
>    7731`, so the cheapest path to low loss must run *through* the memory — exactly the
>    property [mixed_decoder.md](mixed_decoder.md) §5 demands and that QnA lacks.
> 2. **Programmatically generated at unlimited scale**, with **free, exact labels** and
>    **tunable difficulty** — no annotation, no leakage, no dataset-imbalance confound
>    (the very confounds flagged in [mixed_decoder.md](mixed_decoder.md) §2 Phase 3).
> 3. **A query is intrinsic to the task.** XPath / JSONPath / a SQL `WHERE`/`SELECT` / an XML
>    tag *is* the prompt. This is precisely the "Visit" signal the loop needs — the prompt
>    selects a sub-region of a structured record, which is the cleanest possible training
>    signal for query-conditioned pooling.
> 4. **Difficulty is a single knob** (number of fields/rows/nesting depth), so the same family
>    doubles as the **capacity-vs-ratio probe** of [mixed_decoder.md](mixed_decoder.md) §4.4.

This generalizes the existing `Cite` task (`MixedDecoderDsType.Cite`,
[mllm/config/model.py](../../mllm/config/model.py)), which already extracts a contiguous span
out of a chunk and works well (Phase 2 "Plain Cite: very strong"). Structured extraction is
"Cite with a *structured query* instead of a span index" — a natural, in-house extension of the
pipeline that produced `MaskedCiteBatch`.

### 5.1 Proposed synthetic structured-extraction generators

| Family | Context (encoded) | Prompt (VISIT query) | Target | Difficulty knob |
|---|---|---|---|---|
| **JSON field** | a JSON object | `JSONPath` / `key:` | field value | #keys, nesting depth |
| **XML / XPath** | an XML fragment | `XPath` expr | node text / attr | #nodes, attr count, depth |
| **SQL-select** | a small table (CSV/markdown rows) | `SELECT col WHERE k=v` | cell(s) | #rows, #cols, predicate count |
| **Key-value recall** | `k1:v1; k2:v2; …` | `kI` | `vI` | #pairs (→ capacity curve) |
| **Multi-field** | record with M fields | several keys | tuple | M, #queried |

All five are trivially generated from random tokens or from the existing wiki corpus
(entity/number slot-filling), produce a dense per-position target, and need **no human labels**.
Start them at chunk size = `inp_len = 128` so they drop straight into the current encoder.

### 5.2 How to mix with existing data

Following [mixed_decoder.md](mixed_decoder.md) §5.6, with structured extraction added as the
context-grinding spine:

1. **Stage-1 train (context grinding):** structured extraction (JSON/XML/SQL/KV) **+** salient-span
   cloze on wiki **+** context reconstruction (AE always on).
2. **Stage-1 forcing add-on:** **VitaminC** (counterfactual; near-identical claims, opposite
   labels — a model ignoring evidence scores at chance).
3. **Stage-2 bridge:** the existing QnA aggregate (`QnaAns` / `QnaAnsCite`).
4. **Diagnostics (never trained on):** passkey, MQAR, key-value recall, swept over
   `emb_exp_rate`/`slots_per_chunk` and chunk length.

**Caveat (avoid a new shortcut).** Purely synthetic, uniform-format records risk teaching a
*format-parsing* skill rather than general extraction. Mitigate by: randomizing serialization
(spacing, key order, quoting), interleaving real wiki/NER spans, and validating on a *held-out
format* (e.g. train JSON+SQL, test XPath). If the loop only works on the trained serializer,
it memorized the grammar, not the content.

---

## 6. Tests

Concrete, mostly synthetic, runnable before/after every architecture change — a regression
harness ([mixed_decoder.md](mixed_decoder.md) §4.9 step 1).

### 6.1 Capacity / extraction probes (held-out, diagnostic)
- **Passkey / needle-in-haystack** — does the memory keep one random token through compression.
- **MQAR** (multi-query associative recall) — selective recall of several values.
- **Key-value / JSON-field recall** — accuracy vs. #fields → the capacity curve per
  `slots_per_chunk` × chunk length.
- **Renamed-entity QnA (Apono/Hypoon at scale)** — the §2 grounding test; priors cannot help.

### 6.2 Loop-mechanism tests (white-box)
- **VISIT attention concentration** (§4.1): on `Key: X`, fraction of attention mass on the
  correct slot; expect it to rise across loops and over training.
- **Ablate the loop:** `num_loops = 0` (pure expand+pool, no VISIT) vs. `1` vs. `2`; extraction
  accuracy must increase with loops, else the prompt-visiting adds nothing.
- **Prompt-visibility ablation:** prompt only in VISIT vs. also in causal stream; if accuracy
  is unchanged when the prompt is removed from the causal stream, the loop is doing the work.
- **Identity-collapse check:** with the decoder frozen, confirm loss does **not** improve when
  the extractor is bypassed (random/zeroed memory) — guards against the decoder cheating.

### 6.3 Format-generalization tests (anti-shortcut, §5.2 caveat)
- Train on JSON+SQL, **test on XPath** (held-out serializer).
- Randomized-serialization eval: same content, shuffled key order / spacing.

### 6.4 Existing-task non-regression
- **Plain Cite** must stay near-perfect (it already is — Phase 2).
- **QnA val loss** and the **val−train gap** vs. the linear-expansion baseline (the §2 metric
  that exposed Phase-3 stagnation).

---

## 7. Suggested ladder

1. Build the **synthetic structured-extraction generators** (§5.1) + the **diagnostic harness** (§6.1).
2. Add **`freeze_decoder` / LoRA** to Qwen (§4 step 1) — measure the val−train gap on current
   linear-expansion baseline first (isolates the forcing-function effect).
3. Implement **Option B** `IterativeExtractor` (frozen/LoRA Qwen); sweep
   `slots_per_chunk ∈ {4,8,16}`, `num_loops ∈ {0,1,2}`; track §6.2 loop tests.
4. Add **reconstruction + distillation** (§4 step 2) and the **Stage-1/Stage-2 curriculum**.
5. Fold in **Option C** (expand from full encoder hidden states) if the loop helps.
6. Map the **ratio/accuracy curve** (§6.1) and pick an operating point.
7. Only then consider **Option A** (in-decoder layer-wise) and port to a larger decoder.

---

## 8. References

This plan is a concrete instantiation of the directions in
[mixed_decoder.md](mixed_decoder.md) §4 (architecture) and §5 (data). Key external anchors:
Perceiver / Q-Former (learned-query resampler), Flamingo (frozen LLM + layer-wise
cross-attention), ICAE / xRAG (freeze decoder, reconstruction + distillation), ColBERT
(per-token addressability), VitaminC (counterfactual grounding). Full citations in
[mixed_decoder.md](mixed_decoder.md) §6.
