# MixedDecoder

> A BERT encoder compresses fixed-size text chunks into single embeddings; a causal
> LLM decoder (Qwen2.5 / GPT-2 / BERT-dec) consumes those embeddings as *soft tokens*
> mixed with ordinary token embeddings and generates text conditioned on them.

---

## 1. Architecture

### 1.1 High-level data flow

```
raw text
  │  chunk into windows of inp_len = 128 tokens (BERT WordPiece)
  ▼
[CLS] t_1 … t_126 [SEP]          ──►  EncoderBert (bert-base-uncased)  ──►  CLS vector  e ∈ ℝ^768
  (one chunk)                          12 layers, d_model = 768                (one per chunk)
  ▼
optional expansion  e ∈ ℝ^768  ──►  Linear(768 → emb_exp_rate · d_dec)  ──►  reshape to (emb_exp_rate, d_dec)
  ▼
context window of N chunks  ──►  N · emb_exp_rate soft vectors in decoder space ℝ^{d_dec}
  ▼
decoder input  =  [ ctx_embs , (SEP/EOS) , prompt_tok_embs , target_tok_embs ]   (all in ℝ^{d_dec})
  ▼
Causal decoder (Qwen2.5-1.5B, d_dec = 1536, RoPE, bf16 compute)
  ▼
cross-entropy loss on target positions only
```

### 1.2 Components and dimensions

| Component | Spec | Notes |
|---|---|---|
| **Encoder** | `bert-base-uncased`, 12 layers, `d_model = 768` | `EncBertCfg(inp_len=128, emb_type=Cls)`; emits the `[CLS]` last-hidden-state vector per chunk (`run_enc`, [mllm/model/mixed_decoder.py](../../mllm/model/mixed_decoder.py)). |
| **Chunk size** | `inp_len = 128` WordPiece tokens | 126 content + `[CLS]` + `[SEP]`. **128:1 compression** into one 768-d vector. |
| **Embedding expansion** | `emb_exp = nn.Linear(768, emb_exp_rate · d_dec, bias=False)` | `emb_exp_rate = 4` in current runs; one 768-d vector → 4 vectors of `d_dec`. No non-linearity. |
| **Context window** | `emb_win_min_size … emb_win_max_size` (e.g. 2…6) | Number of chunk embeddings placed in the prefix; randomized per step at train time, fixed at eval. Filler embeddings from other batch samples pad short windows (`run_on_qna`). |
| **Decoder** | `Qwen/Qwen2.5-1.5B`, `d_dec = 1536`, RoPE, gradient-checkpointed, fp32 weights / bf16 autocast | Alternatives: `GPT-2` (`d_dec=768`, learned abs-pos), `BertGenerationDecoder`. For Qwen, `pos_emb = None` (RoPE handles position). |
| **Projection** | `enc_proj = Linear(768, d_dec)` | Only used when `emb_exp_rate ≤ 0` and `d_model ≠ d_dec`. With expansion active it is unused. |
| **Delimiter** | `use_sep` → one `SEP`/`EOS` embedding between context and prompt | Configurable; recent Qwen runs use `sepF`. |
| **Sequence budget** | `max_seq_len = 400` | Target is right-truncated to fit (`build_decoder_input`). |
| **Loss** | causal CE on target span only; prefix labels = `-100` | Optional mask-aware weighting only when encoder/decoder share a vocab (not for BERT-enc + Qwen-dec). |

### 1.3 Key design choices and their consequences

1. **Single-vector bottleneck.** Each 128-token chunk is reduced to exactly one `[CLS]` vector. All downstream "expansion" is a deterministic linear map of that one vector, so the *information content* per chunk is capped at one 768-d embedding regardless of `emb_exp_rate`.
2. **Soft-token mixing.** The decoder never sees context token IDs (unless `decoder_only`); it sees continuous vectors interleaved with real token embeddings — the same mechanism multimodal models use for image tokens (see §3).
3. **Decoder is fully fine-tuned** (`frzencF` toggles the *encoder*; there is no decoder-freeze flag). The Qwen weights are free to absorb task regularities directly.
4. **Causal attention over the whole sequence.** Context embeddings, prompt, and target are one causal stream; the target can attend to all context embeddings but context embeddings cannot re-read the question.

---

## 2. Training phases and results

### Phase 1 — BERT autoencoder (compression / decompression)

**Setup.** Encoder–decoder BERT trained to compress a 128-token chunk into a single `[CLS]` embedding and reconstruct the original tokens (the `pretrained_encdec` checkpoint that seeds MixedDecoder).

**Result.** Converges cleanly. 128 tokens → one 768-d vector → near-lossless reconstruction. This establishes that a single embedding can *hold* most of a 128-token chunk's surface content, and provides the encoder initialization for all later phases.

**Caveat.** The objective is *reconstruction* — the embedding is organized for regenerating surface form, not for being *queried*. This distinction reappears in Phase 3.

### Phase 2 — Qwen + embeddings on the Cite task

**Setup.** Feed chunk embeddings (×4 expansion) into Qwen; ask the model to emit a **cited subsequence** of the 128-token source. Two variants:
- **Plain Cite** — copy/extract an arbitrary contiguous span out of the chunk.
- **Masked Cite** — mask the cite span in the encoder input, encode, then ask Qwen to recover the *unmasked* span (Masked Cite → Encoder → Qwen → Unmasked Cite).

**Results.**
- **Plain Cite: very strong.** The model extracts essentially any requested subsequence out of a single 128-token chunk almost perfectly. This proves the soft-token interface works and that Qwen can read positionally-addressed content out of the expanded embedding.
- **Masked Cite: noticeably weaker.** Convergence is real and loss drops, but recovery of the masked span is imprecise. The likely cause is the **128-token ceiling**: the masked span must be inferred from a *single* chunk's embedding that itself had the span removed, leaving little redundancy to reconstruct from.

**Interpretation.** Phase 2 separates two skills: *copying* content that is present in the embedding (easy) vs. *inferring* content that was removed (hard). The interface is fine; the bottleneck is information capacity and the reconstruction-style geometry of the embedding.

### Phase 3 — QnA training (the failure mode)

**Setup.** Aggregate QnA datasets (SQuAD v2, NaturalQuestions, TriviaQA, NewsQA, QuAC, CoQA, MRQA, AdversarialQA), `QnaAns` / `QnaAnsCite` regimes, Qwen2.5-1.5B decoder, `emb_exp_rate=4`, `emb_win 2×6`, `max_seq_len=400`. Numbers from [s_03_12_eval_mixed_decoder.md](../../s_03_12_eval_mixed_decoder.md).

**Symptoms.**
- Train loss keeps decreasing from ~epoch 2 onward; **validation stagnates early**.
- The model often answers from Qwen's parametric priors rather than from the context embeddings. The diagnostic **Apono/Hypoon** example (renamed entities so priors cannot help — "Who was the first person to walk on the Hypoon?" → should be *Nelly Armweak*) is **not** answered correctly: specific, context-only information is not extracted.

**Representative numbers** (Run 2, fixed `QnaAns` indexing):

| Dataset | Train loss | Val loss | Gap (val−train) |
|---|---:|---:|---:|
| squad_v2 | 1.072 | 2.499 | **+1.427** |
| natural_questions | 0.917 | 1.561 | +0.644 |
| triviaqa | 1.893 | 2.607 | +0.713 |
| newsqa | 1.264 | 2.550 | **+1.286** |
| mrqa | 0.533 | 1.198 | +0.664 |
| adversarialqa | 1.243 | 2.939 | **+1.696** |
| quac | 2.098 | 2.469 | +0.371 |
| coqa | 2.457 | 2.502 | +0.044 |
| **Overall** | **1.435** | **2.291** | **+0.856** |

A later `QnaAnsCite` checkpoint (Run 3) *narrows* the gap (overall +0.422) by **raising train loss** (multi-task regularization), not by lowering val loss meaningfully — i.e., it regularizes rather than teaches extraction.

**Confounds flagged in the eval report** (must be controlled before drawing architecture conclusions): an aggregator indexing bug that leaked unanswerable rows, train/val composition mismatch, dataset-difficulty imbalance (easy `mrqa` dominates; hard `adversarialqa`/`newsqa` carry the gap), and high-variance validation (80-batch sampling). See the "Ranked Flaws" section of the eval report.

**Diagnosis.** Three compounding causes:
1. **Capacity** — answer-critical tokens (names, numbers, dates) are exactly what a reconstruction-trained gist discards first; a 128:1 single-vector bottleneck cannot reliably preserve them.
2. **Cheating path** — with the decoder fully trainable and the question already in the prompt, the cheapest gradient route is to memorize question→answer regularities in Qwen's weights. Many QnA answers are guessable from question + priors, so the signal that *forces* grounding is weak. This is precisely why val stagnates while train improves.
3. **Objective mismatch** — reconstruction geometry ≠ queryable geometry. Nothing in the QnA CE loss makes the embedding *addressable by a question*.

---

## 3. Analysis of related work

The MixedDecoder problem sits at the intersection of six research lines: **(A) multimodal soft-token bridges**, **(B) text-into-embedding context compression**, **(C) retrieval / late-interaction representations**, **(D) knowledge distillation**, **(E) memory-augmented / recurrent compression**, and **(F) soft-prompt / prefix conditioning**. Each contributes a concrete, transferable design lesson. This section is deliberately detailed because the choice of *next* direction depends on understanding precisely *why* each prior system works.

### 3.0 Framing: what "extract information from an embedding" actually requires

Three orthogonal properties decide whether a compressed representation is useful for downstream extraction. Keeping them separate clarifies every paper below.

- **Capacity** — how many bits the representation can hold. Upper-bounded by `(#vectors × d × effective_bits_per_dim)`. MixedDecoder's single 768-d `[CLS]` per 128 tokens is the binding constraint; linear ×4 expansion does **not** raise it (a deterministic linear map cannot add information).
- **Addressability** — whether a *query* (the question) can selectively route to the right stored content. A gist optimized for reconstruction is organized by *surface-form likelihood*, not by *query relevance*; these are different geometries. Cross-attention with learned queries (Perceiver/Q-Former) and per-token late interaction (ColBERT) are the two canonical ways to build addressable representations.
- **Forcing function** — whether the *training signal* makes the model actually use the representation rather than a shortcut. With a trainable decoder and guessable targets, the model bypasses the representation entirely. Freezing the LLM and using unguessable / distilled targets are the two standard forcing functions.

MixedDecoder's Phase 3 fails on all three: low capacity, reconstruction-not-query geometry, and a weak forcing function. Every recommendation in §4 maps back to one of these axes.

### 3.1 Multimodal soft-token bridges (the closest analogy)

> **Thread:** [Multimodal soft-token bridges](multimodal/multimodal.md) — evolution table and MixedDecoder-specific notes.

The user's mental model — "do with text what models do with images" — is exactly the VLM bridge problem, and the literature is unambiguous about what makes it work.

**Flamingo** ([local recap](../papers/multimodal_2022_flamingo-perceiver-resampler.md), Alayrac et al. 2022). A frozen vision encoder produces a variable-length feature grid; a **Perceiver Resampler** maps it to a *fixed, small* set of latent tokens (e.g. 64) via cross-attention from learned latent queries; the frozen LM ingests them through **gated cross-attention** inserted between its existing layers. The `tanh`-gating is initialized at zero so the pretrained LM is unperturbed at step 0 and the visual pathway is introduced gradually. **Four lessons:** (i) the bridge emits *many* tokens, not one; (ii) the LM is **frozen**, forcing the bridge to do the representation work; (iii) conditioning enters at *every* layer via cross-attention, not only as a prefix; (iv) zero-init gating stabilizes the cold-start of a new modality.

**Perceiver / Perceiver IO** (Jaegle et al. 2021). The architectural ancestor: a fixed set of latent vectors cross-attends to an arbitrarily large input array, decoupling compute from input size. The key idea reused everywhere downstream is *learned latent queries as an information bottleneck* — exactly the component MixedDecoder is missing between BERT and Qwen.

**BLIP-2 / Q-Former** ([local recap](../papers/multimodal_2023_blip2-qformer.md), Li et al. 2023). The single most relevant paper. A small Querying Transformer holds **K = 32 learned query embeddings** that cross-attend to the frozen image features and emit 32 soft tokens to a frozen LLM. Three design points map directly onto MixedDecoder's failure:
- **Multiple independent queries.** 32 queries each attend to the *whole* input and specialize — unlike MixedDecoder's single `[CLS]` blown up linearly into 4 rank-≤1 copies of the same vector. This is the *addressability* fix.
- **Two-stage curriculum that separates *extract* from *express*.** Stage 1 trains the bridge with representation losses (ITC/ITM/ITG) against the encoder *only*; Stage 2 bridges to the frozen LLM. BLIP-2 explicitly argues that forcing the bridge to learn *what to extract* and *how to express it* simultaneously (Flamingo's single LM loss) is sample-inefficient — exactly the regime MixedDecoder's Phase 3 is stuck in.
- **Masking patterns as objective control.** The three Stage-1 objectives differ only by the self-attention mask over queries↔text, showing how much can be taught with one architecture and a careful mask schedule.

**LLaVA / LLaVA-1.5** (Liu et al. 2023). The counterpoint: it keeps **all** patch tokens (no learned-query bottleneck) and a simple MLP (later 2-layer GELU) projector, and fine-tunes the LLM with visual instruction data. It works because it does *not* compress — it pays full token cost. The lesson for MixedDecoder: if you insist on heavy compression you need Q-Former-style machinery; a linear/MLP projector alone (LLaVA-style) only works at *low* compression. MixedDecoder currently uses a LLaVA-grade projector at a Q-Former-grade compression ratio — the worst of both.

**Honeybee** (Cha et al. 2023) and **Q-Former ablations.** Later VLM work (e.g. Honeybee's "locality-enhanced projector") finds that abstractor design — whether the bridge preserves *spatial/positional locality* — measurably affects downstream fine-grained tasks (OCR, counting). The text analog: a bridge that preserves *token-order locality* should help recall of specific spans (names/numbers). This argues for slot-per-region pooling rather than a single global `[CLS]`.

> **Takeaway for MixedDecoder:** replace single-`[CLS]` + linear expansion with a *real resampler* (K learned queries cross-attending to the encoder's full 128-token hidden states), freeze/LoRA the decoder, and consider layer-wise cross-attention (Flamingo) instead of prefix-only conditioning.

### 3.2 Text-into-embedding context compression (the direct prior art)

This is the text analog of the image bridge and the most directly applicable body of work. The throughline: **freeze the LLM, emit multiple memory slots, and pretrain with reconstruction + continuation objectives — not task CE alone.** Typical working compression ratios are **4×–16×**, far below MixedDecoder's 128×.

**Gisting** (Mu, Li, Goodman 2023). Trains *gist tokens* that compress a prompt into a few KV slots using only an attention-mask modification (everything after the gist tokens cannot attend to the original prompt, forcing the gist to carry it), LLM frozen. Demonstrates up to ~26× prompt compression with minimal quality loss — but for *instructions/prompts*, which are low-entropy and reusable, not arbitrary factual context. Capacity for fine-grained facts degrades as ratio grows, foreshadowing MixedDecoder's recall failures.

**AutoCompressor** (Chevalier et al. 2023). Recursively summarizes long context into *summary vectors* using the LM itself, with an unsupervised next-token objective and segment-level recursion (summary vectors of one segment are prepended to the next). Shows summary vectors extend effective context cheaply, but with smooth degradation — consistent with MixedDecoder's "copy works, recall of specifics fails." Also demonstrates that the *same* model can both produce and consume the compressed vectors, an alternative to MixedDecoder's separate-BERT design.

**ICAE — In-Context Autoencoder** (Ge et al. 2023). The single closest architecture to MixedDecoder: a LoRA-adapted encoder (the LLM itself in encoder mode) compresses context into a small number of **memory slots**, and the *same frozen LLM* decodes. **Crucially pretrained with autoencoding (reconstruct the context) + text continuation, then instruction-tuned.** This is almost exactly the user's intuition, but with three differences that each matter: (i) multiple slots, not one; (ii) **frozen** decoder; (iii) reconstruction kept as a first-class objective throughout, not just as a Phase-1 initializer that is then discarded. ICAE reports ~4× compression with high fidelity and explicitly frames the memory slots as a learned, queryable representation.

**xRAG** (Cheng et al. 2024). Compresses a retrieved document to **one** token via a projector over a frozen dense-retriever embedding, and trains with **self-distillation**: a teacher LLM sees the full document text, the student sees the single embedding, and a KL loss aligns their next-token distributions. This is precisely the user's "Idea 1," validated at scale — the KL term is the dense, per-token signal that *pushes* information into the embedding (the *forcing function*). xRAG works for RAG answer generation but is explicitly extreme (1 token) and trades away fine detail; it succeeds partly because the projected vector starts from a *retrieval-optimized* (addressable) embedding rather than a reconstruction-optimized one.

**COCOM — Context Compression** (Rau et al. 2024). Compresses context into a handful of embeddings for QA and **ablates the compression-ratio/accuracy curve directly** — the empirical map MixedDecoder is missing. Confirms accuracy falls monotonically with ratio and that several embeddings beat one; also studies online vs. offline compression cost trade-offs relevant to a RAG deployment.

**500xCompressor** (Li et al. 2024). Pushes ratios to extremes (up to 480×) by compressing into the *KV cache* rather than input embeddings, and quantifies the steep capacity wall: extraction/QA accuracy degrades gracefully for "gist"-style queries but collapses for verbatim recall — a direct empirical statement of MixedDecoder's ceiling.

**CEPE — Context Expansion with Parallel Encoding** (Yen et al. 2024). Uses a small parallel encoder to process long context in chunks and feeds the chunk representations into a frozen decoder via added **cross-attention layers**. Architecturally the closest "encoder feeds frozen decoder via cross-attention" design to a Flamingo-for-text, and a concrete blueprint for §4.6.

**LLMLingua / LongLLMLingua / LLMLingua-2** (Jiang et al. 2023–2024). The *discrete* alternative: a small LM scores token informativeness and **drops** low-information tokens rather than compressing to vectors. Useful as (i) a strong, training-free baseline to beat, and (ii) a reminder that "compression" need not be continuous. They cannot reach sub-token density, but they never lose verbatim fidelity for the tokens they keep — the opposite failure profile to MixedDecoder.

**Soft-prompt compression lineage.** The continuous-vector compression idea descends from prompt-tuning (§3.6); ICAE, Gisting, and AutoCompressor are best read as "learned soft prompts that *encode a specific context*" rather than a task.

> **Takeaway:** MixedDecoder operates at ~128× where this literature operates at 4–16×, with a *trainable* decoder where this literature *freezes* it, and with task-CE where this literature uses **autoencoding + continuation + distillation**. All three gaps point the same direction, and COCOM/500xCompressor give the ratio curve to choose an operating point.

### 3.3 Retrieval and late-interaction representations

These inform *how many* vectors and *what geometry* preserve queryable information — the *addressability* axis.

**ColBERT / ColBERTv2** ([local recap](../papers/retrieval_2020_colbert-late-interaction.md), Khattab & Zaharia 2020; Santhanam et al. 2021). The defining lesson: a *single* pooled vector per passage loses fine-grained term information; **late interaction** keeps **one vector per token** and matches query terms against passage terms individually (MaxSim), dramatically improving retrieval — especially for queries hinging on a specific entity or number. This is the strongest external evidence that MixedDecoder's single-`[CLS]` design is the core limitation: a content-addressable *bag of K vectors* is far more queryable than one gist of the same total width. ColBERTv2 adds residual compression, showing the per-token vectors can themselves be quantized cheaply.

**DPR** (Karpukhin et al. 2020) and **Sentence-BERT** (Reimers & Gurevych 2019). The single-vector dense baselines ColBERT improves upon: excellent for *topical* similarity, weak for *specific-token* recall — precisely MixedDecoder's symptom (good gist, poor names/numbers).

**Contriever / GTR / E5 / BGE** (Izacard et al. 2021; Ni et al. 2021; Wang et al. 2022; Xiao et al. 2023). Modern single-vector encoders trained with large-scale contrastive learning. The relevant lesson is *training signal*: contrastive objectives produce embeddings organized by *retrievability* (addressability), unlike a reconstruction autoencoder. Initializing or co-training MixedDecoder's encoder with a contrastive/retrieval objective could move the embedding geometry toward "queryable."

**ColBERT-style late interaction inside an LLM bridge** is an open design: emit K per-region vectors *and* let the decoder's question tokens do MaxSim-like selective attention over them. MixedDecoder already has the cross-attention substrate (the decoder attends to soft tokens); the missing piece is having *enough, distinct* soft tokens to attend to.

> **Takeaway:** number of slots and their *independence/locality* matter more than total parameter count or vector width. K content-addressable vectors ≫ one gist of equal width; contrastive/retrieval pretraining biases the geometry toward addressability.

### 3.4 Knowledge distillation (the forcing function)

**Hinton, Vinyals, Dean 2015** (soft-target KD) and **Kim & Rush 2016** (sequence-level KD) are the basis for the teacher–student recipe the user proposed. **DistilBERT** (Sanh et al. 2019) and **TinyBERT** (Jiao et al. 2020) show distillation transfers most of a model's competence at a fraction of the size, and that *intermediate-feature* matching (hidden states, attention maps) accelerates it.

Applied here (the user's Ideas 1 & 3): a **frozen full-context teacher** (Qwen sees real context tokens) supervises a **student** (Qwen sees context *embeddings*) via per-token KL on continuations. Why this is the right forcing function:
- **Density.** KL is computed at *every* continuation position, not just on a short answer span — orders of magnitude more signal than QnA CE.
- **Unfakeable when grounded.** If the continuation depends on context-only facts, the student can match the teacher's distribution *only* by having preserved those facts in the embeddings. This directly closes the Phase-3 cheating path.
- **Optionally feature-level.** Beyond output KL, matching the teacher's hidden states at the answer positions (TinyBERT-style) gives an even denser target and is cheap when teacher and student share the architecture (same Qwen).

A subtlety: distillation only forces grounding to the extent the *teacher itself* relies on the context. For facts the teacher already knows parametrically, the KL target is also guessable. Hence distillation must be paired with **context-only / renamed-entity data** (§5) so the teacher, too, must read the context.

### 3.5 Memory-augmented and recurrent compression (adjacent architectures)

These offer an alternative to the encoder-bridge route: instead of a separate BERT, let the decoder maintain its own compressed memory.

**Compressive Transformer** (Rae et al. 2019). Compresses old activations into a coarser memory via a learned pooling, extending range with graceful forgetting — an explicit study of *what* to keep when compressing past context.

**Memorizing Transformers** (Wu et al. 2022). Adds a non-differentiable kNN lookup over a large external memory of past (key, value) pairs, read by one attention head. Shows verbatim recall benefits from *exact* stored vectors — a counterpoint to lossy compression and an argument for keeping some per-token vectors (cf. ColBERT).

**Recurrent Memory Transformer / RMT** (Bulatov et al. 2022) and **Block-Recurrent Transformers** (Hutchins et al. 2022). Carry a small set of memory tokens across segments, reading and rewriting them recurrently. Architecturally similar to AutoCompressor and a candidate if MixedDecoder needs to compress *multi-chunk* documents rather than independent 128-token chunks.

**AutoCompressor** (already in §3.2) is the bridge between this line and text compression.

> **Takeaway:** if verbatim recall of specific tokens is the priority, a hybrid — lossy gist for topical context *plus* a few exact/per-token vectors for entities — matches the empirical lesson from Memorizing Transformers and ColBERT.

### 3.6 Soft-prompt and prefix conditioning (the mechanism MixedDecoder uses)

MixedDecoder conditions a frozen-or-tuned decoder on *continuous vectors in the input-embedding slot* — mechanically identical to soft prompting.

**Prefix-Tuning** (Li & Liang 2021) and **Prompt-Tuning** (Lester et al. 2021) show that a handful of trained continuous vectors can steer a frozen LLM to a whole task, and that *more* soft tokens help up to a point — direct evidence that the *number* of context vectors matters and that a frozen LLM can be driven entirely through the embedding slot. **P-Tuning v2** (Liu et al. 2021) adds prefixes at *every* layer (like Flamingo's per-layer conditioning) and closes the gap to full fine-tuning on hard tasks — an argument for layer-wise (not prefix-only) injection of the context vectors. The distinction from MixedDecoder: soft prompts are *task-constant*, whereas MixedDecoder's vectors are *input-dependent* (produced by the encoder per chunk) — i.e. MixedDecoder is "input-conditioned soft prompting," which is exactly ICAE/Gisting.

### 3.7 Synthesis — what the literature collectively prescribes

| Problem in MixedDecoder | Axis (§3.0) | Field consensus | Source lines |
|---|---|---|---|
| One `[CLS]` per chunk | Capacity + Addressability | Emit **K independent slots** (learned queries / per-token) | §3.1 BLIP-2, §3.3 ColBERT |
| Linear ×4 "expansion" adds no info | Capacity | Use **cross-attention resampler** over full hidden states | §3.1 Perceiver/Q-Former |
| Reconstruction ≠ queryable | Addressability | Contrastive/retrieval pretraining; learned queries | §3.3 ColBERT/Contriever |
| Decoder fully trainable → cheats | Forcing function | **Freeze or LoRA** the decoder | §3.1, §3.2 (ICAE/xRAG/Gisting) |
| Task-CE only | Forcing function | Add **reconstruction + continuation + KL distillation** | §3.2 ICAE/xRAG, §3.4 |
| 128× ratio | Capacity | Operate at **4–16×**; map the ratio/accuracy curve | §3.2 COCOM/500x |
| Prefix-only conditioning | Addressability | Consider **layer-wise cross-attention** | §3.1 Flamingo, §3.6 P-Tuning v2 |
| Reconstruction discarded after Phase 1 | Forcing function | Keep AE **on** as a permanent auxiliary | §3.2 ICAE |

---

## 4. Architectural changes and experiments

Ordered by expected impact-to-effort. Each is a discrete, testable change, tagged with the §3.0 axis it targets.

### 4.1 Replace the bottleneck with a learned-query resampler *(highest impact; capacity + addressability)*

Swap single-`[CLS]` + `emb_exp` linear for a small **Q-Former / Perceiver head**: K learned queries cross-attend to the encoder's **full 128 token hidden states**, emitting K genuinely independent soft tokens per chunk.

- Why: today's 4 expanded vectors are a rank-≤1 image of one vector (no added information). K queries attending to different spans add real capacity *and* addressability (§3.1, §3.3).
- Experiment: sweep `K ∈ {4, 8, 16, 32}`; compare passkey/MQAR recall (§5) and QnA val loss vs. the linear-expansion baseline at matched token budget.
- Locality variant (§3.1 Honeybee): also try K = (#regions) average/attention pools over strided spans so each slot owns a contiguous region — expected to help name/number recall.
- Cheaper interim: pool the encoder's *sequence* (not just `[CLS]`) into K attention-pooled or strided vectors, so the K outputs depend on different input regions.

### 4.2 Freeze or LoRA the decoder *(highest impact, lowest effort; forcing function)*

Add a `freeze_decoder` path mirroring `freeze_encoder`, or attach LoRA to Qwen and train only {bridge, LoRA}.

- Why: removes the memorize-in-weights cheating path that produces "train improves, val stagnates" (§2 Phase 3, §3.2). Every comparable system freezes the LLM.
- Experiment: 2×2 — {decoder frozen, LoRA, full} × {linear expansion, resampler}. Track the val−train gap; expect it to shrink because the only loss-reducing route becomes "read the embeddings."

### 4.3 Multi-objective training (reconstruction + continuation + distillation) *(forcing function)*

Train the bridge with a weighted sum:

$$\mathcal{L} = \lambda_{\text{recon}}\,\mathcal{L}_{\text{AE}} + \lambda_{\text{cont}}\,\mathcal{L}_{\text{LM}} + \lambda_{\text{KL}}\,\mathcal{L}_{\text{distill}} + \lambda_{\text{task}}\,\mathcal{L}_{\text{QnA}}$$

- $\mathcal{L}_{\text{AE}}$ — regenerate the chunk from its embeddings (Phase 1 already converges; keep it **on** as a capacity guarantee). §3.2 ICAE.
- $\mathcal{L}_{\text{distill}}$ — per-token KL vs. a **frozen full-context Qwen teacher** on context-grounded continuations. §3.2 xRAG, §3.4. This is the user's Idea 1, and the dense signal that *forces* information into the embeddings. Optionally add TinyBERT-style hidden-state matching at answer positions.
- Curriculum (BLIP-2 style, §3.1): **Stage 1** representation (AE + continuation + distill, *no* QnA) → **Stage 2** QnA bridging.

**How to force reading from embeddings, not previous tokens** (the user's explicit question): make the continuation/answer depend on information that exists *only* in the context — context-grounded cloze, renamed entities (the Apono/Hypoon trick at scale), and KL against the full-context teacher. If the target is recoverable from priors, no architecture will force grounding; the *data* must remove the shortcut (§5).

### 4.4 Lower and instrument the compression ratio *(capacity)*

Run at **4–16×** (smaller chunk `inp_len`, or more slots per chunk) and **plot the ratio/accuracy curve** (COCOM/500xCompressor-style, §3.2). MixedDecoder currently sits far past where the literature degrades; you need the curve to choose an operating point.

### 4.5 Bias the encoder geometry toward addressability *(addressability)*

Before/while bridging, co-train the encoder with a **contrastive retrieval** objective (in-batch negatives, question↔chunk) so its embeddings are organized by *retrievability* rather than reconstruction (§3.3 Contriever/ColBERT). Even a light auxiliary contrastive loss should improve query-conditioned routing.

### 4.6 Expose context to layer-wise cross-attention *(addressability)*

Instead of (or in addition to) prefixing soft tokens, add Flamingo/CEPE-style **gated cross-attention** so the decoder *re-reads* the compressed memory at every layer (§3.1, §3.2 CEPE, §3.6 P-Tuning v2), with zero-init gates for stable cold-start. More invasive; keep as a later experiment if prefix-only saturates.

### 4.7 Hybrid lossy-gist + exact-slot memory *(capacity for verbatim recall)*

Keep the lossy gist for topical context **and** attach a few exact/per-token vectors for entity-dense spans (§3.3 ColBERT, §3.5 Memorizing Transformers). Targets the specific failure of names/numbers without paying full-token cost everywhere.

### 4.8 Smaller, faster decoder for the ablation loop *(velocity)*

Qwen2.5-1.5B fully fine-tuned is expensive and confounds the diagnosis. Run §4.1–4.5 ablations on **GPT-2 / Qwen-0.5B frozen + LoRA** first; port the winner to 1.5B.

### 4.9 Suggested experiment ladder

1. **Diagnostics first** (§5): passkey + MQAR + key-value recall harness → quantify current capacity.
2. **4.2** decoder freeze/LoRA on small decoder → measure gap change.
3. **4.1** resampler head, sweep K → measure recall change.
4. **4.3** add reconstruction + distillation → measure grounding (Apono/Hypoon-style eval).
5. **4.5** contrastive encoder pretraining → measure query-conditioned recall.
6. **4.4** ratio sweep → pick operating point.
7. Port winner to Qwen2.5-1.5B; revisit **4.6 / 4.7** if needed.

---

## 5. Datasets for information retrieval from embeddings

The property to optimize for: **the target is provably in the context and not guessable from the decoder's priors**, ideally with a **dense (per-token) signal**. QnA fails both (short target, often guessable). Grouped from best *diagnostic* to best *at-scale training*.

### 5.1 Synthetic recall probes — best for *diagnosis*

Unguessable by construction; let you measure exactly how much survives compression.

- **Passkey / needle-in-a-haystack** (Mohtashami & Jaggi 2023; Kamradt). Bury a random key in filler, ask for it. Pure "did the embedding keep this token." Trivial to generate from your wiki corpus.
- **MQAR — Multi-Query Associative Recall** (Arora et al., *Zoology* 2023). Random key→value list; query several values. The standard compression/long-context probe; stresses *selective* recall.
- **Key-value / JSON field recall.** Encode `{"name": …, "id": 7731, …}`, ask one field. Difficulty scales smoothly with #fields → directly maps capacity to `emb_exp_rate` × chunk size.
- **RULER** (Hsieh et al. 2024). A configurable synthetic long-context battery (multi-hop, aggregation, tracing, variable tracking) — a ready-made capacity harness.
- **bAbI** (Weston et al. 2015), **CLUTRR** (Sinha et al. 2019). Synthetic reasoning over *stated* facts; priors cannot help.

> Use these as a **regression harness** before and after every architecture change (§4.9 step 1).

### 5.2 Cloze / masked-span recovery — dense, self-supervised, unguessable

Highest-leverage *training* family; free from your existing wiki pipeline; loss on every position. Generalizes your `Cite` task.

- **LAMBADA** (Paperno et al. 2016). Predict a final word resolvable *only* from broader context (designed so local n-gram priors fail).
- **Children's Book Test** (Hill et al. 2015). Cloze on named entities / nouns; the NE subset is extraction-heavy.
- **CNN/DailyMail cloze** (Hermann et al. 2015). Entities anonymized as `@entityN` — anonymization deliberately removes priors, the scaled version of your Apono/Hypoon trick.
- **WikiText / your own wiki, salient-span masked.** Mask entities/numbers/dates and recover them through the bottleneck. Bias masking toward NER/number spans to maximize extraction pressure. (Closest to your existing Masked Cite, but over multiple chunks for redundancy.)

### 5.3 Structured extraction — explicit "pull spans/relations out"

Targets are spans or tuples grounded in the input; little room for guessing. Pairs naturally with the distillation-teacher idea (§4.3).

- **NER:** CoNLL-2003, OntoNotes 5.0, WNUT-17 (rare entities → low priors).
- **Relation extraction:** TACRED (Zhang et al. 2017), DocRED (Yao et al. 2019, document-level/multi-hop), FewRel.
- **Coreference:** OntoNotes, GAP, WSC.
- **Text-to-SQL:** WikiSQL (Zhong et al. 2017), Spider (Yu et al. 2018) — schema/values live in context, output is checkable.
- **Slot filling:** ATIS, SNIPS, MIT Movie/Restaurant.

### 5.4 Counterfactual / evidence-grounded verification — best anti-cheating signal

- **VitaminC** (Schuster et al. 2021). *Purpose-built* so near-identical claims get **opposite** labels depending on a small evidence revision — a model ignoring evidence cannot beat chance. The single best dataset for *forcing* grounding; directly attacks the Phase-3 stagnation.
- **FEVER** (Thorne et al. 2018). Claim + evidence → SUPPORTS/REFUTES/NEI. Large; weaker than VitaminC (some claims guessable).

### 5.5 Full-information-preservation objectives

Require *all* of the context to survive, not just an answer span.

- **Context reconstruction / autoencoding** — regenerate the passage from its embeddings. Already converges (Phase 1); keep as a permanent auxiliary (strongest guarantee information is *present*). §3.2 ICAE.
- **Data-to-text:** ToTTo (Parikh et al. 2020, controlled/extractive), WikiBio, DART, E2E. Hallucination = loss.
- **Machine translation** — forces near-lossless semantic preservation (mixes in a translation skill; use as stress test).

### 5.6 Recommended starting mix

1. **Train:** salient-span cloze on your wiki (extends `Cite`, free, dense) **+ VitaminC** (counterfactual grounding).
2. **Diagnostics:** passkey + MQAR + key-value recall, swept over `emb_exp_rate` and chunk length.
3. **Auxiliary:** context reconstruction always on.

The unifying principle: **the cheapest path to low loss must run *through* the embeddings.** Cloze-with-anonymization (§5.2) and VitaminC (§5.4) enforce it most directly; the synthetic probes (§5.1) measure whether it is happening.

---

## 6. References

### 6.1 Multimodal soft-token bridges

**Thread:** [Multimodal soft-token bridges](multimodal/multimodal.md) (Perceiver · Flamingo · BLIP-2/Q-Former · LLaVA · InstructBLIP · Honeybee)

- Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning.* arXiv:2204.14198, 2022. ([local recap](../papers/multimodal_2022_flamingo-perceiver-resampler.md))
- Li et al. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.* arXiv:2301.12597, ICML 2023. ([local recap](../papers/multimodal_2023_blip2-qformer.md))
- Jaegle et al. *Perceiver: General Perception with Iterative Attention.* arXiv:2103.03206, 2021.
- Jaegle et al. *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* arXiv:2107.14795, 2021.
- Liu et al. *Visual Instruction Tuning (LLaVA).* arXiv:2304.08485, 2023.
- Liu et al. *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5).* arXiv:2310.03744, 2023.
- Dai et al. *InstructBLIP.* arXiv:2305.06500, 2023.
- Cha et al. *Honeybee: Locality-enhanced Projector for Multimodal LLM.* arXiv:2312.06742, 2023.

### 6.2 Text context compression into embeddings
- Mu, Li, Goodman. *Learning to Compress Prompts with Gist Tokens.* arXiv:2304.08467, 2023.
- Chevalier et al. *Adapting Language Models to Compress Contexts (AutoCompressor).* arXiv:2305.14788, 2023.
- Ge et al. *In-context Autoencoder for Context Compression in a Large Language Model (ICAE).* arXiv:2307.06945, 2023.
- Cheng et al. *xRAG: Extreme Context Compression for RAG with One Token.* arXiv:2405.13792, 2024.
- Rau et al. *Context Embeddings for Efficient Answer Generation in RAG (COCOM).* arXiv:2407.09252, 2024.
- Li et al. *500xCompressor: Generalized Prompt Compression for LLMs.* arXiv:2408.03094, 2024.
- Yen et al. *Long-Context Language Modeling with Parallel Context Encoding (CEPE).* arXiv:2402.16617, 2024.
- Jiang et al. *LLMLingua: Compressing Prompts for Accelerated Inference of LLMs.* arXiv:2310.05736, 2023.
- Jiang et al. *LongLLMLingua.* arXiv:2310.06839, 2023.
- Pan et al. *LLMLingua-2.* arXiv:2403.12968, 2024.

### 6.3 Retrieval and late interaction
- Khattab & Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* arXiv:2004.12832, 2020. ([local recap](../papers/retrieval_2020_colbert-late-interaction.md))
- Santhanam et al. *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* arXiv:2112.01488, 2021.
- Karpukhin et al. *Dense Passage Retrieval for Open-Domain QA (DPR).* arXiv:2004.04906, 2020.
- Reimers & Gurevych. *Sentence-BERT.* arXiv:1908.10084, 2019.
- Izacard et al. *Unsupervised Dense Information Retrieval (Contriever).* arXiv:2112.09118, 2021.
- Ni et al. *Large Dual Encoders Are Generalizable Retrievers (GTR).* arXiv:2112.07899, 2021.
- Wang et al. *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).* arXiv:2212.03533, 2022.
- Xiao et al. *C-Pack / BGE: Packed Resources for General Chinese Embeddings.* arXiv:2309.07597, 2023.

### 6.4 Knowledge distillation
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015.
- Kim & Rush. *Sequence-Level Knowledge Distillation.* arXiv:1606.07947, 2016.
- Sanh et al. *DistilBERT.* arXiv:1910.01108, 2019.
- Jiao et al. *TinyBERT: Distilling BERT for Natural Language Understanding.* arXiv:1909.10351, 2020.

### 6.5 Memory-augmented and recurrent compression
- Rae et al. *Compressive Transformers for Long-Range Sequence Modelling.* arXiv:1911.05507, 2019.
- Wu et al. *Memorizing Transformers.* arXiv:2203.08913, 2022.
- Bulatov et al. *Recurrent Memory Transformer.* arXiv:2207.06881, 2022.
- Hutchins et al. *Block-Recurrent Transformers.* arXiv:2203.07852, 2022.

### 6.6 Soft-prompt and prefix conditioning
- Li & Liang. *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* arXiv:2101.00190, 2021.
- Lester et al. *The Power of Scale for Parameter-Efficient Prompt Tuning.* arXiv:2104.08691, 2021.
- Liu et al. *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks.* arXiv:2110.07602, 2021.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021.

### 6.7 Diagnostic / probing datasets
- Mohtashami & Jaggi. *Landmark Attention (passkey retrieval).* arXiv:2305.16300, 2023.
- Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models (MQAR).* arXiv:2312.04927, 2023.
- Hsieh et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* arXiv:2404.06654, 2024.
- Weston et al. *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks (bAbI).* arXiv:1502.05698, 2015.
- Sinha et al. *CLUTRR.* arXiv:1908.06177, 2019.
- Paperno et al. *The LAMBADA Dataset.* arXiv:1606.06031, 2016.
- Hill et al. *The Goldilocks Principle (Children's Book Test).* arXiv:1511.02301, 2015.
- Hermann et al. *Teaching Machines to Read and Comprehend (CNN/DailyMail cloze).* arXiv:1506.03340, 2015.

### 6.8 Extraction / structured-output / verification datasets
- Rajpurkar et al. *Know What You Don't Know: Unanswerable Questions for SQuAD (SQuAD 2.0).* arXiv:1806.03822, 2018.
- Tjong Kim Sang & De Meulder. *Introduction to the CoNLL-2003 Shared Task: Language-Independent NER.* 2003.
- Pradhan et al. *Towards Robust Linguistic Analysis using OntoNotes (CoNLL-2012).* 2013.
- Zhang et al. *Position-aware Attention and Supervised Data Improve Slot Filling (TACRED).* EMNLP 2017.
- Yao et al. *DocRED: A Large-Scale Document-Level Relation Extraction Dataset.* arXiv:1906.06127, 2019.
- Han et al. *FewRel.* arXiv:1810.10147, 2018.
- Zhong et al. *Seq2SQL / WikiSQL.* arXiv:1709.00103, 2017.
- Yu et al. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Text-to-SQL.* arXiv:1809.08887, 2018.
- Parikh et al. *ToTTo: A Controlled Table-to-Text Generation Dataset.* arXiv:2004.14373, 2020.
- Thorne et al. *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* arXiv:1803.05355, 2018.
- Schuster et al. *Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence (VitaminC).* arXiv:2103.08541, NAACL 2021.

### 6.9 Core architecture components (in-repo recaps)
- Vaswani et al. *Attention Is All You Need.* arXiv:1706.03762, 2017. ([local recap](../papers/attention_2017_transformer.md))
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864, 2021. ([local recap](../papers/positional_2021_rope-roformer.md))
- Zhang & Sennrich. *Root Mean Square Layer Normalization.* arXiv:1910.07467, 2019. ([local recap](../papers/attention_2019_rmsnorm.md))
- Shazeer. *GLU Variants Improve Transformer (SwiGLU).* arXiv:2002.05202, 2020. ([local recap](../papers/attention_2020_swiglu.md))
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* arXiv:2305.13245, 2023. ([local recap](../papers/attention_2023_gqa.md))

### 6.10 Internal sources
- [mllm/model/mixed_decoder.py](../../mllm/model/mixed_decoder.py) — model implementation.
- [mllm/config/model.py](../../mllm/config/model.py) — `MixedDecoderCfg`, `EncBertCfg`, dataset/type enums.
- [s_03_11_train_mixed_decoder.py](../../s_03_11_train_mixed_decoder.py) — training driver.
- [s_03_12_eval_mixed_decoder.md](../../s_03_12_eval_mixed_decoder.md) — QnA evaluation report (loss/PPL/gap numbers in §2).
