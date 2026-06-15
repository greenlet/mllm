# Auto‑generated structured‑extraction datasets — a plan

> How to extend MixedDecoder's existing `Cite` extraction task into a family of
> **query‑conditioned structured‑extraction generators** (JSON field, JSONata, XML/XPath,
> SQL `SELECT`, key‑value recall, …) whose sole purpose is to **force the decoder to read
> the answer out of the context embeddings** rather than from Qwen's parametric priors.
>
> Companion to [mixed_decoder.md](../mixed_decoder.md) (diagnosis: capacity / addressability /
> forcing‑function) and [interactive_extractor.md](../interactive_extractor.md) (the
> `Expand → Visit → Pool` bridge that consumes these data). This doc answers the two
> questions raised there: **(1)** a concrete generation plan per family — what data, and
> whether to auto‑generate or pull from existing corpora; **(2)** what prior art exists for
> learning data extraction, including the more natural‑language‑oriented lines.

---

## 0. Why structured extraction is the right forcing function

From [mixed_decoder.md](../mixed_decoder.md) §5 and §3.0, a training target is useful only if
the cheapest path to low loss runs **through** the compressed representation. Structured
extraction maximizes all three axes simultaneously:

| Property | Why structured extraction has it |
|---|---|
| **Unguessable / grounded** (capacity) | A random `"id": 7731` or attribute value lives *only* in the provided record; priors cannot supply it. |
| **Query is intrinsic** (addressability) | An XPath / JSONPath / SQL `WHERE` / a key *is* the prompt — the cleanest possible signal for query‑conditioned pooling (the "Visit" step of [interactive_extractor.md](../interactive_extractor.md)). |
| **Dense, exact, free labels** (forcing fn) | Programmatic generation → unlimited scale, exact targets, no annotation, no leakage. |
| **Difficulty is one knob** (ratio probe) | #fields / #rows / nesting depth → doubles as the capacity‑vs‑ratio probe of [mixed_decoder.md](../mixed_decoder.md) §4.4. |

This is a strict generalization of the working `Cite` task
([mllm/data/utils.py](../../../mllm/data/utils.py) `RandomInputTokenizerV2`), which already
extracts a contiguous span between two random *anchor* tokens (Phase 2 "Plain Cite: very
strong"). Structured extraction is **"Cite with a structured query instead of an anchor
span"**.

### 0.1 The extraction / computation boundary (read first)

Keep two tiers strictly separate, because they train different skills and have different
grounding guarantees:

- **Tier‑E — verbatim extraction (primary).** The answer is a *substring* of the context
  (a field value, an attribute, a cell, a captured group). Grounding is provable: the target
  token sequence appears in the input. **This is the spine of the curriculum.**
- **Tier‑C — computed answers (secondary, gated).** The answer is *derived* (SQL `COUNT`/`SUM`,
  JSONata `$sum()`, a sorted top‑k, a date diff). The result is **not** verbatim in context, so
  it mixes an arithmetic/reasoning skill into the grounding signal and can be partially guessed.
  Useful later as a stress test, **not** as the initial forcing function. Tag every generated
  item with its tier so the loss/curriculum can weight them.

> **Rule:** start Tier‑E only. Introduce Tier‑C after the extractor demonstrably reads
> Tier‑E targets out of the memory (per the loop tests in
> [interactive_extractor.md](../interactive_extractor.md) §6.2).

---

## 1. Shared generation framework (plug into the existing pipeline)

All generators emit the **same three token streams** the current `Cite` task already produces,
so they drop into `MaskedCiteBatch` / `MixedDecoderDsType` with no model surgery:

| Stream | Vocab | Role | Existing field |
|---|---|---|---|
| **context** | encoder (BERT WordPiece, `inp_len=128`) | the serialized record fed to `EncoderBert` | `toks_inp` |
| **query** | decoder (Qwen) | the path / selector / question — the **Visit** signal | `toks_prompt` |
| **target** | decoder (Qwen) | the extracted value(s) | `toks_cite_dec` |

Concretely, add a sibling to `RandomInputTokenizerV2` — call it
`StructuredExtractionTokenizer` — that, per sample, (a) builds a record `R` and a query `q`
with a known answer `a ⊆ R`, (b) serializes `R` into ≤ `n_cite_toks` encoder tokens,
(c) tokenizes `q` and `a` with the decoder tokenizer, (d) returns a `TokensSubsetV2`‑shaped
object. A new `MixedDecoderDsType.Struct` (or per‑family values) routes it through the
unchanged `run_on_text_citation` path
([mllm/model/mixed_decoder.py](../../../mllm/model/mixed_decoder.py)).

**Three invariants every generator must enforce** (anti‑shortcut, see §6):

1. **Verbatim answer (Tier‑E).** `a` appears as a contiguous decoder‑token subsequence of the
   serialized record. Assert it at generation time (catch tokenizer‑boundary mismatches early).
2. **Randomized surface form.** Randomize key order, whitespace, quoting style, value
   vocabulary, and entity names per sample so the model cannot learn a fixed grammar position.
3. **Capacity bookkeeping.** Record `(#fields, depth, #rows, serialized_token_len)` on each
   item → lets the eval harness plot accuracy vs. compression ratio directly.

### 1.1 Value vocabularies (make answers truly unguessable)

Draw field *values* from a mix so priors never help:
- **random alphanumeric IDs / numbers** (pure needle — strongest grounding),
- **renamed entities** (the Apono/Hypoon trick at scale — names that don't exist),
- **real wiki spans** (entities/numbers pulled from the existing wiki corpus — keeps the
  surface realistic and shares vocabulary with the rest of training).

---

## 2. Per‑family plans

Ordered from simplest (best first probe) to richest. Each: *context · query · target ·
difficulty knob · generate‑vs‑online · pitfalls.* All are **Tier‑E** unless noted.

### 2.1 Key‑value recall — the baseline capacity probe

- **Context:** `k1: v1; k2: v2; … kM: vM` (random separators, shuffled order).
- **Query:** `kI` (or `Value of kI?`).
- **Target:** `vI`.
- **Difficulty knob:** `M` (#pairs) → the cleanest **capacity curve** per `emb_exp_rate` /
  `slots_per_chunk` and chunk length.
- **Data:** **auto‑generate only.** Trivial; no online source needed or wanted (control is the
  point). Reuse wiki spans as values for realism.
- **Pitfalls:** if keys are always English words in sorted order the model learns position, not
  recall — shuffle order and mix in random‑token keys.
- **Doubles as a held‑out diagnostic** (never train on the exact `M` you evaluate at).

### 2.2 JSON field (JSONPath / dotted key)

- **Context:** a JSON object, randomized nesting depth `d` and breadth `b`, e.g.
  `{"user":{"id":7731,"name":"Nelly Armweak","tags":["x","y"]}}`.
- **Query:** a **JSONPath** (`$.user.id`) or dotted path (`user.id`) or NL (`What is user.id?`).
  Provide all three phrasings across the dataset (paraphrase robustness, §5).
- **Target:** the field value (scalar verbatim; for arrays, the indexed element).
- **Difficulty knob:** #keys, nesting depth `d`, array length, decoy keys with similar names.
- **Data:**
  - **Auto‑generate (primary):** full control over depth/breadth, exact labels.
  - **Online (realism augment):** sample real JSON from public API response corpora / JSON
    Schema example sets / GitHub `*.json` configs to learn realistic key distributions; then
    *re‑label programmatically* (pick a random leaf path as the query). Treat these as a
    held‑out *format‑realism* split, not the training spine.
- **Pitfalls:** BERT WordPiece splits long numbers/IDs into multiple pieces — assert the
  decoder‑side verbatim target still aligns; prefer values that tokenize cleanly for the first
  experiments, then relax.

### 2.3 JSONata (and `jq`) — selection + light transform

- **Context:** same JSON records as §2.2.
- **Query:** a **JSONata**/`jq` expression. Split by tier:
  - **Tier‑E:** pure navigation/filter that *returns a stored value* —
    `users[role="admin"].email`, `orders[2].id`.
  - **Tier‑C:** aggregation/transform — `$sum(orders.amount)`, `$count(users)`,
    `items[price>10].name` returning a *derived list*. Gate per §0.1.
- **Target:** the selected value(s) (Tier‑E) or computed result (Tier‑C).
- **Difficulty knob:** predicate count, filter selectivity, transform complexity.
- **Data:** **auto‑generate.** A small JSONata/`jq` evaluator (or the reference libraries) gives
  exact labels for free; no good labelled online corpus exists at the needed scale.
- **Pitfalls:** keep Tier‑E vs Tier‑C labelled; an unfiltered JSONata corpus silently mixes
  computation into the grounding signal.

### 2.4 XML / XPath

- **Context:** an XML fragment with elements, attributes, and text nodes; randomized tag
  names, attribute order, and depth.
- **Query:** an **XPath** (`//user[@id='7731']/name/text()`, `/root/item[2]/@sku`) or NL
  equivalent.
- **Target:** node text or attribute value (verbatim).
- **Difficulty knob:** depth, sibling count, attribute count, predicate complexity, namespaces.
- **Data:**
  - **Auto‑generate (primary):** build a random element tree, pick a random node, derive the
    XPath that selects it (exact label by construction).
  - **Online (augment):** real XML/HTML from open corpora (Wikipedia XML dumps, public RSS,
    OpenStreetMap, SEC/EDGAR XBRL, arXiv metadata) re‑labelled programmatically with
    `lxml` XPath evaluation; good for realistic structure, namespaces, mixed content.
- **Pitfalls:** XPath is verbose and may eat into the prompt budget (`ie_max_prompt_len`);
  cap expression length. HTML adds attribute noise — useful as a harder held‑out format.

### 2.5 SQL `SELECT` over a small table

- **Context:** a small relational table serialized as CSV / markdown rows, e.g.
  ```
  | id | name          | city    |
  | 7  | Nelly Armweak | Apono   |
  | 8  | Bert Quill    | Hypoon  |
  ```
- **Query:** `SELECT city FROM t WHERE id = 7` (and NL paraphrase: *"Which city has id 7?"*).
- **Target:**
  - **Tier‑E:** projected cell(s) for a single/looked‑up row (verbatim).
  - **Tier‑C:** `COUNT`/`SUM`/`MAX`/`ORDER BY … LIMIT k`/`GROUP BY` — derived. Gate per §0.1.
- **Difficulty knob:** #rows, #cols, predicate count, join across two tiny tables (advanced).
- **Data:**
  - **Auto‑generate (primary):** random schema + rows + a templated query; label by executing
    the query with an in‑memory SQLite (`sqlite3`) → exact answers, including Tier‑C.
  - **Online (strongly recommended for NL realism):** **WikiSQL**, **Spider**, **BIRD**,
    **WikiTableQuestions**, **FeTaQA** provide *natural‑language* questions paired with tables +
    gold SQL/answers — the best bridge from synthetic selectors to real questions (§5). Filter
    to single‑table, small results for Tier‑E; keep multi‑table/aggregate for Tier‑C.
- **Pitfalls:** large tables overflow `inp_len=128` — start with ≤ 8 rows × ≤ 4 cols; scale via
  the multi‑chunk window (`emb_win_*`) only after single‑chunk works.

### 2.6 Additional families (round out coverage)

Each follows the same context/query/target schema; all auto‑generatable, most with an optional
real‑corpus augment.

| Family | Context | Query | Target | Tier | Online augment |
|---|---|---|---|---|---|
| **CSV/TSV cell** | delimited rows | `(row r, col c)` or header+key | cell | E | UCI / Kaggle small tables |
| **Markdown table** | `\| … \|` table | NL: "value in row X, col Y" | cell | E | GitHub READMEs |
| **YAML / TOML / INI** | config blob | dotted key | value | E | GitHub config files |
| **Regex capture** | free text + pattern | a regex w/ a group | captured group | E | logs, emails |
| **Log / grep field** | log lines | "field after `status=`" | value | E | public log datasets |
| **Function‑call args** | a tool schema + NL request | "extract args as JSON" | arg values | E | BFCL, ToolBench, API‑Bank |
| **List indexing** | a list | "the k‑th item" / "item after X" | element | E | — (synthetic) |
| **RDF / SPARQL** | triples `(s,p,o)` | `SELECT ?o WHERE { s p ?o }` | object | E | DBpedia, Wikidata subsets |
| **GraphQL** | JSON + query | GraphQL selection set | sub‑object | E | public GraphQL APIs |
| **Coref / entity track** | short narrative w/ renamed entities | "who did X?" | entity | E | (synthetic; CLUTRR for harder) |
| **Date/number normalize** | text w/ a date/number | "the date" | normalized value | E→C | — |
| **SQL aggregate / sort** | table | `COUNT`/`SUM`/top‑k | derived | C | WikiSQL/Spider agg subset |

> **"Anything else?"** The highest‑value additions beyond your list are **function‑call /
> tool‑argument extraction** (directly transfers to agent use‑cases and has strong online
> corpora — BFCL, ToolBench, API‑Bank, Gorilla), **regex/log field extraction** (free,
> infinite, very NL‑adjacent), and **RDF triple → SPARQL** (clean graph‑addressed recall).

---

## 3. Auto‑generate vs. use online data — the decision

| | Auto‑generate | Online / real corpora |
|---|---|---|
| **Strength** | unlimited scale, exact labels, tunable difficulty, no leakage, full control of the capacity curve | realistic surface forms, real key/value distributions, **natural‑language queries**, format diversity |
| **Weakness** | risks teaching a *format‑parsing* shortcut (§6) | finite, noisier labels, harder to control difficulty, must re‑label for extraction |
| **Use as** | the **training spine** (Tier‑E) + the **diagnostic probes** | **realism augment** + the **NL‑query bridge** + **held‑out format generalization** |

**Recommendation:** auto‑generate the spine; fold in online data for (a) realistic JSON/XML/SQL
*structure*, and (b) *natural‑language questions* over tables (WikiSQL/Spider/WTQ/FeTaQA), so
the Visit signal learns to handle real questions, not just formal selectors.

---

## 4. Existing approaches to data‑extraction learning (prior art)

The user's Q2: *are there existing approaches, maybe more NL‑oriented?* Yes — across five
lines. Each contributes a transferable lesson for what to generate and how to phrase queries.

### 4.1 Synthetic recall / associative‑memory probes (closest to our spine)
- **MQAR / Zoology** (Arora et al. 2023, arXiv:2312.04927) — multi‑query associative recall;
  the canonical "can the compressed state recall a stored value" probe. Our key‑value family
  (§2.1) is essentially MQAR with a serialized context.
- **Passkey / needle‑in‑a‑haystack** (Mohtashami & Jaggi 2023, arXiv:2305.16300; Kamradt) —
  single‑token recall through compression.
- **RULER** (Hsieh et al. 2024, arXiv:2404.06654) — a *configurable* synthetic battery
  (multi‑key, multi‑value, multi‑hop, variable tracking, aggregation) — a ready‑made template
  catalog for several of our families; mirror its task taxonomy.
- **bAbI** (Weston et al. 2015), **CLUTRR** (Sinha et al. 2019) — reasoning over *stated*
  facts; priors can't help. Good Tier‑E/Tier‑C bridges for coref/entity tracking.

### 4.2 Semantic parsing & text‑to‑SQL (the NL‑oriented selector line)
- **WikiSQL** (Zhong et al. 2017), **Spider** (Yu et al. 2018), **BIRD** (Li et al. 2023) —
  *natural‑language question → SQL → answer* over tables. The single best source of **real NL
  queries** paired with structured contexts; use the answers directly (Tier‑E single‑table) and
  the gold SQL to label difficulty.
- **Text‑to‑SPARQL / KBQA** (LC‑QuAD, QALD) — NL → graph query over triples; the RDF analog.
- **Text‑to‑API / function calling** (Berkeley **BFCL**, **ToolBench**, **API‑Bank**,
  **Gorilla**) — NL request → structured argument extraction; directly reusable as a §2.6
  family and highly transferable to agent settings.

### 4.3 Table / structured QA (NL questions over structured context)
- **WikiTableQuestions** (Pasupat & Liang 2015), **TAT‑QA**, **HybridQA**, **FeTaQA**
  (free‑form table QA), **TabFact** (table fact verification — counterfactual, anti‑guessing).
  These give **natural‑language** questions whose answers are table cells or short derivations —
  exactly the NL‑oriented extraction the user asks about, and a natural realism layer over §2.5.

### 4.4 Classic NL extraction (span / entity / relation / slot)
- **Span QA:** SQuAD / SQuAD2.0, NewsQA, Natural Questions, TriviaQA — answer is a span of the
  passage. The NL ancestor of our Tier‑E; weaker forcing function alone (often guessable), but
  the *renamed‑entity* variant (CNN/DailyMail anonymized cloze) restores grounding.
- **NER:** CoNLL‑2003, OntoNotes, WNUT‑17 (rare entities → low priors).
- **Relation extraction:** TACRED, DocRED (doc‑level, multi‑hop), FewRel.
- **Slot filling / intent:** ATIS, SNIPS, MultiWOZ (dialogue state = key‑value extraction over
  conversation — a very natural NL key‑value family).
- **Discrete reasoning over text:** DROP (extract‑then‑compute — a Tier‑C NL analog).

### 4.5 Counterfactual / evidence‑grounded verification (best anti‑cheating signal)
- **VitaminC** (Schuster et al. 2021) — near‑identical claims get *opposite* labels after a
  small evidence edit; a model ignoring the context scores at chance. The strongest NL forcing
  function; pair it with the synthetic spine (already recommended in
  [interactive_extractor.md](../interactive_extractor.md) §5.2).
- **FEVER** (Thorne et al. 2018) — claim + evidence → SUPPORTS/REFUTES/NEI.

### 4.6 Document‑to‑structure / information extraction as generation
- **Data‑to‑text inverted** (ToTTo, WikiBio, DART, E2E) used *backwards*: given the realized
  text, extract the source table cells — controlled, extractive, hallucination = loss.
- **OpenIE / closed IE as seq2seq**, **KILT** (knowledge‑intensive tasks with provenance) —
  frame extraction as constrained generation grounded in retrieved evidence.

> **Synthesis for our setting:** the NL‑oriented literature (4.2–4.4) supplies *real questions*
> and proves the task is learnable; the synthetic literature (4.1) supplies *unlimited, exactly
> labelled, unguessable* training and the *capacity probes*. Our plan uses synthetic as the
> spine and NL corpora as the realism + paraphrase bridge — the same split the long‑context
> compression literature converged on ([mixed_decoder.md](../mixed_decoder.md) §3.2).

---

## 5. Making it natural‑language‑oriented (the user's Q2, applied)

Pure formal selectors risk teaching grammar parsing, not content extraction. Bias toward NL:

1. **Triple‑phrase every query.** For each generated item, emit the formal selector *and* one
   or two NL paraphrases (`$.user.id` → "What is the user's id?" / "Give me user id."). Train on
   a random phrasing; evaluate cross‑phrasing.
2. **Inject NL corpora.** Mix WikiSQL/Spider/WTQ/FeTaQA (NL questions over tables) and BFCL/
   ToolBench (NL → args) into the structured spine so the Visit signal sees real questions.
3. **Prose‑embedded records.** Serialize some records inside natural sentences ("The user, whose
   id is 7731 and who lives in Apono, …") and ask NL questions — closer to real RAG context than
   bare JSON.
4. **Renamed entities everywhere** (Apono/Hypoon) so even NL questions can't be answered from
   priors — the grounding guarantee survives the move to natural language.

---

## 6. Anti‑shortcut discipline (do not skip)

A synthetic generator is only as good as its negatives. Enforce:
- **Randomize serialization** (key order, spacing, quoting, delimiters) per sample.
- **Decoy fields** with near‑identical keys/values so the query must disambiguate (not just
  pattern‑match the lone number).
- **Held‑out format generalization:** train JSON+SQL, **test XPath**; train formal selectors,
  **test NL paraphrases**. If accuracy collapses, the model learned the grammar, not the
  content ([interactive_extractor.md](../interactive_extractor.md) §6.3).
- **Verbatim assertion** at generation time (Tier‑E): the decoder‑tokenized target *is* a
  subsequence of the decoder‑tokenized record.
- **Tier tagging** so Tier‑C (computed) items never silently dominate the grounding signal.

---

## 7. Curriculum & mixing (where these data sit)

Following [mixed_decoder.md](../mixed_decoder.md) §5.6 and
[interactive_extractor.md](../interactive_extractor.md) §5.2:

1. **Stage‑1 (context grinding):** key‑value + JSON‑field + SQL‑select (Tier‑E) **+** salient‑span
   cloze on wiki **+** context reconstruction (AE always on).
2. **Stage‑1 forcing add‑on:** VitaminC (counterfactual) + renamed‑entity QnA.
3. **Stage‑1 NL bridge:** WikiSQL/Spider/WTQ single‑table + BFCL args.
4. **Stage‑2 bridge:** the existing QnA aggregate (`QnaAns` / `QnaAnsCite`).
5. **Tier‑C (later):** JSONata/SQL aggregates, XPath predicates, DROP‑style compute.
6. **Diagnostics (never trained on):** passkey, MQAR, key‑value recall, XPath held‑out, swept
   over `emb_exp_rate` / `ie_exp_rate` and chunk length → the capacity curve.

Difficulty schedule: start at the easy end of every knob (few fields, depth 1, ≤ 8 rows) and
anneal upward, logging accuracy vs. serialized‑token length to keep the operating point on the
working side of the ratio wall.

---

## 8. Implementation checklist (this repo)

- [ ] `StructuredExtractionTokenizer` next to `RandomInputTokenizerV2`
      ([mllm/data/utils.py](../../../mllm/data/utils.py)) emitting `TokensSubsetV2`‑shaped items
      (`toks_inp` enc‑vocab record, `toks_prompt` dec‑vocab query, `toks_cite_dec` dec‑vocab
      target) + per‑item metadata `(family, tier, #fields, depth, #rows, ctx_len)`.
- [ ] Per‑family record generators (start: key‑value, JSON‑field, SQL‑select) with
      verbatim‑assertion + serialization randomization (§1, §6).
- [ ] New `MixedDecoderDsType` value(s) routing structured batches through the unchanged
      `run_on_text_citation` path ([mllm/model/mixed_decoder.py](../../../mllm/model/mixed_decoder.py)).
- [ ] Loaders for WikiSQL/Spider/WTQ/FeTaQA + BFCL re‑labelled into the same item shape (NL bridge).
- [ ] Diagnostic harness: key‑value/MQAR/passkey/XPath‑held‑out, accuracy vs. ratio
      ([interactive_extractor.md](../interactive_extractor.md) §6.1).
- [ ] Triple‑phrase query emitter (formal + NL paraphrases) for cross‑phrasing eval (§5).

---

## 9. Open questions

1. **Tokenizer alignment.** BERT WordPiece (encoder) vs. Qwen BPE (decoder) split structural
   punctuation and long IDs differently — does the verbatim‑subsequence guarantee survive for
   all value vocabularies, or only "clean" ones? (Affects which values are safe at first.)
2. **Single‑chunk budget.** How much structure fits in `inp_len=128` before tables/XML must span
   the multi‑chunk window — and does cross‑chunk extraction need a different query layout?
3. **Synthetic→real transfer.** Does training on auto‑generated JSON/SQL transfer to *real* API
   responses and NL questions, or is a minimum real‑corpus fraction required? (Measure via the
   held‑out NL/format splits.)
4. **Tier‑C timing.** At what extraction accuracy is it safe to introduce computed answers
   without re‑opening the cheating path?

---

## 10. References

Anchors only; full citations and recaps in [mixed_decoder.md](../mixed_decoder.md) §6 and the
[papers index](../../papers).

- **Synthetic recall:** Arora et al. *Zoology/MQAR* 2023 · Mohtashami & Jaggi *passkey* 2023 ·
  Hsieh et al. *RULER* 2024 · Weston et al. *bAbI* 2015 · Sinha et al. *CLUTRR* 2019.
- **Text‑to‑SQL / parsing:** Zhong et al. *WikiSQL* 2017 · Yu et al. *Spider* 2018 ·
  Li et al. *BIRD* 2023.
- **Table QA:** Pasupat & Liang *WikiTableQuestions* 2015 · *FeTaQA* · *TAT‑QA* · *TabFact* · *HybridQA*.
- **Function calling:** Berkeley *BFCL* · *ToolBench* · *API‑Bank* · Patil et al. *Gorilla* 2023.
- **NL extraction:** Rajpurkar et al. *SQuAD/2.0* · *CoNLL‑2003* · Zhang et al. *TACRED* 2017 ·
  Yao et al. *DocRED* 2019 · Dua et al. *DROP* 2019.
- **Counterfactual grounding:** Schuster et al. *VitaminC* 2021 · Thorne et al. *FEVER* 2018.
- **Data‑to‑text (inverted):** Parikh et al. *ToTTo* 2020 · *DART* · *WikiBio* · *KILT*.
- **In‑repo:** [mixed_decoder.md](../mixed_decoder.md) · [interactive_extractor.md](../interactive_extractor.md) ·
  [mllm/data/utils.py](../../../mllm/data/utils.py) · [mllm/model/mixed_decoder.py](../../../mllm/model/mixed_decoder.py) ·
  [mllm/train/encdec_graph_bert.py](../../../mllm/train/encdec_graph_bert.py).
