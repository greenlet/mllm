# Know What You Don't Know: Unanswerable Questions for SQuAD (SQuAD 2.0) — Rajpurkar, Jia & Liang, 2018

> **arXiv:** 1806.03822v1 · **Venue:** ACL 2018 · **Affiliation:** Stanford University

## TL;DR
SQuAD 2.0 augments the extractive reading-comprehension dataset SQuAD 1.1 with **50,000+ unanswerable
questions**, written **adversarially by crowdworkers** to look plausible against a paragraph that in fact
contains no answer. To score well a system must both **extract the answer span when one exists** and
**abstain** ("no answer") when none is supported — the ability prior extractive models lacked, since they
tended to return a confident guess for every question. The jump in difficulty is dramatic: a strong neural
model at **86% F1 on SQuAD 1.1** drops to **66% F1 on SQuAD 2.0**, versus **~89% human** — making
answer-abstention a first-class evaluation target and the de-facto benchmark for "knowing what you don't
know."

## Problem & motivation
Extractive QA models locate answer spans well but are **overconfident**: on questions with no supported
answer they still emit a span. Existing "unanswerable" data was inadequate — either datasets contained
**only answerable** questions, or their unanswerable examples were **auto-generated** (e.g. random or
distant questions) and trivially detectable by topic/entity mismatch. A good benchmark needs unanswerable
questions that are **relevant, plausible, and hard to distinguish** from answerable ones, so that
abstention requires genuine comprehension rather than a keyword-overlap heuristic.

## Key idea
Keep SQuAD 1.1's answerable (paragraph, question, span) triples and add **crowd-authored unanswerable
questions** under an adversarial protocol: shown a Wikipedia paragraph, workers write questions that
(a) are **on-topic** and **answerable-looking**, and (b) have **no answer** in the paragraph, often by
referencing a **plausible (but wrong) answer** — an entity of the right type present in the text. This
defeats type-matching shortcuts: the paragraph typically **contains a plausible answer of the correct
answer type**, so a model can't abstain just because "no person is mentioned."

Formally, a system outputs either a span $\hat a$ or the special **no-answer** decision. Evaluation extends
SQuAD's **Exact Match (EM)** and **token-level F1** to handle abstention: for unanswerable questions the
gold answer is the empty string, so predicting **no-answer** scores 1 and any span scores 0. A common
implementation compares a **no-answer score** (e.g. the score of the null/`[CLS]` span) against the best
span score with a tuned threshold $\delta$:

$$
\text{predict no-answer if } s_{\text{null}} - \max_{i\le j} s_{ij} > \delta .
$$

Token-level F1 for answerable questions is the usual

$$
\text{F1} = \frac{2\,\text{prec}\cdot\text{rec}}{\text{prec}+\text{rec}},\quad
\text{prec}=\frac{|\text{pred}\cap\text{gold tokens}|}{|\text{pred}|},\ 
\text{rec}=\frac{|\text{pred}\cap\text{gold}|}{|\text{gold}|}.
$$

## How it works
```mermaid
flowchart LR
  P["Wikipedia paragraph"] --> W["crowdworker"]
  W -->|answerable (from SQuAD 1.1)| A["question + gold span"]
  W -->|adversarial, on-topic, no supported answer| U["unanswerable question (references a plausible wrong answer)"]
  A --> DS["SQuAD 2.0"]
  U --> DS
  DS --> M["reading-comprehension model"]
  M --> DEC{"answerable?"}
  DEC -->|yes| SPAN["extract answer span"]
  DEC -->|no: abstain| NULL["predict no-answer"]
  SPAN --> EVAL["EM / F1 (gold='' for unanswerable)"]
  NULL --> EVAL
```

Because the negatives are **hand-written to be confusable**, the benchmark rewards **calibration** — the
model must estimate whether the evidence actually supports *any* span, not merely rank candidate spans.

## Training / data
Built on **Wikipedia** paragraphs (same source articles as SQuAD 1.1). ~**150k** total questions:
the original **~100k answerable** plus **>50k unanswerable**. Standard train / dev split; the **test set is
hidden** (leaderboard-evaluated). Metrics: **EM** and **F1**, reported overall and split by
answerable/unanswerable ("HasAns"/"NoAns"). Baselines at release include **DocQA** with a no-answer option
and a strong 1.1 model retrofitted with abstention.

## Results
| System | SQuAD 1.1 F1 | SQuAD 2.0 F1 | Source |
|---|---:|---:|---|
| Strong neural extractive model | 86 | **66** | Abstract |
| Human | ~91 | **~89.5** | §Results |

- **20-point drop** from 1.1 to 2.0 for the same strong model — abstention is genuinely hard and not solved
  by span-ranking alone.
- **Human ceiling ~89%** leaves a large model–human gap at release (later closed by BERT-era models, which
  is precisely why SQuAD 2.0 became the standard stress test for no-answer calibration).
- Splitting scores into **HasAns** vs **NoAns** exposes systems that over- or under-abstain, a diagnostic
  the original SQuAD couldn't provide.

## Limitations & follow-ups
- **Single-paragraph, extractive** — answers are contiguous spans in a given passage; no multi-document
  retrieval, no free-form generation, no numeric reasoning.
- Crowd-authored negatives share **annotation style**; models can partly learn the "unanswerable register,"
  so high scores don't fully certify open-world abstention.
- Directly motivated no-answer heads in **BERT/RoBERTa** QA and open-domain/abstaining QA; the
  "know-what-you-don't-know" framing recurs in retrieval-augmented and fact-verification tasks
  ([FEVER](../mixed_decoder/mixed_decoder.md), VitaminC) that also require a "NOT ENOUGH INFO" / abstain
  decision.

## Links
- **arXiv:** [abs](https://arxiv.org/abs/1806.03822) · [html](https://arxiv.org/html/1806.03822v1) · [pdf](https://arxiv.org/pdf/1806.03822)
- **Data / leaderboard:** <https://rajpurkar.github.io/SQuAD-explorer/>
- **BibTeX:**
  ```bibtex
  @inproceedings{rajpurkar2018squad2,
    title     = {Know What You Don't Know: Unanswerable Questions for {SQuAD}},
    author    = {Rajpurkar, Pranav and Jia, Robin and Liang, Percy},
    booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year      = {2018},
    url       = {https://arxiv.org/abs/1806.03822}
  }
  ```
- **Related papers:** [CoNLL-2003 NER](dataset_2003_conll2003-ner.md) ·
  [TACRED](dataset_2017_tacred.md) · [OntoNotes / CoNLL-2012](dataset_2013_ontonotes.md)
- **In-repo:** [§6.8 in mixed_decoder](../mixed_decoder/mixed_decoder.md)
