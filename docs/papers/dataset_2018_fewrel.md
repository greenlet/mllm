# FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset — Han, Zhu et al., 2018

> **arXiv:** 1810.10147v2 · **Venue:** EMNLP 2018 · **Affiliation:** Tsinghua NLP (THUNLP)

## TL;DR
FewRel reframes relation extraction as a **few-shot** problem: **70,000 sentences** over **100 relations**
(700 clean instances each) from Wikipedia + Wikidata, split so that **train, validation, and test use
disjoint relation sets**. At test time a model sees an **N-way K-shot** episode — $N$ novel relations, each
with only $K$ labeled example sentences — and must classify a query sentence's relation from that tiny
support set. The paper adapts state-of-the-art few-shot learners (Prototypical/Matching/GNN/SNAIL nets) and
finds they **struggle vs. humans**, and that success needs varied reasoning skills — establishing few-shot
RE as an open problem and a standard meta-learning benchmark.

## Problem & motivation
Supervised RE ([TACRED](dataset_2017_tacred.md)) needs **many labeled examples per relation** and works on a
**fixed schema**; real applications constantly encounter **new, long-tail relations** with few examples,
and **distant supervision** is noisy. FewRel asks: can a model learn a *new* relation from just a **handful
of clean examples**? This requires **generalizing the notion of "relation"** across types, not memorizing
type-specific patterns.

## Key idea
Build a **balanced, clean** corpus, then evaluate under **episodic meta-learning**.

**Data.** Sentences from Wikipedia are aligned to **Wikidata** triples by distant supervision, then
**crowd-filtered** so every kept sentence genuinely expresses the labeled relation between the marked head
and tail entities — yielding **100 relations × 700 sentences = 70k** high-quality instances. Relations are
partitioned into **64 train / 16 validation / 20 test** — **disjoint**, so test relations are **never seen**
in training.

**N-way K-shot evaluation.** Each episode samples a **support set** $S$ of $N$ relations with $K$ labeled
sentences each, and a **query** sentence $x$ from one of those $N$ relations; the model predicts which of
the $N$. A **Prototypical Network** computes a class prototype as the mean of its support embeddings and
classifies by nearest prototype:

$$
\mathbf c_n=\frac{1}{K}\sum_{(x_i)\in S_n} f_\phi(x_i),\qquad
P(y{=}n\mid x)=\frac{\exp(-\lVert f_\phi(x)-\mathbf c_n\rVert^2)}{\sum_{n'}\exp(-\lVert f_\phi(x)-\mathbf c_{n'}\rVert^2)} .
$$

Standard settings are **5-way-1-shot, 5-way-5-shot, 10-way-1-shot, 10-way-5-shot** — accuracy drops as $N$
grows (harder) and rises with $K$ (more support). The sentence encoder $f_\phi$ (CNN or, later, BERT)
marks the head/tail entity positions, as in position-aware RE.

## How it works
```mermaid
flowchart LR
  WIKI["Wikipedia sentences"] --> DS["distant supervision vs Wikidata triples"]
  DS --> CROWD["crowd filter: keep sentences that truly express the relation"]
  CROWD --> POOL["100 relations × 700 clean sentences"]
  POOL --> SPLIT["disjoint: 64 train / 16 val / 20 test relations"]
  SPLIT --> EP["episode: N-way K-shot support set + query"]
  EP --> ENC["encoder f_φ (mark head/tail)"]
  ENC --> PROTO["prototypes c_n = mean of support embeddings"]
  PROTO --> CLS["classify query by nearest prototype"]
  CLS --> EVAL["accuracy (5/10-way, 1/5-shot)"]
```

Because train and test relations don't overlap, high accuracy requires learning a **metric space where
'same relation' clusters** generically — not a per-relation classifier.

## Training / data
- **Size:** 70,000 sentences, **100** relations (700 each), head/tail entities marked, from Wikipedia +
  Wikidata.
- **Splits:** **64 / 16 / 20** relations for train / val / test (relation-disjoint); the **test relations
  are held out** on a public leaderboard.
- **Protocol:** episodic **N-way K-shot** sampling; report accuracy across the four standard settings.
- Baselines: Meta Network, GNN, SNAIL, **Prototypical Network**, Matching Network — plus a **human** upper
  bound.

## Results
- **Prototypical networks lead**, but all methods fall well **below human** accuracy — e.g. on the hardest
  standard setting (**10-way-1-shot**) models trail humans by a wide margin, and even **5-way-1-shot** shows
  a clear gap.
- **N and K behave as expected:** accuracy **drops with more classes** ($N$) and **rises with more shots**
  ($K$); 5-way-5-shot is comparatively easy, 10-way-1-shot the stress test.
- **Diverse skills needed:** analysis shows correct classification often requires **coreference, syntactic,
  and world-knowledge** cues — a single surface heuristic doesn't suffice.
- Became the canonical **few-shot RE** benchmark; the follow-up **FewRel 2.0** added **domain adaptation**
  (biomedical test relations) and a **none-of-the-above** option, both of which sharply degrade the 1.0
  leaders — confirming brittleness.

## Limitations & follow-ups
- **Balanced & clean** (700/relation, crowd-filtered) — easier than the noisy, negative-heavy real world;
  FewRel 2.0's NOTA + domain shift is more realistic.
- **Single sentence, single relation per instance** — no document-level or multi-hop relations (that's
  [DocRED](dataset_2019_docred.md)).
- Episodic accuracy depends on the **sampling protocol**, complicating cross-paper comparison.
- Complements supervised sentence RE ([TACRED](dataset_2017_tacred.md)) and document RE
  ([DocRED](dataset_2019_docred.md)); shares Wikipedia/Wikidata construction with DocRED (overlapping
  authors).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/1810.10147) · [html](https://arxiv.org/html/1810.10147v2) · [pdf](https://arxiv.org/pdf/1810.10147)
- **Code / data / leaderboard:** <https://github.com/thunlp/FewRel> · <http://zhuhao.me/fewrel>
- **BibTeX:**
  ```bibtex
  @inproceedings{han2018fewrel,
    title     = {{FewRel}: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation},
    author    = {Han, Xu and Zhu, Hao and Yu, Pengfei and Wang, Ziyun and Yao, Yuan and Liu, Zhiyuan and Sun, Maosong},
    booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year      = {2018},
    url       = {https://arxiv.org/abs/1810.10147}
  }
  ```
- **Related papers:** [TACRED](dataset_2017_tacred.md) · [DocRED](dataset_2019_docred.md) ·
  [CoNLL-2003 NER](dataset_2003_conll2003-ner.md)
- **In-repo:** [§6.8 in mixed_decoder](../mixed_decoder/mixed_decoder.md)
