---
name: paper-recap
description: |
  Generate a **comprehensive**, self-contained recap of a research paper or technical
  report and save it to `docs/papers/`. The recap must convey the *full meaning* of the
  paper: complete method description detailed enough to reimplement, all key equations
  (in KaTeX), architecture/flow diagrams (Mermaid or ASCII), and the paper's essential
  figures downloaded locally. Prefer thoroughness over brevity — length is dictated by
  what it takes to fully reveal the idea, not by a page limit. Generic and
  project-agnostic. Triggers on phrases like "recap paper", "summarize paper",
  "add paper recap", "build recap thread", "generate paper review", "review this arXiv".
  Inputs: an arXiv ID, a paper title, or a URL. Output: a new file at
  `docs/papers/<category>_<YYYY>_<keywords>.md` plus its figure assets.
tools:
  - fetch_webpage
  - read_file
  - grep_search
  - file_search
  - list_dir
  - create_file
  - replace_string_in_file
  - multi_replace_string_in_file
  - run_in_terminal
  - runSubagent
---

# Paper Recap Skill

Build a generic, project-agnostic recap of a research paper. Recaps are stored in
`docs/papers/` and follow a stable naming convention so they sort chronologically
within a category.

## Inputs you accept

- An **arXiv ID** (e.g. `2104.09864` or `2104.09864v5`).
- A **paper title** (search arXiv if needed).
- A **URL** (arXiv abs/html/pdf, OpenReview, project page, blog post, GitHub README).

If the user names a *thread* (e.g. "recap the positional thread") instead of a single
paper, expand it to the list of papers documented in
`docs/qwen/<thread>/<thread>.md` (or ask the user for the list) and run one recap per
paper, parallelizing fetches via the `Explore` subagent.

## File naming convention

```
docs/papers/<category>_<YYYY>_<keywords>.md
docs/papers/_assets/<category>_<YYYY>_<keywords>/figureN.png
```

- `<category>` — short identifying concept; 1–3 hyphen-joined tokens. Use the thread
  name when the paper belongs to one (e.g. `positional`, `moe`, `peft-lora`,
  `alignment-rl`, `multimodal-vision`); otherwise the most descriptive concept
  (`qwen2`, `qwen2-vl`, `tokenization`, `benchmark-math`). **Listed first** so
  files cluster by topic when sorted lexicographically inside `docs/papers/`.
- `<YYYY>` — original publication year (arXiv v1 if multiple).
- `<keywords>` — concise model/method name, hyphen-separated, lowercase, ASCII, 2–4
  tokens (e.g. `rope-roformer`, `lora-low-rank-adaptation`, `flash-attention-2`).

## Source-fetch policy (in order)

1. **arXiv HTML** — `https://arxiv.org/html/<id>v<n>` or `https://arxiv.org/html/<id>`.
   Best structured source. Use `fetch_webpage` with targeted queries
   ("abstract and introduction", "method section", "experiments and results").
2. **arXiv abstract** — `https://arxiv.org/abs/<id>`. Fallback for metadata.
3. **arXiv PDF** — `https://arxiv.org/pdf/<id>` if HTML unavailable.
4. **GitHub README / project page / blog post** — when no arXiv version exists
   (e.g. NTK-aware RoPE, transformer-circuits posts) or to enrich Links.

For long papers, **split-fetch by section** with separate `fetch_webpage` calls.
When recapping a whole thread, dispatch one `Explore` subagent per paper in parallel.

## Image policy

- **Always include the paper's essential figures.** Download **2–4** figures per paper
  that are load-bearing for understanding: the architecture/method diagram (mandatory when
  one exists), the key results plot, and any figure that illustrates the core mechanism.
  Skip only purely decorative or redundant figures.
- Find figure URLs from arXiv HTML — typically
  `https://arxiv.org/html/<id>v<n>/x1.png`, `x2.png`, ...
  or `https://arxiv.org/html/<id>v<n>/extracted/<hash>/figureN.png`.
- Download with PowerShell:
  ```powershell
  New-Item -ItemType Directory -Force -Path docs/papers/_assets/<full-stem> | Out-Null
  Invoke-WebRequest -Uri "<url>" -OutFile "docs/papers/_assets/<full-stem>/<name>.png"
  ```
  (use `curl.exe` on non-Windows). Verify each download succeeded before embedding.
- Reference in markdown as `![caption](_assets/<full-stem>/<name>.png)` with a caption that
  explains what the figure shows and why it matters.
- If a figure download fails, **also** provide a Mermaid/ASCII diagram reproducing its
  content — never embed a remote URL as a substitute for a local asset.
- **In addition to downloaded figures, author your own diagram** (Mermaid ` ```mermaid ` or
  an ASCII block) that reconstructs the architecture / data flow in your own terms. Every
  recap should contain at least one such diagram even when figures download successfully.

## Per-paper recap template

Use this exact section structure. **Be comprehensive:** the recap must stand on its own
as a full explanation of the paper — a reader should understand the problem, the exact
mechanism, every key equation, the architecture (with a diagram), the training recipe,
and the headline results **without opening the original paper**. Do not compress to fit a
page; expand every section as far as the paper's content requires. Prefer more detail,
more equations, and an explanatory diagram over terse bullet points.

```markdown
# <Short title> — <First author> et al., <Year>

> **arXiv:** <id>v<n> · **Venue:** <conf/journal or "preprint"> · **Affiliation:** <lab>

## TL;DR
2–4 sentences capturing the contribution and why it matters.

## Problem & motivation
What was broken before, what prior art assumed, and why it matters. Explain the
quantitative or structural pain point the paper attacks (with numbers where the paper
gives them).

## Key idea
The core mechanism in plain language, then made precise. State the central equation(s)
in KaTeX and **define every symbol**:

$$
\text{equation here}
$$

## How it works
Comprehensive, reimplementation-grade walkthrough. Include, as needed:
- Step-by-step algorithm (numbered) or architecture description, component by component.
- Inputs / outputs / tensor shapes at each stage.
- All key equations with symbols defined.
- Critical hyperparameters and their typical/default values.
- A **self-authored diagram** of the architecture or data flow:
  ```mermaid
  flowchart LR
    A[input] --> B[...] --> C[output]
  ```
- The paper's essential **figures**, embedded with explanatory captions:
  ![Figure 1: architecture — what it shows and why](_assets/<full-stem>/x1.png)

## Training / data
Datasets, objective(s) (with loss equations), compute budget, optimizer / schedule,
calibration, and any recipe specifics needed to reproduce.

## Results
| Benchmark | This paper | Baseline | Notes |
|---|---|---|---|
| ...       | ...        | ...      | source: §X |

Report the headline numbers **and** enough secondary numbers to convey the picture, each
sourced. Mark uncertainty with "(per abstract)" or "(per §X)". Embed a results figure when
it reveals the trend better than a table.

## Limitations & follow-ups
Known issues acknowledged in the paper plus successor work
(link to local recap if it exists, otherwise external).

## Links
- **arXiv:** [abs](https://arxiv.org/abs/<id>) · [html](https://arxiv.org/html/<id>) · [pdf](https://arxiv.org/pdf/<id>)
- **Code:** <github URL or "—">
- **Hugging Face:** <model / dataset / space URL or "—">
- **Project page:** <url or "—">
- **Blog posts:** <url(s) or "—">
- **Talks / videos:** <url(s) or "—">
- **OpenReview / venue page:** <url or "—">
- **Papers-with-Code:** <url or "—">
- **BibTeX:** <inline or link>
- **Related / successor papers:** <local recap links or external>
```

## Quality rules (strict)

- **Never invent numbers.** If not found in source, omit the row or write "n/a".
- **Cite source location** for every benchmark number (`(per Table 3)`, `(per §4.2)`).
- **Pin arXiv version** in the header (use the latest v shown on the abs page).
- **Equations in KaTeX** (`$...$` inline, `$$...$$` block).
- **No project-specific content** in per-paper recaps — no "Why it matters for
  Qwen / for project X" section. That belongs in the thread doc that uses the
  paper. The single exception is when the paper IS the project's tech report
  (e.g. recapping a Qwen tech report itself).

## Cross-link rule (overview update)

After creating the recap file, search `docs/` for any `[Tag]: <url>` link-table
entry that points to this paper's arXiv URL and **repoint it to the local recap**:

```
grep_search for: arxiv.org/(abs|html)/<id>
```

Replace e.g.
```
[RoPE]: https://arxiv.org/abs/2104.09864 "Su et al., RoFormer (2021)"
```
with
```
[RoPE]: ../papers/positional_2021_rope-roformer.md "Su et al., RoFormer (2021)"
```

(Path is computed relative to the file containing the link table — typically
`docs/qwen/overview.md` → `../papers/...`.)

This makes the link table double as a TODO tracker: tags still pointing at arXiv
are papers that haven't been recapped yet.

## Subagent rule

When asked to recap a whole thread (multiple papers), dispatch one `Explore`
subagent per paper to fetch the arXiv HTML and extract: abstract, method, key
equations, hyperparameters, results table, and image URLs. Each subagent returns
a structured summary. Then write each recap file sequentially.

## Thread review rule (mandatory)

A thread review (`docs/qwen/<thread>/<thread>.md`) is **not complete** without
per-paper recaps for every entry in its evolution table. When asked to add or
update a thread:

1. Build the thread doc (evolution table, Qwen-specific notes, see-also).
2. For each paper referenced in the evolution table, check whether a recap
   file already exists in `docs/papers/`. If not, generate it via the
   per-paper workflow (template + figures + cross-link repointing).
3. After all per-paper recaps exist, replace external arXiv URLs in the thread
   doc's evolution table and "papers" sub-bullet list (in `overview.md` §9)
   with relative links to the local recaps.

Never leave a thread review pointing only to external arXiv URLs when local
recaps are feasible — the thread doc should be a hub of local recaps.

## Workflow checklist

1. Resolve input → arXiv ID + canonical URL.
2. Choose `<category>_<keywords>` per the naming convention above.
3. Fetch sources (arXiv HTML preferred; **split-fetch method + experiments sections**, not
   just the abstract, so the recap can be comprehensive).
4. Identify 2–4 essential figures, download to `_assets/<full-stem>/`; verify each download.
5. Write a **comprehensive** recap per template: full method walkthrough, all key equations
   in KaTeX (symbols defined), at least one self-authored Mermaid/ASCII diagram, embedded
   figures with explanatory captions, and sourced result numbers.
6. Repoint any `[Tag]` in `docs/` that previously pointed to this paper's arXiv URL.
7. If part of a thread, update the thread doc's evolution table.
8. Report file path(s) and any tags repointed.
