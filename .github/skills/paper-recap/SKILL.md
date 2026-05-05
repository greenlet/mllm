---
name: paper-recap
description: |
  Generate a structured, single-page (or longer when needed for implementability) recap
  of a research paper or technical report and save it to `docs/papers/`. Generic and
  project-agnostic. Triggers on phrases like "recap paper", "summarize paper",
  "add paper recap", "build recap thread", "generate paper review", "review this arXiv".
  Inputs: an arXiv ID, a paper title, or a URL. Output: a new file at
  `docs/papers/p<NNN>_<YYYY>_<category>_<keywords>.md` plus optional figure assets.
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
docs/papers/p<NNN>_<YYYY>_<category>_<keywords>.md
docs/papers/_assets/p<NNN>_<YYYY>_<category>_<keywords>/figureN.png
```

- `<NNN>` — 3-digit zero-padded sequence number. Determine the next free number by
  listing `docs/papers/p*.md` and incrementing the max. Reserve gaps if the user
  asks for logical thread grouping.
- `<YYYY>` — original publication year (arXiv v1 if multiple).
- `<category>` — short identifying concept; 1–3 hyphen-joined tokens. Use the thread
  name when the paper belongs to one (e.g. `positional`, `peft-lora`, `alignment-rl`,
  `multimodal-vision`); otherwise the most descriptive concept (`qwen2`, `qwen2-vl`,
  `tokenization`, `benchmark-math`).
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

- Download **at most 1–3 essential figures** per paper (architecture diagram, key
  result plot). Skip decorative figures, tables, and non-essential plots.
- Find figure URLs from arXiv HTML — typically
  `https://arxiv.org/html/<id>v<n>/x1.png`, `x2.png`, ...
  or `https://arxiv.org/html/<id>v<n>/extracted/<hash>/figureN.png`.
- Download with PowerShell:
  ```powershell
  New-Item -ItemType Directory -Force -Path docs/papers/_assets/<full-stem> | Out-Null
  Invoke-WebRequest -Uri "<url>" -OutFile "docs/papers/_assets/<full-stem>/<name>.png"
  ```
  (use `curl.exe` on non-Windows). Verify each download succeeded before embedding.
- Reference in markdown as `![caption](_assets/<full-stem>/<name>.png)`.
- If no good figure exists or download fails, omit the image — never embed a remote
  URL as a substitute for a local asset.

## Per-paper recap template

Use this exact section structure. Keep it ~1 page; expand only when needed for
implementability of the core idea.

```markdown
# <Short title> — <First author> et al., <Year>

> **arXiv:** <id>v<n> · **Venue:** <conf/journal or "preprint"> · **Affiliation:** <lab>

## TL;DR
2–3 sentences capturing the contribution.

## Problem & motivation
What was broken before. What prior art assumed. Why it matters.

## Key idea
The core mechanism in plain language. Include the central equation in KaTeX
when applicable:

$$
\text{equation here}
$$

## How it works
- Step-by-step algorithm or architecture description, detailed enough to reimplement.
- Inputs / outputs / shapes.
- Critical hyperparameters and their typical values.
- Embed 1–3 essential figures here:
  ![Figure 1: architecture](_assets/<full-stem>/x1.png)

## Training / data
Datasets, objective(s), compute budget, optimizer / schedule, recipe specifics.

## Results
| Benchmark | This paper | Baseline | Notes |
|---|---|---|---|
| ...       | ...        | ...      | source: §X |

Headline numbers only, sourced. Mark uncertainty with "(per abstract)" or "(per §X)".

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
[RoPE]: ../papers/p002_2021_positional_rope-roformer.md "Su et al., RoFormer (2021)"
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

## Workflow checklist

1. Resolve input → arXiv ID + canonical URL.
2. Determine `<NNN>` (next free in `docs/papers/`).
3. Fetch sources (arXiv HTML preferred; split by section if long).
4. Identify 1–3 essential figures, download to `_assets/<full-stem>/`.
5. Write recap file per template; KaTeX for equations; cite all numbers.
6. Repoint any `[Tag]` in `docs/` that previously pointed to this paper's arXiv URL.
7. If part of a thread, update the thread doc's evolution table.
8. Report file path(s) and any tags repointed.
