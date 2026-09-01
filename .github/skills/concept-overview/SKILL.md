---
name: concept-overview
description: |
  Research, create, or revise an exhaustive concept-family overview under `docs/`.
  Use when the user asks for a concept overview, technology landscape, model-family
  survey, state-of-the-art review, exhaustive research guide, or a document intended
  to help readers decide what to study next. The output must explain rather than list:
  every bullet or table cell containing a concept needs several sentences of context,
  every technical concept needs a globally numbered citation, and the final references
  must be organized into chronological/evolutionary "stories". Link to primary external
  sources (arXiv, venue, official project, GitHub, or model card), not local paper recaps.
---

# Concept Overview Skill

Create a repository-quality, self-contained research guide for a technical concept or
model family. The document is a map of the field, not a glossary or link dump: readers
must understand what each approach changes, why it exists, what it trades away, how it
relates to neighboring approaches, and which source to read next.

## Inputs

Accept any of:

- A concept or family name, such as “BERT-family encoders”, “long-context attention”,
  “retrieval models”, or “mixture-of-experts”.
- A requested destination under `docs/`.
- Existing repository reviews whose organization and depth should be matched.
- Specific project concerns that require a final project-facing decision section.

If no destination is supplied, infer a stable path such as
`docs/<concept>/overview.md`. Ask only when two destinations are equally plausible.

## Required research workflow

1. Inspect two or more related repository reviews before drafting. Reuse their useful
   conventions, but improve weak conventions rather than copying them mechanically.
2. Define the scope and taxonomy. Separate architectures, objectives, training recipes,
   downstream systems, and deployment techniques so that unlike categories are not
   presented as direct competitors.
3. Research primary sources first. Prefer the latest arXiv version, venue paper,
   official project page, official GitHub repository, and official model card. Use a
   secondary source only for implementation guidance or when no primary source exists.
4. Verify model dates, dimensions, context lengths, objectives, data scales, benchmark
   versions, and licenses before stating them. If a claim cannot be verified, qualify
   it explicitly or omit it.
5. Distinguish nominal capability from demonstrated capability. Examples include
   configured context length versus effective long-range use, pretrained backbone
   versus task-fine-tuned system, and total parameters versus active parameters.
6. Connect the landscape to the repository when relevant. Identify concrete candidate
   models, controlled ablations, evaluation criteria, integration constraints, and
   failure modes instead of ending with generic recommendations.

## Explanation-depth rules

- Do not use one-clause glossary bullets such as “RoPE — better long context.” Explain
  the mechanism, motivation, consequence, and limitation in at least several complete
  sentences.
- A bullet that introduces a technical idea must normally contain 3–6 sentences. If
  this becomes awkward, replace the bullet with a titled subsection and prose.
- A table is a navigation aid, not a substitute for explanation. Every important table
  must be followed by prose that interprets the rows, cautions against invalid
  comparisons, and tells the reader which branch to investigate next.
- Define symbols in every important equation. State tensor shapes where they clarify
  architecture or implementation.
- Include diagrams when information flow or family relationships are otherwise hard to
  infer. Mermaid and compact ASCII diagrams are preferred over decorative figures.
- Explain trade-offs. For every proposed benefit, discuss costs such as compute,
  memory, index size, training complexity, information loss, language coverage,
  tokenizer fertility, latency, or ecosystem maturity.

## Numeric citation rules

Use one global numeric namespace throughout the document:

```markdown
ModernBERT alternates local and global attention to reduce average long-sequence
cost while periodically restoring document-wide communication [27].
```

Requirements:

1. Every named research concept, model family, objective, dataset, benchmark, or
   non-obvious empirical claim must have at least one numeric citation such as `[12]`.
2. A citation number always resolves to exactly one entry in the final References
   section. Do not reuse one number for different sources.
3. Number references by first appearance in the body. Preserve those numbers in all
   later mentions.
4. Put citations immediately after the claim they support, not at the end of a long
   paragraph containing unrelated claims.
5. For broad synthesis, cite multiple sources, for example `[14, 18, 23]`.
6. Internal repository files may be linked inline, but they do not replace primary
   research citations.
7. Do not redirect citations to local paper recaps. Cite the external primary source:
   arXiv, ACL Anthology, OpenReview, official GitHub repository, official project page,
   or official model card.

## Reference-story structure

The final `References` section must be split into thematic “stories”. Each story is a
short narrative followed by its numbered sources. A story explains how an idea evolved,
which limitation caused the next paper to appear, and how approaches diverged.

Example:

```markdown
### Story 3 — From one-vector retrieval to late interaction

Early sentence encoders made corpus-scale similarity practical, but forced every fact
into one vector. Dense passage retrieval adapted this to open-domain QA. ColBERT then
retained one vector per token and delayed query-document interaction, trading a larger
index for fine-grained matching.

12. Reimers and Gurevych. *Sentence-BERT*. EMNLP 2019. [arXiv](...).
13. Karpukhin et al. *Dense Passage Retrieval*. EMNLP 2020. [arXiv](...).
14. Khattab and Zaharia. *ColBERT*. SIGIR 2020. [arXiv](...) · [Code](...).
```

Rules:

- Organize by conceptual lineage, not alphabetically and not merely by year.
- Keep global citation numbers stable even though entries are grouped by story.
- Within a story, order sources chronologically or causally.
- Include 2–6 sentences introducing each story before its source list.
- Add official code, model, dataset, or project links when they materially help the
  reader continue. Prefer arXiv/venue for papers and GitHub/model cards for artifacts.
- A source may belong to only one primary story. Cross-reference its number from other
  stories rather than duplicating it.

## Recommended document structure

1. Title, status date, scope, and reading guidance.
2. Executive synthesis with several-sentence decisions rather than slogan bullets.
3. Taxonomy and family/evolution diagram.
4. Historical foundations.
5. Core mechanics and equations.
6. Major contemporary branches, each with architecture, training, evidence, and
   limitations.
7. Representation and prediction interfaces.
8. Training, fine-tuning, distillation, and deployment.
9. Evaluation methodology and benchmark caveats.
10. Application-to-method decision guide.
11. Repository-specific implications and experiment plan, when applicable.
12. References organized as stories.

Adjust headings to fit the concept, but preserve the explanatory progression from
foundations to choices to evidence to next reading.

## Quality gate

Before finishing, verify all of the following:

- The requested file exists at the requested path.
- No technical bullet is a fragment or one-sentence label.
- Every major concept has a numeric citation resolving to the final section.
- Citation numbering is unique and reasonably follows first appearance.
- Reference entries point to external primary sources, not local recaps.
- References are grouped into narrated evolutionary stories.
- Tables are interpreted in surrounding prose.
- Backbone checkpoints are distinguished from downstream fine-tunes.
- Benchmark comparisons state protocol/version caveats where needed.
- Nominal context length is distinguished from effective long-context use.
- Project recommendations include alternatives, trade-offs, and controlled tests.
- Links and relative repository paths resolve.
- No unsupported “best”, “SOTA”, license, parameter-count, or benchmark claim remains.

## Output report

Report the created or updated file and the skill file. Summarize the document’s major
stories and call out any claims intentionally qualified because source evidence or
comparison protocols differ.
