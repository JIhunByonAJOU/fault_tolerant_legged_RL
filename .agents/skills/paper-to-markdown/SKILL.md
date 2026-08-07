---
name: paper-to-markdown
description: Convert an attached or local academic PDF, DOI, URL, arXiv identifier, or paper title into a faithful, searchable Markdown record under docs/papers; discover and record official public code; and maintain the citation graph without duplicates. Use when the user says paper-to-markdown or p2m, or when Codex must preserve paper structure, equations, tables, references, metadata, and legged-robot/RL implementation details.
---

# Paper to Markdown

Create an auditable Markdown copy of a paper, not merely a summary. Keep source-derived text separate from agent-authored analysis.

## Resolve the input

1. Accept a local/attached PDF, URL, DOI, arXiv ID, or exact/approximate title.
2. For a local PDF, use the supplied file and never modify it in place.
3. For a title or identifier, search the web and confirm the exact title and authors. Prefer, in order:
   - an openly accessible publisher or DOI landing page;
   - the latest matching arXiv version;
   - an author or institutional repository.
4. Record both the canonical landing-page URL and the actual PDF URL. Do not bypass a paywall. If no lawful full-text PDF is available, stop and ask the user to provide it; never expand an abstract into a fake full paper.
5. Detect duplicates before converting by DOI, arXiv ID, normalized title, then PDF SHA-256.

## Discover public code

For every paper, check whether implementation code is publicly available even when the paper itself does not print a repository URL. Search in this order:

1. repository or project URL printed in the paper;
2. official publisher, DOI, arXiv, supplementary-material, or project page;
3. corresponding-author pages and author-owned GitHub/GitLab organizations;
4. exact-title web search followed by identity verification.

Classify a repository as `official` only when the paper, project page, author, or publisher establishes the relationship. Use `author_related`, `third_party`, or `uncertain` otherwise. Do not call a reimplementation official merely because its name matches the paper.

Record the repository URL, relationship, evidence URL, checked date, default branch, paper-specific tag or commit when known, license when visible, and notes about missing releases or archived state. Do not deeply analyze its parameters during paper ingestion; leave that to Searcher. Preserve an explicit `no_public_code_found` or `unknown` result rather than omitting the field.

## Choose the destination

Use the repository root as the base. Save the explicitly identified main paper under:

```text
docs/papers/main/<year>-<first-author>-<short-title>/
```

Save other papers under:

```text
docs/papers/ref/<year>-<first-author>-<short-title>/
```

Use lowercase ASCII kebab-case. If year or author is initially unknown, resolve metadata before naming the directory. Store:

```text
source.pdf
paper.md
metadata.json
implementation-notes.md   # only for robotics/RL papers with relevant details
figures/                  # only images needed to understand a result or diagram
```

Do not retain temporary page renders or raw extraction text in the final paper directory.

## Prepare and inspect the PDF

Create a temporary directory with `mktemp -d`, then run:

```bash
python3 <skill-dir>/scripts/prepare_pdf.py <input.pdf> <temporary-directory>
```

The script validates the PDF, records PDF metadata and SHA-256, and creates layout-preserving text. Use `--render-pages 1,3-5` when visual inspection is required. Inspect the first page, all pages containing complex equations or tables, and at least one references page. If extracted text is empty or badly corrupted and no OCR tool is available, report the limitation rather than inventing content.

## Write `paper.md`

Start with YAML frontmatter:

```yaml
---
id: <year-first-author-short-title>
title: "<exact title>"
authors:
  - "<author>"
year: <year>
venue: "<venue or unknown>"
doi: "<DOI or null>"
arxiv: "<arXiv ID or null>"
landing_page: "<canonical URL>"
pdf_url: "<download URL or local>"
source_pdf: "source.pdf"
sha256: "<digest>"
retrieved_at: "<YYYY-MM-DD>"
conversion_status: "verified"
---
```

After the frontmatter, reproduce the paper in its original language and order:

- title, author block, abstract, and every numbered section;
- equations in LaTeX, retaining equation numbers;
- tables as Markdown or HTML when Markdown cannot represent merged cells;
- figure/table captions and in-text references;
- acknowledgements, appendices, footnotes, and the complete reference list.

Normalize repeated page headers/footers, line-wrap artifacts, and obvious end-of-line hyphenation. Do not paraphrase source prose, silently repair technical claims, translate text, or guess unreadable symbols. Mark uncertainty as `[UNCLEAR: page N, brief reason]`. Attribute any omitted non-textual material explicitly, for example `[Figure 4 omitted; caption preserved]`.

## Extract implementation details separately

For legged locomotion, reinforcement learning, sim-to-real, or control papers, create `implementation-notes.md`. Cite section, equation, table, appendix, or page for every value. Include only applicable fields:

- state versus policy observation, privileged observation, history, dimensions, and normalization;
- action semantics, bounds, scaling, decimation, simulation timestep, and control frequency;
- transition/dynamics model and actuator or PD-control parameters;
- every reward term, coefficient, schedule, and clipping rule;
- reset and termination conditions;
- terrain, curriculum, domain randomization, disturbances, and system identification;
- optimizer/PPO hyperparameters, rollout horizon, batch sizes, and training duration;
- robot model, simulator, hardware, and evaluation protocol;
- parameters inherited from cited papers, with the cited paper key and the unresolved item.

Label every statement as one of `Explicit`, `Derived`, `Inherited`, `Unspecified`, or `Ambiguous`. Put calculations under `Derived`; do not present them as quoted paper values.

## Write `metadata.json`

Write machine-readable metadata matching the Markdown frontmatter, plus:

- `role`: `main` or `reference`;
- `aliases`: alternate titles or preprint titles;
- `cites`: known paper IDs already present in the repository;
- `cited_by`: known paper IDs already present in the repository;
- `conversion`: extractor, page count, conversion date, and any warnings.
- `code`: public-code search result using this shape:

```json
{
  "availability": "yes | no_public_code_found | unknown",
  "checked_at": "YYYY-MM-DD",
  "repositories": [
    {
      "url": "https://github.com/owner/repository",
      "relation": "official | author_related | third_party | uncertain",
      "evidence_url": "https://...",
      "default_branch": "main",
      "paper_version": "tag, branch, commit, or null",
      "license": "SPDX identifier or null",
      "notes": "brief factual note or null"
    }
  ]
}
```

Use `null` for unknown values. Never use guessed bibliographic values.

## Update the index

Create or update `docs/papers/00_index.md`. Keep one row per paper with ID, year, title, authors, role, source link, local Markdown link, code availability/link, conversion status, and notes status. Preserve existing rows and sort main papers first, then references by year and title. Update cross-links only when supported by the paper's reference list.

## Maintain the citation graph

Create or update `docs/papers/citation-tree.md` as the human-readable graph entrypoint. Give each main paper its own rooted tree, link every local node to `paper.md`, and show unresolved external references as non-local leaves only when they are relevant to the tracked implementation lineage. A shared reference may appear under multiple roots; do not duplicate its paper directory.

Keep `metadata.json.cites` and `metadata.json.cited_by` reciprocal for every local edge. Before adding any node, search the full corpus by DOI, arXiv ID, normalized title, and SHA-256. Reuse an existing ID regardless of whether it is already under `main` or `ref`; never create a second record to satisfy another tree branch. Detect and report cycles rather than expanding them indefinitely.

## Verify before completion

Check all of the following:

1. `source.pdf` opens and its SHA-256 matches `metadata.json` and `paper.md`.
2. Title, authors, year, DOI/arXiv ID, and page count agree with primary-source metadata.
3. Every top-level source section appears in `paper.md` in order.
4. Abstract start/end, several random body passages, equations containing key parameters, tables, and the final references agree with rendered pages.
5. No replacement characters (`�`), extraction control characters, invented prose, or unresolved placeholders remain except explicit `[UNCLEAR: ...]` markers.
6. Links in `paper.md`, `implementation-notes.md`, and `00_index.md` resolve locally.
7. Public-code metadata has an explicit availability value and every `official` relation has an evidence URL.
8. Local citation edges are reciprocal and `citation-tree.md` contains the ingested paper under the appropriate main root.

Set `conversion_status` to `needs-review` instead of `verified` when any material equation, table, section, or reference remains uncertain. Report the saved paths, source/version chosen, status, and unresolved issues to the user.
