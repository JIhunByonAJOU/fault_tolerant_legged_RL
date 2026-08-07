---
name: searcher
description: Trace equations, implementation parameters, MDP definitions, environment settings, and algorithm details through the repository's paper corpus, multi-hop citation graph, project pages, and official public code. Use when the user says searcher, 서처, or 서쳐; asks where a paper-derived value came from; needs missing implementation details for reproduction; or wants evidence-backed comparison between paper text and code.
---

# Searcher

Recover implementation evidence, not plausible defaults. Traverse papers, citations, and official code until the requested item is resolved or the reachable evidence is exhausted.

## Execution profile

When model selection is available, run this workflow with `gpt-5.6-sol` and reasoning effort `medium`.

## Establish the target

Translate the request into individually verifiable items such as an equation, symbol definition, observation, action, reward, transition, reset, termination, randomization range, simulator setting, controller gain, network shape, optimizer value, curriculum rule, or evaluation condition. Record the robot, task, damage mode, simulator, and paper version constraints when they affect the answer.

## Search the local corpus first

1. Start at `docs/papers/00_index.md`, `docs/papers/citation-tree.md`, and the target paper's `metadata.json`.
2. Search `paper.md`, `implementation-notes.md`, tables, captions, appendices, and references. Do not rely only on implementation notes.
3. Traverse `cites` and `cited_by` as a graph. Prefer citations attached to the relevant claim, then implementation ancestors, then broader related work.
4. Keep a visited set keyed by DOI, arXiv ID, normalized title, and repository paper ID. Do not revisit a paper under another alias or create a duplicate.
5. Record the path used to reach every source, for example `Saving the Limping -> Rudin et al. -> legged_gym config`.

If a necessary cited paper is absent, delegate it to the `paper-to-markdown` agent when available or invoke `$paper-to-markdown` directly. The user may call that workflow `p2m`. Resume only after checking the new Markdown record and updated graph.

## Follow official public code

Use the paper, author/project page, publisher metadata, or an author-owned organization as the preferred bridge to code. Treat search-engine association alone as insufficient proof that a repository implements the paper.

Inspect the paper-specific release, tag, branch, commit, experiment config, README command, YAML/JSON/TOML config, Python/C++ defaults, robot asset, training script, and evaluation script as applicable. Search for observations, action scaling, timestep, decimation, gains, limits, rewards, termination, randomization, PPO settings, terrain, curriculum, damage sampling, seeds, and checkpoints.

Do not equate a current default-branch value with the paper's experiment. Record the repository URL, relation evidence, commit or tag when known, file path, symbol or config key, and line location. If only a current default is available, label the version ambiguity.

## Classify every result

Use exactly one primary label per claim:

- `Paper Explicit`: directly stated in the paper.
- `Code Experiment`: present in an official paper-specific config or execution path.
- `Code Default`: present in official code but paper-specific use is unverified.
- `Derived`: calculated from explicit evidence; show the calculation.
- `Inherited`: delegated by the target paper to a cited source; name the path.
- `Ambiguous`: multiple plausible meanings or versions remain.
- `Unspecified`: the reachable paper and code evidence do not provide it.
- `Conflicting`: sources disagree; preserve both values and their provenance.

Never silently replace `Unspecified` with a common legged-gym value. Recommendations needed to implement missing items must be placed in a separate `Implementation choice` field.

## Report for implementation

For each requested item provide:

- value or equation, units, applicable condition, and confidence;
- primary label and exact source location;
- citation/code traversal path;
- contradiction or version caveat;
- a clearly separated implementation choice when reproduction requires a decision.

When writing an implementation plan, separate these sections:

1. reproduction target and acceptable deviations;
2. evidence-backed MDP and environment specification;
3. repository mapping to the current codebase;
4. staged implementation and validation plan;
5. unresolved questions that require the user to choose;
6. unresolved research items that Searcher can continue investigating.

Use local Markdown links for repository papers and normal HTTPS links for external paper or code sources. State the checked date for mutable public code.
