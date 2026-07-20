# Topic Discovery Skill — Design

**Date:** 2026-07-20
**Status:** Approved design, pending implementation plan
**Owner:** Daniel Pina

## Purpose

A personal Claude Code skill (`/topic-discovery`) that runs bottom-up (grounded-theory) auto-discovery of a user-defined concept ("domain") — e.g. contact reasons, frictions, promise types — over call-center transcriptions stored in BigQuery. It generalizes the proven ELAL discovery pipeline (Sonnet map → Opus consolidate → cross-batch merge) into a reusable, client-agnostic skill, replacing the one-shot cross-batch merge with a **persistent collector memory** (`codebook.json`) that incrementally absorbs batch aggregates, collapses overlapping categories into meaningful ones, and tracks saturation across runs.

## Decisions (locked with user)

1. **Input:** a BigQuery table id (`project.dataset.table`); the skill asks for it and inspects the schema.
2. **Domain definition:** created via **guided intake** interview → `domain.md`, approved by the user before the first run; reused on subsequent runs.
3. **Collector memory:** **persistent across runs** — a living codebook per client+domain with saturation tracking.
4. **Run size:** configurable deterministic sample (default 5,000 calls, `FARM_FINGERPRINT` ordering) per run.
5. **Taxonomy shape:** intake decides per domain — flat list or two-level `MAIN[SUB]`; the memory schema supports both.
6. **Orchestration:** Approach A — Workflow-orchestrated map-reduce; the **collector is the main session (fable-5)** doing serial memory merges.
7. **UMAP lane:** embeddings-based clustering (local sentence-transformers, BGE-M3) as an independent cross-check signal for the collector; advisory, not authoritative.

## Architecture

```
transcriptions (BQ table)
        │ pull_and_shard.py (deterministic sample → 20-call chunks)
        ▼
  chunks ──► sonnet-5 map agents (open-coding, 1 agent/chunk)   [Workflow]
        │                                    │
        │                                    ├──► opus-4.8 group aggregators (1/group, concurrent)
        │                                    │         groups/group_NN.json
        └──► embed_cluster.py (BGE-M3 → UMAP → HDBSCAN)
                  umap/clusters.json + scatter.html
                                             │
                                             ▼
                             fable-5 collector (main session)
                     serial merge into codebook.json (persistent memory)
                                             │
                                             ▼
                          RUN_REPORT.md + CODEBOOK.md + git commit
```

### Skill package — `~/.claude/skills/topic-discovery/`

```
topic-discovery/
├── SKILL.md                      # process: intake → ingest → engine → collect → report
├── templates/
│   └── domain.md                 # domain definition template the intake fills in
├── prompts/
│   ├── map_extraction.md         # sonnet-5 per-chunk open-coding (parametrized by domain.md)
│   ├── aggregate_group.md        # opus-4.8 per-group consolidation
│   └── collector_merge.md        # fable-5 memory-merge + collapse rules
├── scripts/
│   ├── pull_and_shard.py         # bq pull → deterministic sample → chunk files
│   ├── validate_extractions.py   # per-chunk completeness check
│   └── embed_cluster.py          # BGE-M3 embeddings → UMAP → HDBSCAN → cluster cards + scatter.html
├── workflow/
│   └── discovery_run.js          # Workflow template instantiated per run
└── docs/superpowers/specs/       # this spec
```

Scripts use `uv` for dependency management (sentence-transformers, umap-learn, hdbscan, numpy, plotly) so nothing pollutes system Python.

### Per-client workspace (created where the skill is invoked)

```
<workspace>/                      # e.g. ~/Desktop/elal-reasons/
├── domain.md                     # approved domain definition
├── codebook.json                 # THE persistent collector memory
├── CODEBOOK.md                   # human-readable mirror, regenerated each run
└── runs/<run_id>/
    ├── chunks/chunk_NNNN.jsonl
    ├── extractions/chunk_NNNN.out.jsonl
    ├── groups/group_NN.json
    ├── umap/{embeddings.npy, clusters.json, scatter.html}
    └── RUN_REPORT.md
```

Invoking the skill in a workspace that already has `domain.md` + `codebook.json` skips intake and starts a new run feeding the same memory.

## Guided intake

One question at a time (AskUserQuestion, multiple-choice where possible):

1. **Concept** — what to discover + 1-2 sentence working definition.
2. **Unit of analysis** — per call, or per event within call (multiple detections per call).
3. **Scope filters** — what to segregate (default-on gate for agent-internal coordination and unintelligible/non-call audio — the ~27% lesson from ELAL).
4. **Evidence rule** — verbatim quote required per detection (default on).
5. **Taxonomy shape** — flat or `MAIN[SUB]`, with target caps (e.g. ≤8 families / ≤6 subs each).
6. **Data source** — BQ table id; skill runs `bq show`, proposes transcription + call-id columns, asks for optional date column/window.
7. **Run sizing** — sample size (default 5,000) and language notes.

Answers land in `domain.md`; the user approves it; it parametrizes all three prompt templates.

## Engine (one run, defaults for 5k calls)

Defaults: 20 calls/chunk, 1,000 calls/group → 50 chunks/group, 5 groups.

1. **Ingest** — `pull_and_shard.py --table <id> --sample N` pulls the deterministic sample (`ORDER BY FARM_FINGERPRINT(<id_col>)`, optional date window/offset so later runs see fresh calls) and writes chunk files.
2. **Map (sonnet-5)** — one Workflow; per chunk, a sonnet-5 agent open-codes every call against `domain.md`: free-form `candidate_category` (+ optional candidate sub), verbatim evidence quote, unit-of-analysis fields, scope-gate verdict (segregated calls recorded but not coded). Appends to `extractions/chunk_NNNN.out.jsonl`, returns its call count. The workflow script re-dispatches any chunk returning fewer calls than expected (automates the ELAL 19/20 patch).
3. **Aggregate (opus-4.8)** — pipeline, no global barrier: when a group's chunks are done, one opus-4.8 agent clusters that group's raw codes into an emergent group codebook (clusters, frequencies, exemplar quotes) → `groups/group_NN.json`.
4. **UMAP lane (parallel with 3)** — `embed_cluster.py` embeds each detection (`candidate_category + evidence quote`, one vector per detection) with BGE-M3; UMAP (fixed `random_state`) to ~10-15 dims for HDBSCAN and to 2D for plotting; outputs cluster cards (size, top terms, 5 nearest-to-centroid exemplars). The interactive scatter (hover = quote) is rendered as the final collection step, once detections can be colored by final category via the raw-cluster → alias mapping in the codebook.
5. **Collect (fable-5, main session)** — after the workflow returns: run `validate_extractions.py`; patch residual gaps with direct agents; then merge group aggregates into `codebook.json` **serially, one group at a time**, with the UMAP cluster cards as cross-check evidence.

## Collector memory — `codebook.json`

Top level: `domain`, `client`, `taxonomy_shape`, timestamps, `runs[]` ledger, `categories[]`, `segregated[]`, `open_questions[]`.

Category entry: `id` (`FAMILY[SUB]` or flat), `label`, `definition`, `status` (`emerging → stable → merged/retired`), `aliases[]` (absorbed raw cluster names), capped exemplar `evidence[]` (quote + call_id + run_id), `counts` (per-run + total), `first_seen_run`, `last_seen_run`, `provenance[]` (run/group/raw-cluster).

`runs[]` ledger entry: run_id, date, table, n_calls, groups merged (the **collection journal**), new/merged/confirmed category counts — this is also what makes interrupted collection resumable without double-counting.

### Merge rules (`collector_merge.md`)

Per incoming group aggregate, each incoming cluster is matched **on meaning, not name** against existing categories; outcomes:
- **Absorb** — counts +=, add alias, keep best exemplars (cap enforced).
- **Spawn** — new `emerging` category.
- **Merge proposal** — the incoming cluster bridges two existing categories.

End-of-run **collapse pass**: merge overlapping categories; enforce caps from `domain.md`; rare singletons stay `emerging` (never silently deleted); merged categories keep tombstones (`status: merged` + alias link).

**UMAP cross-check** during collapse:
- one embedding cluster spanning several categories → over-splitting signal (merge candidates);
- one category scattered across many clusters → incoherent definition (split/refine candidate);
- dense cluster matching no category → possible missed topic.
Agreements/mismatches are reported in `RUN_REPORT.md`. The taxonomy remains LLM-authored — UMAP advises, the collector decides.

**Saturation:** the ledger tracks new-stable-categories-per-run; a run adding zero declares the taxonomy saturated in the report.

**Safety:** `codebook.json` backed up before collection; schema validated after every merge.

## Outputs per run

- `RUN_REPORT.md` — new/merged/confirmed categories, distributions, UMAP-vs-taxonomy agreement table, saturation verdict, segregated share, notable evidence.
- `umap/scatter.html` — interactive projection.
- Updated `CODEBOOK.md` + `codebook.json`.
- Workspace is a git repo (skill offers `git init` on first run); one commit per run → codebook diff history.

## Resumability & failure handling

Three ELAL layers kept: chunk-level (existing `.out.jsonl` skipped), workflow-level (`resumeFromRunId`), group-level (existing `groups/group_NN.json` skipped). Additions: in-workflow short-chunk re-dispatch; collection journal in the ledger; embedding cache (`umap/embeddings.npy`).

## Testing

- **Smoke mode** built into SKILL.md: before any full run, a 40-call mini-run (2 chunks → 1 group → UMAP → collection → report) exercises the whole pipeline in minutes.
- `validate_extractions.py` asserts per-chunk completeness.
- Collector validates codebook schema after each merge.
- Acceptance: first real use on the ELAL table with a "contact reasons" domain; discovered taxonomy checked for consistency with the known-good manual pipeline results.

## Non-goals (YAGNI)

- No production prompt generation (that is a downstream productionization step, as in ELAL).
- No dashboards beyond the UMAP scatter + markdown reports.
- No non-BigQuery ingestion in v1.
- No automatic scheduling; runs are user-invoked.
