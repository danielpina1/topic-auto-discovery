---
name: topic-auto-discovery
description: Use when the user wants to discover what topics or categories exist in call-center transcriptions stored in a BigQuery table — building a bottom-up taxonomy of contact reasons, frictions, promises, complaints, or any recurring concept — or wants to feed new calls into a previously discovered taxonomy. Triggers: "discover topics", "auto discovery", "build a taxonomy from calls", "what are customers calling about", "/topic-auto-discovery".
---

# Topic Auto-Discovery

Bottom-up (grounded-theory) taxonomy discovery over call transcriptions: sonnet map
agents open-code calls → opus agents aggregate each group into an emergent codebook →
you (the collector) merge everything serially into a persistent, collapsing
`codebook.json` → a reporter agent writes the cumulative domain report with UMAP
visuals.

**SKILL_DIR** = the directory containing this file.
**WORKSPACE** = where this client+domain lives (default: cwd of invocation). All state
is in WORKSPACE; the ONLY cross-run state is `WORKSPACE/codebook.json`.

## 0. Workspace + intake

- If `WORKSPACE/domain.md` exists: confirm with the user, skip to step 1.
- Else run the guided intake — one AskUserQuestion at a time:
  1. concept + 1-2 sentence working definition;
  2. unit of analysis (`call` = at most one detection per call | `event` = several);
  3. scope gates (default ON: `agent_internal`, `unintelligible`);
  4. evidence rule (default ON: verbatim quote, original language);
  5. taxonomy shape (`flat` | `two_level`) + caps (e.g. ≤8 families / ≤6 subs);
  6. BigQuery table id — then run `bq show --schema <table>` and propose id/text/date
     columns for the user to confirm;
  7. sample size (default 5000) + language notes.
- Fill `SKILL_DIR/templates/domain.md` placeholders, write `WORKSPACE/domain.md`, ask
  the user to approve it. Offer `git init` if WORKSPACE is not a repo.

## 1. Run setup

- `RUN_ID=run_$(date +%Y%m%d_%H%M%S)`; `RUN_DIR=WORKSPACE/runs/$RUN_ID`.
- **First run in a workspace MUST be a smoke run:** `--sample 40 --group-size 40`, and
  use `--fake-embeddings` in step 6. Only after the smoke run completes end-to-end and
  the outputs look sane, start the full run (new RUN_ID).

## 2. Ingest

```bash
uv run SKILL_DIR/scripts/pull_and_shard.py --table <t> --id-col <id> --text-col <txt> \
  [--date-col X --start Y --end Z] [--where "<sql>"] \
  --sample <N> [--offset <k*N> when re-sampling the same window] --run-dir $RUN_DIR
```

## 3. Render prompts

Copy `prompts/map_extraction.md` + `prompts/aggregate_group.md` to `$RUN_DIR/prompts/`,
replacing every `{{PLACEHOLDER}}` from domain.md. `{{UNIT_RULE}}`: call → "emit at most
ONE detection per call (the dominant one)"; event → "emit one detection per distinct
instance in the call". `{{GROUP_SIZE}}` from manifest. `{{GROUP_ID_2D}}` stays literal —
the aggregator formats it. Verify: `grep -c '{{' $RUN_DIR/prompts/*.md` → 1 hit
(GROUP_ID_2D) in aggregate, 0 in map.

## 4. Map + aggregate (Workflow)

Invoke the Workflow tool with `scriptPath = SKILL_DIR/workflow/discovery_run.js` and
`args = {runDir, mapPrompt, aggPrompt, groups}` where `groups =
[{id, chunks: [{file, n}]}]` built from `$RUN_DIR/manifest.json` (chunks grouped by
their `group` field). On interruption: relaunch with `resumeFromRunId`.

## 5. Validate + patch

```bash
uv run SKILL_DIR/scripts/validate_extractions.py --run-dir $RUN_DIR
```

For any non-OK chunk, dispatch a direct sonnet Agent with the rendered map prompt in
PATCH MODE listing the missing ids. Re-run the validator until `"ok": true`.

## 6. UMAP lane

```bash
uv run SKILL_DIR/scripts/embed_cluster.py embed --run-dir $RUN_DIR   # --fake-embeddings on smoke runs
uv run SKILL_DIR/scripts/embed_cluster.py cluster --run-dir $RUN_DIR
```

First real run downloads BGE-M3 (~2GB, one-time).

## 7. Collect (you, serially)

Follow `SKILL_DIR/prompts/collector_merge.md` to the letter: backup, journal, per-group
serial merge in group-id order, UMAP cross-check, collapse pass, caps, saturation,
regenerate `CODEBOOK.md`. Never merge groups in parallel.

## 8. Render visuals

```bash
uv run SKILL_DIR/scripts/embed_cluster.py render --run-dir $RUN_DIR --workspace WORKSPACE
```

## 9. RUN_REPORT.md (you)

Write `$RUN_DIR/RUN_REPORT.md` — THIS run's delta only: new/merged/confirmed table,
category distributions, UMAP agreement summary + the decisions you made about its
signals, segregated share, saturation verdict, notable evidence.

## 10. Report node (subagent)

Dispatch one Agent (session model): "Follow SKILL_DIR/prompts/discovery_report.md.
WORKSPACE=<abs path>. Latest run=$RUN_ID." Verify its return JSON and that every image
it embedded exists in `WORKSPACE/reports/assets/`.

## 11. Commit

In WORKSPACE: `git add -A && git commit -m "discovery run $RUN_ID: <n_new> new,
<n_merged> merged, saturated=<bool>"`. Tell the user the taxonomy delta, saturation
status, and where `CODEBOOK.md`, `RUN_REPORT.md`, and `reports/DISCOVERY_REPORT.md`
live.

## Resumability

| Layer | Mechanism |
|---|---|
| Chunk | existing `extractions/*.out.jsonl` skipped (validator + patch agents) |
| Workflow | relaunch with `resumeFromRunId` |
| Group | existing `groups/group_NN.json` skipped |
| Collection | `groups_merged` journal in the codebook's `runs[]` ledger |
| Embeddings | reuse `umap/embeddings.npy` when detections are unchanged |

## Common mistakes

- Skipping the smoke run on a new workspace → pipeline bugs surface after 250 agents.
- Merging groups in parallel → corrupted codebook counts; collection is serial by design.
- Letting the aggregator drop `member_codes` → UMAP render can't map detections to
  categories (everything "unassigned").
- Deleting categories instead of tombstoning → cross-run memory loses history.
