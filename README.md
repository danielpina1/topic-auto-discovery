# topic-auto-discovery

A [Claude Code](https://claude.com/claude-code) skill that runs **bottom-up
auto-discovery of a topic taxonomy** over call-center transcriptions stored in
BigQuery. You define *what* to discover (contact reasons, frictions, promise types,
complaints — any recurring concept); the skill discovers *which categories exist*,
collapses them into a meaningful taxonomy, and keeps refining it run after run through
a persistent collector memory.

```
transcriptions (BigQuery table)
        │ pull_and_shard.py (deterministic sample → 20-call chunks)
        ▼
  chunks ──► sonnet map agents (open-coding, 1 agent/chunk)      [Workflow]
        │                                    │
        │                                    ├──► opus group aggregators (1/group)
        │                                    │         groups/group_NN.json
        └──► embed_cluster.py (BGE-M3 → UMAP → HDBSCAN)
                  umap/clusters.json + scatter
                                             │
                                             ▼
                          collector (main session, fable)
                  serial merge into codebook.json (persistent memory)
                                             │
                                             ▼
                          reporter agent (report node)
              reports/DISCOVERY_REPORT.md + UMAP visual assets
```

## Features

- **Guided domain intake** — a short interview produces an approved `domain.md`
  (concept definition, unit of analysis, scope gates, evidence rule, taxonomy shape
  and caps, data source, sizing). One domain definition, reused across runs.
- **Deterministic BigQuery sampling** — `ORDER BY FARM_FINGERPRINT(id)` sampling makes
  every run reproducible; `--offset` feeds later runs fresh calls from the same window.
- **Map-reduce agent engine** — one sonnet agent open-codes each 20-call chunk (free
  vocabulary, verbatim evidence quotes, scope gating); one opus agent per 1,000-call
  group consolidates raw codes into an emergent group codebook; groups run
  concurrently; short chunks are automatically re-dispatched in-flight.
- **Persistent collector memory** — the main session merges group codebooks serially
  into `codebook.json`: match-on-meaning absorb/spawn/bridge rules, an end-of-run
  collapse pass, category lifecycle (`emerging → stable → merged`), tombstones instead
  of deletions, per-run counts, and **saturation tracking** ("no new stable categories
  this run" = your taxonomy is complete).
- **UMAP cross-check lane** — every detection is embedded locally (BGE-M3,
  multilingual), clustered with UMAP + HDBSCAN, and compared against the LLM taxonomy:
  over-splitting, incoherent categories, and missed topics show up as geometry. UMAP
  advises; the collector decides.
- **Report node** — a reporter agent writes `reports/DISCOVERY_REPORT.md`: the full
  cumulative account of the discovery — every methodology assumption made explicit,
  every category with definition/boundaries/evidence, embedded UMAP visualizations,
  cluster-vs-category agreement matrix, and the cross-run saturation picture.
- **Three-layer resumability** — chunk level (existing outputs skipped), workflow level
  (`resumeFromRunId`), collection level (merge journal). A crashed run resumes without
  double-counting.
- **Smoke mode** — the first run in a workspace is a mandatory 40-call mini-run that
  exercises the whole pipeline in minutes before you spend real tokens.

## Requirements

- Claude Code with subagent + Workflow support.
- An authenticated [`bq` CLI](https://cloud.google.com/bigquery/docs/bq-command-line-tool)
  with access to your transcription table.
- [`uv`](https://docs.astral.sh/uv/) (scripts are self-contained PEP 723 uv scripts).
- Disk/CPU note: the first real embed run downloads the BGE-M3 model (~2GB, one-time).

## Install

```bash
git clone https://github.com/danielpina1/topic-auto-discovery.git \
  ~/.claude/skills/topic-auto-discovery
```

## Usage

1. **Create a workspace** (one per client + domain) and invoke the skill in it:

   ```bash
   mkdir elal-reasons && cd elal-reasons
   claude
   > /topic-auto-discovery
   ```

2. **Intake** — answer the interview (concept, unit, gates, evidence, shape/caps,
   table id, sample size). The skill inspects your table schema with `bq show` and
   proposes the id/transcription columns. You approve the generated `domain.md`.

3. **Smoke run** — the skill runs a 40-call end-to-end mini-run first. Check the
   outputs look sane.

4. **Full run** — the real sample (default 5,000 calls ≈ 250 sonnet map agents +
   ~5 opus aggregators + collection). At the end you get:

   | Output | What it is |
   |---|---|
   | `codebook.json` | The persistent taxonomy memory (machine-readable) |
   | `CODEBOOK.md` | Human-readable taxonomy mirror |
   | `runs/<run_id>/RUN_REPORT.md` | This run's delta: new/merged/confirmed, distributions, UMAP agreement |
   | `reports/DISCOVERY_REPORT.md` | Cumulative full-domain report with UMAP visuals |
   | `reports/assets/` | UMAP scatter (PNG + interactive HTML), agreement matrix, family panels |

5. **Later runs** — invoke the skill in the same workspace: intake is skipped, new
   calls are pulled (use a new date window or the sample offset), and the collector
   updates the same codebook. Watch the `saturated` flag in the runs ledger — when new
   runs stop producing new stable categories, discovery is done.

### Workspace layout

```
<workspace>/
├── domain.md              # approved domain definition
├── codebook.json          # THE persistent collector memory
├── CODEBOOK.md            # human-readable mirror
├── reports/
│   ├── DISCOVERY_REPORT.md
│   └── assets/
└── runs/<run_id>/
    ├── chunks/            # sampled calls, 20/file
    ├── extractions/       # per-call open codes (JSONL)
    ├── groups/            # emergent group codebooks
    ├── umap/              # embeddings, clusters, plots
    └── RUN_REPORT.md
```

## FAQ

**How does "collapsing into meaningful categories" work?** Group aggregators cluster
raw open codes by meaning; the collector then matches each incoming cluster against the
persistent codebook — absorbing synonyms as aliases, spawning genuinely new categories
as `emerging`, and running a collapse pass (merge overlaps, enforce caps, promote
recurring categories to `stable`). Nothing is deleted: merged categories keep
tombstones, so counts and history survive.

**How do I reset the memory?** Delete `codebook.json` (and `CODEBOOK.md`). The next
run starts a fresh taxonomy. Git history keeps the old one.

**What does a 5k-call run cost?** ~250 sonnet chunk agents (low effort), ~5 opus
aggregators, one collection pass in the main session, and one reporter agent. Scale
the sample down if you just want a first look.

**Non-BigQuery sources?** Not in v1 — the contract is a `project.dataset.table` id.

## Development

```bash
uv run --with pytest --with numpy --with umap-learn --with hdbscan \
  --with plotly --with matplotlib python -m pytest tests/ -v
```

Design docs: `docs/superpowers/specs/` (approved design) and
`docs/superpowers/plans/` (implementation plan).
