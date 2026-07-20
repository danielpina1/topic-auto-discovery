# Collector merge — absorb group codebooks into the persistent memory

You (the main session) are the collector. You own `codebook.json` — the ONLY cross-run
state. Work serially: one group at a time, in group-id order. Never parallelize merges.

## Before anything

1. `cp codebook.json codebook.json.bak` (first run: create `codebook.json` from the
   top-level schema below with empty `categories`/`runs`).
2. Append this run to `runs[]` with `"groups_merged": []` — that list is the journal; a
   group id present there is already merged. On resume, skip those groups.

Top-level schema:
`{"skill_version": "1.0", "client", "domain", "taxonomy_shape": "flat"|"two_level",
  "created", "updated", "caps", "runs": [], "categories": [], "segregated": [],
  "open_questions": []}`
Category schema:
`{"id", "family"|null, "sub"|null, "label", "definition",
  "status": "emerging"|"stable"|"merged"|"retired", "aliases": [], "evidence":
  [{"quote","call_id","run_id"}] (cap 5), "counts": {"total", "by_run": {}},
  "first_seen_run", "last_seen_run", "provenance": [{"run_id","group","raw_cluster"}]}`
Runs-ledger entry:
`{"run_id", "date", "table", "n_calls", "groups_merged": [], "new_categories",
  "merged", "confirmed", "saturated": false}`

## Per group (serial)

For each cluster in the group codebook, match ON MEANING (definitions + exemplars, never
name-string equality) against existing categories:

- **Absorb** — same concept as one existing category: `counts.total += count`,
  `counts.by_run[run_id] += count`, append cluster `name` to `aliases` (dedup), append
  `{run_id, group, raw_cluster: name}` to `provenance`, keep the 5 strongest `evidence`
  quotes overall, update `last_seen_run`.
- **Spawn** — genuinely new concept: create a category with `status: "emerging"`, an id
  in the domain's shape (`FAMILY[SUB]` if two_level, UPPER_SNAKE if flat), definition
  written from the cluster's definition + exemplars.
- **Bridge** — the cluster spans two existing categories: split its count by meaning
  across them (note the split in `open_questions` if uncertain) and record a merge
  proposal for the collapse pass.

Segregated gate counts roll into top-level `segregated[]` (same counts structure).
After each group: append its id to the journal, write `codebook.json`, re-validate:
unique category ids; every category has definition + ≥1 provenance; counts consistent.

## Collapse pass (after all groups)

1. Apply merge proposals + any pair of categories whose definitions describe one
   concept: keep the better id, absorb counts/aliases/evidence/provenance, loser gets
   `status: "merged"` and `definition: "→ merged into <winner id>"` (tombstone — never
   delete).
2. **UMAP cross-check** (`umap/clusters.json`): one embedding cluster spanning several
   categories → merge candidates; one category scattered across many clusters → refine
   or split its definition; a sizable cluster with no matching category → missed topic:
   spawn `emerging` from its exemplars. Record each signal + your decision for
   RUN_REPORT.md. UMAP advises; you decide.
3. Enforce caps from domain.md by merging nearest-meaning categories (never by deleting).
   Promotion: `emerging` → `stable` when seen in ≥2 groups or ≥2 runs.
4. Finish the ledger entry: `new_categories` (spawned this run, still present),
   `merged`, `confirmed` (absorbed into pre-existing), `saturated` = (new stable
   categories this run == 0). Update `updated`. Write `codebook.json`.
5. Regenerate `CODEBOOK.md`: one section per non-tombstone category — id, label, status,
   definition, aliases, total + per-run counts, 3 exemplar quotes; then a Tombstones
   table and the runs ledger.
