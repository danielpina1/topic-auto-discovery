# Topic Auto-Discovery Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `/topic-auto-discovery` Claude Code skill: guided-intake domain definition → BigQuery deterministic sample → sonnet-5 map / opus-4.8 group-aggregate Workflow → fable-5 collector with persistent `codebook.json` → UMAP cross-check lane → reporter node writing a cumulative `DISCOVERY_REPORT.md` with UMAP visuals.

**Architecture:** A skill package (SKILL.md process + parametrized prompt templates + uv PEP 723 Python scripts + a Workflow script driven entirely by `args`). All run state lives in a per-client workspace; the only cross-run state is `codebook.json`. See spec: `docs/superpowers/specs/2026-07-20-topic-discovery-design.md`.

**Tech Stack:** Claude Code skills, Workflow tool, `bq` CLI, Python 3.11+ via `uv` single-file scripts (numpy, umap-learn, hdbscan, sentence-transformers/BGE-M3, plotly, matplotlib), pytest.

## Global Constraints

- Repo root: `~/.claude/skills/topic-auto-discovery/` (remote `origin` = `github.com/danielpina1/topic-auto-discovery`, branch `main`).
- Python scripts are self-contained uv scripts (PEP 723 inline metadata), shebang `#!/usr/bin/env -S uv run --script`, `requires-python = ">=3.11"`.
- All heavy imports (`umap`, `hdbscan`, `sentence_transformers`, `plotly`, `matplotlib`) are lazy (inside functions) so tests import the modules with numpy only.
- Determinism: every UMAP call uses `random_state=42`; sampling uses `ORDER BY FARM_FINGERPRINT(CAST(<id> AS STRING))`.
- Model roster: map = `sonnet`, group aggregate = `opus`, collector = main session, reporter = session default (omit model).
- Workflow scripts must not use `Date.now()` / `Math.random()` (harness restriction) — run ids and timestamps are passed in via `args`.
- Extraction record schema (one JSON line per call): `{"call_id": str, "gate": "coded"|<gate-name>, "detections": [{"candidate_category": str, "candidate_sub": str, "evidence_quote": str}]}`.
- Tests run with: `cd ~/.claude/skills/topic-auto-discovery && uv run --with pytest --with numpy python -m pytest tests/ -v`.
- Commit after every task; push once at the end.

---

### Task 1: Domain definition template

**Files:**
- Create: `templates/domain.md`

**Interfaces:**
- Produces: the `{{PLACEHOLDER}}` vocabulary the intake fills and every prompt template references: `{{CLIENT}}`, `{{DOMAIN_NAME}}`, `{{DOMAIN_DEFINITION}}`, `{{UNIT}}`, `{{GATES}}`, `{{EVIDENCE_RULE}}`, `{{TAXONOMY_SHAPE}}`, `{{CAPS}}`, `{{TABLE}}`, `{{ID_COL}}`, `{{TEXT_COL}}`, `{{DATE_COL}}`, `{{WINDOW}}`, `{{WHERE}}`, `{{SAMPLE_SIZE}}`, `{{LANGUAGE_NOTES}}`.

- [ ] **Step 1: Write `templates/domain.md`**

```markdown
# Domain Definition — {{CLIENT}} / {{DOMAIN_NAME}}

Approved by user on: {{APPROVED_DATE}}

## Concept
**Name:** {{DOMAIN_NAME}}
**Working definition:** {{DOMAIN_DEFINITION}}

## Unit of analysis
{{UNIT}}
<!-- "call" (at most one detection per call) or "event" (a call can carry several detections) -->

## Scope gates (segregate — record but never code)
{{GATES}}
<!-- Default-on, one per line, format: gate_name — description. Baseline:
agent_internal — agent-to-agent / internal coordination, no customer on the line
unintelligible — noise, dead air, non-call audio, or transcript too corrupted to code -->

## Evidence rule
{{EVIDENCE_RULE}}
<!-- Default: every detection MUST carry a verbatim quote from the transcription, in its original language. -->

## Taxonomy shape
**Shape:** {{TAXONOMY_SHAPE}}   <!-- flat | two_level -->
**Caps:** {{CAPS}}              <!-- e.g. two_level: max 8 families, max 6 subs each. flat: max 25 categories -->

## Data source
- **Table:** `{{TABLE}}`
- **Call id column:** `{{ID_COL}}`
- **Transcription column:** `{{TEXT_COL}}`
- **Date column:** `{{DATE_COL}}` (blank if none)
- **Window:** {{WINDOW}}
- **Extra WHERE:** `{{WHERE}}` (blank if none)

## Run sizing
- **Sample size:** {{SAMPLE_SIZE}} (deterministic FARM_FINGERPRINT order; use --offset on later runs for fresh calls)
- **Chunk size:** 20 calls/agent · **Group size:** 1000 calls/group

## Language notes
{{LANGUAGE_NOTES}}
```

- [ ] **Step 2: Verify placeholder inventory**

Run: `grep -o '{{[A-Z_]*}}' templates/domain.md | sort -u`
Expected: exactly the 17 placeholders listed in Interfaces (plus `{{APPROVED_DATE}}`).

- [ ] **Step 3: Commit**

```bash
git add templates/domain.md && git commit -m "feat: domain definition template"
```

---

### Task 2: `pull_and_shard.py` (TDD)

**Files:**
- Create: `scripts/pull_and_shard.py`, `tests/conftest.py`, `tests/test_pull_and_shard.py`

**Interfaces:**
- Produces: `build_query(table, id_col, text_col, sample, offset=0, date_col=None, start=None, end=None, where=None) -> str`; `make_chunks(rows: list[dict], chunk_size: int) -> list[list[dict]]`; `write_run(rows, run_dir: Path, chunk_size: int, group_size: int, meta: dict) -> dict` (returns the manifest it wrote).
- Manifest schema consumed by Tasks 4, 9, 10: `{"run_id", "table", "sample", "offset", "chunk_size", "group_size", "n_calls", "chunks": [{"file": "chunk_0000.jsonl", "n": int, "group": int}], "groups": [{"id": int, "n_chunks": int, "n_calls": int}]}`.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
```

- [ ] **Step 2: Write failing tests `tests/test_pull_and_shard.py`**

```python
import json
import pull_and_shard as ps


def rows(n):
    return [{"call_id": f"c{i:04d}", "transcription": f"hello {i}"} for i in range(n)]


def test_build_query_deterministic_order_and_limit():
    q = ps.build_query("p.d.t", "name_id", "transcription", sample=100)
    assert "FROM `p.d.t`" in q
    assert "ORDER BY FARM_FINGERPRINT(CAST(name_id AS STRING))" in q
    assert q.rstrip().endswith("LIMIT 100")
    assert "AS call_id" in q and "AS transcription" in q


def test_build_query_window_where_offset():
    q = ps.build_query("p.d.t", "id", "txt", sample=10, offset=20,
                       date_col="create_date", start="2026-01-01", end="2026-02-01",
                       where="reason != 'x'")
    assert "create_date >= '2026-01-01'" in q
    assert "create_date < '2026-02-01'" in q
    assert "reason != 'x'" in q
    assert "LIMIT 10 OFFSET 20" in q


def test_make_chunks_sizes():
    chunks = ps.make_chunks(rows(45), 20)
    assert [len(c) for c in chunks] == [20, 20, 5]


def test_write_run_manifest_and_files(tmp_path):
    manifest = ps.write_run(rows(45), tmp_path, chunk_size=20, group_size=40,
                            meta={"run_id": "r1", "table": "p.d.t", "sample": 45, "offset": 0})
    files = sorted(p.name for p in (tmp_path / "chunks").glob("*.jsonl"))
    assert files == ["chunk_0000.jsonl", "chunk_0001.jsonl", "chunk_0002.jsonl"]
    assert manifest["n_calls"] == 45
    # groups: 40 calls/group => chunks 0,1 in group 0, chunk 2 in group 1
    assert [c["group"] for c in manifest["chunks"]] == [0, 0, 1]
    assert manifest["groups"][0]["n_calls"] == 40
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
    first = (tmp_path / "chunks" / "chunk_0000.jsonl").read_text().splitlines()
    assert json.loads(first[0]) == {"call_id": "c0000", "transcription": "hello 0"}
```

- [ ] **Step 3: Run tests, expect import failure**

Run: `uv run --with pytest --with numpy python -m pytest tests/test_pull_and_shard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pull_and_shard'`.

- [ ] **Step 4: Write `scripts/pull_and_shard.py`**

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Pull a deterministic sample from BigQuery and shard it into chunk files.

Usage:
  pull_and_shard.py --table p.d.t --id-col name_id --text-col transcription \
      --sample 5000 [--offset 0] [--date-col create_date --start 2026-01-01 --end 2026-07-01] \
      [--where "reason != 'x'"] --run-dir runs/run_20260720_1200 [--chunk-size 20] [--group-size 1000]

Requires an authenticated `bq` CLI. Writes chunks/chunk_NNNN.jsonl + manifest.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_query(table, id_col, text_col, sample, offset=0, date_col=None,
                start=None, end=None, where=None):
    conds = [f"{text_col} IS NOT NULL"]
    if date_col and start:
        conds.append(f"{date_col} >= '{start}'")
    if date_col and end:
        conds.append(f"{date_col} < '{end}'")
    if where:
        conds.append(where)
    limit = f"LIMIT {sample} OFFSET {offset}" if offset else f"LIMIT {sample}"
    return (
        f"SELECT CAST({id_col} AS STRING) AS call_id, {text_col} AS transcription\n"
        f"FROM `{table}`\n"
        f"WHERE {' AND '.join(conds)}\n"
        f"ORDER BY FARM_FINGERPRINT(CAST({id_col} AS STRING))\n"
        f"{limit}"
    )


def run_bq(query, max_rows):
    cmd = ["bq", "query", "--nouse_legacy_sql", "--format=json",
           f"--max_rows={max_rows}", query]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"bq query failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def make_chunks(rows, chunk_size):
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]


def write_run(rows, run_dir, chunk_size, group_size, meta):
    run_dir = Path(run_dir)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (run_dir / "extractions").mkdir(exist_ok=True)
    (run_dir / "groups").mkdir(exist_ok=True)
    (run_dir / "umap").mkdir(exist_ok=True)
    chunks = make_chunks(rows, chunk_size)
    chunks_per_group = max(1, group_size // chunk_size)
    chunk_entries = []
    for i, chunk in enumerate(chunks):
        name = f"chunk_{i:04d}.jsonl"
        with open(run_dir / "chunks" / name, "w") as f:
            for row in chunk:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        chunk_entries.append({"file": name, "n": len(chunk), "group": i // chunks_per_group})
    groups = []
    for gid in sorted({c["group"] for c in chunk_entries}):
        members = [c for c in chunk_entries if c["group"] == gid]
        groups.append({"id": gid, "n_chunks": len(members),
                       "n_calls": sum(c["n"] for c in members)})
    manifest = {**meta, "chunk_size": chunk_size, "group_size": group_size,
                "n_calls": len(rows), "chunks": chunk_entries, "groups": groups}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--date-col")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--where")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=1000)
    args = ap.parse_args()

    query = build_query(args.table, args.id_col, args.text_col, args.sample,
                        args.offset, args.date_col, args.start, args.end, args.where)
    print(query, file=sys.stderr)
    rows = run_bq(query, args.sample)
    if not rows:
        sys.exit("bq returned 0 rows — check table/window/filters")
    run_id = Path(args.run_dir).name
    manifest = write_run(rows, args.run_dir, args.chunk_size, args.group_size,
                         {"run_id": run_id, "table": args.table,
                          "sample": args.sample, "offset": args.offset})
    print(json.dumps({"run_id": run_id, "n_calls": manifest["n_calls"],
                      "n_chunks": len(manifest["chunks"]),
                      "n_groups": len(manifest["groups"])}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, expect pass**

Run: `uv run --with pytest --with numpy python -m pytest tests/test_pull_and_shard.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/pull_and_shard.py tests/ && git commit -m "feat: BigQuery pull + deterministic sharding with manifest"
```

---

### Task 3: `validate_extractions.py` (TDD)

**Files:**
- Create: `scripts/validate_extractions.py`, `tests/test_validate_extractions.py`

**Interfaces:**
- Consumes: manifest + chunk/extraction files from Task 2's layout.
- Produces: `validate_chunk(expected_ids: list[str], out_lines: list[str]) -> dict` with keys `status` (`"OK"|"MISSING_FILE"|"SHORT"|"BAD"`), `missing: list[str]`, `extra: list[str]`, `bad_lines: int`; CLI exit 0 iff every chunk OK; stdout = JSON `{"chunks": {name: result}, "ok": bool}` (re-dispatch input for SKILL.md step 5).

- [ ] **Step 1: Write failing tests `tests/test_validate_extractions.py`**

```python
import json
import validate_extractions as ve


def line(cid, gate="coded", dets=None):
    return json.dumps({"call_id": cid, "gate": gate, "detections": dets or []})


def test_ok_chunk():
    r = ve.validate_chunk(["a", "b"], [line("a"), line("b", gate="agent_internal")])
    assert r["status"] == "OK" and r["missing"] == [] and r["bad_lines"] == 0


def test_short_chunk_lists_missing():
    r = ve.validate_chunk(["a", "b", "c"], [line("a")])
    assert r["status"] == "SHORT" and r["missing"] == ["b", "c"]


def test_bad_json_line_counted():
    r = ve.validate_chunk(["a"], ["not-json"])
    assert r["status"] == "BAD" and r["bad_lines"] == 1 and r["missing"] == ["a"]


def test_extra_and_duplicate_ids_reported():
    r = ve.validate_chunk(["a"], [line("a"), line("a"), line("z")])
    assert r["extra"] == ["z"] and r["status"] == "OK"


def test_cli_run_dir(tmp_path):
    (tmp_path / "chunks").mkdir(); (tmp_path / "extractions").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"chunks": [{"file": "chunk_0000.jsonl", "n": 1, "group": 0}]}))
    (tmp_path / "chunks" / "chunk_0000.jsonl").write_text(
        json.dumps({"call_id": "a", "transcription": "t"}) + "\n")
    report = ve.validate_run(tmp_path)
    assert report["ok"] is False
    assert report["chunks"]["chunk_0000.jsonl"]["status"] == "MISSING_FILE"
```

- [ ] **Step 2: Run tests, expect ModuleNotFoundError**

Run: `uv run --with pytest --with numpy python -m pytest tests/test_validate_extractions.py -v`

- [ ] **Step 3: Write `scripts/validate_extractions.py`**

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Validate that every sampled call appears exactly once in the map extractions.

Usage: validate_extractions.py --run-dir runs/<run_id>
Exit 0 iff all chunks OK. Stdout: JSON report with per-chunk missing ids.
"""
import argparse
import json
import sys
from pathlib import Path


def validate_chunk(expected_ids, out_lines):
    seen, bad = [], 0
    for raw in out_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
            seen.append(str(rec["call_id"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
    missing = [i for i in expected_ids if i not in seen]
    extra = sorted(set(seen) - set(expected_ids))
    if bad:
        status = "BAD"
    elif missing:
        status = "SHORT"
    else:
        status = "OK"
    return {"status": status, "missing": missing, "extra": extra, "bad_lines": bad}


def validate_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    results = {}
    for entry in manifest["chunks"]:
        chunk_file = run_dir / "chunks" / entry["file"]
        expected = [str(json.loads(l)["call_id"])
                    for l in chunk_file.read_text().splitlines() if l.strip()]
        out_file = run_dir / "extractions" / entry["file"].replace(".jsonl", ".out.jsonl")
        if not out_file.exists():
            results[entry["file"]] = {"status": "MISSING_FILE", "missing": expected,
                                      "extra": [], "bad_lines": 0}
            continue
        results[entry["file"]] = validate_chunk(expected, out_file.read_text().splitlines())
    return {"chunks": results, "ok": all(r["status"] == "OK" for r in results.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    report = validate_run(args.run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, expect 5 passed**

Run: `uv run --with pytest --with numpy python -m pytest tests/test_validate_extractions.py -v`

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_extractions.py tests/test_validate_extractions.py \
  && git commit -m "feat: extraction completeness validator with re-dispatch report"
```

---

### Task 4: `embed_cluster.py` (TDD; embed/cluster/render subcommands)

**Files:**
- Create: `scripts/embed_cluster.py`, `tests/test_embed_cluster.py`, `tests/fixtures/` (synthetic mini-run)

**Interfaces:**
- Consumes: Task 2 layout (`manifest.json`, `extractions/*.out.jsonl`); for `render` also `<workspace>/codebook.json` (Task 6 schema) + `groups/group_NN.json` (Task 5 schema).
- Produces (files): `umap/embeddings.npy`, `umap/detections.json`, `umap/xy.npy`, `umap/labels.npy`, `umap/clusters.json` (cluster cards for the collector), `umap/scatter.html`, `umap/scatter_categories.png`, `umap/agreement_matrix.png`, `umap/families.png` (two_level only), copied into `<workspace>/reports/assets/`.
- Produces (functions): `load_detections(run_dir) -> list[dict]` (keys `call_id, chunk, group, candidate_category, candidate_sub, quote`); `fake_embed(texts, dim=64) -> np.ndarray`; `make_cluster_cards(emb, labels, dets, k_exemplars=5) -> dict`; `build_assignment(dets, groups_dir, codebook) -> list[str]` (final category id or `"unassigned"` per detection).

- [ ] **Step 1: Write fixture mini-run `tests/fixtures/make_fixture.py`** (checked in; also used by the Task 10 smoke test)

```python
"""Generate a tiny synthetic run dir + codebook for tests and smoke runs."""
import json
import random
from pathlib import Path

TOPICS = [
    ("refund status not received", "customer waiting for promised refund"),
    ("website login failure", "app or site blocks the customer"),
    ("seat assignment problem", "seat lost or cannot be chosen"),
]


def build(root: Path, n_calls=40, chunk_size=20):
    rng = random.Random(42)
    root = Path(root)
    (root / "chunks").mkdir(parents=True, exist_ok=True)
    (root / "extractions").mkdir(exist_ok=True)
    (root / "groups").mkdir(exist_ok=True)
    (root / "umap").mkdir(exist_ok=True)
    chunks = []
    for c in range((n_calls + chunk_size - 1) // chunk_size):
        name = f"chunk_{c:04d}.jsonl"
        with open(root / "chunks" / name, "w") as cf, \
             open(root / "extractions" / name.replace(".jsonl", ".out.jsonl"), "w") as xf:
            for i in range(chunk_size):
                cid = f"call_{c:02d}_{i:02d}"
                topic, blurb = TOPICS[rng.randrange(len(TOPICS))]
                cf.write(json.dumps({"call_id": cid, "transcription": f"... {blurb} ..."}) + "\n")
                xf.write(json.dumps({"call_id": cid, "gate": "coded", "detections": [
                    {"candidate_category": topic, "candidate_sub": "",
                     "evidence_quote": f"{blurb} #{i}"}]}) + "\n")
        chunks.append({"file": name, "n": chunk_size, "group": 0})
    manifest = {"run_id": "fixture", "table": "p.d.t", "sample": n_calls, "offset": 0,
                "chunk_size": chunk_size, "group_size": n_calls, "n_calls": n_calls,
                "chunks": chunks, "groups": [{"id": 0, "n_chunks": len(chunks), "n_calls": n_calls}]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "groups" / "group_00.json").write_text(json.dumps({
        "group": 0, "n_calls": n_calls, "n_detections": n_calls,
        "clusters": [
            {"cluster_id": "g00_c00", "name": "refund_not_received",
             "definition": "refund promised but not received", "count": 10,
             "member_codes": [{"code": "refund status not received", "count": 10}],
             "exemplars": [{"quote": "waiting for refund", "call_id": "call_00_01"}]},
            {"cluster_id": "g00_c01", "name": "digital_failure",
             "definition": "site/app blocks customer", "count": 10,
             "member_codes": [{"code": "website login failure", "count": 10}],
             "exemplars": [{"quote": "cannot log in", "call_id": "call_00_02"}]},
            {"cluster_id": "g00_c02", "name": "seat_problem",
             "definition": "seat lost or unavailable", "count": 10,
             "member_codes": [{"code": "seat assignment problem", "count": 10}],
             "exemplars": [{"quote": "lost my seat", "call_id": "call_00_03"}]}],
        "segregated": {}, "notes": []}, indent=2))
    codebook = {"skill_version": "1.0", "client": "fixture", "domain": "reasons",
                "taxonomy_shape": "two_level", "created": "2026-07-20", "updated": "2026-07-20",
                "caps": {"max_families": 8, "max_subs": 6},
                "runs": [], "segregated": [], "open_questions": [],
                "categories": [
                    {"id": "REFUND[STATUS_NOT_RECEIVED]", "family": "REFUND",
                     "sub": "STATUS_NOT_RECEIVED", "label": "Refund not received",
                     "definition": "promised refund missing", "status": "stable",
                     "aliases": ["refund_not_received"], "evidence": [],
                     "counts": {"total": 10, "by_run": {}}, "first_seen_run": "fixture",
                     "last_seen_run": "fixture",
                     "provenance": [{"run_id": "fixture", "group": 0,
                                     "raw_cluster": "refund_not_received"}]},
                    {"id": "DIGITAL[ACCESS_FAILURE]", "family": "DIGITAL",
                     "sub": "ACCESS_FAILURE", "label": "Digital access failure",
                     "definition": "site/app blocks customer", "status": "stable",
                     "aliases": ["digital_failure"], "evidence": [],
                     "counts": {"total": 10, "by_run": {}}, "first_seen_run": "fixture",
                     "last_seen_run": "fixture",
                     "provenance": [{"run_id": "fixture", "group": 0,
                                     "raw_cluster": "digital_failure"}]}]}
    (root.parent / "codebook.json").write_text(json.dumps(codebook, indent=2))
    return manifest


if __name__ == "__main__":
    import sys
    build(Path(sys.argv[1]))
```

- [ ] **Step 2: Write failing tests `tests/test_embed_cluster.py`**

```python
import json
import numpy as np
import pytest
import embed_cluster as ec
from fixtures.make_fixture import build


@pytest.fixture()
def run_dir(tmp_path):
    root = tmp_path / "ws" / "runs" / "fixture"
    build(root)
    return root


def test_load_detections(run_dir):
    dets = ec.load_detections(run_dir)
    assert len(dets) == 40
    assert set(dets[0]) == {"call_id", "chunk", "group",
                            "candidate_category", "candidate_sub", "quote"}


def test_fake_embed_deterministic_and_normalized():
    a = ec.fake_embed(["refund missing", "refund missing", "login broken"])
    assert a.shape == (3, 64)
    assert np.allclose(a[0], a[1])
    assert not np.allclose(a[0], a[2])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_make_cluster_cards_structure():
    emb = np.vstack([np.tile([1.0, 0.0], (6, 1)), np.tile([0.0, 1.0], (5, 1))])
    labels = np.array([0] * 6 + [1] * 5)
    dets = [{"call_id": f"c{i}", "chunk": "chunk_0000", "group": 0,
             "candidate_category": "x", "candidate_sub": "", "quote": f"q{i}"}
            for i in range(11)]
    cards = ec.make_cluster_cards(emb, labels, dets)
    assert [c["size"] for c in cards["clusters"]] == [6, 5]
    assert len(cards["clusters"][0]["exemplars"]) == 5
    assert cards["noise"] == 0 and cards["total"] == 11


def test_build_assignment_maps_via_groups_and_provenance(run_dir):
    dets = ec.load_detections(run_dir)
    codebook = json.loads((run_dir.parent.parent / "codebook.json").read_text())
    assigned = ec.build_assignment(dets, run_dir / "groups", codebook)
    cats = set(assigned)
    assert "REFUND[STATUS_NOT_RECEIVED]" in cats
    assert "DIGITAL[ACCESS_FAILURE]" in cats
    assert "unassigned" in cats  # seat topic has no category in fixture codebook
```

Also create `tests/fixtures/__init__.py` (empty) so the fixture module imports.

- [ ] **Step 3: Run tests, expect ModuleNotFoundError**

Run: `uv run --with pytest --with numpy python -m pytest tests/test_embed_cluster.py -v`

- [ ] **Step 4: Write `scripts/embed_cluster.py`**

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "umap-learn>=0.5.5",
#   "hdbscan>=0.8.36",
#   "sentence-transformers>=3.0",
#   "plotly>=5.20",
#   "matplotlib>=3.8",
# ]
# ///
"""UMAP lane: embed detections, cluster them, render visuals.

Subcommands:
  embed   --run-dir R [--fake-embeddings]     -> umap/embeddings.npy, umap/detections.json
  cluster --run-dir R [--min-cluster-size N]  -> umap/{xy,labels}.npy, umap/clusters.json
  render  --run-dir R --workspace W           -> umap/scatter.html + PNGs, copied to W/reports/assets/
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 42


# ---------- embed ----------

def load_detections(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    chunk_group = {c["file"].replace(".jsonl", ""): c["group"] for c in manifest["chunks"]}
    dets = []
    for out in sorted((run_dir / "extractions").glob("chunk_*.out.jsonl")):
        chunk = out.name.replace(".out.jsonl", "")
        for raw in out.read_text().splitlines():
            if not raw.strip():
                continue
            rec = json.loads(raw)
            if rec.get("gate") != "coded":
                continue
            for d in rec.get("detections", []):
                dets.append({"call_id": str(rec["call_id"]), "chunk": chunk,
                             "group": chunk_group.get(chunk, 0),
                             "candidate_category": d.get("candidate_category", ""),
                             "candidate_sub": d.get("candidate_sub", "") or "",
                             "quote": d.get("evidence_quote", "")})
    return dets


def detection_text(d):
    return f"{d['candidate_category']} | {d['candidate_sub']} | {d['quote']}".strip()


def fake_embed(texts, dim=64):
    import hashlib
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        grams = [t[j:j + 3] for j in range(max(len(t) - 2, 1))]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vecs[i, h % dim] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def real_embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")
    return np.asarray(model.encode(texts, normalize_embeddings=True,
                                   show_progress_bar=True), dtype=np.float32)


def cmd_embed(args):
    run_dir = Path(args.run_dir)
    dets = load_detections(run_dir)
    if not dets:
        raise SystemExit("no coded detections found — run the map phase first")
    texts = [detection_text(d) for d in dets]
    emb = fake_embed(texts) if args.fake_embeddings else real_embed(texts)
    np.save(run_dir / "umap" / "embeddings.npy", emb)
    (run_dir / "umap" / "detections.json").write_text(
        json.dumps(dets, ensure_ascii=False, indent=1))
    print(json.dumps({"n_detections": len(dets), "dim": int(emb.shape[1]),
                      "fake": bool(args.fake_embeddings)}))


# ---------- cluster ----------

TOKEN_RE = re.compile(r"[\w֐-׿']{3,}", re.UNICODE)


def top_terms(texts, k=8):
    counts = Counter(t.lower() for txt in texts for t in TOKEN_RE.findall(txt))
    return [w for w, _ in counts.most_common(k)]


def make_cluster_cards(emb, labels, dets, k_exemplars=5):
    cards = []
    for lab in sorted(set(int(l) for l in labels) - {-1}):
        idx = np.where(labels == lab)[0]
        centroid = emb[idx].mean(axis=0)
        order = np.argsort(((emb[idx] - centroid) ** 2).sum(axis=1))
        nearest = idx[order[:k_exemplars]]
        cards.append({"cluster_id": lab, "size": int(len(idx)),
                      "top_terms": top_terms([detection_text(dets[i]) for i in idx]),
                      "exemplars": [{"quote": dets[i]["quote"],
                                     "call_id": dets[i]["call_id"],
                                     "candidate_category": dets[i]["candidate_category"]}
                                    for i in nearest]})
    return {"clusters": cards, "noise": int((labels == -1).sum()), "total": int(len(labels))}


def cmd_cluster(args):
    import hdbscan
    import umap
    run_dir = Path(args.run_dir)
    emb = np.load(run_dir / "umap" / "embeddings.npy")
    dets = json.loads((run_dir / "umap" / "detections.json").read_text())
    n = len(dets)
    reduced = umap.UMAP(n_components=min(12, max(2, n - 2)), metric="cosine",
                        random_state=SEED).fit_transform(emb)
    min_size = args.min_cluster_size or max(5, n // 200)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_size).fit_predict(reduced)
    xy = umap.UMAP(n_components=2, metric="cosine", random_state=SEED).fit_transform(emb)
    np.save(run_dir / "umap" / "labels.npy", labels)
    np.save(run_dir / "umap" / "xy.npy", xy)
    cards = make_cluster_cards(emb, labels, dets)
    (run_dir / "umap" / "clusters.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=1))
    print(json.dumps({"n_clusters": len(cards["clusters"]), "noise": cards["noise"],
                      "min_cluster_size": min_size}))


# ---------- render ----------

def build_assignment(dets, groups_dir, codebook):
    code2cluster = {}
    for gf in sorted(Path(groups_dir).glob("group_*.json")):
        g = json.loads(gf.read_text())
        for cl in g.get("clusters", []):
            for mc in cl.get("member_codes", []):
                code2cluster[(g["group"], mc["code"].strip().lower())] = cl["name"]
    cluster2cat = {}
    for cat in codebook.get("categories", []):
        for prov in cat.get("provenance", []):
            cluster2cat[(prov["group"], prov["raw_cluster"])] = cat["id"]
        for alias in cat.get("aliases", []):
            cluster2cat.setdefault((None, alias), cat["id"])
    assigned = []
    for d in dets:
        cluster = code2cluster.get((d["group"], d["candidate_category"].strip().lower()))
        cat = (cluster2cat.get((d["group"], cluster))
               or cluster2cat.get((None, cluster)) if cluster else None)
        assigned.append(cat or "unassigned")
    return assigned


def cmd_render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.express as px
    run_dir, ws = Path(args.run_dir), Path(args.workspace)
    dets = json.loads((run_dir / "umap" / "detections.json").read_text())
    xy = np.load(run_dir / "umap" / "xy.npy")
    labels = np.load(run_dir / "umap" / "labels.npy")
    codebook = json.loads((ws / "codebook.json").read_text())
    cats = build_assignment(dets, run_dir / "groups", codebook)

    fig = px.scatter(x=xy[:, 0], y=xy[:, 1], color=cats,
                     hover_name=[d["candidate_category"] for d in dets],
                     hover_data={"quote": [d["quote"][:160] for d in dets]},
                     title=f"UMAP — {codebook['domain']} ({run_dir.name})")
    fig.write_html(run_dir / "umap" / "scatter.html", include_plotlyjs="cdn")

    uniq = sorted(set(cats))
    cmap = plt.get_cmap("tab20")
    color_of = {c: cmap(i % 20) for i, c in enumerate(uniq)}
    plt.figure(figsize=(9, 7))
    for c in uniq:
        m = np.array([x == c for x in cats])
        plt.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, color=color_of[c],
                    label=f"{c} ({int(m.sum())})")
    plt.legend(fontsize=6, loc="best", markerscale=1.5)
    plt.title(f"Detections by final category — {run_dir.name}")
    plt.tight_layout()
    plt.savefig(run_dir / "umap" / "scatter_categories.png", dpi=160)
    plt.close()

    clusters = sorted(set(int(l) for l in labels))
    matrix = np.zeros((len(clusters), len(uniq)), dtype=int)
    for l, c in zip(labels, cats):
        matrix[clusters.index(int(l)), uniq.index(c)] += 1
    plt.figure(figsize=(max(6, len(uniq) * 0.7), max(4, len(clusters) * 0.35)))
    plt.imshow(matrix, aspect="auto", cmap="Blues")
    plt.xticks(range(len(uniq)), uniq, rotation=60, ha="right", fontsize=6)
    plt.yticks(range(len(clusters)),
               [f"cluster {c}" if c >= 0 else "noise" for c in clusters], fontsize=6)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=5)
    plt.title("HDBSCAN clusters × final categories")
    plt.tight_layout()
    plt.savefig(run_dir / "umap" / "agreement_matrix.png", dpi=160)
    plt.close()

    if codebook.get("taxonomy_shape") == "two_level":
        fams = sorted({c.split("[")[0] for c in uniq if c != "unassigned"})
        if fams:
            cols = min(3, len(fams))
            rows = -(-len(fams) // cols)
            plt.figure(figsize=(4 * cols, 3.2 * rows))
            for i, fam in enumerate(fams, 1):
                ax = plt.subplot(rows, cols, i)
                ax.scatter(xy[:, 0], xy[:, 1], s=4, color="lightgray")
                m = np.array([x.startswith(fam + "[") for x in cats])
                ax.scatter(xy[m, 0], xy[m, 1], s=8, color="crimson")
                ax.set_title(f"{fam} ({int(m.sum())})", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(run_dir / "umap" / "families.png", dpi=160)
            plt.close()

    assets = ws / "reports" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for png in (run_dir / "umap").glob("*.png"):
        shutil.copy2(png, assets / png.name)
    shutil.copy2(run_dir / "umap" / "scatter.html", assets / "scatter.html")
    print(json.dumps({"categories": len(uniq),
                      "unassigned": int(sum(1 for c in cats if c == "unassigned")),
                      "assets": sorted(p.name for p in assets.iterdir())}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed"); e.add_argument("--run-dir", required=True)
    e.add_argument("--fake-embeddings", action="store_true"); e.set_defaults(fn=cmd_embed)
    c = sub.add_parser("cluster"); c.add_argument("--run-dir", required=True)
    c.add_argument("--min-cluster-size", type=int); c.set_defaults(fn=cmd_cluster)
    r = sub.add_parser("render"); r.add_argument("--run-dir", required=True)
    r.add_argument("--workspace", required=True); r.set_defaults(fn=cmd_render)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, expect all pass (numpy only — heavy imports are lazy)**

Run: `uv run --with pytest --with numpy python -m pytest tests/test_embed_cluster.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/embed_cluster.py tests/ && git commit -m "feat: UMAP lane — embed/cluster/render with fake-embeddings test mode"
```

---

### Task 5: Map + aggregate prompt templates

**Files:**
- Create: `prompts/map_extraction.md`, `prompts/aggregate_group.md`

**Interfaces:**
- Consumes: `{{...}}` values from the approved `domain.md` (Task 1). SKILL.md renders these templates into `runs/<run_id>/prompts/` before launching the workflow.
- Produces: the extraction record schema (Global Constraints) and the group codebook schema `{"group": int, "n_calls": int, "n_detections": int, "clusters": [{"cluster_id": "gNN_cNN", "name": snake_case, "definition": str, "count": int, "member_codes": [{"code": str, "count": int}], "exemplars": [{"quote": str, "call_id": str}]}], "segregated": {gate: int}, "notes": [str]}` consumed by Tasks 4, 6, 7.

- [ ] **Step 1: Write `prompts/map_extraction.md`**

````markdown
# Map extraction — open-code one chunk of calls

You are open-coding call-center transcriptions for bottom-up discovery. Do not use any
predefined taxonomy: name what you see, in the data's own terms.

**Domain:** {{DOMAIN_NAME}} — {{DOMAIN_DEFINITION}}
**Unit of analysis:** {{UNIT}}
**Language notes:** {{LANGUAGE_NOTES}}

Your dispatch message tells you the CHUNK FILE (input), the OUTPUT FILE, and the EXPECTED
number of calls.

## Procedure

1. Read the chunk file. Each line is `{"call_id": ..., "transcription": ...}`.
2. For EVERY call — no exceptions, even garbage — write exactly ONE JSON line to the
   output file (create it; if you were dispatched to patch specific missing call ids,
   APPEND only lines for those ids):
   `{"call_id": "...", "gate": "...", "detections": [...]}`
3. **Gate first.** Before coding, classify the call:
{{GATES}}
   If a gate applies, set `"gate"` to that gate's name and `"detections": []`. Otherwise
   `"gate": "coded"`.
4. **Code coded calls.** A detection is an instance of the domain concept:
   - `candidate_category`: a short free-form label (3-6 words, lowercase) in YOUR words,
     specific enough that two different problems never share a label.
   - `candidate_sub`: optional finer label, else `""`.
   - `evidence_quote`: {{EVIDENCE_RULE}}
   - Unit rule: {{UNIT_RULE}}
     <!-- rendered as: unit=call -> "emit at most ONE detection per call (the dominant
     one)"; unit=event -> "emit one detection per distinct instance in the call" -->
   A coded call with no instance of the concept gets `"detections": []` — that is a
   valid, common outcome. Never invent detections.
5. Self-check: count your output lines vs EXPECTED. Report any call ids you could not
   emit in `missing`.

## Return (this exact JSON, nothing else)

{"chunk": "<chunk file name>", "expected": <int>, "written": <int>, "missing": ["<call_id>", ...]}
````

- [ ] **Step 2: Write `prompts/aggregate_group.md`**

````markdown
# Group aggregation — consolidate raw codes into an emergent group codebook

You are consolidating open codes from ~{{GROUP_SIZE}} calls into emergent clusters.
Bottom-up: cluster by MEANING; do not force any external taxonomy.

**Domain:** {{DOMAIN_NAME}} — {{DOMAIN_DEFINITION}}

Your dispatch message lists this group's EXTRACTION FILES (JSONL), the GROUP ID, and the
OUTPUT FILE path.

## Procedure

1. Read every extraction file. Collect (a) all detections from `"gate": "coded"` lines,
   (b) counts per gate for segregated lines.
2. Cluster the detections' `candidate_category`/`candidate_sub`/`evidence_quote` by
   meaning. Aim for the natural grain: neither one mega-cluster nor a cluster per
   phrasing. Singletons that share no meaning with anything stay their own cluster.
3. For each cluster write:
   - `cluster_id`: `g{{GROUP_ID_2D}}_c<NN>` (e.g. `g00_c03`)
   - `name`: snake_case, ≤5 words, descriptive
   - `definition`: 1-2 sentences — what unites the members, boundary notes
   - `count`: number of member detections
   - `member_codes`: EVERY distinct `candidate_category` string absorbed, with its count
     — `[{"code": "...", "count": N}, ...]`. This mapping is required downstream; never
     omit or truncate it.
   - `exemplars`: 3-5 most representative `{"quote": ..., "call_id": ...}`
4. Write the OUTPUT FILE as JSON:
   `{"group": <int>, "n_calls": <int>, "n_detections": <int>, "clusters": [...],
     "segregated": {"<gate>": <count>}, "notes": ["anything surprising"]}`

## Return (this exact JSON, nothing else)

{"group": <int>, "n_clusters": <int>, "n_detections": <int>, "path": "<output file>"}
````

- [ ] **Step 3: Verify placeholders resolvable from domain.md + manifest**

Run: `grep -oh '{{[A-Z_0-9]*}}' prompts/map_extraction.md prompts/aggregate_group.md | sort -u`
Expected: `{{DOMAIN_DEFINITION}} {{DOMAIN_NAME}} {{EVIDENCE_RULE}} {{GATES}} {{GROUP_ID_2D}} {{GROUP_SIZE}} {{LANGUAGE_NOTES}} {{UNIT}} {{UNIT_RULE}}` — all derivable from `domain.md` (Task 1) or the manifest.

- [ ] **Step 4: Commit**

```bash
git add prompts/ && git commit -m "feat: map extraction + group aggregation prompt templates"
```

---

### Task 6: Collector merge prompt (`collector_merge.md`)

**Files:**
- Create: `prompts/collector_merge.md`

**Interfaces:**
- Consumes: group codebooks (Task 5 schema), `umap/clusters.json` (Task 4), `domain.md`.
- Produces: the canonical `codebook.json` schema (also consumed by Tasks 4, 7): top level `{"skill_version", "client", "domain", "taxonomy_shape", "created", "updated", "caps", "runs": [...], "categories": [...], "segregated": [...], "open_questions": [...]}`; category `{"id", "family"|null, "sub"|null, "label", "definition", "status": "emerging"|"stable"|"merged"|"retired", "aliases": [str], "evidence": [{"quote","call_id","run_id"}] (cap 5), "counts": {"total": int, "by_run": {run_id: int}}, "first_seen_run", "last_seen_run", "provenance": [{"run_id","group","raw_cluster"}]}`; runs-ledger entry `{"run_id","date","table","n_calls","groups_merged": [int],"new_categories","merged","confirmed","saturated": bool}`.

- [ ] **Step 1: Write `prompts/collector_merge.md`**

````markdown
# Collector merge — absorb group codebooks into the persistent memory

You (the main session) are the collector. You own `codebook.json` — the ONLY cross-run
state. Work serially: one group at a time, in group-id order. Never parallelize merges.

## Before anything

1. `cp codebook.json codebook.json.bak` (first run: create `codebook.json` from the
   top-level schema below with empty `categories`/`runs`).
2. Append this run to `runs[]` with `"groups_merged": []` — that list is the journal; a
   group id present there is already merged. On resume, skip those groups.

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
````

- [ ] **Step 2: Verify the schema block matches `tests/fixtures/make_fixture.py`'s codebook** (field-for-field; fix whichever is wrong).

- [ ] **Step 3: Commit**

```bash
git add prompts/collector_merge.md && git commit -m "feat: collector merge + collapse rules (persistent codebook)"
```

---

### Task 7: Reporter prompt (`discovery_report.md`) — the report node

**Files:**
- Create: `prompts/discovery_report.md`

**Interfaces:**
- Consumes: `domain.md`, `codebook.json`, `runs/<run_id>/umap/clusters.json`, `reports/assets/*.png|html` (Task 4 render), the runs ledger.
- Produces: `reports/DISCOVERY_REPORT.md` (cumulative, overwritten each run).

- [ ] **Step 1: Write `prompts/discovery_report.md`**

````markdown
# Discovery report — write the cumulative domain report

You are the reporter node. Write `reports/DISCOVERY_REPORT.md` for a reader who has
NEVER seen this project: a complete, self-contained account of the auto-discovery — its
assumptions, its taxonomy, and its evidence. Overwrite the previous version (git keeps
history). Your dispatch message gives the workspace path; read `domain.md`,
`codebook.json`, and the latest run's `umap/clusters.json` before writing.

Image embeds use relative paths from `reports/`: `![...](assets/scatter_categories.png)`.

## Required structure

1. **Executive summary** — the domain in one paragraph; taxonomy size and stability;
   headline findings; saturation verdict.
2. **Methodology & assumptions** — make every assumption explicit:
   - domain definition quoted verbatim from `domain.md`;
   - sampling design: table, window, FARM_FINGERPRINT determinism, sample size/offset
     per run (from the ledger);
   - unit of analysis and what it implies; scope gates and what they exclude (with
     segregated volumes); evidence rule;
   - pipeline: sonnet-5 open-coding → opus-4.8 group aggregation → fable-5 collector
     with persistent memory; UMAP/BGE-M3 cross-check lane (advisory);
   - taxonomy caps; known limitations and biases (sample ≠ population, LLM coder
     variance, embedding-model behavior on the corpus languages).
3. **The taxonomy** — for EVERY non-tombstone category: id, label, status, full
   definition WITH boundary description (what's in, what's out, adjacent categories),
   lifecycle (first/last seen run), absorbed aliases, total + per-run counts, 2-3
   verbatim exemplar quotes with call ids. Order by total count, grouped by family when
   two_level. Then a short Tombstones subsection (what merged into what, why).
4. **UMAP evidence** — embed `assets/scatter_categories.png`, `assets/families.png`
   (when present) and `assets/agreement_matrix.png`; link `assets/scatter.html`.
   Interpret: where geometry and taxonomy agree; every over-split / scatter /
   missed-topic signal and what the collector decided about it.
5. **Cross-run picture** — runs ledger table (run, date, calls, new, merged, confirmed,
   saturated); category-arrival timeline; saturation trajectory and expected next-run
   yield; `open_questions` from the codebook.
6. **Appendix** — file inventory of the workspace and how to reproduce a run.

Style: plain prose, no hedging, every claim traceable to codebook fields or assets.

## Return (this exact JSON, nothing else)

{"path": "reports/DISCOVERY_REPORT.md", "categories_documented": <int>, "images_embedded": <int>}
````

- [ ] **Step 2: Commit**

```bash
git add prompts/discovery_report.md && git commit -m "feat: reporter node prompt — cumulative discovery report"
```

---

### Task 8: Workflow script (`workflow/discovery_run.js`)

**Files:**
- Create: `workflow/discovery_run.js`

**Interfaces:**
- Consumes (as Workflow `args`): `{"runDir": abs path, "mapPrompt": abs path to rendered map prompt, "aggPrompt": abs path to rendered aggregate prompt, "groups": [{"id": int, "chunks": [{"file": "chunk_0000.jsonl", "n": 20}]}]}` — built by SKILL.md from `manifest.json`.
- Produces: `runs/<run_id>/extractions/*.out.jsonl`, `runs/<run_id>/groups/group_NN.json`; returns `{groups: [{group, n_clusters, n_detections, path}]}`.

- [ ] **Step 1: Write `workflow/discovery_run.js`**

```javascript
export const meta = {
  name: 'topic-discovery-run',
  description: 'Map (sonnet) + per-group aggregation (opus) for one topic-discovery run',
  whenToUse: 'Dispatched by the topic-auto-discovery skill; not for direct use.',
  phases: [
    { title: 'Map', detail: 'sonnet open-coding, one agent per 20-call chunk', model: 'sonnet' },
    { title: 'Aggregate', detail: 'opus emergent codebook per group', model: 'opus' },
  ],
}

const MAP_SCHEMA = {
  type: 'object',
  properties: {
    chunk: { type: 'string' },
    expected: { type: 'integer' },
    written: { type: 'integer' },
    missing: { type: 'array', items: { type: 'string' } },
  },
  required: ['chunk', 'expected', 'written', 'missing'],
}

const AGG_SCHEMA = {
  type: 'object',
  properties: {
    group: { type: 'integer' },
    n_clusters: { type: 'integer' },
    n_detections: { type: 'integer' },
    path: { type: 'string' },
  },
  required: ['group', 'n_clusters', 'n_detections', 'path'],
}

const { runDir, mapPrompt, aggPrompt, groups } = args

const outName = (f) => f.replace('.jsonl', '.out.jsonl')

const mapChunk = (c, extra) =>
  agent(
    `Follow the instructions in ${mapPrompt} EXACTLY.\n` +
      `CHUNK FILE: ${runDir}/chunks/${c.file}\n` +
      `OUTPUT FILE: ${runDir}/extractions/${outName(c.file)}\n` +
      `EXPECTED: ${c.n}\n` + (extra || ''),
    { label: `map:${c.file}`, phase: 'Map', model: 'sonnet', effort: 'low', schema: MAP_SCHEMA },
  )

const results = await pipeline(
  groups,
  async (g) => {
    const outs = (await parallel(g.chunks.map((c) => () => mapChunk(c)))).filter(Boolean)
    const short = outs.filter((o) => o.missing && o.missing.length > 0)
    for (const s of short) {
      const c = g.chunks.find((x) => x.file === s.chunk) || { file: s.chunk, n: s.expected }
      log(`re-dispatching ${s.chunk}: ${s.missing.length} missing`)
      await mapChunk(c, `PATCH MODE: the output file exists; APPEND lines ONLY for these missing call ids: ${s.missing.join(', ')}\n`)
    }
    return g
  },
  async (g) => {
    const gid = String(g.id).padStart(2, '0')
    const files = g.chunks.map((c) => `${runDir}/extractions/${outName(c.file)}`).join('\n')
    return agent(
      `Follow the instructions in ${aggPrompt} EXACTLY.\n` +
        `GROUP ID: ${g.id}\n` +
        `EXTRACTION FILES:\n${files}\n` +
        `OUTPUT FILE: ${runDir}/groups/group_${gid}.json\n`,
      { label: `aggregate:g${gid}`, phase: 'Aggregate', model: 'opus', schema: AGG_SCHEMA },
    )
  },
)

return { groups: results.filter(Boolean) }
```

- [ ] **Step 2: Syntax check**

Run: `node --input-type=module --check < workflow/discovery_run.js` — but `args`/`agent` are runtime globals, so only verify it PARSES: expected silent exit 0 (reference errors appear only at run time, not parse time).

- [ ] **Step 3: Commit**

```bash
git add workflow/discovery_run.js && git commit -m "feat: map+aggregate workflow with in-flight short-chunk re-dispatch"
```

---

### Task 9: `SKILL.md`

**Files:**
- Create: `SKILL.md`

**Interfaces:**
- Consumes: every artifact above. This file IS the orchestration: it tells the executing Claude the exact sequence and commands.

**Note for implementer:** invoke `superpowers:writing-skills` before authoring; keep frontmatter `name: topic-auto-discovery` and a trigger-rich `description`.

- [ ] **Step 1: Write `SKILL.md`** with frontmatter + the 11-step process below (full text in spec §Engine/§Report node; the file must contain, per step, the exact commands shown here):

```markdown
---
name: topic-auto-discovery
description: Use when the user wants bottom-up auto-discovery of a topic/category taxonomy (contact reasons, frictions, promises, any recurring concept) over call-center transcriptions in a BigQuery table. Runs guided domain intake, a sonnet map / opus group-aggregation workflow, a persistent collapsing codebook (collector memory), a UMAP cross-check, and writes a cumulative discovery report. Triggers: "discover topics", "auto discovery", "build a taxonomy from calls", "/topic-auto-discovery".
---

# Topic Auto-Discovery

SKILL_DIR = the directory containing this file. WORKSPACE = the directory where the
user wants this client+domain to live (default: cwd of invocation). All state is in
WORKSPACE; the ONLY cross-run state is WORKSPACE/codebook.json.

## 0. Workspace + intake
- If WORKSPACE/domain.md exists: confirm with the user, skip to step 1.
- Else run the guided intake — one AskUserQuestion at a time:
  concept+definition · unit (call|event) · scope gates (default agent_internal +
  unintelligible ON) · evidence rule (default verbatim quote ON) · taxonomy shape
  (flat|two_level)+caps · table id (then run `bq show --schema <table>` and propose
  id/text/date columns) · sample size (default 5000) + language notes.
- Fill SKILL_DIR/templates/domain.md placeholders, write WORKSPACE/domain.md, ask the
  user to approve it. Offer `git init` if WORKSPACE is not a repo.

## 1. Run setup
- RUN_ID=run_$(date +%Y%m%d_%H%M%S); RUN_DIR=WORKSPACE/runs/$RUN_ID
- First run in a workspace MUST be a smoke run: --sample 40 --group-size 40. Only after
  the smoke run completes end-to-end and looks sane, do the full run (new RUN_ID).

## 2. Ingest
uv run SKILL_DIR/scripts/pull_and_shard.py --table <t> --id-col <id> --text-col <txt> \
  [--date-col/--start/--end/--where from domain.md] --sample <N> [--offset <k*N> on
  re-runs against the same window] --run-dir $RUN_DIR

## 3. Render prompts
Copy prompts/map_extraction.md + aggregate_group.md to $RUN_DIR/prompts/, replacing
every {{PLACEHOLDER}} from domain.md ({{UNIT_RULE}}: call→"at most ONE detection per
call"; event→"one detection per distinct instance"; {{GROUP_ID_2D}} stays literal — the
aggregator formats it). Verify: `grep -c '{{' $RUN_DIR/prompts/*.md` → 1 hit
(GROUP_ID_2D) in aggregate, 0 in map.

## 4. Map + aggregate (Workflow)
Invoke the Workflow tool with scriptPath=SKILL_DIR/workflow/discovery_run.js and
args={runDir, mapPrompt, aggPrompt, groups} built from $RUN_DIR/manifest.json
(groups[].chunks from manifest chunks by group id). On interruption: resumeFromRunId.

## 5. Validate + patch
uv run SKILL_DIR/scripts/validate_extractions.py --run-dir $RUN_DIR
For any non-OK chunk, dispatch a direct sonnet Agent with the map prompt in PATCH MODE
listing the missing ids; re-run until ok=true.

## 6. UMAP lane
uv run SKILL_DIR/scripts/embed_cluster.py embed --run-dir $RUN_DIR   # add --fake-embeddings on smoke runs
uv run SKILL_DIR/scripts/embed_cluster.py cluster --run-dir $RUN_DIR

## 7. Collect (you, serially)
Follow SKILL_DIR/prompts/collector_merge.md to the letter: backup, journal, per-group
serial merge in group order, UMAP cross-check, collapse, caps, saturation, CODEBOOK.md.

## 8. Render visuals
uv run SKILL_DIR/scripts/embed_cluster.py render --run-dir $RUN_DIR --workspace WORKSPACE

## 9. RUN_REPORT.md (you)
Write $RUN_DIR/RUN_REPORT.md — THIS run's delta only: new/merged/confirmed table,
distributions, UMAP agreement summary + decisions, segregated share, saturation verdict,
notable evidence.

## 10. Report node (subagent)
Dispatch one Agent (session model): "Follow SKILL_DIR/prompts/discovery_report.md.
WORKSPACE=<abs path>. Latest run=$RUN_ID." Verify its return JSON; confirm the images it
embedded exist.

## 11. Commit
In WORKSPACE: git add -A && git commit -m "discovery run $RUN_ID: <n_new> new,
<n_merged> merged, saturated=<bool>". Tell the user: taxonomy delta, saturation status,
where the reports live.

## Resumability
Chunk: existing *.out.jsonl skipped (validator + patch). Workflow: resumeFromRunId.
Group: existing groups/group_NN.json skipped. Collection: journal in runs[] ledger.
Embeddings: reuse umap/embeddings.npy when detections unchanged.
```

- [ ] **Step 2: Verify frontmatter parses** (`head -5 SKILL.md` shows `---`, `name:`, `description:`) and every referenced path exists in the repo.

- [ ] **Step 3: Commit**

```bash
git add SKILL.md && git commit -m "feat: SKILL.md orchestration process"
```

---

### Task 10: Fixture smoke test (fake end-to-end)

**Files:**
- Create: `tests/test_smoke_pipeline.py`

**Interfaces:**
- Consumes: `tests/fixtures/make_fixture.py`, `validate_extractions.validate_run`, `embed_cluster` embed/cluster/render mains.

- [ ] **Step 1: Write `tests/test_smoke_pipeline.py`**

```python
"""End-to-end (minus LLM agents): fixture run -> validate -> embed(fake) -> cluster -> render."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
from fixtures.make_fixture import build  # noqa: E402


def run(script, *argv):
    proc = subprocess.run([sys.executable, str(SCRIPTS / script), *argv],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_full_fake_pipeline(tmp_path):
    ws = tmp_path / "ws"
    run_dir = ws / "runs" / "fixture"
    build(run_dir)

    import validate_extractions as ve
    assert ve.validate_run(run_dir)["ok"] is True

    run("embed_cluster.py", "embed", "--run-dir", str(run_dir), "--fake-embeddings")
    run("embed_cluster.py", "cluster", "--run-dir", str(run_dir), "--min-cluster-size", "5")
    cards = json.loads((run_dir / "umap" / "clusters.json").read_text())
    assert cards["total"] == 40 and len(cards["clusters"]) >= 1

    run("embed_cluster.py", "render", "--run-dir", str(run_dir), "--workspace", str(ws))
    assets = ws / "reports" / "assets"
    assert (assets / "scatter_categories.png").exists()
    assert (assets / "agreement_matrix.png").exists()
    assert (assets / "scatter.html").exists()
```

- [ ] **Step 2: Run the full suite (needs the heavy deps this time)**

Run: `uv run --with pytest --with numpy --with umap-learn --with hdbscan --with plotly --with matplotlib python -m pytest tests/ -v`
Expected: all tests pass (test_smoke_pipeline exercises umap/hdbscan/plotly/matplotlib; `sys.executable` inside the uv env sees them).

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_pipeline.py && git commit -m "test: fake-embeddings end-to-end smoke of the non-LLM pipeline"
```

---

### Task 11: `README.md` + `.gitignore` + push

**Files:**
- Modify: `README.md` (replace the one-line stub)
- Create: `.gitignore`

- [ ] **Step 1: Write `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 2: Write `README.md`** covering, in this order: what it is (1 paragraph + the architecture diagram from the spec, as ASCII); features list (guided intake, deterministic BQ sampling, sonnet/opus map-reduce workflow with auto re-dispatch, persistent collapsing codebook + saturation, UMAP/BGE-M3 cross-check, reporter node with UMAP visuals, 3-layer resumability, smoke mode); requirements (Claude Code, authed `bq`, `uv`; first embed run downloads BGE-M3 ~2GB); install (`git clone https://github.com/danielpina1/topic-auto-discovery.git ~/.claude/skills/topic-auto-discovery`); usage walkthrough (mkdir workspace → `/topic-auto-discovery` → intake → smoke run → full run → what each output file is → re-runs with `--offset` and saturation); workspace layout tree (from spec); FAQ (how collapsing works, how to reset memory = delete codebook.json, cost expectations ~250 sonnet + 5 opus agents per 5k run); development (run tests command).

- [ ] **Step 3: Verify README links/paths** — every path mentioned exists; clone URL matches `git remote -v`.

- [ ] **Step 4: Commit and push everything**

```bash
git add README.md .gitignore && git commit -m "docs: README — features and usage"
git push origin main
```

---

## Self-Review (run after writing, fix inline)

1. **Spec coverage:** intake ✓(T9§0) ingest ✓(T2) map/aggregate workflow ✓(T5,T8) validator ✓(T3) UMAP lane ✓(T4) collector+codebook ✓(T6) report node ✓(T7) RUN_REPORT ✓(T9§9) README ✓(T11) smoke mode ✓(T9§1,T10) resumability ✓(T8,T9§Resumability).
2. **Placeholder scan:** none — every file's full content is in its task.
3. **Type consistency:** manifest keys (T2↔T4↔T8↔T9), extraction schema (Global↔T3↔T4↔T5), group codebook (T5↔T4.build_assignment↔T6), codebook (T6↔T4 fixture) — verified field-for-field.
