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
