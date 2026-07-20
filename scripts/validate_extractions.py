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
