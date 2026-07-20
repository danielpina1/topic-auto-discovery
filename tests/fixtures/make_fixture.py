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
