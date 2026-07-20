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
