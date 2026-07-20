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
