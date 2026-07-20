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
