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
