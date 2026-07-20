#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "numba>=0.60",
#   "umap-learn>=0.5.5",
#   "hdbscan>=0.8.36",
#   "sentence-transformers>=3.0",
#   "plotly>=5.20",
#   "pandas>=2.0",
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
        cat = None
        if cluster:
            cat = cluster2cat.get((d["group"], cluster)) or cluster2cat.get((None, cluster))
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
