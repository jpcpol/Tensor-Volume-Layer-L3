# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
S4 (definitive) — Manifold test on the n=90 synthetic causal corpus.

The §6.3 S4 run used the n=12 L2 ground-truth corpus, where PCA recovered a
3-dim manifold (tw=0.99) but UMAP underestimated it because n=12 is too sparse
for its k-NN graph. This is the definitive run: the S1 synthetic corpus provides
n=90 points (30 per causal graph G1/G2/G3), enough density for UMAP to either
corroborate or contradict the preliminary PCA result.

Each session tensor T^(s) ∈ ℝ^(11×4×4×12) is collapsed to its representative
governance vector v ∈ [0,1]^11 by averaging over stages, agents, and cycles —
the same "current snapshot" reduction used at L2 inference, but pooled across the
session. The 90 vectors form the input matrix; the sweep + gate logic is the
audited harness from s4_manifold.py (multi-seed UMAP + deterministic PCA).

Pre-registration (unchanged from 2026-06-07):
  H_manifold: dim(M_gov) <= 3
  acceptance: trustworthiness >= 0.85 at dim <= 3
  borderline: 0.70
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Reuse the audited harness (same directory).
from s4_manifold import (
    DIM_NAMES,
    gate_decision,
    run_pca_baseline,
    run_umap_sweep,
)

HERE = Path(__file__).parent
CORPUS_DIR = HERE.parent / "synthetic_corpus" / "corpus"
OUTPUT_DIR = HERE / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_COLORS = {"G1": "red", "G2": "blue", "G3": "green"}


def load_corpus() -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load all session tensors, collapse each to its representative vector.

    Returns (X[n,11], graph_labels, session_labels).
    """
    manifest = json.loads((CORPUS_DIR / "corpus_manifest.json").read_text())
    graph_ids = sorted(manifest["graphs"].keys())

    rows, graphs, labels = [], [], []
    for gid in graph_ids:
        gdir = CORPUS_DIR / gid
        for npy in sorted(gdir.glob("session_*.npy")):
            T = np.load(npy)                 # (11, 4, 4, 12)
            v = T.mean(axis=(1, 2, 3))       # collapse stage, agent, cycle -> (11,)
            rows.append(v)
            graphs.append(gid)
            labels.append(f"{gid}/{npy.stem}")
    X = np.array(rows, dtype=float)
    return X, graphs, labels


def plot_results(sweep_results: dict, pca_results: dict, graphs: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("  [skip] matplotlib not installed")
        return

    # 2D scatter colored by causal graph.
    if 2 in sweep_results:
        emb = sweep_results[2]["embedding"]
        fig, ax = plt.subplots(figsize=(8, 6))
        for gid, color in GRAPH_COLORS.items():
            idxs = [i for i, g in enumerate(graphs) if g == gid]
            ax.scatter(emb[idxs, 0], emb[idxs, 1], c=color, label=gid, s=45,
                       edgecolors="black", linewidths=0.3, alpha=0.8)
        ax.set_title(f"S4 definitive (n=90) — UMAP 2D by causal graph\n"
                     f"trustworthiness={sweep_results[2]['trustworthiness']:.4f}")
        ax.legend(); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / "s4def_umap_2d.png", dpi=150)
        plt.close(fig); print("  Saved: s4def_umap_2d.png")

    # 3D scatter.
    if 3 in sweep_results:
        emb = sweep_results[3]["embedding"]
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for gid, color in GRAPH_COLORS.items():
            idxs = [i for i, g in enumerate(graphs) if g == gid]
            ax.scatter(emb[idxs, 0], emb[idxs, 1], emb[idxs, 2], c=color, label=gid,
                       s=35, edgecolors="black", linewidths=0.3, alpha=0.8)
        ax.set_title(f"S4 definitive (n=90) — UMAP 3D by causal graph\n"
                     f"trustworthiness={sweep_results[3]['trustworthiness']:.4f}")
        ax.legend(); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / "s4def_umap_3d.png", dpi=150)
        plt.close(fig); print("  Saved: s4def_umap_3d.png")

    # Trustworthiness curve: UMAP mean±std + PCA overlay.
    dims = sorted(sweep_results.keys())
    tw = [sweep_results[d]["trustworthiness"] for d in dims]
    std = [sweep_results[d].get("std", 0.0) for d in dims]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(dims, tw, yerr=std, fmt="o-", color="steelblue", linewidth=2,
                markersize=8, capsize=4, label="UMAP (mean±std)")
    if pca_results:
        pd = sorted(pca_results.keys())
        ax.plot(pd, [pca_results[d]["trustworthiness"] for d in pd], "s--",
                color="firebrick", linewidth=1.5, markersize=6, label="PCA")
    ax.axhline(0.85, color="green", linestyle="--", linewidth=1, label="acceptance (0.85)")
    ax.axhline(0.70, color="orange", linestyle="--", linewidth=1, label="borderline (0.70)")
    for d, t in zip(dims, tw):
        ax.annotate(f"{t:.3f}", (d, t), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("n_components"); ax.set_ylabel("Trustworthiness")
    ax.set_title("S4 definitive (n=90) — trustworthiness vs. dim")
    ax.set_xticks(dims); ax.legend(fontsize=8); ax.set_ylim(0.5, 1.05)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "s4def_trustworthiness_curve.png", dpi=150)
    plt.close(fig); print("  Saved: s4def_trustworthiness_curve.png")


def main() -> int:
    print("S4 (definitive) — Manifold test on synthetic corpus (n=90)")
    print(f"  Corpus: {CORPUS_DIR}")
    print(f"  Output: {OUTPUT_DIR}\n")

    if not (CORPUS_DIR / "corpus_manifest.json").exists():
        print("ERROR: corpus not found. Run synthetic_corpus/causal_generator.py first.")
        return 1

    X, graphs, labels = load_corpus()
    print(f"  Input matrix: {X.shape}  (n_points={X.shape[0]}, ambient_dim={X.shape[1]})")
    print(f"  Graphs: {sorted(set(graphs))}\n")

    dims = [2, 3, 4, 5]
    seeds = list(range(10))
    n_neighbors_grid = [5, 10, 15]                      # denser corpus -> larger neighborhoods
    n_neighbors_grid = [nn for nn in n_neighbors_grid if nn <= len(X) - 1]
    tw_k = min(10, len(X) - 1)
    config = {"seeds": seeds, "n_neighbors_grid": n_neighbors_grid, "tw_k": tw_k}

    print(f"Running UMAP sweep (seeds={len(seeds)}, n_neighbors={n_neighbors_grid}, tw_k={tw_k})...")
    sweep_results = run_umap_sweep(X, dims, seeds, n_neighbors_grid, tw_k)
    print()
    print("Running PCA triangulation...")
    pca_results = run_pca_baseline(X, dims, tw_k)

    decision = gate_decision(sweep_results, pca_results)

    print("\n" + "=" * 65)
    print("S4 DEFINITIVE (n=90) — RESULTS")
    print("=" * 65)
    print("  UMAP trustworthiness by dim (mean±std):")
    for d in dims:
        v = sweep_results[d]
        bar = "#" * int(v["trustworthiness"] * 20)
        mk = " <-- pre-reg target" if d == 3 else ""
        print(f"    dim={d}  {v['trustworthiness']:.4f}±{v['std']:.4f} (nn={v['best_n_neighbors']})  {bar}{mk}")
    print("\n  PCA trustworthiness by dim:")
    for d in dims:
        p = pca_results[d]
        print(f"    dim={d}  {p['trustworthiness']:.4f}  cum_var={p['cum_explained_variance']:.4f}")
    print(f"\n  dim(M_gov) estimate:  {decision['dim_gov']}  (best-tw dim)")
    print(f"  crosses 0.85 at dim:  {decision['crossing_dim_0p85']}")
    print(f"  Recommended strategy: {decision['strategy']}")
    print(f"\n  VERDICT: {decision['verdict']}")
    print(f"  NEXT:    {decision['proceed']}")
    print("=" * 65)

    output = {
        "experiment": "S4 (definitive) — Manifold test on synthetic corpus",
        "corpus": "S1 synthetic (n=90, 30 per graph G1/G2/G3)",
        "pre_registration": {
            "H_manifold": "dim(M_gov) <= 3",
            "acceptance_criterion": "trustworthiness >= 0.85 at dim <= 3",
            "borderline_threshold": 0.70,
            "commit_date": "2026-06-07",
        },
        "methodology": {
            "vector_reduction": "T (11x4x4x12) -> mean over stage,agent,cycle -> v in [0,1]^11",
            "umap_seeds": seeds,
            "umap_n_neighbors_grid": n_neighbors_grid,
            "trustworthiness_k": tw_k,
            "pca_triangulation": True,
        },
        "dataset": {"n_points": int(X.shape[0]), "n_dims_ambient": int(X.shape[1]),
                    "graphs": sorted(set(graphs))},
        "umap": {str(d): {"trustworthiness_mean": v["trustworthiness"], "trustworthiness_std": v["std"],
                          "best_n_neighbors": v["best_n_neighbors"],
                          "per_n_neighbors": {str(nn): c for nn, c in v["per_nn"].items()}}
                 for d, v in sweep_results.items()},
        "pca": {str(d): {"trustworthiness": v["trustworthiness"],
                         "cum_explained_variance": v["cum_explained_variance"]}
                for d, v in pca_results.items()},
        "decision": decision,
    }
    out_path = OUTPUT_DIR / "manifold_results_definitive.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved: {out_path}")

    print("\nGenerating plots...")
    plot_results(sweep_results, pca_results, graphs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
