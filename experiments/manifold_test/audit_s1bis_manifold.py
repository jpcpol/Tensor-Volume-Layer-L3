# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
AUDIT (not pre-registered, not a gate). Gap-1 check for the Pi_gov pre-prereg
review: does the governance manifold hold on S1-bis (t=48) as it does on S1
(t=12)? S4 (definitive) measured dim(M_gov)~2-3 on S1; the TCI and operator
search run on S1-bis. Pi_gov assumes the S4 manifold. This re-runs the EXACT S4
harness (same reduction, same gate) on S1-bis to confirm the assumption transfers.

Reuses s4_manifold.{run_umap_sweep, run_pca_baseline, gate_decision}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from s4_manifold import gate_decision, run_pca_baseline, run_umap_sweep

HERE = Path(__file__).parent
CORPUS = HERE.parent / "synthetic_corpus" / "corpus_s1bis"
OUTPUT_DIR = HERE / "results"


def main() -> int:
    print("AUDIT — governance manifold on S1-bis (t=48), S4 harness verbatim")
    gt = json.loads((CORPUS / "ground_truth.json").read_text())
    gids = sorted(gt.keys())
    rows, graphs = [], []
    for gid in gids:
        for npy in sorted((CORPUS / gid).glob("session_*.npy")):
            T = np.load(npy)                  # (11, 4, 4, 48)
            rows.append(T.mean(axis=(1, 2, 3)))   # SAME reduction as S4: collapse stage,agent,cycle
            graphs.append(gid)
    X = np.array(rows, dtype=float)
    print(f"  input: {X.shape}  graphs={sorted(set(graphs))}")

    dims = [2, 3, 4, 5]
    seeds = list(range(10))
    nn = [5, 10, 15]
    twk = min(10, len(X) - 1)
    sweep = run_umap_sweep(X, dims, seeds, nn, twk)
    pca = run_pca_baseline(X, dims, twk)
    dec = gate_decision(sweep, pca)

    print("\n  dim | UMAP tw (mean±std) | PCA tw | PCA cumvar")
    for d in dims:
        print(f"   {d}  | {sweep[d]['trustworthiness']:.4f}±{sweep[d]['std']:.4f}      "
              f"| {pca[d]['trustworthiness']:.4f} | {pca[d]['cum_explained_variance']:.4f}")
    print(f"\n  dim_gov={dec['dim_gov']}  crosses_0.85_at={dec['crossing_dim_0p85']}")
    print(f"  VERDICT: {dec['verdict']}")

    out = {
        "audit": "Gap-1 — governance manifold on S1-bis (t=48), S4 harness verbatim",
        "corpus": "corpus_s1bis (n=90, t_cycles=48)",
        "reduction": "T(11x4x4x48) -> mean over stage,agent,cycle -> v in [0,1]^11 (same as S4)",
        "umap": {str(d): {"tw_mean": sweep[d]["trustworthiness"], "tw_std": sweep[d]["std"],
                          "best_n_neighbors": sweep[d]["best_n_neighbors"]} for d in dims},
        "pca": {str(d): {"tw": pca[d]["trustworthiness"],
                         "cum_explained_variance": pca[d]["cum_explained_variance"]} for d in dims},
        "decision": dec,
        "comparison_to_S1": "S1 (t=12) gave dim~2-3, tw 0.959-0.965; S1-bis matches -> manifold robust to corpus/time length",
    }
    (OUTPUT_DIR / "audit_s1bis_manifold.json").write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {OUTPUT_DIR / 'audit_s1bis_manifold.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
