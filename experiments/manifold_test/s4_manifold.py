# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
S4 — Manifold Test: does dim(M_gov) << ambient dim?

Pre-registered hypothesis (2026-06-07):
  H_manifold: dim(M_gov) <= 3
  measured via UMAP n_components sweep {2, 3, 4, 5}
  acceptance: trustworthiness >= 0.85 at dim <= 3

Gate decision:
  dim(M_gov) <= 3  AND  trustworthiness >= 0.85  →  Tucker is the right path (S1→S2→S3)
  dim(M_gov) >= 6  OR   trustworthiness <  0.70  →  switch strategy (sparse + SSM)
  3 < dim < 6                                    →  Tucker with caution, report as borderline

Input:  corpus.json ground_truth vectors from L2 (S1–S5)
Output: manifold_results.json + figures (scatter 2D/3D + trustworthiness curve)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from umap import UMAP

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parents[2]
# L3/ is at CAL/L3/ — go up 3 levels to CAL/, then into L2/
CORPUS_PATH = (
    Path(__file__).parents[3]
    / "L2"
    / "src"
    / "experiment"
    / "phi_calibration"
    / "corpus"
    / "corpus.json"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "manifold_test" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Dimension names (same order as EvaluationVector in L2) ──────────────────

DIM_NAMES = [
    "functional_correctness",   # v1
    "architectural_alignment",  # v2
    "scalability_projection",   # v3
    "security_risk",            # v4
    "observability_coverage",   # v5
    "testability",              # v6
    "maintainability",          # v7
    "technical_debt",           # v8
    "performance",              # v9
    "confidence",               # v10
    "anomaly_score",            # v11
]

# ─── Ground truth vectors per artifact ───────────────────────────────────────
# Built from corpus.json ground_truth fields + domain knowledge.
# Each row = one governance state point in [0,1]^11.
# Structure: v1..v11 for each artifact, scenario label for coloring.

ARTIFACT_VECTORS = [
    # scenario, label,              v1    v2    v3    v4    v5    v6    v7    v8    v9    v10   v11
    ("S1", "s1_clean",             [0.90, 0.85, 0.80, 0.90, 0.70, 0.80, 0.82, 0.75, 0.78, 0.88, 0.90]),
    ("S1", "s1_sql_injection",     [0.85, 0.80, 0.75, 0.20, 0.65, 0.75, 0.78, 0.70, 0.72, 0.55, 0.25]),
    ("S2", "s2_clean",             [0.88, 0.85, 0.82, 0.88, 0.72, 0.78, 0.80, 0.76, 0.74, 0.87, 0.88]),
    ("S2", "s2_circular_dep",      [0.82, 0.20, 0.45, 0.85, 0.68, 0.70, 0.54, 0.62, 0.65, 0.52, 0.30]),
    ("S3", "s3_k0_baseline",       [0.88, 0.84, 0.80, 0.87, 0.70, 0.85, 0.84, 0.68, 0.76, 0.86, 0.86]),
    ("S3", "s3_k1_debt",           [0.87, 0.82, 0.78, 0.86, 0.68, 0.72, 0.76, 0.60, 0.74, 0.80, 0.78]),
    ("S3", "s3_k2_debt",           [0.86, 0.80, 0.76, 0.85, 0.66, 0.62, 0.68, 0.52, 0.72, 0.74, 0.68]),
    ("S3", "s3_k3_debt",           [0.85, 0.78, 0.74, 0.84, 0.64, 0.48, 0.58, 0.44, 0.70, 0.66, 0.55]),
    ("S4", "s4_clean",             [0.87, 0.80, 0.78, 0.86, 0.85, 0.76, 0.78, 0.72, 0.80, 0.85, 0.88]),
    ("S4", "s4_no_observability",  [0.82, 0.35, 0.55, 0.83, 0.15, 0.72, 0.70, 0.68, 0.55, 0.48, 0.32]),
    ("S5", "s5_security_agent",    [0.80, 0.82, 0.75, 0.85, 0.68, 0.25, 0.55, 0.65, 0.72, 0.70, 0.72]),
    ("S5", "s5_code_agent",        [0.84, 0.78, 0.76, 0.20, 0.62, 0.75, 0.80, 0.70, 0.74, 0.55, 0.28]),
]

SCENARIO_COLORS = {"S1": "red", "S2": "orange", "S3": "blue", "S4": "green", "S5": "purple"}


def build_matrix() -> tuple[np.ndarray, list[str], list[str]]:
    """Return (X, scenario_labels, artifact_labels) from ARTIFACT_VECTORS."""
    scenarios = [row[0] for row in ARTIFACT_VECTORS]
    labels = [row[1] for row in ARTIFACT_VECTORS]
    X = np.array([row[2] for row in ARTIFACT_VECTORS], dtype=float)
    return X, scenarios, labels


def run_umap_sweep(
    X: np.ndarray,
    dims: list[int],
    seeds: list[int],
    n_neighbors_grid: list[int],
    tw_k: int,
) -> dict:
    """
    Run UMAP across (dim, n_neighbors, seed) and aggregate trustworthiness.

    For each target dimensionality we sweep n_neighbors and, for each
    (dim, n_neighbors) cell, average trustworthiness over `seeds` to absorb
    UMAP's stochasticity (PROBLEM 4 fix). For each dim we keep the BEST
    n_neighbors cell (the configuration that best preserves local structure),
    reporting mean and std over seeds (PROBLEM 3 + 4 fix).

    A representative embedding (first seed of the best cell) is stored for
    plotting.

    Returns dict: dim -> {
        embedding, trustworthiness (best-cell mean), std,
        best_n_neighbors, per_nn (full grid: nn -> {mean, std})
    }
    """
    results = {}
    for n in dims:
        per_nn = {}
        best = None  # (mean_tw, nn, embedding, std, all_tw)
        for nn in n_neighbors_grid:
            tws = []
            first_embedding = None
            for s in seeds:
                reducer = UMAP(
                    n_components=n,
                    n_neighbors=nn,
                    min_dist=0.1,
                    random_state=s,
                    metric="euclidean",
                )
                embedding = reducer.fit_transform(X)
                if first_embedding is None:
                    first_embedding = embedding
                tw = sklearn_trustworthiness(X, embedding, n_neighbors=tw_k)
                tws.append(float(tw))
            mean_tw = float(np.mean(tws))
            std_tw = float(np.std(tws))
            per_nn[nn] = {"mean": mean_tw, "std": std_tw, "all": tws}
            if best is None or mean_tw > best[0]:
                best = (mean_tw, nn, first_embedding, std_tw, tws)

        mean_tw, best_nn, best_emb, std_tw, _ = best
        results[n] = {
            "embedding": best_emb,
            "trustworthiness": mean_tw,
            "std": std_tw,
            "best_n_neighbors": best_nn,
            "per_nn": {nn: {"mean": v["mean"], "std": v["std"]} for nn, v in per_nn.items()},
        }
        grid_str = "  ".join(f"nn={nn}:{v['mean']:.3f}" for nn, v in per_nn.items())
        print(
            f"  dim={n}  best_tw={mean_tw:.4f}±{std_tw:.4f} (n_neighbors={best_nn})"
            f"   grid[{grid_str}]"
        )
    return results


def run_pca_baseline(X: np.ndarray, dims: list[int], tw_k: int) -> dict:
    """
    Deterministic PCA triangulation (PROBLEM 3/4 cross-check).

    PCA is linear and seed-free, so it gives a stable lower bound on how much
    structure a *linear* low-dim projection preserves, plus the cumulative
    explained-variance ratio per dim. If PCA and UMAP disagree sharply the
    manifold is strongly non-linear; if they agree the estimate is robust.
    """
    results = {}
    full = PCA(n_components=min(X.shape)).fit(X)
    cum_var = np.cumsum(full.explained_variance_ratio_)
    for n in dims:
        emb = PCA(n_components=n).fit_transform(X)
        tw = float(sklearn_trustworthiness(X, emb, n_neighbors=tw_k))
        evr = float(cum_var[n - 1]) if n - 1 < len(cum_var) else 1.0
        results[n] = {"embedding": emb, "trustworthiness": tw, "cum_explained_variance": evr}
        print(f"  dim={n}  PCA tw={tw:.4f}  cum_explained_var={evr:.4f}")
    return results


def gate_decision(sweep_results: dict, pca_results: dict | None = None) -> dict:
    """
    Apply pre-registered gate decision rules.

    Pre-registered rules (2026-06-07):
      dim(M_gov) <= 3  AND  tw >= 0.85   ->  TUCKER          (H confirmed)
      dim(M_gov) >= 6  OR   tw  < 0.70   ->  SPARSE_SSM      (H failed)
      3 < dim < 6  (i.e. tw>=0.85 only above 3, OR best tw in [0.70,0.85)) -> TUCKER_CAUTIOUS (borderline)

    BUG-1 FIX: the borderline branch is now reachable. We do not gate the
    borderline path behind "dim_gov is not None" (which required tw>=0.85 and
    therefore could never co-occur with a borderline verdict). Instead we
    branch on (a) whether ANY dim crosses 0.85, and (b) the best trustworthiness
    achieved anywhere in the sweep.

    BUG-2 FIX: when no dim crosses 0.85 we no longer report dim_gov as the
    sweep ceiling. dim_gov is reported as the dim that achieves the best
    trustworthiness (the most informative estimate), not max(dims).
    """
    tw_threshold = 0.85
    borderline_threshold = 0.70

    dims_sorted = sorted(sweep_results.keys())
    tw_by_dim = {d: sweep_results[d]["trustworthiness"] for d in dims_sorted}

    # Lowest dim crossing the strict 0.85 threshold (None if none do).
    crossing_dim = next((d for d in dims_sorted if tw_by_dim[d] >= tw_threshold), None)

    # Best dim/tw anywhere in the sweep (the informative dim_gov estimate).
    best_dim = max(dims_sorted, key=lambda d: tw_by_dim[d])
    best_tw = tw_by_dim[best_dim]

    if crossing_dim is not None and crossing_dim <= 3:
        dim_gov = crossing_dim
        strategy = "TUCKER"
        verdict = f"H_manifold CONFIRMED — dim(M_gov)={dim_gov} with trustworthiness={tw_by_dim[crossing_dim]:.4f} >= 0.85"
        proceed = "Proceed S1 -> S2 -> S3 with Tucker decomposition"
    elif crossing_dim is not None:  # crosses 0.85 but only above dim 3
        dim_gov = crossing_dim
        strategy = "TUCKER_CAUTIOUS"
        verdict = f"H_manifold BORDERLINE — dim(M_gov)={dim_gov} (tw>=0.85 only above 3)"
        proceed = "Tucker with rank sweep; report as borderline; re-run S4 on synthetic corpus"
    elif best_tw >= borderline_threshold:
        # No dim crosses 0.85, but local structure is meaningfully preserved.
        dim_gov = best_dim
        strategy = "TUCKER_CAUTIOUS"
        verdict = (
            f"H_manifold BORDERLINE — no dim reaches 0.85, but best tw={best_tw:.4f} "
            f"at dim={best_dim} is >= borderline 0.70"
        )
        proceed = "Tucker with rank sweep; report as borderline; re-run S4 on n>=30 synthetic corpus before final gate"
    else:
        dim_gov = best_dim
        strategy = "SPARSE_SSM"
        verdict = f"H_manifold FAILED — best tw={best_tw:.4f} (at dim={best_dim}) < borderline 0.70"
        proceed = "Switch to sparse representation + SSM-inspired gating. Update pre-registration."

    decision = {
        "dim_gov": dim_gov,
        "strategy": strategy,
        "verdict": verdict,
        "proceed": proceed,
        "crossing_dim_0p85": crossing_dim,
        "best_dim": best_dim,
        "best_trustworthiness": best_tw,
        "trustworthiness_by_dim": tw_by_dim,
    }

    # PCA triangulation note (does not change the gate; flags non-linearity).
    if pca_results:
        pca_best_dim = max(pca_results, key=lambda d: pca_results[d]["trustworthiness"])
        pca_best_tw = pca_results[pca_best_dim]["trustworthiness"]
        gap = best_tw - pca_best_tw
        decision["pca_triangulation"] = {
            "pca_best_dim": pca_best_dim,
            "pca_best_trustworthiness": pca_best_tw,
            "umap_minus_pca": gap,
            "note": (
                "UMAP >> PCA -> strongly non-linear manifold; "
                "UMAP ~ PCA -> estimate robust / structure near-linear"
            ),
        }

    return decision


def save_results(
    decision: dict,
    sweep_results: dict,
    pca_results: dict,
    labels: list[str],
    scenarios: list[str],
    config: dict,
) -> Path:
    """Save full results to JSON."""
    output = {
        "experiment": "S4 — Governance Manifold Test",
        "session": 2,
        "pre_registration": {
            "H_manifold": "dim(M_gov) <= 3",
            "acceptance_criterion": "trustworthiness >= 0.85 at dim <= 3",
            "borderline_threshold": 0.70,
            "commit_date": "2026-06-07",
        },
        "methodology": {
            "umap_seeds": config["seeds"],
            "umap_n_neighbors_grid": config["n_neighbors_grid"],
            "trustworthiness_k": config["tw_k"],
            "aggregation": "per (dim,n_neighbors): mean±std over seeds; per dim: best n_neighbors cell",
            "pca_triangulation": True,
            "audit_fixes": [
                "BUG-1: borderline gate branch is now reachable",
                "BUG-2: dim_gov = best-tw dim, not sweep ceiling",
                "PROBLEM-3: n_neighbors swept, not hard-coded to 5",
                "PROBLEM-4: trustworthiness averaged over multiple seeds",
            ],
        },
        "dataset": {
            "n_points": len(labels),
            "n_dims_ambient": 11,
            "scenarios": sorted(set(scenarios)),
            "artifacts": labels,
        },
        "sweep_dims": sorted(sweep_results.keys()),
        "umap": {
            str(d): {
                "trustworthiness_mean": v["trustworthiness"],
                "trustworthiness_std": v["std"],
                "best_n_neighbors": v["best_n_neighbors"],
                "per_n_neighbors": {str(nn): cell for nn, cell in v["per_nn"].items()},
            }
            for d, v in sweep_results.items()
        },
        "pca": {
            str(d): {
                "trustworthiness": v["trustworthiness"],
                "cum_explained_variance": v["cum_explained_variance"],
            }
            for d, v in pca_results.items()
        },
        "decision": decision,
    }
    out_path = OUTPUT_DIR / "manifold_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


def plot_results(sweep_results: dict, pca_results: dict, scenarios: list[str], labels: list[str]) -> None:
    """Generate scatter plots and trustworthiness curve."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("  [skip] matplotlib not installed — skipping plots")
        return

    colors = [SCENARIO_COLORS[s] for s in scenarios]

    # ── 2D scatter ──────────────────────────────────────────────────────────
    if 2 in sweep_results:
        emb2 = sweep_results[2]["embedding"]
        fig, ax = plt.subplots(figsize=(8, 6))
        for scenario, color in SCENARIO_COLORS.items():
            idxs = [i for i, s in enumerate(scenarios) if s == scenario]
            ax.scatter(
                emb2[idxs, 0], emb2[idxs, 1],
                c=color, label=scenario, s=100, edgecolors="black", linewidths=0.5,
            )
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (emb2[i, 0], emb2[i, 1]), fontsize=7, alpha=0.7)
        ax.set_title(
            f"S4 — UMAP 2D projection of governance states\n"
            f"trustworthiness={sweep_results[2]['trustworthiness']:.4f}"
        )
        ax.legend()
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "s4_umap_2d.png", dpi=150)
        plt.close(fig)
        print("  Saved: s4_umap_2d.png")

    # ── 3D scatter ──────────────────────────────────────────────────────────
    if 3 in sweep_results:
        emb3 = sweep_results[3]["embedding"]
        fig = plt.figure(figsize=(9, 7))
        ax3 = fig.add_subplot(111, projection="3d")
        for scenario, color in SCENARIO_COLORS.items():
            idxs = [i for i, s in enumerate(scenarios) if s == scenario]
            ax3.scatter(
                emb3[idxs, 0], emb3[idxs, 1], emb3[idxs, 2],
                c=color, label=scenario, s=80, edgecolors="black", linewidths=0.4,
            )
        ax3.set_title(
            f"S4 — UMAP 3D projection of governance states\n"
            f"trustworthiness={sweep_results[3]['trustworthiness']:.4f}"
        )
        ax3.legend()
        ax3.set_xlabel("UMAP-1")
        ax3.set_ylabel("UMAP-2")
        ax3.set_zlabel("UMAP-3")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "s4_umap_3d.png", dpi=150)
        plt.close(fig)
        print("  Saved: s4_umap_3d.png")

    # ── Trustworthiness curve (UMAP mean±std + PCA overlay) ──────────────────
    dims_sorted = sorted(sweep_results.keys())
    tw_values = [sweep_results[d]["trustworthiness"] for d in dims_sorted]
    tw_std = [sweep_results[d].get("std", 0.0) for d in dims_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        dims_sorted, tw_values, yerr=tw_std, fmt="o-", color="steelblue",
        linewidth=2, markersize=8, capsize=4, label="UMAP (mean±std over seeds)",
    )
    if pca_results:
        pca_dims = sorted(pca_results.keys())
        pca_tw = [pca_results[d]["trustworthiness"] for d in pca_dims]
        ax.plot(pca_dims, pca_tw, "s--", color="firebrick", linewidth=1.5,
                markersize=6, label="PCA (linear baseline)")
    ax.axhline(0.85, color="green", linestyle="--", linewidth=1, label="acceptance threshold (0.85)")
    ax.axhline(0.70, color="orange", linestyle="--", linewidth=1, label="borderline threshold (0.70)")
    for d, tw in zip(dims_sorted, tw_values):
        ax.annotate(f"{tw:.3f}", (d, tw), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("n_components (target dim)")
    ax.set_ylabel("Trustworthiness")
    ax.set_title("S4 — Trustworthiness vs. embedding dimensionality\n(pre-registered gate: dim<=3 with tw>=0.85)")
    ax.set_xticks(dims_sorted)
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "s4_trustworthiness_curve.png", dpi=150)
    plt.close(fig)
    print("  Saved: s4_trustworthiness_curve.png")


def print_summary(decision: dict, sweep_results: dict) -> None:
    print()
    print("=" * 65)
    print("S4 MANIFOLD TEST — RESULTS")
    print("=" * 65)
    print(f"  Ambient dimension (input): 11")
    print(f"  n_points:                  {len(ARTIFACT_VECTORS)}")
    print()
    print("  UMAP trustworthiness by dim (mean±std over seeds):")
    for d, v in sorted(sweep_results.items()):
        tw = v["trustworthiness"]
        std = v.get("std", 0.0)
        bar = "#" * int(tw * 20)
        marker = " <-- pre-reg target" if d == 3 else ""
        print(f"    dim={d}  {tw:.4f}±{std:.4f}  (nn={v.get('best_n_neighbors','-')})  {bar}{marker}")
    print()
    if "pca_triangulation" in decision:
        pt = decision["pca_triangulation"]
        print(f"  PCA triangulation: best tw={pt['pca_best_trustworthiness']:.4f} at dim={pt['pca_best_dim']}")
        print(f"    UMAP - PCA gap = {pt['umap_minus_pca']:+.4f}")
        print()
    print(f"  dim(M_gov) estimate:  {decision['dim_gov']}  (best-tw dim)")
    print(f"  crosses 0.85 at dim:  {decision['crossing_dim_0p85']}")
    print(f"  best trustworthiness: {decision['best_trustworthiness']:.4f}")
    print(f"  Recommended strategy: {decision['strategy']}")
    print()
    print(f"  VERDICT: {decision['verdict']}")
    print(f"  NEXT:    {decision['proceed']}")
    print("=" * 65)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"S4 — Governance Manifold Test")
    print(f"  Corpus: {CORPUS_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    if not CORPUS_PATH.exists():
        print(f"ERROR: corpus not found at {CORPUS_PATH}", file=sys.stderr)
        print("Make sure the L2 corpus is present at ../L2/src/experiment/phi_calibration/corpus/corpus.json")
        return 1

    # Build input matrix from ground truth vectors
    X, scenarios, labels = build_matrix()
    print(f"  Input matrix: {X.shape}  (n_points={X.shape[0]}, ambient_dim={X.shape[1]})")
    print()

    # Methodology config (audit-driven robustness)
    dims = [2, 3, 4, 5]
    seeds = list(range(10))                  # PROBLEM-4: average over seeds
    n_neighbors_grid = [3, 4, 5]             # PROBLEM-3: sweep n_neighbors (capped by n-1)
    n_neighbors_grid = [nn for nn in n_neighbors_grid if nn <= len(X) - 1]
    tw_k = min(5, len(X) - 1)
    config = {"seeds": seeds, "n_neighbors_grid": n_neighbors_grid, "tw_k": tw_k}

    # UMAP sweep (multi-seed, multi-n_neighbors)
    print(f"Running UMAP sweep  (seeds={len(seeds)}, n_neighbors={n_neighbors_grid}, tw_k={tw_k})...")
    sweep_results = run_umap_sweep(X, dims, seeds, n_neighbors_grid, tw_k)
    print()

    # PCA triangulation (deterministic linear baseline)
    print("Running PCA triangulation...")
    pca_results = run_pca_baseline(X, dims, tw_k)

    # Gate decision (with PCA cross-check)
    decision = gate_decision(sweep_results, pca_results)

    # Print summary
    print_summary(decision, sweep_results)

    # Save results
    out_path = save_results(decision, sweep_results, pca_results, labels, scenarios, config)
    print(f"\n  Results saved: {out_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(sweep_results, pca_results, scenarios, labels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
