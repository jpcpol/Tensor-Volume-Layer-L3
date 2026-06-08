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


def run_umap_sweep(X: np.ndarray, dims: list[int], seed: int = 42) -> dict:
    """
    Run UMAP for each target dimensionality.
    Returns dict: dim -> {embedding, trustworthiness}
    """
    results = {}
    for n in dims:
        reducer = UMAP(
            n_components=n,
            n_neighbors=min(5, len(X) - 1),  # small dataset: use n-1
            min_dist=0.1,
            random_state=seed,
            metric="euclidean",
        )
        embedding = reducer.fit_transform(X)
        tw = sklearn_trustworthiness(X, embedding, n_neighbors=min(5, len(X) - 1))
        results[n] = {"embedding": embedding, "trustworthiness": float(tw)}
        print(f"  dim={n}  trustworthiness={tw:.4f}")
    return results


def gate_decision(sweep_results: dict) -> dict:
    """
    Apply pre-registered gate decision rules.
    Returns decision dict with recommended_strategy and dim_gov.
    """
    # Find lowest dim where trustworthiness >= 0.85
    tw_threshold = 0.85
    borderline_threshold = 0.70

    dim_gov = None
    for dim in sorted(sweep_results.keys()):
        tw = sweep_results[dim]["trustworthiness"]
        if tw >= tw_threshold:
            dim_gov = dim
            break

    if dim_gov is not None and dim_gov <= 3:
        strategy = "TUCKER"
        verdict = "H_manifold CONFIRMED — dim(M_gov) <= 3 with trustworthiness >= 0.85"
        proceed = "Proceed S1 → S2 → S3 with Tucker decomposition"
    elif dim_gov is not None and dim_gov <= 5:
        tw_at_3 = sweep_results.get(3, {}).get("trustworthiness", 0.0)
        if tw_at_3 >= borderline_threshold:
            strategy = "TUCKER_CAUTIOUS"
            verdict = f"H_manifold BORDERLINE — dim(M_gov)={dim_gov}, trustworthiness at 3={tw_at_3:.4f}"
            proceed = "Tucker with rank sweep; report as borderline in paper; run S4 with synthetic corpus too"
        else:
            strategy = "SPARSE_SSM"
            verdict = f"H_manifold FAILED — dim(M_gov)={dim_gov} >= 4, trustworthiness at 3 < 0.70"
            proceed = "Switch to sparse representation + SSM-inspired gating. Update pre-registration."
    else:
        strategy = "SPARSE_SSM"
        dim_gov = max(sweep_results.keys())
        verdict = "H_manifold FAILED — no dim achieves trustworthiness >= 0.85 within sweep range"
        proceed = "Switch to sparse representation + SSM-inspired gating. Update pre-registration."

    return {
        "dim_gov": dim_gov,
        "strategy": strategy,
        "verdict": verdict,
        "proceed": proceed,
        "trustworthiness_by_dim": {d: v["trustworthiness"] for d, v in sweep_results.items()},
    }


def save_results(decision: dict, sweep_results: dict, labels: list[str], scenarios: list[str]) -> Path:
    """Save full results to JSON."""
    output = {
        "experiment": "S4 — Governance Manifold Test",
        "pre_registration": {
            "H_manifold": "dim(M_gov) <= 3",
            "acceptance_criterion": "trustworthiness >= 0.85 at dim <= 3",
            "commit_date": "2026-06-07",
        },
        "dataset": {
            "n_points": len(labels),
            "n_dims_ambient": 11,
            "scenarios": sorted(set(scenarios)),
            "artifacts": labels,
        },
        "sweep_dims": sorted(sweep_results.keys()),
        "trustworthiness": {str(d): v["trustworthiness"] for d, v in sweep_results.items()},
        "decision": decision,
    }
    out_path = OUTPUT_DIR / "manifold_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


def plot_results(sweep_results: dict, scenarios: list[str], labels: list[str]) -> None:
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

    # ── Trustworthiness curve ────────────────────────────────────────────────
    dims_sorted = sorted(sweep_results.keys())
    tw_values = [sweep_results[d]["trustworthiness"] for d in dims_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dims_sorted, tw_values, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.axhline(0.85, color="green", linestyle="--", linewidth=1, label="acceptance threshold (0.85)")
    ax.axhline(0.70, color="orange", linestyle="--", linewidth=1, label="borderline threshold (0.70)")
    for d, tw in zip(dims_sorted, tw_values):
        ax.annotate(f"{tw:.3f}", (d, tw), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("UMAP n_components (target dim)")
    ax.set_ylabel("Trustworthiness")
    ax.set_title("S4 — Trustworthiness vs. embedding dimensionality\n(pre-registered gate: dim<=3 with tw>=0.85)")
    ax.set_xticks(dims_sorted)
    ax.legend()
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
    print("  Trustworthiness by dim:")
    for d, v in sorted(sweep_results.items()):
        tw = v["trustworthiness"]
        bar = "#" * int(tw * 20)
        marker = " <-- pre-reg target" if d == 3 else ""
        print(f"    dim={d}  {tw:.4f}  {bar}{marker}")
    print()
    print(f"  dim(M_gov) estimate:  {decision['dim_gov']}")
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
    print(f"  Input matrix: {X.shape}  (n_points=12, ambient_dim=11)")
    print()

    # UMAP sweep
    print("Running UMAP sweep...")
    dims = [2, 3, 4, 5]
    sweep_results = run_umap_sweep(X, dims)

    # Gate decision
    decision = gate_decision(sweep_results)

    # Print summary
    print_summary(decision, sweep_results)

    # Save results
    out_path = save_results(decision, sweep_results, labels, scenarios)
    print(f"\n  Results saved: {out_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(sweep_results, scenarios, labels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
