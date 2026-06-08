# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
S3 — Causal conservation test (Property 1 of C).

Pre-registered in PRE_REGISTRATION.md (commit 1a3076d, 2026-06-08). This code
implements that protocol verbatim. Do not change parameters post-hoc; if a
revision is needed, amend the pre-registration in a new commit first.

Question: does the Tucker composition operator C preserve the causal structure
between quality dimensions? We run pairwise Granger causality on the Tucker
RECONSTRUCTION V̂ and score recovered edges against the ground truth. As an
attribution control we run the identical method on the RAW corpus.

H_causal: micro-F1 (on V̂) >= 0.70.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

from sys import path as _sys_path
_sys_path.insert(0, str(Path(__file__).parent.parent / "tucker_composition"))
from tucker_operator import TuckerCompositionOperator  # noqa: E402

warnings.filterwarnings("ignore")  # statsmodels emits many runtime warnings

HERE = Path(__file__).parent
CORPUS_DIR = HERE.parent / "synthetic_corpus" / "corpus"
OUTPUT_DIR = HERE / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Pre-registered fixed parameters ─────────────────────────────────────────
TUCKER_RANK = (3, 3, 3, 3, 6)   # S2 dim-mode-justified rank
MAXLAG = 1                       # injected causal lag
ALPHA = 0.01                     # edge significance threshold
N_DIMS = 11

PASS_THRESHOLD = 0.70
PARTIAL_THRESHOLD = 0.50


def load_graph_sessions(gid: str) -> list[np.ndarray]:
    return [np.load(p) for p in sorted((CORPUS_DIR / gid).glob("session_*.npy"))]


def to_dim_series(T: np.ndarray) -> np.ndarray:
    """T (11×4×4×12) -> per-dim trajectory (t × 11): mean over stage, agent."""
    return T.mean(axis=(1, 2)).T  # (t, 11)


def granger_pvalue(child: np.ndarray, parent: np.ndarray, maxlag: int) -> float:
    """
    p-value that `parent` Granger-causes `child` (F-test, ssr-based).

    statsmodels expects a 2-col array [child, parent]; it tests whether the
    second column helps predict the first.
    """
    data = np.column_stack([child, parent])
    try:
        res = grangercausalitytests(data, maxlag=[maxlag], verbose=False)
        return float(res[maxlag][0]["ssr_ftest"][1])
    except TypeError:
        # Older/newer statsmodels dropped the verbose kwarg; suppress stdout.
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            res = grangercausalitytests(data, maxlag=[maxlag])
        return float(res[maxlag][0]["ssr_ftest"][1])
    except Exception:
        return 1.0  # non-estimable -> treat as no evidence


def discover_edges(series_list: list[np.ndarray], maxlag: int, alpha: float) -> dict:
    """
    Pairwise Granger over all ordered dim pairs, pooling per-session trajectories.

    Returns dict: (i, j) -> p-value, plus the predicted edge set (p < alpha).
    """
    # Pool lagged pairs across sessions by concatenating series with NaN breaks
    # avoided: we instead run granger per session and combine via Fisher's method
    # would over-engineer; the pre-reg says "pooling the per-session trajectories".
    # We concatenate trajectories end-to-end (boundary effect is negligible at
    # maxlag=1 over 30 sessions × 12 cycles).
    pooled = np.vstack(series_list)  # (n_sessions*t, 11)

    pvals = {}
    predicted = set()
    for i in range(N_DIMS):
        for j in range(N_DIMS):
            if i == j:
                continue
            p = granger_pvalue(pooled[:, j], pooled[:, i], maxlag)  # i -> j
            pvals[(i, j)] = p
            if p < alpha:
                predicted.add((i, j))
    return {"pvalues": pvals, "predicted": predicted}


def score(predicted: set, truth: set) -> dict:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def run_condition(label: str, series_by_graph: dict, truth_by_graph: dict) -> dict:
    """Run discovery + scoring for one condition (raw or reconstruction)."""
    print(f"\n--- {label} ---")
    per_graph = {}
    all_pred, all_truth = set(), set()
    for gid in sorted(series_by_graph.keys()):
        disc = discover_edges(series_by_graph[gid], MAXLAG, ALPHA)
        truth = set(map(tuple, truth_by_graph[gid]))
        sc = score(disc["predicted"], truth)
        per_graph[gid] = {
            "predicted": sorted(map(list, disc["predicted"])),
            "truth": sorted(map(list, truth)),
            **sc,
        }
        # offset edges per graph by a graph tag so micro-union doesn't merge across graphs
        all_pred |= {(gid, *e) for e in disc["predicted"]}
        all_truth |= {(gid, *e) for e in truth}
        print(f"  {gid}: pred={sorted(map(list, disc['predicted']))}  "
              f"truth={sorted(map(list, truth))}  "
              f"P={sc['precision']:.2f} R={sc['recall']:.2f} F1={sc['f1']:.2f}")

    micro = score(all_pred, all_truth)
    print(f"  MICRO: tp={micro['tp']} fp={micro['fp']} fn={micro['fn']}  "
          f"P={micro['precision']:.3f} R={micro['recall']:.3f} F1={micro['f1']:.3f}")
    return {"per_graph": per_graph, "micro": micro}


def main() -> int:
    print("S3 — Causal Conservation Test (Property 1 of C)")
    print(f"  Tucker rank={TUCKER_RANK}  maxlag={MAXLAG}  alpha={ALPHA}")
    print(f"  Pre-registration: PRE_REGISTRATION.md (commit 1a3076d)")

    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text())
    graphs = sorted(gt.keys())
    truth_by_graph = {gid: gt[gid]["edge_index"] for gid in graphs}
    op = TuckerCompositionOperator()

    raw_series, recon_series = {}, {}
    for gid in graphs:
        sessions = load_graph_sessions(gid)
        # Raw: per-session dim series from original tensors.
        raw_series[gid] = [to_dim_series(T) for T in sessions]
        # Reconstruction: Tucker-compose then reconstruct, then per-session series.
        T5 = op.stack(sessions)
        core, factors, _ = op.compose(T5, TUCKER_RANK)
        import tensorly as tl
        recon5 = tl.to_numpy(tl.tucker_to_tensor(
            (tl.tensor(core), [tl.tensor(f) for f in factors])))
        recon_series[gid] = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]

    # Control first (attribution), then the pre-registered primary condition.
    raw_result = run_condition("RAW corpus (control)", raw_series, truth_by_graph)
    recon_result = run_condition("Tucker reconstruction (PRIMARY)", recon_series, truth_by_graph)

    recon_f1 = recon_result["micro"]["f1"]
    raw_f1 = raw_result["micro"]["f1"]

    if recon_f1 >= PASS_THRESHOLD:
        verdict = "PASS — Property 1 holds; C preserves causality"
    elif recon_f1 >= PARTIAL_THRESHOLD:
        verdict = "PARTIAL — C partially preserves causality (borderline)"
    else:
        verdict = "FAIL — Property 1 violated (semantic collapse)"
    attribution = ""
    if recon_f1 < PASS_THRESHOLD and raw_f1 < PASS_THRESHOLD:
        attribution = " [NOTE: raw F1 also < 0.70 -> attribute to method, S3 inconclusive]"

    print("\n" + "=" * 65)
    print("S3 VERDICT")
    print("=" * 65)
    print(f"  Reconstruction micro-F1: {recon_f1:.3f}  (primary)")
    print(f"  Raw corpus micro-F1:     {raw_f1:.3f}  (control)")
    print(f"  VERDICT: {verdict}{attribution}")
    print("=" * 65)

    output = {
        "experiment": "S3 — Causal Conservation Test",
        "pre_registration": "PRE_REGISTRATION.md (commit 1a3076d, 2026-06-08)",
        "fixed_params": {"tucker_rank": list(TUCKER_RANK), "maxlag": MAXLAG, "alpha": ALPHA,
                         "pass_threshold": PASS_THRESHOLD, "partial_threshold": PARTIAL_THRESHOLD},
        "raw_control": raw_result,
        "reconstruction_primary": recon_result,
        "verdict": {"reconstruction_micro_f1": recon_f1, "raw_micro_f1": raw_f1,
                    "verdict": verdict, "attribution_note": attribution.strip()},
    }
    out_path = OUTPUT_DIR / "s3_causal_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=lambda o: list(o) if isinstance(o, set) else o))
    print(f"\n  Results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
