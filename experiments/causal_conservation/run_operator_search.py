# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
Causal-Aware Operator Search.

Pre-registered: PRE_REGISTRATION_OPERATOR_SEARCH.md (commit 5bb1b2d). Implements
that document verbatim. Searches a Tucker rank config that maximizes the VALIDATED
TCI metric U directly (exhaustive grid; no proxy -- the proxy audit, commit
abd7871, refuted differentiable surrogates because U's ordering depends on PCMCI's
discrete PC selection step).

Primary outcome: C* = argmax_grid U.
Gate G1 (blocking, anti-overfit): supervised micro-F1(C*) >= micro-F1(C_base),
  C_base = TCI calibration baseline (best-r0 at ambient (3,3,3,6)). F1 is an
  EXTERNAL check only; we never select on F1.
Gate G2 (reported): sign of Spearman(U, F1) over the scored configs.
"""
from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# Reuse the EXACT S3-bis + TCI machinery.
from run_s3_run2 import (PC_ALPHA, TAU, VOTE_THRESHOLD, discover_edges_majority,
                         reconstruct, score, to_dim_series)
from run_tci import (CORPUS_DIR, F1_BY_RANK, N_DIMS, U, flow_matrix,
                     load_graph_sessions)
from tucker_operator import TuckerCompositionOperator

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Pre-registered fixed search space ────────────────────────────────────────
R0_GRID = [1, 2, 3, 5, 8]
RDIM_GRID = [2, 3, 4]
RCYCLE_GRID = [4, 6, 9]
RSTAGE = 3
RAGENT = 3
BASE_AMBIENT = (3, 3, 3, 6)   # TCI calibration baseline ambient (dim,stage,agent,cycle)


def recon_series_for(op, sessions, rank5):
    recon5 = reconstruct(op, sessions, rank5)
    return [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]


def supervised_f1(op, sessions_by_graph, truth_by_graph, rank5) -> float:
    """S3-bis micro-F1 across graphs for a given Tucker rank (external check only)."""
    all_pred, all_truth = set(), set()
    for gid, sessions in sessions_by_graph.items():
        rs = recon_series_for(op, sessions, rank5)
        pred = discover_edges_majority(rs)
        truth = set(map(tuple, truth_by_graph[gid]))
        all_pred |= {(gid, *e) for e in pred}
        all_truth |= {(gid, *e) for e in truth}
    return score(all_pred, all_truth)["f1"]


def main() -> int:
    print("Causal-Aware Operator Search")
    print(f"  Objective: max U(C(T))  (TCI metric verbatim, tau={TAU}, alpha={PC_ALPHA})")
    print(f"  Pre-registration: PRE_REGISTRATION_OPERATOR_SEARCH.md (commit 5bb1b2d)")
    print(f"  Corpus: {CORPUS_DIR.name}")

    if not (CORPUS_DIR / "corpus_manifest.json").exists():
        print("ERROR: S1-bis corpus not found.")
        return 1

    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text())
    graphs = sorted(gt.keys())
    truth_by_graph = {gid: gt[gid]["edge_index"] for gid in graphs}
    op = TuckerCompositionOperator()

    sessions_by_graph = {gid: load_graph_sessions(gid) for gid in graphs}
    raw_series = {gid: [to_dim_series(T) for T in sessions_by_graph[gid]] for gid in graphs}
    phi_raw = {gid: flow_matrix(raw_series[gid]) for gid in graphs}

    # ── Primary: exhaustive U over the 45-config grid ────────────────────────
    grid = list(product(R0_GRID, RDIM_GRID, RCYCLE_GRID))
    print(f"\n--- Exhaustive U over {len(grid)} configs (r0 x r_dim x r_cycle) ---")
    print(f"  {'r0':>3} {'rdim':>4} {'rcyc':>4} {'U':>8}")
    results = []
    for (r0, rdim, rcyc) in grid:
        rank5 = (r0, rdim, RSTAGE, RAGENT, rcyc)
        us = []
        for gid in graphs:
            rs = recon_series_for(op, sessions_by_graph[gid], rank5)
            us.append(U(phi_raw[gid], flow_matrix(rs)))
        u_mean = float(np.mean(us))
        results.append({"r0": r0, "r_dim": rdim, "r_cycle": rcyc, "rank5": list(rank5), "U": u_mean})
        print(f"  {r0:>3} {rdim:>4} {rcyc:>4} {u_mean:>8.4f}")

    results.sort(key=lambda d: d["U"], reverse=True)
    best = results[0]
    rank_star = tuple(best["rank5"])
    print(f"\n  C* = argmax U: rank={rank_star}  U*={best['U']:.4f}")

    # Baseline: TCI calibration config at best r0, ambient (3,3,3,6).
    base_rank = (best["r0"], *BASE_AMBIENT)
    base_U = next(d["U"] for d in results
                  if d["r0"] == best["r0"] and d["r_dim"] == 3 and d["r_cycle"] == 6)
    print(f"  C_base (TCI calib): rank={base_rank}  U={base_U:.4f}")

    # ── G1: supervised F1 of C* vs C_base (external anti-overfit check) ───────
    print("\n--- G1: supervised micro-F1 (external check; never a selection target) ---")
    f1_star = supervised_f1(op, sessions_by_graph, truth_by_graph, rank_star)
    f1_base = supervised_f1(op, sessions_by_graph, truth_by_graph, base_rank)
    g1_pass = f1_star >= f1_base
    print(f"  F1(C*)    = {f1_star:.4f}  rank={rank_star}")
    print(f"  F1(C_base)= {f1_base:.4f}  rank={base_rank}")
    print(f"  -> G1 {'PASS' if g1_pass else 'FAIL'} (need F1(C*) >= F1(C_base))")

    # ── G2: sign of Spearman(U, F1) over a scored subset (reported) ──────────
    # Score F1 on the unique r0 configs along the calibration line (cheap, known
    # to S3-bis) to get a U-vs-F1 trend without scoring all 45.
    print("\n--- G2: Spearman(U, F1) over calibration-line configs (reported) ---")
    g2_rows = []
    for r0 in R0_GRID:
        rank5 = (r0, *BASE_AMBIENT)
        u_here = next(d["U"] for d in results
                      if d["r0"] == r0 and d["r_dim"] == 3 and d["r_cycle"] == 6)
        f1_here = F1_BY_RANK[r0]  # from S3-bis, same configs
        g2_rows.append((r0, u_here, f1_here))
    us_g2 = [r[1] for r in g2_rows]
    f1_g2 = [r[2] for r in g2_rows]
    rho_g2 = float(spearmanr(us_g2, f1_g2).correlation)
    print(f"  Spearman(U, F1) over r0 calibration line = {rho_g2:.4f}  "
          f"({'positive' if rho_g2 > 0 else 'NON-positive — overfit flag'})")

    # ── Verdict ──────────────────────────────────────────────────────────────
    if g1_pass:
        verdict = ("U is usable as a DIRECT optimization objective. C* is the "
                   "U-optimal Tucker operator. Proceed to Pi_gov (lexicographic "
                   "topological tie-break) and Q3 composition (separate preregs).")
    else:
        verdict = ("U ranks but does NOT safely optimize: argmax-U config has worse "
                   "supervised F1 than baseline (instrument overfitting). Next prereg "
                   "must add a structure-preservation guard before using U as target.")

    print("\n" + "=" * 70)
    print("OPERATOR SEARCH VERDICT")
    print("=" * 70)
    print(f"  C* = {rank_star}  U* = {best['U']:.4f}")
    print(f"  G1 (anti-overfit): {'PASS' if g1_pass else 'FAIL'}  "
          f"(F1*={f1_star:.3f} vs base={f1_base:.3f})")
    print(f"  G2 (reported): Spearman(U,F1)={rho_g2:.3f}")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    out = {
        "experiment": "Causal-Aware Operator Search",
        "pre_registration": "PRE_REGISTRATION_OPERATOR_SEARCH.md (commit 5bb1b2d)",
        "objective": "max U(C(T)) directly; U = TCI metric verbatim",
        "grid": {"r0": R0_GRID, "r_dim": RDIM_GRID, "r_cycle": RCYCLE_GRID,
                 "r_stage": RSTAGE, "r_agent": RAGENT, "n_configs": len(grid)},
        "all_configs_by_U": results,
        "C_star": {"rank5": list(rank_star), "U": best["U"]},
        "C_base": {"rank5": list(base_rank), "U": base_U},
        "G1": {"f1_star": f1_star, "f1_base": f1_base, "pass": g1_pass},
        "G2": {"spearman_U_F1_calib_line": rho_g2, "rows": g2_rows},
        "verdict": {"g1_pass": g1_pass, "text": verdict},
    }
    out_path = OUTPUT_DIR / "operator_search_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
