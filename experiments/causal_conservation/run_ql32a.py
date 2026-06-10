# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
Q_L3.2A — Causal-Invariant Survival under Tucker.

Pre-registered: PRE_REGISTRATION_QL32A.md (commit f61472f). Implements it verbatim.
Characterizes WHICH observational causal invariants (R, C, S) survive Tucker, over
{raw, r=1,2,3,5,8}. Descriptive, no PASS/FAIL. D and P are out of scope by
pre-registration (deferred to Q_L3.2B / S-Omega corpus).

Invariants (Omega0 subset, per consultant brief §9):
  R  reachability  = transitive-closure ordered-pair fraction of the edge set
  C  coverage      = |E ∩ E_ref| / |E_ref|   (E_ref = GT or raw)
  S  consistency   = 1 - (#sign-flips / |E|)  vs reference signs (GT or raw)
d_Omega is the vectorial (R, C_raw, S_raw) difference vs raw. No scalar collapse.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from run_s3_run2 import (PC_ALPHA, TAU, VOTE_THRESHOLD, discover_edges_majority,
                         reconstruct, score, to_dim_series)
from run_tci import (CORPUS_DIR, F1_BY_RANK, N_DIMS, flow_matrix,
                     load_graph_sessions, val_matrix_one_session)
from tucker_operator import TuckerCompositionOperator

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANKS = [1, 2, 3, 5, 8]
TUCKER_AMBIENT = (3, 3, 3, 6)


# ── Invariants ───────────────────────────────────────────────────────────────
def reachability(edges: set) -> float:
    """Transitive-closure ordered-pair fraction. R = #(i->...->j) / (n(n-1))."""
    A = np.zeros((N_DIMS, N_DIMS), dtype=bool)
    for (i, j) in edges:
        A[i, j] = True
    # Floyd-Warshall style transitive closure.
    reach = A.copy()
    for k in range(N_DIMS):
        reach |= reach[:, [k]] & reach[[k], :]
    np.fill_diagonal(reach, False)
    return float(reach.sum()) / (N_DIMS * (N_DIMS - 1))


def coverage(edges: set, ref_edges: set) -> float:
    """C = |E ∩ E_ref| / |E_ref|."""
    if not ref_edges:
        return 0.0
    return len(edges & ref_edges) / len(ref_edges)


def consistency(edges: set, phi: np.ndarray, ref_signs: dict) -> float:
    """S = 1 - (#sign-flips / |E|). Sign-flip: edge in both E and ref but opposite
    val_matrix-flow sign vs the reference sign. Only edges present in ref are checked."""
    if not edges:
        return 1.0
    checked = [e for e in edges if e in ref_signs]
    if not checked:
        return 1.0
    flips = 0
    for (i, j) in checked:
        s_here = np.sign(phi[i, j])
        if s_here != 0 and s_here != ref_signs[(i, j)]:
            flips += 1
    return 1.0 - flips / len(checked)


def edges_and_phi(series_list):
    """Recovered majority-vote edge set + median val_matrix flow for a condition."""
    edges = discover_edges_majority(series_list)
    phi = flow_matrix(series_list)
    return edges, phi


def main() -> int:
    print("Q_L3.2A — Causal-Invariant Survival under Tucker (R, C, S)")
    print(f"  Pre-registration: PRE_REGISTRATION_QL32A.md (commit f61472f)")
    print(f"  Corpus: {CORPUS_DIR.name}  | D,P out of scope (Q_L3.2B)")

    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text())
    graphs = sorted(gt.keys())
    op = TuckerCompositionOperator()

    # Ground-truth edges per graph + signs (paired with edge_index by order).
    gt_edges = {g: {tuple(e) for e in gt[g]["edge_index"]} for g in graphs}
    gt_signs = {}
    for g in graphs:
        signs = {}
        for ed, idx in zip(gt[g]["edges"], gt[g]["edge_index"]):
            signs[tuple(idx)] = int(ed.get("sign", 1))
        gt_signs[g] = signs

    sessions_by_graph = {g: load_graph_sessions(g) for g in graphs}
    raw_series = {g: [to_dim_series(T) for T in sessions_by_graph[g]] for g in graphs}

    # Reference structures on RAW (deployment refs) + signs from raw flow.
    raw_edges, raw_phi = {}, {}
    for g in graphs:
        e, phi = edges_and_phi(raw_series[g])
        raw_edges[g] = e
        raw_phi[g] = phi
    raw_signs = {g: {(i, j): np.sign(raw_phi[g][i, j]) for (i, j) in raw_edges[g]} for g in graphs}

    def invariants_for(series_by_graph_cond):
        """Mean over graphs of R, C_GT, C_raw, S_GT, S_raw, |E|."""
        Rs, Cgt, Craw, Sgt, Sraw, Es = [], [], [], [], [], []
        for g in graphs:
            e, phi = edges_and_phi(series_by_graph_cond[g])
            Rs.append(reachability(e))
            Cgt.append(coverage(e, gt_edges[g]))
            Craw.append(coverage(e, raw_edges[g]))
            Sgt.append(consistency(e, phi, gt_signs[g]))
            Sraw.append(consistency(e, phi, raw_signs[g]))
            Es.append(len(e))
        return {"R": float(np.mean(Rs)), "C_GT": float(np.mean(Cgt)),
                "C_raw": float(np.mean(Craw)), "S_GT": float(np.mean(Sgt)),
                "S_raw": float(np.mean(Sraw)), "E": float(np.mean(Es))}

    rows = {}
    # raw condition (the Omega reference).
    rows["raw"] = invariants_for(raw_series)
    # Tucker conditions.
    for r in RANKS:
        cond_series = {}
        for g in graphs:
            recon5 = reconstruct(op, sessions_by_graph[g], (r, *TUCKER_AMBIENT))
            cond_series[g] = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]
        rows[f"r{r}"] = invariants_for(cond_series)

    # ── Report table ─────────────────────────────────────────────────────────
    print("\n--- Invariant survival (mean over G1/G2/G3) ---")
    print(f"  {'cond':>5} {'R':>6} {'C_GT':>6} {'C_raw':>6} {'S_GT':>6} {'S_raw':>6} "
          f"{'|E|':>5} {'U':>6} {'F1':>6}")
    u_known = {1: 0.1952, 2: 0.2399, 3: 0.3575, 5: 0.3916, 8: 0.4415}
    order = ["raw"] + [f"r{r}" for r in RANKS]
    for k in order:
        v = rows[k]
        r = int(k[1:]) if (k != "raw" and k.startswith("r")) else None
        u = 1.0 if k == "raw" else u_known.get(r, 1.0)
        f1 = 1.0 if k == "raw" else F1_BY_RANK.get(r, 1.0)
        print(f"  {k:>5} {v['R']:>6.3f} {v['C_GT']:>6.3f} {v['C_raw']:>6.3f} "
              f"{v['S_GT']:>6.3f} {v['S_raw']:>6.3f} {v['E']:>5.1f} {u:>6.3f} {f1:>6.3f}")

    # ── d_Omega (vectorial) vs raw ───────────────────────────────────────────
    print("\n--- d_Omega(raw, r) = (|dR|, |dC_raw|, |dS_raw|)  [no scalar collapse] ---")
    d_omega = {}
    for r in RANKS:
        v = rows[f"r{r}"]; ref = rows["raw"]
        d = (abs(ref["R"] - v["R"]), abs(ref["C_raw"] - v["C_raw"]), abs(ref["S_raw"] - v["S_raw"]))
        d_omega[r] = d
        print(f"  r{r}: ({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f})")

    # ── Per-invariant monotonicity vs rank (does it degrade with compression?) ─
    print("\n--- Spearman(invariant, rank) over r=1..8 (does it track compression?) ---")
    rank_order = RANKS
    mono = {}
    for inv in ["R", "C_GT", "C_raw", "S_GT", "S_raw"]:
        vals = [rows[f"r{r}"][inv] for r in RANKS]
        rho = float(spearmanr(rank_order, vals).correlation) if np.std(vals) > 1e-12 else 0.0
        mono[inv] = rho
        print(f"  {inv:>6}: rho={rho:+.3f}")
    # also U and F1 for reference
    rho_u = float(spearmanr(rank_order, [u_known[r] for r in RANKS]).correlation)
    rho_f1 = float(spearmanr(rank_order, [F1_BY_RANK[r] for r in RANKS]).correlation)
    print(f"  {'U':>6}: rho={rho_u:+.3f}   {'F1':>6}: rho={rho_f1:+.3f}  (reference)")

    out = {
        "experiment": "Q_L3.2A — Causal-Invariant Survival under Tucker (R,C,S)",
        "pre_registration": "PRE_REGISTRATION_QL32A.md (commit f61472f)",
        "scope": "R,C,S only; D,P deferred to Q_L3.2B (S-Omega corpus)",
        "rows": rows,
        "d_omega_vectorial": {f"r{r}": list(d_omega[r]) for r in RANKS},
        "monotonicity_spearman_vs_rank": mono,
        "reference_U": u_known, "reference_F1": {str(k): v for k, v in F1_BY_RANK.items()},
        "note": "Descriptive characterization; no PASS/FAIL. Omega1 (drift/conflict) NOT claimed.",
    }
    out_path = OUTPUT_DIR / "ql32a_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
