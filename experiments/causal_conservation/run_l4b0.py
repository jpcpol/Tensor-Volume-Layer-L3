# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
L4-B0 — Residual Characterization.

Pre-registered: CAL/L4/.../PRE_REGISTRATION_L4B0_RESIDUAL.md (commit ad668ca).
Implements it verbatim. Descriptive only — builds NO inverse projection V'.

Answers: of the residual ΔU = 1 − U(Φ_masked) ≈ 0.138 that the Form-1 prune did NOT
recover, what fraction is linear-edge-representable (B1 magnitude + B2 sign — what a
linear inverse projection could fold into V') vs not carriable by a single linear V'
(B3-lin complement + B4 lag>1)?

Everything reuses the FROZEN ParCorr machinery (run_tci.U / flow_matrix,
run_form1.masked_flow / measure, run_s3_run2.discover_edges_majority). The only run
beyond the lag-1 machinery is one declared auxiliary PCMCI at tau_max=3 for B4
(same ParCorr, same pc_alpha) — §1 of the prereg.

Decision (§5): share_lin = (ΔU_B1 + ΔU_B2) / ΔU → GO ≥0.70 / CONDITIONAL / NO-GO <0.40.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# Portable unicode output (Windows consoles default to cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from run_s3_run2 import PC_ALPHA, TAU, discover_edges_majority, reconstruct, to_dim_series
from run_tci import (CORPUS_DIR, N_DIMS, U, flow_matrix, load_graph_sessions,
                     val_matrix_one_session)
from tucker_operator import TuckerCompositionOperator

# Write results into the L4 repo (this is an L4 experiment run from L3's machinery dir).
L4_RESULTS = (Path(__file__).resolve().parents[3] / "L4" / "experiments" /
              "efficiency_hypothesis" / "results")
L4_RESULTS.mkdir(parents=True, exist_ok=True)

TUCKER_R8 = (8, 3, 3, 3, 6)   # the L4-A / Form-1 operator (κ=1296), fixed a priori
DELTA_W = 0.10                # §1: magnitude threshold on the normalized Φ scale
PREREG = "PRE_REGISTRATION_L4B0_RESIDUAL.md (commit ad668ca)"


def bucket_partition(phi_raw: np.ndarray, phi_tucker: np.ndarray,
                     support: set) -> tuple[set, set]:
    """
    Partition the on-support off-diagonal entries into B1 (magnitude) and B2 (sign).

    B2 = sign-flip (incl. one side ≈ 0 with opposite sign of the other) — reported
    separately because U is Pearson over SIGNED Φ, so a flip costs U non-linearly.
    B1 = same sign but |Φ_tucker − Φ_raw| > DELTA_W. Disjoint by construction.
    """
    b1, b2 = set(), set()
    for (i, j) in support:
        if i == j:
            continue
        r, t = phi_raw[i, j], phi_tucker[i, j]
        sr, st = np.sign(r), np.sign(t)
        if sr != 0 and st != 0 and sr != st:
            b2.add((i, j))
        elif sr != st:                       # one side ~0 with the other signed → flip-like
            b2.add((i, j))
        elif abs(t - r) > DELTA_W:
            b1.add((i, j))
    return b1, b2


def corrected(phi_masked: np.ndarray, phi_raw: np.ndarray, bucket: set) -> np.ndarray:
    """correct(Φ_masked, B_k): copy raw Φ into the masked Φ for the bucket's entries."""
    out = phi_masked.copy()
    for (i, j) in bucket:
        out[i, j] = phi_raw[i, j]
    return out


def val_matrix_tau3_one_session(series: np.ndarray, tau_max: int = 3) -> np.ndarray:
    """
    DECLARED auxiliary run (prereg §1, B4 only): PCMCI at tau_max=3, same ParCorr /
    pc_alpha. Returns the per-(i,j) MAX |val| over lags 2..tau_max — the lag>1 flow
    the lag-1 machinery drops. Lag-1 is excluded here (it lives in the main Φ).
    """
    df = pp.DataFrame(series, var_names=[str(d) for d in range(N_DIMS)])
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    try:
        res = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=PC_ALPHA)
    except Exception:
        return np.zeros((N_DIMS, N_DIMS))
    vm = np.asarray(res["val_matrix"])  # (N, N, tau_max+1)
    higher = vm[:, :, 2:tau_max + 1]
    idx = np.argmax(np.abs(higher), axis=2)
    return np.take_along_axis(higher, idx[:, :, None], axis=2)[:, :, 0]


def higher_order_flow(series_list: list[np.ndarray]) -> np.ndarray:
    """Median over sessions of the lag>1 (tau 2..3) flow — the B4 object."""
    mats = [val_matrix_tau3_one_session(s) for s in series_list]
    return np.median(np.stack(mats, axis=0), axis=0)


def run_once(graphs, sessions_by_graph, raw_series, tucker_series, phi_raw,
             raw_edges) -> dict:
    """One full attribution pass over G1/G2/G3. Deterministic (D1)."""
    per_graph = {}
    for g in graphs:
        pr = phi_raw[g]
        pt = flow_matrix(tucker_series[g])
        support = raw_edges[g]
        # G_pruned's Φ via the EXACT Form-1 primitive.
        from run_form1 import masked_flow
        pm = masked_flow(tucker_series[g], support)
        u0 = U(pr, pm)
        dU = 1.0 - u0

        b1, b2 = bucket_partition(pr, pt, support)
        dU_b1 = U(pr, corrected(pm, pr, b1)) - u0
        dU_b2 = U(pr, corrected(pm, pr, b2)) - u0

        # B4 (lag>1) — DIAGNOSTIC, not a U-recovery. U is defined on lag-1 Φ, so an
        # edge living only at lag>1 is invisible to BOTH phi_raw and ΔU; "correcting"
        # it against the lag-1 reference is incoherent (it injects off-axis flow and
        # destroys the Pearson, the -120% artifact of the first run). Instead B4
        # measures how much of the raw causal SUPPORT carries lag>1 structure the
        # lag-1 regime cannot see: |val_hi| > |val_lag1| on support entries. This is
        # reported as a SHARE OF SUPPORT (out-of-U-scope structure), separate from the
        # ΔU budget — it flags structure a lag-1 linear V' could never carry, without
        # double-counting it into ΔU.
        pr_hi = higher_order_flow(raw_series[g])
        b4_support = {(i, j) for (i, j) in support
                      if i != j and abs(pr_hi[i, j]) > abs(pr[i, j])}
        b4_share = len(b4_support) / max(len(support), 1)

        # B3-lin = the in-ΔU complement of B1+B2 (prereg §0.3, §2): linear lag-1
        # residual B1+B2 cannot recover. B4 is OUT of this budget (different axis).
        dU_b3lin = dU - (dU_b1 + dU_b2)
        per_graph[g] = {
            "U0": float(u0), "dU": float(dU),
            "dU_B1": float(dU_b1), "dU_B2": float(dU_b2),
            "dU_B3lin": float(dU_b3lin),
            "B4_support_share": float(b4_share),
            "B1_n": len(b1), "B2_n": len(b2),
            "B4_n": len(b4_support), "support_n": len(support),
        }
    return per_graph


def main() -> int:
    print("L4-B0 — Residual Characterization")
    print(f"  Pre-registration: {PREREG}")
    print(f"  Operator: Tucker r8 {TUCKER_R8} (κ=1296); δ_w={DELTA_W}; corpus S1-bis t=48")

    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text())
    graphs = sorted(gt.keys())
    op = TuckerCompositionOperator()

    sessions_by_graph = {g: load_graph_sessions(g) for g in graphs}
    raw_series = {g: [to_dim_series(T) for T in sessions_by_graph[g]] for g in graphs}
    phi_raw = {g: flow_matrix(raw_series[g]) for g in graphs}
    raw_edges = {g: discover_edges_majority(raw_series[g]) for g in graphs}

    tucker_series = {}
    for g in graphs:
        recon5 = reconstruct(op, sessions_by_graph[g], TUCKER_R8)
        tucker_series[g] = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]

    # D1: run twice, require byte-identical attribution.
    run_a = run_once(graphs, sessions_by_graph, raw_series, tucker_series, phi_raw, raw_edges)
    run_b = run_once(graphs, sessions_by_graph, raw_series, tucker_series, phi_raw, raw_edges)
    d1_pass = run_a == run_b

    # Aggregate (mean over graphs).
    def mean(key):
        return float(np.mean([run_a[g][key] for g in graphs]))

    agg = {k: mean(k) for k in ("dU", "dU_B1", "dU_B2", "dU_B3lin", "U0",
                                "B4_support_share")}
    dU = agg["dU"]
    share_lin = (agg["dU_B1"] + agg["dU_B2"]) / dU if dU > 1e-9 else 0.0
    sum_measured = agg["dU_B1"] + agg["dU_B2"]

    print("\n--- Per-graph attribution (fraction of ΔU; B4 = share of support, off-axis) ---")
    print(f"  {'G':>3} {'U0':>6} {'ΔU':>7} {'B1':>7} {'B2':>7} {'B3lin':>7} {'B4*':>6}")
    for g in graphs:
        r = run_a[g]
        du = r["dU"] if r["dU"] > 1e-9 else 1.0
        print(f"  {g:>3} {r['U0']:>6.3f} {r['dU']:>7.3f} "
              f"{r['dU_B1']/du:>7.2f} {r['dU_B2']/du:>7.2f} "
              f"{r['dU_B3lin']/du:>7.2f} {r['B4_support_share']:>6.2f}")

    print("\n--- Aggregate (mean over G1/G2/G3) ---")
    print(f"  ΔU = {dU:+.3f}  (U0 = {agg['U0']:.3f})")
    print(f"  ΔU_B1 (magnitude)  = {agg['dU_B1']:+.3f}  ({agg['dU_B1']/dU:.1%} of ΔU)")
    print(f"  ΔU_B2 (sign-flip)  = {agg['dU_B2']:+.3f}  ({agg['dU_B2']/dU:.1%} of ΔU)")
    print(f"  ΔU_B3lin (compl.)  = {agg['dU_B3lin']:+.3f}  ({agg['dU_B3lin']/dU:.1%} of ΔU)")
    print(f"  B4 lag>1 (OFF-axis, share of support): {agg['B4_support_share']:.1%} "
          f"— structure outside U's lag-1 scope, NOT in the ΔU budget")

    # D2: exhaustiveness honesty.
    print("\n--- D2 exhaustiveness (report) ---")
    print(f"  In-ΔU budget: B1+B2 = {sum_measured:+.3f}; B3lin (complement) = "
          f"{agg['dU_B3lin']:+.3f}; sum = {sum_measured + agg['dU_B3lin']:+.3f} ≈ ΔU = {dU:+.3f}")
    print(f"  B4 reported as off-axis support share (lag>1), not double-counted into ΔU.")

    # §5 decision logic.
    if share_lin >= 0.70:
        decision = ("GO for L4-B — residual is dominantly linear-edge-representable; "
                    "open PRE_REGISTRATION_L4B_INVERSE.md targeting B1+B2.")
    elif share_lin >= 0.40:
        decision = ("CONDITIONAL — a linear V' recovers part of ΔU but cannot close it; "
                    "open L4-B only with an explicit partial-recovery target.")
    else:
        decision = ("NO-GO for a single linear V' — residual lives in non-linearity/"
                    "lag>1; the dual (V_Tucker, G_pruned) is the terminal representation. "
                    "Publish as a negative result (Causality ≻ Reconstruction).")

    print("\n" + "=" * 72)
    print("L4-B0 DECISION")
    print("=" * 72)
    print(f"  share_lin = (ΔU_B1 + ΔU_B2) / ΔU = {share_lin:.3f}")
    print(f"  D1 reproducibility (two runs byte-identical): {'PASS' if d1_pass else 'FAIL'}")
    print(f"  DECISION: {decision}")
    print("=" * 72)

    out = {
        "experiment": "L4-B0 — Residual Characterization",
        "pre_registration": PREREG,
        "operator_tucker": list(TUCKER_R8),
        "delta_w": DELTA_W,
        "per_graph": run_a,
        "aggregate": agg,
        "share_lin": share_lin,
        "D1_reproducible": d1_pass,
        "D2_sum_measured": sum_measured,
        "D3_machinery": ("ParCorr lag-1 (frozen) + one declared tau_max=3 aux for B4; "
                         "no non-linear estimator (GPDC/CMIknn infeasible in-env, "
                         "and redundant with the B1+B2 complement since V' is linear)."),
        "decision": decision,
    }
    out_path = L4_RESULTS / "l4b0_residual_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
