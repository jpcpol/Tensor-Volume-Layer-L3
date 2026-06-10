# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
Form 1 — Structural-Mask Falsifier.

Pre-registered: PRE_REGISTRATION_FORM1.md (commit b2c843a). Implements it verbatim.
Q_L3.2A showed Tucker over-generates causal edges (fabrication), while keeping C
(coverage) and S (consistency). Form 1 tests the sharp prediction: pruning V_hat's
flow to a causal support should lift U toward raw while keeping C>=0.95 and S=1.0.

Scientific question: is Tucker's causal loss explained PRINCIPALLY by spurious
connections? CONFIRMED if dU(masked vs Tucker r8) >= +0.15 with the safeguard held.

Two pre-registered supports: support_raw (primary, self-referential deployment
surrogate, DECIDES) and support_GT (oracle ceiling). raw<->GT gap reported.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from run_s3_run2 import (PC_ALPHA, TAU, discover_edges_majority, reconstruct,
                         to_dim_series)
from run_tci import (CORPUS_DIR, N_DIMS, U, flow_matrix, load_graph_sessions,
                     val_matrix_one_session)
from run_ql32a import consistency, coverage, reachability
from tucker_operator import TuckerCompositionOperator

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TUCKER_R8 = (8, 3, 3, 3, 6)     # best-U Tucker (C* from operator search)
CONFIRM_DU = 0.15               # a priori: confirm if dU >= +0.15
REFUTE_DU = 0.05                # a priori: refute if dU < +0.05
SAFE_C = 0.95                   # safeguard: coverage >= 0.95
# safeguard S must equal 1.0


def masked_flow(series_list, support: set) -> np.ndarray:
    """
    Flow matrix Phi of a corpus, with flow zeroed OUTSIDE `support` (the prune).

    Per session, take the val_matrix, zero every off-support (i!=j) entry, then
    median across sessions. This is the structural mask applied at the flow level
    so U/R/C/S all see the pruned structure with the SAME machinery.
    """
    mats = []
    for s in series_list:
        vm = val_matrix_one_session(s)
        masked = np.zeros_like(vm)
        for (i, j) in support:
            masked[i, j] = vm[i, j]
        # keep diagonal (excluded by U anyway), copy as-is
        np.fill_diagonal(masked, np.diag(vm))
        mats.append(masked)
    return np.median(np.stack(mats, axis=0), axis=0)


def masked_edges(series_list, support: set) -> set:
    """Edge set of the pruned corpus = recovered edges intersected with support."""
    rec = discover_edges_majority(series_list)
    return rec & support


def main() -> int:
    print("Form 1 — Structural-Mask Falsifier")
    print(f"  Pre-registration: PRE_REGISTRATION_FORM1.md (commit b2c843a)")
    print(f"  Baseline: Tucker r8 {TUCKER_R8}; masks: support_raw (primary), support_GT (ceiling)")

    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text())
    graphs = sorted(gt.keys())
    op = TuckerCompositionOperator()
    gt_edges = {g: {tuple(e) for e in gt[g]["edge_index"]} for g in graphs}
    gt_signs = {g: {tuple(idx): int(ed.get("sign", 1))
                    for ed, idx in zip(gt[g]["edges"], gt[g]["edge_index"])} for g in graphs}

    sessions_by_graph = {g: load_graph_sessions(g) for g in graphs}
    raw_series = {g: [to_dim_series(T) for T in sessions_by_graph[g]] for g in graphs}
    phi_raw = {g: flow_matrix(raw_series[g]) for g in graphs}
    raw_edges = {g: discover_edges_majority(raw_series[g]) for g in graphs}
    raw_signs = {g: {(i, j): np.sign(phi_raw[g][i, j]) for (i, j) in raw_edges[g]} for g in graphs}

    # Tucker r8 reconstruction series.
    tucker_series = {}
    for g in graphs:
        recon5 = reconstruct(op, sessions_by_graph[g], TUCKER_R8)
        tucker_series[g] = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]

    def measure(series_by_graph, support_by_graph=None):
        """Mean over graphs of (U vs raw, R, C_raw, S_raw, |E|). support=None -> unmasked."""
        Us, Rs, Cs, Ss, Es = [], [], [], [], []
        for g in graphs:
            if support_by_graph is None:
                phi = flow_matrix(series_by_graph[g])
                edges = discover_edges_majority(series_by_graph[g])
            else:
                phi = masked_flow(series_by_graph[g], support_by_graph[g])
                edges = masked_edges(series_by_graph[g], support_by_graph[g])
            Us.append(U(phi_raw[g], phi))
            Rs.append(reachability(edges))
            Cs.append(coverage(edges, raw_edges[g]))
            Ss.append(consistency(edges, phi, raw_signs[g]))
            Es.append(len(edges))
        return {"U": float(np.mean(Us)), "R": float(np.mean(Rs)),
                "C_raw": float(np.mean(Cs)), "S_raw": float(np.mean(Ss)),
                "E": float(np.mean(Es))}

    # Baseline (unmasked Tucker r8) and the two masked variants.
    base = measure(tucker_series)
    masked_raw = measure(tucker_series, raw_edges)             # primary, self-ref
    masked_gt = measure(tucker_series, gt_edges)               # ceiling, oracle
    raw_ref = measure(raw_series)                              # raw reference (U=1 sanity)

    print("\n--- Measurements (mean over G1/G2/G3) ---")
    print(f"  {'cond':>14} {'U':>7} {'R':>7} {'C_raw':>7} {'S_raw':>7} {'|E|':>6}")
    for name, m in [("raw (ref)", raw_ref), ("Tucker r8", base),
                    ("masked(raw)", masked_raw), ("masked(GT)", masked_gt)]:
        print(f"  {name:>14} {m['U']:>7.3f} {m['R']:>7.3f} {m['C_raw']:>7.3f} "
              f"{m['S_raw']:>7.3f} {m['E']:>6.1f}")

    dU_raw = masked_raw["U"] - base["U"]
    dU_gt = masked_gt["U"] - base["U"]
    dE_raw = masked_raw["E"] - base["E"]
    dR_raw = masked_raw["R"] - base["R"]
    raw_gt_gap = masked_gt["U"] - masked_raw["U"]

    # Safeguard on the PRIMARY (raw) mask.
    safe = (masked_raw["C_raw"] >= SAFE_C) and (abs(masked_raw["S_raw"] - 1.0) < 1e-9)

    print("\n--- Primary (support_raw) ---")
    print(f"  dU = {dU_raw:+.3f}  (confirm>=+{CONFIRM_DU}, refute<+{REFUTE_DU})")
    print(f"  d|E| = {dE_raw:+.1f}   dR = {dR_raw:+.3f}")
    print(f"  safeguard C>=0.95 & S=1.0: C={masked_raw['C_raw']:.3f} S={masked_raw['S_raw']:.3f} "
          f"-> {'HELD' if safe else 'FAILED'}")
    print(f"  raw<->GT gap (oracle - deploy) = {raw_gt_gap:+.3f}  (GT dU={dU_gt:+.3f})")

    # Verdict (pre-registered).
    if not safe and dU_raw > 0:
        verdict = ("INVALID — U rose but the safeguard failed (prune destroyed true "
                   "structure); not evidence either way.")
    elif dU_raw >= CONFIRM_DU and safe:
        verdict = ("CONFIRMED — Tucker's causal loss is principally spurious inflation. "
                   "L3 causal conservation = preserving structural sparsity. Justifies a "
                   "richer causal-pruning C_causal (separate prereg).")
    elif dU_raw < REFUTE_DU:
        verdict = ("REFUTED — pruning barely moves U; spurious inflation is a symptom, "
                   "not the cause. C_causal must be rethought.")
    else:
        verdict = ("PARTIAL — structure matters but is not the whole story; design a "
                   "combined operator.")

    print("\n" + "=" * 70)
    print("FORM 1 VERDICT")
    print("=" * 70)
    print(f"  primary dU(masked_raw vs Tucker r8) = {dU_raw:+.3f}   safeguard {'HELD' if safe else 'FAILED'}")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    out = {
        "experiment": "Form 1 — Structural-Mask Falsifier",
        "pre_registration": "PRE_REGISTRATION_FORM1.md (commit b2c843a)",
        "baseline_tucker": list(TUCKER_R8),
        "measurements": {"raw_ref": raw_ref, "tucker_r8": base,
                         "masked_raw": masked_raw, "masked_gt": masked_gt},
        "primary_support_raw": {"dU": dU_raw, "dE": dE_raw, "dR": dR_raw,
                                "safeguard_held": safe,
                                "C": masked_raw["C_raw"], "S": masked_raw["S_raw"]},
        "ceiling_support_gt": {"dU": dU_gt, "raw_gt_gap": raw_gt_gap},
        "thresholds": {"confirm_dU": CONFIRM_DU, "refute_dU": REFUTE_DU, "safe_C": SAFE_C},
        "verdict": verdict,
    }
    (OUTPUT_DIR / "form1_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved: {OUTPUT_DIR / 'form1_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
