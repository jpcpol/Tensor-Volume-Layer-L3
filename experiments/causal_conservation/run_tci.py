# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
TCI — Instrument Calibration Test for the unsupervised causal-conservation metric U.

Pre-registered: PRE_REGISTRATION_TCI.md (commit 1766b86). This implements that
document verbatim. Do NOT change U's definition, the gates, or the verdict logic
post-hoc; a different U requires a new pre-registration.

U is built on the PCMCI val_matrix (the SAME estimator as S3-bis run_s3_run2 —
ParCorr, tau=1, pc_alpha=0.01), so U and the supervised F1 share the causal
machinery and differ only in what they extract (continuous flow vs thresholded
edges). The TCI validates U against the S3-bis supervised F1 via two gates:
  M1  — U ranks the Tucker family {1,2,3,5,8} as F1 does (Spearman rho = 1.0)
  C2  — U collapses when causality is destroyed at fixed marginals (shuffle < 0.5)
Both required. ONE U; if it fails, report an honest negative (no method-shopping).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from sys import path as _sys_path

import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

# Reuse the EXACT S3-bis machinery (same estimator params, reconstruction, series).
from run_s3_run2 import PC_ALPHA, TAU, reconstruct, to_dim_series
from tucker_operator import TuckerCompositionOperator  # path added by run_s3_run2

_sys_path  # keep import-time path side effect explicit

HERE = Path(__file__).parent
CORPUS_DIR = HERE.parent / "synthetic_corpus" / "corpus_s1bis"
OUTPUT_DIR = HERE / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_DIMS = 11

# ── Pre-registered fixed parameters ──────────────────────────────────────────
M1_FAMILY_RANKS = [1, 2, 3, 5, 8]
# Supervised F1 per rank, taken verbatim from S3-bis results (cdd64e8).
F1_BY_RANK = {1: 0.0898876404494382, 2: 0.1038961038961039, 3: 0.1348314606741573,
              5: 0.17391304347826084, 8: 0.24489795918367346}
TUCKER_AMBIENT = (3, 3, 3, 6)          # (dim, stage, agent, cycle) ranks, as S3-bis
SHUFFLE_SEED = 20260609
M1_GATE = 1.0                          # Spearman rho must equal 1.0
C2_FLOOR = 0.5                         # U(raw, shuffle) must be < 0.5


def val_matrix_one_session(series: np.ndarray) -> np.ndarray:
    """
    PCMCI val_matrix lag-1 slice for one session's (t x 11) series.

    Returns Phi_s (11 x 11): Phi_s[i,j] = partial-correlation strength of
    var_i(t-1) -> var_j(t), conditioned on other parents. Same estimator/params
    as S3-bis (ParCorr, tau=1, pc_alpha=0.01).
    """
    df = pp.DataFrame(series, var_names=[str(d) for d in range(N_DIMS)])
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    try:
        res = pcmci.run_pcmci(tau_min=TAU, tau_max=TAU, pc_alpha=PC_ALPHA)
    except Exception:
        return np.zeros((N_DIMS, N_DIMS))
    return np.asarray(res["val_matrix"][:, :, TAU])  # (11, 11)


def flow_matrix(series_list: list[np.ndarray]) -> np.ndarray:
    """Phi[i,j] = median over sessions of val_matrix[i,j,1]. Diagonal kept (excluded in U)."""
    mats = np.stack([val_matrix_one_session(s) for s in series_list], axis=0)
    return np.median(mats, axis=0)  # (11, 11)


def U(phi_ref: np.ndarray, phi_test: np.ndarray) -> float:
    """Pearson correlation of off-diagonal entries (i != j) of two flow matrices."""
    off = ~np.eye(N_DIMS, dtype=bool)
    a, b = phi_ref[off], phi_test[off]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(pearsonr(a, b)[0])


def u_l1(phi_ref: np.ndarray, phi_test: np.ndarray) -> float:
    """Secondary, non-gating: mean |.| off-diagonal distance (context only)."""
    off = ~np.eye(N_DIMS, dtype=bool)
    return float(np.mean(np.abs(phi_ref[off] - phi_test[off])))


def load_graph_sessions(gid: str) -> list[np.ndarray]:
    return [np.load(p) for p in sorted((CORPUS_DIR / gid).glob("session_*.npy"))]


def shuffle_series(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    C2 perturbation: independently permute the cycle order of each of the 11
    dimensions. Destroys lag-1 cross-dim causal structure; preserves each
    dimension's marginal exactly (same values, reordered); no compression.
    """
    out = series.copy()
    t = series.shape[0]
    for d in range(N_DIMS):
        out[:, d] = series[rng.permutation(t), d]
    return out


def main() -> int:
    print("TCI — Instrument Calibration Test for unsupervised causal metric U")
    print(f"  U = Pearson corr of off-diag PCMCI val_matrix flow (tau={TAU}, alpha={PC_ALPHA})")
    print(f"  Pre-registration: PRE_REGISTRATION_TCI.md (commit 1766b86)")
    print(f"  Corpus: {CORPUS_DIR.name}")

    if not (CORPUS_DIR / "corpus_manifest.json").exists():
        print("ERROR: S1-bis corpus not found.")
        return 1

    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text())
    graphs = sorted(gt.keys())
    op = TuckerCompositionOperator()

    # Per-graph raw series + reference flow matrix Phi_raw.
    raw_series = {gid: [to_dim_series(T) for T in load_graph_sessions(gid)] for gid in graphs}
    phi_raw = {gid: flow_matrix(raw_series[gid]) for gid in graphs}

    # ── M1: does U rank the Tucker family as the supervised F1 does? ──────────
    print("\n--- M1: causal monotonicity (U vs supervised F1 over Tucker ranks) ---")
    print(f"  {'rank':>4} {'F1(sup)':>8} {'U(mean over graphs)':>20}")
    u_by_rank = {}
    for r in M1_FAMILY_RANKS:
        us = []
        for gid in graphs:
            sessions = load_graph_sessions(gid)
            recon5 = reconstruct(op, sessions, (r, *TUCKER_AMBIENT))
            recon_series = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]
            phi_recon = flow_matrix(recon_series)
            us.append(U(phi_raw[gid], phi_recon))
        u_mean = float(np.mean(us))
        u_by_rank[r] = u_mean
        print(f"  {r:>4} {F1_BY_RANK[r]:>8.4f} {u_mean:>20.4f}")

    f1_ordered = [F1_BY_RANK[r] for r in M1_FAMILY_RANKS]
    u_ordered = [u_by_rank[r] for r in M1_FAMILY_RANKS]
    rho = float(spearmanr(f1_ordered, u_ordered).correlation)
    # Gate is "perfect ordering" (rho = 1.0). Compare with float tolerance:
    # spearmanr returns 0.9999999999999999 for a perfectly monotone sequence.
    m1_pass = (rho >= M1_GATE - 1e-9)
    print(f"  Spearman rho(U, F1) = {rho:.4f}  ->  M1 {'PASS' if m1_pass else 'FAIL'} (gate rho=1.0)")

    # ── C2: does U collapse when causality is destroyed at fixed marginals? ───
    print("\n--- C2: causal perturbation control (shuffle at fixed marginals, no compression) ---")
    rng = np.random.default_rng(SHUFFLE_SEED)
    u_shuffle, u_self = [], []
    for gid in graphs:
        shuffled = [shuffle_series(s, rng) for s in raw_series[gid]]
        phi_shuffle = flow_matrix(shuffled)
        u_shuffle.append(U(phi_raw[gid], phi_shuffle))
        u_self.append(U(phi_raw[gid], phi_raw[gid]))  # trivial ceiling = 1.0
    u_shuffle_mean = float(np.mean(u_shuffle))
    u_self_mean = float(np.mean(u_self))
    c2_pass = (u_shuffle_mean < C2_FLOOR)
    print(f"  U(raw, raw)     = {u_self_mean:.4f}  (trivial sanity ceiling, expect 1.0)")
    print(f"  U(raw, shuffle) = {u_shuffle_mean:.4f}  ->  C2 {'PASS' if c2_pass else 'FAIL'} (gate < 0.5)")

    # ── Verdict (pre-registered: BOTH required) ──────────────────────────────
    if m1_pass and c2_pass:
        verdict = ("U is a VALIDATED instrument. Proceed to design C_causal / Pi_gov "
                   "with U in the objective (separate pre-registration).")
    elif m1_pass and not c2_pass:
        verdict = ("REJECT — U passes M1 but fails C2: it orders by compression, not "
                   "causality (a compression detector in disguise).")
    else:
        verdict = ("REJECT — U fails M1: it does not track known causal fidelity. "
                   "Linear-flow invariant insufficient; next prereg needs a richer one.")

    print("\n" + "=" * 70)
    print("TCI VERDICT")
    print("=" * 70)
    print(f"  M1 (monotonicity): {'PASS' if m1_pass else 'FAIL'}  (Spearman rho={rho:.4f})")
    print(f"  C2 (perturbation): {'PASS' if c2_pass else 'FAIL'}  (U_shuffle={u_shuffle_mean:.4f})")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    out = {
        "experiment": "TCI — Instrument Calibration Test",
        "pre_registration": "PRE_REGISTRATION_TCI.md (commit 1766b86)",
        "U_definition": "Pearson corr of off-diagonal median PCMCI val_matrix (lag-1)",
        "fixed_params": {"tau": TAU, "pc_alpha": PC_ALPHA, "tucker_ambient": list(TUCKER_AMBIENT),
                         "shuffle_seed": SHUFFLE_SEED, "m1_gate": M1_GATE, "c2_floor": C2_FLOOR},
        "M1": {"ranks": M1_FAMILY_RANKS, "f1_by_rank": F1_BY_RANK,
               "u_by_rank": u_by_rank, "spearman_rho": rho, "pass": m1_pass},
        "C2": {"u_raw_raw": u_self_mean, "u_raw_shuffle": u_shuffle_mean, "pass": c2_pass},
        "verdict": {"m1_pass": m1_pass, "c2_pass": c2_pass, "validated": m1_pass and c2_pass,
                    "text": verdict},
    }
    out_path = OUTPUT_DIR / "tci_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
