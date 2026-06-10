# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
AUDIT (not pre-registered, not a gate). Verifies the design decisions for the
upcoming Causal-Aware Tucker prereg BEFORE writing it. Three checks:

  A. Does a FIXED-conditioning flow matrix Phi_fix (no PCMCI discovery step)
     exist and is it cheap? -> required for E_flow to be a usable training proxy.
  B. Mini-M1: does the proxy built on Phi_fix order the Tucker family {1,2,3,5,8}
     the SAME way as the validated U-via-PCMCI does (Spearman rho)?
  C. Sanity: is the proxy's C2 behavior plausible (shuffle collapses it)?

If A fails -> proxy not computable as claimed (gap -> consultant).
If B fails -> proxy diverges from the validated metric (gap -> revise or consultant).
If both pass -> the prereg design is sound; proceed to write it.

This reuses the EXACT S3-bis machinery + the TCI's U for ground truth.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

from run_s3_run2 import PC_ALPHA, TAU, reconstruct, to_dim_series
from run_tci import (CORPUS_DIR, F1_BY_RANK, M1_FAMILY_RANKS, N_DIMS,
                     SHUFFLE_SEED, TUCKER_AMBIENT, U, flow_matrix,
                     load_graph_sessions, shuffle_series)
from tucker_operator import TuckerCompositionOperator


def phi_fixed_one_session(series: np.ndarray) -> np.ndarray:
    """
    FIXED-conditioning lag-1 partial-correlation flow matrix (NO PCMCI discovery).

    Phi_fix[i,j] = partial correlation between x_i(t-1) and x_j(t), conditioned on
    ALL other dimensions at t-1 (a fixed, non-discrete conditioning set). This is
    a smooth function of the series (least-squares residual correlations), so it is
    a candidate differentiable proxy for U's PCMCI Phi.

    Implementation: build lagged design X_prev = x(t-1) (n-1, 11), Y = x(t) (n-1, 11).
    For target j, regress Y[:,j] on X_prev; for source i, partial corr of x_i(t-1)
    with the residual of Y[:,j] after removing the OTHER sources. We use the
    precision-matrix form on the joint [X_prev, Y[:,j]] for speed.
    """
    t = series.shape[0]
    Xp = series[:-1]                      # (t-1, 11)  predictors at t-1
    Y = series[1:]                        # (t-1, 11)  targets at t
    phi = np.zeros((N_DIMS, N_DIMS))
    # Standardize predictors once.
    Xc = Xp - Xp.mean(0)
    Xstd = Xc.std(0)
    Xstd[Xstd < 1e-12] = 1.0
    Xz = Xc / Xstd
    for j in range(N_DIMS):
        yj = Y[:, j] - Y[:, j].mean()
        ystd = yj.std()
        if ystd < 1e-12:
            continue
        yz = yj / ystd
        # Joint design [Xz (11 sources at t-1), yz (target at t)] -> precision.
        M = np.column_stack([Xz, yz])     # (t-1, 12)
        C = np.corrcoef(M, rowvar=False)  # (12, 12)
        try:
            P = np.linalg.pinv(C)
        except np.linalg.LinAlgError:
            continue
        # Partial corr between source i (idx i) and target (idx 11):
        #   rho_partial = -P[i,11] / sqrt(P[i,i] P[11,11])
        for i in range(N_DIMS):
            denom = np.sqrt(P[i, i] * P[11, 11])
            if denom < 1e-12:
                continue
            phi[i, j] = -P[i, 11] / denom
    return phi


def phi_bivariate_one_session(series: np.ndarray) -> np.ndarray:
    """
    REPAIR candidate 1: NO conditioning at all — plain lag-1 cross-correlation
    Phi_biv[i,j] = corr(x_i(t-1), x_j(t)). Differentiable, no discovery. This is
    the opposite extreme from full-conditioning; if Tucker's failure is *adding*
    flow, even a bivariate magnitude pattern might track it.
    """
    t = series.shape[0]
    Xp = series[:-1]; Y = series[1:]
    Xp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    return (Xp.T @ Yz) / (t - 1)   # (11,11) Phi[i,j] = corr(x_i(t-1), x_j(t))


def phi_ridge_one_session(series: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """
    REPAIR candidate 2: ridge-regularized VAR(1) coefficient matrix.
    Phi_ridge = argmin ||Y - Xp B||^2 + lam||B||^2, i.e. B = (Xp'Xp + lam I)^-1 Xp'Y.
    A *regularized* full-conditioning that does not over-condition as hard as the
    precision matrix, and is fully differentiable. Phi[i,j] = B[i,j].
    """
    Xp = series[:-1]; Y = series[1:]
    Xc = Xp - Xp.mean(0); Yc = Y - Y.mean(0)
    Xs = Xc.std(0); Xs[Xs < 1e-12] = 1.0
    Xz = Xc / Xs
    G = Xz.T @ Xz + lam * np.eye(N_DIMS)
    B = np.linalg.solve(G, Xz.T @ Yc)     # (11,11)
    return B


def flow_matrix_fixed(series_list: list[np.ndarray]) -> np.ndarray:
    mats = np.stack([phi_fixed_one_session(s) for s in series_list], axis=0)
    return np.median(mats, axis=0)


def _flow_med(fn, series_list):
    return np.median(np.stack([fn(s) for s in series_list], axis=0), axis=0)


def main() -> int:
    print("AUDIT — Causal-Aware Tucker proxy (pre-prereg, not a gate)")
    print(f"  Phi_fix = fixed-conditioning lag-1 partial corr (no PCMCI discovery)")
    print(f"  Corpus: {CORPUS_DIR.name}\n")

    import json
    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text())
    graphs = sorted(gt.keys())
    op = TuckerCompositionOperator()

    raw_series = {gid: [to_dim_series(T) for T in load_graph_sessions(gid)] for gid in graphs}

    # --- A. cost: time one fixed-conditioning Phi vs one PCMCI Phi -----------
    s0 = raw_series[graphs[0]][0]
    t0 = time.perf_counter(); phi_fixed_one_session(s0); t_fix = time.perf_counter() - t0
    from run_tci import val_matrix_one_session
    t0 = time.perf_counter(); val_matrix_one_session(s0); t_pcmci = time.perf_counter() - t0
    print(f"--- A. cost per session ---")
    print(f"  Phi_fix (proxy):  {t_fix*1000:8.2f} ms")
    print(f"  Phi PCMCI (U):    {t_pcmci*1000:8.2f} ms")
    print(f"  speedup: {t_pcmci/max(t_fix,1e-9):.1f}x  -> A {'PASS' if t_fix < t_pcmci else 'CHECK'}\n")

    phi_raw_fix = {gid: flow_matrix_fixed(raw_series[gid]) for gid in graphs}
    phi_raw_pcmci = {gid: flow_matrix(raw_series[gid]) for gid in graphs}

    # --- B. mini-M1: proxy ordering vs U ordering over Tucker family ---------
    print("--- B. mini-M1: proxy E_flow ordering vs validated U ---")
    print(f"  {'rank':>4} {'F1':>7} {'U(PCMCI)':>9} {'proxy_corr':>11}")
    u_by_rank, proxy_by_rank = {}, {}
    for r in M1_FAMILY_RANKS:
        us, ps = [], []
        for gid in graphs:
            sessions = load_graph_sessions(gid)
            recon5 = reconstruct(op, sessions, (r, *TUCKER_AMBIENT))
            recon_series = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]
            us.append(U(phi_raw_pcmci[gid], flow_matrix(recon_series)))
            ps.append(U(phi_raw_fix[gid], flow_matrix_fixed(recon_series)))  # same corr stat
        u_by_rank[r] = float(np.mean(us))
        proxy_by_rank[r] = float(np.mean(ps))
        print(f"  {r:>4} {F1_BY_RANK[r]:>7.3f} {u_by_rank[r]:>9.4f} {proxy_by_rank[r]:>11.4f}")

    f1_ord = [F1_BY_RANK[r] for r in M1_FAMILY_RANKS]
    u_ord = [u_by_rank[r] for r in M1_FAMILY_RANKS]
    p_ord = [proxy_by_rank[r] for r in M1_FAMILY_RANKS]
    rho_proxy_f1 = float(spearmanr(f1_ord, p_ord).correlation)
    rho_proxy_u = float(spearmanr(u_ord, p_ord).correlation)
    print(f"  Spearman rho(proxy, F1) = {rho_proxy_f1:.4f}")
    print(f"  Spearman rho(proxy, U)  = {rho_proxy_u:.4f}")
    b_pass = rho_proxy_f1 >= 1.0 - 1e-9
    print(f"  -> B (mini-M1) {'PASS' if b_pass else 'FAIL'} (gate rho(proxy,F1)=1.0)\n")

    # --- C. sanity: shuffle collapses the proxy too --------------------------
    print("--- C. proxy C2 sanity (shuffle at fixed marginals) ---")
    rng = np.random.default_rng(SHUFFLE_SEED)
    u_sh = []
    for gid in graphs:
        shuffled = [shuffle_series(s, rng) for s in raw_series[gid]]
        u_sh.append(U(phi_raw_fix[gid], flow_matrix_fixed(shuffled)))
    u_sh_mean = float(np.mean(u_sh))
    c_pass = u_sh_mean < 0.5
    print(f"  proxy U(raw, shuffle) = {u_sh_mean:.4f}  -> C {'PASS' if c_pass else 'CHECK'} (<0.5)\n")

    # --- B2. repair candidates: bivariate + ridge VAR(1) --------------------
    print("--- B2. repair candidates (ordering vs F1) ---")
    repair_rho = {}
    for name, fn in [("bivariate", phi_bivariate_one_session),
                     ("ridge_VAR1", phi_ridge_one_session)]:
        phi_raw_r = {gid: _flow_med(fn, raw_series[gid]) for gid in graphs}
        cand_by_rank = {}
        for r in M1_FAMILY_RANKS:
            ps = []
            for gid in graphs:
                sessions = load_graph_sessions(gid)
                recon5 = reconstruct(op, sessions, (r, *TUCKER_AMBIENT))
                rs = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]
                ps.append(U(phi_raw_r[gid], _flow_med(fn, rs)))
            cand_by_rank[r] = float(np.mean(ps))
        c_ord = [cand_by_rank[r] for r in M1_FAMILY_RANKS]
        rho = float(spearmanr(f1_ord, c_ord).correlation)
        repair_rho[name] = {"rho": rho, "by_rank": cand_by_rank}
        vals = " ".join(f"{cand_by_rank[r]:+.3f}" for r in M1_FAMILY_RANKS)
        print(f"  {name:>11}: rho(.,F1)={rho:+.3f}  [{vals}]  {'PASS' if rho>=1-1e-9 else 'fail'}")
    print()

    print("=" * 60)
    print("AUDIT VERDICT")
    print("=" * 60)
    print(f"  A cost:    proxy {'cheaper' if t_fix < t_pcmci else 'NOT cheaper'}")
    print(f"  B mini-M1: rho(proxy,F1)={rho_proxy_f1:.3f}  {'PASS' if b_pass else 'FAIL'}")
    print(f"  C sanity:  proxy shuffle={u_sh_mean:.3f}  {'PASS' if c_pass else 'CHECK'}")
    ok = (t_fix < t_pcmci) and b_pass and c_pass
    print(f"  -> {'DESIGN SOUND — proceed to write prereg' if ok else 'GAP — revise / consult'}")
    print("=" * 60)

    # Persist the audit evidence (NOT a pre-registered result; an audit record).
    rec = {
        "audit": "Causal-Aware Tucker proxy (pre-prereg, not a gate)",
        "corpus": CORPUS_DIR.name,
        "cost_ms": {"phi_fixed": t_fix * 1000, "phi_pcmci": t_pcmci * 1000},
        "ordering_rho_vs_F1": {
            "full_conditioning_precision": rho_proxy_f1,
            "U_via_PCMCI_reference": 1.0,
        },
        "u_by_rank_PCMCI": u_by_rank,
        "proxy_full_cond_by_rank": proxy_by_rank,
        "repair_candidates": repair_rho,
        "f1_by_rank": {str(k): v for k, v in F1_BY_RANK.items()},
        "proxy_shuffle_sanity": u_sh_mean,
        "verdict": "GAP — no differentiable proxy orders like U; escalate to consultant",
    }
    out = Path(__file__).parent / "results" / "audit_proxy_results.json"
    out.write_text(__import__("json").dumps(rec, indent=2))
    print(f"  audit record: {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
