# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
S3 Run 2 — Causal conservation test with PCMCI (Property 1 of C).

Pre-registered: PRE_REGISTRATION.md AMENDMENT 1 (commit ed60d77, 2026-06-08).
This implements that amendment verbatim. Run-1 code (run_s3.py) is retained.

Fix vs run 1: pairwise Granger flagged transitive/reverse false positives
(3→6 = composition of 3→5→6), so even the raw control failed. PCMCI + ParCorr
conditions each link on the target's other parents, separating direct from
indirect links. Fit per session, aggregate by majority vote (≥50% of sessions).

Conditions / scoring / acceptance unchanged from the original pre-registration:
primary = Tucker reconstruction V̂, control = raw corpus, micro-F1 across
G1/G2/G3 vs 6 ground-truth edges, PASS ≥ 0.70 / PARTIAL 0.50–0.70 / FAIL < 0.50.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from sys import path as _sys_path

import numpy as np

warnings.filterwarnings("ignore")

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

_sys_path.insert(0, str(Path(__file__).parent.parent / "tucker_composition"))
from tucker_operator import TuckerCompositionOperator  # noqa: E402

HERE = Path(__file__).parent
CORPUS_DIR = HERE.parent / "synthetic_corpus" / "corpus"
OUTPUT_DIR = HERE / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Pre-registered fixed parameters (amendment 1) ───────────────────────────
TUCKER_RANK = (3, 3, 3, 3, 6)   # primary reconstruction rank (S2-justified)
TAU = 1                          # tau_min = tau_max = 1 (injected lag)
PC_ALPHA = 0.01                  # ParCorr significance
VOTE_THRESHOLD = 0.50            # majority vote across sessions
N_DIMS = 11

PASS_THRESHOLD = 0.70
PARTIAL_THRESHOLD = 0.50

# Secondary (exploratory) — kappa(V) vs causal-F1 trade-off across session ranks.
SECONDARY_SESSION_RANKS = [1, 2, 3, 5, 8]


def load_graph_sessions(gid: str) -> list[np.ndarray]:
    return [np.load(p) for p in sorted((CORPUS_DIR / gid).glob("session_*.npy"))]


def to_dim_series(T: np.ndarray) -> np.ndarray:
    """T (11×4×4×12) -> per-dim trajectory (t × 11): mean over stage, agent."""
    return T.mean(axis=(1, 2)).T  # (t, 11)


def pcmci_edges_one_session(series: np.ndarray) -> set:
    """Run PCMCI on one session's (t×11) series; return predicted lag-1 edges."""
    dataframe = pp.DataFrame(series, var_names=[str(d) for d in range(N_DIMS)])
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    try:
        res = pcmci.run_pcmci(tau_min=TAU, tau_max=TAU, pc_alpha=PC_ALPHA)
    except Exception:
        return set()
    pmat = res["p_matrix"]  # (N, N, tau_max+1)
    edges = set()
    for i in range(N_DIMS):
        for j in range(N_DIMS):
            if i == j:
                continue
            if pmat[i, j, TAU] < PC_ALPHA:
                edges.add((i, j))
    return edges


def discover_edges_majority(series_list: list[np.ndarray]) -> set:
    """Fit PCMCI per session; predict an edge if it appears in >= VOTE_THRESHOLD."""
    n = len(series_list)
    counts: dict[tuple[int, int], int] = {}
    for series in series_list:
        for e in pcmci_edges_one_session(series):
            counts[e] = counts.get(e, 0) + 1
    return {e for e, c in counts.items() if c / n >= VOTE_THRESHOLD}


def score(predicted: set, truth: set) -> dict:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def run_condition(label: str, series_by_graph: dict, truth_by_graph: dict) -> dict:
    print(f"\n--- {label} ---")
    per_graph = {}
    all_pred, all_truth = set(), set()
    for gid in sorted(series_by_graph.keys()):
        pred = discover_edges_majority(series_by_graph[gid])
        truth = set(map(tuple, truth_by_graph[gid]))
        sc = score(pred, truth)
        per_graph[gid] = {"predicted": sorted(map(list, pred)),
                          "truth": sorted(map(list, truth)), **sc}
        all_pred |= {(gid, *e) for e in pred}
        all_truth |= {(gid, *e) for e in truth}
        print(f"  {gid}: pred={sorted(map(list, pred))}  truth={sorted(map(list, truth))}  "
              f"P={sc['precision']:.2f} R={sc['recall']:.2f} F1={sc['f1']:.2f}")
    micro = score(all_pred, all_truth)
    print(f"  MICRO: tp={micro['tp']} fp={micro['fp']} fn={micro['fn']}  "
          f"P={micro['precision']:.3f} R={micro['recall']:.3f} F1={micro['f1']:.3f}")
    return {"per_graph": per_graph, "micro": micro}


def reconstruct(op: TuckerCompositionOperator, sessions: list[np.ndarray], rank: tuple) -> np.ndarray:
    """Tucker-compose then reconstruct -> 5d tensor (n×11×4×4×12)."""
    import tensorly as tl
    T5 = op.stack(sessions)
    core, factors, _ = op.compose(T5, rank)
    return tl.to_numpy(tl.tucker_to_tensor((tl.tensor(core), [tl.tensor(f) for f in factors])))


def main() -> int:
    print("S3 Run 2 — Causal Conservation Test with PCMCI (Property 1 of C)")
    print(f"  tau={TAU}  pc_alpha={PC_ALPHA}  vote>={VOTE_THRESHOLD}  rank={TUCKER_RANK}")
    print(f"  Pre-registration: AMENDMENT 1 (commit ed60d77)")

    gt = json.loads((CORPUS_DIR / "ground_truth.json").read_text())
    graphs = sorted(gt.keys())
    truth_by_graph = {gid: gt[gid]["edge_index"] for gid in graphs}
    op = TuckerCompositionOperator()

    raw_series, recon_series = {}, {}
    for gid in graphs:
        sessions = load_graph_sessions(gid)
        raw_series[gid] = [to_dim_series(T) for T in sessions]
        recon5 = reconstruct(op, sessions, TUCKER_RANK)
        recon_series[gid] = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]

    # Control first (attribution), then primary.
    raw_result = run_condition("RAW corpus (control)", raw_series, truth_by_graph)
    recon_result = run_condition("Tucker reconstruction (PRIMARY)", recon_series, truth_by_graph)

    raw_f1 = raw_result["micro"]["f1"]
    recon_f1 = recon_result["micro"]["f1"]

    # Pre-registered (amended) verdict logic.
    if raw_f1 < PASS_THRESHOLD:
        verdict = ("INCONCLUSIVE — raw control F1 < 0.70; method still the bottleneck. "
                   "Report the limit; do not iterate methods further.")
    elif recon_f1 >= PASS_THRESHOLD:
        verdict = "PASS — Property 1 holds; C preserves causality"
    elif recon_f1 < PARTIAL_THRESHOLD:
        verdict = ("CLEAN REFUTATION — raw passes but reconstruction collapses "
                   "(semantic collapse); Property 1 violated by C")
    else:
        verdict = "PARTIAL — C partially preserves causality (borderline)"

    print("\n" + "=" * 70)
    print("S3 RUN 2 VERDICT")
    print("=" * 70)
    print(f"  Raw corpus micro-F1:     {raw_f1:.3f}  (control — must reach 0.70 first)")
    print(f"  Reconstruction micro-F1: {recon_f1:.3f}  (primary)")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    # ── Secondary: kappa(V) vs causal-F1 trade-off (exploratory) ─────────────
    print("\n--- SECONDARY (exploratory): kappa(V) vs causal-F1 across Tucker ranks ---")
    secondary = []
    for r0 in SECONDARY_SESSION_RANKS:
        rank = (r0, 3, 3, 3, 6)
        rseries, all_pred, all_truth, kappa = {}, set(), set(), None
        for gid in graphs:
            sessions = load_graph_sessions(gid)
            T5 = op.stack(sessions)
            _, _, res = op.compose(T5, rank)
            kappa = res.kappa
            recon5 = reconstruct(op, sessions, rank)
            rseries[gid] = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]
            pred = discover_edges_majority(rseries[gid])
            truth = set(map(tuple, truth_by_graph[gid]))
            all_pred |= {(gid, *e) for e in pred}
            all_truth |= {(gid, *e) for e in truth}
        f1 = score(all_pred, all_truth)["f1"]
        secondary.append({"session_rank": r0, "kappa": kappa, "recon_micro_f1": f1})
        print(f"  r0={r0}  kappa={kappa:5d}  recon_micro_F1={f1:.3f}")

    output = {
        "experiment": "S3 Run 2 — Causal Conservation Test (PCMCI)",
        "pre_registration": "PRE_REGISTRATION.md AMENDMENT 1 (commit ed60d77, 2026-06-08)",
        "fixed_params": {"tucker_rank": list(TUCKER_RANK), "tau": TAU, "pc_alpha": PC_ALPHA,
                         "vote_threshold": VOTE_THRESHOLD,
                         "pass_threshold": PASS_THRESHOLD, "partial_threshold": PARTIAL_THRESHOLD},
        "raw_control": raw_result,
        "reconstruction_primary": recon_result,
        "verdict": {"raw_micro_f1": raw_f1, "reconstruction_micro_f1": recon_f1, "verdict": verdict},
        "secondary_kappa_vs_f1": secondary,
    }
    out_path = OUTPUT_DIR / "s3_run2_pcmci_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
