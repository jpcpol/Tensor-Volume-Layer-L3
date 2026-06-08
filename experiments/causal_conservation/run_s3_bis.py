# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
S3-bis — Causal conservation gate on the S1-bis (48-cycle) corpus.

Pre-registered: synthetic_corpus/PRE_REGISTRATION_S1bis.md (commit d38a366).
This wrapper REUSES the exact PCMCI method, unit of analysis, and scoring from
run_s3_run2.py by direct import — the only difference from run 2 is the corpus
(S1-bis, t_cycles=48). Holding the method byte-identical is the methodological
guarantee that S1-bis isolates session length as the single change.

Two-stage gate (pre-registered):
  Stage 1 (precondition): raw micro-F1 >= 0.70  → proceed; else stop (the
           per-session+vote unit of analysis is too stringent regardless of length).
  Stage 2 (Property 1):   reconstruction micro-F1
           >= 0.70  PASS (C preserves causality)
           0.50-0.70 PARTIAL
           < 0.50    CLEAN REFUTATION (semantic collapse).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# Reuse the EXACT run-2 method (import the pure functions + fixed params).
from run_s3_run2 import (
    PARTIAL_THRESHOLD,
    PASS_THRESHOLD,
    PC_ALPHA,
    SECONDARY_SESSION_RANKS,
    TAU,
    TUCKER_RANK,
    VOTE_THRESHOLD,
    discover_edges_majority,
    reconstruct,
    score,
    to_dim_series,
)
from tucker_operator import TuckerCompositionOperator  # noqa: E402 (added to path by run_s3_run2)

HERE = Path(__file__).parent
CORPUS_DIR = HERE.parent / "synthetic_corpus" / "corpus_s1bis"
OUTPUT_DIR = HERE / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_graph_sessions(gid: str) -> list[np.ndarray]:
    return [np.load(p) for p in sorted((CORPUS_DIR / gid).glob("session_*.npy"))]


def run_condition(label: str, series_by_graph: dict, truth_by_graph: dict) -> dict:
    print(f"\n--- {label} ---")
    per_graph, all_pred, all_truth = {}, set(), set()
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


def main() -> int:
    print("S3-bis — Causal Conservation Gate on S1-bis (48-cycle corpus)")
    print(f"  tau={TAU}  pc_alpha={PC_ALPHA}  vote>={VOTE_THRESHOLD}  rank={TUCKER_RANK}")
    print(f"  Pre-registration: PRE_REGISTRATION_S1bis.md (commit d38a366)")
    print(f"  Corpus: {CORPUS_DIR}")

    if not (CORPUS_DIR / "corpus_manifest.json").exists():
        print("ERROR: S1-bis corpus not found. Generate it first:")
        print("  python ../synthetic_corpus/causal_generator.py "
              "--scaled-length --t-cycles 48 --seed 5678 --out corpus_s1bis")
        return 1

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

    raw_result = run_condition("RAW corpus (control / Stage-1 precondition)", raw_series, truth_by_graph)
    recon_result = run_condition("Tucker reconstruction (PRIMARY / Stage 2)", recon_series, truth_by_graph)

    raw_f1 = raw_result["micro"]["f1"]
    recon_f1 = recon_result["micro"]["f1"]

    # ── Two-stage pre-registered gate ────────────────────────────────────────
    if raw_f1 < PASS_THRESHOLD:
        stage1 = "NOT MET"
        verdict = ("STAGE 1 FAILED — raw F1 < 0.70 even at 48 cycles. The "
                   "per-session+vote unit of analysis is too stringent for a "
                   "2-link chain regardless of length. Stop; a clean gate needs "
                   "a DIFFERENT unit of analysis (new pre-registration).")
    else:
        stage1 = "MET"
        if recon_f1 >= PASS_THRESHOLD:
            verdict = "PASS — Property 1 holds; C (low-rank Tucker) preserves causal structure"
        elif recon_f1 < PARTIAL_THRESHOLD:
            verdict = ("CLEAN REFUTATION — raw passes but reconstruction collapses: "
                       "C achieves high variance while destroying causal structure "
                       "(semantic collapse). Property 1 violated.")
        else:
            verdict = "PARTIAL — C partially preserves causality (borderline)"

    print("\n" + "=" * 70)
    print("S3-bis VERDICT (two-stage gate)")
    print("=" * 70)
    print(f"  Stage 1 (raw >= 0.70 precondition): {stage1}  (raw F1={raw_f1:.3f})")
    print(f"  Stage 2 (reconstruction F1):        {recon_f1:.3f}")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    # ── Secondary: kappa(V) vs causal-F1 (levels now interpretable if Stage 1 met) ──
    print("\n--- SECONDARY: kappa(V) vs causal-F1 across Tucker ranks ---")
    secondary = []
    for r0 in SECONDARY_SESSION_RANKS:
        rank = (r0, 3, 3, 3, 6)
        all_pred, all_truth, kappa = set(), set(), None
        for gid in graphs:
            sessions = load_graph_sessions(gid)
            _, _, res = op.compose(op.stack(sessions), rank)
            kappa = res.kappa
            recon5 = reconstruct(op, sessions, rank)
            rseries = [to_dim_series(recon5[s]) for s in range(recon5.shape[0])]
            pred = discover_edges_majority(rseries)
            truth = set(map(tuple, truth_by_graph[gid]))
            all_pred |= {(gid, *e) for e in pred}
            all_truth |= {(gid, *e) for e in truth}
        f1 = score(all_pred, all_truth)["f1"]
        secondary.append({"session_rank": r0, "kappa": kappa, "recon_micro_f1": f1})
        print(f"  r0={r0}  kappa={kappa:5d}  recon_micro_F1={f1:.3f}")

    output = {
        "experiment": "S3-bis — Causal Conservation Gate (S1-bis, 48-cycle)",
        "pre_registration": "PRE_REGISTRATION_S1bis.md (commit d38a366)",
        "corpus": "S1-bis synthetic (90 sessions, t_cycles=48)",
        "method_identical_to": "run_s3_run2.py (PCMCI, per-session + vote>=0.50)",
        "fixed_params": {"tucker_rank": list(TUCKER_RANK), "tau": TAU, "pc_alpha": PC_ALPHA,
                         "vote_threshold": VOTE_THRESHOLD,
                         "pass_threshold": PASS_THRESHOLD, "partial_threshold": PARTIAL_THRESHOLD},
        "raw_control": raw_result,
        "reconstruction_primary": recon_result,
        "verdict": {"stage1_precondition": stage1, "raw_micro_f1": raw_f1,
                    "reconstruction_micro_f1": recon_f1, "verdict": verdict},
        "secondary_kappa_vs_f1": secondary,
    }
    out_path = OUTPUT_DIR / "s3bis_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
