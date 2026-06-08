# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
S1 sanity check: confirm the injected causal signal is recoverable.

For each graph, for each ground-truth edge (i -> j), the child j at time t
should correlate with the parent i at time t-lag MORE than with a random
non-edge pair. We collapse the tensor to its per-cycle mean vector (average over
stages and agents), pool the lagged pairs across all sessions, and report:

  - mean lagged Pearson r for TRUE edges
  - mean lagged Pearson r for a sample of NON-edges (control)

A clear separation (true >> control) confirms the corpus carries the structure
S3 must recover. This is a generator self-test, NOT the S3 causal test itself.
"""
from __future__ import annotations

import json
import argparse
from itertools import product
from pathlib import Path

import numpy as np

DEFAULT_CORPUS = Path(__file__).parent / "corpus"
N_DIMS = 11


def load_sessions(gdir: Path) -> list[np.ndarray]:
    return [np.load(p) for p in sorted(gdir.glob("session_*.npy"))]


def mean_trajectory(T: np.ndarray) -> np.ndarray:
    """T (11×s×a×t) -> V (t×11): mean over stage and agent axes."""
    return T.mean(axis=(1, 2)).T  # (t, 11)


def lagged_corr(sessions: list[np.ndarray], i: int, j: int, lag: int) -> float:
    """Pooled Pearson r between parent_i[t-lag] and child_j[t] across sessions."""
    parent, child = [], []
    for T in sessions:
        V = mean_trajectory(T)
        parent.extend(V[:-lag, i])
        child.extend(V[lag:, j])
    parent, child = np.array(parent), np.array(child)
    if parent.std() < 1e-9 or child.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(parent, child)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="corpus self-test: lagged causal signal")
    parser.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS),
                        help="corpus directory (default: ./corpus)")
    args = parser.parse_args()
    CORPUS = Path(args.corpus)

    gt = json.loads((CORPUS / "ground_truth.json").read_text())
    manifest = json.loads((CORPUS / "corpus_manifest.json").read_text())
    lag = manifest["config"]["lag"]
    rng = np.random.default_rng(0)

    print(f"corpus sanity check (lagged Pearson r, lag={lag})  [{CORPUS.name}]")
    print("=" * 65)

    all_true, all_ctrl = [], []
    for gid, g in gt.items():
        sessions = load_sessions(CORPUS / gid)
        true_edges = {(e[0], e[1]) for e in g["edge_index"]}

        print(f"\n{gid}: {len(sessions)} sessions")
        true_rs = []
        for (i, j) in true_edges:
            r = lagged_corr(sessions, i, j, lag)
            true_rs.append(abs(r))
            print(f"  TRUE edge {i:2d}->{j:2d}   r={r:+.3f}")

        # Control: sample non-edge pairs (exclude self-loops and true edges).
        all_pairs = [(i, j) for i, j in product(range(N_DIMS), range(N_DIMS))
                     if i != j and (i, j) not in true_edges]
        ctrl_pairs = rng.choice(len(all_pairs), size=min(20, len(all_pairs)), replace=False)
        ctrl_rs = [abs(lagged_corr(sessions, *all_pairs[k], lag)) for k in ctrl_pairs]

        mt, mc = np.mean(true_rs), np.mean(ctrl_rs)
        all_true.extend(true_rs)
        all_ctrl.extend(ctrl_rs)
        print(f"  mean |r| TRUE edges:  {mt:.3f}")
        print(f"  mean |r| control(20): {mc:.3f}")
        print(f"  separation:           {mt - mc:+.3f}  {'OK' if mt > mc + 0.15 else 'WEAK'}")

    print("\n" + "=" * 65)
    gt_mean, ctrl_mean = np.mean(all_true), np.mean(all_ctrl)
    print(f"OVERALL  TRUE |r|={gt_mean:.3f}   CONTROL |r|={ctrl_mean:.3f}   "
          f"sep={gt_mean - ctrl_mean:+.3f}")
    verdict = "PASS — causal signal recoverable" if gt_mean > ctrl_mean + 0.15 else "FAIL — signal too weak"
    print(f"VERDICT: {verdict}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
