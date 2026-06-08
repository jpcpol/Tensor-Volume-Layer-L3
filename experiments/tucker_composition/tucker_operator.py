# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
S2 — Composition operator C as Tucker decomposition.

C aggregates a set of session tensors {T^(s)}_{s=1..n} into a volume V:

    V = C({T^(s)}) = Tucker_core( stack_s T^(s) )

The n session tensors (each 11×4×4×12) are stacked into a 5th-order tensor

    𝒯 ∈ ℝ^(n × 11 × 4 × 4 × 12)      (mode-0 = sessions)

and Tucker-HOOI compresses it to a core G of multilinear rank
(r0, r1, r2, r3, r4). The compressed core G IS the volume V. The effective rank
κ(V) is the product of the core dimensions (the structural complexity measure
that the L4 Efficiency Hypothesis predicts bounds Cost(M(V))).

Property 4 (Tractability) is the central test here: if a rank with r0 << n
reconstructs 𝒯 with low error, then V does NOT grow linearly with n_sessions —
the composition is tractable for continuous pipelines.

This module implements the operator and a rank sweep; metrics (κ, reconstruction
error, compression ratio, variance explained) are reported per rank.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

tl.set_backend("numpy")

# Ambient mode sizes of a session tensor (excluding the session/stack mode).
AMBIENT_SHAPE = (11, 4, 4, 12)   # dim × stage × agent × cycle


@dataclass
class TuckerResult:
    rank: tuple[int, ...]            # multilinear rank (r0..r4)
    core_shape: tuple[int, ...]      # actual core G shape
    kappa: int                       # κ(V) = product of core dims (effective rank)
    reconstruction_error: float      # relative Frobenius error ||𝒯 - 𝒯̂|| / ||𝒯||
    variance_explained: float        # 1 - error²
    compression_ratio: float         # |𝒯| / (|G| + |factors|)
    n_params_full: int               # |𝒯|
    n_params_tucker: int             # |G| + Σ|U_i|


class TuckerCompositionOperator:
    """C: {T^(s)} → V (Tucker core), with a rank sweep for κ(V) characterization."""

    def stack(self, sessions: list[np.ndarray]) -> np.ndarray:
        """Stack n session tensors into 𝒯 ∈ ℝ^(n × 11 × 4 × 4 × 12)."""
        for T in sessions:
            if T.shape != AMBIENT_SHAPE:
                raise ValueError(f"session tensor shape {T.shape} != {AMBIENT_SHAPE}")
        return np.stack(sessions, axis=0)

    def compose(self, tensor_5d: np.ndarray, rank: tuple[int, ...]) -> tuple[np.ndarray, list, TuckerResult]:
        """
        Apply Tucker-HOOI at the given multilinear rank.

        Returns (core G, factors [U0..U4], TuckerResult metrics).
        """
        core, factors = tucker(tl.tensor(tensor_5d), rank=list(rank), init="svd", random_state=0)
        core = tl.to_numpy(core)
        factors = [tl.to_numpy(f) for f in factors]

        recon = tl.to_numpy(tl.tucker_to_tensor((tl.tensor(core), [tl.tensor(f) for f in factors])))
        err = float(np.linalg.norm(tensor_5d - recon) / np.linalg.norm(tensor_5d))

        n_full = int(np.prod(tensor_5d.shape))
        n_core = int(np.prod(core.shape))
        n_factors = int(sum(f.size for f in factors))
        n_tucker = n_core + n_factors

        result = TuckerResult(
            rank=tuple(rank),
            core_shape=tuple(core.shape),
            kappa=n_core,
            reconstruction_error=err,
            variance_explained=float(1.0 - err ** 2),
            compression_ratio=float(n_full / n_tucker),
            n_params_full=n_full,
            n_params_tucker=n_tucker,
        )
        return core, factors, result

    def rank_sweep(
        self,
        tensor_5d: np.ndarray,
        session_ranks: list[int],
        dim_rank: int = 3,
        stage_rank: int = 3,
        agent_rank: int = 3,
        cycle_rank: int = 6,
    ) -> list[TuckerResult]:
        """
        Sweep the SESSION-mode rank r0 ∈ session_ranks, holding the ambient-mode
        ranks fixed at values motivated by S4 (dim≈3 manifold) and the tensor
        geometry. The session-mode rank is the Property-4 quantity: how few
        latent session-patterns reconstruct all n sessions.

        Ambient ranks default: dim=3 (S4 manifold dim), stage=3, agent=3, cycle=6.
        """
        n = tensor_5d.shape[0]
        results = []
        for r0 in session_ranks:
            r0_eff = min(r0, n)
            rank = (r0_eff, dim_rank, stage_rank, agent_rank, cycle_rank)
            _, _, res = self.compose(tensor_5d, rank)
            results.append(res)
        return results
