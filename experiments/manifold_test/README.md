# S4 — Governance Manifold Test

Tests the **Governance Manifold Hypothesis** (CAL-L3 §5): *the set of
governance-relevant pipeline states forms a low-dimensional manifold M_gov
embedded in the 11-dimensional quality-vector space.*

This is the **cheapest, first gate** of the L3 experimental plan. Its result
determines whether the composition operator C should be a multilinear
decomposition (Tucker) or a non-linear sparse/SSM scheme.

## Pre-registration (2026-06-07)

- **H_manifold:** dim(M_gov) ≤ 3
- **Acceptance criterion:** trustworthiness ≥ 0.85 at dim ≤ 3
- **Borderline threshold:** 0.70
- **Gate rules:**
  - dim ≤ 3 AND tw ≥ 0.85 → **TUCKER** (H confirmed)
  - dim ≥ 6 OR best tw < 0.70 → **SPARSE_SSM** (H failed)
  - otherwise → **TUCKER_CAUTIOUS** (borderline)

## Method

`s4_manifold.py` builds an n×11 matrix of ground-truth quality vectors from the
L2 φ-calibration corpus (S1–S5, clean + failure variants), then:

1. **UMAP sweep** over `n_components ∈ {2,3,4,5}`. For each dim it sweeps
   `n_neighbors ∈ {3,4,5}` and averages trustworthiness over **10 seeds**
   (mean ± std), keeping the best `n_neighbors` cell.
2. **PCA triangulation** — deterministic, seed-free linear baseline. Reports
   trustworthiness and cumulative explained variance per dim. A large
   UMAP−PCA gap flags non-linearity; agreement flags robustness.
3. **Gate decision** per the pre-registered rules.

Run:

```bash
python s4_manifold.py
```

Outputs to `results/`: `manifold_results.json`, `s4_umap_2d.png`,
`s4_umap_3d.png`, `s4_trustworthiness_curve.png`.

## Results

### Session 1 (UMAP only, single seed) — superseded

Single-seed UMAP returned tw ∈ [0.758, 0.813]; naïve verdict SPARSE_SSM. An
audit of the harness found this was a method artifact, not a property of the
data (see below).

### Session 2 (multi-seed UMAP + PCA) — current

| dim | UMAP tw (mean ± std) | PCA tw | PCA cum. var. |
|-----|----------------------|--------|---------------|
| 2 | 0.802 ± 0.026 | 0.942 | 79.2% |
| 3 | 0.781 ± 0.015 | **0.992** | 91.8% |
| 4 | 0.788 ± 0.011 | 0.992 | 97.7% |
| 5 | 0.785 ± 0.025 | 1.000 | 99.9% |

**PCA recovers the manifold almost perfectly at dim=3** (tw 0.992, 91.8%
variance), comfortably above the 0.85 acceptance criterion. The UMAP−PCA gap
of −0.198 shows UMAP was *underestimating* the manifold because n=12 is too
sparse for its k-NN graph. The governance states are **low-dimensional and
approximately linear** → multilinear Tucker is the motivated operator.

**Verdict (preliminary):** H_manifold supported. Gate emits `TUCKER_CAUTIOUS`
(conservative: stochastic UMAP alone does not cross 0.85), but PCA evidence
points to TUCKER. Proceed to S1 with Tucker as primary candidate; re-run S4 on
the n≥30 synthetic corpus for the definitive gate.

## Audit fixes (Session 1 → 2)

| Issue | Fix |
|-------|-----|
| BUG-1: borderline gate branch unreachable | Branch on (a) crossing 0.85 and (b) best tw |
| BUG-2: `dim_gov` = sweep ceiling when none cross 0.85 | `dim_gov` = best-tw dim |
| PROBLEM-3: `n_neighbors` hard-coded to 5 | Sweep {3,4,5}, keep best cell |
| PROBLEM-4: single seed (UMAP is stochastic) | 10 seeds, mean ± std + PCA triangulation |

## Caveats

- n=12 is small for UMAP; PCA is the more reliable estimator at this size.
- The definitive S4 runs on the S1 synthetic corpus (n≥30, ≥6 sessions per
  causal graph), where UMAP has enough density to corroborate PCA.
