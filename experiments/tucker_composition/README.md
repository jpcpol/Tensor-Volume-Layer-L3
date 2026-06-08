# S2 — Tucker Composition Operator C

Implements the composition operator **C = Tucker decomposition** (motivated by
the S4 manifold result) and characterizes κ(V), reconstruction error, and the
Property-4 tractability claim on the S1 synthetic corpus.

## What C is

C aggregates a set of session tensors into a volume V:

```text
V = C({T^(s)}_{s=1..n}) = Tucker_core( stack_s T^(s) )
```

The n session tensors (each `11×4×4×12`) are stacked into a 5th-order tensor
`𝒯 ∈ ℝ^(n × 11 × 4 × 4 × 12)` (mode-0 = sessions). Tucker-HOOI compresses it to
a core G of multilinear rank `(r0, r1, r2, r3, r4)`. **The core G is V.** The
effective rank `κ(V) = product of core dims` — the structural complexity measure
the L4 Efficiency Hypothesis predicts bounds `Cost(M(V))`.

The session-mode rank `r0` is the Property-4 quantity: how few latent
session-patterns reconstruct all n sessions. Ambient-mode ranks are fixed at
`dim=3` (S4 confirmed dim(M_gov)≈2–3), `stage=3, agent=3, cycle=6`.

## Run

```bash
python run_s2.py   # writes results/s2_tucker_results.json
```

## Results (S1 corpus, 30 sessions/graph)

### Analysis A — session-rank sweep

Even at `r0=1`, Tucker explains **~98% of variance** with **197× compression**.
The 30 sessions of each graph share a near-identical latent structure (they share
one causal graph), so a single session-pattern already captures the structure.

| graph | r0=1 var_exp | r0=1 compression | r0=8 var_exp |
|-------|--------------|------------------|--------------|
| G1 | 0.9810 | 197× | 0.9865 |
| G2 | 0.9769 | 197× | 0.9850 |
| G3 | 0.9792 | 197× | 0.9861 |

Relative Frobenius error floors at ~8–12% even with full ambient ranks — this is
the **irreducible injected noise** of the S1 corpus (noise_std=0.03 + random walk
+ stage/agent offsets). Tucker correctly captures structure and discards noise,
so **variance explained** — not error ceiling — is the meaningful budget.

### Analysis B — Property 4 (Tractability)

Minimal session-rank `r0*` to reach variance_explained ≥ 0.98, as n grows:

| graph | r0* @ n=10 | r0* @ n=20 | r0* @ n=30 | r0 ratio | verdict |
|-------|-----------|-----------|-----------|----------|---------|
| G1 | 1 | 1 | 1 | 1.00 | sub-linear |
| G2 | 1 | 2 | 2 | 2.00 | sub-linear |
| G3 | 1 | 2 | 2 | 2.00 | sub-linear |

As n triples (10→30), r0* grows at most 2× (G2/G3) or stays constant (G1) —
well below the linear ratio of 3.0. **Property 4 HOLDS: κ(V) grows sub-linearly
with n_sessions.** C is tractable for continuous pipelines.

## Verdict

C = Tucker is a working composition operator: high compression, near-complete
variance retention, and sub-linear κ(V) growth. The two open properties remain:

- **Property 1 (Causal preservation)** — tested in **S3**: does M(V) recover the
  ground-truth causal edges from `ground_truth.json`?
- **Property 2 (Temporal coherence)** — partially exercised here (cycle mode
  retained at rank 6); full cross-session drift test deferred to S3/S4-temporal.

## Files

- `tucker_operator.py` — `TuckerCompositionOperator`: stack, compose, rank_sweep
- `run_s2.py` — Analysis A (rank sweep) + Analysis B (Property-4 tractability)
- `results/s2_tucker_results.json` — full metrics
