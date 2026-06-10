# Causal-Aware Operator Search — Pre-Registration

**Commit this document BEFORE writing or running any search code.** Its commit SHA
is the pre-registration timestamp. This experiment searches for a Tucker rank
configuration that maximizes the **validated** unsupervised causal metric U, and
checks whether the U-optimal operator is also better by the *supervised* causal F1
— i.e. whether optimizing the validated compass actually buys causal fidelity.

- **Date:** 2026-06-09
- **Reuses:** S1-bis corpus (`corpus_s1bis/`, t_cycles=48), the TCI metric U
  (`run_tci.py`, commit c8cf496), and the S3-bis machinery (`run_s3_run2.py`,
  ParCorr/PCMCI, tau=1, pc_alpha=0.01). No new corpus, no new estimator, no new
  metric.
- **Depends on:** TCI verdict (U validated, M1∧C2 PASS) and the proxy audit
  (`audit_proxy.py`, commit abd7871) which **refuted** a differentiable proxy:
  U's ordering depends on PCMCI's discrete PC selection step. Therefore we
  optimize U **directly** via derivative-free search — no proxy.

## Motivation and the precise question

S3-bis showed low-rank Tucker destroys causal structure. The TCI gave us U: an
unsupervised metric that orders causal fidelity (ρ(U,F1)=1.0 on the rank family).
The audit closed the door on a cheap differentiable surrogate. So the operator
problem is a **search**, not a gradient descent. The question this experiment
answers:

> Over the Tucker rank space, does maximizing U(C(T)) select an operator that also
> improves the supervised causal F1 — or does U-maximization decouple from causal
> fidelity (instrument overfitting)?

This is the smallest honest step: it does **not** introduce a new operator family
(still Tucker — per the consultant, we have evidence *pure low-rank* Tucker fails,
not that Tucker does), and it does **not** yet build Π_gov or the full
`C = Π_gov ∘ C_causal ∘ C_compress`. It tests whether U is a usable *objective*,
having already proven it a usable *instrument*.

## The operator under search

`C` = `TuckerCompositionOperator.compose` at multilinear rank
`(r0, r_dim, r_stage, r_agent, r_cycle)` on the stacked 5-tensor
(n_sessions × 11 × 4 × 4 × 48). The reconstruction `to_dim_series(recon)` feeds the
flow matrix Φ exactly as in the TCI.

## Objective (fixed a priori)

```
score(C) = mean over graphs of  U(Φ_raw_g, Φ_recon_g(C))
```

where U is the **validated** TCI metric verbatim (Pearson correlation of
off-diagonal median PCMCI val_matrix, tau=1, pc_alpha=0.01). We **maximize** score.
U is the primary objective directly — no proxy, no differentiable surrogate.

## Search space (fixed a priori)

A small, exhaustively enumerable grid (derivative-free search is just full
enumeration here — the space is O(10²), not O(10⁶), so Bayesian opt / CMA-ES would
be over-engineering and we fix exhaustive grid for full reproducibility):

- `r0` (session rank): {1, 2, 3, 5, 8}  — the audited Property-4 axis.
- `r_dim`: {2, 3, 4}  — brackets the S4 manifold dim (≈2–3) on both sides.
- `r_cycle`: {4, 6, 9}  — brackets the TCI's fixed 6 on both sides.
- `r_stage`, `r_agent`: **fixed at 3** (S3-bis/TCI value; not the object of study).

Grid size: 5 × 3 × 3 = **45 configurations**. All enumerated; the full
(config → score) table is reported, not just the argmax. The TCI baseline
`(r0, 3, 3, 3, 6)` is contained in the grid (r_dim=3, r_cycle=6) so results are
directly comparable to the calibration family.

## Primary outcome (fixed a priori)

`C* = argmax_C score(C)` over the 45-config grid, with its score `U* = score(C*)`.

Reference points (from the TCI, same corpus/metric):
- shuffle floor U ≈ 0.068, raw ceiling U = 1.000, best calibration-family Tucker
  (r0=8, ambient (3,3,3,6)) U ≈ 0.441.

## Confirmatory gate G1 — U-optimum must also improve supervised F1 (anti-overfit)

The danger of optimizing U directly: the search could inflate U without improving
real causal fidelity (overfitting the instrument). To rule this out, we compute the
**supervised** causal F1 (S3-bis scoring vs ground-truth edges, `discover_edges_majority`
+ micro-F1 across G1/G2/G3) for the U-optimal config `C*` and for the TCI baseline
`C_base = (best-r0, 3, 3, 3, 6)`.

- **PASS G1:** `F1(C*) ≥ F1(C_base)` — the U-optimal operator is at least as good
  by the *independent supervised* metric. U-maximization did not decouple from
  causal fidelity.
- **FAIL G1:** `F1(C*) < F1(C_base)` — U-maximization found a config that scores
  high on U but *worse* on supervised F1. This would mean U, while a valid
  *ranking* instrument over the calibration family, is **not safe as a direct
  optimization objective** outside that family (instrument overfitting). Honest
  negative; report it and do not deploy U as an objective without a guard.

> Why G1 and not "max F1": we do NOT optimize F1 (it needs ground truth, absent at
> deployment — the whole reason U exists). F1 is used **only** as an external
> confirmatory check on the U-selected operator, exactly as the TCI used F1 to
> validate U. We never select on F1.

## Confirmatory gate G2 — monotone-in-U sanity

Across the grid, the Spearman correlation between `score(C)` and supervised F1(C)
**on the subset of configs we score** must be **positive** (we score the full grid's
U but, for cost, compute supervised F1 only on the configs needed for G1 plus the
calibration family already known from S3-bis). G2 is reported, non-blocking — a
negative ρ would flag instrument overfitting even if G1 passes by luck on two
points.

## Verdict logic (fixed a priori)

| G1 | Verdict |
|----|---------|
| PASS | **U is usable as a direct optimization objective.** `C*` is the U-optimal Tucker operator; proceed to design Π_gov (lexicographic topological tie-break) and Q3 composition as separate preregs. |
| FAIL | **U ranks but does not safely optimize.** Report the decoupling config; the next prereg must constrain the search (e.g. structure-preservation guard) before using U as an objective. |

**No method-shopping clause.** ONE objective (U verbatim), ONE grid (fixed above),
ONE primary outcome (argmax), ONE confirmatory gate (G1). If G1 fails, that is the
result — we do not re-pick the grid or the objective to make `C*` look good.

## What each outcome means

- **G1 PASS** → L3 has not just a validated instrument but a validated *objective*:
  operator search under U yields causally-better operators. The program can move to
  the governance projection Π_gov with the search loop established.
- **G1 FAIL** → an important hazard: U is a good *referee* (orders known cases) but
  a leaky *target* (optimizing it directly exploits its blind spots). This would
  echo S3-bis (reconstruction was a leaky target) and the audit (the proxy was a
  leaky target) at one level up — and would force a structure-preservation
  constraint into the objective before any operator is trusted.

## Fixed parameters (summary)

- Operator: Tucker via `TuckerCompositionOperator.compose`.
- Objective: U (TCI metric verbatim), maximized, directly (no proxy).
- Search: exhaustive grid, 45 configs: r0∈{1,2,3,5,8} × r_dim∈{2,3,4} ×
  r_cycle∈{4,6,9}, r_stage=r_agent=3.
- Primary: argmax U over the grid → C*, U*.
- G1 (confirmatory, blocking): supervised micro-F1(C*) ≥ micro-F1(C_base), where
  C_base = (argmax-r0 at ambient (3,3,3,6)) — the TCI calibration baseline.
- G2 (reported, non-blocking): sign of Spearman(score, F1) over scored configs.
- Corpus: S1-bis (corpus_s1bis, t_cycles=48). PCMCI ParCorr, tau=1, pc_alpha=0.01.
