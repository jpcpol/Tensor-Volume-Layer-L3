# TCI — Test de Calibración Instrumental (Pre-Registration)

**Commit this document BEFORE writing or running any TCI code.** Per the §6.1
protocol, this commit SHA is the pre-registration timestamp. The TCI validates a
*measurement instrument* (an unsupervised causal-conservation metric U), NOT an
operator. Same logic as discovering that pairwise Granger was a defective
instrument for Property 1: validate the instrument before trusting it.

- **Date:** 2026-06-09
- **Reuses:** S1-bis corpus (`corpus_s1bis/`, t_cycles=48) and the byte-identical
  PCMCI machinery of `run_s3_run2.py` (commit cdd64e8). No new corpus, no new
  causal estimator.
- **Motivation:** S3-bis refuted Property 1 for low-rank Tucker (raw F1=1.000,
  reconstruction F1=0.135). The framework reframed L3 around causal-observability
  preservation (Causality ≻ Topology ≻ Reconstruction). Before designing any
  causality-aware operator C, we need a metric U that scores causal conservation
  **without ground truth** — because outside the lab there is no known causal
  graph. The TCI tests whether a candidate U is a trustworthy instrument.

## Audit findings that shaped this pre-registration (2026-06-09)

A pre-registration-stage audit (no operator run) established two facts that fix
the design below:

1. **The naive "PCMCI persistence" U is broken.** Defined as
   `|E_raw ∩ E_recon| / |E_raw|` it equals **1.0** for the Tucker reconstruction
   while the supervised F1 is 0.135. It is a *recall* — blind to false positives.
   But Tucker's failure mode is **fabricating** spurious edges (G1: 2 true edges
   in `E_raw`, 28 edges in `E_recon` → 26 spurious), not losing true ones. Any
   recall-style U is blind to the actual failure and is rejected a priori.
2. **The correct invariant is causal-flow magnitude, not edge presence.** Tucker
   redistributes causal flow into spurious channels. U must compare the
   *continuous flow pattern*, not a thresholded edge set. The PCMCI `val_matrix`
   provides exactly this (partial correlation of var_i(t-1) with var_j(t),
   conditioned on other parents), is already computed in S3-bis, conditions out
   transitive links, and needs no threshold.

## Definition of U (fixed a priori)

For a set of session series, define the **causal-flow matrix** Φ ∈ ℝ^(11×11):

```
Φ[i,j] = median over sessions of  val_matrix[i, j, 1]
```

i.e. the lag-1 PCMCI partial-correlation strength of dimension i → j, aggregated
across the 30 sessions of a graph by the **median** (robust to per-session
outliers; matches the per-session spirit of the S3-bis unit of analysis). The
diagonal (i=j) is excluded.

**U compares the flow pattern of the original vs a transformed corpus:**

```
U(Φ_ref, Φ_test) = Pearson correlation between the off-diagonal entries of
                   Φ_ref and Φ_test   (flattened, i≠j)
```

- `U = 1`  → the transformed corpus routes causal flow through the *same channels
  with the same relative strengths* as the original.
- `U ≈ 0`  → the flow pattern is unrelated (causal structure not preserved).

**Why correlation, not a distance:** Tucker *redistributes* flow (concentrates
part into spurious channels) while possibly preserving global magnitude. A
matrix correlation penalizes that redistribution — it asks "does flow go through
the same channels?", which is the causal question. A raw L2/Frobenius distance
would be partly fooled by overall-magnitude preservation. Correlation is fixed as
the **primary** statistic; secondary L1 distance is reported for context only and
is NOT part of any gate.

**Scope (honest caveat):** val_matrix measures *linear, direct* lagged causal
flow. U is therefore an instrument for linear-direct causal conservation. This is
appropriate for validating against S3-bis, whose causality is linear-by-
construction and whose governance manifold S4 found approximately linear. If
future corpora/real data carry non-linear causality, U-via-val_matrix would be
blind to it and transfer-entropy / CMI would be needed — a separate future
pre-registration, not this one.

## What U is validated against

The supervised causal F1 from S3-bis (against ground-truth edges). U is a *good
instrument* iff it agrees with F1 on the ordering of a family of corpora whose
true causal fidelity is known. This is tested by two pre-registered criteria.

## Criterion M1 — Causal monotonicity (primary gate)

For the family of Tucker reconstructions `{V_r}` at session-ranks
r ∈ {1, 2, 3, 5, 8}, with **known** supervised causal F1
(0.090, 0.104, 0.135, 0.174, 0.245 from S3-bis, monotone in r), the candidate U
must preserve the ordering:

```
F1(V_r) > F1(V_s)  ⇒  U(Φ_raw, Φ_{V_r}) > U(Φ_raw, Φ_{V_s})
```

U does NOT need to be linear in F1, nor calibrated to F1's scale — only to
**rank** the family in the same order. Operationalized as:

- **PASS M1:** Spearman rank correlation ρ(U, F1) over the 5-point family **= 1.0**
  (strict — perfect ordering, the family is small and the order is unambiguous).
- **FAIL M1:** ρ < 1.0 (any inversion). U is rejected as an instrument.

The 5-point κ-F1 family is the calibration scale. A metric that only separated
"original vs any compression" (a compression detector in disguise) would NOT
necessarily order the *intermediate* ranks correctly — M1 over the full family
rules that out.

> **Honest limit of M1 alone:** with 5 points, ρ=1.0 has a 1/120 ≈ 0.8% chance
> under a random-ordering null — it is necessary but NOT strong on its own. M1 is
> not the validation; **M1 ∧ C2 is.** M1 checks that U tracks causal fidelity; C2
> checks that what U tracks is causality and not compression. Neither alone
> validates U.

## Criterion C2 — Causal perturbation control (identifiability gate)

M1 alone cannot prove U measures *causality* rather than *compression*: under
compression, causality and κ move together. C2 breaks that confound. Take the
**original, uncompressed** corpus and destroy its temporal dependencies while
holding κ fixed and marginals unchanged:

- **Perturbation (fixed):** the flow is measured on the per-dimension series
  `to_dim_series(T)` (mean over stage/agent → series t×11). The shuffle is applied
  **at that series level**: for each session and each of the 11 dimensions,
  independently permute the cycle order of that dimension's time series with a
  fresh random permutation (seed fixed = 20260609 for reproducibility). This
  destroys lag-1 causal structure between dimensions, preserves each dimension's
  marginal exactly (same values, reordered in time), and applies no compression
  (κ undefined/unchanged — the corpus is uncompressed).
- **The real gate is the floor, not the ceiling.** `Φ` is a deterministic
  function of the corpus, so `U(raw, raw) = 1.0` trivially (re-estimation on
  identical input). The ceiling is a sanity check, not the test. The test is
  whether destroying causality at fixed marginals collapses U.

Operationalized:

- **PASS C2:** `U(Φ_raw, Φ_shuffle) < 0.5` (the instrument collapses when causal
  flow is destroyed though marginals are intact). Report `U(raw,raw)=1.0` as the
  trivial sanity ceiling.
- **FAIL C2:** `U(Φ_raw, Φ_shuffle) ≥ 0.5` → U responds to something other than
  lagged causal flow (e.g. marginal geometry or static correlation); reject U.

> **Marginal-preservation safeguard:** the shuffle MUST be a pure temporal
> permutation per dimension — no added noise, no altered values. If marginals
> changed, a drop in U could be attributed to changed geometry rather than
> destroyed causality, reintroducing the confound C2 exists to remove.

## Verdict logic (fixed a priori)

| M1 | C2 | Verdict |
|----|----|---------|
| PASS (ρ=1.0) | PASS | **U is a validated instrument.** Proceed to design C_causal / Π_gov with U in the objective (separate pre-registration). |
| PASS | FAIL | U orders by compression, not causality — **reject**. Instrument invalid. |
| FAIL | — | U does not track causal fidelity — **reject**. Instrument invalid. |

**No method-shopping clause.** This pre-registration fixes ONE U (val_matrix flow
correlation). If it fails M1 or C2, the TCI reports **"no validated instrument in
this family"** as an honest negative — we do NOT iterate formulas until one
passes. A different U family would require a new, separately justified
pre-registration. (Two causal-discovery methods already hit ceilings in S3 runs
1–2; the discipline that made S3-bis clean applies here too.)

## What would falsify / what each outcome means

- **U passes both** → L3 gains an observable objective `U(C(T))` evaluable without
  ground truth. The problem shifts from "design C blind" to "optimize C under a
  validated causal metric." Strongest possible outcome.
- **U passes M1, fails C2** → U is a sophisticated compression detector. Important
  negative: it means edge/flow-overlap metrics on this corpus cannot be trusted
  to isolate causality from compression. Documents a real instrument hazard.
- **U fails M1** → val_matrix flow correlation does not even track known causal
  fidelity; the linear-flow invariant is insufficient and the next pre-registration
  must consider a richer invariant (CMI, interventional).

## Fixed parameters (summary)

- Flow source: PCMCI `val_matrix[i,j,1]`, ParCorr, tau=1, pc_alpha=0.01
  (identical to S3-bis run_s3_run2).
- Φ aggregation across sessions: median.
- U statistic: Pearson correlation of off-diagonal Φ entries (primary). L1
  distance reported, non-gating.
- M1 family: Tucker session-ranks {1,2,3,5,8}, F1 from S3-bis. Gate: Spearman ρ=1.0.
- C2 perturbation: per-dimension temporal permutation at the to_dim_series level,
  shuffle seed=20260609, marginals preserved, no compression. Gate:
  U(raw,shuffle)<0.5 (floor is the real test; U(raw,raw)=1.0 trivial ceiling).
- Verdict requires BOTH M1 (ρ=1.0) AND C2; neither alone validates U.
- Corpus: S1-bis (corpus_s1bis, t_cycles=48). Reuses cdd64e8 machinery.
