# S3 — Run 1 Result: INCONCLUSIVE (method-attributed)

Per the §6.1 protocol, this records the S3 outcome verbatim. The pre-registered
run (commit 1a3076d) returned a negative result that the pre-registration's own
attribution rule classifies as **inconclusive**, not as a refutation of C.

## Result (run 1, 2026-06-08)

| Condition | micro-F1 | precision | recall |
|-----------|---------|-----------|--------|
| Raw corpus (control) | **0.545** | 0.375 | 1.000 |
| Tucker reconstruction (primary) | **0.048** | 0.025 | 0.500 |

Pre-registered verdict mapping: reconstruction F1 = 0.048 < 0.50 → FAIL. **But**
the attribution rule fires: raw F1 = 0.545 < 0.70, so the negative is attributed
to the *discovery method*, not to C. **S3 is inconclusive; the method must be
revised and re-run (after amending the pre-registration).**

## Diagnosis

Two distinct problems, both real:

### 1. The method (pairwise Granger) is not selective enough — even on raw data

Per-graph raw recall is **1.00 everywhere** (all 6 true edges found), but
precision is low (0.25–0.67) due to false positives. The FPs are not random —
they are structurally-correlated pairs:

- G1 raw predicts `3→6` (security→maintainability): the transitive composition
  of the true chain `3→5→6`. Pairwise Granger cannot separate a direct edge from
  a transitive/confounded one.
- G1 also predicts `6→5` (the reverse of a true edge) and `7→5`.

So pairwise Granger at α=0.01 recovers every true edge but adds transitive and
reverse edges. F1 is capped by precision, not recall.

### 2. Tucker reconstruction at low rank injects spurious cross-dim coupling

The reconstruction condition explodes to **117 FPs** (F1=0.048). The low-rank
Tucker core (r0=3, cycle=6) that achieved 98% variance (S2) **smooths the
trajectories and mixes dimensions through the factor matrices**, creating
lagged correlations between nearly all pairs. This is consistent with the
"semantic collapse" risk §3 warns about: high numerical fidelity (variance) does
NOT guarantee preserved causal *structure*.

However, because the control method already fails its own threshold, we **cannot
yet attribute the reconstruction collapse to C alone** — a sharper method might
recover edges from the reconstruction that pairwise Granger misses. This must be
disambiguated before any claim about Property 1.

## Revision plan (requires amended pre-registration before re-running)

1. **Replace pairwise Granger with a method that controls for indirect paths:**
   - **Conditional / multivariate Granger** (condition on the other dims), or
   - **PCMCI** (`tigramite`) — designed exactly for distinguishing direct from
     indirect causal links in time series with autocorrelation, or
   - **Transfer entropy with conditioning**.
2. **Establish a raw-corpus ceiling first.** The revised method must reach
   F1 ≥ 0.70 on the RAW corpus before the reconstruction test is meaningful.
   If raw passes and reconstruction fails, *that* is a clean Property-1 refutation.
3. **Consider the Tucker rank.** If reconstruction at r0=3 collapses causality
   but a higher cycle-mode rank preserves it, report κ(V) vs. causal-F1 as a
   trade-off curve — this is itself a publishable characterization of C.
4. **Amend `PRE_REGISTRATION.md`** with the new method + thresholds in a new
   commit (the SHA is the new pre-registration), then re-run as S3-run-2.

## What this does NOT change

- S4 (manifold) and S2 (tractability, κ(V) sub-linear) stand — they do not depend
  on the causal-discovery method.
- The Property-1 question remains **open**, not refuted. The pre-registration's
  attribution clause did its job: it prevented a method artifact from being
  mistaken for a refutation of C.

---

# S3 — Run 2 Result: INCONCLUSIVE (method ceiling), with a publishable trade-off

Run 2 (PCMCI, pre-registered in AMENDMENT 1, commit ed60d77) fixed run-1's
false-positive problem but hit a recall ceiling. Per the amended attribution
rule, since the raw control still falls below 0.70, **we stop iterating methods
and report the limit** — this is pre-registration discipline, not a failure to
try harder.

## Result (run 2, 2026-06-08)

| Condition | micro-F1 | precision | recall |
|-----------|---------|-----------|--------|
| Raw corpus (control) | **0.667** | 1.000 | 0.500 |
| Tucker reconstruction (primary) | **0.089** | 0.047 | 0.833 |

PCMCI achieved **precision 1.00** on raw (zero transitive/reverse FPs — it solved
run-1's exact failure), but recovered only the **first edge of each causal chain**
(`3→5`, `7→6`, `4→8`) and missed the **second edge** (`5→6`, `6→1`, `8→9`). With
only 12 cycles per session and a strict ≥50% majority vote across 30 sessions,
the attenuated second-link signal does not reach significance in enough sessions.
Raw F1=0.667 sits just below the 0.70 ceiling.

**Verdict: INCONCLUSIVE (method ceiling).** The amended rule fires: raw < 0.70 →
the method, not C, is the bottleneck → stop and report.

## The publishable finding: κ(V)-vs-causal-F1 trade-off

The secondary analysis (exploratory, pre-registered) is the scientifically
valuable result. Re-running PCMCI on Tucker reconstructions across session-ranks:

| r0 | κ(V) | reconstruction causal-F1 |
|----|------|--------------------------|
| 1 | 162 | 0.058 |
| 2 | 324 | 0.042 |
| 3 | 486 | 0.089 |
| 5 | 810 | 0.185 |
| 8 | 1296 | 0.226 |

Causal-F1 rises **monotonically with κ(V)** (less compression → more causal
structure preserved). This is direct evidence that the Tucker compression which
achieved 98% variance (S2) **trades away causal structure** — the semantic-collapse
mechanism §3 warns about, now measured as a curve rather than asserted. See
`results/s3_kappa_vs_causalf1.png`.

**Important caveat:** because the raw ceiling is 0.667, the reconstruction
absolute F1 values are not directly interpretable as "C preserves X% of
causality." The *shape* (monotone increase with κ) is the robust finding; the
*levels* are confounded by the method's recall ceiling.

## Decision (per pre-registration)

1. **Stop iterating discovery methods.** The amended rule explicitly forbids
   indefinite method-shopping once raw < 0.70. Two methods (Granger, PCMCI) have
   now hit method-side ceilings on a 12-cycle-per-session corpus.
2. **Property 1 is not cleanly testable on the current corpus geometry.** The
   bottleneck is temporal resolution: 12 cycles is too short for robust
   per-session causal discovery of a 2-link chain.
3. **Two clean paths forward (each a NEW pre-registration, not a method swap):**
   - **S1-bis (corpus revision):** regenerate the corpus with longer sessions
     (e.g. t_cycles = 40–60) so per-session causal discovery has the temporal
     resolution to recover both chain links → re-establishes a raw ceiling ≥ 0.70,
     after which the reconstruction test becomes a clean Property-1 gate.
   - **Report the trade-off as the L3 causal result:** the κ(V)-vs-F1 curve is
     itself a characterization of C (compression vs. causal fidelity), reportable
     independent of the raw-ceiling issue, with the caveat above.

## What still stands

- S4 (manifold, dim≈2–3) and S2 (κ(V) sub-linear, Property 4) are unaffected.
- Property 1 remains **open**. Run 2 strengthens the *qualitative* claim that
  low-rank Tucker erodes causal structure (monotone κ-F1 curve), but a clean
  quantitative gate awaits a longer-session corpus (S1-bis).
