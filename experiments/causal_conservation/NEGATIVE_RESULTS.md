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
