# S1-bis + S3-bis — Pre-Registration (joint)

**Commit this document BEFORE generating the S1-bis corpus or writing any S3-bis
code.** Per the §6.1 protocol, this commit SHA is the pre-registration timestamp.
Generated as a single joint pre-registration so the causal gate (S3-bis) is fixed
*before* the corpus exists — this makes it impossible to tune the corpus toward
the gate.

- **Date:** 2026-06-08
- **Motivation:** S3 runs 1–2 were inconclusive. Diagnosis (consultant-reviewed):
  the bottleneck is the pre-registered per-session-plus-vote **unit of analysis**,
  which lacks temporal degrees of freedom for PCMCI's *partial* correlation on the
  second edge of each chain. The signal itself is strong (bivariate lagged
  r ≈ 0.72–0.85 on every edge including second links). S1-bis tests whether
  **longer sessions, with the unit of analysis held fixed**, lift the raw ceiling
  to ≥ 0.70 and thereby make a clean Property-1 gate possible.
- **Depends on / reuses:** `causal_generator.py` (parameterized for length),
  `validate_corpus.py` (self-test), `run_s3_run2.py` (S3-bis = same PCMCI method).

## Part A — S1-bis corpus

### What changes (and ONLY this)

| Parameter | S1 | S1-bis | Rule |
|-----------|----|--------|------|
| `t_cycles` | 12 | **48** | the single factor under test (4× temporal resolution) |
| `recover_rate` | U(0.04, 0.12) | **U(0.01, 0.03)** | scaled ÷(48/12)=÷4 a priori, so the shock traverses a comparable fraction of its recovery over the longer session (stays slow-decaying, not flat by mid-session) |
| `drift_std` | 0.01 | **0.0025** | scaled ÷4 a priori, so accumulated drift over 48 cycles ≈ S1's over 12 (avoids pushing dims out of [0,1] → nonlinear clipping) |
| seed | 1234 | **5678** | fresh seed (independent draw) |

**Everything else is identical to S1:** the 3 causal graphs (G1/G2/G3), edge
weights, `lag=1`, `shock_strength`, `noise_std`, `n_sessions=30`, the AR(1)
single-shock dynamics, and the tensor lift. Only session length and the two
length-scaled parameters change.

### Scaling rule (fixed a priori — no post-hoc tuning)

`recover_rate` and `drift_std` are set by the deterministic rule
`new = old / (t_cycles / 12)`. These values are **not** to be re-tuned after
inspecting the self-test. If the self-test fails, the corpus is documented as a
generator failure — parameters are NOT searched to make it pass.

### S1-bis self-test (abort gate — NOT the causal gate)

`validate_corpus.py` is run on the S1-bis corpus. It measures **bivariate** pooled
lagged Pearson r for true edges vs. a 20-pair non-edge control (exactly as in S1).

- **Pass (proceed to S3-bis):** overall true-edge mean |r| − control mean |r| ≥ 0.15
  (the same separation S1 achieved: 0.53 vs 0.12).
- **Fail (abort):** separation < 0.15 → document as generator failure; do NOT run
  S3-bis; do NOT re-tune.

> **This self-test is deliberately weaker than the S3-bis gate.** It is a
> *bivariate* sanity check on the generator. Passing it does NOT predict that
> S3-bis (PCMCI partial correlation under per-session + vote) passes — they test
> different things. The self-test only confirms the corpus carries signal; it
> cannot and must not be read as evidence about the causal gate.

## Part B — S3-bis causal gate (fixed BEFORE the corpus exists)

### Method (identical to S3 run 2 — the unit of analysis is held fixed)

PCMCI + ParCorr, fit **per session**, aggregate by **majority vote ≥ 50%**,
`tau_min = tau_max = 1`, `pc_alpha = 0.01`. Primary condition = Tucker
reconstruction V̂ at rank (3,3,3,3,6); control = raw corpus. Micro-F1 across
G1/G2/G3 vs the 6 ground-truth edges. The unit of analysis (per-session + vote)
is **unchanged from run 2 by design** — S1-bis isolates session length as the
only difference.

### Two-stage gate (fixed a priori)

**Stage 1 — precondition (does longer-session fix the unit-of-analysis ceiling?):**

- Raw-corpus micro-F1 ≥ 0.70 → precondition MET; proceed to Stage 2.
- Raw-corpus micro-F1 < 0.70 → precondition NOT met. Longer sessions did not lift
  the ceiling under the fixed unit of analysis. Stop; report. (No further method
  iteration — that remains off-limits per the S3 amendment.)

**Stage 2 — Property 1 verdict (only if Stage 1 met):**

| Reconstruction micro-F1 | Verdict |
|-------------------------|---------|
| ≥ 0.70 | **PASS** — C preserves causal structure (Property 1 holds) |
| 0.50 – 0.70 | **PARTIAL** — C partially preserves; report as borderline |
| < 0.50 | **CLEAN REFUTATION** — C achieves high variance but destroys causal structure (semantic collapse). Publishable negative. |

This is the clean gate that runs 1–2 could not reach: with a valid raw ceiling,
a low reconstruction F1 is unambiguously attributable to C, not the method.

### Secondary (exploratory, not gating)

If Stage 1 is met, re-report the κ(V)-vs-causal-F1 trade-off across Tucker ranks
on the S1-bis corpus. With a valid raw ceiling, the curve's **absolute levels**
become interpretable (not just its direction) — closing the caveat left open in
S3 run 2. Exploratory: reported as a curve, not subject to the gate.

## What would make this whole experiment fail cleanly

- **S1-bis self-test fails** → generator failure; the length-scaling rule was
  wrong. Documented, no S3-bis.
- **Stage 1 fails (raw < 0.70 even at 48 cycles)** → the per-session + vote unit
  of analysis is fundamentally too stringent for a 2-link chain regardless of
  length; the clean Property-1 gate is not reachable with this unit of analysis,
  and any future attempt requires a *different* unit fixed in a new pre-registration.
- **Stage 2 = CLEAN REFUTATION** → C = low-rank Tucker does not preserve
  causality; this refutes Property 1 and is a primary publishable result.

## Fixed parameters (summary)

- Corpus: t_cycles=48, recover_rate∈U(0.01,0.03), drift_std=0.0025, seed=5678,
  n_sessions=30, 3 graphs, weights/lag/shock/noise unchanged from S1.
- Self-test abort threshold: separation ≥ 0.15.
- S3-bis: PCMCI+ParCorr, per-session + majority vote ≥0.50, tau=1, pc_alpha=0.01,
  Tucker rank (3,3,3,3,6).
- Gate: Stage 1 raw ≥0.70 precondition; Stage 2 PASS ≥0.70 / PARTIAL / REFUTE <0.50.
