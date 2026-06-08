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

# S3 — Run 2 Result: INCONCLUSIVE, with a publishable trade-off

Run 2 (PCMCI, pre-registered in AMENDMENT 1, commit ed60d77) fixed run-1's
false-positive problem but the raw control landed just below the 0.70 ceiling.
Per the amended attribution rule we **stop and report the limit** — this is
pre-registration discipline, not a failure to try harder.

> **External methodological review (2026-06-08).** A consultant reviewed runs 1–2
> for methodological validity. Verdicts: (1) the attribution rule is sound and the
> stop is correct — *but* 0.667 and 0.15 both trip the rule while being very
> different scientific situations, so the discussion must separate the
> *experimental conclusion* (INCONCLUSIVE) from the *accumulated evidence* (0.667
> ≈ 0.70). (2) Run 2 is not invalidated; it correctly answers the pre-registered
> question, which turned out to be more restrictive than initially understood.
> (3) Granger→PCMCI is controlled methodological debugging, not method-shopping.
> (4) The κ(V)–F1 trade-off, not the PASS/FAIL of Property 1, is the strongest
> result of S3. The reframing below adopts this review.

## Result (run 2, 2026-06-08)

| Condition | micro-F1 | precision | recall |
|-----------|---------|-----------|--------|
| Raw corpus (control) | **0.667** | 1.000 | 0.500 |
| Tucker reconstruction (primary) | **0.089** | 0.047 | 0.833 |

PCMCI achieved **precision 1.00** on raw (zero transitive/reverse FPs — it solved
run-1's exact failure), but recovered only the **first edge of each causal chain**
(`3→5`, `7→6`, `4→8`) and missed the **second edge** (`5→6`, `6→1`, `8→9`).

**What this actually measures (consultant reframing).** The bottleneck is **not**
the algorithm and **not** a defect in the corpus. PCMCI *can* recover the full
chain (see the diagnostic below). What Run 2 demonstrates is narrower and
legitimate:

> The second-link causal signal is not strong enough to survive an *independent
> per-session estimate followed by a majority vote* — the unit of analysis that
> the pre-registration fixed. The pre-registered question turned out to be more
> restrictive than we understood when we wrote it.

So the correct framing is not "the method failed" but "Run 2 correctly answered
the pre-registered question; the question demanded more per-session temporal
evidence than an attenuated second link provides in 12 cycles."

**Conclusion vs. evidence (consultant point 1).** The control fell **0.033**
below the pre-registered threshold. By experimental discipline the classification
stays **INCONCLUSIVE**, but the accumulated evidence shows the method is
*approaching* the required ceiling — qualitatively different from a method that
fails outright (e.g. raw = 0.15). Both trip the same rule; they are not the same
scientific situation.

## Diagnostic (post-hoc, NOT pre-registered — for review only)

To locate the bottleneck we applied PCMCI to the RAW corpus three ways (this
diagnostic does **not** change Run 2's verdict; it is reported, not substituted):

| Aggregation | True edges found (of 2/graph) | Total predictions |
|-------------|-------------------------------|-------------------|
| Per-session + majority vote (Run-2 unit of analysis) | **1 / 2** (first link) | 1 |
| Sessions **concatenated** (30×12 = 360 points) | **2 / 2** (both links) | 4 |
| Union over per-session fits | 2 / 2 | ~85 (noise) |

Concatenation recovers BOTH true edges in all three graphs with only 4 total
predictions. This confirms the recall ceiling is a property of the
**pre-registered per-session-plus-vote unit of analysis**, not a bug and not an
irrecoverable corpus.

> ⚠️ **This is a trap, not an option.** It is tempting to say "concatenation
> works, so switch to concatenation." That would change the **unit of analysis
> after seeing the results** — the exact move pre-registration exists to prevent.
> We therefore do **not** rerun Run 2 with concatenation. The diagnostic is
> reported to explain the ceiling; the pre-registered Run 2 stands as-is. Any use
> of concatenation must be a NEW pre-registration with the unit of analysis fixed
> in advance (see Decision below).

**Verdict: INCONCLUSIVE** (control 0.033 below ceiling; method approaching it).
The amended rule fires: raw < 0.70 → stop and report, do not iterate further.

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

**Important caveat (and a point where we are more cautious than the reviewer).**
The reviewer suggested the trade-off "would likely survive even if a future
corpus clears the 0.70 ceiling." We agree only for the **shape**. Because the raw
ceiling is 0.667, the reconstruction *absolute* F1 values are not interpretable as
"C preserves X% of causality," and if S1-bis lifts the raw ceiling the curve's
*levels* will re-scale — its slope could change. What survives with confidence is
the **direction** (more compression → less causal structure), not the quantitative
curve. We report the monotone direction as robust and the levels as provisional.

## Decision (per pre-registration)

1. **Stop iterating discovery methods.** The amended rule forbids indefinite
   method-shopping once raw < 0.70. Two methods (Granger, PCMCI) have now reached
   their ceiling under the pre-registered unit of analysis. Per the reviewer, this
   stop is correct and is *not* method-shopping — it was controlled methodological
   debugging with an explicit stop rule that we honored.
2. **The bottleneck is the unit of analysis, not "a bad corpus" and not a bug.**
   The pre-registered per-session-plus-vote estimate demands more temporal
   evidence than an attenuated second link supplies in 12 cycles. The corpus
   carries the signal (concatenation recovers both edges); the per-session unit of
   analysis is simply more stringent than anticipated.
3. **Two clean paths forward — each a NEW pre-registration with the unit of
   analysis fixed in advance (never a post-hoc swap):**
   - **Path A — S1-bis (longer sessions):** regenerate the corpus with
     t_cycles ≈ 40–60 so the *per-session* unit of analysis (unchanged) has the
     temporal resolution to recover both chain links → re-establishes a raw
     ceiling ≥ 0.70 → the reconstruction test becomes a clean Property-1 gate.
   - **Path B — new unit of analysis, pre-registered up front:** if a future
     pre-registration *fixes* a pooled/concatenated unit of analysis BEFORE
     looking at results, that is legitimate. Reusing concatenation now, after the
     diagnostic, is the trap flagged above and is explicitly off-limits.
   - **Either way — report the trade-off as the L3 causal result:** the κ(V)-vs-F1
     curve characterizes C (compression vs. causal fidelity), reportable
     independent of the raw-ceiling issue, with the direction/levels caveat above.

## What still stands (as of runs 1–2)

- S4 (manifold, dim≈2–3) and S2 (κ(V) sub-linear, Property 4) are unaffected.
- Property 1 remains **open**. Run 2 strengthens the *qualitative* claim that
  low-rank Tucker erodes causal structure (monotone κ-F1 curve), but a clean
  quantitative gate awaits a longer-session corpus (S1-bis).

---

# S3-bis — Result: CLEAN REFUTATION of Property 1

S3-bis (Path A) ran the **byte-identical** PCMCI method and per-session+vote unit
of analysis (imported from `run_s3_run2.py`) on the S1-bis corpus (t_cycles=48,
the only change), per the joint pre-registration `PRE_REGISTRATION_S1bis.md`
(commit d38a366). The two-stage gate resolved cleanly.

## Result (S3-bis, 2026-06-08)

| Condition | micro-F1 | precision | recall |
|-----------|---------|-----------|--------|
| Raw corpus (Stage-1 precondition) | **1.000** | 1.000 | 1.000 |
| Tucker reconstruction (Stage-2 primary) | **0.135** | 0.072 | 1.000 |

**Stage 1 — precondition MET.** With 48 cycles and the unchanged unit of
analysis, PCMCI recovers all 6 ground-truth edges across G1/G2/G3 with zero false
positives — raw F1 = **1.000**. This quantitatively confirms the consultant's
diagnosis: the run-2 ceiling of 0.667 was a shortage of per-session temporal
degrees of freedom for the partial-correlation test on the second chain edge,
not the method and not C. Longer sessions fix it. The unit of analysis is no
longer the bottleneck.

**Stage 2 — Property 1 REFUTED.** With a valid raw ceiling of 1.000, the
reconstruction result is unambiguous: F1 = **0.135** (77 false positives, perfect
recall). Low-rank Tucker reconstruction invents dozens of spurious causal edges.
Because the raw control is perfect, this collapse is attributable to **C, not the
method**. This is a **clean refutation**, not an inconclusive: C = low-rank Tucker
achieves 98 % variance (S2) while **destroying causal structure** — the semantic
collapse §3 warned about, now a pre-registered, conclusive negative result.

## Secondary — κ(V) vs causal-F1, now with interpretable levels

With raw F1 = 1.000, the trade-off **levels** are interpretable (closing the
direction-only caveat of run 2):

| r0 | κ(V) | reconstruction causal-F1 |
|----|------|--------------------------|
| 1 | 162 | 0.090 |
| 2 | 324 | 0.104 |
| 3 | 486 | 0.135 |
| 5 | 810 | 0.174 |
| 8 | 1296 | 0.245 |

The curve is monotone (more κ → more causal structure) **and** its ceiling is
low: even at κ=1296 (minimal compression in the tested range), F1 reaches only
0.245 against a perfect control. **No tested compression regime preserves
causality.** This is stronger than run 2's direction-only finding.

## Implication for C = low-rank Tucker

Property 1 (causal preservation) is **refuted** for low-rank Tucker as the sole
composition operator C. C satisfies Property 3 (dimensional stability) and
Property 4 (tractability, κ(V) sub-linear), and achieves high variance fidelity —
but it does **not** preserve the causal structure between quality dimensions.

This does not sink L3; it sharpens it. The result means **variance-optimal
compression and causality-preserving compression are different objectives**, and
a composition operator for governance must optimize for the latter, not the
former. Candidate directions (each a future, separately pre-registered
experiment, not a post-hoc patch):

1. **Causality-aware C:** add a causal-structure-preserving penalty to the Tucker
   objective, or use a decomposition that respects the lagged-influence graph
   (e.g. a structured/sparse core constrained to the causal support).
2. **Higher cycle-mode rank as a knob:** the κ-F1 curve suggests causal fidelity
   improves with the cycle-mode rank; characterize whether any tractable rank
   reaches an acceptable F1, accepting the variance/causality trade-off explicitly.
3. **Report the negative as a primary L3 contribution:** "low-rank Tucker is a
   variance-optimal but causality-destroying composition operator" is a clean,
   publishable finding that motivates the search for causality-aware C.

## What still stands

- S4 (manifold, dim≈2–3) and S2 (κ(V) sub-linear, Property 4) are unaffected and
  remain positive results.
- Property 1 is now **refuted** for low-rank Tucker (not open) — a stronger,
  pre-registered, method-clean outcome than the inconclusive runs 1–2.
- The pre-registration discipline paid off: by fixing the two-stage gate before
  the corpus existed, the refutation is attributable to C rather than to method
  choices made after seeing data.
