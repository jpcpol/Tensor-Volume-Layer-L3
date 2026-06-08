# S3 — Pre-Registration: Causal Conservation Test

**Commit this document BEFORE running any S3 code.** Per the §6.1 protocol, the
merge/commit SHA of this file is the pre-registration timestamp. Results are
reported verbatim, positive or negative.

- **Date:** 2026-06-08
- **Experiment:** S3 — does the composition operator C (Tucker) preserve the
  causal structure between quality dimensions? (Property 1 of C.)
- **Depends on:** S1 corpus (`synthetic_corpus/corpus/`, ground truth in
  `ground_truth.json`), S2 operator (`tucker_composition/tucker_operator.py`).

## Hypothesis

> **H_causal:** Causal edges between quality dimensions, present in the original
> session tensors, remain recoverable from the Tucker reconstruction V̂ = C(𝒯).
> Formally, a causal-discovery method applied to V̂ recovers the ground-truth
> directed edges with **F1 ≥ 0.70**.

This tests Property 1 (causal preservation): C must not collapse genuine causal
dependencies into spurious correlations, nor invent edges absent from the source.

## What is tested on what

The causal-discovery method is run on the **Tucker reconstruction** V̂, NOT on
the raw corpus. The raw corpus is known to carry the signal (S1 self-test:
true-edge |r|=0.53 vs control 0.12). S3's question is specifically whether C
*preserves* that signal through compression. As a control, we also run the same
method on the raw corpus and report both, so a failure can be attributed to C
(if raw passes but reconstruction fails) vs. to the method (if both fail).

## Method (fixed before running)

1. **Reconstruction.** For each graph G ∈ {G1, G2, G3}, stack its 30 sessions
   into 𝒯 (30×11×4×4×12), apply Tucker at the **pre-registered S2 rank
   (r0=3, dim=3, stage=3, agent=3, cycle=6)**, and reconstruct V̂. Collapse V̂ to
   per-session dimension trajectories: V̂ → series x_d[t] for each dim d, by
   averaging over stage and agent axes (the same reduction used in
   `validate_corpus.py`).

2. **Causal discovery: pairwise Granger causality.** For every ordered pair of
   distinct dimensions (i → j), test whether past values of x_i Granger-cause
   x_j, pooling the per-session trajectories. The generator's model is a
   lagged-linear influence, so Granger causality at **maxlag = 1** (the injected
   lag) is the matched test. Use the F-test p-value from `statsmodels`
   `grangercausalitytests`.

3. **Edge prediction.** Predict edge (i → j) if p-value < **α = 0.01**
   (Bonferroni-style strict threshold to control the 11×10 = 110 candidate
   ordered pairs per graph). The set of predicted edges is the discovered graph.

4. **Scoring.** Compare predicted edges to the ground-truth `edge_index` per
   graph. Compute precision, recall, F1. The primary metric is **micro-averaged
   F1 across all three graphs** (6 true edges total).

## Acceptance criteria (fixed before running)

| Outcome | Condition | Verdict |
|---------|-----------|---------|
| **PASS** | micro-F1 (on reconstruction V̂) ≥ 0.70 | Property 1 holds; C preserves causality |
| **PARTIAL** | 0.50 ≤ micro-F1 < 0.70 | C partially preserves; report as borderline, investigate which edges are lost |
| **FAIL** | micro-F1 < 0.50 | Property 1 violated; document in NEGATIVE_RESULTS.md, reconsider C or its rank |

If the **raw-corpus** F1 is itself < 0.70, the negative result is attributed to
the discovery method, not to C — in that case S3 is inconclusive and the method
must be revised (e.g., transfer entropy, or PCMCI) before re-running.

## Fixed parameters

- Tucker rank: (3, 3, 3, 3, 6) — the S2 dim-mode-justified rank.
- Granger maxlag: 1 (the injected causal lag).
- Edge α: 0.01.
- Trajectory reduction: mean over (stage, agent) → per-dim series per session.
- Pooling: concatenate per-session lagged pairs (as in `validate_corpus.py`).
- Direction handling: directed edges only; (i→j) and (j→i) scored separately.

## What would falsify H_causal

- C reconstructs V̂ with high variance (S2: 98%) but the causal edges are NOT
  recoverable from V̂ (F1 < 0.50) → C achieves numerical fidelity while
  destroying causal structure (semantic collapse). This is the key failure mode
  §3 warns about, and it would be a publishable negative result.

---

# AMENDMENT 1 — S3 Run 2 (2026-06-08)

Run 1 (commit 929c229) returned **inconclusive**: pairwise Granger reached
recall=1.00 but low precision (transitive/reverse false positives), so even the
RAW-corpus control failed the 0.70 ceiling (raw micro-F1=0.545). Per the run-1
attribution rule, the negative was attributed to the *method*, not to C. This
amendment fixes the method **before** S3 run 2 is executed. Run 1 code, results,
and `NEGATIVE_RESULTS.md` are retained unchanged.

## Revised method (fixed before running run 2)

1. **Causal discovery: PCMCI with partial correlation (ParCorr).** PCMCI
   (`tigramite`) is designed to separate *direct* causal links from indirect /
   transitive ones in autocorrelated time series, by conditioning each pairwise
   link on the parents of the target. This directly addresses the run-1 failure
   (3→6 flagged as the transitive composition of 3→5→6).

2. **Per-session time-series handling.** PCMCI is fit per graph on the stacked
   per-session dimension trajectories (mean over stage, agent → t×11 per
   session). To respect session boundaries we fit PCMCI **per session** and
   aggregate links by majority vote across the 30 sessions (a link is predicted
   if it is significant in ≥ 50% of sessions). This avoids the cross-session
   concatenation boundary artifact of run 1.

3. **Parameters (fixed):**
   - `tau_min = 1`, `tau_max = 1` (the injected lag).
   - ParCorr independence test, significance `pc_alpha = 0.01` (matched to run-1 α).
   - Link predicted at lag 1 only; contemporaneous links ignored (the generator
     injects lagged influence only).
   - Majority-vote threshold: 0.50 of sessions.

4. **Conditions, scoring, and acceptance are UNCHANGED from the original
   pre-registration:** primary = Tucker reconstruction V̂; control = raw corpus;
   micro-F1 across G1/G2/G3 vs the 6 ground-truth edges; PASS ≥ 0.70,
   PARTIAL 0.50–0.70, FAIL < 0.50.

## Revised attribution rule (fixed before running run 2)

- The revised method must first reach **raw-corpus micro-F1 ≥ 0.70** for the
  reconstruction test to be interpretable.
- If raw ≥ 0.70 **and** reconstruction ≥ 0.70 → **PASS**: C preserves causality.
- If raw ≥ 0.70 **and** reconstruction < 0.50 → **clean refutation** of
  Property 1: C achieves high variance but destroys causal structure
  (semantic collapse). Publishable negative.
- If raw < 0.70 → still inconclusive; the method, not C, is the bottleneck;
  document and stop (do not iterate methods indefinitely — report the limit).

## Secondary analysis (exploratory, not gating)

- **κ(V)-vs-causal-F1 trade-off:** repeat the reconstruction-condition discovery
  across Tucker session-ranks r0 ∈ {1,2,3,5,8} (cycle-mode rank fixed) and report
  reconstruction F1 as a function of κ(V). This characterizes how much causal
  structure survives at each compression level. Exploratory — reported as a curve,
  not subject to the PASS/FAIL gate.
