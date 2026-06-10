# Form 1 — Structural-Mask Falsifier (Pre-Registration)

**Commit this document BEFORE writing or running any Form-1 code.** Its commit SHA
is the pre-registration timestamp. Form 1 is a **falsifier**, not the definitive
C_causal. It tests one sharp prediction made falsifiable by Q_L3.2A.

- **Date:** 2026-06-09
- **Reuses:** S1-bis corpus (t=48), the S3-bis PCMCI machinery, the TCI U, the
  Tucker operator. No new corpus, no new estimator, no new metric.
- **Builds on:** Q_L3.2A (commit e743a52) decomposed Tucker's failure: S
  (consistency) = 1.0 everywhere (no sign corruption), C (coverage) high (true
  edges recovered, no omission), but R/|E| explode (raw 2 edges → Tucker 14–28) —
  **Tucker over-generates causal structure (spurious-edge fabrication).** Form 1
  attacks exactly that failure mode.

## The scientific question (not "does a mask work?")

> Is Tucker's causal loss explained **principally** by spurious connections?

- If pruning spurious edges lifts U materially toward raw → the loss is structural,
  and causal conservation in L3 depends on preserving structural **sparsity**, not
  geometric reconstruction. (Strong result — identifies the mathematical property
  that defines governable causality in CAL.)
- If pruning barely moves U → spurious inflation is a *symptom*, not the cause, and
  C_causal must be rethought entirely.

## The operator under test (fixed a priori)

```
V̂_Tucker  =  reconstruct(Tucker, rank=(8,3,3,6))      # the best-U Tucker (C* from operator search)
V̂_masked  =  mask( V̂_Tucker , support )               # zero the flow on channels outside `support`
```

The mask operates on the **recovered structure**: after estimating the per-session
val_matrix / edge set of V̂_Tucker, retain flow only on edges in `support`; the
masked corpus's flow matrix Φ is recomputed on the pruned structure. Concretely the
mask is applied to the edge set used to build Φ for U (zero off-support entries of
the val_matrix before forming Φ), so U/R/C/S are all measured on the pruned
structure with the SAME machinery as everywhere else.

### Two pre-registered supports (both run)

- **`support_raw` (PRIMARY, self-referential):** the majority-vote edge set
  recovered on the **raw** corpus. Uses NO ground truth → a valid deployment
  surrogate (raw exists outside the lab). **This is the falsifier that decides.**
- **`support_GT` (reference ceiling, oracle):** the true ground-truth edges. Not
  available at deployment; gives the upper bound of what pruning can achieve. The
  **raw↔GT gap** quantifies what the deployment surrogate loses vs the oracle.

## Predicted signature (fixed a priori, from Q_L3.2A's H3)

If structural inflation is the dominant failure (H3), masking should produce:

```
|E|  ↓   (toward raw's 2)
R    ↓   (toward raw's 0.027 — R is a SOVER-connectivity detector here, NOT
          "propagation quality"; with a 2-edge GT chain, LOW R is correct)
U    ↑   (materially above Tucker r=8's 0.442)
C    ≈   unchanged (true edges kept)
S    =   1.0 (no sign change)
```

A note on R's interpretation (consultant correction, adopted): R measures **causal
connectivity capacity**, not propagation goodness. A rise in R under Tucker means
the graph densified (spurious paths). Form 1 should DRIVE R DOWN toward raw.

## Outcomes measured (fixed a priori)

Per support, relative to the Tucker r=8 baseline (U=0.442, |E|≈14.3, R≈0.233):

- **Primary:** `ΔU = U(masked) − U(Tucker_r8)`.
- **Secondary:** `Δ|E|`, `ΔR`.
- **Reported:** raw↔GT gap = `U(masked, support_GT) − U(masked, support_raw)`.

## Safeguard gate (fixed a priori — prevents a cheating prune)

A prune could raise U dishonestly by also removing TRUE edges (over-pruning that
happens to inflate the flow correlation). To rule that out, any U-improvement counts
ONLY if the prune preserves the true structure:

```
SAFEGUARD:  C ≥ 0.95  AND  S = 1.0   (on the masked corpus)
```

If U rises but the safeguard fails → the improvement is an artifact of destroying
real structure, NOT evidence for the structural thesis; reported as a FAIL of the
falsifier's validity, not a success.

## Verdict logic (fixed a priori)

| Primary (support_raw) | Safeguard | Verdict |
|---|---|---|
| ΔU materially > 0 (e.g. ≥ +0.15, ~30% of the 0.56 headroom) | C≥0.95 ∧ S=1.0 | **CONFIRMED** — Tucker's causal loss is principally spurious inflation; L3 causal conservation = preserving sparsity. Justifies a richer causal-pruning C_causal (separate prereg). |
| ΔU ≈ 0 (< +0.05) | (any) | **REFUTED** — structural inflation is a symptom, not the cause; C_causal must be rethought. Re-open design. |
| ΔU > 0 but safeguard FAILS | C<0.95 or S<1.0 | **INVALID** — prune improved U by destroying true edges; not evidence either way. |
| intermediate (0.05 ≤ ΔU < 0.15), safeguard OK | OK | **PARTIAL** — structure matters but is not the whole story; report, design a combined operator. |

The ΔU thresholds (+0.15 confirm / +0.05 refute) are fixed here a priori and not
adjusted post-hoc.

## No-method-shopping / honesty clauses

- ONE mask form (zero off-support flow), TWO pre-registered supports (raw decides,
  GT is the ceiling). We do not try alternative mask formulations until one "works."
- The safeguard is a real gate, not a footnote: a U gain that fails C≥0.95 ∧ S=1.0
  does NOT count as confirmation.
- Form 1 is a falsifier, not the operator. CONFIRMED justifies designing C_causal;
  it is not itself C_causal.

## Fixed parameters (summary)

- Tucker baseline: r=8 ambient (3,3,3,6) — the best-U config from the operator search.
- Mask: zero val_matrix flow off `support` before forming Φ; U/R/C/S on the pruned Φ.
- Supports: support_raw (primary, self-ref), support_GT (ceiling, oracle).
- Primary ΔU vs Tucker r=8; secondary Δ|E|, ΔR; reported raw↔GT gap.
- Safeguard: C ≥ 0.95 ∧ S = 1.0 on the masked corpus.
- Verdict thresholds: confirm ΔU ≥ +0.15, refute ΔU < +0.05 (a priori).
- Corpus S1-bis; PCMCI ParCorr tau=1 pc_alpha=0.01; same machinery as TCI/Q_L3.2A.
