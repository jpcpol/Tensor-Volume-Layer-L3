# TCI — Consultant Interconsult: Validated Instrument → Designing C_causal / Π_gov

**Focus requested:** the unsupervised causal metric U is now validated. The open
decisions are no longer "do we have a compass" but "how do we build the operator
under it." This document is self-contained — it does not require reading prior
commits, but cites them so any claim is auditable.

---

## 1. Where we are (one paragraph)

S3-bis refuted Property 1 for low-rank Tucker (raw causal F1 = 1.000,
reconstruction F1 = 0.135): Tucker preserves 98% of variance yet destroys the
causal structure governance needs. We reframed L3 around **causal-observability
preservation**, committing to the validation order **Causality ≻ Topology ≻
Reconstruction** and to the operator decomposition **C = Π_gov ∘ C_causal ∘
C_compress** (compress reduces dim; causal preserves structure; Π_gov projects
to the Governance Manifold). The bottleneck was never the operator — it was the
**instrument** that would judge any future operator without ground truth. The TCI
(Test de Calibración Instrumental) was run to validate a candidate metric U
*before* designing any operator on top of it. **U passed both gates.** L3 now has
a validated, ground-truth-free objective `U(C(T))`. The next step is the
operator — and that is what this interconsult is about.

---

## 2. The TCI procedure (what we did, and the one mistake we caught first)

### 2.1 Pre-run audit caught a broken metric (the important part)

The seeded candidate — "PCMCI persistence," `|E_raw ∩ E_recon| / |E_raw|` — was
**audited before any code ran** and found **broken**. It is a *recall*: it equals
**1.0** for the Tucker reconstruction while the supervised F1 is 0.135, because it
is blind to false positives. But Tucker's failure mode is precisely **fabricating
spurious edges** (graph G1: 2 true edges in `E_raw`, 28 in `E_recon` → 26
spurious), not losing true ones. A recall-style metric is structurally blind to
the actual failure. Had we run the TCI with it, M1 would have returned a **false
PASS** (recall = 1.0 for Tucker). The audit-before-run discipline — the same
discipline that earlier exposed pairwise Granger as a defective instrument for
Property 1 — prevented building the whole program on a defective compass.

### 2.2 Re-derivation of the correct invariant

The correct invariant is **causal-flow magnitude, not edge presence**. Tucker
*redistributes* causal flow into spurious channels while roughly preserving global
magnitude — so U must compare the *continuous flow pattern*, not a thresholded
edge set. We chose the PCMCI **`val_matrix`** as the flow source over transfer
entropy, for three reasons: (a) it is already computed in the S3-bis pipeline
(coherence — U and the supervised F1 share the exact estimator and differ only in
what they extract, continuous flow vs thresholded edges); (b) it conditions out
transitive links (partial correlation); (c) it is threshold-free and stable at 48
cycles/session. The scope caveat was pre-registered: val_matrix measures **linear,
direct** lagged flow — appropriate for S1-bis (linear by construction) and for the
S4 manifold (≈ linear), but blind to non-linear causality, which would need
TE/CMI under a *separate* future pre-registration.

### 2.3 U, fixed a priori

```
Φ[i,j] = median over sessions of  val_matrix[i, j, 1]      (11×11 causal-flow matrix)
U(Φ_ref, Φ_test) = Pearson correlation of the off-diagonal entries (i≠j)
```

Correlation (not distance) is the primary statistic **on purpose**: it penalizes
Tucker's *redistribution* of flow — it asks "does flow go through the same
channels with the same relative strengths?", which is the causal question. A raw
Frobenius distance would be partly fooled by overall-magnitude preservation.

### 2.4 Two pre-registered gates (BOTH required; neither alone validates U)

- **M1 — Causal monotonicity.** U must rank the Tucker family {1,2,3,5,8} as the
  *known* supervised F1 does. Gate: Spearman ρ(U, F1) = 1.0. (Honest limit: with
  5 points ρ=1.0 has ~0.8% chance under a random null — necessary, not strong
  alone. That is why M1 ∧ C2 is the validation, not M1.)
- **C2 — Causal perturbation (identifiability).** M1 alone cannot separate
  causality from compression (under compression they move together). C2 breaks the
  confound: take the **original, uncompressed** corpus and **shuffle each
  dimension's cycle order independently** (seed 20260609) — this destroys lag-1
  cross-dimensional causality, **preserves each marginal exactly**, and applies no
  compression. Gate: U(raw, shuffle) < 0.5. The marginal-preservation safeguard is
  load-bearing: if marginals changed, a U drop could be blamed on geometry, not
  destroyed causality.

A **no-method-shopping clause** fixed ONE U: if it had failed either gate, the TCI
reports an honest negative — no iterating formulas until one passes.

---

## 3. Results

**Verdict: U is a VALIDATED instrument. Both gates PASS** (prereg `1766b86`, run
`c8cf496`, results `results/tci_results.json`).

### M1 — passes perfectly

| Tucker rank | F1 (supervised, S3-bis) | U (val_matrix corr) |
|------------:|------------------------:|--------------------:|
| 1 | 0.0899 | 0.1952 |
| 2 | 0.1039 | 0.2399 |
| 3 | 0.1348 | 0.3575 |
| 5 | 0.1739 | 0.3916 |
| 8 | 0.2449 | 0.4415 |

Spearman ρ(U, F1) = **1.000** → **M1 PASS**. U orders the entire κ–F1 family in
the same order as the supervised metric, including the intermediate ranks (which a
mere "compression detector" would not be guaranteed to order correctly).

### C2 — passes decisively

| Comparison | U | Meaning |
|---|---:|---|
| U(raw, raw) | 1.000 | trivial sanity ceiling (re-estimation on identical input) |
| U(raw, shuffle) | **0.068** | causality destroyed at fixed marginals |

U(raw, shuffle) = 0.068 ≪ 0.5 → **C2 PASS.** Destroying causality while holding
marginals fixed collapses U to near zero — U responds to lagged causal flow, not
to marginal geometry or static correlation.

*(Implementation note: the M1 gate had a float-epsilon bug — `spearmanr` returns
0.9999999999999999 for a perfectly monotone 5-element sequence, so `>= 1.0`
evaluated False. Fixed with a 1e-9 tolerance. This is an implementation fix, not a
pre-registration change — the ordering IS perfect.)*

### The dynamic range U gives us (the number that matters for §4)

Placing all four reference points on U's scale:

```
U(raw, raw)          = 1.000   identity ceiling
U(raw, Tucker r=8)   = 0.441   BEST Tucker (most components)
U(raw, Tucker r=1)   = 0.195   most compressed
U(raw, shuffle)      = 0.068   causality destroyed (floor)
```

The best available operator (Tucker r=8) sits at only **~40% of the way from the
shuffle-floor to the raw-ceiling**. There is **0.559 of headroom** between the
best current operator and perfect causal conservation. **This is the gap the new
operator C_causal exists to close** — and now we can measure progress toward it
without ground truth.

---

## 4. What changes now (the methodological shift)

Before the TCI, L3 was "design C blind" — any candidate operator could only be
judged by reconstruction (which S3-bis proved is the *wrong* target) or by
ground-truth F1 (which **does not exist at deployment**, outside the lab). After
the TCI, L3 is **"optimize C under a validated causal objective"**:
`U(C(T))` is observable, requires no ground-truth graph, and provably tracks
causal fidelity (M1) for the right reason (C2). The problem changed type. This is
the most consequential methodological move since S3-bis: S3-bis showed the current
operator fails; the TCI establishes that the program has a **trustworthy compass**
for the next one. Without it, any future "improvement" might just be a new way to
optimize reconstruction.

---

## 5. The open decisions for C_causal / Π_gov (what we want the consultant's input on)

We deliberately did NOT design the operator yet — each piece gets its own
pre-registration. The questions below are where we want refinement *before* we
commit any prereg.

### Q1 — Form of C_causal: penalty, constraint, or structured factorization?

We have a validated differentiable-ish objective `U(C(T))`, but U routes through
PCMCI (a discovery step), so it is **not smoothly differentiable** in the operator
parameters. Three candidate forms:

- **(a) Penalized Tucker objective:** keep C_compress = Tucker but add a causal
  term — illustrative target `C* = argmax[ E_causal − μ·κ(V) ]  s.t. E_recon ≤ ε`,
  with `E_causal` a surrogate of U. *Risk:* if E_causal is only U-via-PCMCI, the
  inner loop is expensive and non-smooth. *Question: is there a cheap differentiable
  proxy for U (e.g. a direct val_matrix-correlation term computed on a linear
  surrogate) we can put in the objective and validate against full U post-hoc?*
- **(b) Causal-support-constrained core:** factor Tucker but constrain the core G
  to the causal support recovered by PCMCI on the raw corpus (zero out off-support
  channels). *Risk:* this presumes the raw causal support is itself trustworthy and
  could overfit to the linear-direct regime U measures.
- **(c) Separate decomposition entirely:** abandon Tucker as C_compress; use a
  graph-tensor / causal-structured factorization where causal channels are
  first-class. *Risk:* loses the κ(V) tractability result (Property 4) that Tucker
  gave us.

**We lean toward (a)** because it preserves the validated κ(V) tractability and
treats causality as an additive correction to a known-good compressor — but we
want the consultant's read on whether the non-smoothness of U-via-PCMCI makes (b)
or a hybrid more honest. **What is the right surrogate to put *in* the objective,
given that full U is the validator, not necessarily the trainer?**

### Q2 — How does Π_gov relate to the already-confirmed Governance Manifold?

S4 confirmed the Governance Manifold at **dim ≈ 2–3** (n=90, twin-width ≥ 0.96).
The paper's seed reading is literal: *"the ideal C is Π_{M_gov}, and Tucker is
only a blind approximation."* Open question: **is Π_gov a learned projection onto
the empirically-recovered M_gov, or a fixed analytic projection?** And the seam we
flagged at reframing time: **topology must be subordinate to causality, not a hard
co-constraint** — a hard `tw(M_T, M_V) ≥ 0.95` risks re-rewarding the geometric
preservation that S3-bis showed is *insufficient*. *Question: how do we encode
"topology subordinate to causality" operationally — soft penalty, lexicographic
objective, or a staged optimization (causal first, then topological tie-break)?*

### Q3 — Formal composition C = Π_gov ∘ C_causal ∘ C_compress

We have the three pieces as a sketch. Open: **does the composition order commute
with the validation order (Causality ≻ Topology ≻ Reconstruction)?** Intuitively
C_compress runs first (cheapest, reconstruction-level) and Π_gov last (highest,
governance-level), but C_causal sitting in the middle must *correct* what
C_compress destroyed — which means C_compress should not be optimized to its own
reconstruction optimum in isolation (that is exactly the S3-bis trap). *Question:
should the three be trained jointly under a single causal-first objective, or
staged with C_causal allowed to override C_compress's reconstruction choices?*

### Q4 — The deployment surrogate and U's linear scope

U-via-val_matrix measures **linear-direct** causal flow. It is the validated
*training/selection* objective on S1-bis. Two deployment concerns:

- **Non-linearity:** real pipelines may carry non-linear causality U is blind to.
  We pre-registered that TE/CMI is the future extension — but *when* does that
  become load-bearing? Is a linear U acceptable for the **first** C_causal (whose
  job is just to beat Tucker's 0.441), deferring non-linearity to a v2 operator?
- **Cross-scale persistence (the L4 destination):** the memory notes flag "causal
  stability across scales" `sim(G_L2, G_L3)` as the metric that best fits CAL /
  Governance Manifold — but as the *destination* (L4), not the start, because it
  drags the open assumption that G_L2 and G_L3 are comparable. *Question: should
  C_causal be designed now with the cross-scale objective in mind (so it doesn't
  need redesign at L4), or is it correct to optimize the within-scale U first and
  treat cross-scale as a separate later prereg?*

---

## 6. What we are NOT asking (settled)

- **Whether U is a valid instrument** — the TCI settled this (M1 ∧ C2 PASS).
- **Whether to save Tucker by adding components/variance** — no. S3-bis already
  proved more variance ≠ more causality; the U=0.441 ceiling for Tucker r=8 (most
  components) confirms it on the new instrument.
- **The validation order** — Causality ≻ Topology ≻ Reconstruction is a committed
  framework constraint, not up for re-litigation.
- **S4 (manifold dim ≈ 2–3) and S2 (κ(V) sub-linear, Property 4)** — independent
  of all this, stand on their own.

---

## 7. Documents to read (in order)

1. `PRE_REGISTRATION_TCI.md` — the instrument under validation (commit 1766b86)
2. `run_tci.py` — implements it verbatim (commit c8cf496)
3. `results/tci_results.json` — the numbers in §3
4. `../../CONSULTANT_REVIEW_L3.md` — the foundational reframing (the "why" of §1)
5. `NEGATIVE_RESULTS.md` / `PRE_REGISTRATION.md` — S3 + S3-bis context (the F1 scale)

---

## 8. The single decision we most want refined

Of §5, **Q1 (form of C_causal) is the load-bearing one** — it determines the next
pre-registration. Our default is the penalized-Tucker form (a), choosing a cheap
differentiable causal surrogate for the *training* objective and validating it
against full U-via-PCMCI post-hoc. We want the consultant to either confirm that
the train-surrogate / validate-with-U split is methodologically sound, or tell us
it reintroduces the S3-bis trap (optimizing a proxy that diverges from the thing
we actually validated). Everything else in §5 can follow that decision.
