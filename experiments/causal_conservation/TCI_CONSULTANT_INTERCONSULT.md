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

---

## 9. Reconciliation with the consultant's TCI response (state sync)

The consultant reviewed an **earlier** state (the seeded plan, before the
pre-run audit) and proposed a TCI design. This section maps each point onto what
was **actually executed**, so the next round starts from the current state, not
the prior one. Net: the consultant's framing is right; two of his specifics were
already superseded by stricter executed versions, and two of his signals are
genuinely useful.

### 9.1 Confirmed — already executed, stronger than proposed

- **TCI-1 (causal monotonicity** `F(Tᵢ)>F(Tⱼ) ⇒ U(Tᵢ)>U(Tⱼ)`**)** is exactly our
  gate **M1**, run over the full family {1,2,3,5,8} (not just A>B>C): Spearman
  ρ = 1.0. The consultant's "naturally calibrated benchmark" (A=1.000, B=0.245,
  C=0.135) is the κ–F1 scale we used — and we used **all five** points, so the
  intermediate-rank ordering (which a compression detector could fake on three
  points) is also pinned. ✅ Done.
- **TCI-2 (minimum separation** `U(T) − U(T_κl) > ε`**)** is **weaker** than what
  we ran. Our **C2** does not merely require a gap on two points; it destroys
  causality at **fixed marginals** (per-dim temporal shuffle) and shows U collapses
  1.0 → 0.068. That proves the separation is caused by *causality, not
  compression* — which a two-point ε threshold cannot establish. C2 ⊃ TCI-2. ✅ Done.
- **"The TCI is the real scientific gate of L3"** — agreed and already acted on.
  The TCI was run **before** any new operator, exactly as a gate.

### 9.2 Superseded — do NOT adopt

- **U₁ = `|E_T ∩ E_V| / |E_T|` (PCMCI edge preservation)** is **the metric we
  audited and rejected a priori.** It is a *recall*: it returns **1.0** for Tucker
  while supervised F1 = 0.135, because it is blind to false positives — and Tucker's
  failure mode is *fabricating* spurious edges (G1: 2 true edges vs 28 in the
  reconstruction → 26 spurious), not losing true ones. Running the TCI with U₁
  would have produced a **false PASS on M1**. This is not a matter of preference;
  it is a verified property of the metric on this corpus. We replaced it with the
  **Pearson correlation of off-diagonal `val_matrix` flow** (continuous flow, which
  penalizes Tucker's redistribution). The consultant's recommendation predates the
  audit that ruled U₁ out.

### 9.3 Useful new signals — adopted / catalogued

- **U₂ = Causal Manifold Stability** (PCMCI structure on T vs V, measuring causal
  *neighborhood geometry*, aligned with the Governance Manifold). The consultant is
  right that **M_gov is the real scientific object, not Tucker.** This matches our
  catalogued "cross-scale" candidate — flagged as the **L4 destination, not the
  starting point**, because it drags the still-open assumption that G_L2 and G_L3
  are comparable. It is **Q4** in §5. It confirms the destination; it does not
  change the immediate next step (within-scale U first).
- **Reordering TCI → S4 → operators** (instead of S4 → operator → TCI). **Adopted
  as a dependency principle.** His argument is sound — if U were invalid, S4 would
  lose value and every operator comparison would be in question. De facto we
  *already* satisfy this (the TCI ran first); we are now formalizing it: a validated
  U is a **precondition** for S4's interpretability and for any operator comparison.

### 9.4 Framing sentence (adopted verbatim into the L3 program statement)

> The immediate objective of L3 is not to find a composition operator C, but to
> establish an unsupervised metric U capable of discriminating between known
> levels of causal fidelity. Without a validated U, Property 1 (Causal
> Preservation) of operator C is not evaluable outside synthetic corpora with
> ground truth.

This sentence connects S3-bis, the TCI, README Property 1, and the Governance
Manifold hypothesis, and is now part of how L3 is stated. (As of this round, U
**is** validated — so the precondition the sentence names is met, and the program
may proceed to the operator under §5.)

---

## 10. Consultant resolution of §5 (second round — read the full report)

The consultant reviewed this full report and resolved the load-bearing open
questions. Q1, Q2, and the next-prereg scope are now **decided**; Q4 stands as
catalogued (L4). Recorded here as the design the next pre-registration commits to.

### 10.1 Q1 — form of C_causal: **(a) Penalized Tucker** (decided)

Confirmed, with the decisive argument: *we have evidence that **pure** Tucker is
insufficient, not that Tucker is — those are different.* Tucker's computational
behavior, κ(V), stability, and scalability are known and validated (S2); there is
no warrant to discard it, only to correct it. (b) core-restricted-by-causal-support
is **rejected** (overfits the linear-PCMCI regime we ourselves flagged as limited);
(c) a new causal factorization is **premature** (too many simultaneous unknowns).

### 10.2 The train/validate split + the differentiable proxy (the key unlock)

Confirmed: **do not put full U inside the optimizer.** U routes
operator → PCMCI → val_matrix → corr, which is costly, non-differentiable, and
risks optimizing estimator noise. Instead:

```
Training objective:   E_flow = 1 − corr(Φ_T, Φ_V)     (continuous flow, no thresholds)
Validation objective: U  (full PCMCI val_matrix correlation)   ← external check
```

The proxy is drawn from U's own machinery (the flow matrix Φ), so proxy and
validator are aligned by construction — the standard "train on a proxy, validate
on the real metric" discipline.

> **Implementation caveat we add (makes the proxy executable, does not weaken it):**
> in U, `Φ = val_matrix` is the PCMCI *partial* correlation conditioned on the
> parents **PCMCI selected** — that selection is a discrete discovery step, so a
> Φ built that way is **not** smoothly differentiable. To get a genuinely cheap,
> differentiable `E_flow`, the proxy's Φ must use a **fixed-conditioning** partial
> correlation (e.g. condition on all other 9 dimensions, no PC selection step). The
> next prereg must (i) fix that conditioning a priori, and (ii) run a mini-M1: check
> the fixed-conditioning proxy Φ still orders the Tucker family as full-U-via-PCMCI
> does. If it does not, the proxy diverges from the validated metric and must be
> revised before training on it (this is exactly the S3-bis-trap guard, applied to
> the proxy rather than the operator).

### 10.3 Q2 — Π_gov: **lexicographic tie-break, not a constraint** (decided)

Π_gov is NOT a hard constraint (a hard `tw ≥ 0.95` would re-reward the geometric
preservation S3-bis showed is insufficient). It is a **lexicographic objective**:

```
Level 1:  max U            (causal preservation — primary)
Level 2:  max tw           among solutions already causally equivalent (tie-break)
```

This implements **Causality ≻ Topology ≻ Reconstruction** operationally without
re-entering the S3-bis trap. Closes Q2.

### 10.4 The deeper scientific claim the report under-emphasized

The consultant's strongest point: **the TCI does not only validate U — it
establishes that causality can be detected without the ground-truth causal
graph.** That was not demonstrated before; now there is evidence it can. This is a
standalone, publishable claim (independent of any operator), and is the reason the
line is worth a paper. It should be foregrounded, not buried under "U passed."

### 10.5 Next pre-registration scope (decided)

NOT yet "new L3 operator." Scoped to answer Q1 only, reusing S3-bis + TCI:

> **"Causal-Aware Tucker: evaluation of differentiable causal-flow proxies as a
> training objective, with external validation via U."**

This answers Q1 without committing the full `C = Π_gov ∘ C_causal ∘ C_compress`
architecture, and reuses everything already validated. Q3 (composition order) and
Q4 (non-linearity / cross-scale, L4) remain for later, separately pre-registered
steps.

---

## 11. Pre-prereg audit: the differentiable proxy is REFUTED (third round)

Before writing the §10.5 prereg, we **audited** the load-bearing assumption — that
a cheap differentiable proxy `E_flow` could order the Tucker family as the
validated U does. The audit (`audit_proxy.py`, record `results/audit_proxy_results.json`)
ran three differentiable Φ candidates against the S3-bis F1 scale. **It refuted
the assumption** — and the consultant, shown the result, agreed and reframed it as
a finding, not a defect.

### 11.1 What the audit found

| Flow matrix Φ | Differentiable? | Cost/session | ρ(·, F1) |
|---|---|---|---|
| U-via-PCMCI (validated metric) | **No** (discrete PC parent-selection step) | ~190–540 ms | **+1.00** |
| Full-conditioning partial corr (precision matrix) | Yes | ~2.5 ms (≈80× faster) | **−0.30** |
| Bivariate lag-1 cross-corr | Yes | cheap | **+0.60** |
| Ridge VAR(1) coefficients | Yes | cheap | **+0.60** |

Every differentiable candidate fails the mini-M1 ordering gate (ρ=1.0). The two
best (bivariate, ridge) invert at **rank 5 < rank 3** while F1 rises monotonically.
The C2-style shuffle sanity passes for all (proxy collapses to ~0.02 under
shuffle) — so the proxies *do* respond to temporal causality, they just don't
**rank fidelity** correctly. Cost is not the problem; **ordering** is.

### 11.2 The finding (consultant, adopted)

The diagnosis: **U's ordering power depends on PCMCI's discrete structural
selection step (PC)** — which is exactly what makes U non-differentiable. Removing
it to gain a gradient destroys the very property we want to preserve. This is the
**second instance of the S3-bis pattern**:

```
S3-bis  :  Reconstruction        ≠  Causality
this audit: Differentiable proxy ≠  Causal ordering
```

Chasing a "more convenient" continuous metric that then becomes the objective is
the Tucker mistake again. We do NOT repeat it.

> **Deeper claim (consultant, to be tested):** the failure of coefficient-based
> proxies is **not a negative result for CAL** — it is evidence that the causal
> information relevant to governance lives in the **structure** of relations, not
> in the **continuous magnitudes** attached to them. It explains why Tucker
> preserves reconstruction and local topology yet destroys causality, and why
> parent-selection works where coefficient magnitudes fail. The fundamental causal
> question is `X → Y?` (discrete), not "what is the coefficient value?"
> (continuous). If this holds in further experiments, Property 1 will end up
> defined by **discrete causal-structure preservation** (graph similarity), not by
> a continuous energy — and L3 stops being "compression-operator design" and
> starts **characterizing which part of a representation carries causal
> governability.**

### 11.3 Design decision (revised — supersedes §10.2 and §10.5)

**Drop the differentiability requirement.** There is no evidence it is necessary
(it came from a deep-learning reflex, not from CAL/TCO/L3), and there is now
evidence that trying to satisfy it degrades the target property. L3 is an
**operator-search** problem, not a differentiable-optimization one:

```
REVISED objective:   max  U(C(T))        ← directly, the validated metric
REVISED search:      black-box / derivative-free over the operator's few
                     parameters (Tucker ranks): Bayesian optimization, CMA-ES,
                     differential evolution, or exhaustive small-grid.
```

This is standard and accepted (AutoML, NAS, hyperparameter search all optimize
non-differentiable external metrics). U stays the **primary objective**, not a
proxy. Π_gov remains the lexicographic tie-break of §10.3.

### 11.4 Revised next-prereg scope (supersedes §10.5)

> **"Causal-Aware operator search: maximize the validated metric U(C(T)) directly
> via derivative-free search over Tucker ranks, with Π_gov as lexicographic
> topological tie-break."**

No proxy. The expensive-but-correct U is affordable because the search space (a
few Tucker ranks) is tiny — the cost concern that motivated the proxy does not
bind when there are O(10) operator configurations, not O(10⁶) gradient steps.

### 11.5 Research-notebook entry (consultant, verbatim)

> Audit result: U's ability to order causal fidelity appears to depend on PCMCI's
> discrete structural-selection step (PC). Three continuous proxies fail to fully
> reproduce that ordering. Absent contrary evidence, the program temporarily
> abandons the hypothesis that a simple differentiable proxy for U exists and
> adopts U as the primary evaluation objective for future causal-composition
> operators.
>
> The failure of the proxies is not a negative result for CAL. It is evidence that
> the causal information relevant to governance is contained chiefly in the
> structure of relations and not in the continuous magnitudes associated with
> them.

---

## 12. Operator search executed — U is a usable objective (G1 PASS, with caveat)

Pre-registered `PRE_REGISTRATION_OPERATOR_SEARCH.md` (commit 5bb1b2d), run
`run_operator_search.py`, results `results/operator_search_results.json`.
Exhaustive 45-config Tucker grid maximizing the validated U directly (no proxy).

### 12.1 Result

```
C* = argmax U  =  (r0=8, r_dim=3, r_stage=3, r_agent=3, r_cycle=6)   U* = 0.4415
G1 (anti-overfit):  F1(C*)=0.245  ≥  F1(C_base)=0.245   PASS
G2 (reported):      Spearman(U, F1) over calibration line = 1.000   positive
```

**Honest caveat on G1.** The argmax-U config **coincides with the TCI calibration
baseline** `(8,3,3,3,6)`, so C* = C_base and G1 passes *trivially* (0.245 = 0.245).
G1 was therefore not a strong test this run — there was no displacement away from
the baseline to stress-test. The real evidence that U does not overfit is
structural, below.

### 12.2 The structural evidence (stronger than the trivial G1)

The full grid shows U and the **known** governance structure coincide
*geographically* in rank space — U never rewards a causally-degenerate config:

- **Top-8 U configs all lie at r_dim ≥ 3** — at or above the S4 governance-manifold
  dimension (≈2–3). U's preferred region is exactly the region S4 marked as where
  causal/governance structure lives.
- **r_dim = 2 (below the manifold dim) is confined to U ≤ 0.217** — compressing the
  dimension mode *below* the causal manifold is consistently penalized by U. A
  metric that overfit (rewarded compression per se) could have placed a high-U
  config here; none did.
- Along the calibration line (r_dim=3, r_cycle=6) U rises monotonically with r0
  (0.195→0.240→0.358→0.392→0.442), tracking the S3-bis F1 order (G2 ρ=1.0).

So U does not just *rank* the calibration family (TCI) — across an **independent**
grid it concentrates its optimum in the governance-manifold region and refuses to
reward sub-manifold compression. That is the anti-overfit signal, independent of
the trivial G1.

### 12.3 What this does and does NOT establish

- **Does:** U is usable as a **direct optimization objective**, not only as a
  referee. Maximizing it over an independent rank grid lands in the causally-correct
  region and never in a degenerate one. The derivative-free search loop works.
- **Does NOT:** prove U is *exploit-proof* under a *richer* operator family. Within
  plain Tucker, the U-optimum simply confirms the calibration baseline — Tucker has
  no "cheat" config to find. The genuine overfit stress-test arrives when the
  operator family is enlarged (structured cores, Π_gov), where U *could* be gamed.
  G1 must be re-run there with a non-trivial baseline. Flag carried forward.

### 12.4 Decision and next step

U is confirmed as the L3 objective. The plain-Tucker search has no operator beyond
the calibration optimum to offer (expected: S3-bis already showed Tucker's ceiling
is low — U*≈0.44 vs raw 1.0, the 0.56 headroom is **not reachable inside Tucker**).
This makes the next prereg's necessity concrete: **to close the headroom, the
operator family must change** — and that is exactly where Π_gov (lexicographic
topological tie-break, §10.3) and a structure-preserving C_causal enter. The U
objective and the derivative-free search loop are now validated infrastructure for
that step; G1 must use a non-trivial baseline once the family is enlarged.
