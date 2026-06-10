# L3 Closure — Causal Conservation under Compression

**Juan Pablo Chancay** · Aural Syncro Research Lab · jpcpol@gmail.com
**Date:** June 2026 · **Status:** L3 characterization phase complete (~92–95%).
**License:** CC BY-NC 4.0 (this document) · AGPL-3.0 (src/)

This document consolidates the L3 research line into a single methodological
narrative, formalizes the operator property the experiments converged on, and
states the residual frontier. It is the closure synthesis (Bloque D) prescribed by
the methodological consultant; it does **not** introduce new experiments. Every
claim is traceable to a pre-registered run committed in
`experiments/causal_conservation/`.

---

## 0. Executive summary

L3 began as a compression problem — how to aggregate a sequence of cognitive
tensors `{T⁽ˢ⁾}` into a volume `V = C({T⁽ˢ⁾})` — and ended as a **causal-
observability** problem. The decisive results:

1. **Reconstruction ≠ causality.** Low-rank Tucker preserves 98% of variance yet
   destroys the causal structure governance needs (S3-bis: raw F1=1.000,
   reconstruction F1=0.135).
2. **A ground-truth-free causal metric exists.** U (PCMCI val-matrix flow
   correlation) orders causal fidelity (TCI: M1 ρ=1.0, C2 collapse 1.0→0.068) and
   is usable as a direct optimization objective.
3. **Causality is structural, not magnitude.** No differentiable magnitude proxy
   reproduces U's ordering (proxy audit); U's power comes from PCMCI's discrete
   parent-selection.
4. **Tucker's failure mode is fabrication, not loss.** Tucker keeps coverage and
   sign-consistency but inflates the graph 7–14× (Q_L3.2A).
5. **Pruning recovers 75% of the gap, with no ground truth.** A raw-support
   structural mask lifts U 0.441→0.862, equal to the GT-oracle mask (Form 1).

**The L3 thesis, now empirically established:** *causal conservation under
compression is principally the preservation of structural sparsity* — keeping the
true causal support and suppressing fabricated edges, not maximizing reconstruction
fidelity.

---

## C1. Formal definition: a causally-conservative operator

L3's experiments converge on an operational property, stated over the observational
invariants Ω₀.

### C1.1 The observational invariants Ω₀

Given a PCMCI causal graph `G = (V, E, W)` recovered from a corpus (ParCorr, lag-1,
pc_alpha=0.01, majority vote ≥0.5), define Ω₀ = (R, C, S):

- **R — Reachability** (connectivity capacity): `R = #{(i,j): i⇝j} / (n(n−1))`,
  the transitive-closure ordered-pair fraction. R is an **over-connectivity
  detector**: for a sparse true graph, low R is correct; a rise signals spurious
  densification.
- **C — Coverage:** `C = |E ∩ E_ref| / |E_ref|`, the fraction of reference edges
  retained. `E_ref` = ground truth (training) or raw (deployment surrogate).
  Omission = 1 − C.
- **S — Consistency:** `S = 1 − (#sign-violations / |E|)`, the fraction of recovered
  edges whose causal flow sign matches the reference. Structural violation = 1 − S.

Ω₀ are observational primitives; the cognitive categories Ω₁ = {drift, conflict,
propagation, omission, structural violation} are **derived** from Ω₀, never defined
directly (a semantic definition would inject theory into the instrument).

### C1.2 The property

> **Definition (causal conservation).** A composition operator `C` is *causally
> conservative under compression* if, for the compressed volume `V = C(T)`, the
> observational invariants Ω₀(V) match Ω₀(T) within tolerance:
> `R(V) ≈ R(T)`, `C(V) ≥ 1 − ε_C`, `S(V) = 1`,
> while achieving a compression ratio κ(V) < |T|.

Equivalently, using the validated metric: `C` is causally conservative iff it keeps
`U(C(T))` near the raw ceiling **for the right reason** — by preserving the causal
support (R, C, S), not by preserving variance.

### C1.3 Why this is the right property (empirical anchor)

S3-bis showed variance preservation (Property 4 / 98% variance) is *insufficient*:
an operator can be variance-optimal and causally destructive. Q_L3.2A + Form 1 show
the **separable, sufficient** property: Tucker already holds C and S (it recovers
true edges with correct signs) and fails **only** on R (it fabricates edges).
Restoring R to the raw value — a pure structural prune — recovers 75% of the causal
gap while holding C=S=1. So Ω₀-preservation, not reconstruction, is the operative
criterion. This replaces the original four properties' implicit reconstruction goal
with an explicit causal-structure goal, in the validation order
**Causality ≻ Topology ≻ Reconstruction.**

---

## C2. The methodological narrative (one sequence)

The knowledge is distributed across many committed experiments; here it is one arc.

### S2 — Tucker as C_compress (tractability holds)
Tucker-HOOI compresses `{T⁽ˢ⁾}` with κ(V) sub-linear in n_sessions: Property 4
(tractability) and Property 3 (dimensional stability) hold, 98% variance retained.
*Result: Tucker is a valid compressor.* (`results/s2_tucker_results.json`)

### S3 runs 1–2 — the instrument was the bottleneck, not C
Pairwise Granger (run 1) flagged transitive false positives; PCMCI per-session +
majority vote (run 2) hit a recall ceiling (raw F1=0.667) from too few per-session
time points. A pre-registered attribution rule (*if the raw control can't clear
0.70, blame the method, stop*) prevented mistaking a weak instrument for a
refutation. *Lesson: validate the instrument before trusting a negative.*

### S3-bis — Property 1 cleanly REFUTED
S1-bis (t=48, recovering raw F1=1.000) gave a perfect control. Against it, low-rank
Tucker scored causal F1=0.135 — inventing spurious edges. *Result:
**reconstruction ≠ causality**; the refutation is attributable to C, not the
method. Topology ≠ causality too — the hierarchy reordered.*
(`NEGATIVE_RESULTS.md`)

### TCI — a ground-truth-free causal instrument U
A pre-run audit caught the seeded metric (edge-recall) as broken (1.0 for Tucker, a
false PASS). Re-derived U = Pearson correlation of off-diagonal PCMCI val-matrix
flow. Two gates: M1 (orders the Tucker family as supervised F1 does, ρ=1.0) and C2
(collapses 1.0→0.068 under temporal shuffle at fixed marginals). *Result: U is a
validated unsupervised causal instrument.* (`results/tci_results.json`)

### Proxy audit — causality is structural, not magnitude
To optimize U, a differentiable surrogate was sought. Three magnitude-based proxies
(full-conditioning, bivariate, ridge-VAR) **failed** to reproduce U's ordering
(ρ ≤ 0.6 vs 1.0). *Result: U's discriminating power lives in PCMCI's **discrete**
parent-selection — causality is in the graph structure, not coefficient magnitudes.
Optimize U directly via derivative-free search; no proxy.*
(`results/audit_proxy_results.json`)

### Operator search — U is a usable objective
Exhaustive 45-config Tucker grid maximizing U directly. The U-optimum coincided with
the calibration baseline (Tucker has no "cheat" config), and the high-U region sat
entirely at r_dim ≥ the governance-manifold dimension — U never rewards
sub-manifold compression. *Result: U works as a direct objective; the loop is
validated infrastructure.* (`results/operator_search_results.json`)

### Π_gov — suspended (Option C)
The S4 governance manifold is **static** (reconstructible with the time axis
averaged out, trustworthiness 0.96 even at t=48 — confirmed by audit on both S1 and
S1-bis), while causality is **temporal**. A topological tie-break toward the static
manifold would re-enter the S3-bis trap via the tolerance band. *Decision: suspend
Π_gov (not delete); M_gov becomes a descriptor, not a driver; the operator reduces
to `C = C_causal ∘ C_compress`. Whether a causal manifold exists (Q_L3.2) is an
L4 question.* (`CONSULTANT_BRIEF_PIGOV.md` §9, `audit_s1bis_manifold.json`)

### Ω — the object to preserve, as observational invariants
With Π_gov out, C_causal is the only new piece. Its target was defined as Ω₀ =
(P, D, R, C, S) — observational invariants, not semantic categories. An audit
fixed the scope: R, C, S are high-power on S1-bis; D (divergence) needs per-agent
series and injected conflict; P (persistence) needs longer windows — both deferred
to an S-Ω corpus. *Result: Ω operationalized; Q_L3.2A scoped to R, C, S.*
(`CONSULTANT_BRIEF_OMEGA.md` §9)

### Q_L3.2A — Tucker's failure DECOMPOSED
Across {raw, r=1,2,3,5,8}: **S = 1.0 everywhere** (no sign corruption),
**C high** (true edges recovered, no omission), **R and |E| explode** (raw 2 edges
→ Tucker 14–28). *Result: Tucker does not lose or corrupt causality — it
**over-generates** it. C_causal must be a **pruning** operator, not a recovery one.*
(`results/ql32a_results.json`)

### Form 1 — the structural thesis CONFIRMED
A raw-support structural mask (zero flow off the raw causal support) was the
falsifier. Prediction met exactly: U 0.441→**0.862**, |E| 14.3→2.0, R 0.233→0.027,
with the safeguard held (C=1.0, S=1.0 — no true edge destroyed). The deployment
mask (raw support, no GT) **equalled** the oracle (GT support): raw↔GT gap = 0.000.
*Result: Tucker's causal loss is **principally** structural inflation; pruning
recovers 75% of the headroom **without ground truth**.*
(`results/form1_results.json`)

---

## C3. The residual frontier (declared, not hidden)

Form 1 reaches U = 0.862, not 1.000. The residual is explicit:

```
ΔU_residual ≈ 1.000 − 0.862 = 0.138   (≈25% of the raw→Tucker headroom)
```

This is a **frontier, not a defect.** It bounds what *pure binary structural
pruning* can achieve. The residual must live in something the lag-1 binary causal
support cannot capture:

- **Flow magnitude on true edges** — the prune keeps the true edges but does not
  re-weight their strengths to the raw values.
- **Higher-order / non-linear structure** — U-via-val_matrix measures linear-direct
  flow only (a pre-registered scope caveat); non-linear causal content would need
  TE/CMI and an S-Ω corpus.

A future C_causal can attack the residual with **graded** (not binary) pruning —
weighting retained edges toward raw flow magnitude — but that is a new operator
generation (L4 territory), not a requirement for the L3 thesis.

---

## D. What L3 closes, and what it explicitly does not

### Closed (L3 thesis demonstrated)
- Reconstruction and topology are insufficient criteria for causal compression.
- A ground-truth-free causal metric (U) exists, is validated, and is a usable
  objective.
- Causality is carried by discrete structure, not continuous magnitudes.
- Tucker's failure is spurious-edge fabrication; structural pruning recovers 75% of
  the causal gap with no ground truth at deployment.
- The operative property of a causally-conservative operator is **Ω₀-preservation**.

### Deliberately NOT in L3 (L4 / future)
- **Q_L3.2B** — drift, conflict, temporal persistence (D, P): needs an S-Ω corpus
  with injected inter-agent conflict and temporal drift.
- **Causal manifold** — whether a low-dimensional M_causal exists (vs the static
  M_gov^static): open, does not block L3.
- **Advanced C_causal** — graded/weighted pruning, structural corrector,
  graph-aware Tucker: a new operator generation, not required for the thesis.
- **L4 governance claim** — that causal observability improves *human* governance:
  belongs to the RCT (Paper 2).

### Defensible framing (conservative)
> L3 produced evidence that CAL can function as a **causal-observability layer** for
> human governance of multi-agent systems: a representation engineered to preserve
> what a supervisor needs to retain attribution and correction capacity over
> autonomous agents. The stronger claim — that CAL *is* an AI-governance theory —
> awaits the RCT.

---

## E. Provenance (pre-registration → run, every claim)

| Result | Pre-registration | Run / artifact |
|--------|------------------|----------------|
| Tucker tractable (S2) | paper §6.6 | `results/s2_tucker_results.json` |
| Property 1 refuted (S3-bis) | `PRE_REGISTRATION.md` + S1-bis prereg | `results/s3bis_results.json`, `NEGATIVE_RESULTS.md` |
| U validated (TCI) | `PRE_REGISTRATION_TCI.md` (1766b86) | `results/tci_results.json` (c8cf496) |
| Proxy refuted | (audit, pre-prereg) | `results/audit_proxy_results.json` (abd7871) |
| U usable objective | `PRE_REGISTRATION_OPERATOR_SEARCH.md` (5bb1b2d) | `results/operator_search_results.json` (1fc0bb8) |
| Π_gov suspended | `CONSULTANT_BRIEF_PIGOV.md` §9 | `audit_s1bis_manifold.json` (bf1c589) |
| Ω operationalized | `CONSULTANT_BRIEF_OMEGA.md` §9 | — |
| Tucker fabricates (Q_L3.2A) | `PRE_REGISTRATION_QL32A.md` (f61472f) | `results/ql32a_results.json` (e743a52) |
| Structural thesis confirmed (Form 1) | `PRE_REGISTRATION_FORM1.md` (b2c843a) | `results/form1_results.json` (35d4cb0) |

The full interconsultation record (consultant rounds, design decisions, audits) is
`experiments/causal_conservation/TCI_CONSULTANT_INTERCONSULT.md` §§9–14.
