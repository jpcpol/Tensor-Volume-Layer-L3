# CAL-L3 — Foundational Reframing after S3-bis (Consultant Review)

**Date:** 2026-06-08
**Status:** research-program principle (not an implementation commitment)
**Evidence base:** S3-bis (`experiments/causal_conservation/`, commit cdd64e8):
raw causal F1 = 1.000, Tucker-reconstruction causal F1 = 0.135.

---

## The decision (epistemological, not technical)

S3-bis settled a question that had two compatible hypotheses:

- **Hypothesis A:** causality is lost because the corpus is insufficient.
- **Hypothesis B:** causality is lost because the compression operator is not
  optimized to preserve it.

The result `raw = 1.000, recon = 0.135` is very hard to reconcile with A (the raw
control recovers the full causal graph perfectly) and shifts the weight of
evidence decisively to **B**.

Therefore L3 is **redefined**:

> **L3 is not a compression problem. L3 is a problem of causal-observability
> preservation under compression.**

The objective is no longer `min ||T − T̂||` (reconstruction / explained variance).
The framework now commits to a **validation ordering**:

> **Causality ≻ Topology ≻ Reconstruction**

An L3 operator must demonstrate **causal conservation before** geometric/topological
conservation before reconstruction fidelity — never the reverse, and never
causality and geometry at the same level (S3-bis is a concrete case where they
conflict).

## Founding constraint (the negative result, promoted)

> S3-bis showed that **geometric preservation is insufficient as a validation
> criterion for cognitive compression operators.** An operator can preserve
> almost all observable variance (98%, S2) while simultaneously destroying the
> causal structure required for governance (causal F1 = 0.135, S3-bis).

This converts a negative result into a **foundational restriction** of the
framework, compatible with TCO, SID, the Governance Manifold, and causal
observability.

## The conceptual error this fixes

The current operator implicitly maximizes **I(V, V̂)** — how much the
reconstruction resembles the original tensor. But Property 1 (causal preservation)
requires **I(Causal(T), Causal(V))** — how much the *causal structure* survives.
These are different objects. Optimizing the first does not optimize the second;
S3-bis is the demonstration.

## Operator reframing

Tucker is accepted as a candidate for **C_compress**, but **not** assumed to be a
candidate for **C** itself. The target operator likely decomposes as:

```
C = Π_gov ∘ C_causal ∘ C_compress
```

- `C_compress` — dimensionality reduction (Tucker is a candidate here).
- `C_causal` — preserves causal structure.
- `Π_gov` — projection onto the Governance Manifold M_gov.

The paper already contains the seed: *"If confirmed, C = projection onto M_gov."*
S3-bis supports reading that literally — the ideal C is **Π_{M_gov}**, the
projection onto the manifold of governable states, of which Tucker is only an
approximate (and causality-blind) implementation.

## Two seams to resolve before implementing (not yet solved)

These refine the consultant's proposal; both must be handled when L3 resumes.

### Seam 1 — Topology must not compete with causality

A hard constraint like `tw(M_T, M_V) ≥ 0.95` (manifold trustworthiness) risks
re-rewarding exactly what failed: geometric preservation. S3-bis proved an
operator can preserve geometry and destroy causality. So topology (Level 2) must
be **subordinate** to causality (Level 1), not a co-equal hard constraint.
Causality dominates.

### Seam 2 — F1_causal cannot be in the *deployment* objective

`F1_causal` (and SHD, SID against ground truth) require a known causal graph.
That exists in the synthetic corpus but **disappears outside the lab**. So:

- **Training / validation metrics** (lab, ground truth known): F1_causal, SID, SHD.
- **Deployment metrics** (no ground truth): an *unsupervised* causal-conservation
  surrogate — PCMCI stability, transfer-entropy consistency, cross-scale causal
  persistence, or equivalent.

An operator defined with F1_causal in its objective is only computable where the
answer is already known; a deployable C needs the unsupervised surrogate.

## Candidate objective (illustrative, to be refined per Seam 1 & 2)

A causality-first objective, NOT the current variance objective:

```
C* = argmax [ E_causal − μ·κ(V) ]   subject to   E_recon ≤ ε,  Π_gov enforced
```

where `E_causal` uses ground-truth metrics at training and an unsupervised
surrogate at deployment, κ(V) is the effective rank (tractability, P4), and
reconstruction is a *floor constraint*, not the maximization target. The exact
form is open; the **ordering** (causality first) is the commitment.

## What is committed vs. open

- **Committed (principle):** Causality ≻ Topology ≻ Reconstruction. Future L3
  operators must demonstrate causal conservation first. This does not bind any
  implementation; if Tucker-modified / supervised-HOSVD / graph-tensor
  compression later wins, the principle still holds.
- **Open (implementation):** the exact form of C_causal, Π_gov, the unsupervised
  deployment surrogate, and the objective. Each is separately pre-registered
  future work.

## Provenance

This reframing emerged from an external consultant review of the L3 README and
paper draft, cross-checked against S3-bis evidence. The principal investigator
(JP Chancay) adopted the causality-first ordering as a foundational commitment.
Related: `experiments/causal_conservation/NEGATIVE_RESULTS.md`,
paper §6.7–6.8, §3 (Property 1).
