# CAL-L3: The Tensor Volume Layer
## Composition Operator C and the Governance Manifold Hypothesis

**Juan Pablo Chancay**  
Aural Syncro Research Lab  
jpcpol@gmail.com

**Version:** 0.1 — Draft / Working Paper  
**Date:** June 2026  
**Status:** Pre-experimental. Operator C is an open problem. Hypotheses to be pre-registered before data collection.  
**Part of:** CAL architecture — [CAL pre-paper DOI 10.5281/zenodo.20430343](https://doi.org/10.5281/zenodo.20430343)  
**Repository:** [github.com/jpcpol/Tensor-Volume-Layer-L3](https://github.com/jpcpol/Tensor-Volume-Layer-L3)  
**License:** CC BY-NC 4.0 (this document) · AGPL-3.0 (src/)  
**Target venue:** NeurIPS / ICML

---

## Abstract

The Cognitive Abstraction Layer (CAL) architecture proposes a five-level hierarchy for compressing multi-agent AI pipeline state into decision-relevant representations. Layer 2 (TCO-L2) produces a cognitive tensor T[d,i,j,k] per pipeline session, validated empirically via a between-subjects RCT (n=40). Layer 3 (L3) — this paper — addresses the question: how do we aggregate a *sequence* of cognitive tensors into a unified volume V that preserves their causal and temporal structure, while remaining tractable for Layer 4 meta-inference?

The central open problem is the **composition operator C**: a mapping `V = C({T⁽ˢ⁾}_{s=1}^{n})` that satisfies four formal requirements — causal preservation, temporal coherence, dimensional stability, and tractability. We introduce the **Governance Manifold Hypothesis**: that governance-relevant pipeline states occupy a low-dimensional manifold M_gov embedded in the full tensor space, bounded by the number of distinct failure patterns. If confirmed, C reduces to a manifold projection and the L4 Efficiency Hypothesis becomes a geometric consequence.

This paper reports the pre-experimental framework. We describe the formal requirements for C, the Tucker decomposition as a primary candidate, the synthetic validation plan (S1–S4), and the pre-registration protocol. Results will be reported as they are obtained — negative results are treated as publishable under the methodological principle that pre-registered negatives are necessary for the research program.

**Keywords:** tensor composition, causal preservation, governance manifold, Tucker decomposition, semantic information density, multi-agent AI, cognitive abstraction

---

## 1. Introduction

### 1.1 Motivation

TCO-L2 [Chancay 2026] demonstrated that human operators achieve superior governance accuracy when working with the cognitive tensor T[d,i,j,k] rather than with raw artifact streams. The tensor abstracts per-session pipeline state into a four-dimensional structure — quality dimension × pipeline stage × agent × time — that supports efficient supervisory reasoning.

A single session tensor, however, captures a bounded window of pipeline activity. Real deployment involves continuous pipelines with hundreds of sessions, progressive architectural drift, and cross-sprint dependency patterns that no single T can represent. L3 addresses this: it aggregates session tensors into a volume V that encodes cross-session dynamics while preserving the causal and dimensional structure that L4 meta-inference requires.

The challenge is formal. Any naive aggregation — tensor concatenation, averaging, direct product — either destroys causal structure, grows combinatorially with n, or loses dimensional compatibility with L4. A principled composition operator C is required.

### 1.2 Position in the CAL Architecture

```
L4  Meta-Inference         M(V) → {decisions, predictions, adaptations}
     ↑ consumes V from L3
L3  Tensor Volume          V = C({T⁽ˢ⁾}_{s=1}^{n})          ← THIS PAPER
     ↑ aggregates T from L2 sessions
L2  Cognitive Tensor       T[d,i,j,k] + NCF                   ← TCO-L2 [Chancay 2026]
     ↑ vectorizes from L1
L1  Semantic Features      quality vectors V ∈ [0,1]¹¹
L0  Raw Artifacts          AI-generated code, configs, logs
```

L3 is the first layer that operates above human working memory. Once V exists, L4 can perform governance inference without requiring a human supervisor at each step — but only if C is semantically conservative.

### 1.3 Relation to AMD-Instinct Collaboration

The cost contrast between O(n²) flat-context attention (which processes raw L0 artifacts) and O(κ(V)) inference on compressed volumes (the L4 Efficiency Hypothesis) requires empirical hardware measurement. AMD-Instinct Labs provides this via `fa_dme` (Flash Attention with DME async, validated on MI300X). Step S5 of the experimental plan — the cost contrast — is delegated to AMD-Instinct and deferred until C is validated. Sections 5 and 6 describe this gate.

---

## 2. Formal Definition

Let `T⁽ˢ⁾ ∈ ℝⁿ×|S|×|A|×|T_idx|` denote the cognitive tensor produced by pipeline session s, where:
- n = 11 (quality dimensions v₁...v₁₁)
- |S| = number of pipeline stages
- |A| = number of agents
- |T_idx| = number of time indices in session s

The **tensor volume** is:

```
V = C({T⁽ˢ⁾}_{s=1}^{n_sessions})
```

where C is the **composition operator** — the central open problem of this paper.

### 2.1 The Quality Vector Basis

V inherits the 11-dimensional quality structure of L2:

| Dim | Name | Type |
|-----|------|------|
| v₁ | functional_correctness | supervisory estimator |
| v₂ | architectural_alignment | supervisory estimator |
| v₃ | scalability_projection | supervisory estimator |
| v₄ | security_risk | verifiable (Bandit) — inverted |
| v₅ | observability_coverage | supervisory estimator |
| v₆ | testability | verifiable (Radon) |
| v₇ | maintainability | verifiable (Radon) |
| v₈ | technical_debt | verifiable (Radon) — inverted |
| v₉ | performance | supervisory estimator |
| v₁₀ | confidence | supervisory estimator |
| v₁₁ | anomaly_score | composite — inverted |

Dimensional stability (Property 3 below) requires that V exposes these 11 dimensions consistently across all slices that L4 will query.

---

## 3. Requirements for the Composition Operator C

For V to be semantically conservative with respect to L4 inference, C must satisfy four properties (§5.2 CAL pre-paper):

### Property 1 — Causal Preservation

Causal relationships between quality dimensions recorded within any T⁽ˢ⁾ must be encodeable in V. Formally, for any pair of quality dimensions (dᵢ, dⱼ) where dᵢ causally produces dⱼ within session s, the conditional interventional distribution P(dⱼ | do(dᵢ)) must remain estimable from V.

*Example:* if a security hardening decision at k=2 caused a testability degradation at k=3 within session s (pattern present in scenario S1 of the L2 corpus), C must represent this dependency — not collapse it into a spurious correlation.

Failure mode: **semantic collapse** — C preserves acceptable SID for each individual governance class while destroying the diversity of governance states across classes. This passes contemporaneous SID checks while producing systematic failures on novel scenarios.

### Property 2 — Temporal Coherence

Time indices of constituent tensors must be composable into a global temporal ordering. Cross-session drift (e.g., progressive architectural degradation across 10 sprint cycles) must be representable in V.

*Example:* the gradual v₈ drift in S3 (−0.08/cycle, detectable only at 3-cycle accumulation in L2) must remain representable in V across sessions, since L4 may need to project this drift 5–10 sessions forward.

### Property 3 — Dimensional Stability

The 11-dimensional quality vector structure must be preserved across the composition. L4 must be able to apply the same inference operations (Ω, Δ, Ρ, Ξ) to slices of V that it applies to individual T⁽ˢ⁾ tensors.

### Property 4 — Tractability

C must be computable for continuous pipelines. The size of V must not grow linearly with n_sessions for large n. A naive tensor product C = T⁽¹⁾ ⊗ T⁽²⁾ ⊗ ... ⊗ T⁽ⁿ⁾ violates this property: it grows exponentially in dimensionality.

---

## 4. Tractability Mitigation Strategies

Four candidate approaches from §5.3 of the CAL pre-paper:

| Strategy | Mechanism | Tractability claim | Key assumption |
|----------|-----------|-------------------|----------------|
| **Sparse representation** | COO/CSR format; zero entries for never-co-occurring combinations | O(k) storage, k = nonzero entries | Pipeline interactions are sparse |
| **Low-rank approximation (Tucker/CP)** | Factor matrices compress V to core G of effective rank r | O(r) inference if r << ambient dim | Governance patterns have low effective rank |
| **Selective retention (dynamic forgetting)** | SSM/LSTM-like gating; discard low-SID sessions | Effective n bounded at any time | SID estimable without full human reference |
| **Manifold projection** | Project onto M_gov; discard off-manifold components | dim(M_gov) << ambient dim | Governance Manifold Hypothesis (§5) |

The four strategies are not mutually exclusive. The primary candidate for the first validation is low-rank Tucker decomposition (§4.1), with manifold projection as the key theoretical backing hypothesis (§5).

### 4.1 Primary Candidate: Tucker Decomposition

Tucker decomposition represents a tensor V as:

```
V ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
```

where G is the compressed core tensor and U₁...U₄ are factor matrices. The effective rank κ(V) = rank of G. If governance-relevant patterns recur across sessions (a testable empirical claim), then r << n and the decomposition is tractable.

**Implementation:** `tensorly` library (Python). Tucker-HOOI (Higher-Order Orthogonal Iteration) algorithm.

**κ(V) as the structural complexity measure:** κ(V) is the key quantity for the L4 Efficiency Hypothesis — if Cost(M(V)) = O(κ(V)) and κ(V) << O(n²), the efficiency hypothesis holds empirically.

---

## 5. The Governance Manifold Hypothesis

> **Governance Manifold Hypothesis:** the set of governance-relevant pipeline states is a low-dimensional manifold M_gov embedded in the full tensor space. The intrinsic dimensionality of M_gov is bounded by the number of distinct governance-relevant failure patterns the pipeline can produce — not by the full tensor dimensionality.

If confirmed, the consequences cascade through the architecture:

- **Tractability:** V only needs to represent M_gov; effective dimensionality = dim(M_gov), potentially orders of magnitude smaller than ambient.
- **C definition:** C = projection onto M_gov — not a general approximation, but a geometrically motivated operator.
- **L4 Efficiency:** if M(V) operates on M_gov, Cost(M(V)) = O(dim(M_gov)), decoupled from n_sessions.
- **SID geometry:** semantic conservation becomes a topological question — does C preserve the connectivity of M_gov?
- **Semantic collapse framing:** collapse = C projects M_gov onto a lower-dimensional submanifold, destroying distinctions between failure modes that are topologically separated in M_gov.

### 5.1 Testability at L2 (Before New Data Collection)

The hypothesis is testable on the existing TCO-L2 corpus (S1–S5) without new data:

If quality vector trajectories across S1–S5 can be embedded in a low-dimensional space (via UMAP or Isomap) while preserving inter-scenario distances, this is empirical support for a low-dimensional governance manifold at L2. The embedding dimensionality is the first measurable proxy for dim(M_gov).

This is Step S4 of the experimental plan — run first, cheapest gate.

---

## 6. Experimental Plan

**Pre-registration required before S3 and S4.** Commit hypotheses to a dated git commit before running experiments. Negative results are publishable.

| Step | Task | Method | Status | Notes |
|------|------|--------|--------|-------|
| **S4** | Manifold test: is dim(M_gov) << ambient? | UMAP/Isomap on L2 corpus (S1–S5) | **Run first** | Cheapest gate — if fails, switch to sparse/SSM strategy |
| S1 | Synthetic pipeline generator | Reuse `fault_injector.py` from L2; known causal graph ground truth | Pending | CPU-only |
| S2 | C = Tucker on {T⁽ˢ⁾} stack; measure κ(V) | `tensorly` Tucker-HOOI | Pending | CPU-only |
| S3 | Causal conservation test: does M(V) recover ground-truth causal graph? | Compare recovered vs. known causal edges | Pending | Pre-register before running |
| S5 | O(n²) flat vs O(κ) cost contrast | `fa_dme` on AMD MI300X | **Deferred → AMD-Instinct** | Gate: C validated on S1–S4 |

### 6.1 Pre-Registration Protocol

Before running S3 or S4:

1. Open a PR in this repo with the exact hypotheses to be tested, the statistical tests to be used, and the acceptance criteria.
2. Merge the PR. The merge commit SHA is the pre-registration timestamp.
3. Run the experiment. Report results verbatim — positive or negative.
4. If negative: document in a `NEGATIVE_RESULTS.md` and update the roadmap with the alternative strategy.

### 6.2 Decision Gate: S4

If S4 result shows dim(M_gov) ≥ 6 (high-dimensional manifold):
- Tucker decomposition is likely insufficient as a sole strategy.
- Switch primary candidate to sparse + selective retention (SSM-inspired gating).
- S3 continues but Tucker claim is weakened to "one component of C" rather than "C".

If S4 shows dim(M_gov) ≤ 3 (low-dimensional manifold):
- Tucker decomposition is strongly motivated.
- Proceed with S1 → S2 → S3.
- Pre-register: "C = Tucker projection preserves causal structure with SID(L2→L3) > 0.70."

---

## 7. Semantic Information Density at L3

SID(L2→L3) measures how much decision-relevant information is preserved through C. Without a human reference signal at L3, measurement relies on the synthetic benchmark with known causal ground truth (S3):

```
SID_synthetic(L2→L3) ≈ (causal edges recovered by M(V)) / (total causal edges in ground truth)
```

This is an approximation of the formal SID (§7 CAL pre-paper). Full operationalization requires:
- Ground truth governance decisions for each synthetic pipeline configuration
- Comparison of M(V) decisions vs. ground truth across configurations
- Both correct decisions AND recovered causal structure count

The target threshold is SID(L2→L3) > 0.70 — consistent with the L2 empirical anchor (H2: Cohen's d > 0.50 ≈ SID(L0→L2) > 0.70).

---

## 8. Open Research Questions

1. **Algebraic structure of C:** does C belong to a known algebraic family? Tucker is a candidate, not a proof.
2. **Attractor states:** does V exhibit stable configurations across sessions? These would represent semantic invariants of the pipeline.
3. **Phase transitions:** does V exhibit discontinuous changes in global state analogous to phase transitions? Detecting precursors is a key L4 use case.
4. **Scalability:** how does κ(V) grow with n_sessions? Polynomial growth would indicate SID(L2→L3) degrades with pipeline scale — a critical failure mode to characterize.
5. **Semantic collapse detection:** what test suite design maximizes detection of collapse on rare/tail governance scenarios?

---

## 9. Relationship to Existing Work

- **Tucker/CP decomposition** [Kolda & Bader 2009]: standard tools for tractable low-rank tensor approximation. Used here as C candidate.
- **State Space Models (Mamba/S4)** [Gu & Dao 2023]: SSMs demonstrate sub-quadratic temporal compression with long-range dependency preservation — structural analog of what C must achieve on quality tensors.
- **Causal Representation Learning** [Schölkopf et al. 2021]: causal graph priors and causal regularization provide two actionable tools for Property 1 (causal preservation).
- **Renormalization Group Theory** [Wilson 1971]: L3 is a semantic coarse-graining — the RG formalism may provide mathematical guarantees on what is preserved/lost under C.

---

## 10. Roadmap

| Milestone | Gate | Timeline |
|-----------|------|----------|
| S4: manifold test on L2 corpus | — | First (before any other experiment) |
| Pre-register S3 hypotheses | S4 result determines which hypotheses | Before running S3 |
| S1–S3: synthetic validation | Pre-registration complete | Weeks 1–4 (35-day window) |
| Report S4 result (positive or negative) | — | Week 1 |
| Report S3 result (positive or negative) | S1–S2 complete | Week 4 |
| S5: cost contrast on MI300X | C validated on S1–S4 | AMD-Instinct gate |
| L3 paper draft | S1–S4 results available | Post-experiment |
| Submission (NeurIPS/ICML) | L2 paper accepted/submitted | TBD |

---

## 11. References

- Chancay, J.P. (2026). *Cognitive Abstraction Layers (CAL): A Research Architecture for Hierarchical Semantic Compression in AI Systems*. Zenodo. DOI: 10.5281/zenodo.20430343
- Chancay, J.P. (2026). *Tensor-Based Cognitive Oversight (TCO-L2)*. Working paper v3.0. [github.com/jpcpol/TENSOR-BASED-COGNITIVE-OVERSIGHT-TCO](https://github.com/jpcpol/TENSOR-BASED-COGNITIVE-OVERSIGHT-TCO)
- Kolda, T.G. & Bader, B.W. (2009). Tensor Decompositions and Applications. *SIAM Review*, 51(3), 455–500.
- Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.
- Schölkopf, B. et al. (2021). Toward Causal Representation Learning. *Proceedings of the IEEE*, 109(5), 612–634.
- Wilson, K.G. (1971). Renormalization Group and Critical Phenomena I. *Physical Review B*, 4(9), 3174.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

---

## Appendix A: Notation Summary

| Symbol | Definition |
|--------|------------|
| T⁽ˢ⁾ | Cognitive tensor from L2 session s: ℝⁿ×\|S\|×\|A\|×\|T_idx\| |
| V | Tensor volume: output of composition operator C |
| C | Composition operator: maps {T⁽ˢ⁾} → V (open problem) |
| κ(V) | Effective rank of V (Tucker core G rank) |
| M_gov | Governance manifold — low-dim subspace hypothesis |
| dim(M_gov) | Intrinsic dimension of M_gov (measured via UMAP/Isomap) |
| SID(Lk→Lk+1) | Semantic Information Density at layer boundary |
| φ | Vectorization function: artifact → V ∈ [0,1]¹¹ |
| n_sessions | Number of L2 sessions aggregated into V |
