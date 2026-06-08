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
| **S4** | Manifold test: is dim(M_gov) << ambient? | UMAP + PCA; n=12 prelim (§6.3) + n=90 definitive (§6.5) | ✅ Confirmed (§6.5) | dim(M_gov)≈2–3, tw≥0.96 by both methods → **TUCKER** |
| S1 | Synthetic pipeline generator | New `causal_generator.py`; 3 known causal graphs (G1–G3), 90 sessions | ✅ Done (§6.4) | Signal validated: true-edge \|r\|=0.53 vs control 0.12 |
| S2 | C = Tucker on {T⁽ˢ⁾} stack; measure κ(V) | `tensorly` Tucker-HOOI | ✅ Done (§6.6) | 98% var @ r₀=1, 197× compress; Property 4 holds (κ sub-linear) |
| S3 | Causal conservation test: does M(V) recover ground-truth causal graph? | Granger (run 1) → PCMCI (run 2) | ⚠️ Inconclusive ×2 (§6.7) | Method ceiling (raw F1=0.67<0.70, 12-cycle limit); κ(V)-F1 trade-off found; needs S1-bis (longer sessions) |
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

### 6.3 S4 Results (Preliminary, n=12 corpus)

S4 was run on the n=12 ground-truth quality vectors derived from the L2 φ-calibration corpus (S1–S5, clean + failure variants). The test was run in two sessions; the second session followed a code audit of the experiment harness.

**Session 1 (UMAP only, single seed):** A single-seed UMAP sweep over `n_components ∈ {2,3,4,5}` returned trustworthiness in [0.758, 0.813], with no dimension crossing the pre-registered 0.85 threshold. The naïve gate verdict was SPARSE_SSM. An audit of the harness then identified two bugs and two methodological weaknesses (see below).

**Audit findings (`s4_manifold.py`):**

| Issue | Description | Fix |
|-------|-------------|-----|
| BUG-1 | Borderline gate branch was unreachable: `dim_gov` was only set when tw ≥ 0.85, so the [0.70, 0.85) borderline rule could never fire. | Branch on (a) whether any dim crosses 0.85 and (b) the best tw achieved. |
| BUG-2 | When no dim crossed 0.85, `dim_gov` was reported as the sweep ceiling (5), a meaningless value. | Report `dim_gov` as the best-trustworthiness dimension. |
| PROBLEM-3 | `n_neighbors` hard-coded to 5; with 2 points per scenario the local neighborhood is dominated by inter-scenario mixing. | Sweep `n_neighbors ∈ {3,4,5}`, keep best cell per dim. |
| PROBLEM-4 | Single seed; UMAP is stochastic and on n=12 the seed variance can exceed the distance to threshold. | Average trustworthiness over 10 seeds; report mean ± std. Add a deterministic PCA triangulation. |

**Session 2 (multi-seed UMAP + PCA triangulation):**

| dim | UMAP tw (mean ± std, 10 seeds) | PCA tw (deterministic) | PCA cum. variance |
|-----|-------------------------------|------------------------|-------------------|
| 2 | 0.802 ± 0.026 | **0.942** | 79.2% |
| 3 | 0.781 ± 0.015 | **0.992** | 91.8% |
| 4 | 0.788 ± 0.011 | 0.992 | 97.7% |
| 5 | 0.785 ± 0.025 | 1.000 | 99.9% |

**Interpretation.** The deterministic PCA baseline reverses the Session-1 reading. PCA achieves trustworthiness 0.992 at dim=3 and captures 91.8% of variance in three dimensions — comfortably above the pre-registered 0.85 acceptance criterion at dim ≤ 3. The large UMAP−PCA gap (−0.198) indicates that UMAP was *underestimating* the manifold: with only n=12 points, UMAP's k-NN graph is too sparse to reconstruct the topology reliably, whereas PCA recovers it directly. The governance states are low-dimensional **and approximately linear** — the latter is direct support for a multilinear operator (Tucker) rather than a non-linear sparse/SSM scheme.

**Verdict (preliminary).** H_manifold is supported at n=12: dim(M_gov) ≈ 3 with PCA trustworthiness ≥ 0.99 and >90% variance retained. The conservative gate emits `TUCKER_CAUTIOUS` (because the stochastic UMAP estimate alone does not cross 0.85), but the PCA evidence points to **TUCKER**. This is preliminary — the n=12 corpus is too small for a definitive gate. The definitive S4 will be re-run on the n≥30 synthetic corpus from S1 (≥6 sessions per causal graph), where UMAP will have sufficient density to corroborate or contradict the PCA result.

**Decision:** Proceed to S1 (synthetic corpus generator) with Tucker as the primary candidate for C. Re-run S4 on the synthetic corpus before committing the final operator choice.

### 6.4 S1 Results: Synthetic Causal Corpus

S1 produces the corpus that S2 (Tucker), S3 (causal conservation), and the definitive S4 all consume. Rather than reusing L2's `fault_injector.py` (which transforms source-code *artifacts*), S1 introduces a new generator, `causal_generator.py`, that synthesizes *quality-vector trajectories* with explicit causal edges between the 11 dimensions — because the ground truth S3 must recover is dimension-level causal structure, not artifact faults.

**Model.** Each session is a trajectory V[t] ∈ [0,1]¹¹ over T_cyc=12 cycles. A parent dimension's deviation from baseline at t−lag propagates into its child at t, scaled by an edge weight (lagged linear influence — Granger/transfer-entropy recoverable, Tucker compressible). The shock dimension that seeds each cascade follows an AR(1) recovery trajectory so it carries genuine temporal variance; an early prototype kept it at a flat depressed level, which left the first edge of each chain with no signal to propagate (lagged r ≈ 0). The fix raised every first-link correlation into the detectable range. Each trajectory is lifted to a tensor T⁽ˢ⁾ ∈ ℝ^(11×4×4×12) with small stage/agent offsets (non-trivial, non-rank-1).

**Ground-truth causal graphs (90 sessions, 30 per graph):**

| Graph | Cascade | Mirrors L2 |
|-------|---------|------------|
| G1 | security_risk → testability → maintainability | S1 (security) |
| G2 | technical_debt → maintainability → architectural_alignment | S3 (debt) |
| G3 | observability_coverage → performance → confidence | S4 (observability) |

**Generator self-test** (`validate_corpus.py`, pooled lagged Pearson r, true edges vs. 20-pair non-edge control):

| Graph | mean \|r\| true edges | mean \|r\| control | separation |
|-------|----------------------|--------------------|------------|
| G1 | 0.565 | 0.098 | +0.467 |
| G2 | 0.582 | 0.122 | +0.459 |
| G3 | 0.450 | 0.130 | +0.320 |
| **overall** | **0.532** | **0.117** | **+0.415** |

The injected causal signal is cleanly recoverable; every edge sits well above control. This is a *generator* self-test, not the S3 causal test — S3 will apply a formal causal-recovery method and report F1 against `ground_truth.json` with a pre-registered threshold (target F1 ≥ 0.70). The corpus is ground-truth-labelled and ready for S2/S3 and the definitive n=90 S4 re-run.

### 6.5 S4 Definitive: Manifold Test on n=90 — H_manifold CONFIRMED

The §6.3 S4 run (n=12 L2 corpus) was preliminary: PCA recovered a low-dimensional manifold but UMAP underestimated it because n=12 is too sparse for its k-NN graph (UMAP−PCA gap −0.198). The definitive run uses the S1 synthetic corpus (n=90, 30 per graph), with each session tensor collapsed to its representative vector v = mean over (stage, agent, cycle). The harness is the audited multi-seed UMAP + deterministic PCA from §6.3.

| dim | UMAP tw (mean ± std, 10 seeds) | PCA tw | PCA cum. variance |
|-----|-------------------------------|--------|-------------------|
| **2** | **0.9587 ± 0.003** | 0.9458 | 64.2% |
| **3** | **0.9649 ± 0.001** | 0.9616 | 71.7% |
| 4 | 0.9694 ± 0.002 | 0.9729 | 77.6% |
| 5 | 0.9696 ± 0.002 | 0.9773 | 82.1% |

**Result.** With sufficient density UMAP and PCA converge (gap ≈ 0.01, vs. −0.198 at n=12) — confirming the n=12 discordance was a sampling artifact, not a property of the data. Trustworthiness crosses the pre-registered 0.85 acceptance threshold at **dim=2** by *both* methods (UMAP 0.959, PCA 0.946). The curve is near-flat from dim 2 to 5 while explained variance rises slowly (64%→82%), indicating the governance manifold genuinely lives in ~2–3 dimensions and extra dimensions add little structure.

**Gate decision: TUCKER (confirmed).** dim(M_gov) ≈ 2–3 with trustworthiness ≥ 0.96 at dim=3. The Governance Manifold Hypothesis is supported on the definitive corpus. The manifold is low-dimensional and approximately linear, so the multilinear Tucker decomposition is the empirically motivated composition operator C. The C-gate is closed positive. **Proceed to S2** (Tucker-HOOI implementation, rank sweep, κ(V) measurement).

### 6.6 S2: Tucker Composition Operator — κ(V) and Tractability

S2 implements C = Tucker decomposition. The n session tensors of each causal graph (each 11×4×4×12) are stacked into a 5th-order tensor 𝒯 ∈ ℝ^(n×11×4×4×12) with mode-0 = sessions; Tucker-HOOI (`tensorly`) compresses it to a core G of multilinear rank (r₀, 3, 3, 3, 6). The core G is V; κ(V) is the product of core dimensions. The ambient dimension-mode rank is fixed at 3, justified by the S4 manifold result. The session-mode rank r₀ is the quantity under test (Property 4).

**Analysis A — rank sweep.** Even at r₀=1, Tucker explains ~98% of variance with 197× compression across all three graphs:

| graph | r₀=1 var. explained | r₀=1 compression | r₀=8 var. explained |
|-------|--------------------|------------------|---------------------|
| G1 | 0.981 | 197× | 0.987 |
| G2 | 0.977 | 197× | 0.985 |
| G3 | 0.979 | 197× | 0.986 |

The 30 sessions of each graph share a near-identical latent structure (they share one causal graph), so a single session-pattern already captures the structure. Relative Frobenius error floors at ~8–12% even with full ambient ranks — this is the irreducible injected noise of the S1 corpus (Tucker correctly captures structure and discards noise), so variance explained, not an error ceiling, is the meaningful budget.

**Analysis B — Property 4 (Tractability).** Minimal session-rank r₀* to reach variance ≥ 0.98 as n_sessions grows:

| graph | r₀* @ n=10 | r₀* @ n=20 | r₀* @ n=30 | r₀ ratio | verdict |
|-------|-----------|-----------|-----------|----------|---------|
| G1 | 1 | 1 | 1 | 1.00 | sub-linear |
| G2 | 1 | 2 | 2 | 2.00 | sub-linear |
| G3 | 1 | 2 | 2 | 2.00 | sub-linear |

As n triples (10→30), r₀* grows at most 2× (or stays constant), well below the linear ratio of 3.0. **Property 4 holds: κ(V) grows sub-linearly with n_sessions** — C is tractable for continuous pipelines, satisfying the L4 Efficiency Hypothesis precondition that κ(V) ≪ O(n²).

**Status.** C = Tucker satisfies Property 3 (dimensional stability, by construction — the 11-dim mode is preserved) and Property 4 (tractability, measured). Property 1 (causal preservation) is the subject of **S3**: does M(V) recover the ground-truth causal edges? Property 2 (temporal coherence) is partially exercised (cycle mode retained at rank 6) and fully tested alongside S3.

### 6.7 S3 Run 1: Causal Conservation — Inconclusive (method-attributed)

S3 tests Property 1: does C preserve the causal structure between quality dimensions? The protocol was pre-registered (§6.1) before any code: pairwise Granger causality (maxlag=1, the injected lag), α=0.01, micro-F1 across G1/G2/G3 against the 6 ground-truth edges, run on the Tucker reconstruction V̂ (primary) and the raw corpus (attribution control). Acceptance: F1 ≥ 0.70.

**Result (run 1):**

| Condition | micro-F1 | precision | recall |
|-----------|---------|-----------|--------|
| Raw corpus (control) | 0.545 | 0.375 | 1.000 |
| Tucker reconstruction (primary) | 0.048 | 0.025 | 0.500 |

**Verdict: inconclusive.** The reconstruction F1 (0.048) falls below the FAIL threshold, but the pre-registered attribution rule fires: the raw-corpus control also falls below 0.70, so the negative is attributed to the *discovery method*, not to C. The pre-registration's attribution clause did exactly its intended job — it prevented a method artifact from being mistaken for a refutation of C.

**Diagnosis.** Two distinct issues, both real. (1) Pairwise Granger recovers every true edge (raw recall = 1.00) but adds transitive and reverse false positives — e.g., for G1 it flags 3→6, the composition of the true chain 3→5→6 — because it cannot separate direct from indirect links. Precision, not recall, caps F1. (2) The low-rank Tucker reconstruction (r₀=3, cycle=6) that achieved 98% variance (§6.6) smooths trajectories and mixes dimensions through the factor matrices, producing 117 spurious lagged correlations. This is consistent with the semantic-collapse risk (§3): high numerical fidelity does not guarantee preserved causal *structure*. But the failing control means this cannot yet be attributed to C alone.

**Revision (pre-registered before S3 run 2, AMENDMENT 1).** Replace pairwise Granger with PCMCI (`tigramite`, ParCorr), which conditions each link on the target's other parents to separate direct from indirect paths; fit per session and aggregate by majority vote (≥50%); first establish a raw-corpus ceiling of F1 ≥ 0.70 before the reconstruction test is interpretable; if raw passes and reconstruction collapses, that is a clean Property-1 refutation.

**S3 Run 2 result (PCMCI):**

| Condition | micro-F1 | precision | recall |
|-----------|---------|-----------|--------|
| Raw corpus (control) | 0.667 | 1.000 | 0.500 |
| Tucker reconstruction (primary) | 0.089 | 0.047 | 0.833 |

PCMCI fixed run-1's false-positive problem (raw precision 1.00, zero transitive/reverse edges) but hit a **recall ceiling**: it recovered the first edge of each chain (3→5, 7→6, 4→8) and missed the second (5→6, 6→1, 8→9), because 12 cycles per session under a strict majority vote is too little temporal resolution for the attenuated second link. Raw F1=0.667 sits just below the 0.70 ceiling, so the amended attribution rule fires — **inconclusive (method ceiling)**; per pre-registration we stop iterating discovery methods.

**The publishable finding — κ(V)-vs-causal-F1 trade-off.** The pre-registered secondary analysis re-runs PCMCI on Tucker reconstructions across session-ranks:

| r₀ | κ(V) | reconstruction causal-F1 |
|----|------|--------------------------|
| 1 | 162 | 0.058 |
| 2 | 324 | 0.042 |
| 3 | 486 | 0.089 |
| 5 | 810 | 0.185 |
| 8 | 1296 | 0.226 |

Causal-F1 rises **monotonically with κ(V)**: the compression that achieved 98% variance (§6.6) trades away causal structure. This measures the semantic-collapse mechanism (§3) as a curve rather than asserting it. The *shape* is robust; the *absolute levels* are confounded by the raw recall ceiling, so they are not read as "C preserves X% of causality." **Decision (per pre-registration):** stop method iteration; the bottleneck is corpus temporal resolution (12 cycles). A clean Property-1 gate requires a new pre-registration with a longer-session corpus (S1-bis, t_cycles≈40–60) that re-establishes a raw ceiling ≥ 0.70. S2 and S4 are unaffected; Property 1 remains **open**, with qualitative evidence (monotone κ–F1 curve) that low-rank Tucker erodes causal structure. Runs 1–2 documented in `causal_conservation/NEGATIVE_RESULTS.md`.

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
