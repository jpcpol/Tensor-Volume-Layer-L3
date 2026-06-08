# CAL-L3 — Tensor Volume Layer

**Part of:** [CAL — Cognitive Abstraction Layers](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL)  
**Author:** Juan Pablo Chancay · Aural Syncro  
**Status:** In development — composition operator C is an open problem  
**Target venue:** NeurIPS / ICML  
**License:** CC BY-NC 4.0 (docs) · AGPL-3.0 (src)

---

## What is L3?

L3 is the **Tensor Volume Layer** of the CAL architecture. It defines the operator that compresses a *sequence* of cognitive tensor snapshots (L2 sessions) into a single unified volume, preserving the causal relationships that make downstream meta-inference tractable.

```
L3:  V = C({T⁽ˢ⁾}_{s=1}^{n})
```

Where:
- `T⁽ˢ⁾ ∈ ℝⁿˣˢˣᵃˣᵗ` — cognitive tensor from L2 session s (dimension × stage × agent × time)
- `C` — composition operator (open problem, this repo)
- `V` — tensor volume; input to L4 meta-inference

---

## The Central Problem: Operator C

C must simultaneously satisfy four properties (§5.2 CAL pre-paper):

| Property | Requirement |
|----------|-------------|
| **Causal preservation** | Causal relations between tensor dimensions must be encodable in V |
| **Temporal coherence** | Temporal indices of each T⁽ˢ⁾ must compose into a consistent global order |
| **Dimensional stability** | The 11-dim quality vector structure must be preserved; L4 applies the same operations |
| **Tractability** | Size of V must not grow linearly with n (number of sessions) |

C is currently an **open problem**. This repo exists to find and validate a candidate.

---

## Current Candidate: Tucker Decomposition

Authorized by §5.3.2 and §8.2 (method 2) of the CAL pre-paper.

**Approach:** Stack `{T⁽ˢ⁾}` into a higher-order tensor; apply Tucker decomposition to obtain a compressed core G and factor matrices. κ(V) = effective rank of G.

**Library:** `tensorly`  
**Validation:** Synthetic benchmark corpus with known causal ground truth (fault_injector.py from L2).

### Governance Manifold Hypothesis (§5.6)

> The set of governance-relevant states forms a low-dimensional manifold M_gov embedded in the full tensor space. The intrinsic dimension of M_gov is bounded by the number of distinct failure patterns — not by the ambient tensor dimension.

If confirmed: C = manifold projection, and the L4 Efficiency Hypothesis becomes a *consequence*, not an independent claim.

---

## Experimental Plan

**Pre-register hypotheses (dated commit) before running S3 or S4.**

| Step | Task | Status |
|------|------|--------|
| S4 | Manifold test: is dim(M_gov) << ambient dim? ← **run first** | Pending |
| S1 | Synthetic pipeline generator with known causal graph (reuses L2 fault_injector.py) | Pending |
| S2 | C = Tucker over {T⁽ˢ⁾} stack; measure κ(V) | Pending |
| S3 | Conservation test: does M(V) recover ground-truth causal graph? | Pending |
| S5 | O(n²) flat vs O(κ) cost contrast — **AMD-Instinct runs this on MI300X** | Deferred → gate C |

S4 is the cheapest early gate: if M_gov is high-dimensional, switch strategy (sparse / SSM) before investing in Tucker implementation.

---

## Repository Structure

```
L3/
├── README.md
├── paper/                  ← L3 paper (in development)
├── src/
│   └── composition/        ← operator C implementations (Tucker, sparse, manifold)
├── benchmarks/
│   └── synthetic/          ← corpus with known causal ground truth
└── experiments/
    ├── manifold_test/      ← S4: intrinsic dimension of M_gov
    └── causal_conservation/ ← S3: does V preserve causal structure?
```

---

## Dependencies

- **Consumes:** L2 validated corpus (`T⁽ˢ⁾` from scenarios S1–S5 in TCO-L2)  
- **Blocks:** L4 Rol 2 — `fa_dme` as M(V) kernel proxy requires C to exist first  
- **AMD-Instinct collaboration:** S5 (O(n²) vs O(κ) contrast) runs on MI300X post gate-C  

Full collaboration context: [CAL collaboration doc](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL)

---

## Related Repos

| Repo | Role |
|------|------|
| [CAL](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL) | Framework root — pre-paper, architecture |
| [L2 — TCO](https://github.com/jpcpol/TENSOR-BASED-COGNITIVE-OVERSIGHT-TCO) | Produces T⁽ˢ⁾ corpus that L3 consumes |
| [L4 — Meta-Inference](https://github.com/jpcpol/Meta-Inference-Layer-L4) | Consumes V produced by C |
