# CAL-L3 — Tensor Volume Layer

**Part of:** [CAL — Cognitive Abstraction Layers](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL) — the research starts at CAL; L3 is its Tensor Volume Layer.  
**Author:** Juan Pablo Chancay · Aural Syncro  
**Status:** Characterization closed (~95%, 2026-06) — operator C characterized; causal conservation = structural sparsity preservation. See [`L3_CLOSURE.md`](L3_CLOSURE.md).  
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

## The Central Problem: Operator C (characterized)

C must satisfy four properties (§5.2 CAL pre-paper): causal preservation, temporal coherence, dimensional stability, tractability. L3 **characterized** what causal preservation actually requires — the headline result:

> **Reconstruction fidelity ≠ causal conservation.** Low-rank Tucker preserves 98% of variance yet destroys causal structure (S3-bis: reconstruction causal F1 = 0.135 against a perfect raw control). Variance-optimal compression and causality-preserving compression are *different objectives*.

This reordered the framework to the validation order **Causality ≻ Topology ≻ Reconstruction**, and built C as:

```
C = C_causal ∘ C_compress
    C_compress = Tucker   (validated tractable: κ(V) sub-linear in n, 195.6× compression)
    C_causal   = structural prune to the causal support
```

**Operative property (the definition L3 closed on):** an operator is *causally conservative* iff it preserves the observational invariants **Ω₀ = (R, C, S)** — reachability, coverage, consistency — under compression, **not** reconstruction error.

---

## What L3 found (the characterization, closed)

| Result | Finding |
|--------|---------|
| **S2** | Tucker is tractable: κ(V)=1296, 195.6× compression, sub-linear in n (Property 4 holds). |
| **S4** | Governance manifold confirmed: dim(M_gov)≈2–3, trustworthiness ≥0.96. *Static* (time-averaged). |
| **S3-bis** | Property 1 **refuted** for plain Tucker: reconstruction ≠ causality (F1=0.135 vs raw 1.000). |
| **TCI** | A ground-truth-free causal metric **U** (PCMCI val-matrix flow) validated as both instrument and objective. |
| **Proxy audit** | Causality is **structural, not magnitude**: no differentiable magnitude proxy reproduces U's ordering. |
| **Q_L3.2A** | Tucker's failure is **spurious-edge fabrication**, not loss: coverage + sign held, reachability/\|E\| exploded. |
| **Form 1** | A structural prune to the raw causal support recovers **75% of the causal-conservation headroom with no ground truth** (U 0.441→0.862, raw↔GT gap = 0.000). |

Full narrative + formal definition + provenance: [`L3_CLOSURE.md`](L3_CLOSURE.md).

**Downstream — the residual frontier, now characterized (L4-B0, 2026-06).** Form 1's prune leaves a residual `ΔU≈0.138` (the gap from U=0.862 to 1.0), declared by L3 as a frontier (flow magnitude / higher-order structure). L4 characterized it: only ~19% is linear-edge-representable (magnitude + sign); **~81% is non-linearity** a single *linear* volume cannot carry (lag>1 ≈0%). The frontier is therefore **non-linear**, not a tuning gap — which is why L4 keeps the dual `(V_Tucker, G_pruned)` as terminal and does not pursue a single linear V′ (L4 §5.5). L3's "structural sparsity preservation" thesis is unaffected; the residual it declared is now explained.

### Governance Manifold Hypothesis (§5.6) — confirmed, but demoted to descriptor

dim(M_gov)≈2–3 is confirmed. However L3 found this manifold is *static* (reconstructible with the time axis averaged out) while causality is *temporal* — so M_gov is retained as a **descriptor**, not as a projection driver (`Π_gov` suspended). Whether a low-dimensional *causal* manifold exists is an open L4 question (Q_L3.2).

---

## Experimental record (all pre-registered, all run)

Every step committed its pre-registration *before* code; negatives reported honestly.

| Step | Task | Result |
|------|------|--------|
| S4 | Manifold test: dim(M_gov) << ambient? | ✅ dim≈2–3, tw≥0.96 (confirmed on S1 and S1-bis) |
| S1 / S1-bis | Synthetic corpus with known causal graph | ✅ 3 graphs × 30 sessions, t=48 |
| S2 | C = Tucker over {T⁽ˢ⁾}; κ(V) | ✅ κ=1296, 195.6×, tractable |
| S3 / S3-bis | Does the reconstruction recover the causal graph? | ❌ Property 1 refuted (clean) → reframing |
| TCI · proxy audit · operator search · Ω · Q_L3.2A · Form 1 | Characterize causal conservation | ✅ Closed — sparsity-preservation thesis confirmed |
| S5 | O(n²) flat vs O(κ) cost contrast on MI300X | ✅ Run & frozen on MI300X (D(n)→52.8× @4k, mechanism confirmed) — via L4-A / AMD |

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
- **Unblocks:** L4 — with C characterized and κ(V) concrete, the L4-A operator delivers the dual volume `(V_Tucker, G_pruned)` that L4 / AMD's κ vs n² contrast consumes (gate-C closed)  
- **AMD-Instinct collaboration:** the O(n²) vs O(κ) contrast runs on MI300X; L3 supplies the O(κ) side (κ=1296)  

Full collaboration context: [CAL collaboration doc](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL)

---

## Related Repos

| Repo | Role |
|------|------|
| [CAL](https://github.com/jpcpol/Cognitive-Abstraction-Layer-CAL) | Framework root — pre-paper, architecture |
| [L2 — TCO](https://github.com/jpcpol/TENSOR-BASED-COGNITIVE-OVERSIGHT-TCO) | Produces T⁽ˢ⁾ corpus that L3 consumes |
| [L4 — Meta-Inference](https://github.com/jpcpol/Meta-Inference-Layer-L4) | Consumes V produced by C |
