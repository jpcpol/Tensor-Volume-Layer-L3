# Consultant Brief — Π_gov: static governance manifold vs temporal causality

**Focus requested:** ONE foundational design decision, before we pre-register the
next operator. Not interpretation, not the whole roadmap — just: *what manifold
should Π_gov project onto?* The brief is self-contained; all evidence is included.

---

## 1. One-paragraph context

L3 compresses multi-agent pipeline state with an operator `C: T → V` that must
**preserve causal observability** (the order Causality ≻ Topology ≻ Reconstruction,
adopted after S3-bis refuted reconstruction as a valid criterion). We validated an
unsupervised causal metric **U** (TCI) and confirmed it works as a direct
optimization objective (operator search). The agreed next operator is
`C = Π_gov ∘ C_causal ∘ C_compress`, where **Π_gov projects V onto the Governance
Manifold M_gov**, and you proposed Π_gov act as a **lexicographic tie-break**:
`max U` first (causal), then `max tw` (topological) among causally-equivalent
solutions. Before pre-registering it, a coherence review surfaced a tension in
what M_gov actually *is*. That is the only question here.

---

## 2. The two relevant prior results (verbatim, with numbers)

### 2.1 The governance manifold M_gov (S4) is STATIC

S4 estimates dim(M_gov) by reducing each session tensor `T ∈ ℝ^(11×4×4×t)` to a
single vector and measuring trustworthiness of low-dim embeddings:

```
reduction:  v = mean over (stage, agent, CYCLE)  →  v ∈ [0,1]^11     ← time averaged out
```

Result (S1, n=90, t=12): **dim(M_gov) ≈ 2–3, trustworthiness 0.96** (UMAP) /
PCA corroborates. **The manifold is highly reconstructible with the time axis
averaged away.** It is a *static / marginal* object.

### 2.2 The causality U preserves is TEMPORAL

U is built on the PCMCI val_matrix (lag-1 partial-correlation flow). The TCI's C2
gate destroyed temporal dependencies at fixed marginals (per-dimension cycle
shuffle) and U collapsed:

```
U(raw, raw)     = 1.000      U(raw, Tucker r=8) = 0.441
U(raw, shuffle) = 0.068      ← causality destroyed, marginals intact
```

So U measures something that **lives in the time axis** — the exact axis S4
averages out. Causality is temporal; the S4 manifold is static.

---

## 3. The tension (the question)

Π_gov, as a lexicographic tie-break, would at Level 2 maximize `tw` — i.e. project
toward / reward proximity to the **S4 static manifold**. But that manifold was
shown (§2.1) to be reconstructible *without* temporal information, while the causal
structure we are trying to preserve (§2.2) is *entirely* temporal.

> **The hazard:** Π_gov's topological tie-break optimizes a *static* geometry. That
> is precisely the move S3-bis warned against — preserving geometry while losing
> causality — re-entering through the back door of the tie-break. The `tw` Π_gov
> would maximize is static topology, not causal topology.

This connects to a seam we already flagged when adopting the framework: *topology
must be subordinate to causality, not a hard co-constraint.* The review makes that
seam concrete and unavoidable for Π_gov's definition.

---

## 4. What we audited before bringing this to you

We did NOT bring a raw worry — we closed everything closable first.

### 4.1 Gap 1 (corpus mismatch) — CLOSED by audit

S4/S2 were measured on **S1 (t=12)**; U and the operator search run on **S1-bis
(t=48)**. Risk: the manifold Π_gov assumes might not transfer across corpora. We
re-ran the **exact S4 harness** (same reduction, same UMAP+PCA gate) on S1-bis
(`audit_s1bis_manifold.py`, results `manifold_test/results/audit_s1bis_manifold.json`):

| dim | UMAP tw (S1-bis) | PCA tw | PCA cum-var | S1 (t=12) for comparison |
|----:|-----------------:|-------:|------------:|--------------------------|
| 2 | 0.9611 | 0.9422 | 0.6245 | ~0.959 |
| 3 | 0.9630 | 0.9549 | 0.7471 | ~0.965 |
| 4 | 0.9651 | 0.9682 | 0.8005 | — |
| 5 | 0.9637 | 0.9814 | 0.8469 | — |

**Verdict: dim(M_gov) ≈ 2–3 holds on S1-bis, essentially identical to S1.** The
manifold is robust to corpus / time-length. Π_gov can safely assume dim≈2–3 on the
corpus U validates. **This gap is resolved — not part of the question.**

Note the audit *sharpens* §3: even at t=48 (4× the cycles), the time-collapsed
manifold still reconstructs at tw=0.96. The static structure is not a t=12
artifact; it is genuinely there with the temporal axis removed.

### 4.2 Gap 2 (this brief) — NOT closable by code

Whether Π_gov should target a static or a causal manifold is a *definitional*
choice about M_gov, not an empirical one. Hence this consult.

---

## 5. The three options we see (your call on which, or a fourth)

**Option A — Keep the S4 static manifold, accept a static tie-break.**
Π_gov projects onto the validated S4 geometry; the tie-break is explicitly
*static topology among causally-equivalent operators*. Defensible IF the
lexicographic order genuinely insulates causality: causality is decided at Level 1
(U) before topology is ever consulted, so a static Level-2 can't override it.
*Risk:* "causally-equivalent" is never exact in practice (U ties are approximate),
so a static tie-break could still drift toward geometry within the tolerance band.

**Option B — Redefine M_gov as the CAUSAL-STRUCTURE manifold.**
The tie-break geometry becomes the space of causal graphs (e.g. PCMCI
val_matrix / edge-structure embeddings), not time-averaged vectors. This aligns
with your own finding that *causal governability lives in structure, not in
magnitudes* — the manifold relevant to governance might be the manifold of causal
structures, which is temporal by construction. *Risk:* we have not shown this
object is low-dimensional or even well-defined; it would need its own S4-style
characterization before Π_gov could use it.

**Option C — Drop Π_gov as a projection; make topology a soft reported diagnostic.**
Optimize U directly (already validated as an objective), and report `tw` to the
static manifold only as a *post-hoc descriptor*, never in the objective. Honors
"topology subordinate to causality" maximally — topology never competes. *Risk:*
we lose the governance-manifold projection the paper's seed ("ideal C = Π_{M_gov}")
pointed at; M_gov becomes descriptive, not constructive.

We lean toward **B as the principled answer but C as the safe next step** (defer
the harder B until the causal-structure manifold is characterized), but this is
exactly the call we want your read on.

---

## 6. The single question

> Given that the S4 governance manifold is **static** (reconstructible with time
> averaged out, tw=0.96 even at t=48) while the causality U preserves is **temporal**
> (collapses to 0.068 under temporal shuffle) — what should Π_gov project onto?
> (A) the validated static S4 manifold as a lexicographic tie-break, (B) a
> redefined causal-structure manifold, or (C) none — topology demoted to a reported
> diagnostic and U optimized directly?

Whatever you choose fixes the Π_gov pre-registration. Everything downstream
(composition order, the enlarged operator family) follows from it.

---

## 7. What is NOT in question

- **U** — validated instrument (M1∧C2 PASS) and usable objective (operator search).
- **dim(M_gov) ≈ 2–3** — confirmed on both S1 and S1-bis (§4.1).
- **The validation order** Causality ≻ Topology ≻ Reconstruction — committed.
- **C_compress = Tucker** — kept (pure low-rank Tucker fails; Tucker per se doesn't).

## 8. Documents / evidence (in order)

1. This brief (self-contained).
2. `manifold_test/results/audit_s1bis_manifold.json` — Gap-1 audit (§4.1).
3. `results/tci_results.json` — U validation (M1 ρ=1.0, C2 U_shuffle=0.068).
4. `results/operator_search_results.json` — U as objective.
5. `TCI_CONSULTANT_INTERCONSULT.md` §10–§12 — the design decisions to date.
6. `manifold_test/manifold_results_definitive.json` — original S4 (S1, t=12).

---

## 9. Consultant resolution — Option C now, B as a research hypothesis; A rejected

The consultant ruled, from methodology (not architecture preference). Decision is
fixed; recorded here as what the Π_gov pre-registration commits to.

### 9.1 Verdict

- **A — REJECTED.** The lexicographic order only protects causality under *exact*
  causal equivalence, which never occurs (e.g. U₁=0.921 vs U₂=0.918). A tolerance
  band `|U₁−U₂|<ε` is unavoidable, and inside it `tw` — a geometry shown to exist
  *without* causality — decides. That indirectly re-enters the S3-bis pattern
  (preserve geometry → lose causality) and is epistemologically inconsistent with
  Causality ≻ Topology. Not fatal, but inconsistent; rejected.
- **B — probably the correct long-term formulation, but TODAY an undemonstrated
  hypothesis.** Everything recent points at it (S3-bis: causality in structure;
  TCI: in structural *selection*; proxy audit: in the graph, not coefficients →
  `M_gov` is plausibly the manifold of causal structures, not of mean states). But
  we have not shown such an object exists, nor its dimension/stability/
  trustworthiness. Methodological principle invoked: **never use as an optimization
  constraint an object whose existence has not yet been demonstrated.**
- **C — RECOMMENDED.** The only option coherent with all accumulated evidence: U is
  valid, `M_gov^static` exists, but we do *not* know `M_gov^static` is relevant to
  preserving causality, nor that `M_gov^causal` exists. So: optimize U, **report**
  the static manifold, put **no manifold in the objective.** `M_gov^static` becomes
  a **descriptor, not a driver.**

### 9.2 The structural consequence (supersedes the §10.3 composition)

The next operator is **`C = C_causal ∘ C_compress`** — **Π_gov is SUSPENDED**, not
deleted, pending evidence of which manifold it should represent. The paper's seed
(`ideal C = Π_{M_gov}`) is not abandoned; it is demoted from constructive to
hypothetical until `M_gov^causal` is characterized.

### 9.3 Pre-registration shape the consultant prescribes

- **Primary objective:** `max U(C(T))` (directly — already validated).
- **Secondary (reported):** characterize `M_gov^static` — dimension, trustworthiness,
  stability. Descriptor only.
- **Exploratory (NOT in the operator):** investigate whether `M_gov^causal` exists,
  defined over PCMCI graphs / val_matrix / Ω-structures.

### 9.4 New formal research question (adopted)

> **Q_L3.2:** Does a low-dimensional manifold of *causal structures* exist that
> explains multi-agent governability better than the static manifold identified in
> S4?

The consultant flags this as "today far more scientifically interesting than any
new Tucker variant" — it sets the program's medium-term direction (and connects to
the cross-scale `M_gov` / L4 destination already catalogued as Q4).

### 9.5 Methodological strategy statement (consultant, verbatim, adopted)

> The program currently has a validated causal metric (U) and a characterized
> static manifold (S4), but not yet sufficient evidence to assert that this
> manifold is the geometric object relevant to causal governance. Consequently the
> Π_gov operator is temporarily withdrawn from the optimization objective and the
> manifold is retained as an observational descriptor. The existence and
> characterization of a possible causal manifold become an explicit hypothesis for
> a later phase of the program.
