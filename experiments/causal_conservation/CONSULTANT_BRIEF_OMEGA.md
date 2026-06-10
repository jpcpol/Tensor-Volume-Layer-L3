# Consultant Brief — operational definition of Ω (before pre-registering Q_L3.2)

**Focus requested:** ONE narrow, concrete thing. You named **Ω** as the causal
object L3 must preserve — `Ω = {drift, conflict, propagation, omission, structural
violation}` — and prescribed Q_L3.2 to measure how much of Ω survives Tucker. To
pre-register Q_L3.2 we need Ω **operationally defined**, and we are deliberately not
inventing those definitions ourselves (that would bias the study from the design).
This brief gives you exactly the substrate available, and asks for the metric.

---

## 1. Why we are asking instead of guessing

Q_L3.2's whole value is that Ω is the *target object*, distinct from the *instrument*
U. If we (the implementers) define drift/conflict/propagation/omission ourselves, we
risk defining them as whatever the existing PCMCI graph happens to express — which
would silently collapse Ω back onto G and defeat the point (the very A-vs-B failure
you warned about: a mask fixed to the original graph). The operational definition of
Ω must come from the methodology, not from convenience. Hence this consult.

## 2. The exact substrate available (so your definition is computable)

Per Tucker rank r (and raw), for each of the 3 causal graphs G1/G2/G3, over 30
sessions of S1-bis (t=48), we have:

- **Per-session PCMCI output:** `val_matrix[i,j,1]` (11×11 lag-1 partial-correlation
  flow) and `p_matrix[i,j,1]` (significance). Already computed in the pipeline.
- **Aggregated flow matrix** `Φ[i,j]` = median over sessions of val_matrix (the
  object U is built on).
- **Thresholded edge set** `E` (p < 0.01, majority vote ≥50% of sessions) — the
  S3-bis discovery output.
- **Ground truth** per graph: a known 2-edge causal chain (e.g. G1:
  security_risk → testability → maintainability), with signs and weights.
- The 11 named dimensions (functional_correctness … anomaly_score), the 4 stages,
  4 agents, 48 cycles of each session tensor `T ∈ ℝ^(11×4×4×48)`.

So Ω-patterns must be expressible as functions of (val_matrix, p_matrix, edge set,
ground truth, named dims, time axis). Anything you can define over those, we can
compute.

## 3. The five Ω elements — what we need defined for each

For each, we need: (a) a **computable definition** over the substrate in §2, and
(b) whether it is measured **against ground truth** (training-time) or
**self-referentially** (deployment-time, no GT — like U).

- **drift** — temporal? our guess would be change in Φ across cycle-windows, but we
  defer to you. What is drift, operationally?
- **conflict** — inter-agent? the tensor has an agent axis (4 agents). Is conflict
  defined at the agent-resolved level (which we currently average out in
  `to_dim_series`), or at the dimension-flow level?
- **propagation** — directed causal paths (i→j→k)? a function of the edge set /
  val_matrix transitive structure?
- **omission** — true edges absent from the recovered structure (a recall-type
  quantity against GT)?
- **structural violation** — edges present that violate a known constraint (sign
  flip? spurious cross-links?)?

## 4. The distance d_Ω

Q_L3.2 Level 3 needs `d_Ω(raw, Tucker_r)` — a distance between causal ontologies.
We need to know: is d_Ω a single scalar aggregating the five elements, a vector
(one per element), or a structured object? And is it defined against raw (raw as
the Ω reference) or against ground truth?

## 5. One design risk we want you to rule on

The S1-bis corpus was generated with **simple lag-1 causal edges** — it was NOT
generated to contain explicit drift/conflict/propagation patterns. Two possibilities:

- **(i)** Ω patterns are *emergent* in S1-bis (e.g. propagation exists because the
  2-edge chains ARE paths; conflict/drift may be weak or absent) — then Q_L3.2 runs
  on existing data but may have low power on the absent patterns.
- **(ii)** Ω requires a corpus *designed* to inject the patterns (an S-Ω corpus) —
  then Q_L3.2 needs a new generator before it can run.

**Which is it?** If (i), we pre-register Q_L3.2 on S1-bis now. If (ii), the prior
step is a corpus pre-registration, and Q_L3.2 waits. This determines the next move.

## 6. The single question

> Give Ω an operational definition over the §2 substrate: for each of {drift,
> conflict, propagation, omission, structural violation}, a computable metric and
> whether it is GT-referenced or self-referenced; plus the form of d_Ω; plus a
> ruling on §5 (does S1-bis already carry Ω, or do we need an S-Ω corpus first?).

## 7. What is NOT in question

- The sequence Q_L3.2 → Form 1 falsifier (settled last consult).
- U as instrument/objective; Tucker as C_compress; Π_gov suspended.
- That Ω, not the full graph G, is the preservation target (your hypothesis, adopted).

## 8. Evidence (in order)

1. This brief.
2. `CONSULTANT_BRIEF_CCAUSAL.md` §9 — the Q_L3.2-first ruling and the Ω hypothesis.
3. `results/tci_results.json`, `results/operator_search_results.json` — U + Tucker ceiling.
4. `../synthetic_corpus/corpus_s1bis/ground_truth.json` — the actual injected causal edges.
