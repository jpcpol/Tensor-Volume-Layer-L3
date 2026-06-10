# Q_L3.2A — Causal-Invariant Survival under Tucker (Pre-Registration)

**Commit this document BEFORE writing or running any Q_L3.2A code.** Its commit SHA
is the pre-registration timestamp. Q_L3.2A characterizes **which observational
causal invariants survive Tucker compression** — it does NOT design an operator and
does NOT claim governance semantics. It measures the object, before anyone builds
C_causal on top of it.

- **Date:** 2026-06-09
- **Reuses:** S1-bis corpus (`corpus_s1bis/`, t=48), the S3-bis PCMCI machinery
  (`run_s3_run2.py`: ParCorr, tau=1, pc_alpha=0.01, majority-vote edges), the TCI U.
  No new corpus, no new estimator.
- **Consultant ruling (4th consult, `CONSULTANT_BRIEF_OMEGA.md` §9):** Ω is defined
  as **observational invariants** Ω₀=(P,D,R,C,S), NOT semantic categories; Q_L3.2
  operates on Ω₀; d_Ω is **vectorial**; S1-bis suffices for Ω₀ but not Ω₁.
- **Scope decision (audited):** Q_L3.2A pre-registers **R, C, S only** — directly
  computable, high power on S1-bis. **D and P are deferred to Q_L3.2B** (needs an
  S-Ω corpus: divergence requires injected inter-agent conflict and per-agent
  series; persistence needs longer windows than 12-cycle slices give).

## The question (fixed a priori)

> Across the Tucker rank family {raw, r=1, 2, 3, 5, 8}, which of the causal
> invariants (R, C, S) are preserved, and which collapse, relative to raw — and how
> does that pattern relate to U and to the supervised F1?

This is descriptive/characterizing, not a pass/fail operator test. There is no
"PASS" verdict; the output is the invariant-survival profile.

## The invariants measured (fixed a priori)

For a corpus condition X (raw, or Tucker reconstruction at rank r), per graph
G1/G2/G3, build the PCMCI structure exactly as S3-bis (`discover_edges_majority`,
ParCorr, tau=1, pc_alpha=0.01, vote≥0.5) plus the median val_matrix Φ:

- **R — Reachability:** on the directed adjacency A from the recovered edge set E,
  `R = #ordered_connected_pairs(A) / (n(n−1))`, n=11, transitive closure counting
  pairs (i→…→j). Self-referential (no GT). Measures propagation capacity.
- **C — Coverage:** `C = |E ∩ E_ref| / |E_ref|`. Two pre-registered variants:
  `C_GT` (E_ref = ground-truth edges, training-time) and `C_raw` (E_ref = edges
  recovered on the raw corpus, deployment-time surrogate). Measures omission =
  1 − C.
- **S — Consistency:** `S = 1 − (#violations / max(|E|,1))`. A violation of edge
  (i,j) ∈ E is a **sign flip** of its val_matrix flow relative to the reference
  (GT sign for S_GT, raw sign for S_raw) for edges that exist in both. Pre-registered
  violation type for Q_L3.2A: **sign inconsistency only** (direction inversion and
  impossible-cycle checks deferred — sign is the unambiguous one on this corpus).

All three are computed on the SAME edge/flow machinery U uses, so they share the
estimator and differ only in what structural property they extract.

## d_Ω (fixed a priori — vectorial)

```
Ω(X)   = (R(X), C_raw(X), S_raw(X))        # the self-referential triple
d_Ω(raw, X) = ( |R(raw) − R(X)|,
                |C_raw(raw) − C_raw(X)|,    # = 1 − C_raw(X), since C_raw(raw)=1
                |S_raw(raw) − S_raw(X)| )
```

Reported as a 3-vector per Tucker rank. **No scalar collapse** (no weights). C_GT
and S_GT are reported alongside as the training-time references.

## What is reported (fixed a priori)

A table over {raw, r=1,2,3,5,8} with columns: R, C_GT, C_raw, S_GT, S_raw, |E|, plus
U (from TCI, already known) and supervised F1 (from S3-bis, already known). Plus the
d_Ω 3-vector per rank. Plus, per invariant, the Spearman correlation with the rank
order (does the invariant degrade monotonically as compression increases?).

## Pre-registered interpretation rules (what each pattern would mean)

These are fixed BEFORE the run so the reading is not post-hoc:

- **If R, C, S all collapse together with U** → Tucker destroys causal structure
  wholesale; the invariants add resolution but tell the same story as U. C_causal
  must restore structure broadly.
- **If some invariant survives while others collapse** (e.g. R preserved, C
  collapses) → the most important result: Tucker preserves *some* causal properties
  and not others. This **decomposes** the failure and tells C_causal precisely what
  to target. This is the outcome that would most change the operator design.
- **If an invariant tracks F1 better than U does** → that invariant is a candidate
  refinement of the unsupervised objective (a future TCI-style validation, separate
  prereg). Reported, not acted on here.

## No-method-shopping / honesty clauses

- Q_L3.2A measures R, C, S as defined above. We do NOT add invariants post-hoc to
  make a cleaner story, nor drop one that behaves awkwardly.
- D and P are **out of scope by pre-registration** (deferred to Q_L3.2B); we do not
  sneak a low-power version of them into the results.
- No PASS/FAIL: this is characterization. Claims about Ω₁ (drift/conflict as
  governance phenomena) are explicitly NOT made — only Ω₀ invariants are reported.

## Fixed parameters (summary)

- Corpus: S1-bis (corpus_s1bis, t=48). Estimator: PCMCI ParCorr, tau=1,
  pc_alpha=0.01, majority vote ≥0.5 (S3-bis machinery verbatim).
- Tucker family: ambient (3,3,3,6), session-ranks {1,2,3,5,8} (the TCI/S3-bis family);
  plus raw as the Ω reference.
- Invariants: R (reachability, transitive-closure pair fraction, self-ref),
  C (coverage, GT and raw refs), S (consistency, sign-flip only, GT and raw refs).
- d_Ω: 3-vector (R, C_raw, S_raw) vs raw, no scalar collapse.
- Cross-reference U (TCI) and F1 (S3-bis), already computed.
- Output: characterization table + d_Ω vectors + per-invariant rank-Spearman. No verdict.
