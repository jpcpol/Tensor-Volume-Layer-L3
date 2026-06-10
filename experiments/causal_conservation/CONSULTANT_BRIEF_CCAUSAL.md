# Consultant Brief — the functional form of C_causal

**Focus requested:** ONE design decision before pre-registering the next operator:
*what is C_causal, concretely?* After Π_gov was suspended, C_causal is the only
remaining piece that can close the causal-conservation gap Tucker cannot. The brief
is self-contained; all numbers are included.

---

## 1. Why this decision is now load-bearing

The agreed operator was `C = Π_gov ∘ C_causal ∘ C_compress`. Two of three pieces
are now fixed:

- **C_compress = Tucker** — kept (pure low-rank Tucker fails causally, but Tucker
  *per se* is not refuted; its κ(V) tractability is validated by S2).
- **Π_gov = SUSPENDED** — the prior consult ruled Option C: the S4 governance
  manifold is *static* (reconstructible with the time axis averaged out, tw=0.96
  even at t=48) while causality is *temporal*; using a static manifold in the
  objective would re-enter the S3-bis trap via the tie-break tolerance band. So
  `M_gov` is now a descriptor, not a driver, and Π_gov waits for evidence of a
  causal manifold (Q_L3.2). Next operator: **`C = C_causal ∘ C_compress`**.

That leaves **C_causal as the single new piece** — and the operator search showed
it is *necessary*, because Tucker alone is capped well below the goal (§2).

---

## 2. The quantitative gap C_causal must close (measured, not assumed)

U is the validated unsupervised causal metric (Pearson corr of off-diagonal PCMCI
val_matrix flow; TCI: M1 ρ=1.0, C2 U_shuffle=0.068). On U's scale:

```
raw ceiling          U = 1.000
best Tucker (45-cfg) U = 0.4415   ← C* = (r0=8, r_dim=3, r_stage=3, r_agent=3, r_cycle=6)
shuffle floor        U = 0.0684
```

- The **entire** Tucker rank grid spans U ∈ [0.187, 0.441]. The best Tucker sits at
  only **40% of the way** from the shuffle-floor to the raw-ceiling.
- **Headroom raw → best-Tucker = 0.5585**, and it is **unreachable inside the Tucker
  family** — exhaustive search found no config beyond C*.

So C_causal is not a refinement; it is the mechanism that must recover causal flow
Tucker's low-rank projection redistributes/destroys. The question is its form.

---

## 3. What we have learned about WHERE causality lives (constrains the form)

Three independent results point the same way and should constrain C_causal:

1. **S3-bis:** causality lives in *structure*, not in reconstruction (Tucker keeps
   98% variance, F1_causal=0.135).
2. **TCI:** U's discriminating power depends on PCMCI's *discrete parent-selection*
   step (the PC step). Continuous proxies that drop it fail to order causal
   fidelity (ρ ≤ 0.6 vs U's 1.0).
3. **Proxy audit:** the causal signal is in the *graph* (which edges), not in the
   *coefficient magnitudes*. Magnitude-based, fully-conditioned, and bivariate
   proxies all fail to rank.

**Implication:** C_causal should preserve/repair *causal structure* (which channels
carry flow), and is probably **not** expressible as a smooth magnitude penalty —
the very thing the audit refuted. The optimization is derivative-free over the
operator's parameters (already established; U is the objective directly).

---

## 4. The candidate forms for C_causal (your call, or a fourth)

All assume the pipeline `T → C_compress(Tucker) → C_causal → V`, objective
`max U(C(T))`, derivative-free search.

**Form 1 — Causal-support reconstruction mask.**
After Tucker reconstructs V̂, estimate the causal support (PCMCI edges) on the raw
corpus once, and *project V̂'s flow onto that support* — zero/attenuate flow on
channels not in the raw causal graph, preserving flow on true channels. C_causal is
a structural mask, not a magnitude transform.
*Pro:* directly targets Tucker's failure mode (fabricated spurious edges — G1: 2
true vs 28 reconstructed). *Con:* presumes the raw causal support is trustworthy
and static; bakes in the linear-direct regime U measures; "estimate once" may not
generalize across sessions.

**Form 2 — Residual causal corrector.**
Keep Tucker for the bulk (compression), and add a low-parameter corrector that
*adds back* the lag-1 causal component Tucker removed, fit to maximize U. C = Tucker
+ Δ_causal, where Δ_causal lives only on the temporal/causal axis.
*Pro:* preserves Tucker's tractability; the corrector is small (few params →
derivative-free is cheap). *Con:* "the causal component" must be defined without a
smooth proxy (the audit's lesson); risks re-introducing magnitudes through Δ.

**Form 3 — Structure-first decomposition (abandon Tucker as the inner step).**
Replace C_compress∘C_causal with a single decomposition whose *latent factors are
constrained to the causal graph* (e.g. a graph-regularized / causal-tensor
factorization) — causal channels are first-class, compression is secondary.
*Pro:* most aligned with "causality in structure"; could reach the headroom Tucker
can't. *Con:* loses the validated κ(V) tractability (S2); many simultaneous
unknowns; effectively a new operator family, not a correction.

**Form 4 — No new operator yet; first characterize the target.**
Defer C_causal. Run the Q_L3.2 exploratory step first (does a low-dim causal-
structure manifold exist?), because the *form* of C_causal depends on the geometry
of what it must preserve. Only design C_causal once that geometry is known.
*Pro:* avoids designing the corrector blind, exactly the discipline that has paid
off three times. *Con:* delays a usable operator; Q_L3.2 may be open-ended.

---

## 5. The tension we want your read on

There is a real ordering question between Form 4 and Forms 1–3:

> Do we design C_causal **now** (Forms 1–3) against the validated objective U — and
> risk designing it before we understand the geometry of causal structure — or do we
> characterize the causal-structure target **first** (Form 4 / Q_L3.2) and let that
> geometry dictate C_causal's form?

Our instinct, given that the last three audits all rewarded "characterize the
instrument/target before building on it," leans to **Form 4 first, then Form 1 as
the most evidence-aligned corrector** (structural mask, directly attacking the
fabricated-edge failure mode). But Form 1 is also cheap enough to *try now* as a
falsification: if a structural mask already moves U materially above 0.441, that is
strong evidence the structure-preservation thesis is right and Form 3/Q_L3.2 are
worth the larger investment.

---

## 6. The single question

> Given that (a) C_causal is now the only piece that can close the 0.56 U-headroom
> Tucker cannot, and (b) all evidence says causality lives in discrete structure not
> magnitudes — what form should C_causal take, and should we design it now against U
> (Forms 1–3) or first characterize the causal-structure target (Form 4 / Q_L3.2)?

---

## 7. What is NOT in question

- **U** — validated instrument and usable direct objective.
- **C_compress = Tucker**, **Π_gov suspended** — settled in prior consults.
- **Derivative-free search**, no differentiable proxy — settled by the proxy audit.
- **dim(M_gov^static) ≈ 2–3** — confirmed on S1 and S1-bis (descriptor only now).

## 8. Evidence (in order)

1. This brief (self-contained).
2. `results/operator_search_results.json` — Tucker ceiling U*=0.4415, full 45-cfg grid.
3. `results/tci_results.json` — U validation (M1 ρ=1.0, C2 0.068).
4. `results/audit_proxy_results.json` — why magnitude proxies fail (causality is structural).
5. `CONSULTANT_BRIEF_PIGOV.md` §9 — Π_gov suspension (why C_causal is the only new piece).
6. `TCI_CONSULTANT_INTERCONSULT.md` §11–§12 — proxy refutation + operator search.
