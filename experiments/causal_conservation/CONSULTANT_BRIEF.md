# S3 — Consultant Brief: Methodological Review

**Focus requested:** methodological validity (not interpretation, not next steps).

## One-paragraph context

S3 tests Property 1 of the L3 composition operator C (= Tucker): does C preserve
the causal structure between quality dimensions? We pre-registered a causal-
discovery protocol, ran it twice (Granger → PCMCI after a pre-registered method
amendment), and both runs returned **INCONCLUSIVE** under a pre-registered
attribution rule: *if the raw-corpus control does not itself reach F1 ≥ 0.70, the
negative is attributed to the discovery method, not to C, and we stop.* Run 2
(PCMCI) got raw F1 = 0.667 — just below the 0.70 ceiling — so we stopped.

## The three questions for review

1. **Is the attribution rule sound, or too conservative?**
   The rule stops the experiment when the control method can't clear 0.70 on
   data known to contain the signal. This prevents mistaking a weak method for a
   refutation of C. But raw F1 = 0.667 is *0.033* from the threshold. Is stopping
   the right call, or are we discarding a usable result on a rigid cutoff?

2. **Is 12 cycles/session the real bottleneck, or an implementation artifact?**
   A post-hoc diagnostic (NOT pre-registered, run for this review) compared three
   ways of applying PCMCI to the RAW corpus:

   | Aggregation | True edges found (of 2/graph) | Total predictions |
   |-------------|-------------------------------|-------------------|
   | Per-session + majority vote (run-2 method) | **1 / 2** (first link only) | 1 |
   | Sessions **concatenated** (360 time points) | **2 / 2** (both links) | 4 |
   | Union over per-session fits | 2 / 2 | ~85 (noise) |

   Concatenation recovers BOTH true edges in all three graphs with only 4 total
   predictions (precision 0.5, recall 1.0 → F1 = 0.67 with both links present).
   This indicates the run-2 recall ceiling is caused by the **pre-registered
   choice of per-session fit + majority vote**, not a bug and not the corpus per
   se: PCMCI needs the time points, and concatenating gives them.
   **Question:** does this make the pre-registered aggregation choice a
   methodological error that invalidates run 2, or a defensible-but-suboptimal
   choice whose limitation we correctly reported?

3. **Was amending the method (run 1 → run 2) legitimate, or method-shopping?**
   We changed the discovery method (Granger → PCMCI) between runs via a committed
   pre-registration amendment, because run 1's failure was diagnosed as a known
   method weakness (transitive false positives). The amended rule explicitly
   forbade further method iteration once raw < 0.70 — and we honored it after
   run 2. Is the single amendment defensible, or does any post-hoc method change
   compromise the pre-registration's validity?

## What is NOT in question

- S4 (manifold, dim ≈ 2–3) and S2 (κ(V) sub-linear, Property 4) — independent of
  the discovery method, stand on their own.
- The corpus carries the signal: generator self-test gives true-edge |r| = 0.53
  vs. control 0.12; the concatenation diagnostic confirms both edges are
  recoverable with enough time points.

## Documents to read (in order)

1. `PRE_REGISTRATION.md` — original protocol + AMENDMENT 1 (the design under review)
2. `NEGATIVE_RESULTS.md` — runs 1–2 verbatim, diagnosis, stop decision
3. `results/s3_kappa_vs_causalf1.png` — the one positive finding (κ–F1 trade-off)
4. (optional, for code audit) `run_s3.py`, `run_s3_run2.py`
5. (optional, raw data) `results/s3_causal_results.json`, `results/s3_run2_pcmci_results.json`
