# S1 — Synthetic Causal Corpus

Generates a corpus of session tensors `T^(s) ∈ ℝ^(11×4×4×12)` whose causal
structure between quality dimensions is **known by construction**. This corpus
is the input for:

- **S2** — Tucker composition operator C (does it compress without losing the causal structure?)
- **S3** — causal conservation test (does M(V) recover the ground-truth edges?)
- **S4 (definitive re-run)** — manifold test with n≥30 per graph (the n=12 §6.3 run was preliminary)

## Why a new generator (vs L2 `fault_injector.py`)

L2's `fault_injector.py` transforms *artifacts* (source code) at the artifact
level. S1 works one layer up: it synthesizes the *quality-vector trajectories* a
pipeline would produce, with explicit causal edges between the 11 quality
dimensions and a controlled time lag. The set of injected edges is the ground
truth that the L3 operator must later recover.

## Model

A session is a trajectory `V[t] ∈ [0,1]^11` over `t = 0..T_cyc-1` cycles:

```text
v_j[t] = clip( baseline_j
               + drift_j * t
               + Σ_{(i→j)∈G} w_ij * (v_i[t-lag] - baseline_i) * sign
               + noise )
```

A parent dimension's *deviation from baseline* at `t-lag` propagates into the
child at `t`, scaled by the edge weight. The **shock dimension** that seeds each
cascade follows an AR(1) recovery trajectory (depressed at t=0, recovering
toward baseline with a random walk) so it carries genuine temporal variance —
without this, the first edge of each chain would have no signal to propagate.

The trajectory is lifted to a tensor `T^(s)` by distributing the per-cycle
vector across 4 stages × 4 agents with small stage/agent offsets (so the tensor
is not a rank-1 broadcast).

## Ground-truth causal graphs

| Graph | Cascade | Mirrors L2 scenario |
|-------|---------|---------------------|
| **G1** | `security_risk → testability → maintainability` | S1 (security) |
| **G2** | `technical_debt → maintainability → architectural_alignment` | S3 (debt) |
| **G3** | `observability_coverage → performance → confidence` | S4 (observability) |

## Run

```bash
python causal_generator.py            # 30 sessions/graph, 90 total
python causal_generator.py --dry-run  # print config, generate nothing
python validate_corpus.py             # sanity check: causal signal recoverable?
```

Config flags: `--n-sessions`, `--t-cycles`, `--n-agents`, `--lag`, `--seed`, `--out`.

## Output

```text
corpus/
  G1/session_000.npy ... session_029.npy   (each: float32, 11×4×4×12)
  G2/...  G3/...
  ground_truth.json      # edges + weights + lag (the S3 target)
  corpus_manifest.json   # full generation config
```

## Self-test result (seed=1234, lag=1)

`validate_corpus.py` measures pooled lagged Pearson r for true edges vs a
20-pair non-edge control:

| Graph | mean \|r\| true edges | mean \|r\| control | separation |
|-------|----------------------|--------------------|------------|
| G1 | 0.565 | 0.098 | +0.467 |
| G2 | 0.582 | 0.122 | +0.459 |
| G3 | 0.450 | 0.130 | +0.320 |
| **overall** | **0.532** | **0.117** | **+0.415** |

**Verdict: PASS** — the injected causal signal is cleanly recoverable. Every
edge (including each chain's first link) sits well above control. This is a
*generator* self-test, not the S3 causal test itself.

## Note

This is a synthetic generator self-test, not a substitute for S3. S3 will use a
proper causal-recovery method (Granger / transfer entropy / mutual information)
and report F1 against `ground_truth.json` with a pre-registered threshold.
