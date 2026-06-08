# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""
S1 — Synthetic pipeline corpus generator with KNOWN causal ground truth.

Purpose
-------
S2 (Tucker composition) and S3 (causal conservation test) both need a corpus of
session tensors T^(s) whose underlying causal structure between quality
dimensions is *known by construction*. This generator produces that corpus.

Where L2's `fault_injector.py` transforms *artifacts* (source code) at the
artifact level, this generator works one layer up: it synthesizes the
*quality-vector trajectories* a pipeline would produce, with explicit causal
edges between the 11 quality dimensions and a controlled time lag. The set of
injected edges is the ground truth that M(V) must later recover (S3).

Model
-----
A session is a trajectory of quality vectors V[t] ∈ [0,1]^11 over t = 0..T_cyc-1
cycles. Each session is generated under one causal graph G:

    v_j[t] = clip( baseline_j
                   + drift_j * t
                   + Σ_{(i→j) ∈ G} w_{ij} * (v_i[t-lag] - baseline_i) * sign
                   + noise )

i.e. a parent dimension's *deviation from its baseline* at time t-lag propagates,
scaled by edge weight w_{ij}, into the child dimension at time t. This makes the
causal edge a lagged linear influence — recoverable by Granger / transfer-entropy
tests in S3, and compressible by a multilinear operator (Tucker) in S2.

The trajectory is then lifted to a tensor T^(s) ∈ ℝ^(11 × s × a × t) by
distributing the per-cycle vector across stages/agents with small stage- and
agent-specific perturbations (so the tensor is not rank-1 trivial).

Causal graphs (ground truth)
----------------------------
G1 (security cascade):
    security_risk → testability → maintainability
G2 (technical-debt cascade):
    technical_debt → maintainability → architectural_alignment
G3 (observability cascade):
    observability_coverage → performance → confidence

These mirror the L2 scenarios (S1 security, S3 debt, S4 observability) but are
expressed as dimension-level causal edges rather than artifact faults.

Output
------
corpus/
  G1/session_000.npy ... session_029.npy        (each: T^(s), shape 11×4×a×t)
  G2/...
  G3/...
  ground_truth.json   (edges per graph, weights, lag, generation config)
  corpus_manifest.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# ─── Quality dimensions (must match L2 EvaluationVector order) ────────────────

DIM_NAMES = [
    "functional_correctness",   # 0
    "architectural_alignment",  # 1
    "scalability_projection",   # 2
    "security_risk",            # 3  (inverted: 1 = no risk)
    "observability_coverage",   # 4
    "testability",              # 5
    "maintainability",          # 6
    "technical_debt",           # 7  (inverted: 1 = no debt)
    "performance",              # 8
    "confidence",               # 9
    "anomaly_score",            # 10 (inverted: 1 = no anomaly)
]
DIM_IDX = {name: i for i, name in enumerate(DIM_NAMES)}
N_DIMS = len(DIM_NAMES)

STAGES = ["design", "build", "test", "deploy"]
N_STAGES = len(STAGES)


# ─── Causal graph definition ─────────────────────────────────────────────────


@dataclass
class CausalEdge:
    src: str          # parent dimension name
    dst: str          # child dimension name
    weight: float     # influence strength (deviation propagation coefficient)
    sign: int = 1     # +1: parent drop pulls child down; -1: parent drop pushes child up


@dataclass
class CausalGraph:
    graph_id: str
    description: str
    edges: list[CausalEdge]
    # The dimension whose baseline is perturbed to seed the cascade (the "shock").
    shock_dim: str = ""
    shock_strength: float = 0.45  # how far below baseline the shock pushes shock_dim

    def edge_index_list(self) -> list[tuple[int, int]]:
        return [(DIM_IDX[e.src], DIM_IDX[e.dst]) for e in self.edges]


def default_graphs() -> dict[str, CausalGraph]:
    """The three pre-registered ground-truth causal graphs."""
    return {
        "G1": CausalGraph(
            graph_id="G1",
            description="security cascade: security_risk -> testability -> maintainability",
            shock_dim="security_risk",
            edges=[
                CausalEdge("security_risk", "testability", weight=0.65),
                CausalEdge("testability", "maintainability", weight=0.55),
            ],
        ),
        "G2": CausalGraph(
            graph_id="G2",
            description="debt cascade: technical_debt -> maintainability -> architectural_alignment",
            shock_dim="technical_debt",
            edges=[
                CausalEdge("technical_debt", "maintainability", weight=0.60),
                CausalEdge("maintainability", "architectural_alignment", weight=0.50),
            ],
        ),
        "G3": CausalGraph(
            graph_id="G3",
            description="observability cascade: observability_coverage -> performance -> confidence",
            shock_dim="observability_coverage",
            edges=[
                CausalEdge("observability_coverage", "performance", weight=0.58),
                CausalEdge("performance", "confidence", weight=0.62),
            ],
        ),
    }


# ─── Generation config ───────────────────────────────────────────────────────


@dataclass
class GenConfig:
    n_sessions: int = 30        # sessions per causal graph
    t_cycles: int = 12          # cycles per session (the t axis of T)
    n_agents: int = 4           # a axis of T
    lag: int = 1                # causal time lag (cycles)
    noise_std: float = 0.03     # per-cell gaussian noise
    drift_std: float = 0.01     # per-session random linear drift magnitude
    baseline_low: float = 0.70  # clean-pipeline baseline range
    baseline_high: float = 0.90
    stage_perturb: float = 0.04 # stage-specific offset magnitude in the tensor lift
    agent_perturb: float = 0.04 # agent-specific offset magnitude in the tensor lift
    # Shock-dimension AR(1) recovery rate bounds (S1 defaults). For longer
    # sessions these are scaled a priori by the rule new = old / (t_cycles/12)
    # via GenConfig.scaled_for_length() — see PRE_REGISTRATION_S1bis.md.
    recover_rate_low: float = 0.04
    recover_rate_high: float = 0.12
    seed: int = 1234

    @classmethod
    def scaled_for_length(cls, t_cycles: int, seed: int, n_sessions: int = 30) -> "GenConfig":
        """
        Build a config for a target session length, scaling the length-sensitive
        parameters (recover_rate, drift_std) by the a-priori rule
        factor = t_cycles / 12, so accumulated drift and shock-recovery fraction
        match the S1 (t_cycles=12) corpus. All other parameters are S1 defaults.
        """
        factor = t_cycles / 12.0
        base = cls()  # S1 defaults
        return cls(
            n_sessions=n_sessions,
            t_cycles=t_cycles,
            drift_std=base.drift_std / factor,
            recover_rate_low=base.recover_rate_low / factor,
            recover_rate_high=base.recover_rate_high / factor,
            seed=seed,
        )


# ─── Trajectory generation ───────────────────────────────────────────────────


def generate_trajectory(graph: CausalGraph, cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one session's quality-vector trajectory V[t] ∈ [0,1]^(t_cycles × 11).

    Baselines are sampled per session; the shock dimension is driven below its
    baseline at t=0 and the deviation propagates along the causal edges.
    """
    T_cyc = cfg.t_cycles
    baselines = rng.uniform(cfg.baseline_low, cfg.baseline_high, size=N_DIMS)
    drift = rng.normal(0.0, cfg.drift_std, size=N_DIMS)

    V = np.zeros((T_cyc, N_DIMS))
    # Initialise at baseline + small noise.
    V[0] = baselines + rng.normal(0.0, cfg.noise_std, size=N_DIMS)

    # Apply the shock to the cascade-seed dimension at t=0.
    shock_i = DIM_IDX[graph.shock_dim]
    shock = graph.shock_strength

    # The shock dimension must have genuine temporal VARIANCE, otherwise its
    # lagged correlation with its child is pure noise (a flat signal carries no
    # information to propagate). We model it as a depressed level that recovers
    # toward baseline at a per-session random rate, with its own random walk.
    recover_rate = rng.uniform(cfg.recover_rate_low, cfg.recover_rate_high)  # per-cycle recovery fraction
    walk_std = cfg.noise_std * 2.5                   # extra volatility on the shock dim
    shock_level = np.zeros(T_cyc)
    shock_level[0] = baselines[shock_i] - shock
    for t in range(1, T_cyc):
        # AR(1)-style recovery toward baseline + random walk -> real variance.
        gap = baselines[shock_i] - shock_level[t - 1]
        shock_level[t] = shock_level[t - 1] + recover_rate * gap + rng.normal(0.0, walk_std)
    shock_level = np.clip(shock_level, 0.0, 1.0)
    V[0, shock_i] = shock_level[0]

    edges = graph.edges
    for t in range(1, T_cyc):
        # Start from baseline + drift + noise.
        V[t] = baselines + drift * t + rng.normal(0.0, cfg.noise_std, size=N_DIMS)
        # Shock dimension follows its own time-varying recovery trajectory.
        V[t, shock_i] = shock_level[t]
        # Propagate parent deviations (at t-lag) into children.
        src_t = t - cfg.lag
        if src_t >= 0:
            for e in edges:
                i, j = DIM_IDX[e.src], DIM_IDX[e.dst]
                parent_dev = V[src_t, i] - baselines[i]
                V[t, j] += e.sign * e.weight * parent_dev

    return np.clip(V, 0.0, 1.0)


def lift_to_tensor(V: np.ndarray, cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Lift a trajectory V[t, 11] to a tensor T^(s) ∈ ℝ^(11 × n_stages × n_agents × t).

    Each (stage, agent) cell is the per-cycle vector plus a small stage- and
    agent-specific offset, so the tensor carries non-trivial stage/agent
    structure (not a rank-1 broadcast) while preserving the causal trajectory.
    """
    T_cyc = V.shape[0]
    stage_off = rng.normal(0.0, cfg.stage_perturb, size=(N_STAGES, N_DIMS))
    agent_off = rng.normal(0.0, cfg.agent_perturb, size=(cfg.n_agents, N_DIMS))

    T = np.zeros((N_DIMS, N_STAGES, cfg.n_agents, T_cyc))
    for s in range(N_STAGES):
        for a in range(cfg.n_agents):
            # broadcast: per-cycle vector + stage offset + agent offset
            cell = V + stage_off[s] + agent_off[a]  # (T_cyc, N_DIMS)
            T[:, s, a, :] = np.clip(cell, 0.0, 1.0).T
    return T


# ─── Corpus generation ───────────────────────────────────────────────────────


def generate_corpus(graphs: dict[str, CausalGraph], cfg: GenConfig, out_dir: Path) -> dict:
    """Generate the full corpus and write it to disk. Returns the manifest."""
    rng = np.random.default_rng(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "S1 — Synthetic causal corpus",
        "config": asdict(cfg),
        "tensor_shape": [N_DIMS, N_STAGES, cfg.n_agents, cfg.t_cycles],
        "dim_names": DIM_NAMES,
        "stages": STAGES,
        "graphs": {},
    }

    for gid, graph in graphs.items():
        gdir = out_dir / gid
        gdir.mkdir(parents=True, exist_ok=True)
        for s in range(cfg.n_sessions):
            V = generate_trajectory(graph, cfg, rng)
            T = lift_to_tensor(V, cfg, rng)
            np.save(gdir / f"session_{s:03d}.npy", T.astype(np.float32))
        manifest["graphs"][gid] = {
            "description": graph.description,
            "shock_dim": graph.shock_dim,
            "shock_strength": graph.shock_strength,
            "n_sessions": cfg.n_sessions,
            "edges": [asdict(e) for e in graph.edges],
            "edge_index": graph.edge_index_list(),
        }
        print(f"  {gid}: {cfg.n_sessions} sessions  ({graph.description})")

    # Ground truth (the thing S3 must recover).
    ground_truth = {
        gid: {
            "edges": [{"src": e.src, "dst": e.dst, "weight": e.weight, "sign": e.sign}
                      for e in g.edges],
            "edge_index": g.edge_index_list(),
            "shock_dim": g.shock_dim,
        }
        for gid, g in graphs.items()
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))
    (out_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))

    return manifest


def summarize(manifest: dict, out_dir: Path) -> None:
    cfg = manifest["config"]
    total = sum(g["n_sessions"] for g in manifest["graphs"].values())
    print()
    print("=" * 65)
    print("S1 — SYNTHETIC CAUSAL CORPUS")
    print("=" * 65)
    print(f"  Graphs:          {len(manifest['graphs'])}")
    print(f"  Sessions/graph:  {cfg['n_sessions']}")
    print(f"  Total sessions:  {total}")
    print(f"  Tensor shape:    {manifest['tensor_shape']}  (dim x stage x agent x cycle)")
    print(f"  Causal lag:      {cfg['lag']} cycle(s)")
    print(f"  Seed:            {cfg['seed']}")
    print()
    print("  Ground-truth causal edges:")
    for gid, g in manifest["graphs"].items():
        chain = "  ".join(f"{e['src']}->{e['dst']}(w={e['weight']})" for e in g["edges"])
        print(f"    {gid}: {chain}")
    print()
    print(f"  Output: {out_dir}")
    print("=" * 65)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="S1 — synthetic causal corpus generator")
    parser.add_argument("--n-sessions", type=int, default=30, help="sessions per causal graph")
    parser.add_argument("--t-cycles", type=int, default=12, help="cycles per session")
    parser.add_argument("--n-agents", type=int, default=4, help="agents per session")
    parser.add_argument("--lag", type=int, default=1, help="causal time lag (cycles)")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "corpus"),
        help="output directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="print config and exit")
    parser.add_argument(
        "--scaled-length",
        action="store_true",
        help="S1-bis mode: scale recover_rate and drift_std a priori by "
             "factor=(t_cycles/12) via GenConfig.scaled_for_length (see "
             "PRE_REGISTRATION_S1bis.md). Use with --t-cycles 48 --seed 5678.",
    )
    args = parser.parse_args()

    if args.scaled_length:
        # S1-bis: a-priori length-scaled config. Only n_sessions/t_cycles/seed
        # are taken from CLI; the length-sensitive params follow the fixed rule.
        cfg = GenConfig.scaled_for_length(
            t_cycles=args.t_cycles, seed=args.seed, n_sessions=args.n_sessions,
        )
        title = "S1-bis — length-scaled synthetic causal corpus"
    else:
        cfg = GenConfig(
            n_sessions=args.n_sessions,
            t_cycles=args.t_cycles,
            n_agents=args.n_agents,
            lag=args.lag,
            seed=args.seed,
        )
        title = "S1 — Synthetic causal corpus generator"
    graphs = default_graphs()
    out_dir = Path(args.out)

    print(title)
    print(f"  Output: {out_dir}")
    print()

    if args.dry_run:
        print("DRY RUN — config:")
        print(json.dumps(asdict(cfg), indent=2))
        print()
        for gid, g in graphs.items():
            print(f"  {gid}: {g.description}")
        return 0

    print("Generating corpus...")
    manifest = generate_corpus(graphs, cfg, out_dir)
    summarize(manifest, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
