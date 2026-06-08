# SPDX-License-Identifier: AGPL-3.0
# Copyright (C) 2026 Juan Pablo Chancay
"""Plot the kappa(V) vs causal-F1 trade-off from S3 run 2 secondary analysis."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
res = json.loads((HERE / "results" / "s3_run2_pcmci_results.json").read_text())
sec = res["secondary_kappa_vs_f1"]

kappa = [r["kappa"] for r in sec]
f1 = [r["recon_micro_f1"] for r in sec]
r0 = [r["session_rank"] for r in sec]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(kappa, f1, "o-", color="darkorange", linewidth=2, markersize=8)
for k, f, r in zip(kappa, f1, r0):
    ax.annotate(f"r0={r}", (k, f), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.axhline(0.70, color="green", linestyle="--", linewidth=1, label="PASS threshold (0.70)")
ax.axhline(0.50, color="orange", linestyle="--", linewidth=1, label="PARTIAL threshold (0.50)")
ax.set_xlabel("kappa(V) — Tucker core size (lower = more compression)")
ax.set_ylabel("Causal recovery micro-F1 (PCMCI on reconstruction)")
ax.set_title("S3 Run 2 — kappa(V) vs causal-F1 trade-off\n"
             "compression that preserves 98% variance destroys causal structure")
ax.legend(fontsize=8)
ax.set_ylim(0, 0.8)
fig.tight_layout()
out = HERE / "results" / "s3_kappa_vs_causalf1.png"
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
