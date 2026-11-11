# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Render the bundled rigid-reference CDFs without running a solver."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
provenance = json.loads((source / "provenance.json").read_text())
assert hashlib.sha256((source / "cdf.csv").read_bytes()).hexdigest() == provenance["cdf_sha256"]
with (source / "cdf.csv").open() as stream:
    rows = list(csv.DictReader(stream))
roles = [("arterial", "#c62828"), ("portal", "#238b45"), ("venous", "#204bb5")]
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "savefig.bbox": "tight", "svg.hashsalt": "liver-rigid-reference-v1"})
for metric, name in [("speed_m_s", "speed"), ("relative_perfusion", "perfusion")]:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, family in zip(axes, ["small", "medium"]):
        count = set()
        for role, color in roles:
            selected = [row for row in rows if row["family"] == family
                        and row["role"] == role and row["metric"] == metric]
            assert len(selected) == 1001
            x = [float(row["value"]) for row in selected]
            y = [float(row["cdf"]) for row in selected]
            assert all(a < b for a, b in zip(x, x[1:]))
            assert all(a <= b + 1e-14 for a, b in zip(y, y[1:]))
            assert min(y) >= 0 and max(y) <= 1 + 1e-12
            count.update(row["terminals_per_role"] for row in selected)
            ax.step(x, y, where="post", color=color, label=role, lw=1.6)
        assert len(count) == 1
        ax.set_title(f"{family}: {int(count.pop()):,} terminals per role")
        ax.grid(alpha=.22)
        ax.set_ylim(0, 1.01)
        if name == "speed":
            ax.set_xscale("log")
            ax.set_xlim(1e-5, 1.)
            ax.set_xlabel("Rigid-reference vessel speed (m/s)")
        else:
            ax.axvline(1, color="0.45", ls=":", lw=1)
            ax.set_xlim(0, 1.6)
            ax.set_xlabel("Terminal flow share / recorded demand share")
        ax.legend(loc="lower right", frameon=False)
    axes[0].set_ylabel("Fraction of physical vessels" if name == "speed"
                       else "Cumulative territory demand fraction")
    fig.suptitle("Rigid hydraulic reference; not an accepted coupled simulation", fontsize=11)
    fig.tight_layout()
    for extension in ["svg", "png"]:
        output = args.output_dir / f"{name}-cdf.{extension}"
        metadata = {"Date": None} if extension == "svg" else {}
        fig.savefig(output, dpi=160, metadata=metadata)
        if extension == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)
print("Rendered both rigid-reference figures from", len(rows), "CDF records")
