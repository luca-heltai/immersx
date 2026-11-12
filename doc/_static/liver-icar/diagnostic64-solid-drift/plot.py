#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0
"""Plot four normalized solid-drift aggregates without loading field samples.

The Euclidean norms use saved Float32 DataOut samples, including repeated
cell vertices. They are neither physical FE norms nor solver-error estimates.
The waveform ratio concatenates all 80 phase pairs before taking norms.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--png", action="store_true", help="Also save an inspection PNG")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    summary = json.loads((source / "summary.json").read_text())
    for name, expected in summary["plot_data_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Plot data SHA-256 mismatch: {name}")
    with (source / "data.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["comparison"] for row in rows] == ["endpoint", "waveform80"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "liver-diagnostic64-solid-drift"})
    figure, axis = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    for offset, field, label, color in [
            (-0.19, "displacement", "Displacement", "#0072B2"),
            (0.19, "velocity", "Velocity", "#D55E00")]:
        values = [100 * float(row[f"{field}_relative_sample_l2"]) for row in rows]
        bars = axis.bar([i + offset for i in range(2)], values, width=0.34,
                        label=label, color=color)
        axis.bar_label(bars, labels=[f"{value:.4g}%" for value in values], padding=4)
    axis.set_xticks([0, 1], ["Cycle-end state\n14.4 to 15.2 s",
                            "Paired waveforms\n80 saved phase pairs"])
    axis.set(ylabel="Relative change of saved samples (%)", ylim=(0, 35),
             title="Diagnostic64: solid change from cycle 9 to 10")
    axis.set_axisbelow(True)
    axis.yaxis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="upper right")
    svg = args.output_dir / "solid-drift.svg"
    figure.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    if args.png:
        figure.savefig(args.output_dir / "solid-drift.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
