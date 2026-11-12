#!/usr/bin/env python3
"""Plot the bundled solver records without rerunning the diagnostic.

The CSVs contain measured solver quantities, not a physical error estimate.
Temporal counters were extracted from six output-statistics log records;
they exclude the separate affine continuation. SHA-256 values in summary.json
identify both the original source records and these reduced plotting tables.
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
    parser.add_argument("--png", action="store_true", help="Also save inspection PNGs")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    summary = json.loads((source / "summary.json").read_text())
    for name, expected in summary["plot_data_sha256"].items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Plot data SHA-256 mismatch: {name}")
    with (source / "continuation.csv").open() as stream:
        stages = list(csv.DictReader(stream))
    with (source / "ida-output.csv").open() as stream:
        outputs = list(csv.DictReader(stream))
    assert len(stages) == 11 and float(stages[-1]["load"]) == 1.0
    assert len(outputs) == 6 and int(outputs[-1]["steps"]) == 10
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "liver-diagnostic64-startup"})

    def save(figure, name):
        svg = args.output_dir / f"{name}.svg"
        figure.savefig(svg, metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        if args.png:
            figure.savefig(args.output_dir / f"{name}.png", dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    load = [float(row["load"]) for row in stages]
    axis.semilogy(load, [float(row["stage_correction_wrms"]) for row in stages],
                  "o-", color="#0072B2", label="Artificial-stage correction")
    axis.semilogy(load, [float(row["original_correction_wrms"]) for row in stages],
                  "s--", color="#D55E00", label="Original-equation correction")
    axis.set(xlabel="Affine load s", ylabel="Reported correction norm (WRMS)",
             title="Diagnostic64: static continuation at fixed time")
    axis.set_xlim(0, 1.02)
    axis.grid(which="both", alpha=0.18)
    axis.legend(loc="upper right", frameon=False)
    save(figure, "continuation-correction")

    figure, axis = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    initial = float(outputs[0]["output_time"])
    elapsed = [1000 * (float(row["output_time"]) - initial) for row in outputs]
    curves = [("steps", "Accepted steps", "#000000", "o"),
              ("newton_iterations", "IDA Newton iterations", "#0072B2", "s"),
              ("jacobian_setups", "Jacobian setups", "#009E73", "^"),
              ("linear_solves", "Linear solves", "#D55E00", "D")]
    for key, label, color, marker in curves:
        axis.plot(elapsed, [int(row[key]) for row in outputs],
                  marker=marker, color=color, label=label)
    axis.set(xlabel="Simulated time after initialization (ms)",
             ylabel="Cumulative IDA count", title="Diagnostic64: accepted 50 ms trajectory")
    axis.set_xticks([0, 10, 20, 30, 40, 50])
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.18)
    axis.legend(loc="upper left", frameon=False)
    save(figure, "accepted-ida-counts")


if __name__ == "__main__":
    main()
