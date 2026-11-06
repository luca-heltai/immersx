#!/usr/bin/env python3
# Copyright (C) 2026 by Luca Heltai
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Render the liver tutorial figures from checksummed scalar snapshots."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def column(rows, name):
    values = np.asarray([float(row[name]) for row in rows])
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite input column: {name}")
    return values


def decorate(fig, title, subtitle, notes):
    fig.suptitle(title, x=.075, y=.975, ha="left", fontsize=16, weight="semibold")
    fig.text(.075, .919, subtitle, fontsize=10.5, color="#505050")
    for i, note in enumerate(notes):
        fig.text(.075, .06 - .027 * i, note, fontsize=9, color="#404040")
    for axis in fig.axes:
        axis.grid(color="#E7E7E7", linewidth=.7)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=9)


def scale_profiles(data):
    rows = data["profiles"]
    if [row["profile"] for row in rows] != ["small", "medium", "large"]:
        raise ValueError("Expected the ordered small, medium, large profiles")
    fig, (counts, ratios) = plt.subplots(1, 2, figsize=(11.8, 5.6), sharey=True)
    fig.subplots_adjust(left=.14, right=.96, bottom=.23, top=.79, wspace=.38)
    ys = [2, 1, 0]
    counts.set_xscale("log")
    counts.set_xlim(8e4, 6e7)
    counts.set_xticks([1e5, 1e6, 1e7], [r"$10^5$", r"$10^6$", r"$10^7$"])
    counts.minorticks_off()
    counts.set_yticks(ys, [row["profile"].capitalize() + "\n" + str(row["numerical_1d_intervals"]) + " line intervals" for row in rows])
    counts.tick_params(axis="y", length=0, pad=12)
    for row, y, color in zip(rows, ys, [BLUE, GREEN, ORANGE]):
        n = row["coupled_total_dofs"]
        lo = float(row["minimum_length_over_h_decimal"])
        hi = float(row["maximum_length_over_h_decimal"])
        if not (n > 0 and np.isfinite(lo) and np.isfinite(hi) and 0 < lo <= hi):
            raise ValueError("Invalid DoF count or observed geometric range")
        counts.scatter(n, y, s=65, color=color, zorder=3)
        counts.annotate(f"{n:,}", (n, y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=11)
        ratios.hlines(y, lo, hi, color=color, linewidth=3)
        ratios.vlines([lo, hi], y - .09, y + .09, color=color, linewidth=2)
        ratios.text((lo + hi) / 2, y + .19, f"{lo:.6f}–{hi:.6f}", ha="center", color=color, fontsize=10)
    for axis in (counts, ratios):
        axis.set_ylim(-.5, 2.6)
        axis.spines["left"].set_visible(False)
    ratios.set_xlim(.45, 2.15)
    ratios.set_xticks([.5, 1, 1.5, 2])
    ratios.tick_params(axis="y", left=False, labelleft=False)
    ratios.axvline(1, color="#666666", linestyle="--", linewidth=1)
    ratios.text(1.025, 2.44, "1:1 size reference", fontsize=9, color="#606060")
    counts.set_title("(a) Measured coupled DoFs", loc="left", fontsize=12, pad=16)
    ratios.set_title("(b) Observed minimum–maximum", loc="left", fontsize=12, pad=16)
    counts.set_xlabel("Total degrees of freedom (log scale)", labelpad=10)
    ratios.set_xlabel(r"Observed $\ell_{1D}/h_{3D}$ (dimensionless)", labelpad=10)
    decorate(fig, "Coupled liver discretization profiles", "P2 solid displacement/velocity; DG1 flow and multiplier", [
        "Three independent meshes; one CCO architecture: 14 physical vessel edges, 6 tissue leaves (2 triplets).",
        "Ranges are observed extrema. The size reference does not establish inf-sup stability or mesh convergence."])
    return fig


def haemodynamics(rows, prescribed, units):
    time = column(rows, "time_s")
    input_time = column(prescribed, "time_s")
    input_flow = column(prescribed, "flow_m3_s")
    # Source endpoint times differ from the requested output by < 1e-12 s.
    if time[0] < input_time[0] - 1e-12 or time[-1] > input_time[-1] + 1e-12:
        raise ValueError("Prescribed input does not cover the plotted times")
    imposed = np.interp(np.clip(time, input_time[0], input_time[-1]), input_time, input_flow)
    factor = units["flow_m3_s_to_ml_min"]
    fig = plt.figure(figsize=(11.8, 6.9))
    grid = fig.add_gridspec(2, 2, left=.09, right=.96, top=.82, bottom=.20, hspace=.48, wspace=.30)
    flow = fig.add_subplot(grid[:, 0])
    ha = fig.add_subplot(grid[0, 1])
    low = fig.add_subplot(grid[1, 1], sharex=ha)
    for role, sign, color, label in [("ha", -1, BLUE, "HA into liver"), ("pv", -1, GREEN, "PV into liver"), ("hv", 1, ORANGE, "HV out of liver")]:
        flow.plot(time, sign * column(rows, role + "_outward_flow_m3_s") * factor, color=color, label=label, linewidth=1.8)
    flow.plot(time, imposed * factor, color="#252525", linestyle=(0, (4, 3)), linewidth=1.2, label="Prescribed HA input")
    flow.set_title("(a) Root flows", loc="left", fontsize=12)
    flow.set_ylabel("Flow (mL/min)")
    flow.set_ylim(bottom=0)
    flow.legend(frameon=False, fontsize=9, loc="upper left")
    ha.plot(time, column(rows, "ha_pressure_Pa") / units["pressure_Pa_per_mmHg"], color=BLUE, linewidth=1.8)
    ha.set_title("(b) Computed HA root pressure", loc="left", fontsize=12)
    ha.set_ylabel("Pressure (mmHg)")
    for role, color, label in [("pv", GREEN, "PV"), ("hv", ORANGE, "HV")]:
        low.plot(time, column(rows, role + "_pressure_Pa") / units["pressure_Pa_per_mmHg"], color=color, label=label, linewidth=1.8)
    low.set_title("(c) Computed PV and HV root pressures", loc="left", fontsize=12)
    low.set_ylabel("Pressure (mmHg)")
    low.legend(frameon=False, ncol=2, fontsize=9)
    for axis in (flow, ha, low):
        axis.set_xlim(time[0], time[-1])
        axis.set_xlabel("Model time (s)")
    decorate(fig, "Small coupled liver: one complete saved cycle", "132,932 DoFs; P2/DG1; MPI16; source a92bee56", [
        "HA and PV are plotted inward; HV outward. Native boundary-normal signs are retained in the input CSV.",
        "Pressures are computed root values. This single cycle does not establish periodicity or physiological calibration."])
    return fig


def conservation(kirchhoff, volume, areas):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.9))
    fig.subplots_adjust(left=.075, right=.975, top=.78, bottom=.22, wspace=.35)
    beds = sorted({row["bed_id"] for row in kirchhoff})
    zero_count = 0
    for index, (bed, color) in enumerate(zip(beds, [BLUE, GREEN])):
        rows = [row for row in kirchhoff if row["bed_id"] == bed]
        ratio = np.abs(column(rows, "defect_m3_s")) / column(rows, "allowed_defect_m3_s")
        zero_count += int(np.count_nonzero(ratio == 0))
        axes[0].plot(column(rows, "time_s"), np.ma.masked_equal(ratio, 0), color=color, label=f"Triplet {index + 1}", linewidth=1.5)
    axes[0].legend(frameon=False, fontsize=9)
    ratio = np.abs(column(volume, "error")) / column(volume, "allowed_absolute_error")
    axes[1].plot(column(volume, "time_s"), np.ma.masked_equal(ratio, 0), color=ORANGE, linewidth=1.8)
    for axis in axes[:2]:
        axis.set_yscale("log")
        axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
        axis.set_ylabel("Absolute defect / original allowed defect")
        axis.set_ylim(top=2)
    axes[0].set_title("(a) Algebraic triplet KCL", loc="left", fontsize=11.5)
    axes[1].set_title("(b) Cumulative root-volume balance", loc="left", fontsize=11.5)
    time = column(areas, "time_s")
    axes[2].plot(time, column(areas, "area_minimum") * 1e6, color=BLUE, label=r"Minimum $A$", linewidth=1.8)
    axes[2].plot(time, column(areas, "A_hat_minimum") * 1e6, color=ORANGE, linestyle="--", label=r"Minimum $\hat{A}$", linewidth=1.4)
    axes[2].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[2].set_ylim(bottom=-.25)
    axes[2].set_title("(c) Saved endpoint areas", loc="left", fontsize=11.5)
    axes[2].set_ylabel(r"Area (mm$^2$)")
    axes[2].legend(frameon=False, fontsize=9, loc="center right")
    for axis in axes:
        axis.set_xlim(time[0], time[-1])
        axis.set_xlabel("Model time (s)")
    decorate(fig, "Small cycle: conservation and saved area positivity", "The dashed bounds reproduce the original audit policy; source a92bee56", [
        "Defect ratios use each original absolute + relative error budget; acceptance requires a ratio at most 1 and area > 0.",
        f"Volume uses trapezoidal saved-time quadrature. Exact-zero KCL samples ({zero_count}) are omitted on log axes; no between-time bound."])
    return fig


def solid_samples(rows):
    time = column(rows, "time_s")
    elapsed = (time - time[0]) * 1000
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8))
    fig.subplots_adjust(left=.09, right=.96, top=.79, bottom=.22, wspace=.30)
    for axis, key, color, title, label in [
        (axes[0], "displacement_max_exported_sample_norm_m", BLUE, "(a) Displacement", "Maximum exported sample norm (mm)"),
        (axes[1], "velocity_max_exported_sample_norm_m_s", ORANGE, "(b) Velocity", "Maximum exported sample norm (mm/s)")]:
        axis.plot(elapsed, column(rows, key) * 1000, color=color, marker="o", markersize=4, linewidth=1.8)
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_ylabel(label)
        axis.set_xlabel("Elapsed time since 7.2 s (ms)")
        axis.set_xlim(0, elapsed[-1])
        axis.set_xticks([0, 5, 10, 15, 20])
        axis.set_ylim(bottom=0)
    decorate(fig, "Medium coupled liver: exported solid samples", "992,584 DoFs; P2/DG1; MPI8; 11 saved times over 20 ms; runtime source b4d85cb6", [
        "Each value is the largest vector norm among exported nodal samples at that saved time.",
        "These are not exact P2 finite-element extrema or strain estimates; this window does not establish mesh convergence."])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--inputs", type=Path, default=here / "inputs")
    parser.add_argument("--output-dir", type=Path, default=here.parents[2] / "doc" / "_static" / "liver-icar")
    args = parser.parse_args()
    provenance = json.loads((args.inputs / "provenance.json").read_text())
    for name, expected in provenance["input_sha256"].items():
        if sha256(args.inputs / name) != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False, "axes.spines.right":False, "axes.edgecolor":"#777777", "text.color":"#202020", "svg.hashsalt":"immersx-liver-tutorial-v1", "svg.fonttype":"path", "savefig.facecolor":"white"})
    figures = {
        "scale-profiles": scale_profiles(json.loads((args.inputs / "scale-profiles.json").read_text())),
        "small-haemodynamics": haemodynamics(read_csv(args.inputs / "small-root-timeseries.csv"), read_csv(args.inputs / "ha-prescribed-input.csv"), provenance["units"]),
        "small-conservation": conservation(read_csv(args.inputs / "small-kirchhoff.csv"), read_csv(args.inputs / "small-cumulative-volume.csv"), read_csv(args.inputs / "small-saved-area-minima.csv")),
        "medium-solid-samples": solid_samples(read_csv(args.inputs / "medium-solid-samples.csv"))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, fig in figures.items():
        for extension in ("svg", "png"):
            path = args.output_dir / f"{name}.{extension}"
            metadata = {"Date":None, "Creator":"ImmersX liver plot_results.py"} if extension == "svg" else {"Software":"Matplotlib " + matplotlib.__version__}
            fig.savefig(path, dpi=220, bbox_inches="tight", metadata=metadata)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            outputs[path.name] = {"sha256":sha256(path), "size_bytes":path.stat().st_size}
        plt.close(fig)
    manifest = {"script_sha256":sha256(Path(__file__)), "provenance_sha256":sha256(args.inputs / "provenance.json"), "matplotlib_version":matplotlib.__version__, "numpy_version":np.__version__, "outputs":outputs}
    (args.output_dir / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
