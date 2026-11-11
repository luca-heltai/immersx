#!/usr/bin/env python3
"""Generate vessel-fitted cube meshes at several refinement levels.

The meshes are written by the fitted_vessel_mesh application, which combines
GridGenerator::plate_with_a_hole() and GridGenerator::subdivided_cylinder().
The default geometry is the MetricFlowX elastodynamics manufactured solution:
the cube [-0.120685, 0.120685]^3 and a vessel of radius sqrt(A0/pi) along the
x axis.

Example:

    python3 scripts/generate_fitted_vessel_meshes.py \
      --executable build/fitted_vessel_mesh --refinements 0 1 2 \
      --output-dir build/test_output/fitted-vessel-meshes

Each level writes fitted_vessel_ref<N>.<format> and its JSON diagnostics.
A summary of all levels is written to summary.json.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--executable", default="fitted_vessel_mesh",
                        help="path to the fitted_vessel_mesh executable")
    parser.add_argument("--refinements", type=int, nargs="+", default=[0],
                        help="global refinement levels to generate")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("fitted-vessel-meshes"),
                        help="directory for meshes and diagnostics")
    parser.add_argument("--format", default="msh",
                        choices=["msh", "vtk", "vtu"],
                        help="mesh file format (default msh)")
    parser.add_argument("--half-width", type=float,
                        help="cube half width (application default 0.120685)")
    parser.add_argument("--radius", type=float,
                        help="vessel radius (application default sqrt(A0/pi))")
    parser.add_argument("--box-factor", type=float,
                        help="half width of the square around the hole, in R")
    parser.add_argument("--axial-cells", type=int,
                        help="even number of coarse cells along the axis")
    parser.add_argument("--vessel-material-id", type=int,
                        help="material id inside the vessel (default 1)")
    return parser.parse_args()


def resolve_executable(name):
    path = Path(name)
    if path.is_file():
        return str(path.resolve())
    found = shutil.which(name)
    if found is None:
        sys.exit(f"error: cannot find executable {name}")
    return found


def main():
    args = parse_arguments()
    if any(level < 0 for level in args.refinements):
        sys.exit("error: refinement levels must be nonnegative")
    executable = resolve_executable(args.executable)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = []
    for option in ("half_width", "radius", "box_factor", "axial_cells",
                   "vessel_material_id"):
        value = getattr(args, option)
        if value is not None:
            geometry += ["--" + option.replace("_", "-"), str(value)]

    summary = []
    for level in sorted(set(args.refinements)):
        mesh = output_dir / f"fitted_vessel_ref{level}.{args.format}"
        diagnostics = output_dir / f"fitted_vessel_ref{level}.json"
        command = [executable, "--refinement", str(level),
                   "--output", str(mesh), "--summary", str(diagnostics)]
        command += geometry
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        with diagnostics.open() as stream:
            record = json.load(stream)
        summary.append(record)
        print(f"refinement {level}: {record['cells']} cells, "
              f"{record['q1_dofs']} Q1 displacement DoFs, "
              f"wall cell diameter/R = "
              f"{record['max_wall_cell_diameter_over_radius']:.4g}, "
              f"vessel volume error = "
              f"{100 * record['vessel_volume_relative_error']:+.4f}% "
              f"-> {mesh}")

    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
