#!/usr/bin/env python3
"""Generate a separated random point cloud for the 0D tutorials.

The output is an unstructured VTK mesh containing one VTK_VERTEX cell per
particle.  ``radius`` and ``rhs`` are stored as point-data arrays so they can
be selected by the ReducedPoisson ``Input file fields`` parameter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def generate_particles(
    n_particles: int,
    dimensions: int,
    min_coord: float,
    max_coord: float,
    min_radius: float,
    max_radius: float,
    min_rhs: float,
    max_rhs: float,
    clearance: float,
    max_attempts: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate non-overlapping particles with a boundary margin."""

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if not max_coord > min_coord:
        raise ValueError("max_coord must be greater than min_coord")
    if not (0.0 < min_radius <= max_radius):
        raise ValueError("require 0 < min_radius <= max_radius")
    if min_rhs > max_rhs:
        raise ValueError("require min_rhs <= max_rhs")
    if clearance < 0.0:
        raise ValueError("clearance must be non-negative")
    if max_radius + clearance >= 0.5 * (max_coord - min_coord):
        raise ValueError("radius and clearance leave no room inside the box")

    rng = np.random.default_rng(seed)
    lower = min_coord + max_radius + clearance
    upper = max_coord - max_radius - clearance
    points: list[np.ndarray] = []
    radii: list[float] = []

    for particle in range(n_particles):
        for _ in range(max_attempts):
            radius = float(rng.uniform(min_radius, max_radius))
            point = rng.uniform(lower, upper, dimensions)
            if all(
                np.linalg.norm(point - other_point)
                >= radius + other_radius + clearance
                for other_point, other_radius in zip(points, radii)
            ):
                points.append(point)
                radii.append(radius)
                break
        else:
            raise RuntimeError(
                f"Could not place particle {particle + 1}/{n_particles}; "
                "reduce N, the radii, or the clearance."
            )

    point_array = np.asarray(points, dtype=float)
    radius_array = np.asarray(radii, dtype=float)
    rhs_array = rng.uniform(min_rhs, max_rhs, n_particles)
    return point_array, radius_array, rhs_array


def save_particles(
    filename: Path,
    points: np.ndarray,
    radii: np.ndarray,
    rhs: np.ndarray,
) -> None:
    """Save points and scalar fields as a VTK unstructured grid."""

    suffix = filename.suffix.lower()
    if suffix not in {".vtk", ".vtu"}:
        raise ValueError("output filename must end in .vtk or .vtu")

    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N, 2) or (N, 3)")
    vtk_points = np.pad(
        points,
        ((0, 0), (0, 3 - points.shape[1])),
        mode="constant",
    )
    cloud = pv.PolyData(vtk_points)
    cloud.point_data["radius"] = radii
    cloud.point_data["rhs"] = rhs
    mesh = cloud.cast_to_unstructured_grid()
    filename.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(filename))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", "--n", type=int, default=16)
    parser.add_argument("--dimensions", type=int, choices=(2, 3), required=True)
    parser.add_argument("--min-coord", type=float, default=0.0)
    parser.add_argument("--max-coord", type=float, default=1.0)
    parser.add_argument("--min-radius", type=float, default=0.015)
    parser.add_argument("--max-radius", type=float, default=0.03)
    parser.add_argument("--min-rhs", type=float, default=0.25)
    parser.add_argument("--max-rhs", type=float, default=1.0)
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.01,
        help="extra distance required between particles and box boundaries",
    )
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    points, radii, rhs = generate_particles(
        n_particles=args.num_particles,
        dimensions=args.dimensions,
        min_coord=args.min_coord,
        max_coord=args.max_coord,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_rhs=args.min_rhs,
        max_rhs=args.max_rhs,
        clearance=args.clearance,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )
    save_particles(args.output_file, points, radii, rhs)
    print(f"Generated {len(points)} particles in {args.output_file}")


if __name__ == "__main__":
    main()
