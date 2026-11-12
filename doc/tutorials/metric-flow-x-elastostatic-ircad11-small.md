# IRCAD11 small with three grown vascular networks

This example reuses the existing **3D-IRCADb-1 case-11 small mesh and its
completed arterial, portal and hepatic venous networks**. It downloads the
accepted physical archive; it does not rerun vascular growth or meshing.

The tissue has 4,804 tetrahedra. Each network has 4,804 paired tissue terminals,
connected through 4,804 saved microvascular triplets. This is one terminal per
background cell **globally for each role**, not a guarantee that each
tetrahedron contains one terminal. The archive also retains 53 unpaired
anatomical ends, which the coupled input explicitly seals.

## Download the existing assets

Choose an absolute build directory outside the source checkout and configure
the application with a MUMPS-enabled deal.II and a MetricFlowX package supporting
native NetworkPhysics input:

~~~bash
build_dir=/absolute/path/to/immersx-ircad11-small-build
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/dealii \
  -DMetricFlowX_DIR=/path/to/MetricFlowX/lib/cmake/MetricFlowX \
  -DENABLE_COUPLED_PROBLEMS=OFF
cmake --build "$build_dir" --target metric_flow_x_elastostatic -j2
python3 scripts/prepare_ircad11_small_elastostatic.py \
  --output "$build_dir/data/ircad11-small"
~~~

The download script uses only the Python standard library. It retrieves
[the existing small mesh and grown networks](https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/ircad11-small-grown-networks-4804-20261112-v1.tar.gz),
checks its pinned SHA-256 and every packaged file checksum, and independently
checks the CSV terminal, triplet and vessel counts before publishing the new
output directory. The asset SHA-256 is
20a8056214b565af988828025259334ea26fa6ec8504aea7d6719112f2456992.

The output contains `small.msh`, the native `network/` archive, `audit/` CSV tables,
`manifest.json`, `SHA256SUMS`, `README.md` and `LICENSE-IRCAD.txt`. Existing directories
are not overwritten. To reuse a previous download, pass
`--archive /path/to/ircad11-small-grown-networks-4804-20261112-v1.tar.gz`;
the same verification applies. Gmsh, VTK, VMTK and growth scripts are not
needed for this workflow.

## What is retained

| Quantity | Arterial | Hepatic venous | Portal |
| --- | ---: | ---: | ---: |
| Paired tissue terminals | 4,804 | 4,804 | 4,804 |
| Additional sealed anatomical terminals | 0 | 38 | 15 |
| Physical vessels | 9,608 | 10,495 | 9,908 |

The networks have 30,011 physical vessels in total. The native reader preserves
their identifiers, endpoint radii, vessel properties and saved microvascular
connections. Numerical intervals are rebuilt on the loaded small background
mesh; the physical networks are not regenerated.

The arterial input began with the case-11 modeled feeder, while the venous and
portal input trees were extracted from the anatomical data. The completed
networks include the subsequent accepted growth. Source parameters, preparation
versions and the independent audits are available in the
[published case-11 provenance snapshot](https://github.com/luca-heltai/immersx/releases/download/liver-h-scaled-2026-11-10/ircad-case11-reproducibility-2026-11-11-v2.zip).
It is a separate supplement; the network asset above supplies the physical
files that snapshot deliberately omitted.

The small mesh uses the original 12 mm target edge length, boundary ID 1, and
coordinates in metres. It is byte-identical to the original small mesh, whose
surface preparation converted the case-11 liver from millimetres to metres.
The downloaded native archive is already in metres and is not rescaled.

The [IRCAD source page](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/)
describes the anatomical dataset. Credit Luc Soler and coauthors, *3D image
reconstruction for comparison of algorithm database: A patient specific
anatomical and medical image database*, IRCAD technical report (2010).
The supplied license is CC BY-NC-ND 4.0; retain `LICENSE-IRCAD.txt`, attribution
and provenance with the data.

Growth used disabled geometric admissibility checks. The retained
topology/hydraulic audits and global terminal count do not certify vessel-wall
containment, inf-sup compatibility, or coupled transient convergence. No
anatomical or physiological calibration is claimed.

## Static coupling and boundary data

The generated parameter file uses the
[elastostatic application](metric-flow-x-elastostatic-coupling.md) with pressure
feedback enabled. Solid density, viscosity and both Rayleigh coefficients are
zero; the positive Robin coefficient 100000 on boundary 1 removes rigid motion.
Its Lamé parameters are 2013.4228187919464 and 98657.7181208053 Pa, as in the
earlier small profile. The initial body load and native flow state are at rest.

Archived wall parameters use the existing role order **arterial, venous,
portal**: Young moduli 400000 Pa, thickness/radius ratios 0.1, and reference
pressures 10064.536060032478, 629.59739556 and 1066.57909932 Pa. Portal and venous
root reservoirs use the corresponding reference pressures and the existing
root resistance 100000. Tissue leaves use the archive's saved triplet
resistances; independent RCR drains are not substituted for these connections.
The 53 unpaired anatomical terminal IDs are listed explicitly in the parameter
file and in the asset manifest.

For a short startup, arterial inflow starts smoothly with zero value and
derivative, at the existing illustrative arterial scale
6.08461327615316e-10 m³/s. This is not the old dynamic profile's MOX waveform
replay at 7.2 seconds. The static problem has no independent solid velocity or
inertia. Materials, reservoirs and flow scale are modelling assumptions, not
case-11 patient measurements.

## Run and inspect

CMake generates a parameter file with absolute build-tree input/output paths.
Use that generated file:

~~~bash
mpiexec -np 2 "$build_dir/metric_flow_x_elastostatic" \
  "$build_dir/tutorials/metric_flow_x_elastostatic/ircad11_small.prm"
~~~

For a setup check without time integration, copy the generated input and
append the existing setup control:

~~~bash
cp "$build_dir/tutorials/metric_flow_x_elastostatic/ircad11_small.prm" \
  "$build_dir/tutorials/metric_flow_x_elastostatic/ircad11_small_setup.prm"
cat >> "$build_dir/tutorials/metric_flow_x_elastostatic/ircad11_small_setup.prm" <<'EOF'
subsection MetricFlowX coupling
  set Setup only = true
end
EOF
mpiexec -np 2 "$build_dir/metric_flow_x_elastostatic" \
  "$build_dir/tutorials/metric_flow_x_elastostatic/ircad11_small_setup.prm"
~~~

Setup checks archive loading, redistribution, discretization and coupled spaces.
It does not execute IDA or demonstrate an accepted timestep.

Co-refinement is disabled for this fixed tetrahedral asset. The ordinary
archived-network discretization controls rebuild numerical vessel intervals
without changing the physical graph or requesting any growth.

The flow block uses MUMPS within the iterative coupled solver. Native componentwise
absolute tolerances scale area by its local reference area (fraction 1e-3),
velocity by 1e-7 m/s and capacitor pressure by 1 Pa. The area fraction and
capacitor scale retain the SDK defaults; the selected velocity scale is below
the archive's median rigid-reference velocity (about 2.5e-5 m/s). The common
solid tolerance remains 1e-9, relative
tolerance 1e-7, and multiplier absolute tolerance 1e-7 Pa. This records the
workflow settings, not a temporal error or physical accuracy certificate. Initial
correction uses `use_y_diff`, and saved output times are internal IDA stop times.
The short requested interval is 0 to 0.002 seconds, with output every 0.001
seconds; it is not a validated long-time or periodic simulation.

Open the collections under
`$build_dir/test_output/metric-flow-x-elastostatic-ircad11-small/`:

- `solid.pvd`: solid displacement;
- `flow.pvd`: native flow and pressure;
- `pressure_jump.pvd`: scalar multiplier;
- `wall_particles.pvd`: selected wall properties and sampled solid displacement.

The dense three-network case is substantially larger than the single-vessel
walkthrough. Record actual setup sizes, accepted steps and solver diagnostics;
a successful download or setup alone is not evidence of an accepted transient.
