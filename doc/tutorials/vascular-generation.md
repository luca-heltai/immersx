# Generate vascular trees in uniform and locally refined domains

These examples run the C++ `vascular_network` application before any transient
simulation. They use synthetic SI data, not a calibrated organ model. The
[vascular generation guide](../how-to/vascular-generation) describes the library,
completion of imported networks, distributed growth, physical laws and failure
reports.

The four domains start with two uniform refinements. The adaptive cube then
refines towards the edge parallel to $z$ at $x=y=0$. The adaptive sphere refines
towards its center. Successive refinement distances are $L/2,L/4,L/8$ for cube
side $L$ and $R/2,R/4,R/8$ for sphere radius $R$. These are ordinary deal.II
refinement passes, completed before generation. Growth does not adapt the mesh.

| Parameter | Value |
|---|---|
| Cube origin and side | $(0,0,0)$; $L=1\ \mathrm{m}$ |
| Sphere center and radius | $(0.5,0.5,0.5)\ \mathrm{m}$; $R=0.5\ \mathrm{m}$ |
| Arterial seed and initial direction | $(0.2,0.5,0.5)\ \mathrm{m}$; positive $x$ |
| Fixed arterial seed radius and initial length | $0.003\ \mathrm{m}$; $0.15\ \mathrm{m}$ |
| Dynamic viscosity | $0.004\ \mathrm{Pa\,s}$ |
| Total new-tissue reference flow | $2\times10^{-9}\ \mathrm{m^3/s}$ |
| Arterial inlet pressure | $12000\ \mathrm{Pa}$ |
| Murray exponents; tissue demand; spatial weight | $3$; $1$; $1$ |
| Sampling seed and initial terminal count | $23$; $64$ |
| Terminal count multiplier factor | `none`: these examples prescribe absolute counts |
| Density mesh exponent $\beta$ | $0$ on uniform backgrounds; $3$ on adaptive backgrounds |
| Maximum service distance | $0.2\ \mathrm{m}$ |
| Terminal-count calibration bounds and budget | $32$–$128$; at most $12$ trials |
| Relative volume tolerance | $5\%$ |
| Minimum arterial coverage | $0.25$ |
| Numerical interval target and desired range | $h$; $0.5h$–$2h$ |

The requested volume fractions are $4.6\times10^{-5}$ for the uniform cube,
$3.4\times10^{-5}$ for the edge cube, $7.0\times10^{-5}$ for the uniform sphere
and $3.5\times10^{-5}$ for the central sphere. They were chosen near the measured
64-terminal pilot values to compare equally sized physical trees. The published serial runs converge at the initial count. In the two-rank
check, the uniform cube instead converges with 92 terminals; its geometry and
count differ while the same volume tolerance and minimum coverage still hold.
This single-seed comparison does not establish statistical equivalence across
partitions. Separate calibration tests exercise bounded count changes and
nonmonotone trials.

The source configurations are
{download}`uniform cube <../../data/vascularization/cube_uniform.prm.in>`,
{download}`edge cube <../../data/vascularization/cube_edge.prm.in>`,
{download}`uniform sphere <../../data/vascularization/sphere_uniform.prm.in>` and
{download}`central sphere <../../data/vascularization/sphere_center.prm.in>`.
CMake generates their runtime `.prm` files below the build tree.
[All effective parameters](../_static/vascular-generation/effective-parameters.txt)
record every native parameter, including defaults. Only the run-directory prefix
is replaced by `<run>` in that published record.

## Networks and backgrounds

```{image} ../_static/vascular-generation/domains.svg
:alt: Four native generated vascular trees with one common oblique projection and radius scale
:width: 100%
```

The camera, spatial scale and vessel-width scale are common to all four panels.
Tube widths are four times physical diameter to make the radii legible; the
stored radii are unchanged. Black dots mark the explicit inlet ports. The gray
outline is the convex hull of projected native mesh vertices, not a generated
replacement mesh.

```{image} ../_static/vascular-generation/meshes.svg
:alt: Wireframes of native background cells intersecting the selected planes, showing edge and central refinement
:width: 100%
```

The cube wireframes show cells intersecting $y=0.0625$, close to the refined
edge. Sphere wireframes use $y=0.5$. Both include cells on either side of the
selected plane. The central sphere panel resolves the nested refinement regions.
First-order mappings approximate the spherical boundary. Reported volume
fractions use each actual native background volume, not $4\pi R^3/3$.

## Physical terminal density and numerical cell length

```{image} ../_static/vascular-generation/terminal-density.svg
:alt: Physical arterial terminal density per native quadrature volume versus distance from the refined edge or center
:width: 100%
```

The denominator is the sum of native background `QGauss(2)` physical weights in
each distance bin. The numerator counts owned physical arterial tissue
terminals. This is a quadrature estimate of shell volume. A shell with no
quadrature volume has no plotted density. The logarithmic vertical coordinate
is $\log_{10}(1+\rho_T)$, with density in terminals per cubic metre.

The adaptive cases use $\beta=3$, concentrating terminal sampling in smaller
background cells. The uniform cases use $\beta=0$. These single-seed examples
show the resulting physical distribution; they do not establish statistical
quality across partitions or seeds.

```{image} ../_static/vascular-generation/length-distribution.svg
:alt: Histograms of numerical interval length divided by the smallest actually crossed background-cell diameter
:width: 100%
```

Each histogram counts numerical intervals. Its denominator $h_{\min}$ comes
from background cells actually crossed by the interval. Short physical branches
and crossing fragments can fall below the desired $0.5h$ bound; the native
interval table records short-cell exceptions. Subdivision preserves each
physical vessel's integrated volume and Poiseuille resistance and creates no
new bifurcation.

## Requested and achieved targets

```{image} ../_static/vascular-generation/targets.svg
:alt: Requested and achieved vascular volume fraction and separate geometric coverage for the four domains
:width: 100%
```

The native runs validate the volume tolerance and minimum coverage separately.
The adaptive trees leave more tissue beyond the service distance than the
uniform trees at the same terminal count and service radius. A positive nearest
terminal assignment does not make such tissue covered. Geometric service
coverage is not a proof of physiological perfusion.

[Measured run summary](../_static/vascular-generation/summary.csv) records counts,
volume fraction, coverage and calibration results. The full owned tables,
archives and meshes remain in the build tree.

## Arterial and venous common territories

```{image} ../_static/vascular-generation/arteriovenous.svg
:alt: Red arterial and blue venous native trees with dashed positive zero-dimensional connections between paired terminals
:width: 100%
```

The {download}`arterial–venous configuration <../../data/vascularization/arteriovenous.prm.in>`
adds a venous outlet at $(0.8,0.5,0.5)\ \mathrm{m}$ with fixed radius
$0.004\ \mathrm{m}$ and pressure $4000\ \mathrm{Pa}$. Terminal partners are
$0.04\ \mathrm{m}$ apart. A positive resistance for each bed is derived from the
explicit pressure data and demand flow, then checked in the actual paired
reference solve. The target volume fraction is $1.26\times10^{-4}$ and the
minimum dual coverage is $0.6$.

Dashed links identify the common territories. They are 0D resistive connections,
not capillary segments with an implicit volume. Veins grow from the outlet
towards the tissue; their reference flow has the opposite orientation. These
reference results precede any transient haemodynamic simulation.

## Rediscretize the same physical graph

```{image} ../_static/vascular-generation/rediscretization.svg
:alt: The same physical vascular graph with additional numerical vertices after rediscretization
:width: 100%
```

The {download}`load-only configuration <../../data/vascularization/rediscretized.prm.in>`
reloads the uniform cube archive without growth. It uses four uniform background
refinements and target interval length $0.5h$ with lower bound $0.25h$. The
pipeline checks identical physical ids, connectivity, geometry, radii and
viscosity, and independently integrates every interval to check preservation of
physical volume and resistance. Only numerical cells and vertices change. Coverage estimates can change with
the background quadrature even though terminal coordinates remain identical.

## Reproduce and check the published figures

Build the standalone target using an installed deal.II, with MetricFlowX disabled
if desired:

```bash
cmake -S . -B /tmp/immersx-vascular-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_MetricFlowX=ON
cmake --build /tmp/immersx-vascular-build --target vascular_network -j
python3 scripts/run_vascular_examples.py \
  --executable /tmp/immersx-vascular-build/vascular_network \
  --parameters /tmp/immersx-vascular-build/data/vascularization \
  --output /tmp/immersx-vascular-build/vascular-examples-1
python3 scripts/update_vascular_figures.py \
  /tmp/immersx-vascular-build/vascular-examples-1 --check
```

The output directory must be new. The runner launches every executable from an
unrelated working directory and retains logs, effective parameters, distributed
archives and tables below that run directory. The Python scripts perform no
vascular growth. They validate and plot owned C++ output with the standard
library, without a rendering dependency.

To refresh selected assets after reviewing a fresh run, omit `--check`. To test
reproducibility, run the generator again into a second new directory and check
it against the published assets. The manifest records the fixed projection,
diameter exaggeration and asset hashes. Equality applies to the recorded seed,
parameters, process count and numerical environment. Different MPI partitions
may produce different valid geometry and are not compared byte for byte.

CTest also runs the six examples through
`ImmersX.Application.vascular_network.IllustratedExamples` (with `_debug` in the
executable name for Debug). It retains a unique build-tree directory for each
run. Pass `--ranks 2 --mpiexec /path/to/mpirun` to the runner for an MPI run; its
invariant checks apply to the combined owned records. MPI scaling benchmarks
are separate from these small illustrated cases.


## Generate before the coupled transient solve

With a compatible MetricFlowX package installed, `metric_flow_x_elastodynamics`
can generate a validated internal network on the solid's final background
before distributing native flow DoFs and constructing the existing wall
coupling. The generator itself still builds without MetricFlowX.

The {download}`small coupled configuration <../../gtests/parameters/metric_flow_x_generated_coupled.prm.in>`
uses a unit cube with 64 solid cells and a four-terminal arterial tree.
Its seed, viscosity and reference-flow data match the synthetic values above;
the terminal count is fixed, with no volume calibration or minimum-coverage
target. Native Square root walls use Young modulus `5e5 Pa`, wall thickness
`0.0003 m`, and reference pressure `125 Pa`. Each tissue outlet uses native
`R1=0`, `R2=1e8 Pa s/m^3`, `C=0` and the same `125 Pa` reservoir pressure.
The equal nonzero reference and outlet pressures balance at rest; the pressure
reference is assigned once to native `p_d`, with native `p0=0`.

The inherited time-dependent inflow is
$Q(t)=5\times10^{-8}(1-\cos(100\pi t))\ \mathrm{m^3/s}$.
Both $Q(0)$ and $Q'(0)$ vanish, matching the initially resting solid; the inflow
is nonzero at subsequent times. This transient forcing is distinct from the
`2e-9 m^3/s` resistive design flow used during generation. IDA advances from
zero to `0.002 s`, with flow, solid and pressure-jump output at
`0`, `0.001` and `0.002 s`.

Configure both packages against the same deal.II installation and linear
algebra backend. This checkout selects Trilinos, so configure the native
package with `-DMETRIC_FLOW_X_BACKEND=Trilinos`. For example:

```bash
cmake -S . -B build-coupled -DCMAKE_BUILD_TYPE=DebugRelease \
  -DMetricFlowX_DIR=/path/to/metric-flow-x-install/lib/cmake/MetricFlowX \
  -DENABLE_GOOGLE_TESTING=ON
cmake --build build-coupled --target \
  metric_flow_x_elastodynamics metric_flow_x_elastodynamics_debug -j
ctest --test-dir build-coupled \
  -R 'application\.MetricFlowXGenerated\.' --output-on-failure
```

CTest creates separate parameter and output paths for each build variant and
solver mode. The checks require actual accepted IDA steps, nonempty internal
vessel geometry, the three output times, positive flow-state change and
positive solid displacement. Available direct-solver cases include serial
UMFPACK and serial/two-rank/four-rank MUMPS. An older MetricFlowX package
without the native in-memory input API skips these generated-input cases.

For a new application, follow the
[native transfer example](vascular-native-transfer).
It explains owner-local material callbacks, explicit terminal data, native
coefficient replication, stable physical IDs and reversed venous execution
orientation. Partially external imported networks remain supported by the
standalone generator; adapting their multiplier space for partial wall coupling
is deferred.

(vascular-measured-mpi-growth)=
## Measured MPI growth

These measurements use three random seeds (`23`, `24`, `25`) for every
process-count and size combination: 1,000 and 10,000 terminals at
`P=1,2,4,8` for strong scaling, and 1,000 terminals per process at
`P=1,8` for weak scaling. Every one of the 30 native runs passed the
independent table checks described in the generation guide. The figures
show medians and full observed seed ranges; they do not fit a model.

| Protocol | MPI ranks | Generation time: median [min, max], seconds |
|---|---:|---:|
| strong, T=1,000 | 1 | 144.3 [143.2, 144.8] |
| strong, T=1,000 | 2 | 46.52 [44.08, 48.44] |
| strong, T=1,000 | 4 | 15.24 [15.21, 15.32] |
| strong, T=1,000 | 8 | 5.549 [5.466, 5.996] |
| strong, T=10,000 | 1 | 1.418e+04 [1.415e+04, 1.427e+04] |
| strong, T=10,000 | 2 | 3691 [3631, 4069] |
| strong, T=10,000 | 4 | 997 [994.2, 1000] |
| strong, T=10,000 | 8 | 283.9 [273.5, 285.4] |
| weak, T=1,000 per rank | 1 | 142.4 [140.5, 145.2] |
| weak, T=1,000 per rank | 8 | 227.9 [227.1, 229.2] |

```{image} ../_static/vascular-benchmarks/scaling.svg
:alt: Measured generation times and same-seed elapsed-time ratios with full three-seed ranges
:width: 100%
```

The strong protocol keeps terminal count and physical controls fixed,
but changing the number of processes changes batch size, sampled geometry
and the number of global physiological updates. Ratios above the process
count therefore include changes in algorithmic work; they are not hardware
speedups for an identical sequence of insertions. The weak protocol grows
one connected tree from one real inlet as the domain and terminal count
increase. It does not create independent roots to give every rank work.

```{image} ../_static/vascular-benchmarks/resources.svg
:alt: Measured peak resident memory, ownership imbalance, MPI activity, physiological updates and startup writer ownership
:width: 100%
```

Peak RSS is the maximum per-rank high-water mark. The CSV also retains
the sum of individual peaks; that sum counts shared pages repeatedly and
does not measure simultaneous node memory. MPI bytes are logical posted
sends plus collective input, not wire traffic. MPI seconds sum rank time
inside the instrumented calls and can exceed elapsed time. The first-range
writer fraction measures ownership of mutable branches across rank-batches;
it does not count all CPU-active proposal producers. The separate first-range
proposer fraction counts ranks with at least one proposal. Whole-growth phase
times use the maximum of per-rank totals rather than a sum of range maxima.

```{image} ../_static/vascular-benchmarks/quality.svg
:alt: Observed volume fraction, geometric coverage, physical branch lengths and junction angles across seeds and partitions
:width: 100%
```

| Protocol | MPI ranks | Volume fraction: median [min, max] | Arterial coverage: median [min, max] |
|---|---:|---:|---:|
| strong, T=1,000 | 1 | 6.905e-05 [6.847e-05, 6.96e-05] | 1 [1, 1] |
| strong, T=1,000 | 2 | 6.846e-05 [6.806e-05, 6.853e-05] | 1 [0.9998, 1] |
| strong, T=1,000 | 4 | 6.819e-05 [6.807e-05, 6.943e-05] | 1 [1, 1] |
| strong, T=1,000 | 8 | 6.85e-05 [6.7e-05, 6.865e-05] | 1 [1, 1] |
| strong, T=10,000 | 1 | 8.913e-05 [8.842e-05, 8.955e-05] | 1 [1, 1] |
| strong, T=10,000 | 2 | 8.875e-05 [8.866e-05, 8.879e-05] | 1 [1, 1] |
| strong, T=10,000 | 4 | 8.898e-05 [8.851e-05, 8.905e-05] | 1 [1, 1] |
| strong, T=10,000 | 8 | 8.889e-05 [8.884e-05, 8.906e-05] | 1 [1, 1] |
| weak, T=1,000 per rank | 1 | 6.905e-05 [6.847e-05, 6.96e-05] | 1 [1, 1] |
| weak, T=1,000 per rank | 8 | 8.684e-05 [8.665e-05, 8.708e-05] | 1 [1, 1] |

All runs satisfy the same topology, Murray, wall-containment, junction-angle,
flow and subdivision checks. The plotted distributions describe the
observed seed and partition variability. Three seeds do not establish
statistical equivalence, and these synthetic geometric measurements do
not establish physiological perfusion. The benchmarks prescribe terminal
counts; they do not calibrate a target volume fraction or enforce a
minimum coverage.

The smallest recorded reference area across these trees is 6.0915206e-08 m².
The native transfer requires an area strictly above `1e-10 m²` and
rejects smaller physical radii instead of clipping them. This comparison
uses the generated radii; the generation benchmarks do not advance the
transient flow solver.

[Per-seed measurements](../_static/vascular-benchmarks/measurements.csv)
include radius and terminal-distance quantiles in addition to the plotted
metrics, plus per-rank sample, proposal and accepted-insertion counts and
their maximum-to-mean ratios. The [manifest](../_static/vascular-benchmarks/manifest.json)
records the source and binary identities, input hashes and asset hashes.

### Recorded environment

The recorded runs use an Intel Xeon E5-2650 v4 host, GCC 11.4.0,
deal.II 9.8.0 and Open MPI 5.0.10. They use the Release targets of a
`DebugRelease` configuration with one OpenMP and one OpenBLAS thread per
process. Effective compilation includes `-DNDEBUG -O2 -fopenmp`,
`-fopenmp-simd -march=znver3 -mtune=znver3`; the architecture flags come
from the installed SDK and are recorded as used, despite the Intel host.
The frozen benchmark executables and library use the gold linker. These
are recorded local build conditions, not required project flags.

MPI processes use `--bind-to core --report-bindings`. Other local builds,
functional tests and documentation work used separate cores where
possible, while the host also served unrelated work. Memory bandwidth,
cache and storage remained shared. Treat the timings as shared-host
measurements rather than isolated machine performance.

The frozen generator source is `76bb381187285e57bdc281a09f2bdd4c09a70703`.
The executable SHA-256 is `d4a70de019707637c76897b44dd33723a5cc3a1805793abc784467096f8a06ca` and the
library SHA-256 is `559e03f47fe7a6e041f40bf6d134a133b8f2c2b3b5b28965dd70e0c29b9eb082`. These files were
copied before the experiments and were not rebuilt between cases.

### Regenerate the measured assets

The measurements were run as three experiments. With a compatible
Release profiling build, run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_vascular_benchmarks.py \
  --executable build-release/benchmarks/vascular_network_profile \
  --library build-release/libimmersx_lib.so \
  --output build-release/vascular-measurements/strong-1000 \
  --kind strong --terminals 1000 --ranks 1,2,4,8 --seeds 23,24,25
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_vascular_benchmarks.py \
  --executable build-release/benchmarks/vascular_network_profile \
  --library build-release/libimmersx_lib.so \
  --output build-release/vascular-measurements/weak-1000 \
  --kind weak --terminals 1000 --ranks 1,8 --seeds 23,24,25
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_vascular_benchmarks.py \
  --executable build-release/benchmarks/vascular_network_profile \
  --library build-release/libimmersx_lib.so \
  --output build-release/vascular-measurements/strong-10000 \
  --kind strong --terminals 10000 --ranks 1,2,4,8 --seeds 23,24,25
python3 scripts/update_vascular_benchmarks.py \
  build-release/vascular-measurements/strong-1000 \
  build-release/vascular-measurements/weak-1000 \
  build-release/vascular-measurements/strong-10000 --check
```

The publisher independently rechecks native tables and frozen binary
hashes. It rejects any incomplete experiment. Omit `--check` to update
selected assets after reviewing new measurements. Repeated timings need
not match; `--check` checks the publication against the supplied completed
dataset, not against an assumed performance target.
