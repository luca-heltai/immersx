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
:alt: Wireframes of native background cells intersecting the middle plane, showing edge and central refinement
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
