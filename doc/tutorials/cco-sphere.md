# Grow a vascular tree in a sphere with CCO

Start with a spherical `Domain`, grow an arterial tree with constrained
constructive optimization (CCO), and save the physical network for use with
MetricFlowX. A separate parameter file prepares a coupled
MetricFlowX–elastodynamics run on the same spherical geometry.

The standalone inputs produce a valid tree with all 56 requested terminals
and save its physical archive and visualization meshes. An export step adds
explicit wall material and terminal RCR data for native MetricFlowX input.
Material, flow and pressure data are illustrative SI values. The coupled
input reads this exported tree; IDA evolution and ParaView views remain pending.

## Geometry and initial vessel

The tissue occupies $\Omega=\{x\in\mathbb{R}^3:|x|<1\}$. The inlet is
at $(0,0,-0.7)$, with radius $0.1$. The initial direction is $(0,0,1)$.
With `Length scale = local mesh size` and `Growth/Seed length = 0.5`, the
initial stem length is $0.5h(x_{\mathrm{inlet}})$. Here $h$ is the local
background-cell diameter; on interfaces the smallest containing diameter is
used. Its initial endpoint is $(0,0,-0.7+0.5h(x_{\mathrm{inlet}}))$.

Coordinates and radii remain in metres. Geometric length coefficients use
the local mesh scale, and the minimum branch length satisfies
$\int_\Gamma ds/h(x)\ge0.1$. The direction is normalized by the generator;
its magnitude does not prescribe the length. CCO can split
this initial segment when inserting bifurcations; the inlet position and
root radius remain fixed.

The inlet is inside the tissue, not on the spherical boundary. Keep
`Seed inward port normal = 0,0,0` so the complete vessel envelope is checked
against the tissue boundary.

## Prepare the spherical Domain

{download}`domain.prm.in <../../tutorials/cco_sphere/domain.prm.in>` describes
the mesh and includes the shared CCO parameters:

```text
subsection Domain
  set Grid generator = hyper_ball
  set Grid generator arguments = 0,0,0:1:false
  set Grid scale = 1
  set Initial refinement = 1
  set Triangulation type = distributed
end
```

The colon-separated arguments are the center, radius and the flag that
additionally attaches the spherical manifold to boundary cells. `hyper_ball`
already attaches a spherical manifold to boundary faces. Its seven coarse
hexahedra become 56 active cells after one global refinement. Generation uses
a first-order mapping, so the computational boundary approximates the sphere
with bilinear faces. Containment refers to that actual mesh.

The following fragment belongs inside an MPI-initialized application. Set
`domain_parameters_file` to the absolute path of the generated
`tutorials/cco_sphere/domain.prm` and `archive_directory` to a new absolute
directory outside the source tree. Every rank executes these calls:

```cpp
#include <deal.II/fe/mapping_q1.h>
#include <immersx/core/domain.h>
#include <immersx/vascularization/generation.h>
#include <immersx/vascularization/io.h>

namespace Vascular = ImmersX::Vascularization;

ImmersX::DomainParameters<3> domain_parameters("/Domain/");
Vascular::Parameters growth;
dealii::ParameterAcceptor::initialize(domain_parameters_file);

ImmersX::Domain<3> domain(domain_parameters, comm);
domain.make_grid();
dealii::MappingQ1<3> mapping;

auto generated = Vascular::generate(domain.triangulation(), mapping,
                                    {}, growth, comm);
const auto valid = dealii::Utilities::MPI::min(
  generated.validation.valid() ? 1u : 0u, comm);
AssertThrow(valid == 1, dealii::ExcMessage("CCO generation failed"));

Vascular::save_network(archive_directory, generated.network,
                       generated.territories, generated.terminal_pairs,
                       generated.terminal_triplets, comm);
```

Finish all mesh refinement before `generate()`. The background remains fixed
during growth. `Domain` owns the mesh; the generator receives its triangulation
and the first-order mapping without creating another tissue domain.

## Request one terminal per background cell

{download}`cco.prm.in <../../tutorials/cco_sphere/cco.prm.in>` is shared by
the Domain fragment and standalone generation. It contains:

```text
subsection Vascularization
  set Length scale = local mesh size
  subsection Input and ports
    set Seed position = 0,0,-0.7
    set Seed direction = 0,0,1
    set Seed radius = 0.1
  end
  subsection Growth
    set Seed length = 0.5
  end
  subsection Sampling
    set Terminal count = 1
    set Terminal count multiplier factor = background cell count
    set Density mesh exponent = 3
    set Random seed = 23
  end
end
```

The cell-based count requests $N_T=N_{\mathrm{active\ cells}}$: 56 tissue
terminals at the initial refinement. This is the available interpretation of
one vessel per cell. Sampling and geometric acceptance determine their
positions; there is no constraint placing exactly one terminal in each cell.
A successful single-root binary tree with $N_T$ terminals has $2N_T-1$
physical branches, and numerical subdivision can create additional 1D cells.
`Density mesh exponent = 3` gives comparable proposal weight per cell when
the spatial weight is one; it does not impose cell occupancy.

Volume-fraction and coverage targets are disabled to retain this terminal
count. The fixed inlet radius is $0.1$; it is not rescaled to meet a volume
target. Increasing refinement also increases the requested terminal count.

### Prescribe the terminal cloud before growth

To prescribe exactly $k$ terminal slots in every active cell, use:

```text
subsection Vascularization
  subsection Sampling
    set Terminal sampling = prescribed per cell
    set Terminal count = 1
    set Terminal count multiplier factor = background cell count
    set Density mesh exponent = 3
    set Spatial weight = 1
  end
end
```

Here `Terminal count` is $k$. With $k=1$, the prescribed position is the
mapped reference-cell centre. For larger counts, hexahedra use a tensor
midpoint grid with the smallest number of subdivisions providing at least
$k$ points. The generator shuffles this grid and selects $k$ positions.
Tetrahedra use interior simplex quadrature locations. Existing mutable
leaves occupy slots in their containing cells; their positions stay fixed.
The generator rejects an initial forest with more than $k$ mutable leaves
in any cell.

Each rank stores its cloud in a deal.II `ParticleHandler` on its owned cells.
CCO attempts to connect these points in shuffled order. A failed point stays
pending and is retried after the other points, when the tree may offer more
connections. Success requires every prescribed slot to be connected; a full
cell-based count alone cannot hide missing cells. Geometry, minimum length,
Murray radii, and collision constraints still apply. A finite search budget
can therefore leave the result invalid.

This mode fixes the count and requires volume and minimum-coverage count
calibration to remain disabled. `Maximum service distance` can still be
positive to measure coverage. The number of points per cell follows mesh
density; it does not impose uniform density per physical volume.

The {download}`benchmark script <../../tutorials/cco_sphere/benchmark_sampling.py>`
runs a JSON list of cases serially, using the same executable and one thread.
Each case names its `mode` (`random` or `prescribed per cell`), `refinement`,
`k`, and `seed`; optional controls include `beta`, `radius`, `candidates`,
`budget`, and `service_distance`. It preserves parameter files, physical
archives, logs, summaries, binary checksum, and a combined `results.csv`.
Use fresh output directories for every experiment:

```bash
python3 "$IMMERSX_SOURCE/tutorials/cco_sphere/benchmark_sampling.py" \
  "$IMMERSX_BUILD/vascular_network" plan.json "$IMMERSX_BUILD/sampling-comparison"
```

Compare successful runs at matching terminal counts and coverage, using the
same positive service distance and coverage quadrature. Report generation
time separately from the process wall time, which also includes startup
and output.

## Generate and save with the standalone application

{download}`generate.prm.in <../../tutorials/cco_sphere/generate.prm.in>` uses
the same spherical geometry through the standalone driver's domain settings:

```text
subsection Vascularization
  subsection Domain
    set Shape = sphere
    set Sphere center = 0,0,0
    set Sphere radius = 1
    set Uniform refinements = 1
    set Refinement focus = none
  end
end
```

The standalone `vascular_network` application needs no MetricFlowX package.
It prepares its own background with `GridGenerator::hyper_ball`; use the
Domain fragment above when an application already owns its background.

CMake expands all tutorial `.prm.in` templates at the next configuration.
The commands below assume an existing build with those generated files and
executables. Pass the generated `.prm` files to applications. For example:

```bash
IMMERSX_BUILD=/absolute/path/to/immersx-build
"$IMMERSX_BUILD/vascular_network" \
  "$IMMERSX_BUILD/tutorials/cco_sphere/generate.prm"
```

Use `vascular_network_debug` in a Debug build. The configured output directory
is `<build>/tutorial-output/cco-sphere/generation` and must not already exist.
Choose a new output directory for each run. The input enables the steady
Poiseuille reference calculation; it does not start a transient simulation.

After successful generation the output includes:

| Output | Purpose |
| --- | --- |
| `network/` | Physical archive with connectivity, coordinates, radii, viscosity, terminal demand and stable identifiers |
| `background*.vtu`, `background*.pvtu` | Tissue mesh for ParaView |
| `line*.vtu`, `line*.pvtu` | Numerically subdivided vascular mesh for ParaView |
| `nodes-RANK.csv`, `vessels-RANK.csv`, `terminals-RANK.csv` | Owned physical data, including vessel radii |
| `summary.csv`, `errors-RANK.txt` | Global report and generation diagnostics |
| `parameters.prm` | Effective parameters with resolved paths |

The physical archive is the saved input for the transfer below. The mesh VTU
files are visualization output; they do not supply the wall material and
terminal physics required by MetricFlowX. Saving an archive preserves the
physical network independently of its numerical subdivision.

With the supplied parameters, a single-process Debug run completed with
`valid = 1`, 56 active background cells, 56 arterial terminals, 112 physical
nodes and 111 physical branches. Numerical subdivision produced 111 line
cells. Growth accepted 55 insertions from 104 sampled points. The steady
reference calculation also completed, and `errors-0.txt` was empty. These
results establish standalone generation and archive output; the coupled
transient has not yet been run with this radius.

### Increase the background refinement

For two additional global refinements, change `Uniform refinements` from 1
to 3 in the standalone generation input. Set `Initial refinement = 3` in
the Domain input and under `/Elastodynamics/Domain/` in the coupled input
when using that finer background. Keep the inlet radius at 0.1 m and the
other shared CCO controls unchanged. Use new generation and export directories.

The cell-based terminal target grows by $8^2=64$:

| Global refinements | Background cells | Arterial terminals | Physical branches | Numerical line cells |
| --- | --- | --- | --- | --- |
| 1 | 56 | 56 | 111 | 111 |
| 3 | 3584 | 3584 | 7167 | 7169 |

Both single-process Debug runs completed with valid networks, empty error
reports and a completed steady reference calculation. The finer run accepted
3583 insertions from 17260 sampled points. Each result has its own saved
physical archive and visualization meshes.

## Export a complete MetricFlowX input

The physical archive and visualization meshes do not contain a complete
MetricFlowX input. The CCO geometry does not determine wall elasticity,
wall thickness or terminal impedances. Supply those data explicitly when
exporting the checked serial arterial tree:

```bash
IMMERSX_SOURCE=/absolute/path/to/immersx
python3 "$IMMERSX_SOURCE/tutorials/cco_sphere/export_metric_flow_x.py" \
  "$IMMERSX_BUILD/tutorial-output/cco-sphere/generation" \
  "$IMMERSX_BUILD/tutorial-output/cco-sphere/metric-flow-x/network.vtk" \
  --young-modulus 5e5 \
  --wall-thickness 0.0003 \
  --reference-pressure 125 \
  --terminal-resistance 1e8 \
  --outlet-pressure 125
```

The {download}`exporter <../../tutorials/cco_sphere/export_metric_flow_x.py>`
uses only the Python standard library. It reads the owned physical CSV tables
from the valid generation result and writes a legacy ASCII VTK file with one
line cell per physical branch. Shared junctions retain shared point indices;
coordinates, branch radii, lengths and viscosity come from the generated tree.
This tutorial exporter requires a single-process arterial result and constant
radius along each branch, as produced by these inputs. It does not regenerate
or rescale the network.

| Native VTK arrays | Values |
| --- | --- |
| Cell `vessel_id`, `L`, `r_d`, `r_in`, `r_out` | Compact branch identifiers, physical length and radii in m |
| Cell `a0`, `a_d` | Reference area $\pi r^2$ in m² |
| Cell `viscosity` | Generated dynamic viscosity in Pa s |
| Cell `E`, `h_wall` | Supplied Young modulus $5\times10^5$ Pa and wall thickness $0.0003$ m |
| Cell `p_d`, `p0` | Reference pressure $125$ Pa and offset $0$ Pa |
| Point `boundary_id` | Inlet 0, distinct terminal identifiers, junctions 255 |
| Point `R1`, `R2`, `C`, `P_out` | Terminal data $R_1=0$, $R_2=10^8$ Pa s/m³, $C=0$, outlet pressure $125$ Pa |

The VTK file contains the wall and outlet data required by the native
Square root model. Inflow, initial state and time-integration controls remain
in the parameter file. A sibling `network.json` records the physical-to-native
identifier correspondence and the supplied physics. Both output files must be
new; the original physical archive is preserved.

## Use the physical archive from C++

On the same prepared Domain, load the archive and reconstruct its numerical
intervals. These collective calls preserve the physical graph and perform no
growth:

```cpp
auto restored = Vascular::load_network(archive_directory,
                                      domain.triangulation(), mapping,
                                      0., comm);
Vascular::GenerationResult input;
input.network = std::move(restored.network);
input.territories = std::move(restored.territories);
input.terminal_pairs = std::move(restored.terminal_pairs);
input.terminal_triplets = std::move(restored.terminal_triplets);
input.discretization = Vascular::discretize(domain.triangulation(), mapping,
                                           input.network,
                                           growth.discretization, comm);
input.validation = input.discretization.validation;
const auto loaded_valid = dealii::Utilities::MPI::min(
  input.validation.valid() ? 1u : 0u, comm);
AssertThrow(loaded_valid == 1,
            dealii::ExcMessage("Network rediscretization failed"));
```

Here the halo is zero because the shared growth parameters specify zero
additional clearance. This fragment reuses the originally validated archive
on the same background. A changed background also requires the corresponding
wall-containment assessment before coupling.

With MetricFlowX enabled, transfer this result into an empty native flow
problem. Set `coupled_parameters_file` to the generated `coupled.prm` and
`flow_parameter_snapshot` to an absolute filename in its output directory:

```cpp
#include <immersx/vascularization/metric_flow_x.h>

using Flow = MetricFlowX::BloodFlowSystem<1, 3>;
Flow flow(comm);
flow.initialize_params(coupled_parameters_file, flow_parameter_snapshot);
auto transfer = Vascular::prepare_metric_flow_x(
  flow, input,
  [](Vascular::Id, const Vascular::Vessel &) {
    Flow::VesselProperties material;
    material.E = 5e5;
    material.h_wall = 0.0003;
    material.p_d = 125;
    material.p0 = 0;
    return material;
  },
  [](Vascular::Id, const Vascular::Node &, const Vascular::Terminal *) {
    return Flow::RCRPhysics{0., 1e8, 0., 125., "sphere"};
  });
```

The transfer supplies geometry, reference areas and viscosity from the archive.
The callbacks add wall Young modulus in Pa, wall thickness in m, pressures in
Pa and terminal resistance in Pa s/m³. They supply native $R_1=0$,
$R_2=10^8$, $C=0$ and outlet pressure $125$. The arterial inlet retains the
native inflow boundary. Keep the returned `transfer` when physical-to-native
identifier correspondence is needed. See the
[native transfer guide](../how-to/vascular-generation#prepare-a-generated-network-for-native-metricflowx)
for API and ownership details.

## Prepare the coupled transient simulation

{download}`coupled.prm.in <../../tutorials/cco_sphere/coupled.prm.in>` sets
the spherical geometry under `/Elastodynamics/Domain/` and reads the exported
native input:

```text
subsection MetricFlowSystem<1, 3>
  set Vtk file path for mesh input = /absolute/path/to/metric-flow-x/network.vtk
  set Wall model = Square root
  set Terminal parameter source = VTK
end
subsection MetricFlowX coupling
  set Generate vascular network = false
end
```

The configured path is
`<build>/tutorial-output/cco-sphere/metric-flow-x/network.vtk`. Export that file
before starting the coupled application. The driver reads its saved geometry,
wall material and outlet data, then constructs the vessel-wall coupling
before IDA starts. Generation is disabled, so the simulation uses the exact
exported physical tree. The native mesh refinement count is zero, preserving
one execution cell per physical branch in this example.

The input uses a clamped outer solid boundary, zero initial displacement and
velocity, a native `rest` flow state and the Square root wall model. Its
nonzero inflow is $Q(t)=5\times10^{-8}(1-\cos(100\pi t))$ m³/s;
$Q(0)=Q'(0)=0$. The design flow used by CCO is separate from this transient
forcing. The input requests a final time of $0.002$ s and output every
$0.001$ s. These are initial settings, without a convergence or response
assessment for this sphere.

An eventual run with existing executables is:

```bash
"$IMMERSX_BUILD/metric_flow_x_elastodynamics" \
  "$IMMERSX_BUILD/tutorials/cco_sphere/coupled.prm"
```

Use the `_debug` executable in a Debug build. Both packages must use the same
deal.II installation and linear algebra backend, and deal.II must have
SUNDIALS support. The configured output directory is
`<build>/tutorial-output/cco-sphere/coupled`. ParaView views of the background
and vessel radii, followed by the coupled run, are the next steps.
