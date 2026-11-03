# Grow a vascular tree in a sphere with CCO

Start with a spherical `Domain`, grow an arterial tree with constrained
constructive optimization (CCO), and save the physical network for use with
MetricFlowX. A separate parameter file prepares a coupled
MetricFlowX–elastodynamics run on the same spherical geometry.

The standalone inputs produce a valid tree with all 56 requested terminals
and save its physical archive and visualization meshes. Material, flow and
pressure data are illustrative SI values. The coupled input remains a draft;
IDA evolution and ParaView views remain pending.

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
the Domain fragment, standalone generation and coupled simulation. It contains:

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

## Reload the saved network for MetricFlowX

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
the spherical geometry under `/Elastodynamics/Domain/`, includes the shared
CCO controls and enables:

```text
subsection MetricFlowX coupling
  set Generate vascular network = true
  set Generated vessel Young modulus = 5e5
  set Generated vessel wall thickness = 0.0003
  set Generated vessel reference pressure = 125
  set Generated terminal resistance = 1e8
  set Generated outlet pressure = 125
end
```

The driver generates a new network on the final solid mesh, prepares native
MetricFlowX data and constructs the vessel-wall coupling before IDA starts.
This parameter file does not reload `generation/network`: the current coupled
driver has no archive-input parameter. Reusing that exact saved tree requires
the load-and-transfer API shown above in the coupled application. The two
commands are separate workflows, not a file handoff between executables.

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
deal.II installation and linear algebra backend. MetricFlowX must provide
`BloodFlowSystem::NetworkPhysics` and `setup(const NetworkPhysics &)`, and
deal.II must have SUNDIALS support. The configured output directory is
`<build>/tutorial-output/cco-sphere/coupled`. ParaView views of the background
and vessel radii, followed by the coupled run, are the next steps.
