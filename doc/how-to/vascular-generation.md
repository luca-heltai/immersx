# Generate a physical vascular network

The vascularization library grows physical vessels inside a prepared 3D
background. It keeps vessel identity separate from numerical 1D cells.
`NetworkData` stores physical node, vessel and tissue-terminal identifiers,
radii, viscosity, system, orientation, ownership and provenance. A successful
`GenerationResult` contains the proposed network, its numerical subdivision,
common tissue territories and validation reports. The caller adopts it only
when `validation.valid()` is true; input data are never modified.

Generation from an empty arterial input is collective and supports one or
several MPI processes. The background point owner creates the single real seed
root; its branch may cross partitions immediately. Imported arterial networks
also support collective startup from one or several selected leaves and
collective terminal-count calibration. Arterio-venous generation starts from
real venous outlets or selected input venous leaves and uses the same MPI
path with one or several processes. Use a first-order mapping on an
all-tetrahedral or all-hexahedral physical background.
Geometry checks include the vessel wall and clearance. An ambiguous boundary
separation at the finite geometry budget fails conservatively. Explicit planar inlet/outlet sections
are supported; a warped boundary face is not replaced by a planar aperture.

## Arterial and venous trees

This synthetic example uses metres, pascals, seconds and cubic metres per
second. Its values are not an organ calibration. The background is a unit cube.
The example is exercised by `VascularAV` integration tests in Debug and Release.

```cpp
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <immersx/vascularization/generation.h>

using namespace dealii;
using namespace ImmersX::Vascularization;

Triangulation<3> background;
GridGenerator::hyper_cube(background);
MappingQ1<3> mapping;
Parameters parameters;
parameters.seed_position  = Point<3>(.2, .5, .5);
parameters.seed_direction = Point<3>(1., 0., 0.);
parameters.seed_radius    = .003;
parameters.seed_length    = .2;
parameters.viscosity      = .004;
parameters.terminal_count = 8;
parameters.reference_flow = 2.e-9;
parameters.tissue_demand  = "1+x+2*y";
parameters.arterial_murray_exponent = 2.7;
parameters.venous.enabled = true;
parameters.venous.seed_position = Point<3>(.8, .5, .5);
parameters.venous.seed_radius = .004;
parameters.venous.murray_exponent = 3.2;
parameters.venous.terminal_separation = .04;
parameters.reference_conditions.pressures = {{0, 12000.}, {2, 4000.}};

auto result = generate(background, mapping, {}, parameters, MPI_COMM_WORLD);
AssertThrow(result.validation.valid(), ExcMessage("Vascular generation failed"));
Triangulation<1, 3> vessels;
const auto physical_cells =
  create_triangulation(result.network, result.discretization, vessels);
```

For empty input, the arterial inlet is physical node `0`. Arterial seed creation
also creates its first endpoint, node `1`. When a new venous outlet is needed,
its physical node is reserved next, node `2`, before further insertions. Its
identifier therefore stays fixed when terminal-count calibration changes the
number of territories. These pressure keys are physical node identifiers,
not triangulation vertex numbers or DoF indices. Imported networks retain their
own port identifiers.

The arterial forest defines common territory centres and their demand. The
same CCO engine then grows veins from the declared outlet or selected venous
attachments toward these territories. The first venous connection goes directly
to an admissible territory, trying territories in distance/id order. Each later
insertion tries six Cartesian endpoint offsets of the configured separation.
The territory centre is the arterial endpoint; its venous endpoint is distinct.
Demand is evaluated once at the common centre and shared exactly by both
terminals. A vein's geometry is oriented from outlet toward tissue, and its
physical reference flow is negative in that orientation.

The arterial forest remains fixed while the venous forest grows. All venous
proposals, including radius changes on their ancestors, are checked against
both forests and imported obstacles. This ordered construction can exhaust its
finite geometry budget even when another joint arrangement might exist. Such
failure is reported; it does not establish geometric impossibility.

Each system has its own Murray exponent. The radius of a generated vessel
satisfies $r_e=R_{\mathrm{root}}(D_e/D_{\mathrm{root}})^{1/\gamma}$, where
$D_e$ is the sum of downstream territory demands. Root radii stay fixed. The
minimum-radius check includes existing leaves when a new demand is added.
`terminal_weight` and `density_beta` control where territories are sampled;
`tissue_demand` controls supply and radii independently. Numerical subdivision
uses diameters of background cells actually crossed by each segment. Refining
these 1D cells changes neither the physical graph nor integrated volume and
resistance.

## Reference pressures, supply and microvascular beds

With `venous.microvascular_resistance == 0`, supply is distributed by relative
territory demand, with total `reference_flow` through each system. The generator
first solves the two forests under the supplied port pressures and terminal
flows. It derives a separate bed resistance
$R_i=(p_{A,i}-p_{V,i})/Q_i$ for every territory and requires it to be finite and
strictly positive. It then solves the connected graph using the supplied
external boundary conditions, with the derived beds and no prescribed flows
at paired tissue terminals. The actual bed flow must realize the demand split
within the configured numerical reference tolerance. No inlet receives both
an independent pressure and flow.

A positive `venous.microvascular_resistance` supplies that resistance directly
for each bed. With `reference_flow == 0`, pressures and resistances determine
the resulting flows. If a positive reference flow is also requested, the result
must realize its demand split; incompatible resistance/pressure/flow data fail
validation. A bed has no implicit capillary volume or geometric collision
segment.

`result.reference` reports actual pressures, signed vessel flows, territory
supply, endpoint wall shear, dissipated power and independent mass-balance
checks. `target_terminal_flows` and `target_attachment_flows` hold desired
flows, not substitutes for the solved values. The reference problem is steady
Poiseuille flow on complete physical vessels, including imported exterior
parts. It is separate from transient hemodynamics and accepted solver state.

`solve_reference()` is collective and uses native distributed vectors,
`dealii::LinearOperator` and CG with Jacobi preconditioning for one or more
processes. It assigns compact algebraic indices independently of physical ids
and numerical cells. Sparse endpoint incidence builds owned graph rows and
neighbor values; neither graph topology nor the solution is gathered.
Pressure offsets follow the nearest explicit anchor in resistive distance,
including across partitions, to retain small vascular drops beside large beds.

Supply each `TerminalPair` on exactly one process; that process receives its
territory flow. The solver resolves the two terminal identities and their
nodes even when the bed's process owns no physical entities. Boundary maps may
be distributed or repeated with identical values. Conflicting duplicates and
pressure/flow overprescription fail collectively. Terminal owners must follow
their nodes, and endpoint ghosts must match their authoritative nodes.
Results contain owned vessel, terminal and boundary records, pressures at
locally relevant nodes, and global scalar diagnostics. `reference_pressure()`
evaluates owned physical vessels.

`demand_flow_conditions()` constructs collective conditions for an arterial or
unpaired arterial/venous design solve. It uses owned terminals, normalizes
arterial sinks and venous sources independently to the same positive total
flow, and returns conditions on their node owners. Pass its result directly
to `solve_reference()`. An empty selection includes all terminals; an explicit
selection must contain the same unique physical terminal ids on every rank.
Only explicitly supplied physical ports receive pressure data. Pressure maps
may be distributed or repeated identically. Demand scaling avoids overflow in
the global sum and retains representable flows even when an intermediate
relative demand would underflow. Nonrepresentable terminal flows are rejected.
The public generated-tree reference test exercises this complete path with
nonzero tissue demand and checks actual solved flows against the requested split.

A nonzero `dissipation_weight` blends generated-vessel volume and dissipated
power, each normalized by the current active forest value. Physical flow is
split globally across the active system's roots by demand. Frozen imported
vessels contribute to the full reference solve but not to this growth objective.

## Complete imported forests

Pass the existing `NetworkData` as `initial`. Set `selected_terminals` to
arterial tissue-terminal identifiers and `venous.selected_terminals` to venous
ones. Empty selections discover available interior leaves of the corresponding
system. If no venous attachment is available, supply a new venous seed radius,
position and viscosity. Multiple arterial attachments and multiple venous
outlets are supported. The territory count must accommodate every selected
attachment in each system.

Every input node and vessel retains its geometry, radii and attributes,
including input entries already marked as generated. Selected tissue-terminal
roles are consumed; `attachment_roots` maps each consumed input terminal to its
new root vessel. Unselected input terminal records remain unchanged. Explicitly
selected exterior terminals fail instead of extending new vessels outside the
background. Supply independent loads on unpaired input terminals through
`reference_conditions`; paired terminals cannot also receive external loads.

Volume fraction includes only inside portions of all 1D vessels, with additive
volumes at junctions. It excludes beds. Coverage is a separate background
quadrature estimate of finite service balls. In arterio-venous mode calibration
uses dual coverage: quadrature points must lie within service distance of both
systems. This geometric coverage is not a proof of physiological perfusion.

`measure_tissue()` is collective on the background communicator and returns
all metrics globally on each process. Only owners of terminal records supply
service centres. Centres travel to every background partition within service
reach, independently of the vessel ghost halo; a service ball can cross several
partitions. Each process integrates its own background cells, then the physical
quadrature weights are summed before normalization. Empty processes participate.
The supplied `discretize()` result already contains the global inside vessel
volume, which is used once. Setting the service radius to zero disables coverage
while retaining the global volume fraction.

Arterial terminal-count calibration is collective. Set a positive
`calibration.volume_fraction` or `calibration.minimum_coverage` to enable it.
Every process must supply the same targets, tolerance, terminal-count bounds,
attempt budget and initial `terminal_count`. Each attempt grows from the
unchanged input with native ownership and measures global volume and coverage.
The finite count search makes the same decisions on every rank without
collecting the graph. It does not assume that volume increases with terminal
count. The defaults are 5% relative volume tolerance and at most 12 attempts.
A volume match still fails when the requested coverage is missing.

When the fixed imported inside volume alone exceeds the upper target tolerance,
calibration reports `incompatible` before growing. Otherwise, exhausting the
allowed counts or attempt budget reports `exhausted`, which does not prove
incompatibility. A failed search can retain its best valid network for
diagnostics, but its overall validation remains invalid. Trial sample counts
are global and include discarded attempts. The supplied parameters are not
modified.

Imported arterial startup first refreshes native point/midpoint ownership and
endpoint incidence. It validates terminal records at their node owners, adds
missing non-port leaf records, and selects interior arterial endpoints.
Explicit selections must name unique input terminal ids; an explicitly selected
exterior leaf is rejected. Unselected and exterior input branches remain fixed,
including input entries marked as generated.

Containing-cell queries for all selected leaves use sparse batches. New roots
inherit the incoming vessel's distal radius, viscosity and extra physical
properties. Node owners propose roots in concurrent geometry rounds; the same
batch checks and joint commit used during growth certify them before adoption.
New identifiers use disjoint integer ranges. No artificial roots or full graph
replicas are introduced. `attachment_roots` records each consumed terminal's
new root id on the consumed terminal's native node owner; root-vessel ownership
may differ after midpoint redistribution.

The final arterio-venous pairing uses sparse exchanges independently of
endpoint ownership. Each common territory has one writer and exactly one
arterial and one venous terminal, with distinct positions and matching demand.
Pair records and actual bed-flow results stay at the territory writer; the
venous terminal need not be present there as a ghost. For derived beds, an
unpaired design solve supplies endpoint pressures from their native owners.
Only new tissue terminals enter the demand normalization. Boundary conditions
on unpaired input terminals remain in both reference solves. The final paired
solve must produce positive perfusion and the requested demand split; desired
flows are never substituted for its measured flow. This finalization is
collective, including processes with no vascular entities or background cells.

A new venous outlet receives its stable node id before arterial CCO insertions.
For imported networks, selected venous leaves retain their incoming distal
radius, viscosity and additional physical properties. Outlet owners query
available common territories in batches; each territory producer returns one
nearest candidate. Its writer grants competing requests deterministically.
A losing outlet retries after committed claims are removed. A geometric
failure tries the remaining Cartesian terminal offsets, then another territory.
New branches pass the common geometry selection and joint commit before any
territory claim is retained. Input vessels and completed arterial geometry
remain fixed obstacles throughout venous startup and growth.

## Save and reload physical data

Save a validated result in a new directory on a filesystem visible to all ranks:

```cpp
#include <immersx/vascularization/io.h>

const auto archive_directory = output_directory / "physical-network";
save_network(archive_directory, result.network, result.territories,
             result.terminal_pairs, communicator);
const auto restored = load_network(archive_directory, background, mapping,
                                   parameters.clearance, communicator);
```

`save_network()` is collective. Each writer stores its owned vessels and
terminals, relevant endpoint nodes, and its supplied territory/bed records.
Files contain at most 1024 owned records of each kind, plus endpoint nodes.
Even an archive with one writer can therefore distribute its parts among many
readers without loading the entire physical graph on one process.
A versioned manifest is published only after every part closes successfully.
The target directory must not already exist. An interrupted or failed write
can leave parts without a manifest; `load_network()` rejects that directory.
Existing archives are never overwritten.

`NetworkArchive` contains physical `network`, `territories` and
`terminal_pairs`. Integer identifiers, coordinates, endpoint order, taper,
viscosity, additional named properties, arterial/venous classification,
provenance, port roles and normals, tissue demand and bed resistance survive the
round trip. Numerical subdivisions and solved reference states are recomputed.
No growth history or solver state is restored. To complete a restored network,
pass its physical metadata as well as its graph:

```cpp
const auto completed = generate(background, mapping, restored.network,
                                parameters, communicator,
                                restored.territories, restored.terminal_pairs);
```

Every input vessel remains fixed. Select both endpoints of any existing bed
that will be extended, using the arterial and venous selected-terminal lists.
The consumed bed and territory are replaced by new tissue territories.
Untouched territories and beds remain in the valid result. Existing beds
require venous growth to be enabled and enter both the design solve and the
final paired solve. Their flows can change when new branches share upstream
vessels; their resistances remain fixed. The specified `reference_flow` and
returned desired-flow maps apply only to new terminals. Volume and coverage
still include the whole physical network. A half-extended bed is rejected
rather than losing its terminal relation. The same rules hold in terminal-count
calibration and on processes with no local metadata.

Readers need not have the writer count. They read old parts cyclically, check
unique authority and matching endpoint data, then use `redistribute()` against
the prepared background. Interior points and vessel midpoints acquire their
native owners. Exterior entities retain their assigned reader. Territories and
beds move to the restored arterial endpoint owner. Only file counts and coarse
background boxes are replicated; no process must gather the complete graph.
Empty writers and readers join all phases. Invalid records, missing or truncated
parts, unsupported versions and filesystem errors fail collectively.

## Discretize across background partitions

`discretize()` is collective and accepts owned physical vessels with their
endpoint nodes, plus optional ghosts. It sends each owned segment to the
background partitions whose coarse boxes it intersects. Exact tetrahedral or
bilinear hexahedral crossings determine resolution; bounding-box overlaps alone
do not count as crossings. Interface portions contribute volume once.

Numerical intervals stay at their physical vessel owner. Each interval records
its execution owner from midpoint containment, choosing the lowest containing
rank on interfaces; exterior intervals retain the physical owner. Inside volume
and short/exterior-cell counts are global on every rank, and the numerical cell
budget applies to the whole network. Processes with no owned background cells or
vessels still participate.

Pass a native fully distributed destination to build the line mesh collectively:

```cpp
#include <deal.II/distributed/fully_distributed_tria.h>

parallel::fullydistributed::Triangulation<1, 3> vessels(communicator);
const auto references = create_triangulation(local_network, subdivision, vessels);
```

Here `subdivision` is the result of `discretize()`. Numerical cells migrate to
their execution owners. A sparse directory of physical endpoint and numerical
cut identities supplies vertex-adjacent ghost cells, including all branches at
a bifurcation. No complete physical graph or mesh is gathered. Compact native
cell and vertex indices are independent of physical identifiers, including
identifiers larger than the exact integer range of a `double`.

`references` maps every locally owned or ghost coarse cell to its physical
vessel and parameter interval. Use `physical_reference(cell->id(), references)`
to recover that interval; skip artificial cells on empty ranks. The physical
network and its attributes remain unchanged. To change distributed numerical
resolution, call `discretize()` again and build a new mesh: native
`parallel::fullydistributed::Triangulation` does not implement refinement flags.
The serial destination continues to support native binary refinement.

## Redistribute physical ownership

For an existing distributed `NetworkData`, `redistribute()` migrates owned
entities against a prepared parallel background and refreshes geometric ghosts.
It is collective, including processes with no physical entities or no owned
background cells. This operation does not generate branches. `generate()`
supports MPI startup from an empty arterial input and from imported arterial
terminals, together with real venous outlets or selected venous leaves in
arterio-venous mode.

```cpp
#include <immersx/vascularization/distribution.h>

const auto relevant_network =
  redistribute(background, mapping, local_input, clearance + search_reach,
               communicator);
```

Each physical entity has one authoritative input owner. A rank needs its own
entities and the endpoint nodes referenced by its locally relevant vessels;
it does not need the complete network. Node and terminal owners follow their
physical points. The physical vessel owner follows its midpoint, matching the
centre-based policy of `ImmersedRepartitioner`. Native `ParticleHandler` search
checks actual background cells. Points on interfaces choose the smallest
containing rank; a bounding-box overlap alone does not assign ownership.
Exterior points and vessel midpoints retain their previous owner. A terminal
always follows its node.

A crossing physical vessel retains one identifier, one endpoint pair and its
original taper and material properties. It is not cut into artificial physical
branches at MPI interfaces. Numerical cell ownership is a separate decision.
Every retained vessel carries its endpoint nodes and any terminal records at
those endpoints. Wire messages retain integer identifiers without converting
them to particle properties of type `double`.

Ghost exchange routes a vessel to background covering boxes intersecting its
segment bounding box expanded by the maximum endpoint radius plus the supplied
halo. The halo can include wall clearance and an additional candidate-search
reach. It can span several cells or partitions and is independent of mesh
adjacency. These are conservative recipient queries; callers must still check
actual segment geometry before accepting an insertion. Vessel ghosts also reach
both endpoint owners, including exterior endpoints beyond every background
covering box. This supplies complete incidence for input-leaf validation and
attachment selection.

Authoritative node values replace stale ghost values before midpoint location.
Ghost edits to geometry, radii or demand cannot override owned data. Sparse
migration and ghost exchanges use `Utilities::MPI::some_to_some`; only coarse
background covering boxes are replicated. Duplicate owned identifiers are
checked through a distributed hash partition of the identifier registry.
Invalid ownership or missing authoritative nodes fail collectively at a phase
boundary. Input data remain unchanged. Repeating the operation after changes
replaces obsolete ghosts, including when vessel radii shrink.

## Validate a batch of proposals

Before accepting a serial CCO insertion, the engine passes its changed vessel
geometry through the same batch validation phase exercised by the MPI tests.
Point location verifies all endpoints globally. Each participating process
checks its owned physical boundary patches and relevant vessel obstacles;
process interfaces are not physical boundaries. An open planar section must
match at least one physical boundary face, and every process must certify
separation from its remaining boundary patches.

A proposed replacement needs consent from the current vessel owner, including
agreement with the recorded endpoint identities and positions. New vessel rays
are also checked against other proposals. Eligible proposals use cost, producer
rank and producer-local identifier as a deterministic priority. One round
selects proposals whose priority is lower than every eligible conflicting
neighbor; the result is independent, but need not be maximal. Invalid proposals
cannot block valid neighbors. All geometric participants and writing owners
receive the same final decisions through sparse exchanges.

This phase evaluates a common snapshot and changes no accepted network or
history. The public generation loop uses its decisions to confirm or discard
the provisional distributed batch after physiological and geometric checks.

## Normalize generated forests

Final serial CCO radii and desired flows use a forest evaluation kernel that
also operates on distributed ownership. It reconstructs incidence at node
owners and exchanges subtree demand and moments from leaves to the declared
attachment roots. A reverse sweep distributes each fixed attachment radius
and total subtree demand. Each vessel owner then computes its Murray radius;
terminal owners receive the desired flows. No process needs the full topology.

The mutable vessel set is explicit: input vessels remain frozen even when
their recorded provenance is generated. Each mutable root retains its existing
uniform radius. Arterial and venous systems use separate exponents and total
demands. Positive reference flow is shared by terminal demand within each
system; zero leaves desired flows unspecified. Common physiological controls
must agree across processes. Invalid connectivity, unrooted cycles, missing
terminal roles and unrepresentable positive demands or radii fail collectively.

The kernel changes no physical data. A caller adopts its radii only after
validation and the required geometric checks. Tests exercise joint ancestor
and descendant splits across partitions, but the public generator still needs
a distributed growth and commit loop before it can generate with multiple
processes.

## Confirm or discard a batch

The internal batch staging path sends each proposed bifurcation to its physical
vessel owner. The owner checks the recorded endpoint identities and positions,
explicit mutability and exclusive writes. The serial CCO engine uses the same
physical split operation. New vessels inherit the selected vessel's material
properties; frozen input vessels remain unchanged.

Tentative geometry migrates against the background with its private mutability
and fixed-root controls. The forest kernel computes the joint demand and radii
of all accepted splits, including ancestor and descendant splits in the same
tree. Ghosts are refreshed for the new radii. Every mutable vessel is then
checked on this final snapshot, using the same distributed geometry checks as
proposal selection. A failed check discards the entire batch on every process.
The caller can resample within its finite budget; it never adopts part of a
failed batch. Staging copies locally relevant data once per batch, rather than
once per candidate.

This path provides distributed physical updates and rollback. It still needs
to be connected to distributed sampling and nearest-vessel selection in the
public generation loop. Certifying all mutable vessels after each batch is a
correctness check whose cost must be included in scaling measurements.

## Optimize from distributed ancestor paths

The serial engine and distributed candidate evaluation use the same CCO cost
and deterministic local search. A candidate needs its selected physical vessel
and the path to its actual attachment root. Subtree demand, minimum leaf demand,
and moments of the complete rooted tree provide the remaining information.
Dissipation moments include each branch's viscosity. Candidate evaluation
changes only the trial ancestral path and never copies the complete network.

MPI path requests go directly to branch owners. Each reply identifies the next
ancestor and its owner; subsequent requests are grouped by destination. Shared
ancestors are retrieved once per requesting process. Processes without requests
still participate and can serve their owned branches. Missing owners, cycles,
and inconsistent endpoint geometry fail collectively. Only the union of the
requested paths is retained, including integer physical identifiers and port
normals. No tree is replaced by a local artificial root.

Geometry checks supplied during optimization are preliminary. A candidate from
a local background partition can cross other partitions and must pass final
owner consent, joint normalization and global geometry certification. An MPI
test performs this sequence and compares the candidate cost with independently
recomputed physical volume and dissipation. Distributed sampling and nearest
candidate selection still need to be connected to the public generation loop.

## Grow a prepared forest in MPI batches

The growth loop operates on already prepared real attachment trees with one
or several processes. For arteries, each process draws raw samples in the
bounding box of its owned background cells. Acceptance includes the ratio of
its box volume to the largest box volume; this avoids prescribing the same
number of accepted terminals to partitions with different physical volumes.
Actual containing-cell queries reject holes and exterior points. On interfaces,
the smallest containing rank proposes the point and the smallest containing
cell diameter determines the density factor. Spatial sampling weight and tissue
demand remain separate expressions.

Nearest-vessel search uses owned physical segments. Only one covering box per
nonempty vessel owner is replicated. A first sparse query obtains a distance
bound; a second queries every remaining box that can improve the nearest
candidate list. Replies contain physical identifiers, owners and exact point-to-
segment distances. Ancestral paths are then fetched for these candidates.

Every producer chooses its best locally optimized candidate. Sparse geometric
votes select independent proposals; shared ancestor demand updates do not lock
the whole tree. Joint normalization and final geometry checks confirm or discard
the selected batch. Provisional integer identifiers are reserved by process and
namespace, independent of numerical cells. The loop never creates roots to
activate idle processes. A Murray bound rejects targets that cannot fit the
minimum radius under the fixed roots. Failed batches consume a finite
consecutive sampling budget, and incomplete runs are reported as failures.

Tests start from one real seed, accept insertions from multiple producers in
the same batch, and reproduce the result at a fixed partition and random seed.
Empty background processes join the same phases without sampling. This loop
serves both public arterial and venous generation and calibration. Distributed
venous startup, exclusive common-territory assignment, positive bed derivation
and actual paired reference validation use the public MPI path. Seed/import
startup, terminal-count calibration, discretization, native mesh construction,
tissue measurement and the resistive reference solve support MPI.
Scaling at the requested sizes has not yet been measured.

Public arterial generation uses the batched engine with an explicit terminal
count for each calibration attempt; it does not copy or re-register the
parameter object. Leaf identities are returned on their terminal owners even
when reference flow is unspecified, so geometric territories do not depend on
whether a flow problem was requested.

Before local optimization, each producer retrieves vessel geometry from
physical owners over the possible trial region and the fixed ancestor paths.
This includes exterior input vessels whose owners may have no nearby background
cells. Trial collision checks use proposed Murray radii, including shrinking
siblings and growing ancestors. A colliding low-cost trial cannot displace an
already feasible position. Preliminary batch votes do not
compare these trials against obsolete snapshot radii; final confirmation always
checks the complete jointly normalized geometry. A regression places a frozen
obstacle outside an old parent's wall and verifies rollback when an insertion
makes that parent's radius reach the obstacle.


The same execution loop grows veins from their actual outlet or attachment
roots. Each common territory has one authoritative producer at the arterial
terminal owner. That producer selects an offset endpoint and optimizes across
nearby venous physical vessels. The offset may belong to another background
process; it is not confined to the producer's partition. Its demand is copied
exactly from the arterial territory, without evaluating the tissue-demand
expression again at the offset point.

A distributed identifier registry checks that common territories have unique
owners. Existing venous leaves claim their territories through this registry;
duplicate claims, missing territories and mismatched demand fail collectively.
Successful new insertions remove only their producer's pending territory and
return the venous terminal identity to that owner, even if the terminal migrates
to another process. No process needs the complete set of territory positions.
This enforces one venous endpoint per territory in addition to geometric
collision checks. The previous separate serial growth implementation has been
removed. Public paired reference solves check positive beds and actual demand-matched
flows through the same distributed execution path.

## Standalone application

The `vascular_network` executable uses the same library implementation and needs
no MetricFlowX installation. In Debug builds its name is
`vascular_network_debug`. In a `DebugRelease` build both names are available.
Print the complete parameter interface into a new file:

```bash
build-debug/vascular_network_debug --write-parameters vascular.prm
```

The following synthetic SI configuration generates one physical vessel with a
fixed radius. It prescribes inlet pressure and total tissue outflow at distinct
nodes. It is an analytic CLI test case, not an organ calibration.

```text
subsection Vascularization
  subsection Input and ports
    set Mode = generate
    set Seed position = .2,.5,.5
    set Seed direction = 1,0,0
    set Seed radius = .005
  end
  subsection Growth
    set Seed length = .4
  end
  subsection Sampling
    set Terminal count = 1
  end
  subsection Physiology
    set Dynamic viscosity = .004
    set Total new-tissue reference flow = 2e-9
    set Reference pressures = 0:12000
  end
  subsection Objectives
    set Maximum service distance = 2
  end
  subsection Output
    set Directory = generated
  end
end
```

Run it with one or several processes:

```bash
mpirun -np 2 build-debug/vascular_network_debug vascular.prm
```

Input archive and output paths are relative to the parameter file. The output
directory must be new. Missing parameter files are errors and are never created
implicitly. `parameters.prm` records the effective configuration with resolved
paths. Invalid generation or reference results return a nonzero exit code and
write `summary.csv` and `errors-RANK.txt`; they are not saved as valid archives.

`Mode = complete` requires `Input archive = generated/network`. It preserves
physical input vessels and extends the selected terminal ids. Supply `Seed
length` for new attachment branches and `Total new-tissue reference flow` for
the new tissue. Viscosity and attachment radii can be inherited from the input.
Retained territories and paired beds follow the library completion rules above.

`Mode = load` requires the same archive path, performs no growth and does not
require seed parameters or viscosity defaults. Choose a new background or
`Discretization/Target length factor` to rediscretize the physical network.
For arterial networks without paired beds, `Physiology/Loaded tissue reference
flow` applies a total demand-based outflow across all loaded terminals. Zero
uses only explicit reference boundary data. `Input and ports/Evaluate reference`
can disable this additional solve. Generated arterio-venous networks always
validate their paired reference solve.

`Domain/Shape` selects `cube` or `sphere`. Configure the cube origin and side
length or the sphere center and radius. `Uniform refinements` prepares the
background before growth. For a cube, `Refinement focus = edge` selects an edge
using `Edge axis` and two `Edge sides` (lower=0, upper=1, in transverse coordinate
order). For a sphere, `Refinement focus = center` selects its center. Each
`Refinement fractions` entry adds one normal deal.II refinement pass where cell
centers lie within that fraction of the side length or sphere radius. The
initial fractions are `.5,.25,.125`. The prepared background remains fixed
during generation.

Each rank writes owned physical nodes, vessels, terminals, territories and pairs
as CSV. Vessel tables include full physical volume, resistance and, when solved,
flow and shear. Interval tables describe numerical subdivisions separately;
background tables record owned cells, vertices and quadrature volumes. Separate
`quadrature-RANK.csv` tables contain native background quadrature positions and
physical weights for spatial statistics. Optional
native parallel VTU files include normal mesh ghost cells. Use owned CSV records
for counts and statistics. `summary.csv` contains global metrics, calibration
status and phase times. `ranks.csv` contains ownership counts and per-rank phase
times, resident memory and high-water resident memory in kB. Memory fields are
zero on platforms where deal.II does not provide them. The generation phase
includes its internal discretization, metrics and paired reference solve;
`additional_reference_seconds` measures only the optional subsequent solve.

Library callers can use `ApplicationParameters` and the collective `run()` in
`immersx/vascularization/application.h` for these prepared domains and outputs.
Supply absolute archive/output paths. Use `generate()` directly when the calling
application already owns its final background.

`GenerationResult::growth_statistics` records local growth work in ranges grouped
by the power of two of the leaf count before each batch. Each row identifies the
active arterial or venous system and requested terminal count. Calibration keeps
rows from all trials, including unsuccessful trials. `first_terminals` and
`last_terminals` are the actual observed counts; a concurrent insertion can cross
a range boundary. Sampling, proposals and accepted insertions count local
producers. Owned-vessel counts include only mutable vessels of the active system,
not geometric ghosts or frozen input vessels. Summing the first owned counts
across ranks therefore shows the real seed/attachment startup, rather than
assuming that every process owns a root.

The standalone writes these rows to `growth-RANK.csv`, even when physical table
output is disabled. Initialization covers preparation inside the forest-growth
call; seed/attachment construction before that call remains part of the overall
generation time. Local sampling, search, optimization, selection and commit times
include communication waits. Redistribution, physiology and certification are
subphases of commit and must not be added to commit again. Physiology measures
native distributed forest evaluation and radius updates. Certification includes
its geometry-index construction and joint owner votes. The data cover completed
batch attempts; an error that exits midway through a batch does not add a partial
row. Collection adds no graph gathering or collective reductions. It stores only
a logarithmic number of rows per system and calibration trial.

For communication measurements, configure with `-DIMMERSX_BUILD_BENCHMARKS=ON`
and build `vascular_network_profile` (or `vascular_network_profile_debug`). On
Unix MPI builds, these optional executables run the same standalone entry point
with standard PMPI interposition. The normal application and library do not link
the interposition code. Set `IMMERSX_MPI_PROFILE_DIRECTORY` to an existing,
absolute output directory before launching. Each process writes `mpi-RANK.csv`
before MPI finalization; existing files are never overwritten. An unset or empty
value disables profiling. A measurement is incomplete if any rank's file is
missing or the application reports an output error.

The counters cover the explicitly listed functions in that CSV, from completed
MPI initialization to the start of finalization, including application setup and
output. They record successful calls, time inside those calls, point-to-point
posted send bytes and collective input bytes. Bytes count logical datatype
contents; they are not measured wire traffic. Collective input counts each
process's contribution, including the root's broadcast/scatter input, without
estimating the MPI implementation's communication algorithm. Receive calls and
waits contribute call counts and time, not additional send bytes. Nonblocking
call time measures initiation; completion time appears in the corresponding
wait/test. Unlisted MPI functions and MPI initialization/finalization are not
included. Counter updates and datatype queries add profiling overhead to the
application's wall time. Compare measurements made with the same instrumentation.

The focused CTest tests `ImmersX.VascularProfile.*.MPI2` compare counters against
known native MPI payloads, including in-place collectives and disabled forwarding.
The profiler adds no mandatory dependency and does not collect physical graphs.

## Reproducible MPI measurements

`scripts/run_vascular_benchmarks.py` launches the instrumented C++ application
sequentially across declared process counts and random seeds. It copies the
executable and ImmersX shared library into a new build-tree run directory so a
later rebuild cannot change an experiment already running. The manifest records
binary and script hashes, source revision and changes, MPI version, linked
libraries, CPU information, affinity and thread settings. Each case retains the
exact command, binding report, input and effective parameters, native tables,
per-rank profiles and independently checked measurements in `case.json`.

For example, with OpenMPI and a Release build:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DIMMERSX_BUILD_BENCHMARKS=ON
cmake --build build-release --target vascular_network_profile -j
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 scripts/run_vascular_benchmarks.py \
  --executable build-release/benchmarks/vascular_network_profile \
  --library build-release/libimmersx_lib.so \
  --output build-release/vascular-strong \
  --kind strong --terminals 1000,10000 --ranks 1,2,4,8 --seeds 23,24,25
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 scripts/run_vascular_benchmarks.py \
  --executable build-release/benchmarks/vascular_network_profile \
  --library build-release/libimmersx_lib.so \
  --output build-release/vascular-weak \
  --kind weak --terminals 1000 --ranks 1,8 --seeds 23,24,25
```

Use `--mpiexec` to select the MPI launcher matching deal.II. The default launch
arguments are OpenMPI's `--bind-to core --report-bindings`; repeat
`--mpi-argument=VALUE` to replace them for another MPI implementation. Use enough
allocated cores and avoid competing jobs while measuring. These experiments can
be expensive: the per-case timeout defaults to four hours and is configurable
with `--timeout` in seconds. Use smaller terminal counts for a driver smoke test.

Strong scaling keeps the unit cube, 512 background cells, terminal count,
fixed seed radius `0.003 m`, reference flow `2e-9 m^3/s` and other parameters
unchanged across process counts. Weak scaling uses cube process counts
`P=1,8,64,...`: side length `P^(1/3) m`, `512P` background cells, `TP` terminals
for the supplied base count `T`, flow `2e-9 P m^3/s`, and seed radius
`0.003 P^(1/3) m`. This preserves background cell size, terminal density, flow
per terminal and Murray leaf radius across the weak cases. Each individual run
keeps its seed radius fixed and starts from exactly one real inlet at the same
relative position `(0.2,0.5,0.5)`. The seed length remains `0.15 m`, service
distance `0.2 m`, viscosity `0.004 Pa s`, density exponent zero, Murray exponent
three and inlet reference pressure `12000 Pa`. These are declared synthetic
experiments, with fixed terminal counts and no volume-fraction calibration or
minimum-coverage target. The growing domain and single-root propagation make
weak scaling a measurement of this protocol, not an assumption of independent
per-rank trees.

Validation checks unique owned identifiers, connected acyclic single-inlet
topology, exact leaf counts, Murray normalization, positive lengths, cube wall
containment, junction angles, uniform terminal flow, pressure drops and exact
volume/resistance preservation after subdivision. It also checks ownership and
growth counters and requires an MPI profile from every process. The recorded
quality distributions include physical vessel lengths, radii, junction angles
and terminal distances from the cube centre. Coverage and volume fraction come
from the native background integration. No claim of physiological adequacy is
made from those geometric measures.

`growth` in `case.json` reports maxima of local phase times in each leaf-count
range and the fraction of rank-batches owning mutable vessels. The separate
`growth_rank_totals` retain local sums across ranges; use their maximum when
reporting a whole-growth phase time. Do not sum per-range maxima as if they were
a single rank's elapsed time. `mpi.seconds` is a sum of rank times inside the
listed MPI functions and can exceed elapsed wall time. `ranks` retains per-rank
resident and peak resident memory. `owned_vessel_max_over_mean` measures final
physical ownership imbalance; it does not describe startup activity.

A new run never overwrites an existing directory. Pass `--resume` with identical
arguments to recheck completed cases and continue an experiment. A case left
running, timed out or failed requires an explicit audit; the runner does not
infer that its processes have stopped or restart it automatically. Keep failed
runs and their diagnostics when reporting measured outcomes.


## Prepare a generated network for native MetricFlowX

The generator has no MetricFlowX dependency. Applications that enable the
optional package can include `<immersx/vascularization/metric_flow_x.h>` and
call `prepare_metric_flow_x()` after generation succeeds. This requires a
MetricFlowX installation providing `BloodFlowSystem::NetworkPhysics` and
`setup(const NetworkPhysics &)`. Older installations continue to support the
ordinary imported-input application path; selecting generation reports the
missing native API before growing a network. Build MetricFlowX with
`-DMETRIC_FLOW_X_BACKEND=Trilinos` for this checkout, whose ImmersX linear
algebra configuration selects Trilinos. Both packages must use the same backend
and deal.II installation. The supporting native API is proposed in
[MetricFlowX #86](https://github.com/luca-heltai/metric-flow-x/pull/86).

Construct the final background first, then generate, validate and prepare the
flow problem. For example, with a configured native Square root wall model:

```cpp
using namespace ImmersX::Vascularization;
using Flow = MetricFlowX::BloodFlowSystem<1, 3>;
Flow flow(comm);
flow.initialize_params(parameter_file);
auto result = generate(background, mapping, {}, parameters, comm);
AssertThrow(result.validation.valid(), dealii::ExcMessage("Generation failed"));
auto transfer = prepare_metric_flow_x(
  flow, result,
  [](Id, const Vessel &) {
    Flow::VesselProperties material;
    material.E = 5.e5;       // Pa, synthetic
    material.h_wall = .0003; // m, synthetic
    material.p_d = material.p0 = 0.;
    return material;
  },
  [](Id, const Node &, const Terminal *) {
    // R1, R2 [Pa s/m^3], C [m^3/Pa], outlet pressure [Pa], label.
    return Flow::RCRPhysics{0., 1.e8, 0., 0., "synthetic"};
  });
```

Both callbacks run only at physical owners. The first receives a physical
vessel ID and its immutable attributes; geometry, length, endpoint radii,
reference areas and viscosity come from the result. The second receives a
physical node ID and a terminal pointer, or a null pointer at a real outlet.
Arterial inlets retain native boundary zero and need no terminal callback.
Material and wall choices must match the native configured model. These
synthetic values illustrate the API; they are not an organ calibration.

The helper constructs the native fully distributed line mesh with execution
owners following the background. It exchanges coefficient tables, identifiers
and endpoint boundary flags, without gathering the complete physical geometry
or interior incidence. The native solver currently replicates coefficient and
terminal tables. This is a remaining memory cost even though the vascular
network and numerical mesh stay distributed.

Physical IDs remain unchanged and may exceed native material-ID capacity in
value. `transfer.vessel_ids`, `boundary_ids` and `terminal_ids` map them to
compact execution IDs. Venous cells and tapered radii are reversed into flow
direction in a temporary mesh input; the physical network retains its original
growth orientation. `transfer.reference(cell->id())` returns the original
physical interval, including after native binary refinement. In a reversed
vessel, local coordinate zero corresponds to the interval's end. Supplied
arteriovenous terminal pairs become native terminal resistive connections;
there is no second 0D model or extra line-cell volume.

The helper rejects a reference area at or below the native numerical floor
`1e-10 m^2`; it does not clip a physical radius. Native setup validates the
material law, terminal records and prepared mesh before distributing flow DoFs.
Use the native `rest` initial-state source. The native VTK-based steady-network
initializer is not available for this in-memory path. Subsequent native
consistent initialization and the ImmersX IDA adapter use the transferred data.

In `metric_flow_x_elastodynamics`, set `/MetricFlowX coupling/Generate vascular
network = true` and provide `/Vascularization/` parameters. The solid domain is
the generator's background; standalone domain and output settings do not create
a second background in this application. Supply the coupling parameters
`Generated vessel Young modulus`, `Generated vessel wall thickness`,
`Generated vessel reference pressure`, `Generated terminal resistance` and
`Generated outlet pressure` in SI units. Young modulus, thickness and terminal
resistance must be positive. The terminal option supplies native R2 with R1 and
C zero; the library callback permits more general native RCR data. The vessel
reference pressure sets native `p_d` once; native `p0` remains zero.

Generation and validation occur before native flow setup, multiplier-space
construction, existing wall coupling and IDA initialization. A quadratic DG
reference-area field represents the physical squared linear radius exactly.
The example uses fully internal walls. Adapting the multiplier to couple only
internal portions of an imported external network remains deferred.

The configured fixture
`gtests/parameters/metric_flow_x_generated_coupled.prm` uses an internal
four-terminal tree, explicitly declared synthetic wall and terminal data, and
nonzero time-dependent inflow. The `MetricFlowXGenerated` application CTest
cases require accepted IDA steps, the requested output times, a nonempty
internal mesh, positive final flow change and positive solid displacement.
The fixture permits 30 nonlinear iterations per IDA corrector, retaining the
same error tolerances. HLL can need more than the imported example's budget
of eight iterations when IDA reuses its Newton matrix. The pressure term
registers both its pressure-coefficient derivative and, with the supporting
native API, the flow-state derivative of HLL weights at fixed pressure.
The separate `MetricFlowX.Vascular` integration cases exercise tapered arterial
and reversed venous inputs, stable large physical IDs, native terminal beds and
nonzero consistent inlet flow on one, two and four ranks. Tests requiring the
new native API are skipped explicitly with an older installed package.


## Publish measured MPI figures

After all planned cases in each experiment have completed, regenerate the
selected benchmark assets from their native outputs:

```bash
python3 scripts/update_vascular_benchmarks.py \
  build-release/vascular-strong build-release/vascular-weak
python3 scripts/update_vascular_benchmarks.py \
  build-release/vascular-strong build-release/vascular-weak --check
```

Supply multiple experiment directories when strong terminal counts were run
separately. A repeated case name is rejected. Each plotted group and seed must
have its one-rank reference case. `--destination /path/to/build/figures`
creates draft assets outside the documentation. The publisher rejects a
destination that overlaps an input experiment, so it cannot overwrite native
measurement manifests.

The publisher verifies the frozen executable and library hashes, requires every
planned case to be complete, reruns the independent native-table checks, and
compares the recomputed measurements with each saved `case.json`. Only then
does it render SVGs and the small `measurements.csv` and `manifest.json`
summaries. Full meshes, tables, profiles and logs remain in the build tree.
`--check` rerenders and compares selected asset bytes against the destination;
it rejects a changed figure or changed native measurements. Use the same
measured run directories for this check. A new benchmark execution yields new
timing samples for review and publication, rather than identical timing figures.

Curves show medians across the supplied seeds; whiskers span their complete
observed ranges. The CSV retains each seed and physical-quality quantiles.
Strong and weak protocols are labeled separately. In strong scaling the plotted
`T(1)/T(P)` is a speedup for this distributed generation protocol. Concurrent
batching changes geometry, accepted work per batch and the number of global
physiological updates across partitions, so it is not an identical-work hardware
scaling experiment. In weak scaling the same ratio compares increasing global
problem sizes under the declared single-inlet protocol.

Resource figures report the maximum per-rank peak RSS, physical ownership
imbalance, summed logical posted-send and collective-input bytes, summed rank
time inside the selected MPI functions, the maximum rank physiological-update
time and startup writer ownership. Logical bytes are not wire traffic; MPI
rank time is not elapsed wall time. The sum of per-rank peaks in the CSV is not a simultaneous node-memory
measurement and double-counts shared pages. Startup writer ownership counts mutable vessel owners, not every rank
that samples or produces proposals. Quality figures show observed volume,
coverage, physical lengths and angles, without fitted physiological acceptance
thresholds or a claim of statistical equivalence from a small seed sample.
