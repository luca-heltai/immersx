# Generate a physical vascular network

The vascularization library grows physical vessels inside a prepared 3D
background. It keeps vessel identity separate from numerical 1D cells.
`NetworkData` stores physical node, vessel and tissue-terminal identifiers,
radii, viscosity, system, orientation, ownership and provenance. A successful
`GenerationResult` contains the proposed network, its numerical subdivision,
common tissue territories and validation reports. The caller adopts it only
when `validation.valid()` is true; input data are never modified.

The current execution path requires one MPI rank and a first-order mapping on
an all-tetrahedral or all-hexahedral background. Geometry checks include the
vessel wall and clearance. An ambiguous boundary separation at the finite
geometry budget fails conservatively. Explicit planar inlet/outlet sections
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

## Redistribute physical ownership

For an existing distributed `NetworkData`, `redistribute()` migrates owned
entities against a prepared parallel background and refreshes geometric ghosts.
It is collective, including processes with no physical entities or no owned
background cells. This operation does not generate branches; `generate()` still
requires one rank in the current implementation.

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
actual segment geometry before accepting an insertion.

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
history. The public generator currently submits single-process batches. MPI
validation of simultaneous proposals does not yet provide a distributed
generation/commit loop.

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

The internal arterial growth loop operates on already prepared real attachment
trees with one or several processes. Each process draws raw samples in the
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
still needs public seed/import startup, shared venous territories and distributed
measurement and solver preparation before it replaces the public generator's
single-process execution path. Scaling at the requested sizes has not yet been
measured.
