# Changelog

All notable changes to ImmersX are documented in this file. Each entry
begins with a link to the pull request that introduced the change.

The file is included in the documentation and is published at
`doc/changes.md`. Add new entries under `[Unreleased]` using the
skeleton in `.github/CHANGELOG_TEMPLATE.md`.

## [Unreleased]

### Added

- [PR #375](https://github.com/luca-heltai/immersx/pull/375) Report actual coupled block DoFs in a setup-only run without initializing or integrating the coupled state.

- [PR #375](https://github.com/luca-heltai/immersx/pull/375) Allow supplied microvascular resistances to determine local perfusion while checking the prescribed total reference flow separately for arterial, portal and venous networks, with optional conductance scaling by relative territory demand.

- [PR #375](https://github.com/luca-heltai/immersx/pull/375) Make geometric CCO admissibility optional while retaining hydraulic, graph ownership and topology checks, and save separate input and result vascular datasets for inspection.

- [PR #372](https://github.com/luca-heltai/immersx/pull/372) Co-refine solid and fluid meshes on physical MetricFlowX lift supports with separate axial and circumferential scale checks, retained input fields, consistent hanging-node condensation and a one-cube manufactured-solution sequence.

- [PR #373](https://github.com/luca-heltai/immersx/pull/373) Write the lifted wall of `LiftedElasticity` and `LiftedPoisson` as particles, one per lifted quadrature point. A `Particle output/Properties` list selects the stored quantities: weight, mode shapes, modal coefficients, lifted value, prescribed modal data, representative point, radius, quadrature indices and the interpolated background field.

- [PR #369](https://github.com/luca-heltai/immersx/pull/369) Add an optional energy projection that removes two fictitious interior elastic and viscous modes in a straight uniform-radius vessel while retaining native inertia, with analytical checks, measured error tables, coarser MMS multiplier meshes, portable serial/MPI runs, and an early error for MetricFlowX packages missing the pressure-state Jacobian API.

- [PR #370](https://github.com/luca-heltai/immersx/pull/370) Save separate diagnostic arterial, portal and venous VTK datasets when liver growth is rejected, preserving crossings and collapsed edges without adopting the invalid network.

- [PR #367](https://github.com/luca-heltai/immersx/pull/367) Continue portal and venous CCO growth from a physical checkpoint while retaining the completed arterial forest and matching its existing common territories.

- [PR #356](https://github.com/luca-heltai/immersx/pull/356) Allow explicit zero-flow declarations for unpaired anatomical archive ends while retaining strict checks for missing or conflicting microvascular connections.

- [PR #347](https://github.com/luca-heltai/immersx/pull/347) Select composed normal or MUMPS-BLR preconditioning for harmonic reinforcement through solver parameters, with controls for inner CG accuracy, the FGMRES basis, BLR truncation and MUMPS workspace.

- [PR #353](https://github.com/luca-heltai/immersx/pull/353) Add the ParaView Area Tube plugin to visualize line datasets with radius computed from cross-sectional area.

- [PR #351](https://github.com/luca-heltai/immersx/pull/351) Select bounded replicated direct inverses for coupled flow fields and materialize small Schur operators per Jacobian snapshot while checking every inner solve against the original action. Configure these policies in the IRCAD liver profiles without changing physical parameters or solver tolerances.

- [PR #345](https://github.com/luca-heltai/immersx/pull/345) Allow IDA to stop at each output time while retaining its BDF history, so nonlinear algebraic constraints are evaluated at internal step endpoints.

- [PR #345](https://github.com/luca-heltai/immersx/pull/345) Allow linear weak terms with fixed geometry, support and homogeneous constraints to retain their first assembly across residual, transpose and matrix requests while keeping state values and affine offsets current.

- [PR #345](https://github.com/luca-heltai/immersx/pull/345) Configure coupled displacement, velocity and wall-pressure absolute tolerances in their physical units, and reuse native MetricFlowX component tolerances for its mixed flow state.

- [PR #345](https://github.com/luca-heltai/immersx/pull/345) Load archived arterial, portal and venous CCO networks in the coupled flow–tissue application, retain their common microvascular compartments, and discretize their edges against the final tissue mesh. Report block DoFs, local mesh scales and physical terminal diagnostics. Save exact generated liver volume meshes and support mesh-only preparation.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Export the coupled solid elastic and Robin spring energy and prescribed-load work at accepted output times.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Export reference pressures and boundary inflows in liver node tables, and reference resistances and flows in liver vessel tables.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Support scalar and vector Robin boundary data in Poisson, static and transient elasticity, harmonic reinforcement, lifted problems and Navier–Stokes, including their execution adapters and coupled applications. Allow spatial coefficients and per-boundary overrides, validate mixed boundary selections collectively, and expose scalar and solid Robin terms in Coral graphs. Add pressure/conductance conditions to vascular reference solves, with pressure anchoring and the prescribed-flux limit.

- [PR #339](https://github.com/luca-heltai/immersx/pull/339) Download the brain and vessel grids from release v0.3 into a gitignored data directory, verify their SHA-256 checksums, and reuse verified files.

- [PR #332](https://github.com/luca-heltai/immersx/pull/332) Name multiplier load operators, vessel elasticity postprocessing and prescribed vessel pressure explicitly. Preserve the existing C++ types, matrix storage, assembly overloads and parameter spellings as compatibility interfaces.

- [PR #336](https://github.com/luca-heltai/immersx/pull/336) Precondition harmonic adapter solves with the standalone elasticity surrogate, including shared reinforcement, using one elasticity AMG hierarchy for both complex components and FGMRES for tolerance-stopped block-diagonal inner solves.

- [PR #330](https://github.com/luca-heltai/immersx/pull/330) Show every recorded benchmark case in the performance overview, with separate compatible history charts and explicit unavailable native comparisons and failed measurements.

- [PR #328](https://github.com/luca-heltai/immersx/pull/328) Integrate nonmatching multiplier couplings on overlapping quadrature samples and impose a homogeneous multiplier mass equation on all exterior samples, including partially covered cells. Support fixed reference geometry in continuity constraints, lifted and modal interactions, fiber dynamics, legacy point couplings, and MetricFlowX coupling while preserving physical material ids.

- [PR #327](https://github.com/luca-heltai/immersx/pull/327) Classify source cells against a distributed background using quadrature or supplied physical points, assign material ids for outside, partial and inside coverage, and synchronize ghost-cell labels.

- [PR #325](https://github.com/luca-heltai/immersx/pull/325) Build flow-oriented vascular meshes directly from immutable physical data, transferring endpoint boundary identifiers only with the numerical cells that need them.

- [PR #314](https://github.com/luca-heltai/immersx/pull/314) Solve arterial, portal and venous networks through a common microvascular pressure in each tissue territory. Report each supply, venous drainage and compartment pressure, and prescribe demand-weighted reference flows separately for each system. Grow all three geometric networks toward the same territories with a configurable supply fraction, retain common compartments in physical archives, and import optional portal trees through the liver application.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Generate three-dimensional Domain meshes from closed surface files using the Gmsh API, with configurable edge length and optional conforming hexahedra.
- [PR #312](https://github.com/luca-heltai/immersx/pull/312) Keep vascular candidate and obstacle centerline indices across growth batches, reuse boundary geometry during commit certification, report per-rank growth and final discretization operation counts, and resolve equal-distance vascular and venous-territory candidates by physical identifier. Reuse native nearest cursors while venous outlets reject or claim candidate territories.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Prepare native MetricFlowX meshes and physics from validated vascular results without collecting the complete graph. Keep physical identifiers separate from compact execution ids, preserve tapered geometry, orient venous cells along flow and reuse native terminal resistors. Generate internal networks on the final solid mesh before setting up the existing elastodynamics coupling and IDA.
- [PR #308](https://github.com/luca-heltai/immersx/pull/308) Support adaptive refinement (`fixed_number`, `fixed_fraction` and `global`) in elasticity with tensor-product coupling. When the co-refinement of the coupling has space pre- or post-refinement cycles, the co-refinement prevails and the remaining adaptive refinement cycles are skipped.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Report local vascular growth activity and wall times across leaf-count ranges, including startup ownership, proposal selection, redistribution, physiological updates and joint geometry certification. Preserve measurements from unsuccessful calibration trials without collecting graph data.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Generate, complete and reload vascular networks through a parameter-driven MPI application. Prepare uniform or locally refined cube and sphere backgrounds, write owned physical and numerical tables with parallel meshes, and report reference flow, calibration and per-rank timings and memory.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Complete vascular networks with their input territory and microvascular-bed metadata. Preserve untouched beds in both reference solves, size new beds against their actual pressure effects, and reject extension of only one endpoint of an existing pair.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Save physical vascular networks, common territories and positive terminal beds in distributed archives, preserving stable integer ids and all physical attributes. Reload with a different process count through native background ownership, without gathering the graph, and reject incomplete files or inconsistent authority collectively.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Start arterio-venous generation collectively from real outlets and selected imported venous leaves, preserving fixed input properties and stable port ids. Assign common territories exclusively, certify concurrent initial branches, and calibrate against global volume and dual coverage.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Build common-territory terminal pairs through sparse MPI exchanges, derive positive beds from remote endpoint pressures, and verify actual paired flow at each territory owner while preserving boundary data on unpaired input terminals.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Construct demand-based terminal loads collectively from owned arterial and venous records, with independent global flow normalization, remote port pressures and consistent explicit selections. Preserve representable flows across extreme demand scales without clipping.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Solve physical vascular Poiseuille networks and positive 0D beds collectively with native distributed algebra. Resolve remote bed endpoints, preserve small pressure drops with resistive-distance offsets, and report actual flow, shear, perfusion and independent global conservation checks without gathering the graph.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Calibrate arterial terminal counts across MPI partitions using global volume and coverage, preserving nonmonotone bounded search, immutable imported geometry and failed-trial diagnostics. Reject inconsistent controls and remote input failures collectively.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Complete imported arterial networks collectively from selected native MPI endpoints, preserving input geometry and properties. Discover missing leaf records, reserve stable integer identifiers by batch and retain incident vessel ghosts at exterior endpoint owners.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Start arterial generation from one real seed at its native background owner and grow it through the public API across MPI partitions. Certify the first branch collectively, preserve its fixed radius, and reject inconsistent controls without adopting a trial.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Measure global vascular volume fraction and arterial, venous and joint tissue coverage across MPI partitions. Exchange terminal service centres independently of vessel ghost halos, integrate owned background cells once, and retain mapping-aware domain volume independently of coverage quadrature.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Build native fully distributed vascular line meshes with background-aligned cell owners and vertex-adjacent ghosts. Preserve shared junction DoFs and physical-vessel references without gathering the complete graph or narrowing physical identifiers into mesh indices.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Subdivide owned physical vessels using exact background-cell crossings exchanged across MPI partitions. Retain exterior portions, assign numerical interval owners by midpoint containment, count interface volume once, and enforce a global numerical cell budget.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Grow prepared arterial forests through concurrent background-owned sampling and sparse global physical-vessel searches. Preserve one real root across MPI partitions, batch independent insertions, and report finite sampling failure without adopting incomplete results.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Optimize vascular bifurcations from physical ancestor paths retrieved from their MPI owners. Reuse the serial CCO objective and local search, including weighted demand, branch viscosities and the global flow shares of other roots.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Stage concurrent vascular bifurcations at their physical owners, carry root controls through ownership changes, and certify geometry after joint Murray normalization. Discard the entire batch when an ownership, radius or geometry check fails.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Normalize generated vascular forests by weighted terminal demand through distributed owner exchanges, preserving attachment radii and frozen input vessels. Check flow conservation across partitions and keep arterial and venous normalization separate.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Validate vascular growth proposals in sparse MPI batches with physical-owner consent, global endpoint and wall checks, deterministic collision/write-conflict selection and consistent decisions for every participant. Reuse these checks before accepting serial CCO insertions.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Redistribute physical vascular data using native background point ownership and sparse MPI exchanges. Preserve crossing vessels and integer identifiers, refresh authoritative endpoint and terminal data, and exchange geometric ghosts across as many partitions as vessel radius and search reach require.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Grow arterial and venous forests toward common weighted tissue territories with distinct paired endpoints, independent Murray exponents, cross-system collision checks and multiple imported outlets. Validate positive supplied or pressure/flow-derived microvascular beds against the complete paired reference problem.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Grow Murray trees using configurable positive tissue demand independently of terminal density. Distribute an explicit new-tissue reference flow by demand, include that distribution in the blended generated-vessel objective, and retain desired terminal and attachment flows for reference validation.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Solve the steady vascular Poiseuille reference problem with tapered vessels and positive microvascular terminal resistances. Validate pressure/flow boundary data and paired territories, and report conservation, signed flow, pressure, territory supply, dissipation and endpoint wall shear independently of numerical subdivision.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Declare planar vascular inlet and outlet sections with inward normals. Permit seeds on those sections while checking the retained vessel envelope, sidewall clearance and interior junctions throughout CCO growth. Preserve port geometry independently of physical node identifiers.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Calibrate new vascular terminal counts against independent volume and tissue coverage targets with finite bounds, recorded trials and explicit convergence or budget failure. Detect when fixed imported volume already exceeds the target, and keep domain volume independent of coverage quadrature order.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Measure vascular volume fraction and geometric tissue coverage independently. Evaluate arterial, venous and joint service-distance coverage by physical background quadrature, excluding final ports from tissue service centres.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Discretize vascular networks using actual tetrahedral and bilinear hexahedral cell crossings, retaining exterior input portions and reporting short-cell exceptions. Build native line meshes with physical-vessel references that survive numerical refinement, and integrate the vessel volume inside the background.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Extend selected arterial endpoints of existing networks with fixed-radius CCO subtrees, including multiple inlets and partially exterior input networks. Preserve all input nodes, vessels, radii and physical properties, and inherit new subtree properties from each attachment. Respect declared arterial and venous outlets when discovering endpoints.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Grow vascular trees with deterministic CCO bifurcations, fixed-root Murray radii, spatial terminal weights and mesh-dependent terminal density. Select physical vessels independently of their numerical subdivision and check walls, junction angles and collisions after radius updates.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Generate an initial vascular seed from explicit SI parameters, checking the complete vessel wall against linear tetrahedral and bilinear hexahedral background boundaries.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Represent physical vascular networks with stable identifiers, explicit arterial and venous roles, ownership and provenance. Integrate tapered-vessel volume and Poiseuille resistance independently of numerical subdivision.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Exercise vesselwise displacement and pressure exchange on tetrahedral tissue meshes in pseudocoupling1D, including repeated solves with updated coupling data.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Use simplex meshes in the tissue component of the legacy 1D coupling drivers. Build this component independently of the optional hemodynamics library and check pointwise displacement and pressure exchange in serial and MPI.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run the coupled MetricFlowX wall model on a quadratic tetrahedral solid, including RCR outlets and default or immersed vessel ownership.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Use simplex multiplier elements and volume quadrature for harmonic reinforcement on triangular sheets and full-dimensional simplex meshes, with independent or shared displacement.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run fiber-reinforced elastodynamics on triangular and tetrahedral matrices with immersed line fibers, including backward-Euler and IDA evolution.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Support the Poisson-to-elasticity pressure-traction application on a tetrahedral solid with an immersed axial line.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Exercise coupled Poisson with a triangular bulk, an immersed line, nonzero boundary values, and independent multiplier DoFs.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Select simplex displacement elements automatically in Lifted Elasticity and support strong and weak boundary data on uniform simplex meshes.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run Lifted Poisson on tetrahedral bulk meshes with simplex error integration and physical mappings.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run Reduced Poisson on fully distributed simplex bulk meshes. Prepare line and point-cloud co-refinement before distribution, complete simplex conformity before particle location, and preserve existing mesh parameter names.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Use simplex Taylor-Hood velocity and pressure elements in Navier-Stokes on uniform simplex meshes, with matching integration, boundary interpolation, divergence diagnostics, and output in the native and IDA paths.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run the immersed Laplacian application on uniform simplex meshes using a fully distributed Domain and matching scalar elements, integration, boundary values, coupling, and output. Select the Domain backend after reading the parameters.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run forward Euler elasticity on uniform simplex meshes with P1 displacement elements and a positive row-sum lumped mass. Reject higher simplex degrees before assembly.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run legacy elasticity on uniform simplex meshes with matching displacement elements, cell and boundary integration, boundary constraints, and stress output.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Solve frequency-domain elasticity on uniform triangle and tetrahedron meshes, with matching displacement elements, cell and face quadrature, boundary interpolation, stress integration, and output.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run the IDA elastodynamics adapter on simplex meshes, skipping its unavailable convergence table. Verify nonzero time-dependent displacement and velocity on simplex and hypercube meshes.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run the static elasticity adapter on simplex meshes and retain hypercube error tables.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Solve static elasticity on uniform simplex meshes with automatic simplex displacement elements and matching cell and boundary quadrature. The simplex lifecycle skips the unavailable error table.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Run the Poisson adapter on simplex meshes. Skip its unavailable simplex error table and provide locally relevant values for error integration on hypercube meshes in MPI.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Assemble weak terms, known sources, and multiplier metrics on uniform simplex meshes. Choose matching mappings for finite-element spaces, imported fields, particle search, immersed partitioning, and lifted triangular sheets. Reject mixed cell meshes in Domains and integration supports.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Allow Poisson to solve on uniform simplex meshes with fully distributed ownership, simplex finite elements and quadrature, and a linear geometry mapping.
- [PR #298](https://github.com/luca-heltai/immersx/pull/298) Select elastodynamics materials by cell material id and named tag, with default coefficients, Rayleigh damping, and Kelvin–Voigt viscosity using the same convention as elasticity.
- [PR #292](https://github.com/luca-heltai/immersx/pull/292) Allow the coupled MetricFlowX–elastodynamics application to partition vessels with the solid cell owners through the `immersed` policy, with serial and MPI RCR regression cases.
- [PR #291](https://github.com/luca-heltai/immersx/pull/291) Allow ElastodynamicsSolver to run on simplex (triangle and tetrahedron) meshes with `FE_SimplexP`, simplex Gauss quadrature, and a linear `MappingFE`. Simplex meshes require the fullydistributed Domain triangulation. `run()` skips the error table on simplex meshes, and `compute_error()` throws there.
- [PR #291](https://github.com/luca-heltai/immersx/pull/291) Report the solid boundary ids and the number of vessel cells with a vertex outside the solid mesh at the start of `metric_flow_x_elastodynamics`.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Select harmonic reinforcement dimensions and displacement formulation from parameters: reduced meshes use a normal volume lift, full-dimensional meshes need no lift, and either independent displacements with continuity multipliers or one shared displacement can be solved.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Integrate known volume sources and linear weak terms on independent physical volume supports, retaining nonzero essential offsets.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Lift scalar representative fields and geometric integration supports to volumes in codimension one and two, using the complete surface tangent space for normal sheet extrusion.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Solve harmonic matrix-plus-excess-fiber systems in 2D and 3D with independent meshes, real and imaginary displacement-continuity constraints, frequency sweeps, and displacement and multiplier output.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Compose harmonic elasticity Problems with `LinearAdapter` using separate real and imaginary displacement Fields, material-dependent damping, and phased body and Neumann loads.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Support Field-local absolute tolerances, a history-preserving startup tolerance ramp, and cumulative IDA progress statistics.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Expose all backported IDA solver controls through IDAParameters while preserving existing defaults.
- [PR #286](https://github.com/luca-heltai/immersx/pull/286) Run coupled elasticity without the optional 1D solver, compute vessel-wall pressure with point or tensor-product coupling, and update radial vessel data between solves.
- [PR #277](https://github.com/luca-heltai/immersx/pull/277) Allow local co-refinement to partition embedded cells with their background owners through the `immersed` policy, preserving imported fields on the final partition.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Allow Domain to observe an externally owned triangulation, retaining its imported fields without taking ownership of the mesh.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Allow serial and distributed coupled IDA solves with native deal.II SparseDirectMUMPS when DEAL_II_WITH_MUMPS is enabled, selected through the application parameter file.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Allow the coupled MetricFlowX–elastodynamics application to select a serial UMFPACK solve from its parameter file. Materialize and factorize each IDA Jacobian once, and reuse the factorization for subsequent right-hand sides.
- [PR #271](https://github.com/luca-heltai/immersx/pull/271) Show ImmersX/native deal.II overhead ratios by default in performance charts, with an equal-time reference and an absolute-timing option. Keep incompatible measurements separate and explain unavailable comparisons and single-sample histories.
- [PR #268](https://github.com/luca-heltai/immersx/pull/268) Backport deal.II's expanded SUNDIALS IDA controls, inequality constraints, N_Vector operations, and expert setup access. Use the backport in IDAAdapter and allow configuration before solver initialization.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Allow lifted Poisson modal data to use imported point/cell field aliases, nonlinear expressions, coordinates, time zero, and `pi`/`E`, evaluated from retained ghosted Domain data without another application FE space.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Expose the 3D scalar primitive execution path and modal lift/data, imported-radius, multiplier, coefficient-output, and lifted-output nodes in Coral, with a runnable component-level graph.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Add the compositional `LiftedPoisson` application with configurable bulk and modal FE spaces, imported radii, section-weighted modal constraints, Schur/AL solves, adaptive bulk refinement, and native/lifted output.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Add a component-level Coral static-elasticity graph with nonzero modal data, serial/MPI execution, and configured absolute parameter/output paths.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Add parameter-driven composable static elasticity in 2D and 3D, with independent vector modal constraints, section-weighted Schur/AL execution, pressure loads, imported radius and modal data, heterogeneous materials, boundary conditions, and refinement cycles. Legacy elasticity remains available.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Add amplitude-preserving selected vector lift modes, supplied physical Constraint metrics, and adapter-bound semantic output.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Expose field-preserving domain co-refinement, named refinement parameters, and global reports in the Coral registry, with a fullydistributed Poisson graph tested in serial and MPI.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Provide executable `both` co-refinement configurations for elasticity and reduced Poisson, with serial/MPI integration tests that refine both meshes and retain imported properties through a nonzero solve.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Add local co-refinement of two domains with space, embedded, and both strategies, preserving imported fields through refinement and delayed fullydistributed partitioning.
- [PR #257](https://github.com/luca-heltai/immersx/pull/257) Compare scalar and vector nonmatching penalty matrices with explicit deal.II particle assembly and step-80, including codimension-two and independent volume supports, in serial and MPI. Check distributed ownership and integration measure, and show separate one/two-rank performance history series.
- [PR #254](https://github.com/luca-heltai/immersx/pull/254) Allow parameter files to select the coupled Poisson multiplier, static elasticity displacement, and fiber velocity multiplier finite elements while preserving degree-based defaults.
- [PR #251](https://github.com/luca-heltai/immersx/pull/251) Add a performance history page with native comparisons, commit coverage, diagnostic regressions, and an archived benchmark pipeline that runs before documentation builds. Keep changed benchmark sources and explicit timing scopes in separate comparison series without relabelling historical data. Restrict official archive writes and Pages deployment to trusted master events, and verify batch provenance before writing.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Added a how-to guide for choosing weak-term integration supports.
- [PR #247](https://github.com/luca-heltai/immersx/pull/247) Added background-to-background weak terms over independent codimension-one supports, including nonlinear terms with frozen inputs.
- [PR #246](https://github.com/luca-heltai/immersx/pull/246) Added geometric quadrature supports for fixed tensor-product lifts and linear weak-term assembly on a distributed bulk mesh.
- [PR #244](https://github.com/luca-heltai/immersx/pull/244) Added support-aware nonlinear weak terms over multiple finite element spaces and frozen inputs.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Added explicit embedded-mesh integration supports and native deal.II nonmatching mass assembly for value pairings.
- [PR #242](https://github.com/luca-heltai/immersx/pull/242) Added boundary integration supports and normal, conormal, and tangential expressions for linear weak terms.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Added cell supports for selecting the integration domain of linear weak terms.
- [PR #240](https://github.com/luca-heltai/immersx/pull/240) Added an opt-in benchmark for comparing matching mass assembly with deal.II.
- [PR #236](https://github.com/luca-heltai/immersx/pull/236) Add an explicit forward-Euler elasticity solver with a lumped mass matrix.
- [PR #235](https://github.com/luca-heltai/immersx/pull/235) Add a frequency-domain elasticity solver.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Add Coral operations for domain triangulations, VTK output, and output handlers.
- [PR #231](https://github.com/luca-heltai/immersx/pull/231) Added primitive static elasticity boundary conditions.
- [PR #229](https://github.com/luca-heltai/immersx/pull/229) Added Poisson primitive right-hand-side terms.
- [PR #228](https://github.com/luca-heltai/immersx/pull/228) Exposed compositional Coral boundary conditions.
- [PR #227](https://github.com/luca-heltai/immersx/pull/227) Exposed compositional Coral finite-element expressions.
- [PR #225](https://github.com/luca-heltai/immersx/pull/225) Added time-aware aggregable boundary conditions.
- [PR #224](https://github.com/luca-heltai/immersx/pull/224) Added solver-neutral known terms and boundary integration.
- [PR #220](https://github.com/luca-heltai/immersx/pull/220) Added the generic primitive scalar Poisson path.
- [PR #217](https://github.com/luca-heltai/immersx/pull/217) Added the solver-neutral `OutputHandler`.
- [PR #195](https://github.com/luca-heltai/immersx/pull/195) Added the KINSOL steady nonlinear adapter.
- [PR #193](https://github.com/luca-heltai/immersx/pull/193) Added nonlinear Observables and the MetricFlowX radial law.
- [PR #190](https://github.com/luca-heltai/immersx/pull/190) Implemented the tensor-product lifting backend and cleaned up transposes.
- [PR #184](https://github.com/luca-heltai/immersx/pull/184) Composed unified constraint sums.
- [PR #183](https://github.com/luca-heltai/immersx/pull/183) Supported scaled weak-term observables.
- [PR #178](https://github.com/luca-heltai/immersx/pull/178) Added the cached nonmatching weak-term backend.
- [PR #175](https://github.com/luca-heltai/immersx/pull/175) Added the finite-element space and Observable foundation API.
- [PR #174](https://github.com/luca-heltai/immersx/pull/174) Restored the standalone Poisson and ElasticStatic execution paths.
- [PR #173](https://github.com/luca-heltai/immersx/pull/173) Supported mixed-dimensional elastodynamics execution paths.
- [PR #168](https://github.com/luca-heltai/immersx/pull/168) Completed two-way MetricFlowX vessel-wall coupling.
- [PR #167](https://github.com/luca-heltai/immersx/pull/167) Added one-way MetricFlowX-Elastodynamics coupling.
- [PR #164](https://github.com/luca-heltai/immersx/pull/164) Added application-consistent initial conditions to the IDA adapter.
- [PR #161](https://github.com/luca-heltai/immersx/pull/161) Added optional MetricFlowX integration.
- [PR #132](https://github.com/luca-heltai/immersx/pull/132) Supported active and frozen finite-element expression sources.
- [PR #131](https://github.com/luca-heltai/immersx/pull/131) Added reusable imported finite-element fields.
- [PR #130](https://github.com/luca-heltai/immersx/pull/130) Coupled Poisson gradients and imported path fields to elasticity.
- [PR #125](https://github.com/luca-heltai/immersx/pull/125) Composed expression representations with the tensor-product lift.
- [PR #124](https://github.com/luca-heltai/immersx/pull/124) Inferred gradient sampling requirements for expressions.
- [PR #122](https://github.com/luca-heltai/immersx/pull/122) Added the scalar expression representation.
- [PR #121](https://github.com/luca-heltai/immersx/pull/121) Added retained finite-element sampling operators.
- [PR #120](https://github.com/luca-heltai/immersx/pull/120) Added the symbolic expression kernel.
- [PR #118](https://github.com/luca-heltai/immersx/pull/118) Added the parameterized tensor-product lift.
- [PR #116](https://github.com/luca-heltai/immersx/pull/116) Productionized the parameterized ElasticStatic problem.
- [PR #115](https://github.com/luca-heltai/immersx/pull/115) Added the coupled Poisson-to-elasticity vertical slice.
- [PR #107](https://github.com/luca-heltai/immersx/pull/107) Added geometry-aware Representation lifting.
- [PR #106](https://github.com/luca-heltai/immersx/pull/106) Added the weak `CouplingOperator` boundary.
- [PR #103](https://github.com/luca-heltai/immersx/pull/103) Added the standalone `ElasticStaticProblem`.
- [PR #102](https://github.com/luca-heltai/immersx/pull/102) Added the representation-driven load interaction.
- [PR #99](https://github.com/luca-heltai/immersx/pull/99) Added mixed Field component views.
- [PR #98](https://github.com/luca-heltai/immersx/pull/98) Added the minimal Field representation concept.
- [PR #94](https://github.com/luca-heltai/immersx/pull/94) Added deal.II semidiscrete contributors and automatic IDA composition.
- [PR #92](https://github.com/luca-heltai/immersx/pull/92) Added composable semidiscrete physics adapters.
- [PR #91](https://github.com/luca-heltai/immersx/pull/91) Added full-order fiber-reinforced elastodynamics.
- [PR #86](https://github.com/luca-heltai/immersx/pull/86) Added distributed IDA adapters for the standalone PDEs.
- [PR #85](https://github.com/luca-heltai/immersx/pull/85) Added the standalone transient Navier-Stokes solver.
- [PR #78](https://github.com/luca-heltai/immersx/pull/78) Added the representation-driven Poisson coupling prototype.
- [PR #75](https://github.com/luca-heltai/immersx/pull/75) Added the dimension-generic Poisson and Laplace-Beltrami solver.
- [PR #71](https://github.com/luca-heltai/immersx/pull/71) Allowed a zero reduced dimension in the tensor-product space.
- [PR #70](https://github.com/luca-heltai/immersx/pull/70) Made imported VTK fields usable in expressions.
- [PR #69](https://github.com/luca-heltai/immersx/pull/69) Added the VTK-backed reduced-coupling path to elasticity.
- [PR #62](https://github.com/luca-heltai/immersx/pull/62) Supported strong dynamic elasticity constraints.

### Changed

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) A legacy `Representative domain/Inclusions file` reads the radius column only with `Thickness = radius`. Previously the exact default string `0.01` meant the radius column and any other spelling of the same number was a constant.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) `elasticity` and `coupled_elasticity` honour `Immersed Problem/Output every n steps` for transient runs; they previously wrote VTU output at every step. Its default is now 1 for these and for `elasticity_explicit`, which used 100.

- [PR #362](https://github.com/luca-heltai/immersx/pull/362) Index CCO candidate obstacles with their own radius envelopes while retaining the existing trial-radius, taper and clearance checks.

- [PR #348](https://github.com/luca-heltai/immersx/pull/348) Reuse Schur operator graphs and lifted trial derivatives during coupled IDA Jacobian setup, preserving the equations, meshes, time intervals and solver tolerances.
- [PR #351](https://github.com/luca-heltai/immersx/pull/351) Set durations, output periods and MPI defaults for each liver profile, reject time increments that cannot advance the start time, and distinguish audited startup intervals from the earlier full beat in run metadata and the tutorial.

- [PR #342](https://github.com/luca-heltai/immersx/pull/342) Reuse deal.II MeshWorker scratch data for linear weak-term cell, boundary, and affine assembly, retaining local buffers between cells and evaluating quadrature weights and face basis values once per point.

- [PR #325](https://github.com/luca-heltai/immersx/pull/325) Release vascular bootstrap and generated-input data after native preparation, retain only live topology in incremental branch records, and construct selected tissue flow conditions without copying the physical graph.

- [PR #324](https://github.com/luca-heltai/immersx/pull/324) Avoid spatial assembly for zero harmonic material coefficients while retaining essential boundary equations, including zero and purely viscous materials.

- [PR #322](https://github.com/luca-heltai/immersx/pull/322) Solve native and IDA elastodynamics with FGMRES and a physical Schur block factorization using CG/AMG inner solves, retaining the mass hierarchy and reusing unchanged Schur combinations.
- [PR #320](https://github.com/luca-heltai/immersx/pull/320) Register linear weak terms as replayable assembly actions without retaining per-term matrices or cell matrices. Reassemble affine boundary contributions separately, avoid building matrices for zero-vector actions, follow the current constraint structure on each build, and materialize sums with one temporary contribution at a time. The standard iterative LinearAdapter solve owns and reuses one assembled system matrix.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Separate optional MPI communication measurements by growth run, system, terminal range and phase without adding communication to production growth.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Share immutable background geometry and native routing across startup, arterial and venous growth, and calibration attempts while keeping each attempt's tissue state independent.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Apply joint integrated territory demands throughout ordinary growth. Update native rows, physiology and geometric certificates together; retain persistent spatial indices and validate the full physical result once at output.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Stage topology and integrated demand changes through a sparse physiology journal, retaining branch mesh constraints and root normalizations across native ownership changes.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Reuse prepared background routing for cell queries and retain ancestor port metadata during candidate optimization instead of rebuilding or rescanning it for each trial.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Certify walls and physical endpoint containment through shared native background searches for edited or enlarged vessels, with physical padding and reusable domains for unchanged geometry.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Stage native vascular rows with sparse before-images and endpoint reference counts. Restore original map nodes on rejection and remove endpoint ghosts when no owned vessel or terminal needs them.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Select pending venous territories by sorted identifier rank in logarithmic time, preserving the proposal order after served territories are removed.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Subdivide vascular profiles with monotone metric-cut traversal and bounded interval queries. Split and merge numerical cuts through stable iterators while preserving the existing length bounds, ownership and budget failures.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Prepare vascular ownership changes from sparse split patches, localizing only new nodes and edited vessel midpoints through the native distributed search. Preserve endpoint authorities, physical properties and the accepted snapshot on failure.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Query the native cached cell index for vascular crossing containment and sweep overlapping MPI interval replies instead of rescanning all cells and intervals for every cut. Preserve finest-cell length scales and lowest-rank execution ownership.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Reuse the latest isolated tissue-demand journal when a growth batch selects only that proposal. Retain one journal at most and recompute joint territories for multiple selected proposals.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Prepare vascular seeds and imported attachment branches without physiological evaluation, then assign all initial territorial integrals before the first Murray normalization and flow calculation, including single-terminal requests.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Default vascular generation to one territory per active cell, volume-based density exponent three and local length coefficients. Derive an automatic calibration maximum from the scaled request instead of imposing a fixed terminal cap. Start veins with a fixed-length stem and a continuation that preserves series flow and Murray radii.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Base terminal proposal density on physical background cell volume so exponent three assigns equal proposal weight per cell, including cells with different aspect ratios.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Limit concurrent vascular proposal comparisons to overlapping vessel envelopes and shared physical identifiers. Preserve exact collision and junction checks, owner votes and deterministic conflict decisions during joint radius validation.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Use one batched growth engine for arteries and veins. Assign venous proposals to common-territory owners, preserve the arterial demand at offset endpoints, and reject duplicate or inconsistent territory claims across MPI owners.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Use batched CCO execution for public arterial generation and calibration. Check local-search trials against owner-provided vessel geometry with their proposed Murray radii, retaining the best feasible trial.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Interpret the harmonic reinforcement root dimension as the reinforcement mesh dimension and the space dimension as the matrix dimension. Use dimension one for a line reinforcement in 2D or 3D.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Share one frequency list between the harmonic matrix and fiber, and preserve the actual initial refinement when saving and reloading either supported parameter alias.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Support harmonic elasticity on embedded domains with `1 <= dim <= spacedim <= 3` and independent parameter subsections. Grid settings use `Domain`, while existing grid and refinement entries remain accepted.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Configure standalone and coupled MetricFlowX IDA integration, time bounds, output cadence, and tolerance ramps in the native `MetricFlowSystem<1, 3>/IDA parameters` section. Remove unused fixed-step controls from the coupled configuration.
- [PR #287](https://github.com/luca-heltai/immersx/pull/287) Refresh eliminated Dirichlet contributions automatically when evaluating supported linear weak-term residuals, reusing the matrix while constraint entries stay fixed. Remove `with_lifted_affine_updates()` from the public API.
- [PR #287](https://github.com/luca-heltai/immersx/pull/287) Write coupled MetricFlowX test and tutorial output in one directory per case, using `solid`, `flow`, and `pressure_jump` filenames.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Keep fiber IDA composition and state local to one solve, write the multiplier through semantic output, and move the uncoupled reference solve to validation tests.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Keep the Poisson-elasticity application focused on pressure traction and semantic output, with residual and traction checks in integration tests.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Simplify coupled Poisson setup and write its solution and multiplier fields directly through semantic output handlers.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Use fixed lower and upper diameter bounds for local mesh adjustment; the old refinement factor remains a deprecated single-mesh input.
- [PR #250](https://github.com/luca-heltai/immersx/pull/250) Include matrix materialization in trace assembly timings and report synchronized maximum-rank MPI timings with explicit setup, action, and operation-count metadata.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Reuse trial and test basis values during linear assembly and input FEValues across cells with a shared mapping during nonlinear assembly.
- [PR #237](https://github.com/luca-heltai/immersx/pull/237) Replace custom MetricFlowX coupling with structural finite-element observables and cell integration.
- [PR #226](https://github.com/luca-heltai/immersx/pull/226) Associated boundary conditions with semantic fields.
- [PR #222](https://github.com/luca-heltai/immersx/pull/222) Returned typed semantic fields from the contributors.
- [PR #215](https://github.com/luca-heltai/immersx/pull/215) Replaced the 2D Coral facade with generic composition.
- [PR #214](https://github.com/luca-heltai/immersx/pull/214) Unified the Coral parameter-loading nodes.
- [PR #213](https://github.com/luca-heltai/immersx/pull/213) Introduced `Domain`-based mesh parameters and migrated the solvers.
- [PR #204](https://github.com/luca-heltai/immersx/pull/204) Refactored time parameter and integrator ownership.
- [PR #200](https://github.com/luca-heltai/immersx/pull/200) Added the deliberate application-coupling roadmap.
- [PR #199](https://github.com/luca-heltai/immersx/pull/199) Reduced memory use in elasticity.
- [PR #189](https://github.com/luca-heltai/immersx/pull/189) Used typed deal.II finite-element expressions for weak terms.
- [PR #188](https://github.com/luca-heltai/immersx/pull/188) Completed the unified fiber-constraint follow-up.
- [PR #187](https://github.com/luca-heltai/immersx/pull/187) Migrated the fiber driver to unified weak-term constraints.
- [PR #185](https://github.com/luca-heltai/immersx/pull/185) Moved the normal pressure load into `weak_term`.
- [PR #181](https://github.com/luca-heltai/immersx/pull/181) Unified Lagrange multiplier constraints over weak terms.
- [PR #176](https://github.com/luca-heltai/immersx/pull/176) Optimized weak-term assembly for simple finite-element paths.
- [PR #166](https://github.com/luca-heltai/immersx/pull/166) Refactored time parameter ownership around `TimeParameters`.
- [PR #162](https://github.com/luca-heltai/immersx/pull/162) Added parameter objects to the execution adapters.
- [PR #160](https://github.com/luca-heltai/immersx/pull/160) Modernized applications around the semantic execution adapters.
- [PR #159](https://github.com/luca-heltai/immersx/pull/159) Gave multiplier Interactions native output ownership.
- [PR #158](https://github.com/luca-heltai/immersx/pull/158) Wrote native output after composed solves.
- [PR #157](https://github.com/luca-heltai/immersx/pull/157) Simplified linear operator composition.
- [PR #152](https://github.com/luca-heltai/immersx/pull/152) Repaired the final semidiscrete composition stack.
- [PR #138](https://github.com/luca-heltai/immersx/pull/138) Hardened field transfer and pruned stale reduced-field APIs.
- [PR #128](https://github.com/luca-heltai/immersx/pull/128) Built pressure from deferred finite-element expressions.
- [PR #123](https://github.com/luca-heltai/immersx/pull/123) Supported semantic expression dependencies.
- [PR #117](https://github.com/luca-heltai/immersx/pull/117) Brought `ElasticStatic` to parity with elasticity.
- [PR #114](https://github.com/luca-heltai/immersx/pull/114) Simplified the public multiphysics composition API.
- [PR #113](https://github.com/luca-heltai/immersx/pull/113) Clarified Representation quantity spaces.
- [PR #109](https://github.com/luca-heltai/immersx/pull/109) Consolidated the Representation domain, evaluation, and transfer boundaries.
- [PR #108](https://github.com/luca-heltai/immersx/pull/108) Separated geometry maps from Representation lifting.
- [PR #105](https://github.com/luca-heltai/immersx/pull/105) Separated the Representation source and evaluation domain.
- [PR #104](https://github.com/luca-heltai/immersx/pull/104) Clarified Representation evaluation and linearization.
- [PR #100](https://github.com/luca-heltai/immersx/pull/100) Made `TensorProduct` a representation-first lifting.
- [PR #97](https://github.com/luca-heltai/immersx/pull/97) Replaced `TimeRole` with per-field differential component masks.
- [PR #83](https://github.com/luca-heltai/immersx/pull/83) Unified semantic residual fields and added mixed DAE integration.
- [PR #81](https://github.com/luca-heltai/immersx/pull/81) Dispatched applications from parameter dimensions.
- [PR #79](https://github.com/luca-heltai/immersx/pull/79) Stabilized the representation and constraint interaction architecture.
- [PR #76](https://github.com/luca-heltai/immersx/pull/76) Separated the reduced-space core from the VTK backend.
- [PR #73](https://github.com/luca-heltai/immersx/pull/73) Improved distributed tensor-product coupling.

### Fixed

- [PR #381](https://github.com/luca-heltai/immersx/pull/381) Share retained linear-operator state across callback copies to avoid copying vectors and nested operators during composition.

- [PR #380](https://github.com/luca-heltai/immersx/pull/380) Share one owned weak-term recipe and target across operator callbacks instead of duplicating their field state, avoiding excessive memory use when registering large coupled terms.

- [PR #375](https://github.com/luca-heltai/immersx/pull/375) Use the native configurable minimum positive area when importing physical vascular radii, retaining the fixed threshold for older MetricFlowX backends.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) The default `elastodynamics` parameter file runs: `Fixed step/Number of time steps` defaults to 10, and the printed `Domain/Initial refinement` default is the 2 actually used. `elastodynamics` no longer prints the unused `IDA` section, and `elastodynamics_adapter` no longer prints the unused `Fixed step` section.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) Give the MetricFlowX copy of every shared `MetricFlowSystem<1, 3>/IDA parameters` entry the value that ImmersX uses and prints. Entries missing from the input previously kept the deal.II defaults on the MetricFlowX side, so consistent initial conditions and the IDA statistics used a different step size, order and tolerances.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) Declare `/MetricFlowX domain/` before the input is parsed, so `Grid scale` and `Triangulation type` can be set and the section is accepted in input files.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) Stop with an error, also in Release builds, when `Inclusions file`, `Data file` or a `file` grid of `elasticity` cannot be opened, when `Read fields = true` names no existing mesh file, when `Coupling type = TensorProduct` is used without VTK, and when `MetricFlowX coupling/Embedded partitioning policy = immersed` is combined with an archived or generated network. These cases previously ran with zero inclusions, without fields, with point coupling or with default partitioning.

- [PR #365](https://github.com/luca-heltai/immersx/pull/365) Keep cell and trace area coefficients positive during coupled flow–tissue IDA solves. Report nonfinite trial residuals collectively as recoverable callback errors instead of passing them to the linear solver; equilibrium initialization may still fail when IDA cannot recover.

- [PR #361](https://github.com/luca-heltai/immersx/pull/361) Certify separation of tapered anatomical vessels using their actual endpoint radii when maximum-radius containers overlap, while retaining current Murray radii for mutable branches and conservative rejection without a separating witness.

- [PR #359](https://github.com/luca-heltai/immersx/pull/359) Retry other nearby vessel candidates when the distributed geometry vote rejects a companion growth target, preserving the same physical checks and finite search budget.

- [PR #358](https://github.com/luca-heltai/immersx/pull/358) Certify separation of flat-ended vascular tubes when their rounded bounding capsules overlap, preserving clearance, junction-angle and domain checks for densely sampled anatomical inputs.

- [PR #357](https://github.com/luca-heltai/immersx/pull/357) Apply up to three MUMPS refinement corrections in IDA using the retained Jacobian action and the existing residual tolerance, so small right-hand sides do not fail solely because the first backsolve is insufficiently accurate.

- [PR #347](https://github.com/luca-heltai/immersx/pull/347) Handle exact Krylov-space closure in LinearAdapter on deal.II 9.7 as well as newer versions, so exact preconditioners converge without NaN residuals.
- [PR #346](https://github.com/luca-heltai/immersx/pull/346) Reject unreadable STL inputs before entering Gmsh and verify that missing MSH or STL files preserve later valid imports in the same session.
- [PR #355](https://github.com/luca-heltai/immersx/pull/355) Rebalance vascular mesh cuts when adjacent intervals can satisfy the existing background length bounds without changing physical vessel geometry.

- [PR #350](https://github.com/luca-heltai/immersx/pull/350) Use the same fixed primal actions in positive-weight Schur construction, elimination and recovery, preserving the physical multiplier row when AMG is approximate.

- [PR #349](https://github.com/luca-heltai/immersx/pull/349) Retain the coupled displacement–velocity inverse when IDA computes consistent initial conditions with zero derivative weight, avoiding singular diagonal AMG setup in coupled elasticity. Use the same fixed pair action in Schur construction and primal recovery so the multiplier row remains consistent when AMG is approximate.

- [PR #345](https://github.com/luca-heltai/immersx/pull/345) Overwrite imported native pressure-operator destinations correctly on ranks without vessel cells, preserving zero and nonzero pressure actions.

- [PR #345](https://github.com/luca-heltai/immersx/pull/345) Check MUMPS corrections against the retained Jacobian, preventing failed native factorizations from silently freezing a coupled trajectory. Allow an explicit workspace margin for the liver profiles.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Reject unreadable Gmsh mesh files before opening them in an existing session, so a missing input does not prevent later valid mesh imports.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Include shared reinforcement Robin springs in the positive harmonic surrogate, using their retained physical boundary support so iterative solves remain anchored without essential conditions.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Preserve small vascular pressure drops next to large pressure offsets by refining the physical flow residual with a separate pressure correction. Keep the original conservation tolerance and total solver iteration budget.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Keep LinearAdapter solves stable near Krylov breakdown with nearly exact preconditioners by selecting modified Gram–Schmidt explicitly on deal.II 9.8 and newer, preserving the requested residual tolerance.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Preserve the cached coupling matrix and the multiplier right-hand-side assembly state when updating a coupled solid at a new time. Keep the coupled displacement and reaction correct on subsequent steps with nonempty inclusions.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Evaluate the legacy coupled solid at the driver's physical time, refreshing Robin springs, applied loads and nonzero essential lifting at every changed time.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Avoid running the tissue solver twice after a legacy 3D/1D coupled run. Execute the pseudocoupled 1D update on its owning rank and reject an MPI-enabled legacy backend with multiple ranks before its incompatible collectives can hang.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Keep quadrature diagnostics on standard error so benchmark output remains valid JSON Lines.

- [PR #340](https://github.com/luca-heltai/immersx/pull/340) Skip empty sparse rows throughout distributed matrix transformations and solver setup, avoiding repeated scans of later rows.

- [PR #336](https://github.com/luca-heltai/immersx/pull/336) Keep memory use linear in the number of contributions when constructing semidiscrete operator sums, avoiding runaway allocation before harmonic fiber-reinforcement solves.

- [PR #329](https://github.com/luca-heltai/immersx/pull/329) Omit exterior reinforcement quadrature points from shared harmonic elasticity, retaining interior samples of partially overlapping fibers and reporting their retained volume.
- [PR #335](https://github.com/luca-heltai/immersx/pull/335) Preserve unnamed Gmsh physical material and boundary ids during Domain imports without changing named material, boundary or manifold selectors. Keep untagged entities, input files and Gmsh output settings intact.
- [PR #328](https://github.com/luca-heltai/immersx/pull/328) Reuse the current linearization in steady Schur and block preconditioners so each application avoids assembling the coupling again. Refresh retained matrices for every solve.
- [PR #328](https://github.com/luca-heltai/immersx/pull/328) Retain affine boundary contributions, full sampled modal Gram matrices, and nominal pressure data when coupling support is partial. Account for the exterior multiplier block in Schur and augmented-Lagrangian solves, and honor the configured inner-solver tolerance in the immersed Laplacian solver.
- [PR #325](https://github.com/luca-heltai/immersx/pull/325) Retain materializable IDA Jacobian contributions per evaluation point and share composed operator callbacks, so iterative global and Schur actions and direct factorization reuse owned matrices without duplicating callback trees or rebuilding contributions during capability checks. Preserve matrix-free actions and fresh residual evaluation.
- [PR #322](https://github.com/luca-heltai/immersx/pull/322) Reuse elastodynamics spatial operators, static loads and boundary constraints until spatial setup changes; build native time systems from cached matrices with affine boundary lifting.
- [PR #320](https://github.com/luca-heltai/immersx/pull/320) Keep caller-owned matrices unchanged when replaying scaled matrix contributions.
- [PR #316](https://github.com/luca-heltai/immersx/pull/316) Open performance charts on the compatible series with the most measurements and show CPU, kernel, point count, and commit range in the series selector.
- [PR #314](https://github.com/luca-heltai/immersx/pull/314) Preserve integrated tissue demand, native startup ownership, cached geometry and mesh-scaled separation and coverage across arterial, portal and venous growth.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Keep liver input lengths in metres and tissue counts absolute when layered on mesh-adaptive generation; retain the independently counted full-commit geometry oracle.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Build surface volume generation and retained vascular centerline and obstacle indices with the deal.II 9.7 and Boost toolchain used by CI, preserving conforming hexahedral output and physical vessel identities during index updates.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Keep new tissue territory identifiers unique when imported leaves have no territory assignment, including arterial growth beside an existing venous tree.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Retain positive tissue demands below the precision of their parent sum when their physical radii and flows remain representable; do not require a strict floating-point increase to accept a new branch.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Keep native parameter snapshots in the selected output directory when MetricFlowX provides an explicit snapshot path, including tests started from the source root. Preserve compatibility with older native initialization APIs.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Preserve native trace couplings in local flow preconditioning across MPI partitions, so generated vascular networks advance with the iterative coupled solver and unchanged error tolerances.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Check the actual linear residual before IDA accepts an FGMRES solve. Correct residual drift within the existing total iteration budget so small coupled vascular loads advance without changing solver tolerances.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Keep newly generated untapered networks at exactly zero initial wall displacement by matching the native area basis. Retain quadratic reference areas when the physical geometry tapers.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Assign a generated vessel's reference pressure once in native MetricFlowX data and balance the coupled example against a nonzero outlet reference pressure.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Include the flow-state derivative of prescribed-pressure HLL fluxes in MetricFlowX wall coupling when the native package provides it, keeping it separate from the derivative with respect to pressure coefficients.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Accept vascular endpoints across warped interior faces at native inverse-map accuracy while keeping strict physical-boundary tests, including nearly exterior thin vessels.
- [PR #301](https://github.com/luca-heltai/immersx/pull/301) Write the HDF5 vessel pressure on every rank, so parallel coupled elasticity runs where some rank owns no vessel no longer hang.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Refresh tissue coupling data through the cached bulk assembly, group pressures by retained multiplier modes, and interpolate gathered pressures without an MPI collective inside the rank-zero callback.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Include prescribed bulk values in the Reduced Poisson coupling right-hand side. Nonzero Dirichlet data now satisfy the same immersed constraint as the complete physical solution.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Condense nonlinear weak residuals through the target constraints and import candidate field states before distributed cell evaluation. Preserve essential rows and evaluate ghost coefficients when the supplied state owns only its local entries.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Retain all prescribed displacement components in the legacy elasticity weak Dirichlet stress terms. This restores consistency on simplex and hypercube meshes.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Identify the physical mesh reference cell collectively when MPI ranks own no cells. Ignore fully distributed artificial placeholders when selecting elements, quadrature, and search mappings; keep field views and geometry accessors local, and reject mixed physical meshes.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Import point and cell fields on simplex meshes with matching finite elements. Preserve those fields during initial refinement and use reference-cell mappings and smoothing rules during Domain co-refinement. Boundary recipes use the target field mapping; point-cloud refinement accepts serial preparation meshes and uses each cell's actual bounding box.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Preserve multiplier mass contributions whose matrix rows belong to other MPI ranks.
- [PR #294](https://github.com/luca-heltai/immersx/pull/294) Avoid division by zero in small fiber multiplier Schur solves with deal.II 9.7, including configurable discontinuous multiplier spaces in serial and MPI.
- [PR #292](https://github.com/luca-heltai/immersx/pull/292) Evaluate lifted observables and nonlinear weak terms with each cell's active finite element and local DoF count, including hp terminal pressure fields. Preserve differentiated fields' execution numbering for compact and permuted layouts and check coupled Jacobians per residual block.
- [PR #292](https://github.com/luca-heltai/immersx/pull/292) Use MetricFlowX's native hp DoFHandler numbering and ghost indices for coupled area and velocity fields, including RCR capacitor pressures represented as a fifth FE component. Remove the conversion for the previous manually numbered capacitor layout.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Preserve saved Coral graphs by keeping known-source factory and execution-node keys unchanged when KnownTerm has a default integration support.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Apply harmonic 1D Dirichlet amplitudes only on physical boundary cells with deal.II 9.7, preserving nonzero complex boundary data across MPI partitions.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Retain essential source offsets in unreindexed nonmatching weak terms, including harmonic continuity between independently meshed full-dimensional materials.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Write harmonic displacement datasets inside the selected output directory even when its path has no trailing separator.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Use the correct Trilinos vector assembly mode in the harmonic complex-Arnoldi updates, which previously failed Debug assertions.
- [PR #289](https://github.com/luca-heltai/immersx/pull/289) Correct LU roundoff in LinearAdapter default direct solves with one residual correction using the cached matrix and factorization, including explicitly reused factorizations.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Report native IDA failures in Release, honor nonzero initial output times, and handle empty ranks, pooled block-layout changes, and callback failures during initialization.
- [PR #287](https://github.com/luca-heltai/immersx/pull/287) Save coupled MetricFlowX fluid output with the accepted simulation times and write the initial state once in each output series. Check fluid, solid, and multiplier output times in the coupled and MMS application tests.
- [PR #287](https://github.com/luca-heltai/immersx/pull/287) Support time-dependent Dirichlet data in the MetricFlowX elastodynamics coupling and use exact displacement and velocity boundary data in the cube MMS tutorials while retaining exact Neumann variants on both meshes.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Apply 1D Poisson Dirichlet data only to non-artificial cells with deal.II 9.7, preserving the pressure solution across MPI partitions.
- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Keep the Coral Poisson–elasticity workflow compatible with the simplified application interfaces and report traction balance from its displacement residual.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Handle exact Krylov breakdown in the default IDA GMRES path with modified Gram-Schmidt, including deal.II 9.7 builds.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Reduce coupled Jacobian setup costs by copying distributed matrices through their native backend, materializing each sum operand once, and avoiding repeated row extraction. Skip quadratic scans of empty blocks when converting the Jacobian to a monolithic matrix. Mark absent operator blocks as zero so block preconditioners skip unnecessary inverse applications.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Apply vessel-wall traction in solid momentum and native external-pressure feedback in flow, including boundary and interface equations, while preserving source-free continuity. Exchange reference-area ghost coefficients before constructing the distributed wall lift. Add an opt-in Schur correction for displacement-velocity coupling, retain small-load corrections, preserve callback errors, and stop on failed IDA steps in Release.
- [PR #264](https://github.com/luca-heltai/immersx/pull/264) Prevent the Navier–Stokes tutorial from timing out in Debug builds by reusing finite-element evaluations during nonlinear Jacobian assembly.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Use one lifted assembly for scalar and vector constraints, preserve constrained master sparsity, and allow explicit source Dirichlet-value updates while retaining collective static-data and constraint-structure guards. Keep scalar coefficient-output and vector coefficient-creation nodes distinct in the combined Coral registry.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Preserve affine Dirichlet contributions in lifted weak terms, assemble known loads and legacy coupling graphs across MPI owners, use relative inner Schur solves and check the true FGMRES residual, and reconstruct hanging-node values before FE output.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Fix vector lifted coupling on locally refined meshes, including nonzero affine boundary data.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Retain static inhomogeneous Dirichlet shifts in lifted weak constraints, update Observable dependencies on semantic registration, and assemble known loads into writable MPI vectors.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Use compiler-independent Coral report-operation keys so co-refinement graphs load on Linux and macOS.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Retain the reduced Poisson augmented-Lagrangian solution in the accepted solution vector as well as the output vector, so solution diagnostics reflect the nonzero solve.
- [PR #254](https://github.com/luca-heltai/immersx/pull/254) Register MetricFlowX time parameters before reading input and provide a mixed-flow preconditioner for coupled IDA solves. Exercise standalone and nonzero-inflow coupled applications in serial and MPI.
- [PR #253](https://github.com/luca-heltai/immersx/pull/253) Preserve every ambient vector component in native nonmatching assembly and give the fiber driver's matrix-only reference independent history.
- [PR #252](https://github.com/luca-heltai/immersx/pull/252) Select all ambient components of embedded vector fields and evolve the fiber driver's matrix-only backward-Euler reference from its own displacement and velocity history.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Use a straight cross-section in the lifted-support benchmark and preserve frozen-input mesh/mapping checks in the nonlinear input cache.
- [PR #247](https://github.com/luca-heltai/immersx/pull/247) Retain off-rank independent-support Jacobian entries and reject frozen inputs on incompatible background meshes or mappings.
- [PR #246](https://github.com/luca-heltai/immersx/pull/246) Request Jacobians for lifted geometry, preserve off-rank physical-support matrix entries, and route quadrature points collectively when a rank has no points.
- [PR #244](https://github.com/luca-heltai/immersx/pull/244) Assemble nonlinear residuals into owned-only vectors, retain off-rank matching and lifted Jacobian entries, and reject frozen inputs on incompatible meshes or mappings.
- [PR #244](https://github.com/luca-heltai/immersx/pull/244) Use ghosted imported inputs and owned-only outputs in distributed frozen-field tests.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Reject unsupported inhomogeneous source constraints in nonmatching and lifted linear pairings, including offsets changed after assembly, and reject inhomogeneous linear multiplier reactions.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Retain relevant rows in nonmatching coupling sparsity.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Preserve all ambient vector components in native nonmatching value pairings, including offset fields and hanging-node constraints.
- [PR #243](https://github.com/luca-heltai/immersx/pull/243) Use particle-based assembly when deal.II's native nonmatching path cannot handle a distributed immersed mesh or a multi-rank background mesh.
- [PR #242](https://github.com/luca-heltai/immersx/pull/242) Include dynamic affine constraint contributions in explicit boundary supports and inferred boundary regions, including normal and conormal contractions.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Keep constrained residual and Jacobian rows condensed for reindexed mixed fields while preserving native diagonals for unreindexed square cell terms.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Preserve and dynamically refresh affine cell contributions in state, frozen-source, and multiplier residuals while retaining native constrained diagonal rows.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Derive locally relevant DoFs for FE-space views by default and retain off-rank matching matrix entries during distributed assembly.
- [PR #241](https://github.com/luca-heltai/immersx/pull/241) Keep persisted Coral weak-term and continuity graphs readable across compilers.
- [PR #238](https://github.com/luca-heltai/immersx/pull/238) Avoid indexed writes to ghosted Trilinos vectors in `OutputHandler`.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Write the short Coral parameter values used by each graph run to a sibling file.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Read point and cell fields from XML VTK mesh files.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Treat grid names with a file extension as mesh files even when the path is unresolved.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Read XML VTK mesh files with VTK versions before 9.3.
- [PR #230](https://github.com/luca-heltai/immersx/pull/230) Fixed Coral first-run parameter initialization.
- [PR #223](https://github.com/luca-heltai/immersx/pull/223) Fixed the Coral CTest filter escaping.
- [PR #210](https://github.com/luca-heltai/immersx/pull/210) Fixed the Schur preconditioner breakdown on lucky termination.
- [PR #208](https://github.com/luca-heltai/immersx/pull/208) Fixed elastodynamics refinement and moving-boundary constraints.
- [PR #207](https://github.com/luca-heltai/immersx/pull/207) Fixed Newmark inhomogeneous boundary handling.
- [PR #205](https://github.com/luca-heltai/immersx/pull/205) Fixed the dynamic elasticity constraint instability.
- [PR #202](https://github.com/luca-heltai/immersx/pull/202) Fixed elastodynamics adapter constrained degrees of freedom.
- [PR #198](https://github.com/luca-heltai/immersx/pull/198) Fixed the inclusion catalog scalar read.
- [PR #191](https://github.com/luca-heltai/immersx/pull/191) Fixed sparse weak-term regression comparisons.
- [PR #180](https://github.com/luca-heltai/immersx/pull/180) Fixed the strict API documentation build.
- [PR #172](https://github.com/luca-heltai/immersx/pull/172) Fixed rotation-aware tensor-product vector modes.
- [PR #163](https://github.com/luca-heltai/immersx/pull/163) Fixed MetricFlowX IDA parameter construction.
- [PR #156](https://github.com/luca-heltai/immersx/pull/156) Fixed distributed Trilinos Schur preconditioner maps.
- [PR #155](https://github.com/luca-heltai/immersx/pull/155) Fixed the Fiber IDA initial solver lifetime.
- [PR #137](https://github.com/luca-heltai/immersx/pull/137) Fixed reduced-coupling assembly and deduplicated imported-field transfer.
- [PR #136](https://github.com/luca-heltai/immersx/pull/136) Fixed elasticity compilation without VTK.
- [PR #135](https://github.com/luca-heltai/immersx/pull/135) Fixed duplicate symbolic outputs in the tensor-product right-hand side.
- [PR #112](https://github.com/luca-heltai/immersx/pull/112) Fixed the matrix-free transfer test build.
- [PR #89](https://github.com/luca-heltai/immersx/pull/89) Fixed parallel elasticity assembly for the coupled case.
- [PR #74](https://github.com/luca-heltai/immersx/pull/74) Addressed a failing test.
- [PR #68](https://github.com/luca-heltai/immersx/pull/68) Removed compiler warnings with deal.II 9.8.
- [PR #67](https://github.com/luca-heltai/immersx/pull/67) Fixed the inclusion basis-function scaling.

### Removed

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) Remove the unused `Constant thickness` lift entry; a numeric `Thickness` is the constant thickness.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) Remove the `Immersed Problem/Time parameters` entries that elasticity never read: `Output frequency`, `Number of time steps`, `Policy` and the IDA subsections `Running parameters`, `Error control` and `Initial condition correction parameters`. `elasticity` and `coupled_elasticity` always advance by `Time step`.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) Give each `Cross section` entry one name. `Maximum inclusion degree`, `Selected indices`, `Inclusion type`, `Number of quadrature points` and `Number of quadrature repetitions` are removed; use `Maximum basis degree`, `Selected modes`, `Geometry type`, `Number of points` and `Number of repetitions`. The flat lift entry `Representative quadrature type` is removed; use `Representative quadrature/Type`. Before this change both names of a pair were printed and the alphabetically later one silently overrode the other.

- [PR #276](https://github.com/luca-heltai/immersx/pull/276) Remove the fiber driver's `matrix_only_displacement_difference()` diagnostic; validation evolves an independent dense reference instead.
- [PR #192](https://github.com/luca-heltai/immersx/pull/192) Purged the transitional representation architecture.
- [PR #186](https://github.com/luca-heltai/immersx/pull/186) Removed the obsolete representation load interaction.
- [PR #96](https://github.com/luca-heltai/immersx/pull/96) Removed the semidiscrete compatibility layers.
- [PR #93](https://github.com/luca-heltai/immersx/pull/93) Removed the obsolete monolithic compatibility layer.

### Documentation

- [PR #383](https://github.com/luca-heltai/immersx/pull/383) Align the newly authored liver setup figure scripts, data and asset license headers with the repository license.

- [PR #382](https://github.com/luca-heltai/immersx/pull/382) Document measured coupled setup counts for separately generated small and medium liver forests, with reproducible block-composition and interval-scale figures and explicit numerical limitations.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) Explain the point, tensor-product, lift and direct-overlap coupling discretisations, the parameter sections each one reads, and which application uses which.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) Describe in the parameter help which coupling section `Coupling type` reads, how `Inclusions`, `Inclusions file`, `Data file` and `Reference inclusion data` combine, when the legacy `Selection of Fourier coefficients` is used, which quadrature the `Representative domain` entries control, and what `Grid generator` means for each `Domain type`.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Document the Robin condition and parameter owner for every application, with static and transient solid examples, physical fiber end sections, lifted scalar data and native terminal flow models.

- [PR #328](https://github.com/luca-heltai/immersx/pull/328) Start GitHub CI after a successful compilation by pushing authorized changes early, while local tests and documentation checks continue in parallel.

- [PR #325](https://github.com/luca-heltai/immersx/pull/325) Register vascular pages in the current documentation sections and distinguish the linear PR stack from archived point-demand measurements.

- [PR #323](https://github.com/luca-heltai/immersx/pull/323) Generate coupled elasticity API documentation without unmatched group warnings.

- [PR #324](https://github.com/luca-heltai/immersx/pull/324) Document the two-minute correctness-test budget, retain the complete source and cost audit, and keep the Navier–Stokes manufactured-solution input as a manual accuracy-study example.

- [PR #322](https://github.com/luca-heltai/immersx/pull/322) Retain the native and IDA elastodynamics performance baseline with revision and artifact checksums, and provide a reproducible comparison driver.
- [PR #316](https://github.com/luca-heltai/immersx/pull/316) Organize site navigation into consistent sections and tutorial groups, fix README documentation URLs, and document page ownership and heading rules.
- [PR #312](https://github.com/luca-heltai/immersx/pull/312) Document the complete vascular cost inventory, invalidation and rollback rules, reproducible four-size performance measurements, memory and communication counts, and inherited arterial/venous convergence limits.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Resolve the refinement-study and measurement-workflow links with explicit documentation labels.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Publish integrated-demand cost checks at 1,000 and 10,000 terminals, with quadrature sizes, memory, operation counts and frozen experiment identities.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Fix the generated-network tutorial link to the native MetricFlowX transfer example.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Explain terminal sampling weights, cell-interface conventions and count normalization separately from numerical vessel resolution.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Describe the completed connection between distributed vascular sampling, candidate optimization, forest normalization and batch adoption in the generation guide. Link the OpenCCO and svVascularize algorithmic references.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Regenerate measured vascular MPI figures and per-seed summaries from independently rechecked native experiments. Show observed variability in timing, memory, communication, physiological updates, startup ownership and tree quality. Distinguish rank-local sampling, proposal and insertion work from physical ownership, scale fractional quantities to their measured values, and reject incomplete inputs or stale published assets.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Complete the application reference with both lifted applications and document uniform simplex selection, refinement limits, and optional coupling dependencies.
- [PR #291](https://github.com/luca-heltai/immersx/pull/291) Document simplex meshes in the elastodynamics tutorial and reading the solid mesh from a file in the MetricFlowX–elastodynamics tutorial.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Document Field-local IDA tolerances, startup ramps, progress statistics, and restart behavior.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Profile the one-vessel elastodynamics MMS, explain Jacobian materialization and nested solver costs, and compare iterative and serial UMFPACK timings, memory, and numerical results.
- [PR #273](https://github.com/luca-heltai/immersx/pull/273) Render the MetricFlowX coupling equations and elastodynamics state variables correctly, and check all generated pages for math rendering errors in CI.
- [PR #272](https://github.com/luca-heltai/immersx/pull/272) Add an English onboarding guide for elastostatic–MetricFlowX coupling, from independent applications and file exchange to a monolithic IDAAdapter solve.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Document lifted Poisson composition, modal coefficient ordering and measure, solver policy, adaptive output, Coral execution, and the tested scope of legacy parity.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Document composable static elasticity, its equations and extension points, generated application/Coral inputs, and unsupported legacy capabilities.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Document application-level `both` configuration, migration from the old refinement factor, and executable elasticity and reduced Poisson examples.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Explain the local co-refinement lifecycle and publish random-tree mesh comparisons generated by a GoogleTest, with commands to reproduce the figures.
- [PR #250](https://github.com/luca-heltai/immersx/pull/250) Explain support-refinement error control, ghosted frozen inputs, and geometry snapshots for independent supports.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Document cell, boundary, independent immersed, and lifted weak-term supports, and distinguish historical reports from new measurements.
- [PR #248](https://github.com/luca-heltai/immersx/pull/248) Document reverse-chronological changelog ordering and checks for missing entries from merged PRs.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Allow pull requests without a related issue to omit the closing reference.
- [PR #196](https://github.com/luca-heltai/immersx/pull/196) Refreshed the documentation for the current architecture.
- [PR #95](https://github.com/luca-heltai/immersx/pull/95) Updated the agent instructions and the contributing guide.
- [PR #82](https://github.com/luca-heltai/immersx/pull/82) Revised the core architecture documentation.
- [PR #77](https://github.com/luca-heltai/immersx/pull/77) Documented the ImmersX core architecture.

### Tutorial

- [PR #380](https://github.com/luca-heltai/immersx/pull/380) Distinguish historical liver setup diagnostics on a fixed network from separately grown CCO families whose terminal counts scale with each background.

- [PR #354](https://github.com/luca-heltai/immersx/pull/354) Document the two-territory IRCAD12 solver pilot, model assumptions and recorded checks, with reproducible scalar plots and snapshots, complete references, and verified ParaView entrypoints.

- [PR #375](https://github.com/luca-heltai/immersx/pull/375) Report accepted dense resistance-driven liver references, local perfusion plots and measured coupled-space sizes, with explicit mesh-length and transient-validation limits.

- [PR #375](https://github.com/luca-heltai/immersx/pull/375) Document optional geometric checks for three-network liver growth, measured rigid capacity and matched terminal radius/flow diagnostics, and dense inspection sets.

- [PR #366](https://github.com/luca-heltai/immersx/pull/366) Add the completed A4245 arterial inspection, its measured coarse-mesh coverage and terminal occupancy, and ParaView paths while retaining the historical A3073 diagnostics.

- [PR #364](https://github.com/luca-heltai/immersx/pull/364) Add anatomical liver CCO inspection figures and measured coverage, interval-length and pressure-head diagnostics, and distinguish the historical feeder-stub pilot from full-tree growth.

- [PR #363](https://github.com/luca-heltai/immersx/pull/363) Explain global liver terminal-count scaling, local mesh coverage distances, per-cell terminal counts and the separate numerical interval-length checks.

- [PR #345](https://github.com/luca-heltai/immersx/pull/345) Provide IRCAD12 liver parameter profiles, bounded MOX arterial input, a two-territory physical vascular archive with CCO bifurcations in all three networks, and reproducible mesh-generation and run commands for a tiny pilot and three problem sizes.

- [PR #314](https://github.com/luca-heltai/immersx/pull/314) Add optional portal-tree preparation and growth to the liver workflow, with a common-compartment diagram, independent checks and rendering for three networks. Preserve the published 1,000-pair arterial–venous reference.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Use project-relative examples in the liver tutorial without machine names or installation-specific paths.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Illustrate the liver tutorial with native before-and-after vascular trees and a tissue-volume section; provide a renderer and a manifest of the checked source data.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Request 1,000 tissue territories with paired arterial and venous terminals in the liver tutorial, with 16 candidate vessels per insertion and a verified 32-rank run; keep a 32-pair option for short trials.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Prepare an IRCAD liver surface and rooted vessel meshes, generate the tissue volume with Gmsh, and grow paired arterial and venous extensions through a runnable tutorial with physical-data checks and ParaView output.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Publish three-seed MPI generation benchmarks at 1,000 and 10,000 terminals, with measured runtime, memory, communication, producer work and geometric quality.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Run native vascular cube and sphere examples with local refinement, common-territory arterial and venous networks, and a fixed-graph rediscretization control. Publish reproducible SVG views and measured density, length, volume and coverage plots from owned C++ output.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Add a frequency-sweep tutorial for a 1D fiber in a 3D matrix, with a 2D section lift and runnable independent and shared displacement inputs.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Explain independent and shared harmonic displacements, normal volume lifts, material units, and full-dimensional reinforcement, with runnable examples of both formulations.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Document harmonic fiber reinforcement, its phase convention and extension API, and add a runnable frequency-sweep tutorial checked in serial and MPI by CTest.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Use a single coarse vessel VTK for the cube MMS cases, with parameter-controlled refinement and reference area retained by Domain during flow-mesh refinement.
- [PR #275](https://github.com/luca-heltai/immersx/pull/275) Add a one-vessel manufactured-solution tutorial with piecewise displacement, nonzero pressure jump, native pressure feedback, and volume supplied by the inlet. Keep centerline cells within a factor of two of the solid axial cell lengths and integrate the lift with four points per solid axial cell.
- [PR #234](https://github.com/luca-heltai/immersx/pull/234) Run the tutorial platform through noVNC with remote Slurm execution and selectable Debug or Release packages.
- [PR #233](https://github.com/luca-heltai/immersx/pull/233) Build multi-architecture Coral, visualizer, and platform images for the dealiiX tutorial stack.
- [PR #127](https://github.com/luca-heltai/immersx/pull/127) Updated the tutorial documentation.
- [PR #84](https://github.com/luca-heltai/immersx/pull/84) Added the standalone first-order elastodynamics solver and its tutorials.
- [PR #80](https://github.com/luca-heltai/immersx/pull/80) Added the random-particle ReducedPoisson tutorials.

### Build and packaging

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Preserve the external 1D solver's global class symbols and expose its MPI header ABI through `IMMERSX_FVM_USE_MPI`, applying the same setting to library consumers and the cached link check.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Avoid implicit floating-point-to-index conversions in vessel-profile interpolation and its regression oracle, preserving endpoint clamping and interpolation values in Debug and Release. Apply the existing SDK preprocessor-warning policy to optional 1D client compilation probes as well. Fix a null-pointer return and remove an unused counter in the legacy 1D clients for Clang compatibility. Add explicit scopes around conditional GoogleTest macros to avoid dangling-else warnings in vascular and MetricFlowX regressions.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Link coupled applications with dimension-typed deal.II reference cells.
- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Link shared libraries on macOS when the optional vascular MPI profiler is absent.
- [PR #312](https://github.com/luca-heltai/immersx/pull/312) Include the Boost segment equality definition explicitly so persistent 3D vascular indices build with deal.II 9.7 headers.
- [PR #311](https://github.com/luca-heltai/immersx/pull/311) Include Boost point and box equality for incremental tissue-territory indices when building with deal.II 9.7 headers.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Fix compilation of venous network generation with deal.II 9.7.1 and its Boost.Geometry headers.
- [PR #303](https://github.com/luca-heltai/immersx/pull/303) Distribute precompiled Release applications in AMD64 and ARM64 OCI images for Docker, Podman, and Apptainer, with default help, application selection, a standalone MPI launcher, and tested GHCR publication.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Generate valid C++ for the optional 1dsolver link probe, include deal.II dependencies, and report incompatible libraries without claiming that RTTI is missing. Compile both driver bodies during Linux OpenMP test builds even when the external archive cannot link.
- [PR #288](https://github.com/luca-heltai/immersx/pull/288) Use the unified solution-transfer header so legacy applications build with deal.II 9.9.
- [PR #269](https://github.com/luca-heltai/immersx/pull/269) Link each ImmersX variant to the matching MetricFlowX and deal.II libraries in DebugRelease builds, retain compatible legacy single-variant packages, and check generated linkage in CTest.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Include set serialization explicitly for distributed static-elasticity material IDs so the application builds with deal.II 9.7.1 and Boost on Linux.
- [PR #219](https://github.com/luca-heltai/immersx/pull/219) Added the Coral Debug and Release CI jobs.
- [PR #211](https://github.com/luca-heltai/immersx/pull/211) Configured the shared ccache and VS Code path mapping.
- [PR #201](https://github.com/luca-heltai/immersx/pull/201) Restored DebugRelease single-config builds.
- [PR #194](https://github.com/luca-heltai/immersx/pull/194) Cancelled superseded GitHub Actions runs.
- [PR #88](https://github.com/luca-heltai/immersx/pull/88) Prepared ImmersX for packaging.
- [PR #87](https://github.com/luca-heltai/immersx/pull/87) Centralized build-tree test inputs and added the contributing guide.

### Testing and CI

- [PR #381](https://github.com/luca-heltai/immersx/pull/381) Count only cycle and mode collections in lifted-elasticity application tests so optional particle output does not fail the record-count check.

- [PR #374](https://github.com/luca-heltai/immersx/pull/374) Check that a native IDA binding and `IDAParameters` on the same section hold the same value for every entry, including entries missing from the input.

- [PR #370](https://github.com/luca-heltai/immersx/pull/370) Check diagnostic VTK identifiers exactly when VTK readers return native unsigned arrays instead of the writer's array subclass.

- [PR #348](https://github.com/luca-heltai/immersx/pull/348) Run validation and application GoogleTests in ordered batches of at most 15 cases with a 115-second CTest timeout, preserving serial and MPI coverage and exclusive access to shared output files.

- [PR #344](https://github.com/luca-heltai/immersx/pull/344) Keep performance history across fast-forward merges with reordered parents, validate the current source before historical backfill, and retain historical failures without blocking a valid current revision.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Request the conservation accuracy checked by the vascular Robin regression explicitly and keep its assertions free of ambiguous-else compiler diagnostics.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Verify Robin pressure and flux laws through the liver application, and time-dependent solid spring energy with a native MetricFlowX RCR network, in Debug and Release with one and two MPI ranks. Bound each application run below two minutes.

- [PR #331](https://github.com/luca-heltai/immersx/pull/331) Exercise Robin boundary conditions with manufactured scalar, solid, flow and vascular solutions, including actual Coral graphs. Bound harmonic application tests to 115 seconds and put a watchdog inside every MPI rank.

- [PR #340](https://github.com/luca-heltai/immersx/pull/340) Send lifted-support benchmark assembly diagnostics to stderr so its JSONL output passes strict correctness validation.

- [PR #334](https://github.com/luca-heltai/immersx/pull/334) Treat changes to the performance viewer browser check as documentation inputs, avoiding numerical, Coral and benchmark jobs when no solver or runtime input changes. Keep mixed changes on the full PR path.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Keep lifted elasticity application checks within 115 seconds and reduce the three-dimensional multiplier fixture while preserving two global refinement cycles, three multiplier fields and solver tolerances.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Bound coupled transient application regressions to 115 seconds and use three-point cross-section quadrature while retaining their meshes, finite-element degrees, forcing and solver tolerances.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Exercise both coupled manufactured boundary conditions with dedicated application fixtures and a 115-second test budget, preserving the resolved tutorial inputs for accuracy studies.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Avoid GMRES breakdown in the static tensor-product coupling regression by using a bulk mesh that can represent its six multiplier constraints, and check both assembled block residuals.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Initialize the vascular crossing-work fixture widths directly, preserving its nonuniform cells while avoiding GCC 13 array-bounds diagnostics from chained vector assignment.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Retain the accepted growth snapshot while traversing its obstacle index in the vascular regression, avoiding GCC 13 lifetime diagnostics.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Read the retained finite-element space directly in the lifted elasticity modal-metric regression, avoiding a temporary Field and GCC 13 lifetime diagnostics while preserving its independent quadrature check.

- [PR #326](https://github.com/luca-heltai/immersx/pull/326) Allow four-rank MPI regressions to launch on hosted runners with two CPU slots, supporting Open MPI 4 and 5 within the Debug and Release CTest steps.

- [PR #325](https://github.com/luca-heltai/immersx/pull/325) Keep complete vascular figure campaigns as manual publication runs, use a two-terminal coupled fixture with free solid displacement DoFs, enforce a 115-second watchdog and vascular CLI limit, remove the Debug-only coupled timeout extension, and retain the independent full-forest oracle only in test support.

- [PR #323](https://github.com/luca-heltai/immersx/pull/323) Limit numerical CI builds to two compiler jobs. Allow 120 minutes for Debug and 90 for Release, and skip incompatible optional 1D solver downloads while retaining client compilation checks.

- [PR #323](https://github.com/luca-heltai/immersx/pull/323) Compile applications and tests without GCC 13 warnings.

- [PR #324](https://github.com/luca-heltai/immersx/pull/324) Limit numerical CI to two concurrent compiler jobs and reserve separate workflow deadlines for cold Debug and Release builds.

- [PR #324](https://github.com/luca-heltai/immersx/pull/324) Reduce harmonic, simplex, and lifted-coupling fixtures, consolidate repeated solver-policy checks, and give the stationary Navier–Stokes simplex smoke test an explicit initial timestep and output assertions.

- [PR #324](https://github.com/luca-heltai/immersx/pull/324) Replace oversized physics and application fixtures with small analytic and native-solver regressions, remove obsolete smoke and convergence studies, and retain nonzero loads, boundary traces, and published error bounds.

- [PR #324](https://github.com/luca-heltai/immersx/pull/324) Replace network and trace convergence demonstrations with small analytic regressions, consolidate duplicate core assertions, and verify all products of the modal basis.

- [PR #324](https://github.com/luca-heltai/immersx/pull/324) Enforce a 120-second wall-time budget on every GoogleTest rank, retain timeout and partial build diagnostics, record Ninja command costs, enlarge the Debug compiler cache, and avoid downloading legacy 1D archives that fail the CI link check while retaining client-interface compilation.

- [PR #322](https://github.com/luca-heltai/immersx/pull/322) Cover elastodynamics cache invalidation, time-dependent loads, affine boundary lifting, and Schur factorization in serial and MPI.
- [PR #321](https://github.com/luca-heltai/immersx/pull/321) Skip numerical and Coral builds for documentation-only pull requests. Record C++ component selection in observation mode, retain every quick test for other changes, and keep full coverage on master with an explicit manual override.
- [PR #321](https://github.com/luca-heltai/immersx/pull/321) Validate documentation previews independently of performance measurements, using two Sphinx workers by default. Keep benchmark correctness smoke tests on numerical pull requests and complete measurements on master, with shared compiler caching.
- [PR #321](https://github.com/luca-heltai/immersx/pull/321) Fix Coral compiler-cache restoration and reuse across checkouts. Run the complete MetricFlowX package configuration matrix once in a dedicated CI job.
- [PR #321](https://github.com/luca-heltai/immersx/pull/321) Split integration GoogleTests into suite groups while checking the complete serial/MPI inventory, retaining shared-output locks, and publishing individual case timings.
- [PR #321](https://github.com/luca-heltai/immersx/pull/321) Build and test container images on master and release tags, while keeping launcher and fingerprint checks on pull requests. Preserve CTest manifests, timings, and logs as CI artifacts.
- [PR #320](https://github.com/luca-heltai/immersx/pull/320) Check deferred vector-constraint and transient fiber assembly, with explicit matrix reuse in the distributed regression tests.
- [PR #318](https://github.com/luca-heltai/immersx/pull/318) Restore serial and MPI coverage for segment-clustered point and tensor-product elasticity coupling. Check the inclusion mass matrix and recovery of nonzero imposed Fourier data instead of printing debug output.
- [PR #314](https://github.com/luca-heltai/immersx/pull/314) Check exact liver vessel counts with the fixed-length portal and venous startup stems and continuations, preserving the independent pairing, perfusion and imported-geometry checks.
- [PR #314](https://github.com/luca-heltai/immersx/pull/314) Check three-network perfusion balance, archive completion and canonical portal-demand updates in serial and MPI, and exercise the liver CLI with two and three input trees.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Count the fixed-length venous startup stem and its continuation separately in the liver smoke checks, while preserving exact terminal counts and imported physical properties.
- [PR #313](https://github.com/luca-heltai/immersx/pull/313) Allow ninety minutes for the complete Debug and Release CI jobs, including cold-cache builds, all quick tests and cache upload. Keep the test selection and individual test timeouts unchanged.
- [PR #312](https://github.com/luca-heltai/immersx/pull/312) Compare sparse vascular terminal-demand updates, candidate costs, obstacle radii, journal rollback and atomic seed initialization with independent full-forest evaluation in serial and distributed tests. Check native ownership and physical aggregation after concurrent branch edits and terminal remapping. Check that a single-root update stays limited to changed dependencies as the number of real roots grows. Cover internal-branch radius constraints on locally refined meshes and physical power when its objective weight is zero. Check cached branch resistance extrema and minimum leaf demand after staged edits. Compare owning execution rejection, retry, native owner migration and SI ghost output with immutable full commits. Check candidate envelope queries against full-forest trial collisions after demand transfers between roots. Keep per-batch physical-bound checks limited to changed roots while retaining the flow minimum on untouched distant leaves. Compare retained venous nearest cursors with exhaustive searches through successive exclusions, concurrent claims and exact-distance ties.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Allow thirty minutes for Debug MetricFlowX coupled application tests, including manufactured solutions, while preserving their physical inputs, solver tolerances and output checks.
- [PR #310](https://github.com/luca-heltai/immersx/pull/310) Exercise native vascular transfer and nonzero coupled IDA evolution in Debug and Release, including two- and four-rank MPI runs. Keep independent geometry and output checks and allow the generated Debug application time for Jacobian materialization.
- [PR #309](https://github.com/luca-heltai/immersx/pull/309) Update simplex elastodynamics and fiber-reinforcement test inputs to the named-material schema so native, IDA, and application tests parse their material coefficients.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Run reproducible strong and weak vascular scaling experiments across seeds with frozen binaries, native per-rank profiles and independent physical-tree checks. Preserve exact parameters, completed measurements and failed-run diagnostics in the build tree.
- [PR #304](https://github.com/luca-heltai/immersx/pull/304) Measure vascular MPI calls, posted payloads and time with optional benchmark-only PMPI profiling. Check native payload accounting and in-place forwarding in Debug and Release.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Include the new explicit elasticity, Laplacian, and reduced Poisson correctness suites in CTest validation runs and keep compiled test-category coverage complete.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Check that standalone MetricFlowX evolves on one-dimensional simplex cells and run the application linked to the installed package with its nonzero inflow fixture.
- [PR #299](https://github.com/luca-heltai/immersx/pull/299) Check native elastodynamics with quadratic displacement elements, nonzero forcing, and time-dependent boundary data on tetrahedra and hexahedra in serial and MPI.
- [PR #291](https://github.com/luca-heltai/immersx/pull/291) Test ElastodynamicsSolver on a 48-tetrahedron Gmsh cube and a hexahedral cube with a nonzero linear-in-space, quadratic-in-time exact solution, in serial and on two MPI ranks.
- [PR #290](https://github.com/luca-heltai/immersx/pull/290) Run harmonic reinforcement sweeps through parameter-file dispatch for all six dimension pairs, both formulations, and serial/two-rank MPI execution, checking the expected displacement and multiplier datasets. Verify owned affine offsets against a separately reconstructed finite-element field.
- [PR #282](https://github.com/luca-heltai/immersx/pull/282) Reduce mesh sizes in the 2D cross-section checks and mixed-boundary time-integrator regression without changing analytic values or tolerances. Keep all eight time steps and run the faster integrator regression in the quick serial/MPI groups. Synchronize MPI application tests before checking files written by rank zero.
- [PR #281](https://github.com/luca-heltai/immersx/pull/281) Keep standalone header and typed category checks while removing unused links and demonstration builds. Skip runtime search paths in configure-only package fixtures and run independent CI tests in two slots with shared output protected.
- [PR #281](https://github.com/luca-heltai/immersx/pull/281) Cancel superseded PR documentation previews while preserving official performance measurements.
- [PR #280](https://github.com/luca-heltai/immersx/pull/280) Preserve compiler cache entries between CI runs by using the restored cache directory and refreshing saved caches.
- [PR #274](https://github.com/luca-heltai/immersx/pull/274) Avoid deal.II deprecation warnings when building the lifted Poisson parity tests.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Bound Coral CI builds to two compiler processes to reduce peak memory when both lifted applications are enabled.
- [PR #263](https://github.com/luca-heltai/immersx/pull/263) Compare lifted Poisson coupling entries, modal loads, and bulk/multiplier coefficients with legacy Schur and AL solves on cylinders, variable radii, and a network; independently check nonzero forcing and Dirichlet data with hanging nodes in serial and MPI. Use feasible primal/multiplier discretizations and check residuals in the prescribed-constraint roadmap examples.
- [PR #262](https://github.com/luca-heltai/immersx/pull/262) Compare full static elasticity fields, modal loads and operator actions with native oracles in serial and MPI, and check generated parameter examples, output fields, cycle counts, and mesh refinement through CTest. Keep the ordinary category-overlap fixture targeted when new lifted suites are added.
- [PR #259](https://github.com/luca-heltai/immersx/pull/259) Run co-refinement serial/MPI cases through CTest using environment-based filters, and check velocity boundary data collectively when a rank owns no prescribed boundary DoFs.
- [PR #250](https://github.com/luca-heltai/immersx/pull/250) Retain original review benchmark reports and their immutable measured source provenance without relabelling them as repaired-head measurements.
- [PR #250](https://github.com/luca-heltai/immersx/pull/250) Check boundary mass against MatrixCreator and piecewise FE trace convergence against a fitted one-dimensional mass matrix, while retaining distributed nonlinear and frozen-input regressions.
- [PR #249](https://github.com/luca-heltai/immersx/pull/249) Compare lifted source-to-bulk assembly with an independent physical-quadrature reference using owned writable vectors, and retain historical Debug/Release benchmark records without relabeling their source.
- [PR #248](https://github.com/luca-heltai/immersx/pull/248) Run pull-request tests for stacked branches targeting any branch.
- [PR #247](https://github.com/luca-heltai/immersx/pull/247) Test replicated and fully distributed embedded support routing, scalar/vector nonlinear actions, transposes, and finite differences across partitions.
- [PR #246](https://github.com/luca-heltai/immersx/pull/246) Classify the compiled GoogleTest registry, reject missing or overlapping categories, discover tests without MPI initialization, and include shared tests in MPI runs.
- [PR #244](https://github.com/luca-heltai/immersx/pull/244) Added a MeshWorker reference and a paired nonlinear residual/Jacobian benchmark for heterogeneous weak terms with frozen inputs.
- [PR #232](https://github.com/luca-heltai/immersx/pull/232) Build and test Coral plugins against the `dealiix-tutorial` Coral branch.
- [PR #221](https://github.com/luca-heltai/immersx/pull/221) Added the issue #209 elastodynamics equivalence regression.
- [PR #182](https://github.com/luca-heltai/immersx/pull/182) Migrated the coupled regressions to unified constraints.
- [PR #177](https://github.com/luca-heltai/immersx/pull/177) Reorganized the tests into labeled CTest suites.
- [PR #169](https://github.com/luca-heltai/immersx/pull/169) Added the one-vessel two-way MetricFlowX-Elastodynamics MMS verification.
- [PR #90](https://github.com/luca-heltai/immersx/pull/90) Reduced the Debug testsuite runtime.

## [0.3] - 2026-05-11

### Added

- [PR #61](https://github.com/luca-heltai/immersx/pull/61) Added per-boundary-id Dirichlet and Neumann data with modulated parsed functions.
- [PR #60](https://github.com/luca-heltai/immersx/pull/60) Added fully distributed triangulation support to the elasticity problem.
- [PR #51](https://github.com/luca-heltai/immersx/pull/51) Added per-material-id material properties.

### Changed

- [PR #55](https://github.com/luca-heltai/immersx/pull/55) Grouped inclusions that fall in the same cell.
- [PR #54](https://github.com/luca-heltai/immersx/pull/54) Moved vessel-pressure computation and output into coupled elasticity.
- [PR #53](https://github.com/luca-heltai/immersx/pull/53) Cleaned up and refactored coupled elasticity.
- [PR #50](https://github.com/luca-heltai/immersx/pull/50) Incorporated the brain-dynamics changes contributed by Yashu.

### Fixed

- [PR #58](https://github.com/luca-heltai/immersx/pull/58) Fixed the reduced Poisson problem.

### Documentation

- [PR #64](https://github.com/luca-heltai/immersx/pull/64) Rebranded the project with a new logo and README.
- [PR #56](https://github.com/luca-heltai/immersx/pull/56) Full documentation pass over the missing classes.

### Build and packaging

- [PR #57](https://github.com/luca-heltai/immersx/pull/57) Updated the GitHub Pages workflow.

### Testing and CI

- [PR #59](https://github.com/luca-heltai/immersx/pull/59) Added the liver-benchmark mesh preparation files.

## [0.2] - 2026-01-29

### Added

- [PR #44](https://github.com/luca-heltai/immersx/pull/44) Added a `properties` member and a vertex- and cell-agnostic field transfer.
- [PR #42](https://github.com/luca-heltai/immersx/pull/42) Added rigid-body mode extraction for AMG preconditioning.
- [PR #40](https://github.com/luca-heltai/immersx/pull/40) Added VTK utilities for reading cell and point data.
- [PR #36](https://github.com/luca-heltai/immersx/pull/36) Added local refinement support for deal.II inclusions.
- [PR #32](https://github.com/luca-heltai/immersx/pull/32) Added initial deal.II-based inclusion support.
- [PR #31](https://github.com/luca-heltai/immersx/pull/31) Added parameter-file-driven inclusion data and corrected pressure and stress output.
- [PR #27](https://github.com/luca-heltai/immersx/pull/27) Added the initial coupled elasticity problem.

### Changed

- [PR #48](https://github.com/luca-heltai/immersx/pull/48) Updated the FVM headers.
- [PR #25](https://github.com/luca-heltai/immersx/pull/25) Updated the Docker environment and the 1D solver library download.

### Fixed

- [PR #49](https://github.com/luca-heltai/immersx/pull/49) Fixed compiler warnings.
- [PR #47](https://github.com/luca-heltai/immersx/pull/47) Removed the duplicated mass matrix from elasticity.
- [PR #46](https://github.com/luca-heltai/immersx/pull/46) Added missing VTK compile flags.
- [PR #43](https://github.com/luca-heltai/immersx/pull/43) Fixed the scaling of the Reduced Poisson problem.
- [PR #30](https://github.com/luca-heltai/immersx/pull/30) Addressed failing tests.

### Testing and CI

- [PR #33](https://github.com/luca-heltai/immersx/pull/33) Added an inclusion-orientation regression test for the solution and multiplier norms.
- [PR #28](https://github.com/luca-heltai/immersx/pull/28) Added the Lorthois datasets.

## [0.1] - 2024-11-07

### Added

- [PR #22](https://github.com/luca-heltai/immersx/pull/22) Added a geometric multigrid matrix-free preconditioner for the elasticity block.
- [PR #20](https://github.com/luca-heltai/immersx/pull/20) Added a matrix-free `CouplingOperator` for the action and transpose of the restriction operator.
- [PR #19](https://github.com/luca-heltai/immersx/pull/19) Added a matrix-free bulk evaluator.
- [PR #18](https://github.com/luca-heltai/immersx/pull/18) Enabled distributed parallel runs.
- [PR #13](https://github.com/luca-heltai/immersx/pull/13) Added local elasticity assembly.
- [PR #11](https://github.com/luca-heltai/immersx/pull/11) Added Dirichlet-only elasticity boundary conditions.
- [PR #5](https://github.com/luca-heltai/immersx/pull/5) Added Neumann and Robin boundary conditions.

### Changed

- [PR #23](https://github.com/luca-heltai/immersx/pull/23) Updated the license and repository metadata.
- [PR #8](https://github.com/luca-heltai/immersx/pull/8) Split declarations and definitions into separate files.
- [PR #6](https://github.com/luca-heltai/immersx/pull/6) Moved the `ReferenceInclusion` struct into its own header.

### Fixed

- [PR #17](https://github.com/luca-heltai/immersx/pull/17) Provided the locally relevant row index set when building distributed sparsity patterns.
- [PR #12](https://github.com/luca-heltai/immersx/pull/12) Fixed rotation handling.

### Testing and CI

- [PR #14](https://github.com/luca-heltai/immersx/pull/14) Added rotation tests.
