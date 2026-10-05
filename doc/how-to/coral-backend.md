# Coral backend

ImmersX can build optional plugins for Coral, the graph runtime used by
dealiiX-platform. Coral is not required by the ordinary ImmersX library or by
its applications.

## Configure and build

Point CMake at the installed Coral package and leave the optional backend
enabled:

```bash
cmake -S . -B build-coral \
  -DCMAKE_BUILD_TYPE=Debug \
  -Dcoral_DIR=/Users/heltai/dualistic/coral/inst/lib/cmake/coral \
  -DIMMERSX_BUILD_CORAL_PLUGINS=ON
cmake --build build-coral -j
```

The option defaults to `ON`, but plugin targets are created only when CMake
finds a compatible shared `coral::core` target. To build the ordinary ImmersX
library without Coral, use:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DIMMERSX_BUILD_CORAL_PLUGINS=OFF
```

`DebugRelease` requires both Debug and Release configurations from Coral. If
the package exposes only one configuration, CMake disables the Coral plugins
and prints a warning instead of mixing ABI variants.

## Plugins and dimensions

The build produces one plugin for each ambient dimension:

```text
build-coral/coral/libcoral_backend_immersx_1_debug.so
build-coral/coral/libcoral_backend_immersx_2_debug.so
build-coral/coral/libcoral_backend_immersx_3_debug.so
```

Each plugin registers only intrinsic dimensions valid for its ambient space:

| plugin | registered `(dim, spacedim)` pairs |
|---|---|
| 1D | `(1,1)` |
| 2D | `(1,2)`, `(2,2)` |
| 3D | `(1,3)`, `(2,3)`, `(3,3)` |

The current stable problem names are `ImmersX::Poisson<dim,spacedim>`,
`ImmersX::ElasticStatic<dim,spacedim>`, and
`ImmersX::Elastodynamics<dim,spacedim>`. Their parameter objects and lifecycle
operations are registered under the same dimension-qualified naming scheme.

To inspect a registry:

```bash
DYLD_LIBRARY_PATH=/Users/heltai/dualistic/coral/inst/lib \
  /Users/heltai/dualistic/coral/inst/bin/coral \
  -p build-coral/coral/libcoral_backend_immersx_2_debug.so \
  register --registry-path /tmp/immersx-coral-2d-registry.json
```

On Linux, use `LD_LIBRARY_PATH` in place of `DYLD_LIBRARY_PATH`.

## Poisson graph

The checked-in graph and parameter file are installed below
`share/immersx/coral/examples`:

```text
poisson_2d.json
poisson_2d.prm
```

The graph makes the mutable lifecycle explicit:

```text
parameters -> initialize -> Poisson -> make_grid -> setup_fe
                                      -> setup_system -> assemble -> solve -> output
```

The edges between mutating operations are required. Coral may execute
independent nodes concurrently, so a graph must pass the object through each
mutation when the order matters.

The plugin initialization block enables MPI. Run the graph with the same MPI
launcher and runtime library used by the deal.II installation, for example:

```bash
cd /path/to/installed/examples
mpirun -np 1 env \
  DYLD_LIBRARY_PATH=/Users/heltai/dualistic/coral/inst/lib:/path/to/immersx/lib \
  THREADS=1 \
  /Users/heltai/dualistic/coral/inst/bin/coral \
  -p /path/to/immersx/lib/immersx/coral/libcoral_backend_immersx_2_debug.so \
  run poisson_2d.json --graph poisson_2d.dot
```

The parameter file path is resolved by Coral from the process working
directory. Therefore the example command is run from the directory containing
the installed graph and parameter file. Output remains controlled by the
parameter file.

If an `Initialize parameters` node refers to a parameter file that does not
exist, the plugin creates its parent directories, declares all connected
`ParameterAcceptor` objects, and writes a documented `ParameterHandler` file
using `DefaultStyle`. Rank zero performs the write and the other ranks wait
before parsing it. The graph then parses the generated file and continues in
the same execution with the default values. An existing file is still parsed
normally, so invalid or unreadable files remain errors. A subsequent
dealiiX-platform run can detect the persistent file and stage it in its run
directory.

The same pass-through lifecycle is used by the checked-in static-elasticity
and standalone-elastodynamics graphs:

```text
elastic_static_2d.json       elastic_static_2d.prm
elastodynamics_2d.json       elastodynamics_2d.prm
```

They use the 2D plugin and expose the parameter bundle, problem construction,
and ordered mutating operations as separate nodes.

When deal.II has SUNDIALS support, the 2D plugin also provides an IDA-backed
execution seam:

```text
ida_elastodynamics_2d.json       ida_elastodynamics_2d.prm
```

`ImmersX::IDAElastodynamics<2,2>` keeps IDA state initialization, accepted
state handoff, and output callbacks inside the binding. The graph exposes only
`run`, finiteness, and current-time nodes.

The 2D and 3D fiber examples are also installed. Their graph exposes the
composition explicitly:

```text
FiberReinforcedElastodynamicsGraph
        |
        +--> matrix Problem ------+
        |                         |
        +--> embedded fiber ------+--> velocity-continuity Interaction
        |                                               |
        +--> Fiber execution adapter -------------------+
                                                        |
                                                       IDA
```

The graph-facing `FiberMatrixProblem<dim>`, `FiberEmbeddedProblem<dim>`, and
`FiberVelocityContinuity<dim>` nodes are small Coral façades over the existing
ImmersX Problems and `make_continuity_constraint()` helper. The execution
adapter owns the IDA state and accepted-state output; when SUNDIALS is not
available, it uses the existing fixed-step fallback. The driver remains the
native owner of the physical assembly and writes matrix, fiber, and multiplier
outputs under the configured output directory.

The 2D plugin also exposes the bulk/embedded Poisson workflow through generic
Problem, finite-element-space, Constraint, and LinearAdapter nodes. Its example
is:

```text
coupled_poisson_2d.json
coupled_poisson_2d.prm
```

The graph constructs two existing `PoissonSolver` Problems, creates the
multiplier finite-element space on the embedded mesh, adds both Problems and a
continuity Constraint to one `LinearAdapter`, transfers the solved fields back
to the Problems, writes bulk, embedded, and multiplier output, and checks the
composed residual. The old standalone `ImmersX::CoupledPoisson<2>` façade is
not part of the 2D registry.

## Primitive scalar solve

The 2D plugin also exposes the lower-level generic path. The checked-in example
`poisson_primitives_2d.json` creates an owning `Domain`, creates an owning
`FiniteElementSpace` directly from that domain, selects boundary conditions for
a field, registers the constrained field as an algebraic execution field, and adds
the weak term

```cpp
weak_term(gradient(u), gradient(test(u)))
```

to one `LinearAdapter`. The same path is available directly in C++:

```cpp
Domain<2> domain(domain_parameters);
domain.make_grid();
FiniteElementSpace<2> space(domain.triangulation(), space_parameters);
BoundaryConditions<2> boundary_conditions(space);
dealii::Functions::ConstantFunction<2> boundary_value(1.0);
boundary_conditions.add_dirichlet(0, boundary_value);
boundary_conditions.update(0.);

const auto unregistered = space.view().field("u");
const auto registration =
  adapter.add(algebraic_field(unregistered), "solution");
const auto u = registration.fields();
adapter.add(weak_term(gradient(u), gradient(test(u))), "laplace");

auto state = adapter.make_state();
adapter.solve(state);
```

The Coral expression binding follows the same compositional grammar as the
C++ API:

```text
Field -> Test -> FE operation
Field -> FE operation
FE expression + test-side FE expression -> Weak term
```

For the Poisson primitive graph, the variational path is therefore:

```text
Field -> Gradient
Field -> Test -> Gradient
Gradient + Test Gradient -> Weak term
```

`Test`, `Value`, `Gradient`, `Divergence`, `Symmetric gradient`, and `Curl`
are operation families registered for the supported scalar/vector overloads.
The node signatures remain concrete C++ value types, while their metadata
keeps the graph vocabulary at the operation-family level. `Weak term` consumes
the outputs of those expression nodes directly; it does not encode a
Poisson- or Laplace-specific shortcut. The current dealiiX-platform UI may
still display concrete overloads separately until it groups nodes using the
shared `operation` metadata.

Boundary conditions follow the same explicit data-flow style:

```text
Field -> Boundary conditions -> Dirichlet boundary condition
Constant function + boundary id + Boundary conditions
    -> Dirichlet boundary condition
Boundary conditions + Field -> Apply boundary conditions -> Field
```

`Boundary conditions` creates the persistent condition set associated with a
field. `Constant function` is a reusable boundary-function node, and
`Dirichlet boundary condition` returns an updated condition set. `Apply boundary
conditions` evaluates that set and passes the constrained field to subsequent
execution nodes. Boundary-function nodes are owned by the condition set, so the
graph does not depend on the lifetime of a temporary C++ function object. The
specialized `Set constant Dirichlet boundary condition` node has been removed;
there is no compatibility alias for it.

The three concrete function sources converge on the same handle:

```cpp
using FunctionHandle =
  std::shared_ptr<const dealii::Function<spacedim>>;
```

Coral transports that handle in `ImmersX::BoundaryFunction<spacedim>`, whose
`value` member has type `FunctionHandle`. In addition to `Constant function`,
the `Parsed function` operation has two variants:

- `Expression` accepts exactly `const std::string &expression` and constructs
  a scalar `dealii::FunctionParser<spacedim>` with
  `default_variable_names() + ",t"`. It supplies the constants string
  `E=<value>,PI=<value>` using `dealii::numbers::E`, `dealii::numbers::PI`, and
  `std::numeric_limits<double>::max_digits10` precision.
- `Parameterized` uses the existing
  `ImmersX::ModulatedParsedFunction<spacedim>`. The concrete object is a
  `ParameterAcceptor` derived type, so it can be connected to the generic
  `Initialize parameters` operation before it is converted to the common
  function handle. Its parameters are `Function constants`, `Function
  expression`, `Variable names`, `Modulation frequency`, and `Phase shift`.

The same handle type can therefore be consumed by boundary conditions today
and by `KnownTerm`/RHS graph operations when those operations are exposed in
Coral.

The boundary helper rebuilds hanging-node and Dirichlet constraints on the
owning space. For an inhomogeneous boundary, the linear weak-term assembly
includes the corresponding affine contribution in the residual. Output uses
the generic `OutputHandler` and the semantic field name; the example writes
`output/solution.pvd`.

The 3D plugin provides the corresponding Poisson-to-elasticity example:

```text
coupled_poisson_elasticity_3d.json
coupled_poisson_elasticity_3d.prm
```

`ImmersX::CoupledPoissonElasticity<3>` uses the existing `value`, scaled
`TensorProductLift`, `normal`, `weak_term`, and `LinearAdapter` components.
Its graph exposes the coupled residual, pressure-scale, and traction-balance
diagnostics after the native Poisson and elasticity outputs are written.

The 3D plugin also provides a point-backed ReducedPoisson example:

```text
reduced_poisson_3d.json       reduced_poisson_3d.prm
```

`ImmersX::ReducedPoissonWorkflow<3>` delegates to the existing
`ReducedPoisson<3,3,0,3>` implementation. The binding supplies one
representative point at `(0.5, 0.5, 0.5)`, so the installed graph does not
depend on a source-tree VTK point-cloud file. It exposes the reduced DoF count,
coupling-matrix norm, solution norms, and a finiteness check after `run`.

## Tests

When the Coral executable target is available, CTest registers one registry
test for each plugin. These tests verify the dimension policy, stable problem
names, and common elementary types:

```bash
ctest --test-dir build-coral -R 'ImmersX.Coral.Registry' \
  --output-on-failure
```

The numerical graph tests are opt-in because they need a working MPI runtime
in addition to the Coral package:

```bash
cmake -S . -B build-coral \
  -DCMAKE_BUILD_TYPE=Debug \
  -Dcoral_DIR=/Users/heltai/dualistic/coral/inst/lib/cmake/coral \
  -DIMMERSX_ENABLE_CORAL_GRAPH_TESTS=ON
ctest --test-dir build-coral -R 'ImmersX.Coral.Graph' \
  --output-on-failure
```

The tests check the generated graphviz files and the configured native output
for Poisson, static elasticity, elastodynamics, both coupled workflows, fiber
reinforcement, and ReducedPoisson.

The registry tests do not replace numerical application tests. Graph execution
also requires a working MPI runtime because the current distributed ImmersX
problems use `MPI_COMM_WORLD` during construction.
