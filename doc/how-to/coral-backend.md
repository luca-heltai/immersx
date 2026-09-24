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
parameters -> load -> Poisson -> make_grid -> setup_fe
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

The same pass-through lifecycle is used by the checked-in static-elasticity
and standalone-elastodynamics graphs:

```text
elastic_static_2d.json       elastic_static_2d.prm
elastodynamics_2d.json       elastodynamics_2d.prm
```

They use the 2D plugin and expose the parameter bundle, problem construction,
and ordered mutating operations as separate nodes.

The 2D plugin also exposes the existing bulk/embedded Poisson workflow as the
graph-facing `ImmersX::CoupledPoisson<2>` façade. Its example is:

```text
coupled_poisson_2d.json
coupled_poisson_2d.prm
```

The façade owns the two existing `PoissonSolver` problems, the multiplier
finite-element space, and the `LinearAdapter`. The graph therefore controls
the workflow lifecycle and can query `ImmersX::CoupledPoisson<2>::residual_norm`
after `run`. It is a first vertical coupling path; the lower-level Problems
and Interaction objects are still assembled by the existing ImmersX code.

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
for Poisson, static elasticity, elastodynamics, or coupled Poisson.

The registry tests do not replace numerical application tests. Graph execution
also requires a working MPI runtime because the current distributed ImmersX
problems use `MPI_COMM_WORLD` during construction.
