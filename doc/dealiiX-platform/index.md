# dealiiX-platform integration

[dealiiX-platform](https://github.com/2listic/dealiiX-platform) is the graph
editor and application interface for Coral workflows. ImmersX supplies Coral
plugins; it does not include the Coral runtime or the platform itself.

The three projects have separate roles:

- [Coral](https://github.com/2listic/coral) provides the graph runtime, command
  line executable, plugin ABI, and CMake package.
- [ImmersX](https://github.com/luca-heltai/immersx) provides the numerical
  problems and the `coral_backend_immersx_*` plugins.
- [Coral Visualizer](https://github.com/2listic/coral-visualizer) displays the
  output files produced by a graph. It is not needed to build or run a plugin.

## Build and install Coral

Coral must be installed before ImmersX is configured. The checkout must include
Coral's `entt` submodule:

```bash
git clone --recurse-submodules https://github.com/2listic/coral.git coral
cd coral
git checkout installable-plugin-sdk
cd /path/to/immersx
base64 -d .github/coral/coral-api.patch.gz.b64 | gzip -d | \
  git -C /path/to/coral apply
```

The remote `installable-plugin-sdk` branch currently predates the registry API
used by ImmersX. Omit the patch step after Coral publishes that API on the
branch. Use a shared Coral core. The plugin and the Coral executable must use
that same core library because the core owns Coral's runtime type registry. The following
build installs both configurations side by side:

```bash
cmake -S . -B build -G "Ninja Multi-Config" \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/coral" \
  -DCORAL_BUILD_BACKEND_DEALII=OFF \
  -DCORAL_BUILD_SHARED_CORE=ON \
  -DCORAL_BUILD_TESTS=OFF \
  -DCORAL_INSTALL=ON
cmake --build build --config Debug
cmake --build build --config Release
cmake --install build --config Debug
cmake --install build --config Release
```

This installs the Coral CMake package at:

```text
$HOME/.local/coral/lib/cmake/coral
```

If the installable package changes have been merged into the Coral branch you
are using, the explicit `git checkout installable-plugin-sdk` step is not
needed. Keep the recursive submodule checkout and the shared-core options.

## Build an ImmersX Coral plugin

Configure ImmersX with the installed Coral package. A `DebugRelease` build is
useful when both Coral configurations are installed:

```bash
cd /path/to/immersx
cmake -S . -B build-coral -G Ninja \
  -DCMAKE_BUILD_TYPE=DebugRelease \
  -Dcoral_DIR="$HOME/.local/coral/lib/cmake/coral" \
  -DIMMERSX_BUILD_CORAL_PLUGINS=ON \
  -DIMMERSX_ENABLE_CORAL_GRAPH_TESTS=ON \
  -DENABLE_GOOGLE_TESTING=OFF \
  -DENABLE_DEAL_II_APP_TESTING=OFF
cmake --build build-coral -j
```

CMake must report:

```text
ImmersX Coral plugins enabled
```

The build creates one plugin for each ambient dimension. In a `DebugRelease`
build, the normal names are Release and the `_debug` names are Debug:

```text
build-coral/coral/libcoral_backend_immersx_1.so
build-coral/coral/libcoral_backend_immersx_2.so
build-coral/coral/libcoral_backend_immersx_3.so
build-coral/coral/libcoral_backend_immersx_1_debug.so
build-coral/coral/libcoral_backend_immersx_2_debug.so
build-coral/coral/libcoral_backend_immersx_3_debug.so
```

The extension is `.dylib` on macOS. For a single-configuration Debug or
Release build, only the matching set is created. The plugin targets link to
both the ImmersX library and `coral::core`; do not link a plugin to a different
Coral build.

To install the plugin and its graph examples:

```bash
cmake --install build-coral --prefix "$HOME/.local/immersx-coral"
```

The installed plugin directory is:

```text
$HOME/.local/immersx-coral/lib/immersx/coral
```

The installed graph and parameter files are under:

```text
$HOME/.local/immersx-coral/share/immersx/coral/examples
```

## Run a Coral network

The checked-in examples exercise complete ImmersX workflows. They include
Poisson, weak-term Poisson, static elasticity, fixed-step and IDA
elastodynamics, coupled Poisson, coupled Poisson/elasticity, fiber-reinforced
elastodynamics, and ReducedPoisson.

Run the registry check first:

```bash
ctest --test-dir build-coral -R '^ImmersX\.Coral\.Registry' \
  --output-on-failure
```

Run the complete graph collection with:

```bash
ctest --test-dir build-coral -R '^ImmersX\.Coral\.' \
  --output-on-failure
```

To run a graph directly, use a working directory containing both the JSON graph
and its parameter file. For example, after installing ImmersX:

```bash
cd "$HOME/.local/immersx-coral/share/immersx/coral/examples"
export CORAL_LIBRARY_DIR="$HOME/.local/coral/lib/Debug"
export PLUGIN="$HOME/.local/immersx-coral/lib/immersx/coral/libcoral_backend_immersx_2_debug.so"

LD_LIBRARY_PATH="$CORAL_LIBRARY_DIR:$HOME/.local/immersx-coral/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$HOME/.local/coral/bin/Debug/coral" \
  -p "$PLUGIN" run poisson_2d.json --graph poisson_2d.dot
```

On macOS, use `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`. The graph
parameter file is resolved relative to the working directory, and its output
is written below the configured output directory. The graph's MPI block should
be run with the MPI implementation used by deal.II and Coral.

## Use the plugin in dealiiX-platform

The platform needs three paths for a local Coral workflow:

1. the Coral executable, such as `$HOME/.local/coral/bin/Debug/coral`;
2. the ImmersX plugin, such as
   `$HOME/.local/immersx-coral/lib/immersx/coral/libcoral_backend_immersx_2_debug.so`;
3. the working directory containing the graph and parameter files.

Use the platform's local Coral execution mode with those paths. The platform
loads the plugin to obtain its registry, then sends the selected graph to the
Coral executable. The same plugin path can be inspected independently with the
`coral register` command shown above.

The platform repository documents its local and container workflows here:

- [Run Coral locally](https://github.com/2listic/dealiiX-platform/blob/main/docs/run-coral-local.md)
- [Run Coral in Docker](https://github.com/2listic/dealiiX-platform/blob/main/docs/run-coral-docker.md)

After a graph writes VTK output, open the result with
[Coral Visualizer](https://github.com/2listic/coral-visualizer). Coral
Visualizer consumes graph output; it does not replace the Coral executable or
the ImmersX plugin.

## Pull the tutorial images

For the fastest browser-based setup, use the published stack from the
`dealiix-tutorial` branch:

```bash
git clone --branch dealiix-tutorial \
  https://github.com/luca-heltai/immersx.git
cd immersx
./scripts/tutorial-init.sh
docker compose pull
docker compose up -d
```

Open <http://localhost:6080/vnc.html?autoconnect=1&resize=scale> for the
dealiiX-platform interface. The stack
pulls the latest multi-architecture images for Coral plus ImmersX and
MetricFlowX, Coral Visualizer, and dealiiX-platform. Set
`DEALIXX_BUILD_TYPE=Debug` before `docker compose up -d --force-recreate
dealiix-platform` to use the Debug Coral and ImmersX variants; Release is the
default. To refresh an existing
checkout, use `git pull && docker compose pull && docker compose up -d`.

For the ImmersX-side CMake and registry details, see the
[Coral backend how-to](../how-to/coral-backend).
