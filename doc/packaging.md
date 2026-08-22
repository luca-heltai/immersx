# Packaging and installation

ImmersX installs its public API below `include/immersx`, the library and
executables in the locations selected by CMake's `GNUInstallDirs`, and a
package configuration that can be consumed with:

```cmake
find_package(ImmersX CONFIG REQUIRED)
target_link_libraries(my_solver PRIVATE ImmersX::immersx)
```

The package configuration still requires the same deal.II installation used
to build ImmersX. It also discovers OpenMP when the build used it. Coupled
3D/1D applications additionally require the `lib1dsolver` library;
that optional library is intentionally not part of the exported core target.
The headers under `include/immersx/fvm` mirror the `1dsolver` interface and
retain the dependency's global class names; the ImmersX
coupling wrapper itself is `ImmersX::CoupledModel1d`. The core ImmersX API is
fully namespaced.

## Shared and static builds

The default build creates a shared ImmersX library:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/deal.II
cmake --build build-release -j
cmake --install build-release --prefix /path/to/stage
```

To create a static ImmersX archive, use `-DIMMERSX_BUILD_STATIC=ON`. This
changes ImmersX's library type, but does not magically make deal.II, Trilinos,
PETSc, MPI, VTK, or the system runtime static. A downstream application may
therefore still need those shared libraries.

## Portable Linux executables

`-DIMMERSX_PORTABLE=ON` selects a static ImmersX archive and adds `-static` to
the application link step. This is intended for a Linux toolchain where
  deal.II and every required dependency are themselves available as matching
static libraries:

```bash
cmake -S . -B build-portable \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/static/deal.II \
  -DIMMERSX_PORTABLE=ON \
  -DENABLE_COUPLED_PROBLEMS=OFF
cmake --build build-portable -j
```

This mode is deliberately Linux-only for now. Static MPI, graphics, VTK, and
system-library support is platform/toolchain dependent, and glibc-linked
executables may still have a minimum host-kernel or libc requirement. Verify a
portable artifact in a clean machine/container; the option is not a promise
that arbitrary deal.II/Trilinos installations can be embedded automatically.

## Installed tests and data

`IMMERSX_INSTALL_TESTS` is enabled by default. GoogleTest binaries are placed
under `${CMAKE_INSTALL_LIBEXECDIR}/immersx/tests`; test fixtures and source
templates are installed below `${CMAKE_INSTALL_DATADIR}/immersx/tests`, while
runtime data is below `${CMAKE_INSTALL_DATADIR}/immersx/data`. Disable test
installation for a runtime-only package with
`-DIMMERSX_INSTALL_TESTS=OFF`.
