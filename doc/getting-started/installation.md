# Installation and build

ImmersX requires CMake 3.18 or newer, a C++ compiler supported by deal.II, and
deal.II 9.7.1 or newer. MPI is required by the distributed applications.
Optional deal.II components enable features such as VTK output, SUNDIALS time
integration, SymEngine expressions, and PETSc or Trilinos backends.

Configure and build out of source:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DDEAL_II_DIR=/path/to/deal.II \
  -DENABLE_GOOGLE_TESTING=ON \
  -DENABLE_DEAL_II_APP_TESTING=ON
cmake --build build -j
```

ImmersX supports `Debug`, `Release`, and `DebugRelease`. `DebugRelease` is a
single-config build-tree mode that builds both variants at once: Release
targets keep their normal names, while Debug targets receive the `_debug`
suffix. It does not select a CMake multi-config generator.

```bash
cmake -S . -B build-debugrelease \
  -DCMAKE_BUILD_TYPE=DebugRelease \
  -DDEAL_II_DIR=/path/to/deal.II
cmake --build build-debugrelease -j
```

The optional `lib1dsolver` dependency enables the 3D/1D `coupled_elasticity`
and `pseudocoupling1D` executables when it is available. The [application reference](../reference/applications)
lists dependencies and supported dimensions for every executable.

Point `FVM_LIB` at a complete library matching the vendored FVM headers, built
with RTTI and with a compatible compiler, C++ standard-library ABI, MPI implementation, and system libraries.
The backend's `USE_MPI` setting changes the layout of its public classes. Match
it explicitly with `IMMERSX_FVM_USE_MPI`: leave the option `OFF` (the default)
for a serial backend, or select `ON` for a backend built with `USE_MPI`:

```bash
cmake -S . -B build-debugrelease \
  -DCMAKE_BUILD_TYPE=DebugRelease \
  -DENABLE_COUPLED_PROBLEMS=ON \
  -DFVM_LIB=/path/to/lib1dsolver.a \
  -DIMMERSX_FVM_USE_MPI=ON
```

The MPI setting requires deal.II with MPI. ImmersX applies the same definition
to its link check and exports it to application and library consumers. The
library filename cannot identify this ABI setting; a successful link alone
does not prove that the backend and header layouts agree. The backend is an
external dependency; ImmersX does not download or rebuild its sources.

To build the published documentation, install the Python requirements and run:

```bash
python3 -m pip install -r doc/requirements.txt
./scripts/build_doc.sh
```

The site is generated in `build/docs/site/`. A configured CMake build also
provides `cmake --build build --target doc`.
