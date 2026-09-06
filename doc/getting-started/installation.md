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

The optional `lib1dsolver` dependency enables the 3D/1D `coupled_elasticity`
and `pseudocoupling1D` executables when it is available. The [application reference](../reference/applications)
lists dependencies and supported dimensions for every executable.

To build the published documentation, install the Python requirements and run:

```bash
python3 -m pip install -r doc/requirements.txt
./scripts/build_doc.sh
```

The site is generated in `build/docs/site/`. A configured CMake build also
provides `cmake --build build --target doc`.
