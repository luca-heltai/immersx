# Build and Run

## Requirements

Required:

- `CMake >= 3.10`
- a C++ compiler supported by your `deal.II` build
- `deal.II` compatible with the APIs used by this project

Optional:

- Trilinos or PETSc as enabled in `deal.II`
- OpenMP
- GoogleTest
- OpenCASCADE and HDF5 for selected features
- `lib1dsolver` for coupled 3D/1D executables

## Build

```bash
mkdir -p build
cd build
cmake -DDEAL_II_DIR=/path/to/deal.II ..
cmake --build . -j
```

Executables are generated from files in `apps/app_*.cc`, for example `elasticity`, `laplacian`, `reduced_poisson`, and `coupled_elasticity`.

## Run

Typical single-run pattern:

```bash
./build/elasticity path/to/input.prm
```

or

```bash
./build/laplacian path/to/input.prm
```

For coupled workflows, inspect the corresponding `apps/app_*.cc` entry point and the matching files under `prms/` or `benchmarks/`.

## Coupled 3D/1D Workflow

```{include} ../COUPLED_PROBLEM.md
:relative-docs: .
:start-after: This file collects the operational notes for the coupled elasticity workflow
```
