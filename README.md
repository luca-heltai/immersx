# Reduced Lagrange Multipliers

![GitHub CI](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/tests.yml/badge.svg)
![Documentation](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/doxygen.yml/badge.svg)
![Indent](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/indentation.yml/badge.svg)

This repository contains C++ implementations of reduced Lagrange multiplier methods for mixed-dimensional coupling problems, built on top of [deal.II](https://www.dealii.org).

Primary reference:

- Luca Heltai, Paolo Zunino, *Reduced Lagrange multiplier approach for non-matching coupling of mixed-dimensional domains*, Mathematical Models and Methods in Applied Sciences (2023).
- DOI: <https://dx.doi.org/10.1142/S0218202523500525>

## What is in this repository

Core code:

- `include/`: public headers for solvers, coupling operators, parameter classes, and utilities.
- `source/`: implementations for the core classes.
- `apps/`: executable entry points (`app_elasticity.cc`, `app_laplacian.cc`, `app_reduced_poisson.cc`, `app_coupled_elasticity.cc`, `app_pseudocoupling1D.cc`).

Testing:

- `tests/`: deal.II-style regression tests.
- `gtests/`: GoogleTest-based unit/integration tests.

Documentation and scripts:

- `doc/`: Doxygen configuration.
- `scripts/`: helper scripts for formatting, checks, and parameter generation.

Inputs, benchmarks, and datasets:

- `prms/`: parameter files for many benchmark setups.
- `benchmarks/`: benchmark geometry/mesh/data files.
- `blood/`: 1D hemodynamics-related assets and scripts used by coupled workflows.
- `notebooks/`: exploratory notebooks and preprocessing utilities.
- `blender/`, `cgal_utilities/`: geometry-generation and geometry-processing helpers.

## Main components

- `ElasticityProblem` (`include/elasticity.h`): bulk elasticity solver with optional transient integration and immersed coupling.
- `PoissonProblem` (`include/laplacian.h`): scalar Poisson/Laplacian immersed-coupling solver.
- `ReducedPoisson` / `ReducedCoupling` / `TensorProductSpace` (`include/reduced_poisson.h`, `include/reduced_coupling.h`, `include/tensor_product_space.h`): reduced-order coupling workflow.
- `Inclusions` (`include/inclusions.h`): immersed geometry, quadrature points, reduced basis data, and coupling metadata.
- `ReferenceCrossSection` (`include/reference_cross_section.h`): reference reduced basis and quadrature construction.
- `ParticleCoupling` (`include/particle_coupling.h`): particle insertion and distributed ownership mapping for coupling points.

## Build requirements

Required:

- CMake >= 3.10
- A C++ compiler supported by your deal.II build
- deal.II (project currently uses APIs compatible with >= 9.2; many workflows target 9.5+)

Optional:

- Trilinos / PETSc as enabled in deal.II
- OpenMP
- GoogleTest (for `gtests`)
- Doxygen (for docs)
- OpenCASCADE, HDF5 (for specific features used by some modules)
- `lib1dsolver` (for coupled 3D/1D executables)

## Build

```bash
mkdir -p build
cd build
cmake -DDEAL_II_DIR=/path/to/deal.II ..
cmake --build . -j
```

Executables are generated from files in `apps/` with names derived from `app_*.cc` (for example `elasticity`, `laplacian`, `reduced_poisson`, `coupled_elasticity`, depending on configuration and build type suffix).

## Running

Typical single-run pattern:

```bash
./elasticity path/to/input.prm
```

or

```bash
./laplacian path/to/input.prm
```

For coupled workflows, use the expected app-specific argument list from the corresponding `apps/app_*.cc` file and matching files in `prms/` or `benchmarks/`.

## Tests

From the build directory:

```bash
ctest --output-on-failure
```

`tests/` are integrated through deal.II testing macros. `gtests/` are enabled when GoogleTest is found.

## Documentation

Generate API docs with:

```bash
doxygen doc/Doxyfile
```

The generated site is written under `doc/` output directories configured in `doc/Doxyfile`.

## Repository notes

- This repository may contain large benchmark/data files and local experiment artifacts.
- Some coupled-elasticity paths depend on an external `lib1dsolver` library and related inputs under `blood/`.

## License

See `LICENSE.md`.
