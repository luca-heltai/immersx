# Coupled 3D/1D Problem Instructions

This file collects the operational notes for the coupled elasticity workflow
(`app_coupled_elasticity.cc`) that were previously in the main `README.md`.

## Install FVM (1D solver)

1. Clone the FVM repository.
2. Start a container with deal.II:

```bash
docker run -ti -v ./:/fvm --platform linux/amd64 dealii/dealii:v9.6.0-jammy
```

3. Inside the container:

```bash
cd /fvm/
make -f MakefileGCCDesktop clean all
```

## Compile the coupled code

Build with OpenMP enabled (needed by the 1D code):

```bash
cmake -DCMAKE_CXX_FLAGS=-fopenmp .
cd build
ninja -jX
```

where `X` is the number of processes used for compilation.

## Run

For parallel runs:

```bash
export OMP_NUM_THREADS=1
mpirun -np n ./build/coupled_elasticity_debug <path_to_input_3d> <path_to_input_1d> <couplingSampling> <couplingStart> 0
```

Notes:

- To run only the 1D simulation, set `couplingStart` to `100`.
- To run only the 3D simulation, provide only `<path_to_input_3d>`.

## Legacy note

An observed compile issue (`invalid template argument`) was historically worked
around by changing include order in `app_*` files.
