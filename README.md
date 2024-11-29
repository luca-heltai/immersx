Reduced Lagrange Multipliers Method
===================================

![GitHub CI](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/tests.yml/badge.svg)
![Documentation](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/doxygen.yml/badge.svg)
![Indent](https://github.com/luca-heltai/reduced_lagrange_multipliers/actions/workflows/indentation.yml/badge.svg)

This repository implements the Reduced Lagrange Multiplier method for non-matching coupling of mixed-dimensional domains, as described in the paper:

**Reduced Lagrange multiplier approach for non-matching coupling of mixed-dimensional domains**  
Authors: Luca Heltai, Paolo Zunino  
Published in *Mathematical Models and Methods in Applied Sciences* (2023)  
DOI: [10.1142/S0218202523500525](https://dx.doi.org/10.1142/S0218202523500525)

## Overview

In many physical problems, especially those involving heterogeneous spatial scales, we encounter coupled partial differential equations (PDEs) defined on domains of different dimensions embedded into each other. Examples include:

- Flow through fractured porous media
- Fiber-reinforced materials
- Modeling small circulation in biological tissues

This repository provides a computational framework for coupling PDEs across dimensions using a reduced Lagrange multiplier approach. The method ensures stability and robustness, with particular attention to the smallest characteristic length of the embedded domain.

## Features

- **General Mathematical Framework**: This framework provides tools for analyzing and approximating coupled PDEs across different dimensions.
- **Non-Matching Coupling**: The method applies to non-matching interfaces, where coupling constraints are enforced via Lagrange multipliers.
- **Dimensionality Reduction**: Supports model reduction techniques that simplify complex 3D problems into more tractable ones by working in lower-dimensional spaces, and employing a *reduced Lagrange multiplier* space.
- **Inf-Sup Stability**: Ensures stable and robust numerical solutions across a range of configurations and mesh sizes.

## Installation

To use this code, you need a working deal.II version (at least version 9.5). Clone the repository and run the following commands:

```bash
git clone https://github.com/luca-heltai/reduced_lagrange_multiplier.git
cd reduced_lagrange_multiplier
mkdir build
cd build
cmake -DDEAL_II_DIR=/path/to/deal.II ..
make
```

## Online documentation

The documentation is built and deployed at each merge to master. You can
find the latest documentation here:

<https://luca-heltai.github.io/reduced_lagrange_multipliers/>

Licence
=======

See the file [LICENSE.md](./LICENSE.md) for details

USE OF COUPLED ELASTICITY
=========================

run in parallel as

export OMP_NUM_THREADS=1

mpirun -np n ./build/coupled_elasticity_debug <path_to_input_3d> <path_to_input_1d> <couplingSampling> <couplingStart> 0
if we only want the the 1D simulation then set coupling Start to 100
if we only want the 3D Simulation then only give <path_to_input_3d>

random error "invalid template argument" solved by changing the order od #include in app_*
