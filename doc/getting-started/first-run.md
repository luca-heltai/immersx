# First run: Poisson

This is the smallest complete ImmersX workflow. It introduces parameter-file
configuration, mesh generation, a scalar finite-element problem, and output.

The executable is `poisson`, and the canonical input is
`tutorials/poisson/poisson_2d.prm.in`. CMake configures the `.in` file into the
build tree so that its output and data paths are independent of the launch
directory.

```{literalinclude} ../../tutorials/poisson/poisson_2d.prm.in
:language: ini
```

From the repository root, run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/path/to/deal.II
cmake --build build -j
./build/poisson build/tutorials/poisson/poisson_2d.prm
```

The solution is written below `build/test_output/tutorial-output/poisson-2d`. The exact
location is controlled by `TEST_OUTPUT_DIR` in the configured input. Continue
with the [Poisson tutorial](../tutorials/poisson) to understand the choices,
then follow the [learning path](../tutorials/index).
