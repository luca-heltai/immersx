# Contributing

This page is the practical guide for contributing code, parameter files,
fixtures, tests, and documentation to ImmersX. The project is a CMake-based
C++ application built on deal.II. Before opening a pull request, read
`AGENTS.md` in the repository root as well; it records the repository-specific
rules and the expected validation commands.

## Repository layout

- `include/` and `source/` contain the library interfaces and implementations.
- `apps/` contains executable entry points. Files named `app_*.cc` produce the
  corresponding application executable without the `app_` prefix.
- `gtests/` contains the GoogleTest unit and integration suite.
- `tests/` contains deal.II-style regression tests and expected output files.
- `gtests/parameters/`, `prms/`, and `data/` contain parameter files and input
  data. Files under `gtests/`, `tests/`, and `data/` ending in `.in` are build
  templates, not runtime files.
- `doc/` contains the Sphinx/MyST documentation and generated API inputs.
- `scripts/` contains repository utilities, including `scripts/indent`.

Keep generated output, build directories, and downloaded libraries out of the
source tree whenever possible. In particular, do not use `data/tests` or a
source-side `tests_*_output` directory as a test output directory.

## Configure and build

Use an out-of-source build. A Debug build is the usual development build:

```bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DDEAL_II_DIR=/path/to/deal.II \
  -DENABLE_GOOGLE_TESTING=ON \
  -DENABLE_DEAL_II_APP_TESTING=ON
cmake --build build-debug -j
```

For a Release build, use `-DCMAKE_BUILD_TYPE=Release`. Depending on the
deal.II installation, the GoogleTest executable is named
`build-debug/gtests/gtests_debug` in Debug and `build-release/gtests/gtests` in
Release. The application executables are placed at the top of the build tree,
for example `elasticity_debug`, `laplacian_debug`, or `reduced_poisson_debug`.

GoogleTest is enabled only when CMake finds GTest. The deal.II regression
tests are enabled by `ENABLE_DEAL_II_APP_TESTING`. OpenMP, VTK, Trilinos,
PETSc, HDF5, and the coupled 1D solver remain feature-dependent on the deal.II
and local installations.

## Parameter-file convention

A parameter file used by a test is kept as a `.prm.in` template. CMake applies
the central rule in the top-level `CMakeLists.txt` to every `.in` file below
`gtests/`, `tests/`, and `data/`:

```text
source:   gtests/parameters/case.prm.in
generated: <build>/gtests/parameters/case.prm
```

The relative path is preserved and only the final `.in` suffix is removed.
The rule applies to any file type, including `.prm.in`, `.txt.in`, `.vtk.in`,
`.vtu.in`, and `.pvtu.in`. `configure_file(... @ONLY)` substitutes the
explicit path variables while leaving other CMake-like text untouched.

Templates may use these absolute CMake variables:

- `@IMMERSX_SOURCE_DIR@` — repository source root;
- `@IMMERSX_BINARY_DIR@` — build root;
- `@TEST_DATA_DIR@` — generated build-tree data root;
- `@TEST_OUTPUT_DIR@` — build-tree scratch/output root.

For example:

```text
set Reduced grid name = @TEST_DATA_DIR@/tests/one_cylinder.vtk
set Output directory  = @TEST_OUTPUT_DIR@/reduced-poisson/one-cylinder
```

Do not add `../data/...`, `./output/...`, or another current-working-directory
assumption to a test parameter file. A consumer must open the generated file
under the build tree. For C++ tests, use the helpers in
`gtests/test_paths.h`, such as `data_filename("tests/one_cylinder.vtk")`,
`parameter_path("gtests/parameters/case.prm")`, and
`output_directory("my-test")`.

When adding a new fixture, make it a `.in` template if it is part of the test
input set. If the fixture contains no substitutions, its contents can simply
be copied into the template. Reconfigure after adding a template so CMake
registers it and emits the corresponding build-tree file.

## Writing robust GoogleTests

The common test driver in `gtests/gtest_main.cc` initializes MPI and selects
tests according to the number of ranks. A serial test must not contain `MPI_`
in its test name. A test intended for the two-rank run must contain an
`MPI_` component, for example:

```cpp
TEST(MyCoupling, MPI_DistributedAssembly)
{
  // ...
}
```

Use `gtests/test_paths.h` for every input and output path. Inputs must be
absolute paths obtained from the configured build/source roots; outputs must
be below `TEST_OUTPUT_DIR`. Do not create symlinks such as `build/data ->
source/data`, do not call `chdir()` to make a relative input work, and do not
write generated VTK, parameter, log, or convergence files into the source
tree. A test that writes files should use a unique subdirectory below
`output_directory("<test-name>")`.

Inline parameter strings need the same treatment. Either compose the string
with a helper-returned absolute path or use a configured token and expand it
with `expand_configured_paths()`. Do not put `../data/...` in a raw parameter
string.

If a test depends on VTK input, guard the test with `DEAL_II_WITH_VTK` and
exercise the same VTK/coupling path used by the application. Keep numerical
assertions focused and document tolerances when they are materially larger
than machine precision. Add a parameter parsing test when introducing or
changing a parameter.

## Adding deal.II regression tests

The `tests/` infrastructure uses `DEAL_II_PICKUP_TESTS()`. A source-driven
test normally consists of:

```text
tests/<category>/<name>.cc
tests/<category>/<name>.output
```

The executable writes deterministic output through `deallog`, and the
`.output` file records the expected result. Parameter-driven tests use an
existing target configured through the relevant CMake variables. Auxiliary
parameter or data files should use the same `.in` convention and must resolve
to generated build-tree paths.

Run one regression test verbosely with:

```bash
ctest --test-dir build-debug -V -R '<category>/<name>'
```

Use `numdiff` when a test intentionally compares floating-point output with a
tolerance, and keep expected output stable across supported configurations.

## Running tests and applications

List configured tests and run the regression suite with:

```bash
ctest --test-dir build-debug -N
ctest --test-dir build-debug --output-on-failure
```

Run the serial GoogleTest binary directly:

```bash
build-debug/gtests/gtests_debug
build-debug/gtests/gtests_debug --gtest_filter='SuiteName.TestName'
```

Run the MPI-selected tests with two ranks:

```bash
mpirun -np 2 build-debug/gtests/gtests_debug
```

The binary must behave identically when launched from the build directory,
the repository root, or an unrelated directory. A useful smoke check is:

```bash
build-debug/gtests/gtests_debug
(cd . && /absolute/path/to/build-debug/gtests/gtests_debug)
(cd /tmp && /absolute/path/to/build-debug/gtests/gtests_debug)
(cd /tmp && mpirun -np 2 /absolute/path/to/build-debug/gtests/gtests_debug)
```

Application tests use generated parameter files and a build-tree scratch
directory. To run an application manually, pass the generated parameter file
explicitly:

```bash
build-debug/elasticity_debug \
  build-debug/gtests/parameters/elasticity_strong_dirichlet.prm
```

Do not use a source-tree `.prm` as a substitute for the generated file when
debugging the test workflow; doing so can hide path and preprocessing errors.

## Documentation

Documentation sources live under `doc/` and use Sphinx with MyST Markdown.
Add a page to the `doc/index.md` toctree when introducing a new topic. Build
the documentation after changes to prose or examples:

```bash
python3 -m pip install -r doc/requirements.txt
./scripts/build_doc.sh
```

Keep commands in documentation aligned with the current CMake and test
conventions. In particular, document generated parameter paths and explicit
working-directory-independent invocations.

## Formatting and review checklist

Before committing or opening a pull request:

1. Run `./scripts/indent` from the repository root.
2. Review the resulting diff, including formatter-only changes.
3. Reconfigure after adding CMake inputs or test templates.
4. Build the relevant Debug and Release targets when the change affects
   configuration or portable C++ code.
5. Run focused tests first, then the relevant serial, MPI, and CTest suites.
6. Confirm that no generated files were written below the source tree.

Pull requests should explain the behavior changed, the tests run, and any
feature requirements such as VTK, MPI, OpenMP, or the coupled 1D solver.
