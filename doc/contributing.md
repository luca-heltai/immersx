# Contributing

This page is the practical guide for contributing code, tests, fixtures,
parameter files, documentation, and build-system changes to ImmersX.

Coding agents must also read the repository-root
[`AGENTS.md`](../AGENTS.md). `AGENTS.md` contains the normative repository and
agent-behavior rules; this page focuses on the human-facing development
workflow and detailed commands.

## Repository layout

The main source-tree areas are:

- `include/immersx/` — public library headers;
- `source/` — compiled implementation sources;
- `apps/` — executable entry points;
- `gtests/` — GoogleTest unit and integration tests;
- `tests/` — deal.II-style regression tests;
- `gtests/parameters/`, `prms/`, and `data/` — parameter files and input data;
- `doc/` — Sphinx/MyST documentation;
- `scripts/` — repository utilities such as `scripts/indent`.

Keep build directories, generated outputs, downloaded dependencies, and test
scratch files out of the source tree whenever possible.

## Configure and build

Always use an out-of-source build.

A typical Debug configuration is:

```bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DDEAL_II_DIR=/path/to/deal.II \
  -DENABLE_GOOGLE_TESTING=ON \
  -DENABLE_DEAL_II_APP_TESTING=ON

cmake --build build-debug -j
```

When `DEAL_II_DIR` is already discoverable, it may be omitted.

For Release:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/path/to/deal.II \
  -DENABLE_GOOGLE_TESTING=ON \
  -DENABLE_DEAL_II_APP_TESTING=ON

cmake --build build-release -j
```

Depending on the deal.II installation, GoogleTest binaries are typically:

```text
build-debug/gtests/gtests_debug
build-release/gtests/gtests
```

Application binaries are generated in the build tree. Debug executables may
carry the `_debug` suffix.

### ccache

ccache is recommended for local development:

```bash
export CCACHE_DIR="$HOME/.ccache"

cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

Parallel git worktrees may share the same ccache, but each worktree must use a
separate CMake build directory.

## Parameter and fixture preprocessing

The top-level `CMakeLists.txt` preprocesses every file ending in `.in` below:

```text
gtests/
tests/
data/
```

The generated file is written under the build tree at the same relative path,
with only the final `.in` suffix removed.

Example:

```text
source:
  gtests/parameters/case.prm.in

generated:
  <build>/gtests/parameters/case.prm
```

The same rule applies to:

```text
.prm.in
.txt.in
.vtk.in
.vtu.in
.pvtu.in
```

and any other `.in` file.

CMake uses `configure_file(... @ONLY)`. Templates may use:

```text
@IMMERSX_SOURCE_DIR@
@IMMERSX_BINARY_DIR@
@TEST_DATA_DIR@
@TEST_OUTPUT_DIR@
```

Example:

```text
set Reduced grid name = @TEST_DATA_DIR@/tests/one_cylinder.vtk
set Output directory  = @TEST_OUTPUT_DIR@/reduced-poisson/one-cylinder
```

Runtime code and tests should open the generated build-tree file, not the
source template.

Do not use relative-runtime assumptions such as:

```text
../data/...
./output/...
```

and do not solve path problems by creating source/build symlinks.

Reconfigure CMake after adding new `.in` files.

## GoogleTest infrastructure

GoogleTests live in `gtests/`.

The common test driver in `gtests/gtest_main.cc` initializes MPI and separates
serial and MPI tests according to the test name.

### Serial and MPI naming

A serial test must not contain `MPI_` in the test-name component:

```cpp
TEST(MyFeature, BasicBehavior)
{
  // ...
}
```

A test intended for MPI execution must contain `MPI_`:

```cpp
TEST(MyFeature, MPI_DistributedBehavior)
{
  // ...
}
```

This convention is used by the shared test driver to select tests according to
the MPI process count.

### Test paths

Use `gtests/test_paths.h` for input and output paths.

Typical helpers include:

```cpp
TestPaths::data_filename("tests/one_cylinder.vtk");
TestPaths::parameter_path("gtests/parameters/case.prm");
TestPaths::output_directory("my-test");
```

Generated output belongs below `TEST_OUTPUT_DIR`.

Do not:

- write generated VTK or result files to the source tree;
- use `chdir()` to make a relative test path work;
- add cwd-dependent paths to inline parameter strings;
- create `build/data -> source/data` symlinks.

For inline parameter text, either construct absolute paths with the helpers or
use configured tokens and `expand_configured_paths()`.

### VTK-dependent tests

When a test requires VTK support, guard it consistently with the project's
existing `DEAL_II_WITH_VTK` convention and exercise the same code path used by
the application where practical.

### Focused test runs

Run one serial test with:

```bash
build-debug/gtests/gtests_debug \
  --gtest_filter='SuiteName.TestName'
```

Run one MPI test with:

```bash
mpirun -np 2 build-debug/gtests/gtests_debug \
  --gtest_filter='SuiteName.MPI_TestName'
```

The MPI test driver may augment the filter to select only MPI-named tests.

### Working-directory robustness

Tests must behave the same when launched from different working directories.

For path-sensitive changes, exercise at least:

```bash
build-debug/gtests/gtests_debug

(cd /path/to/repository && \
  /absolute/path/to/build-debug/gtests/gtests_debug)

(cd /tmp && \
  /absolute/path/to/build-debug/gtests/gtests_debug)

(cd /tmp && \
  mpirun -np 2 /absolute/path/to/build-debug/gtests/gtests_debug)
```

## deal.II-style regression tests

The `tests/` tree uses the deal.II testsuite infrastructure and
`DEAL_II_PICKUP_TESTS()`.

### Source-driven tests

A typical source-driven test consists of:

```text
tests/<category>/<name>.cc
tests/<category>/<name>.output
```

The program writes deterministic output, usually through `deallog`, and the
`.output` file contains the expected result.

### Parameter-driven tests

A parameter-driven test uses an existing executable together with a generated
parameter file.

Store the parameter source as:

```text
tests/<category>/<name>.prm.in
```

CMake generates the runtime `.prm` below the build tree.

The corresponding testsuite directory configures the target through
`TEST_TARGET`, `TEST_TARGET_DEBUG`, or `TEST_TARGET_RELEASE` as appropriate.

### Running regression tests

List tests:

```bash
ctest --test-dir build-debug -N
```

Run all configured tests:

```bash
ctest --test-dir build-debug --output-on-failure
```

Run one regression test verbosely:

```bash
ctest --test-dir build-debug -V -R '<category>/<name>'
```

Use `numdiff` where floating-point expected output requires tolerant
comparison.

## Running applications manually

When debugging application behavior, pass the generated build-tree parameter
file explicitly.

Example:

```bash
build-debug/elasticity_debug \
  build-debug/gtests/parameters/elasticity_strong_dirichlet.prm
```

Do not substitute the source `.prm.in` template or an old source-tree `.prm`
file, because that can hide preprocessing or path problems.

Keep manually generated output in a build-tree or temporary scratch directory.

## CMake changes

When modifying CMake:

- keep builds out of source;
- avoid source-tree-relative runtime assumptions;
- keep installed targets relocatable;
- use standard target-based CMake where possible;
- use `GNUInstallDirs`-style destinations for installable artifacts;
- ensure installed targets do not depend on test-only build-tree resources.

Public headers should remain under:

```text
include/immersx/
```

and public API should live in:

```cpp
namespace ImmersX
{
  // ...
}
```

After adding sources, tests, `.in` templates, or CMake targets, rerun CMake
configuration before concluding that the build system has not detected them.

## Installation and packaging checks

For changes affecting installation or packaging, validate an install into a
temporary prefix.

Example:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/tmp/immersx-install

cmake --build build-release -j
cmake --install build-release
```

Inspect the resulting prefix and verify that:

- public headers are installed in the intended include location;
- libraries are installed in the correct library directory;
- executables are installed in the correct binary directory;
- required runtime data/resources are installed where documented;
- test-only fixtures are not accidentally required at runtime.

If downstream `find_package()` support is changed or added, validate it with a
small external consumer project.

## Documentation

Documentation sources live in `doc/` and use Sphinx/MyST.

When adding a new page, add it to the `doc/index.md` toctree.

Install documentation requirements and build with:

```bash
python3 -m pip install -r doc/requirements.txt
./scripts/build_doc.sh
```

Documentation examples should use the current public API and the real build-tree
path conventions.

Prefer documenting the ordinary user-facing API before internal machinery.

## Validation strategy

During development, use the smallest test that exercises the code being
changed.

Before considering a substantial change complete, broaden validation according
to scope.

For core, solver, distributed, packaging, or cross-cutting changes, typically
run:

1. focused unit/integration tests;
2. complete Debug build;
3. serial GoogleTest suite;
4. two-rank MPI GoogleTest suite;
5. relevant CTest/regression tests;
6. Release build and relevant tests;
7. working-directory robustness checks for path-sensitive changes.

Typical commands:

```bash
cmake --build build-debug -j

build-debug/gtests/gtests_debug

mpirun -np 2 build-debug/gtests/gtests_debug

ctest --test-dir build-debug --output-on-failure
```

For Release:

```bash
cmake --build build-release -j

build-release/gtests/gtests

mpirun -np 2 build-release/gtests/gtests

ctest --test-dir build-release --output-on-failure
```

Keep independent correctness tests when they validate different layers of the
software. For example, a physics-level residual/Jacobian test and a solver-level
integration test are complementary rather than redundant.

## Formatting and commits

The repository formatter is:

```bash
./scripts/indent
```

Run it from the repository root before every commit.

Because it formats the repository globally, inspect the complete resulting diff
and make sure unrelated changes are not included.

Useful checks:

```bash
git diff
git diff --check
git status
```

Implementation work should be split into small, meaningful local commits at
logical milestones.

Do not push, open a pull request, or merge into `master` unless explicitly
requested.

## CI reference

The repository GitHub Actions workflows under `.github/workflows/` are the
authoritative reference for the current CI matrix.

When reproducing CI locally, inspect the workflow first rather than assuming
that image tags, optional dependencies, or exact commands have remained
unchanged.

The normal local baseline remains:

- Debug and Release when supported;
- serial GoogleTests;
- two-rank MPI GoogleTests;
- CTest regression tests.

## Final contributor checklist

Before handing off a substantial change:

- [ ] Build is out of source.
- [ ] No generated output was written into the source tree.
- [ ] New test inputs use the `.in` preprocessing convention where appropriate.
- [ ] MPI GoogleTests contain `MPI_` in the test name.
- [ ] Tests are independent of the working directory.
- [ ] Relevant Debug tests pass.
- [ ] Relevant MPI tests pass.
- [ ] Relevant CTest tests pass.
- [ ] Release validation was performed when appropriate.
- [ ] Documentation was updated when public behavior changed.
- [ ] New documentation pages were added to `doc/index.md`.
- [ ] `./scripts/indent` was run before each commit.
- [ ] The complete diff was reviewed.
- [ ] Commits are small and meaningful.
- [ ] Nothing was pushed and no pull request was opened unless explicitly
      requested.
