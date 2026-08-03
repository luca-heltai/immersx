AGENTS.md

This repository uses CMake as its build system and provides two test infrastructures:

- GoogleTest-based unit tests (under gtests/)
- deal.II-style testsuite/regression tests (under tests/)

This document explains how to add tests to both infrastructures, how to run them locally, and includes tips and common pitfalls.

1) Quick: build & run all tests

  mkdir -p build
  cd build
  cmake ..
  make -j
  ctest -j4

Notes:
- Use -V with ctest for verbose output (ctest -V -R "regex").
- On multi-config generators (Visual Studio / Xcode / CMake multi-config) adapt build commands accordingly.

2) GoogleTest-based tests (gtests/)

Overview
- The directory gtests/ contains GoogleTest-based unit tests. The repository's top-level CMakeLists checks for GTest and adds this subdirectory when found.
- gtests/CMakeLists.txt currently glob-matches *.cc files and builds them into an executable (postfixed per debug/release build type) which links against the compiled project library.

How to add a GoogleTest
- Create a new test source file in gtests/, for example:

    gtests/my_feature_01.cc

  Implement your TEST or TEST_F cases as usual with the GoogleTest API. There is already a gtest_main.cc in gtests/ providing a main() for the suite; do not add another main unless you intend to manage test discovery differently.

- The existing gtests/CMakeLists.txt uses file(GLOB _test_files *cc) so new files will be picked up automatically on the next CMake configure run. If your CMake generator caches file lists, re-run CMake (cmake ..) or reconfigure your build directory.

How the CMake integration works (summary)
- The project builds one executable per build type named gtests or gtests_debug (depending on build type). The executable links against the project's test library (TEST_LIBRARIES_<BUILD_TYPE>), GTest libraries (${GTEST_LIBRARIES} and ${GTEST_MAIN_LIBRARY}), and uses DEAL_II_SETUP_TARGET to get deal.II target settings.
- If you prefer to build separate executables per test, you can edit gtests/CMakeLists.txt and add per-file add_executable() calls, link to TEST_LIBRARIES_<BUILD_TYPE> and call DEAL_II_SETUP_TARGET on the target.

Recommended example test file (gtests/my_feature_01.cc)

  #include <gtest/gtest.h>
  #include "some_project_header.h" // project includes

  TEST(MyFeature, BasicBehaviour)
  {
    EXPECT_EQ(1, my_function_under_test());
  }

CMakeNotes / pitfalls
- If GTest is not found, the top-level CMake will skip gtests and print a warning. Install GTest or provide it via package manager or submodule.
- Re-run CMake after adding sources if the generator caches file lists.
- The gtests/CMakeLists is already set up to build for both Debug and Release configurations when deal.II was built for both; this creates postfixed executable names (e.g., _debug).

Running Google tests
- After build: run ctest which will discover the GoogleTest executable(s). You can also run the test binary directly from build/tests/<...> if you prefer.
- To run a single gtest: ctest -R "gtests" -VV or run the binary with --gtest_filter=MyFeature.*

MPI test naming and execution
- `gtests/gtest_main.cc` automatically changes the GoogleTest filter based on
  the number of MPI ranks. With one rank it excludes tests whose names match
  `*.MPI_*`; with multiple ranks it includes that pattern.
- Name every test that is intended to run in parallel with an `MPI_` test-name
  component, for example `TEST(MyFeature, MPI_DistributedSolve)` or
  `TEST_P(MyFixture, MPI_DistributedSolve)`. A test without `MPI_` is a serial
  test by default and will not be selected by the default `mpirun` invocation.
- Run the binaries from the build directory. Relative test data paths such as
  `../data/tests/...` are resolved from there:

      (cd build-debug && ./gtests/gtests_debug)
      (cd build-debug && mpirun -np 2 ./gtests/gtests_debug)

  Use `gtests` instead of `gtests_debug` for a Release build. A focused serial
  run can select non-MPI tests with `--gtest_filter='SuiteName.*'`; a focused
  parallel run should select the MPI-named suite/test, for example
  `--gtest_filter='*ElasticityTensorProductCouplingTriangulationTypeTest.MPI_*'`. Remember
  that the repository's MPI main adds `*.MPI_*` to the requested filter.

3) deal.II-style tests (tests/)

Overview
- deal.II test infrastructure expects small programs or parameter-driven tests plus an expected output file (.output). The repo already calls DEAL_II_PICKUP_TESTS() in tests/CMakeLists.txt, which scans the tests/ tree for *.cc and *.output pairs and generates CTest entries.
- This mirrors the testsuite used by the deal.II library itself and supports many features: debug/release variants, feature restrictions, mpirun counts, thread settings, expect=..., run_only, and more.

How to add a deal.II testsuite test

Type A: parameter-file driven (use an existing executable)
- Place a parameter file and a .output in tests/<category>/:

    tests/<category>/my_case.prm
    tests/<category>/my_case.output

- In this mode CTest will call a pre-built executable (configured via TEST_TARGET in CMake) with the parameter file as its first argument. You must set TEST_TARGET in the corresponding CMakeLists if necessary (see examples below).

Type B: source-driven test (standalone test executable)
- Add a small executable source with an int main() that writes its results to deallog or to stdout/file "output".
- Place the source and a comparison file of expected output next to each other:

    tests/<category>/my_test.cc
    tests/<category>/my_test.output

- DEAL_II_PICKUP_TESTS() will compile the source into an executable and create CTest entries that build, run, and diff the generated output against the committed .output file.

Minimal example: adding a new test category
- Create a folder tests/my_category/ and add a CMakeLists.txt containing:

    CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)
    INCLUDE(../setup_testsubproject.cmake)
    PROJECT(testsuite CXX)
    INCLUDE(${DEAL_II_TARGET_CONFIG})
    DEAL_II_PICKUP_TESTS()

- Populate the folder with one or more test pairs (my_test.cc + my_test.output) or param/.output combinations.

Useful macros and variables
- DEAL_II_PICKUP_TESTS() — scan directory and create test targets based on *.output files
- TEST_TARGET / TEST_TARGET_DEBUG / TEST_TARGET_RELEASE — specify an executable to be invoked for parameter-driven tests
- TEST_LIBRARIES — if your test is source-built but needs to link additional libraries from your project, set TEST_LIBRARIES in the directory CMake before invoking DEAL_II_PICKUP_TESTS()
- NUMDIFF_EXECUTABLE — path to numdiff to perform tolerant floating-point comparison
- TEST_TIME_LIMIT, TEST_MPI_RANK_LIMIT, TEST_THREAD_LIMIT — limits passed to the testsuite when setting up tests

How to access auxiliary files from a test
- When a test needs auxiliary data files (e.g. .data or .prm that refer to data), use the preprocessor define SOURCE_DIR (passed by the repository CMake to test targets) to form paths relative to the source tree. Example inside test code:

    const std::string data_file = std::string(SOURCE_DIR) + "/tests/my_category/my_data.data";

This ensures tests find their input files regardless of the build directory layout.

Running and debugging an individual deal.II test
- After building, run ctest -V -R "category/my_test" to build, run, and print verbose logs for the test
- Inspect BUILD_DIR/Testing/Temporary/LastTest.log or the output files under BUILD_DIR/tests/... for details if the test failed in BUILD/RUN/DIFF stages.

Common file name conventions
- <testname>.output is mandatory. You can create an empty .output first, run test to capture produced output, then copy build output into the source .output file to accept it as the expected result.
- You can annotate output file names with suffixes to restrict to debug/release or feature variations, e.g. my_test.debug.output, my_test.with_umfpack=on.output, my_test.mpirun=4.output, etc. See deal.II testsuite docs for exact naming rules.

4) Examples (practical)

- Add a new GoogleTest unit:
  - Create gtests/my_new_feature.cc with TEST() cases
  - Run cmake .. to pick it up, build and run tests

- Add a new deal.II test that runs an existing executable with a parameter file:
  - Put my_case.prm and my_case.output in tests/my_category/
  - In tests/my_category/CMakeLists.txt set SET(TEST_TARGET my_executable)
  - Add the directory to top-level CMake if tests folder not already included
  - Reconfigure and run ctest -V -R "my_category/my_case"

- Add a new testsuite source test:
  - Create tests/my_category/my_test.cc (see template below)
  - Create tests/my_category/my_test.output (can be empty; run once to produce output then copy back)
  - run cmake .. && ctest -V -R "my_category/my_test"

Template for a simple deal.II test source:

  #include "../tests.h"   // repository deals.h/tests.h driver includes
  int main()
  {
    initlog(); // routes deallog to file named "output"
    deallog << "Hello test" << std::endl;
    return 0;
  }

5) Troubleshooting and tips

- No tests found by ctest: ensure ENABLE_TESTING() was called (top-level CMake does call it) and that DEAL_II_PICKUP_TESTS() discovered *.output files. Also ensure you re-ran CMake after adding new test files.
- Test fails at DIFF stage: run ctest -V -R "regex" to see generated output path (BUILD_DIR/tests/...) and compare against committed .output. If differences are numerical, install and configure numdiff and set NUMDIFF_EXECUTABLE so tests use tolerant comparison.
- Tests that need data files: use SOURCE_DIR or construct file paths at runtime so tests find resources regardless of build directory.
- Multi-config / build-type issues: the project config builds both Debug and Release variants depending on how deal.II was configured. Look for postfixed binaries (_debug) when invoking them directly.

6) CI (GitHub Actions)

- The repository provides GitHub Actions workflows in .github/workflows. The main tests workflow is .github/workflows/tests.yml and runs two jobs (debug and release) in a dealii container image (dealii/dealii:v9.7.1-noble).
- The workflow downloads `lib1dsolver.a` before each build: release tag `v0.2` for
  Debug and release tag `v0.1` for Release. The file must be in the repository
  root before configuring.
- The CI uses Ninja, OpenMP, and separate build directories. The commands below
  reproduce `.github/workflows/tests.yml` inside the `dealii/dealii:v9.7.1-noble`
  container (the workflow also sets `OMPI_ALLOW_RUN_AS_ROOT=1` and
  `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`):

      # Debug
      rm -rf build_linux_debug
      mkdir build_linux_debug
      cd build_linux_debug
      cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-fopenmp
      ninja
      ctest -N
      ctest --output-on-failure
      ./gtests/gtests_debug
      mpirun -n 2 ./gtests/gtests_debug

      # Release (run from the repository root)
      rm -rf build_linux_release
      mkdir build_linux_release
      cd build_linux_release
      cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fopenmp
      ninja
      ctest -N
      ctest --output-on-failure
      ./gtests/gtests
      mpirun -n 2 ./gtests/gtests

- MPI tests should pass with two ranks. When running locally, `mpirun -np 2` is
  equivalent to the workflow's `mpirun -n 2`.

- Recommendation: when adding tests, mention any CI-specific requirements in the PR (needs lib1dsolver.a, relies on MPI, long-running test, etc.) so reviewers and CI maintainers can adjust or gate the test.

7) References
- deal.II: Setting up testsuite in user projects — https://dealii.org/9.6.0/users/testsuite.html
- deal.II testsuite developer documentation — https://dealii.org/developer/testsuite.html
- GoogleTest docs — https://github.com/google/googletest

If you want, I can:
- Create a short CONTRIBUTING_TESTS.md with a checklist for new tests (suitable for PR templates), and
- Add an example new test in tests/ and gtests/ to show the workflow (I will keep changes minimal and create one test scaffold each).


---
Generated by el Gentleman (Pi coding harness). If you want me to add example tests or CI integration rules (GitHub Actions to run ctest), say which CI provider and acceptance criteria to use.

## Elasticity coupling representations

Elasticity supports two coupling representations selected by
`Immersed Problem / Coupling type`:

- `Point`: discrete inclusion centers with finite radii. Segment clustering is
  an internal grouping of these point inclusions.
- `TensorProduct`: a VTK reduced mesh coupled with a reference cross section
  through `ReducedCoupling`, `TensorProductSpace`, `ParticleCoupling`, and
  `VTKUtils`.

The tensor-product representation currently supports one setup/solve cycle;
adaptive refinement is explicitly rejected until tensor-product particle
transfer is implemented. `ReducedPoisson` remains the reference implementation
of the underlying tensor-product coupling infrastructure.

## Build and testing

Use out-of-source builds from the build directory:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_GOOGLE_TESTING=ON
cmake --build build-debug --target gtests_debug -j$(nproc)
(cd build-debug && ./gtests/gtests_debug)
(cd build-debug && mpirun -np 2 ./gtests/gtests_debug)

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_GOOGLE_TESTING=ON
cmake --build build-release --target gtests -j$(nproc)
(cd build-release && ./gtests/gtests)
(cd build-release && mpirun -np 2 ./gtests/gtests)
```

During development, run the focused tensor-product elasticity tests with:

```bash
./build-debug/gtests/gtests_debug --gtest_filter='*ElasticityTensorProductCoupling*'
```

New or modified code must compile without warnings in both configurations. Existing
third-party deprecation warnings should be documented rather than suppressed
 globally. Add a parameter parsing test whenever a parameter is introduced. New
immersed geometry code must be guarded by `DEAL_II_WITH_VTK` and should read VTK
input through the tensor-product coupling infrastructure.
