# Testing

The repository tests run continuously on GitHub Actions:

[![Tests](https://github.com/luca-heltai/immersx/actions/workflows/tests.yml/badge.svg)](https://github.com/luca-heltai/immersx/actions/workflows/tests.yml)
[![Documentation](https://github.com/luca-heltai/immersx/actions/workflows/doxygen.yml/badge.svg)](https://github.com/luca-heltai/immersx/actions/workflows/doxygen.yml)
[![Indentation](https://github.com/luca-heltai/immersx/actions/workflows/indentation.yml/badge.svg)](https://github.com/luca-heltai/immersx/actions/workflows/indentation.yml)

See the [GitHub Actions workflows](https://github.com/luca-heltai/immersx/actions)
for current runs, logs, and artifacts.

CTest is the authoritative test runner. From the build directory:

```bash
ctest --output-on-failure                 # full suite
ctest -L quick --output-on-failure        # normal development gate
ctest -L unit --output-on-failure
ctest -L integration --output-on-failure
ctest -L validation --output-on-failure
ctest -L application --output-on-failure
```

Test layout:

- `unit` contains focused checks of individual data structures, parsing,
  algebra, geometry, and other reusable components.
- `integration` contains interactions among ImmersX components, distributed
  execution, and solver/adapter composition.
- `validation` contains manufactured-solution, convergence, oracle, and
  physics-validation checks.
- `application` contains executable and end-to-end application behavior.

Both `gtests/` and `tests/` use these primary categories. `quick` is an
orthogonal label applied to all unit and integration tests, while `mpi` marks
tests that require more than one MPI rank. GoogleTest categories are CTest
groups over one aggregate executable, so sources are compiled once and CTest
remains the single orchestration layer.

To inspect the inventory and labels:

```bash
ctest -N
ctest -N -V
```

For focused development, use CTest when possible:

```bash
ctest -R 'ImmersX\.integration\.Serial' --output-on-failure
ctest -R 'ImmersX\.integration\.MPI' --output-on-failure
```

GoogleTest targets are enabled only when `GTest` is available at configure time.

Pull requests run the `quick` label. Pushes to `master` run the full CTest
manifest. CI does not rerun the aggregate GoogleTest binary outside CTest.
