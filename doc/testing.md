# Testing

The repository tests run continuously on GitHub Actions:

[![Tests](https://github.com/luca-heltai/immersx/actions/workflows/tests.yml/badge.svg)](https://github.com/luca-heltai/immersx/actions/workflows/tests.yml)
[![Documentation](https://github.com/luca-heltai/immersx/actions/workflows/doxygen.yml/badge.svg)](https://github.com/luca-heltai/immersx/actions/workflows/doxygen.yml)
[![Indentation](https://github.com/luca-heltai/immersx/actions/workflows/indentation.yml/badge.svg)](https://github.com/luca-heltai/immersx/actions/workflows/indentation.yml)

See the [GitHub Actions workflows](https://github.com/luca-heltai/immersx/actions)
for current runs, logs, and artifacts.

From the build directory:

```bash
ctest --output-on-failure
```

Test layout:

- `tests/` contains `deal.II`-style regression tests.
- `gtests/` contains GoogleTest-based unit and integration tests.

GoogleTest targets are enabled only when `GTest` is available at configure time.
