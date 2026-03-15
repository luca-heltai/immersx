# Testing

From the build directory:

```bash
ctest --output-on-failure
```

Test layout:

- `tests/` contains `deal.II`-style regression tests.
- `gtests/` contains GoogleTest-based unit and integration tests.

GoogleTest targets are enabled only when `GTest` is available at configure time.
