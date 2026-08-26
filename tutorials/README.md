# Tutorial assets

This directory stores canonical runnable inputs used by the documentation
tutorials.

The layout follows learning workflows rather than one page per executable. For
example:

- `tutorials/poisson/`
- `tutorials/reduced_poisson/`
- `tutorials/elastic_static/`
- `tutorials/elasticity/` (verification and specialized legacy cases)
- `tutorials/elastodynamics/`
- `tutorials/navier_stokes/`
- `tutorials/coupled_poisson_elasticity/`
- `tutorials/fiber_reinforced_elastodynamics/`

Inputs ending in `.prm.in` are configured by CMake into the build tree. They
may use `@TEST_DATA_DIR@` and `@TEST_OUTPUT_DIR@` so tests and documentation
commands do not depend on the current working directory. The application smoke
tests use these same generated inputs.
