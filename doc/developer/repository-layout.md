# Repository layout

The main source-tree areas are:

- `include/immersx/` — public headers;
- `source/` — compiled library sources;
- `apps/` — application entry points;
- `gtests/` — GoogleTest unit and integration tests;
- `tests/` — deal.II-style regression tests;
- `gtests/parameters/`, `tutorials/`, and `data/` — parameter files and input
  data;
- `doc/` — Sphinx/MyST documentation;
- `scripts/` — build, formatting, and documentation utilities.

Build directories and generated files belong outside the source tree. CMake
configures `.in` inputs from `gtests/`, `tests/`, `data/`, and `tutorials/`
into the build tree with the same relative path and the final `.in` suffix
removed.
