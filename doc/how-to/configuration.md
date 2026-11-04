# Configure a parameter file

ImmersX applications read a deal.II `ParameterHandler` file. The root
`dimension` and `space dimension` entries select the statically instantiated
problem where an application supports more than one dimension. The input
filename never selects the dimension.

Keep parameter files in a repository or build-tree directory and pass the
path explicitly:

```bash
./build/poisson path/to/poisson.prm
```

For repository tests, use `.prm.in` templates under `gtests/`, `tests/`,
`data/`, or `tutorials/`. CMake replaces `@IMMERSX_SOURCE_DIR@`,
`@IMMERSX_BINARY_DIR@`, `@TEST_DATA_DIR@`, and `@TEST_OUTPUT_DIR@` and writes
the generated file below the build tree.

Use the [application reference](../reference/applications) to identify the
executable and the [API reference](../api/library_root) for the
parameter class behind a section. Tutorials show only the options needed for
their example; they are not complete parameter manuals.

## Gmsh physical identifiers

For a Gmsh mesh read through `Domain`, numeric physical tags select material
ids on cells and boundary ids on lower-dimensional entities. A physical group
does not need a name. Named selectors such as `MaterialID:7,ManifoldID:2` or
`BoundaryID:5,ManifoldID:3` override those defaults. Groups without a physical
tag retain id zero.

Some deal.II Gmsh API readers discard unnamed physical tags. ImmersX works
around this during import by making only unnamed groups' numeric ids explicit
in a temporary MSH 4.1 mesh, then delegating to the same deal.II reader. Named
selectors, untagged entities, and the source file remain unchanged. Imports
requiring this workaround perform an extra Gmsh read/write; temporary files are
removed and the Gmsh output-format options are restored, including on failure.
This protects `Domain` imports, not direct calls to a separately installed
deal.II reader.
