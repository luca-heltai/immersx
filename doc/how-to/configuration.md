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
