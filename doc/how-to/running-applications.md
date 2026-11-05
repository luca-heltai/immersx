# Run an application

Applications are built from the entry points in `apps/`. Pass the parameter
file explicitly; do not rely on the current working directory.

```bash
./build/<executable> path/to/input.prm
```

Debug builds append `_debug` to the executable name. For example:

```bash
./build/elastic_static_debug build/tutorials/elastic_static/elastic_static.prm
```

The explicit elasticity example uses a manufactured wave solution and writes
the error table along with the VTU output:

```bash
./build/elasticity_explicit prms/prm_2d_euler_mms.prm
```

The [application reference](../reference/applications) is the authoritative
inventory of executable names, supported dimension combinations, optional
dependencies, and canonical examples. The [tutorials](../tutorials/index)
explain selected workflows rather than repeating the complete parameter
reference.

## Download brain and vessel meshes

Download the brain grid and its vessel grid from the ImmersX `v0.3` release.
The script requires Python 3 and `curl`:

```bash
python3 scripts/download_brain_assets.py
```

The script places `BG001_skull_back_actuator.inp` and `BG0001_alex_scale.vtk`
in `data/brain/` below the repository containing the script, regardless of the
working directory. This directory is ignored by Git. Use `--directory PATH`
to choose another destination.

Downloads are verified against SHA-256 checksums before being placed at their
final paths. Repeated runs reuse verified files and refuse to overwrite files
whose checksums differ. Use the absolute downloaded mesh paths in application
parameter files.
