# Getting Started

## Build the code

```bash
mkdir -p build
cd build
cmake -DDEAL_II_DIR=/path/to/deal.II ..
cmake --build . -j
```

Executables are generated from `apps/app_*.cc`.

## Build the documentation site

The documentation pipeline requires:

- `doxygen`
- `sphinx-build`
- `breathe`
- `exhale`
- `myst-parser`

Generate the full site from the repository root with:

```bash
./scripts/build_doc.sh
```

The final static site is emitted under `build/docs/site/`.

If the project is already configured with CMake, you can use:

```bash
cmake --build build --target doc
```

## Typical run

```bash
./build/elasticity path/to/input.prm
```

For coupled workflows, inspect the corresponding source file in `apps/` and the matching inputs under `prms/` or `benchmarks/`.
