# Documentation

The documentation site is built with:

- `Doxygen` for API extraction
- `Sphinx` for site generation
- `Breathe` and `Exhale` for rendering Doxygen XML
- `MyST parser` for Markdown support

Build the site with:

```bash
python3 -m pip install -r doc/requirements.txt
./scripts/build_doc.sh
```

The published static site is written to `build/docs/site/`.

Serve it locally with:

```bash
./scripts/serve_doc.sh
```

This is preferable to opening the generated HTML files directly with `file://`, especially for JavaScript-driven features such as MathJax rendering.

If the project is already configured with CMake and the required tools are available, you can also run:

```bash
cmake --build build --target doc
```
