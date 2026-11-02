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

## Navigation and headings

The navigation has nine top-level sections, in this order: Getting started,
Tutorials, How-to guides, Concepts, Reference, Integrations, Performance,
Development, and About. Keep `doc/index.md` limited to those section indexes.
Add each authored page to exactly one owning section or tutorial-group
`toctree`; use ordinary links for other discovery paths. Generated API pages
belong under Reference. Changelog belongs under About.

Classify pages by purpose: first-run instructions under Getting started,
learning sequences under Tutorials, task recipes under How-to guides,
mathematics and explanations under Concepts, factual lookups under Reference,
external-platform workflows under Integrations, benchmark results under
Performance, and contribution or extension instructions under Development.
Tutorials may use the Basic problems, Immersed and coupled problems, Fiber
reinforcement, and MetricFlowX coupling groups. Discuss new top-level sections
or tutorial groups before adding them.

Use `:titlesonly:` in navigation toctrees and `:maxdepth: 1` in section and
group indexes; the root uses `:maxdepth: 2`. The left sidebar lists pages;
the page outline lists internal headings. Each Markdown page has one `#`
title, then `##` sections and `###` subsections without skipped levels. Use
sentence case and descriptive, concise navigation labels. An include-only
page may obtain its title from the included document.

Preserve published page paths when changing navigation ownership. If a page
must move, update incoming links and provide a compatibility page at its old
path. Sphinx's HTML builder emits `page.html` and `section/index.html`;
README links to the published site must use those paths, not invented
extensionless directory URLs. Within MyST, use relative document links.

For navigation changes, build with `./scripts/build_doc.sh`, require no Sphinx
warnings, and inspect the generated sidebar and page outlines on a section
index and a nested page. Check README documentation links against generated
HTML files. Ensure no authored pages are orphaned or registered under multiple
parents. Exercise the performance viewer when its markup or assets change.
