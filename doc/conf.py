import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

project = "ImmersX: embedded & mixed-dimensional simulation framework"
author = "Luca Heltai and contributors"
html_baseurl = "https://luca-heltai.github.io/immersx/"
default_role = "any"

extensions = [
    "sphinx.ext.mathjax",
    "breathe",
    "exhale",
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "html"]

html_static_path = ["_static"]
html_theme = "furo"
html_title = project
html_logo = "immersx-logo.png"
html_favicon = "favicon.png"
html_theme_options = {
    "source_repository": "https://github.com/luca-heltai/immersx/",
    "source_branch": "master",
    "source_directory": "doc/",
    "top_of_page_buttons": ["view", "edit"],
}

sys.path.insert(0, str(_REPO_ROOT))

breathe_projects = {
    project: os.path.abspath("../build/docs/doxygen/xml"),
}
breathe_default_project = project
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library Reference",
    "contentsDirectives": False,
    "doxygenStripFromPath": "..",
    "createTreeView": False,
    "exhaleExecutesDoxygen": False,
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]

mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
)


def render_changelog(app, docname, source):
    if docname == "changes":
        from scripts.changelog import render
        source[0] = render(_REPO_ROOT)


def refresh_changelog(app, env, added, changed, removed):
    # New fragments are not dependencies of the previous build yet.
    return ["changes"]


def setup(app):
    app.connect("source-read", render_changelog)
    app.connect("env-get-outdated", refresh_changelog)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
