import os
import sys

project = "Reduced Lagrange Multipliers"
author = "Luca Heltai and contributors"
html_baseurl = "https://luca-heltai.github.io/reduced_lagrange_multipliers/"
default_role = "any"

extensions = [
    "sphinx.ext.mathjax",
    "breathe",
    "exhale",
    "myst_parser",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "html"]

html_theme = "furo"
html_title = project
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/luca-heltai/reduced_lagrange_multipliers/",
    "source_branch": "master",
    "source_directory": "doc/",
    "top_of_page_buttons": ["view", "edit"],
}

sys.path.insert(0, os.path.abspath(".."))

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
