"""Sphinx configuration for the fastcashflow documentation."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import fastcashflow

project = "fastcashflow"
author = "Seokhoon Joo"
copyright = "2026, Seokhoon Joo"
release = fastcashflow.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "fastcashflow"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "show_prev_next": False,
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/seokhoonj/fastcashflow",
            "icon": "fa-brands fa-github",
        },
    ],
}
