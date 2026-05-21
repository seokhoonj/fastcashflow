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
language = "ko"

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

# The site is Korean. The API reference is built as is -- its autodoc bodies
# are English, which is fine for a reference. The English starter pages
# (getting-started.md, concepts.md) are kept on disk but left unbuilt, for a
# future separate English edition.
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store",
    "getting-started.md", "concepts.md",
]

html_theme = "pydata_sphinx_theme"
html_title = "fastcashflow"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_copy_source = False

html_theme_options = {
    "show_prev_next": True,
    "navbar_align": "content",
    "logo": {
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
        "alt_text": "fastcashflow",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/seokhoonj/fastcashflow",
            "icon": "fa-brands fa-github",
        },
    ],
}
