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
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# sphinx-copybutton -- strip Python REPL and shell prompts on copy.
# `# ` is *not* a prompt here because it doubles as the Python comment
# marker -- including it would also strip section-header lines like
# `# Detail` from copied code.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

templates_path = ["_templates"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["colon_fence", "deflist", "substitution"]

myst_substitutions = {
    "fcf": (
        '<span class="logo__fast">fast</span>'
        '<span class="logo__cashflow">cashflow</span>'
    ),
}

# The site is Korean. The API reference is built as is -- its autodoc bodies
# are English, which is fine for a reference.
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store",
    # Cookbook shared fragments -- pulled into other chapters via the MyST
    # {include} directive, not built as standalone pages.
    "cookbook/_shared/**",
]

html_theme = "pydata_sphinx_theme"
html_title = "fastcashflow"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["ga-consent.js", "ga-events.js"]
html_show_sourcelink = False
html_copy_source = False

# Keep diagrams compact and visually consistent across tutorial and cookbook
# pages. Individual diagrams define semantic classes while this shared theme
# handles typography and spacing.
mermaid_init_config = {
    "startOnLoad": False,
    "securityLevel": "loose",
    "flowchart": {
        "curve": "basis",
        "nodeSpacing": 24,
        "rankSpacing": 34,
        "padding": 8,
        "htmlLabels": True,
    },
    "state": {"useMaxWidth": True},
    "themeVariables": {
        "fontSize": "14px",
        "primaryColor": "#f4f7fa",
        "primaryTextColor": "#24313a",
        "primaryBorderColor": "#9aa9b5",
        "lineColor": "#788a97",
        "secondaryColor": "#e8f4f1",
        "tertiaryColor": "#eef3f8",
        "edgeLabelBackground": "#ffffff",
        "clusterBkg": "#f8fafb",
        "clusterBorder": "#d5dee5",
    },
}

html_theme_options = {
    "show_prev_next": True,
    "navbar_align": "content",
    "external_links": [
        {"name": "데모", "url": "https://demo.fastcashflow.org"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/seokhoonj/fastcashflow",
            "icon": "fa-brands fa-github",
        },
    ],
    "analytics": {
        "google_analytics_id": "G-1W6DP2378L",
    },
}
