# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'BoolForge'
copyright = '2025, Claus Kadelka, Benjamin Coberly'
author = 'Claus Kadelka, Benjamin Coberly'

release = open("../boolforge/_version.py", "rt").read().split('\'')[1]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinxcontrib.collections"
]

collections = {
    "tutorials": {
        "driver": "copy_folder",
        "source": "../tutorials",
        "target": "tutorials/",
        "ignore": ["*.py", "*.sh"],
    }
}

autodoc_member_order = 'bysource'

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
