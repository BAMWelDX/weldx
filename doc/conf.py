# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import weldx

sys.path.insert(0, os.path.abspath(""))


# -- Project information -----------------------------------------------------

project = "weldx"
copyright = "2020, BAM"
author = "BAM"

# The full version, including alpha/beta/rc tags
release = weldx.__version__

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinxcontrib.napoleon",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_asdf",
]

# The suffix of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# nbsphinx
nbsphinx_execute = "always"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_kernel_name = "weldx"

# sphinx-asdf
# This variable indicates the top-level directory containing schemas.
# The path is relative to the location of conf.py in the package
asdf_schema_path = "../weldx/asdf/schemas"
# This variable indicates the standard prefix that is common to all schemas
# provided by the package.
asdf_schema_standard_prefix = "weldx.bam.de/weldx"

# enable references to the ASDF Standard documentation
asdf_schema_reference_mappings = [
    (
        "tag:stsci.edu:asdf",
        "http://asdf-standard.readthedocs.io/en/latest/generated/stsci.edu/asdf/",
    ),
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/WelDX_notext.svg"
html_favicon = "_static/WelDX_notext.ico"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {"logo_only": True}
