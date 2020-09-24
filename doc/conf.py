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
import pathlib
import re
import shutil
import sys

import numpydoc

import weldx

sys.path.insert(0, os.path.abspath(""))

# -- copy files to doc folder -------------------------------------------------
doc_dir = pathlib.Path(".")
changelog_file = pathlib.Path("./../CHANGELOG.md")
shutil.copy(changelog_file, doc_dir)

tutorials_dir = pathlib.Path("./tutorials")
tutorial_files = pathlib.Path("./../tutorials/").glob("*.ipynb")
for f in tutorial_files:
    shutil.copy(f, tutorials_dir)


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
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_asdf",
    "numpydoc",
]

# autosummary --------------------------------------------------------------------------
autosummary_generate = True
numpydoc_show_class_members = False

# version = .__version__
# The full version, including alpha/beta/rc tags.
release = numpydoc.__version__
version = re.sub(r"(\d+\.\d+)\.\d+(.*)", r"\1\2", numpydoc.__version__)
version = re.sub(r"(\.dev\d+).*?$", r"\1", version)
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "type_without_description", "BadException"}


# --------------------------------------------------------------------------------------

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

# Select notebook kernel for nbsphinx
# default "python3" is needed for readthedocs run
# if building locally, this might need to be "weldx" - try setting using -D option:
# -D nbsphinx_kernel_name="weldx"
nbsphinx_kernel_name = "python3"

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

html_theme_options = {
    "external_links": [{"url": "https://asdf.readthedocs.io/", "name": "ASDF Docs"}],
    "github_url": "https://github.com/BAMWelDX/weldx",
    "twitter_url": "https://twitter.com/BAMweldx",
    "use_edit_page_button": False,
}

html_context = {
    "github_user": "BAMWelDX",
    "github_repo": "weldx",
    "github_version": "master",
    "doc_path": "doc",
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {"logo_only": True}


# Disable warnings caused by a bug -----------------------------------------------------

# see this Stack Overflow answer for further information:
# https://stackoverflow.com/a/30624034/6700329

nitpick_ignore = []

for line in open("nitpick_ignore"):
    if line.strip() == "" or line.startswith("#"):
        continue
    dtype, target = line.split(None, 1)
    target = target.strip()
    nitpick_ignore.append((dtype, target))
