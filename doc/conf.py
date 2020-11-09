# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Local build command ------------------------------------------------------------------

# sphinx-build -W -n -b html -d build/doctrees doc build/html --keep-going
# -D nbsphinx_kernel_name="weldx" -D nbsphinx_execute="never"

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pathlib
import shutil
import sys

import traitlets

import weldx
from weldx.asdf.constants import WELDX_TAG_BASE

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
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_asdf",
    "numpydoc",
    "sphinx_autodoc_typehints",  # list after napoleon
]

# autosummary --------------------------------------------------------------------------
autosummary_generate = True

# add __init__ docstrings to class documentation
autoclass_content = "both"

# numpydoc option documentation:
# https://numpydoc.readthedocs.io/en/latest/install.html
numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = True
# numpydoc_citation_re = str - check documentation
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
# numpydoc_xref_aliases = dict - check documentation
# numpydoc_xref_ignore = set - check documentation


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# sphinx-autodoc-typehints https://github.com/agronholm/sphinx-autodoc-typehints
set_type_checking_flag = False
typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = True

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

# -- nbsphinx configuration -------------------------------------------------
nbsphinx_execute = "always"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

if traitlets.__version__ < "5":
    nbsphinx_execute_arguments.append("--InlineBackend.rc={'figure.dpi': 96}")
else:
    nbsphinx_execute_arguments.append("--InlineBackend.rc <figure.dpi=96>")

# Select notebook kernel for nbsphinx
# default "python3" is needed for readthedocs run
# if building locally, this might need to be "weldx" - try setting using -D option:
# -D nbsphinx_kernel_name="weldx"
nbsphinx_kernel_name = "python3"

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None).replace("\\","/") %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::
    
        Run the interactive online version of this notebook (takes 1-2 minutes to load): 
        :raw-html:`<a href="https://mybinder.org/v2/gh/BAMWelDX/weldx/master?urlpath=lab/tree/{{ docname }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`

"""

nbsphinx_epilog = """
----

Generated by nbsphinx_ from a Jupyter_ notebook.

.. _nbsphinx: https://nbsphinx.readthedocs.io/
.. _Jupyter: https://jupyter.org/
"""

# -- sphinx-asdf configuration -------------------------------------------------
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
    (
        WELDX_TAG_BASE,
        "http://weldx.readthedocs.io/en/latest/generated/weldx.bam.de/weldx/",
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
    "external_links": [{"url": "https://asdf.readthedocs.io/", "name": "ASDF"}],
    "github_url": "https://github.com/BAMWelDX/weldx",
    "twitter_url": "https://twitter.com/BAMweldx",
    "use_edit_page_button": False,
    "show_prev_next": False,
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

# Intersphinx mappings -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "xarray": ("http://xarray.pydata.org/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org", None),
    # "dask": ("https://docs.dask.org/en/latest", None),
    # "numba": ("https://numba.pydata.org/numba-doc/latest", None),
    "pint": ("https://pint.readthedocs.io/en/stable", None),
    "jsonschema": ("https://python-jsonschema.readthedocs.io/en/stable/", None),
    "asdf": ("https://asdf.readthedocs.io/en/stable/", None),
}

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

# Enable better object linkage ---------------------------------------------------------

# This option basically turns every Markdown like inline code block into a sphinx
# reference
default_role = "py:obj"

# see:
# https://stackoverflow.com/questions/34052582/how-do-i-refer-to-classes-and-methods-in-other-files-my-project-with-sphinx
