# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-
"""Python install setup."""

from setuptools import setup
from setuptools_scm import get_version

setup(
    version=get_version(),
    use_scm_version={
        "write_to": "weldx/_version.py",
        "write_to_template": '__version__ = "{version}"\n',
    },
)
