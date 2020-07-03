# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-
"""Python Library description and install setup."""

from os import path

import versioneer
from setuptools import find_packages, setup

# add README.md as long description
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "numpy>=0.18",
    "pandas>=1.0",
    "xarray>=0.15",
    "scipy>=1.4",
    "pint>=0.11",
    "asdf>=2.6",
    "bottleneck>=1.3",
    "Jinja2>=2.11",
    "networkx>=2",
    "matplotlib>=3",
]

entry_points = {}
entry_points["asdf_extensions"] = [
    "weldx = weldx.asdf.extension:WeldxExtension",
    "weldx-asdf = weldx.asdf.extension:WeldxAsdfExtension",
]

setup(
    name="weldx",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Cagtay Fabry",
    author_email="Cagtay.Fabry@bam.de",
    packages=find_packages(),
    url="https://www.bam.de/weldx",
    license="BSD License",
    description="Python API for the WelDX file format and standard",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Natural Language :: English",
    ],
    install_requires=requirements,
    include_package_data=True,  # include non-py files listed in MANIFEST.in
    entry_points=entry_points,  # register ASDF Extension entry_points
)
