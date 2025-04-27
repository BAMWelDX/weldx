# WelDX - data and quality standards for welding research data

<hl/>

[![CF](https://anaconda.org/conda-forge/weldx/badges/version.svg)](https://anaconda.org/conda-forge/weldx)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5e7ede6d978249a781e5c580ed1c813f)](https://www.codacy.com/gh/BAMWelDX/weldx)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/BAMWelDX/weldx/?ref=repository-badge)
[![Documentation](https://readthedocs.org/projects/weldx/badge/?version=latest)](https://weldx.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.5565185.svg)](https://doi.org/10.5281/zenodo.5565185)
[![codecov](https://codecov.io/gh/BAMWelDX/weldx/branch/master/graph/badge.svg)](https://codecov.io/gh/BAMWelDX/weldx)
[![package builds](https://github.com/BAMWelDX/weldx/actions/workflows/build_pkg.yml/badge.svg)](https://github.com/BAMWelDX/weldx/actions/workflows/build_pkg.yml)
[![documentation builds](https://github.com/BAMWelDX/weldx/actions/workflows/docs.yml/badge.svg)](https://github.com/BAMWelDX/weldx/actions/workflows/docs.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/BAMWelDX/weldx/master.svg)](https://results.pre-commit.ci/latest/github/BAMWelDX/weldx/master)
[![pytest](https://github.com/BAMWelDX/weldx/actions/workflows/pytest.yml/badge.svg)](https://github.com/BAMWelDX/weldx/actions/workflows/pytest.yml)
[![static analysis](https://github.com/BAMWelDX/weldx/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/BAMWelDX/weldx/actions/workflows/static_analysis.yml)

## Overview

Scientific welding data covers a wide range of physical domains and
timescales and are measured using various different sensors. Complex and
highly specialized experimental setups at different welding institutes
complicate the exchange of welding research data further.

The WelDX research project aims to foster the exchange of scientific
data inside the welding community by developing and establishing a new
open source file format suitable for the documentation of experimental
welding data and upholding associated quality standards. In addition to
fostering scientific collaboration inside the national and international
welding community an associated advisory committee will be established
to oversee the future development of the file format. The proposed file
format will be developed with regard to current needs of the community
regarding interoperability, data quality and performance and will be
published under an appropriate open source license. By using the file
format objectivity, comparability and reproducibility across different
experimental setups can be improved.

The project is under active development by the [Welding Technology](https://www.bam.de/Navigation/EN/About-us/Organisation/Organisation-Chart/President/Department-9/Division-93/division93.html)
division at Bundesanstalt für Materialforschung und -prüfung (BAM).

## Features

WelDX provides several Python API to perform standard tasks like
experiment design, data analysis, and experimental data archiving.

### Planning

- Define measurement chains with all involved devices, error sources,
  and metadata annotations.
- Handle complex coordinate transformations needed to describe the
  movement of welding robots, workpieces, and sensors.
- Planing of welding experiments.
- convenient creation of [ISO 9692-1](https://www.iso.org/standard/62520.html) welding groove types.

### Data analysis

- Plotting routines to inspect measurement chains, workpieces (planned
  and welded).
- Analysis functions for standard measurements like track energy,
  welding speed to fill an ISO groove, and more to come.

### Data archiving

The ultimate goal of this project is to store all information about the
experiment in a single file. We choose the popular [ASDF](https://en.wikipedia.org/wiki/Advanced_Scientific_Data_Format)
format for this task. This enables us to store arbitrary binary data,
while maintaining a human readable text based header. All information is
stored in a tree like structure, which makes it convenient to structure
the data in arbitrary complex ways.

The ASDF format and the provided extensions for WelDX types like

- workpiece information (used alloys, geometries)
- welding process parameters (GMAW parameters)
- measurement data
- coordinate systems (robot movement, sensors)

enables us to store the whole experimental pipeline performed in a
modern laboratory.

## Design goals

We seek to provide a user-friendly, well documented programming
interface. All functions and classes in WelDX have attached
documentation about the involved parameters (types and explanation), see
[API docs](https://weldx.readthedocs.io/en/stable/api.html). Further
we provide rich [Jupyter notebook tutorials](https://weldx.readthedocs.io/en/stable/tutorials.html) about the
handling of the basic workflows.

All involved physical quantities used in `weldx` (lengths, angles,
voltages, currents, etc.) should be attached with a unit to ensure
automatic conversion and correct mathematical handling. Units are being
used in all standard features of WelDX and are also archived in the ASDF
files. This is implemented by the popular Python library [Pint](https://pint.readthedocs.io/en/stable/), which flawlessly handles
the creation and conversion of units and dimensions.

## Publications

- Recommendations for an Open Science approach to welding process
  research data. Fabry, C., Pittner, A., Hirthammer, V. et al. *Weld
  World* (2021). <https://doi.org/10.1007/s40194-021-01151-x>

## Installation

The WelDX package can be installed using any conda or mamba package manager from the [Conda-Forge channel](https://conda-forge.org/#about).
If you have not yet installed a conda package manager, we recommend installing `Miniforge`.
The installer can then be found [here](https://conda-forge.org/download/), and a detailed documentation for the installation process is provided
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).
Once this step has been completed, you will gain access to both the `conda` and the `mamba` command and will be able to proceed with the installation of the WelDX package.

In order to create a new conda environment called `weldx` containing the WeldX package,
run the console command:

```shell
conda create --name weldx --channel conda-forge weldx weldx_widgets
```

To install the WeldX package into your existing environment instead, use:

```shell
conda install weldx weldx_widgets --channel conda-forge
```

If installed, all `conda` commands can be replaced by `mamba` to take advantage
of its faster solver.

The package is also available on pypi and can be installed via:

```shell
pip install weldx weldx-widgets
```

As weldx currently depends on the package `bottleneck`, which contains
C/C++ code, you will need a working C/C++ compiler. The conda package
does not have this requirement as it only installs pre-compiled
binaries. So if you do not know how to install a working compiler, we
strongly encourage using the conda package.

## Documentation

The full documentation is published on readthedocs.org. Click on one of
the following links to get to the desired version:

- [latest](https://weldx.readthedocs.io/en/latest/)
- [stable](https://weldx.readthedocs.io/en/stable/)

## Funding

This research is funded by the Federal Ministry of Education and
Research of Germany under project number 16QK12.
