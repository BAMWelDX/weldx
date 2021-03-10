# WelDX - Welding Data Exchange Format

[![Documentation](https://readthedocs.org/projects/weldx/badge/?version=latest)](https://weldx.readthedocs.io/en/latest/?badge=latest) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BAMWelDX/weldx/master?urlpath=lab/tree/tutorials/welding_example_01_basics.ipynb)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/weldx/badges/version.svg)](https://anaconda.org/conda-forge/weldx)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

Scientific welding data covers a wide range of physical domains and timescales and are measured using various different
sensors. Complex and highly specialized experimental setups at different welding institutes complicate the exchange of
welding research data further.

The WelDX research project aims to foster the exchange of scientific data inside the welding community by developing and
establishing a new open source file format suitable for the documentation of experimental welding data and upholding
associated quality standards. In addition to fostering scientific collaboration inside the national and international
welding community an associated advisory committee will be established to oversee the future development of the file
format. The proposed file format will be developed with regard to current needs of the community regarding
interoperability, data quality and performance and will be published under an appropriate open source license. By using
the file format objectivity, comparability and reproducibility across different experimental setups can be improved.

The project is under active development by
the [Welding Technology](https://www.bam.de/Navigation/EN/About-us/Organisation/Organisation-Chart/President/Department-9/Division-93/division93.html)
division at Bundesanstalt für Materialforschung und -prüfung (BAM).

## Installation

The WelDX package can be installed using conda or mamba package manager from the :code:`conda-forge` channel.
These managers originate from the freely available [Anaconda Python stack](https://docs.conda.io/en/latest/miniconda.html>).
If you do not have Anaconda or Miniconda installed yet, we ask you to install ``Miniconda-3``.
Documentation for the installation procedure can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).
After this step you have access to the conda command and can proceed to installing the WeldX package.
```console
conda install weldx -c conda-forge
```

The package is also available on pypi.
```console
pip install weldx
```

## Documentation

The full documentation is published on readthedocs.org. Click on one of the following links to get to the desired
version:

-   [latest](https://weldx.readthedocs.io/en/latest/)
-   [stable](https://weldx.readthedocs.io/en/stable/)

## Funding

This research is funded by the Federal Ministry of Education and Research of Germany under project number 16QK12.

## Repository status

### Continuous Integration

[![pytest](https://github.com/BAMWelDX/weldx/workflows/pytest/badge.svg?branch=master)](https://github.com/BAMWelDX/weldx/actions?query=workflow%3Apytest+branch%3Amaster)
[![conda build](https://github.com/BAMWelDX/weldx/workflows/conda%20build/badge.svg?branch=master)](https://github.com/BAMWelDX/weldx/actions?query=workflow%3A%22conda+build%22+branch%3Amaster)
[![](https://travis-ci.com/BAMWelDX/weldx.svg?branch=master)](https://travis-ci.com/BAMWelDX/weldx)
[![Build status](https://ci.appveyor.com/api/projects/status/6yvswkpj7mmdbrk1/branch/master?svg=true)](https://ci.appveyor.com/project/BAMWelDX/weldx/branch/master)

### Code Status

[![static analysis](https://github.com/BAMWelDX/weldx/workflows/static%20analysis/badge.svg?branch=master)](https://github.com/BAMWelDX/weldx/actions?query=workflow%3A%22static+analysis%22+branch%3Amaster)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5e7ede6d978249a781e5c580ed1c813f)](https://www.codacy.com/gh/BAMWelDX/weldx?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BAMWelDX/weldx&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/BAMWelDX/weldx/branch/master/graph/badge.svg)](https://codecov.io/gh/BAMWelDX/weldx)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/BAMWelDX/weldx/?ref=repository-badge)

### Documentation build

[![Documentation Status](https://readthedocs.org/projects/weldx/badge/?version=latest)](https://weldx.readthedocs.io/en/latest/?badge=latest)
[![documentation](https://github.com/BAMWelDX/weldx/workflows/documentation/badge.svg?branch=master)](https://github.com/BAMWelDX/weldx/actions?query=workflow%3Adocumentation+branch%3Amaster)
