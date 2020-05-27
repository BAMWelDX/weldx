# WelDX - Welding Data Exchange Format
## Overview
Scientific welding data covers a wide range of physical domains and timescales and are measured using various different sensors.
Complex and highly specialized experimental setups at different welding institutes complicate the exchange of welding research data further.

The WelDX research project aims to foster the exchange of scientific data inside the welding community by developing and establishing a new open source file format suitable for documentation of experimental welding data and upholding associated quality standards.
In addition to fostering scientific collaboration inside the national and international welding community an associated advisory committee will be established to oversee the future development of the file format.
The proposed file format will be developed with regards to current needs of the community regarding interoperability, data quality and performance and will be published under an appropriate open source license.
By using the file format objectivity, comparability and reproducibility across different experimental setups can be improved.

## Installation
The WelDX package can be installed using conda from the `bamwelding` channel (with some required packages available on the `conda-forge` channel).
```console
conda install weldx -c conda-forge -c bamwelding
```

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
