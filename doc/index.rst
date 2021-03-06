WelDX - data and quality standards for welding research data
============================================================

Introduction
############
Scientific welding data covers a wide range of physical domains and timescales and are measured using various different sensors.
Complex and highly specialized experimental setups at different welding institutes complicate the exchange of welding research data further.

The WelDX research project aims to foster the exchange of scientific data inside the welding community by developing and establishing a new open source file format suitable for documentation of experimental welding data and upholding associated quality standards.
In addition to fostering scientific collaboration inside the national and international welding community an associated advisory committee will be established to oversee the future development of the file format.
The proposed file format will be developed with regards to current needs of the community regarding interoperability, data quality and performance and will be published under an appropriate open source license.
By using the file format objectivity, comparability and reproducibility across different experimental setups can be improved.

The project is under active development by the `Welding Technology <https://www.bam.de/Navigation/EN/About-us/Organisation/Organisation-Chart/President/Department-9/Division-93/division93.html>`_ division at Bundesanstalt für Materialforschung und -prüfung (BAM).

Python API
##########
The first core component of the ``WelDX`` project is the Python API.
The API aims to provide a framework for describing welding experiments as well as working and analysing welding research data in Python.

Head over to the :doc:`tutorials` section to see some examples.


WelDX file standard
###################
The second main component is the the ``WelDX`` file standard that is used to define the contents and layouts of welding research data. The file standard is based on the `ASDF standard <https://asdf-standard.readthedocs.io>`_ and consists of custom schema definitions for welding related experiments, measurements, equipment and applications.

Installation
############
The WelDX package can be installed using conda or mamba package manager from the :code:`conda-forge` channel.
These managers originate from the freely available `Anaconda Python stack <https://docs.conda.io/en/latest/miniconda.html>`_.
If you do not have Anaconda or Miniconda installed yet, we ask you to install ``Miniconda-3``.
Documentation for the installation procedure can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_. After this step you have access to the conda command and can proceed to installing the WeldX package.

::

    conda install weldx -c conda-forge


The package is also available on pypi.

::

    pip install weldx


Funding
#######
This research is funded by the Federal Ministry of Education and Research of Germany under project number 16QK12.


.. toctree::
    :hidden:
    :maxdepth: 1

    tutorials
    standard
    api
    CHANGELOG
    legal-notice

