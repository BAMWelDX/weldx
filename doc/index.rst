WelDX - data and quality standards for welding research data
============================================================

Overview
########
Scientific welding data covers a wide range of physical domains and timescales and are measured using various different sensors.
Complex and highly specialized experimental setups at different welding institutes complicate the exchange of welding research data further.

The WelDX research project aims to foster the exchange of scientific data inside the welding community by developing and establishing a new open source file format suitable for documentation of experimental welding data and upholding associated quality standards.
In addition to fostering scientific collaboration inside the national and international welding community an associated advisory committee will be established to oversee the future development of the file format.
The proposed file format will be developed with regards to current needs of the community regarding interoperability, data quality and performance and will be published under an appropriate open source license.
By using the file format objectivity, comparability and reproducibility across different experimental setups can be improved.

The project is under active development by the `Welding Technology <https://www.bam.de/Navigation/EN/About-us/Organisation/Organisation-Chart/President/Department-9/Division-93/division93.html>`_ division at Bundesanstalt für Materialforschung und -prüfung (BAM).

Installation
############
The WelDX package can be installed using conda from the :code:`bamwelding` channel (with some required packages available on the :code:`conda-forge` channel).

::

    conda install weldx -c conda-forge -c bamwelding


Funding
#######
This research is funded by the Federal Ministry of Education and Research of Germany under project number 16QK12.

Contents
########

.. toctree::
    :maxdepth: 1

    tutorials
    asdf-extension
    devdoc
    api
    asdf-schemas
    CHANGELOG
    legal-notice

