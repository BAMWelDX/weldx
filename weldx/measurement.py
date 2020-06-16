"""Contains measurement related classes and functions."""

from dataclasses import dataclass
from typing import List  # noqa: F401


# measurement --------------------------------------------------------------------------
@dataclass
class Error:
    """<TODO CLASS DOCSTRING>"""

    deviation: float


@dataclass
class Data:
    """<TODO CLASS DOCSTRING>"""

    data: str


@dataclass
class DataProcessor:
    """<TODO CLASS DOCSTRING>"""

    name: str
    input_unit: str
    output_unit: str
    error: Error


@dataclass
class Source:
    """<TODO CLASS DOCSTRING>"""

    name: str
    output_unit: str
    error: Error


@dataclass
class MeasurementChain:
    """<TODO CLASS DOCSTRING>"""

    source: Source
    data_processors: List


# equipment ----------------------------------------------------------------------------
@dataclass
class GenericEquipment:
    """<TODO CLASS DOCSTRING>"""

    data: str


@dataclass
class Sensor:
    """<TODO CLASS DOCSTRING>"""

    data: str
