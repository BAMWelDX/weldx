"""Contains measurement related classes and functions."""

from dataclasses import dataclass
from typing import List  # noqa: F401


# measurement --------------------------------------------------------------------------


@dataclass
class Data:
    """<TODO CLASS DOCSTRING>"""

    data: str


@dataclass
class DataProcessor:
    """<TODO CLASS DOCSTRING>"""

    data: str


@dataclass
class Error:
    """<TODO CLASS DOCSTRING>"""

    data: str


# equipment ----------------------------------------------------------------------------
@dataclass
class GenericEquipment:
    """<TODO CLASS DOCSTRING>"""

    data: str


@dataclass
class Sensor:
    """<TODO CLASS DOCSTRING>"""

    data: str
