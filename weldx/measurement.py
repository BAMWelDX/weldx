"""Contains measurement related classes and functions."""

import xarray as xr
from dataclasses import dataclass
from typing import List, Union  # noqa: F401


# measurement --------------------------------------------------------------------------
@dataclass
class Data:
    """<TODO CLASS DOCSTRING>"""

    name: str
    data: xr.DataArray


@dataclass
class Error:
    """<TODO CLASS DOCSTRING>"""

    deviation: float


@dataclass
class Signal:
    """<TODO CLASS DOCSTRING>"""

    signal_type: str
    unit: str
    data: Union[Data, None]


@dataclass
class DataTransformation:
    """<TODO CLASS DOCSTRING>"""

    name: str
    input_signal: Signal
    output_signal: Signal
    error: Error


@dataclass
class Source:
    """<TODO CLASS DOCSTRING>"""

    name: str
    output_signal: Signal
    error: Error


@dataclass
class MeasurementChain:
    """<TODO CLASS DOCSTRING>"""

    name: str
    data_source: Source
    data_processors: List


@dataclass
class Measurement:
    """<TODO CLASS DOCSTRING>"""

    name: str
    data: Data
    measurement_chain: MeasurementChain


# equipment ----------------------------------------------------------------------------
@dataclass
class GenericEquipment:
    """<TODO CLASS DOCSTRING>"""

    name: str
    sources: List
    data_transformations: List


@dataclass
class Sensor:
    """<TODO CLASS DOCSTRING>"""

    data: str
