"""Welding process classes."""

from dataclasses import dataclass
from typing import Dict

from weldx.core import TimeSeries


@dataclass
class GmawProcess:
    """Container class for all GMAW processes."""

    base_process: str
    manufacturer: str
    power_source: str
    parameters: Dict[str, TimeSeries]
    tag: str = None
    meta: dict = None

    def __post_init__(self):
        """Set defaults and convert parameter inputs."""
        if self.tag is None:
            self.tag = "GMAW"

        self.parameters = {
            k: (v if isinstance(v, TimeSeries) else TimeSeries(v))
            for k, v in self.parameters.items()
        }
