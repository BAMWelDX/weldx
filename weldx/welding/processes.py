"""Welding process classes."""

from __future__ import annotations

from dataclasses import dataclass

from weldx.core import TimeSeries


@dataclass
class GmawProcess:
    """Container class for all GMAW processes."""

    base_process: str
    """Base process used, e.g. spray or pulse """
    manufacturer: str
    """Manufacturer of the power source."""
    power_source: str
    """Power source model used in the process."""
    parameters: dict[str, TimeSeries]
    """Process parameters, like U, I or pulsed versions of it."""
    tag: str = None
    """optional tag."""
    meta: dict = None
    """meta can contain custom descriptions of the process."""

    def __post_init__(self):
        """Set defaults and convert parameter inputs."""
        if self.tag is None:
            self.tag = "GMAW"

        self.parameters = {
            k: (v if isinstance(v, TimeSeries) else TimeSeries(v))
            for k, v in self.parameters.items()
        }
