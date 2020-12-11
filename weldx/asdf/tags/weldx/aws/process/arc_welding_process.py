from dataclasses import dataclass

from weldx.asdf.types import WeldxType

__all__ = ["ArcWeldingProcess", "ArcWeldingProcessType"]

_name_to_abbr = {
    "atomicHydrogenWelding": "AHW",
    "bareMetalArcWelding": "BMAW",
    "carbonArcWelding": "CAW",
    "carbonArcWeldingGas": "CAW-G",
    "carbonArcWeldingShielded": "CAW-S",
    "electrogasWelding": "EGW",
    "electroSlagWelding": "ESW",
    "gasMetalArcWelding": "GMAW",
    "gasTungstenArcWelding": "GTAW",
    "plasmaArcWelding": "PAW",
    "shieldedMetalArcWelding": "SMAW",
    "studArcWelding": "SW",
    "submergedArcWelding": "SAW",
    "submergedArcWeldingSeries": "SAW-S",
}

_abbr_to_name = {
    "AHW": "atomicHydrogenWelding",
    "BMAW": "bareMetalArcWelding",
    "CAW": "carbonArcWelding",
    "CAW-G": "carbonArcWeldingGas",
    "CAW-S": "carbonArcWeldingShielded",
    "EGW": "electrogasWelding",
    "ESW": "electroSlagWelding",
    "GMAW": "gasMetalArcWelding",
    "GTAW": "gasTungstenArcWelding",
    "PAW": "plasmaArcWelding",
    "SMAW": "shieldedMetalArcWelding",
    "SW": "studArcWelding",
    "SAW": "submergedArcWelding",
    "SAW-S": "submergedArcWeldingSeries",
}


@dataclass
class ArcWeldingProcess:
    """<CLASS DOCSTRING>"""

    name: str
    abbreviation: str

    def __init__(self, name_or_abbreviation):
        if name_or_abbreviation in _name_to_abbr:
            self.name = name_or_abbreviation
            self.abbreviation = _name_to_abbr[name_or_abbreviation]
        elif name_or_abbreviation in _abbr_to_name:
            self.name = _abbr_to_name[name_or_abbreviation]
            self.abbreviation = name_or_abbreviation
        else:
            raise ValueError(
                f"Could not find process matching description '{name_or_abbreviation}'"
            )


class ArcWeldingProcessType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/process/arc_welding_process"
    version = "1.0.0"
    types = [ArcWeldingProcess]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = ArcWeldingProcess(tree["name"])
        return obj
