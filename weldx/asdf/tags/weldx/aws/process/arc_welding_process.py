from dataclasses import dataclass

from weldx.asdf.util import dataclass_serialization_class

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


def _from_tree(tree):
    return dict(name_or_abbreviation=tree["name"])


ArcWeldingProcessType = dataclass_serialization_class(
    class_type=ArcWeldingProcess,
    class_name="aws/process/arc_welding_process",
    version="1.0.0",
    from_tree_mod=_from_tree,
)
