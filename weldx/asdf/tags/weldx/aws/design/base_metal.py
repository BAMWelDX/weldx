from dataclasses import dataclass

import pint

from weldx.asdf.types import WeldxType

__all__ = ["BaseMetal", "BaseMetalType"]


@dataclass
class BaseMetal:
    """<CLASS DOCSTRING>"""

    common_name: str
    product_form: str
    thickness: pint.Quantity
    m_number: str = None
    group_number: str = None
    diameter: float = None
    specification_number: str = None
    specification_version: str = None
    specification_organization: str = None
    UNS_number: str = None
    CAS_number: str = None
    heat_lot_identification: str = None
    composition: str = None
    manufacturing_history: str = None
    service_history: str = None
    applied_coating_specification: str = None


class BaseMetalType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/base_metal"
    version = "1.0.0"
    types = [BaseMetal]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: BaseMetal, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = node.__dict__
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = BaseMetal(**tree)
        return obj
