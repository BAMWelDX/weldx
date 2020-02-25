from dataclasses import dataclass
from typing import List
from asdf import yamlutil
from weldx.asdf.types import WeldxType


@dataclass
class BaseMetal:
    """<CLASS DOCSTRING>"""

    common_name: str
    m_number: str
    group_number: str
    product_form: str
    thickness: float
    diameter: float
    specification_number: str
    specification_version: str
    specification_organization: str
    UNS_number: str
    CAS_number: str
    heat_lot_identification: str
    composition: str
    manufacturing_history: str
    service_history: str
    applied_coating_specification: str


class BaseMetalType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "aws/design/base_metal"
    version = "1.0.0"
    types = [BaseMetal]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        tree = dict(
            common_name=yamlutil.custom_tree_to_tagged_tree(node.common_name, ctx),
            m_number=yamlutil.custom_tree_to_tagged_tree(node.m_number, ctx),
            group_number=yamlutil.custom_tree_to_tagged_tree(node.group_number, ctx),
            product_form=yamlutil.custom_tree_to_tagged_tree(node.product_form, ctx),
            thickness=yamlutil.custom_tree_to_tagged_tree(node.thickness, ctx),
            diameter=yamlutil.custom_tree_to_tagged_tree(node.diameter, ctx),
            specification_number=yamlutil.custom_tree_to_tagged_tree(
                node.specification_number, ctx
            ),
            specification_version=yamlutil.custom_tree_to_tagged_tree(
                node.specification_version, ctx
            ),
            specification_organization=yamlutil.custom_tree_to_tagged_tree(
                node.specification_organization, ctx
            ),
            UNS_number=yamlutil.custom_tree_to_tagged_tree(node.UNS_number, ctx),
            CAS_number=yamlutil.custom_tree_to_tagged_tree(node.CAS_number, ctx),
            heat_lot_identification=yamlutil.custom_tree_to_tagged_tree(
                node.heat_lot_identification, ctx
            ),
            composition=yamlutil.custom_tree_to_tagged_tree(node.composition, ctx),
            manufacturing_history=yamlutil.custom_tree_to_tagged_tree(
                node.manufacturing_history, ctx
            ),
            service_history=yamlutil.custom_tree_to_tagged_tree(
                node.service_history, ctx
            ),
            applied_coating_specification=yamlutil.custom_tree_to_tagged_tree(
                node.applied_coating_specification, ctx
            ),
        )
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = BaseMetal(**tree)
        return obj
