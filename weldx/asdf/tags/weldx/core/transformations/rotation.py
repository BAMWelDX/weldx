import numpy as np
from scipy.spatial.transform import Rotation

from weldx.asdf.types import WeldxType
from weldx.asdf.validators import wx_unit_validator
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.transformations.rotation import WXRotation


class WXRotationTypeASDF(WeldxType):
    """Serialization class for the 'Scipy.Rotation' type"""

    name = "core/transformations/rotation"
    version = "1.0.0"
    types = [Rotation]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
    validators = {"wx_unit": wx_unit_validator}

    @classmethod
    def to_tree(cls, node: Rotation, ctx):
        """
        Convert an instance of the 'Dimension' type into YAML representations.

        Parameters
        ----------
        node :
            Instance of the 'Dimension' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the 'Dimension' type to be
            serialized.

        """
        tree = {}

        if not hasattr(node, "wx_meta"):  # default to quaternion representation
            tree["quaternions"] = node.as_quat()
        elif node.wx_meta["constructor"] == "from_quat":
            tree["quaternions"] = node.as_quat()
        elif node.wx_meta["constructor"] == "from_matrix":
            tree["matrix"] = node.as_matrix()
        elif node.wx_meta["constructor"] == "from_rotvec":
            tree["rotvec"] = node.as_rotvec()
        elif node.wx_meta["constructor"] == "from_euler":
            seq_str = node.wx_meta["seq"]
            if not len(seq_str) == 3:
                if all([c in "xyz" for c in seq_str]):
                    seq_str = seq_str + "".join([c for c in "xyz" if c not in seq_str])
                elif all([c in "XYZ" for c in seq_str]):
                    seq_str = seq_str + "".join([c for c in "XYZ" if c not in seq_str])
                else:  # pragma: no cover
                    raise ValueError("Mix of intrinsic and extrinsic euler angles.")

            angles = node.as_euler(seq_str, degrees=node.wx_meta["degrees"])
            angles = np.squeeze(angles[..., : len(node.wx_meta["seq"])])

            if node.wx_meta["degrees"]:
                angles = Q_(angles, "degree")
            else:
                angles = Q_(angles, "rad")

            tree["sequence"] = node.wx_meta["seq"]
            tree["angles"] = angles

        else:  # pragma: no cover
            raise NotImplementedError("unknown or missing constructor")

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into custom types.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        Dimension :
            An instance of the 'Dimension' type.

        """
        if "quaternions" in tree:
            return WXRotation.from_quat(tree["quaternions"])
        elif "matrix" in tree:
            return WXRotation.from_matrix(tree["matrix"])
        elif "rotvec" in tree:
            return WXRotation.from_rotvec(tree["rotvec"])
        elif "angles" in tree:
            return WXRotation.from_euler(seq=tree["sequence"], angles=tree["angles"])
