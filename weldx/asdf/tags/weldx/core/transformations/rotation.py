import numpy as np
from scipy.spatial.transform import Rotation

from weldx.asdf.types import WeldxType
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.transformations import WXRotation


class WXRotationTypeASDF(WeldxType):
    """Serialization class for the 'Scipy.Rotation' type"""

    name = "core/transformations/rotation"
    version = "1.0.0"
    types = [Rotation]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

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

        if not hasattr(node, "_wx_meta"):  # default to quaternion representation
            tree["quaternions"] = node.as_quat()

        elif node._wx_meta["constructor"] == "from_euler":
            seq_str = node._wx_meta["seq"]
            if not len(seq_str) == 3:
                seq_str = seq_str + "".join([c for c in "xyz" if c not in seq_str])

            angles = node.as_euler(seq_str, degrees=node._wx_meta["degrees"])
            angles = np.squeeze(angles[..., : len(node._wx_meta["seq"])])

            if node._wx_meta["degrees"]:
                angles = Q_(angles, "degree")
            else:
                angles = Q_(angles, "rad")

            tree["sequence"] = node._wx_meta["seq"]
            tree["angles"] = angles

        else:
            raise NotImplementedError(f"unknown constructor")

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
            return Rotation.from_quat(tree["quaternions"])

        elif "angles" in tree:
            if "degree" in str(tree["angles"].units):
                angles = tree["angles"].to("degree").magnitude
                degrees = True
            else:
                angles = tree["angles"].to("rad").magnitude
                degrees = False
            return WXRotation.from_euler(
                seq=tree["sequence"], angles=angles, degrees=degrees
            )
