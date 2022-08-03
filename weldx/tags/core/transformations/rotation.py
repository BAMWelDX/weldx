import numpy as np
from scipy.spatial.transform import Rotation

from weldx.asdf.types import WeldxConverter
from weldx.constants import Q_
from weldx.transformations.rotation import ROT_META, WXRotation


class WXRotationConverter(WeldxConverter):
    """Serialization class for the 'Scipy.Rotation' type"""

    tags = ["asdf://weldx.bam.de/weldx/tags/core/transformations/rotation-0.1.*"]
    types = [Rotation, WXRotation]

    def to_yaml_tree(self, obj: Rotation, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = {}

        if not hasattr(obj, ROT_META):  # default to quaternion representation
            tree["quaternions"] = obj.as_quat()
        elif getattr(obj, ROT_META)["constructor"] == "from_quat":
            tree["quaternions"] = obj.as_quat()
        elif getattr(obj, ROT_META)["constructor"] == "from_matrix":
            tree["matrix"] = obj.as_matrix()
        elif getattr(obj, ROT_META)["constructor"] == "from_rotvec":
            tree["rotvec"] = obj.as_rotvec()
        elif getattr(obj, ROT_META)["constructor"] == "from_euler":
            seq_str = getattr(obj, ROT_META)["seq"]
            if not len(seq_str) == 3:
                if all(c in "xyz" for c in seq_str):
                    seq_str = seq_str + "".join([c for c in "xyz" if c not in seq_str])
                elif all(c in "XYZ" for c in seq_str):
                    seq_str = seq_str + "".join([c for c in "XYZ" if c not in seq_str])
                else:  # pragma: no cover
                    raise ValueError("Mix of intrinsic and extrinsic euler angles.")

            angles = obj.as_euler(seq_str, degrees=getattr(obj, ROT_META)["degrees"])
            angles = np.squeeze(angles[..., : len(getattr(obj, ROT_META)["seq"])])

            if getattr(obj, ROT_META)["degrees"]:
                angles = Q_(angles, "degree")
            else:
                angles = Q_(angles, "rad")

            tree["sequence"] = getattr(obj, ROT_META)["seq"]
            tree["angles"] = angles

        else:  # pragma: no cover
            raise NotImplementedError("unknown or missing constructor")

        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        if "quaternions" in node:
            return WXRotation.from_quat(node["quaternions"])
        if "matrix" in node:
            return WXRotation.from_matrix(node["matrix"])
        if "rotvec" in node:
            return WXRotation.from_rotvec(node["rotvec"])
        if "angles" in node:
            return WXRotation.from_euler(seq=node["sequence"], angles=node["angles"])
