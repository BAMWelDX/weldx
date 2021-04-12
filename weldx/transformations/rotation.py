"""Contains tools to handle rotations."""

from typing import List, Union

import numpy as np
import pint
from scipy.spatial.transform import Rotation as _Rotation

from weldx.constants import WELDX_UNIT_REGISTRY as UREG

_DEFAULT_LEN_UNIT = UREG.millimeters
_DEFAULT_ANG_UNIT = UREG.rad


class WXRotation(_Rotation):
    """Wrapper for creating meta-tagged `~scipy.spatial.transform.Rotation` objects.

    See `scipy.spatial.transform.Rotation` docs for details.
    """

    @classmethod
    def from_quat(cls, quat: np.ndarray) -> "WXRotation":  # noqa
        """Initialize from quaternions.

        See `scipy.spatial.transform.Rotation.from_quat` docs for details.
        """
        rot = super().from_quat(quat)
        setattr(rot, "wx_meta", {"constructor": "from_quat"})
        return rot

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "WXRotation":  # noqa
        """Initialize from matrix.

        See `scipy.spatial.transform.Rotation.from_matrix` docs for details.
        """
        rot = super().from_matrix(matrix)
        setattr(rot, "wx_meta", {"constructor": "from_matrix"})
        return rot

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> "WXRotation":  # noqa
        """Initialize from rotation vector.

        See `scipy.spatial.transform.Rotation.from_rotvec` docs for details.
        """
        rot = super().from_rotvec(rotvec)
        setattr(rot, "wx_meta", {"constructor": "from_rotvec"})
        return rot

    @classmethod
    @UREG.check(None, None, "[]", None)
    def from_euler(
        cls,
        seq: str,
        angles: Union[pint.Quantity, np.ndarray, List[float], List[List[float]]],
        degrees: bool = False,
    ) -> "WXRotation":  # noqa
        """Initialize from euler angles.

        See `scipy.spatial.transform.Rotation.from_euler` docs for details.
        """

        if isinstance(angles, pint.Quantity):
            if str(angles.u) == "dimensionless":
                angles = angles.to("rad")
            degrees = "rad" not in str(angles.u)
            if degrees:
                angles = angles.to("degree")
            else:
                angles = angles.to("rad")
            angles = angles.m

        rot = super().from_euler(seq=seq, angles=angles, degrees=degrees)
        setattr(
            rot,
            "wx_meta",
            {"constructor": "from_euler", "seq": seq, "degrees": degrees},
        )
        return rot
