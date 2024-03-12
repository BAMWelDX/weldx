"""Contains tools to handle rotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pint
from scipy.spatial.transform import Rotation as _Rotation

from weldx.constants import U_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

_DEFAULT_LEN_UNIT = UREG.millimeters
_DEFAULT_ANG_UNIT = UREG.rad

ROT_META = "wx_rot"


class WXRotation(_Rotation):
    """Wrapper for creating meta-tagged `~scipy.spatial.transform.Rotation` objects.

    See `scipy.spatial.transform.Rotation` docs for details.
    """

    @classmethod
    def from_quat(cls, quat: npt.ArrayLike) -> WXRotation:
        """Initialize from quaternions.

        See `scipy.spatial.transform.Rotation.from_quat` docs for details.
        """
        rot = super().from_quat(quat)
        setattr(rot, ROT_META, {"constructor": "from_quat"})
        return rot

    @classmethod
    def from_matrix(cls, matrix: npt.ArrayLike) -> WXRotation:
        """Initialize from matrix.

        See `scipy.spatial.transform.Rotation.from_matrix` docs for details.
        """
        rot = super().from_matrix(matrix)
        setattr(rot, ROT_META, {"constructor": "from_matrix"})
        return rot

    @classmethod
    def from_rotvec(cls, rotvec: npt.ArrayLike) -> WXRotation:
        """Initialize from rotation vector.

        See `scipy.spatial.transform.Rotation.from_rotvec` docs for details.
        """
        rot = super().from_rotvec(rotvec)
        setattr(rot, ROT_META, {"constructor": "from_rotvec"})
        return rot

    @classmethod
    @UREG.check(None, None, "[]", None)
    def from_euler(
        cls,
        seq: str,
        angles: pint.Quantity | npt.ArrayLike,
        degrees: bool = False,
    ) -> WXRotation:
        """Initialize from euler angles.

        See `scipy.spatial.transform.Rotation.from_euler` docs for details.
        """
        if isinstance(angles, pint.Quantity):
            if angles.u == U_(""):
                angles = angles.to("rad")  # type: ignore[assignment]
            degrees = "rad" not in str(angles.u)
            if degrees:
                angles = angles.to("degree")  # type: ignore[assignment]
            else:
                angles = angles.to("rad")  # type: ignore[assignment]
            angles = angles.m

        rot = super().from_euler(seq=seq, angles=angles, degrees=degrees)
        setattr(
            rot,
            ROT_META,
            {"constructor": "from_euler", "seq": seq, "degrees": degrees},
        )
        return rot
