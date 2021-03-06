"""Contains tools to handle rotations."""

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

from weldx.constants import WELDX_UNIT_REGISTRY as UREG

_DEFAULT_LEN_UNIT = UREG.millimeters
_DEFAULT_ANG_UNIT = UREG.rad


@UREG.wraps(None, _DEFAULT_ANG_UNIT, strict=False)
def rotation_matrix_x(angle):
    """Create a rotation matrix that rotates around the x-axis.

    Parameters
    ----------
    angle :
        Rotation angle

    Returns
    -------
    numpy.ndarray
        Rotation matrix

    """
    return _Rotation.from_euler("x", angle).as_matrix()


@UREG.wraps(None, _DEFAULT_ANG_UNIT, strict=False)
def rotation_matrix_y(angle):
    """Create a rotation matrix that rotates around the y-axis.

    Parameters
    ----------
    angle :
        Rotation angle

    Returns
    -------
    numpy.ndarray
        Rotation matrix

    """
    return _Rotation.from_euler("y", angle).as_matrix()


@UREG.wraps(None, _DEFAULT_ANG_UNIT, strict=False)
def rotation_matrix_z(angle) -> np.ndarray:
    """Create a rotation matrix that rotates around the z-axis.

    Parameters
    ----------
    angle :
        Rotation angle

    Returns
    -------
    numpy.ndarray
        Rotation matrix

    """
    return _Rotation.from_euler("z", angle).as_matrix()


class WXRotation(_Rotation):
    """Wrapper for creating meta-tagged Scipy.Rotation objects."""

    @classmethod
    def from_quat(cls, quat: np.ndarray) -> "WXRotation":
        """Initialize from quaternions.

        scipy.spatial.transform.Rotation docs for details.
        """
        rot = super().from_quat(quat)
        setattr(rot, "wx_meta", {"constructor": "from_quat"})
        return rot

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "WXRotation":
        """Initialize from matrix.

        scipy.spatial.transform.Rotation docs for details.
        """
        rot = super().from_matrix(matrix)
        setattr(rot, "wx_meta", {"constructor": "from_matrix"})
        return rot

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> "WXRotation":
        """Initialize from rotation vector.

        scipy.spatial.transform.Rotation docs for details.
        """
        rot = super().from_rotvec(rotvec)
        setattr(rot, "wx_meta", {"constructor": "from_rotvec"})
        return rot

    @classmethod
    def from_euler(cls, seq: str, angles, degrees: bool = False) -> "WXRotation":
        """Initialize from euler angles.

        scipy.spatial.transform.Rotation docs for details.
        """
        rot = super().from_euler(seq=seq, angles=angles, degrees=degrees)
        setattr(
            rot,
            "wx_meta",
            {"constructor": "from_euler", "seq": seq, "degrees": degrees},
        )
        return rot
