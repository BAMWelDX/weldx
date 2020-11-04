"""ISO 9692-1 welding groove type definitions"""

from dataclasses import dataclass, field
from typing import List

import pint

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.utility import ureg_check_class

__all__ = [
    "IGroove",
    "VGroove",
    "VVGroove",
    "UVGroove",
    "UGroove",
    "HVGroove",
    "HUGroove",
    "DVGroove",
    "DUGroove",
    "DHVGroove",
    "DHUGroove",
    "FFGroove",
]


class IsoBaseGroove:
    """Generic base class for all groove types."""

    def parameters(self):
        """Return groove parameters as dictionary of quantities."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, pint.Quantity)}

    def param_strings(self):
        """Generate string representation of parameters."""
        return [f"{k}={v:~}" for k, v in self.parameters().items()]


@ureg_check_class("[length]", "[length]", None)
@dataclass
class IGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """An I-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t : pint.Quantity
        workpiece thickness
    b : pint.Quantity
        root gap
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.2.1", "1.2.2", "2.1"])

    _mapping = dict(t="workpiece_thickness", b="root_gap")


@ureg_check_class("[length]", "[]", "[length]", "[length]", None)
@dataclass
class VGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A Single-V Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    alpha :
        groove angle
    b :
        root gap
    c :
        root face
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    alpha: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.3", "1.5"])

    _mapping = dict(
        t="workpiece_thickness", alpha="groove_angle", b="root_gap", c="root_face",
    )


@ureg_check_class("[length]", "[]", "[]", "[length]", "[length]", "[length]", None)
@dataclass
class VVGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A VV-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    alpha :
        groove angle
    beta :
        bevel angle
    b :
        root gap
    c :
        root face
    h :
        root face 2
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    alpha: pint.Quantity
    beta: pint.Quantity
    h: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.7"])

    _mapping = dict(
        t="workpiece_thickness",
        alpha="groove_angle",
        beta="bevel_angle",
        b="root_gap",
        c="root_face",
        h="root_face2",
    )


@ureg_check_class("[length]", "[]", "[]", "[length]", "[length]", "[length]", None)
@dataclass
class UVGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A UV-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    alpha :
        groove angle
    beta :
        bevel angle
    R :
        bevel radius
    b :
        root gap
    h :
        root face
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    alpha: pint.Quantity
    beta: pint.Quantity
    R: pint.Quantity
    h: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.6"])

    _mapping = dict(
        t="workpiece_thickness",
        alpha="groove_angle",
        beta="bevel_angle",
        R="bevel_radius",
        b="root_gap",
        h="root_face",
    )


@ureg_check_class("[length]", "[]", "[length]", "[length]", "[length]", None)
@dataclass
class UGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """An U-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    beta :
        bevel angle
    R :
        bevel radius
    b :
        root gap
    c :
        root face
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    beta: pint.Quantity
    R: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.8"])

    _mapping = dict(
        t="workpiece_thickness",
        beta="bevel_angle",
        R="bevel_radius",
        b="root_gap",
        c="root_face",
    )


@ureg_check_class("[length]", "[]", "[length]", "[length]", None)
@dataclass
class HVGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A HV-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    beta :
        bevel angle
    b :
        root gap
    c :
        root face
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    beta: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.9.1", "1.9.2", "2.8"])

    _mapping = dict(
        t="workpiece_thickness", beta="bevel_angle", b="root_gap", c="root_face"
    )


@ureg_check_class("[length]", "[]", "[length]", "[length]", "[length]", None)
@dataclass
class HUGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A HU-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    beta :
        bevel angle
    R :
        bevel radius
    b :
        root gap
    c :
        root face
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    beta: pint.Quantity
    R: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.11", "2.10"])

    _mapping = dict(
        t="workpiece_thickness",
        beta="bevel_angle",
        R="bevel_radius",
        b="root_gap",
        c="root_face",
    )


@ureg_check_class("[length]", "[]", "[]", "[length]", None, None, "[length]", None)
@dataclass
class DVGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A DV-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    alpha_1 :
        groove angle (upper)
    alpha_2 :
        groove angle (lower)
    b :
        root gap
    c :
        root face (middle)
    h1 :
        root face (upper)
    h2 :
        root face (lower)
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    alpha_1: pint.Quantity
    alpha_2: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    h1: pint.Quantity = None
    h2: pint.Quantity = None
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.4", "2.5.1", "2.5.2"])

    _mapping = dict(
        t="workpiece_thickness",
        alpha_1="groove_angle",
        alpha_2="groove_angle2",
        b="root_gap",
        c="root_face",
        h1="root_face2",
        h2="root_face3",
    )

    def __post_init__(self):
        if self.h1 is None and self.h2 is None:
            self.h1 = (self.t - self.c) / 2
            self.h2 = (self.t - self.c) / 2
        elif self.h1 is not None and self.h2 is None:
            self.h2 = self.h1
        elif self.h1 is None and self.h2 is not None:
            self.h1 = self.h2


@ureg_check_class(
    "[length]",
    "[]",
    "[]",
    "[length]",
    "[length]",
    "[length]",
    None,
    None,
    "[length]",
    None,
)
@dataclass
class DUGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A DU-Groove

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    beta_1 :
        bevel angle (upper)
    beta_2 :
        bevel angle (lower)
    R :
        bevel radius (upper)
    R2 :
        bevel radius (lower)
    b :
        root gap
    c :
        root face (middle)
    h1 :
        root face (upper)
    h2 :
        root face (lower)
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    beta_1: pint.Quantity
    beta_2: pint.Quantity
    R: pint.Quantity
    R2: pint.Quantity
    c: pint.Quantity = Q_(3, "mm")
    h1: pint.Quantity = None
    h2: pint.Quantity = None
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.7"])

    _mapping = dict(
        t="workpiece_thickness",
        beta_1="bevel_angle",
        beta_2="bevel_angle2",
        R="bevel_radius",
        R2="bevel_radius2",
        b="root_gap",
        c="root_face",
        h1="root_face2",
        h2="root_face3",
    )

    def __post_init__(self):
        if self.h1 is None and self.h2 is None:
            self.h1 = (self.t - self.c) / 2
            self.h2 = (self.t - self.c) / 2
        elif self.h1 is not None and self.h2 is None:
            self.h2 = self.h1
        elif self.h1 is None and self.h2 is not None:
            self.h1 = self.h2


@ureg_check_class("[length]", "[]", "[]", "[length]", None, None, "[length]", None)
@dataclass
class DHVGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A DHV-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    beta_1 :
        bevel angle (upper)
    beta_2 :
        bevel angle (lower)
    b :
        root gap
    c :
        root face (middle)
    h1 :
        root face (upper)
    h2 :
        root face (lower)
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    beta_1: pint.Quantity
    beta_2: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    h1: pint.Quantity = None
    h2: pint.Quantity = None
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.9.1", "2.9.2"])

    _mapping = dict(
        t="workpiece_thickness",
        beta_1="bevel_angle",
        beta_2="bevel_angle2",
        c="root_face",
        h1="root_face2",
        h2="root_face3",
        b="root_gap",
    )

    def __post_init__(self):
        if self.h1 is None and self.h2 is None:
            self.h1 = (self.t - self.c) / 2
            self.h2 = (self.t - self.c) / 2
        elif self.h1 is not None and self.h2 is None:
            self.h2 = self.h1
        elif self.h1 is None and self.h2 is not None:
            self.h1 = self.h2


@ureg_check_class(
    "[length]",
    "[]",
    "[]",
    "[length]",
    "[length]",
    "[length]",
    None,
    None,
    "[length]",
    None,
)
@dataclass
class DHUGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A DHU-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    beta_1 :
        bevel angle (upper)
    beta_2 :
        bevel angle (lower)
    R :
        bevel radius (upper)
    R2 :
        bevel radius (lower)
    b :
        root gap
    c :
        root face (middle)
    h1 :
        root face (upper)
    h2 :
        root face (lower)
    code_number :
        Numbers of the standard

    """

    t: pint.Quantity
    beta_1: pint.Quantity
    beta_2: pint.Quantity
    R: pint.Quantity
    R2: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    h1: pint.Quantity = None
    h2: pint.Quantity = None
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.11"])

    _mapping = dict(
        t="workpiece_thickness",
        beta_1="bevel_angle",
        beta_2="bevel_angle2",
        R="bevel_radius",
        R2="bevel_radius2",
        b="root_gap",
        c="root_face",
        h1="root_face2",
        h2="root_face3",
    )

    def __post_init__(self):
        if self.h1 is None and self.h2 is None:
            self.h1 = (self.t - self.c) / 2
            self.h2 = (self.t - self.c) / 2
        elif self.h1 is not None and self.h2 is None:
            self.h2 = self.h1
        elif self.h1 is None and self.h2 is not None:
            self.h1 = self.h2


@ureg_check_class(
    "[length]", None, None, None, None, None,
)
@dataclass
class FFGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A Frontal Face Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t_1 :
        workpiece thickness
    t_2 :
        workpiece thickness, if second thickness is needed
    alpha :
        groove angle
    b :
        root gap
    e :
        special depth
    code_number :
        Numbers of the standard

    """

    t_1: pint.Quantity
    t_2: pint.Quantity = None
    alpha: pint.Quantity = None
    # ["1.12", "1.13", "2.12", "3.1.1", "3.1.2", "3.1.3", "4.1.1", "4.1.2", "4.1.3"]
    code_number: str = None
    b: pint.Quantity = None
    e: pint.Quantity = None

    _mapping = dict(
        t_1="workpiece_thickness",
        t_2="workpiece_thickness2",
        alpha="groove_angle",
        b="root_gap",
        e="special_depth",
        code_number="code_number",
    )


# create class <-> name mapping
_groove_type_to_name = {cls: cls.__name__ for cls in IsoBaseGroove.__subclasses__()}
_groove_name_to_type = {cls.__name__: cls for cls in IsoBaseGroove.__subclasses__()}


def get_groove(
    groove_type: str,
    workpiece_thickness: pint.Quantity = None,
    workpiece_thickness2: pint.Quantity = None,
    root_gap: pint.Quantity = None,
    root_face: pint.Quantity = None,
    root_face2: pint.Quantity = None,
    root_face3: pint.Quantity = None,
    bevel_radius: pint.Quantity = None,
    bevel_radius2: pint.Quantity = None,
    bevel_angle: pint.Quantity = None,
    bevel_angle2: pint.Quantity = None,
    groove_angle: pint.Quantity = None,
    groove_angle2: pint.Quantity = None,
    special_depth: pint.Quantity = None,
    code_number=None,
):
    """Create a Groove from weldx.asdf.tags.weldx.core.groove.

    Parameters
    ----------
    groove_type :
        String specifying the Groove type:

        - VGroove_
        - UGroove_
        - IGroove_
        - UVGroove_
        - VVGroove_
        - HVGroove_
        - HUGroove_
        - DVGroove_
        - DUGroove_
        - DHVGroove_
        - DHUGroove_
        - FFGroove_
    workpiece_thickness :
        workpiece thickness (Default value = None)
    workpiece_thickness2 :
        workpiece thickness if type needs 2 thicknesses (Default value = None)
    root_gap :
        root gap, gap between work pieces (Default value = None)
    root_face :
        root face, upper part when 2 root faces are needed, middle part
        when 3 are needed (Default value = None)
    root_face2 :
        root face, the lower part when 2 root faces are needed. upper
        part when 3 are needed - used when min. 2 parts are needed
        (Default value = None)
    root_face3 :
        root face, usually the lower part - used when 3 parts are needed
        (Default value = None)
    bevel_radius :
        bevel radius (Default value = None)
    bevel_radius2 :
        bevel radius - lower radius for DU-Groove (Default value = None)
    bevel_angle :
        bevel angle, usually the upper angle (Default value = None)
    bevel_angle2 :
        bevel angle, usually the lower angle (Default value = None)
    groove_angle :
        groove angle, usually the upper angle (Default value = None)
    groove_angle2 :
        groove angle, usually the lower angle (Default value = None)
    special_depth :
        special depth used for 4.1.2 Frontal-Face-Groove (Default value = None)
    code_number :
        String, used to define the Frontal Face Groove (Default value = None)

    Returns
    -------
    type
        an Groove from weldx.asdf.tags.weldx.core.groove

    Examples
    --------
    Create a V-Groove::

        get_groove(groove_type="VGroove",
                   workpiece_thickness=Q_(9, "mm"),
                   groove_angle=Q_(50, "deg"),
                   root_face=Q_(4, "mm"),
                   root_gap=Q_(2, "mm"))

    Create a U-Groove::

        get_groove(groove_type="UGroove",
                   workpiece_thickness=Q_(15, "mm"),
                   bevel_angle=Q_(9, "deg"),
                   bevel_radius=Q_(6, "mm"),
                   root_face=Q_(3, "mm"),
                   root_gap=Q_(1, "mm"))

    Notes
    -----

    Each groove type has a different set of attributes which are required. Only
    required attributes are considered. All the required attributes for Grooves
    are in Quantity values from pint and related units are accepted.
    Required Groove attributes:

    .. _IGroove:

    IGroove:
        t: workpiece_thickness
            The workpiece thickness is a length Quantity, e.g.: "mm".
            It is assumed that both workpieces have the same thickness.
        b: root_gap
            The root gap is the distance of the 2 workpieces.
            It can be 0 or None.

    .. _VGroove:

    VGroove:
        t: workpiece_thickness
            The workpiece thickness is a length Quantity, e.g.: "mm".
            It is assumed that both workpieces have the same thickness.
        alpha: groove_angle
            The groove angle is the whole angle of the V-Groove.
            It is a pint Quantity in degree or radian.
        b: root_gap
            The root gap is the distance of the 2 workpieces.
            It can be 0 or None.
        c: root_face
            The root face is the length of the Y-Groove which is not
            part of the V. It can be 0.

    .. _UGroove:

    UGroove:
        t: workpiece_thickness
            The workpiece thickness is a length Quantity, e.g.: "mm".
            It is assumed that both workpieces have the same thickness.
        beta: bevel_angle
            The bevel angle is the angle that emerges from the circle segment.
            Where 0 degree would be a parallel to the root face and 90 degree
            would be a parallel to the workpiece.
        R: bevel_radius
            The bevel radius defines the length of the radius of the U-segment.
            It is usually 6 millimeters.
        b: root_gap
            The root gap is the distance of the 2 workpieces.
            It can be 0 or None.
        c: root_face
            The root face is the height of the part below the U-segment.

    .. _UVGroove:

    UVGroove:
        t: workpiece_thickness
            The workpiece thickness is a length Quantity, e.g.: "mm".
            It is assumed that both workpieces have the same thickness.
        alpha: groove_angle
            The groove angle is the whole angle of the V-Groove part.
            It is a pint Quantity in degree or radian.
        beta: bevel_angle
            The bevel angle is the angle that emerges from the circle segment.
            Where 0 degree would be a parallel to the root face and 90 degree
            would be a parallel to the workpiece.
        R: bevel_radius
            The bevel radius defines the length of the radius of the U-segment.
            It is usually 6 millimeters.
        b: root_gap
            The root gap is the distance of the 2 workpieces.
            It can be 0 or None.
        h: root_face
            The root face is the height of the V-segment.

    .. _VVGroove:

    VVGroove:
        t: workpiece_thickness
            The workpiece thickness is a length Quantity, e.g.: "mm".
            It is assumed that both workpieces have the same thickness.
        alpha: groove_angle
            The groove angle is the whole angle of the lower V-Groove part.
            It is a pint Quantity in degree or radian.
        beta: bevel_angle
            The bevel angle is the angle of the upper V-Groove part.
            It is a pint Quantity in degree or radian.
        b: root_gap
            The root gap is the distance of the 2 workpieces.
            It can be 0 or None.
        c: root_face
            The root face is the height of the part below the lower V-segment.
            It can be 0 or None.
        h: root_face2
            This root face is the height of the part of the lower V-segment
            and the root face c.

    .. _HVGroove:

    HVGroove:
        t: workpiece_thickness
            The workpiece thickness is a length Quantity, e.g.: "mm".
            It is assumed that both workpieces have the same thickness.
        beta: bevel_angle
            The bevel angle is the angle of the V-Groove part.
            It is a pint Quantity in degree or radian.
        b: root_gap
            The root gap is the distance of the 2 workpieces.
            It can be 0 or None.
        c: root_face
            The root face is the height of the part below the V-segment.

    .. _HUGroove:

    HUGroove:
        t: workpiece_thickness
            The workpiece thickness is a length Quantity, e.g.: "mm".
            It is assumed that both workpieces have the same thickness.
        beta: bevel_angle
            The bevel angle is the angle that emerges from the circle segment.
            Where 0 degree would be a parallel to the root face and 90 degree
            would be a parallel to the workpiece.
        R: bevel_radius
            The bevel radius defines the length of the radius of the U-segment.
            It is usually 6 millimeters.
        b: root_gap
            The root gap is the distance of the 2 workpieces.
            It can be 0 or None.
        c: root_face
            The root face is the height of the part below the U-segment.

    .. _DVGroove:

    DVGroove:
        t: workpiece_thickness
            The workpiece thickness is a length Quantity, e.g.: "mm".
            It is assumed that both workpieces have the same thickness.
        alpha: groove_angle
            The groove angle is the whole angle of the upper V-Groove part.
            It is a pint Quantity in degree or radian.
        alpha2: groove_angle
            The groove angle is the whole angle of the lower V-Groove part.
            It is a pint Quantity in degree or radian.
        b: root_gap
            The root gap is the distance of the 2 workpieces.
            It can be 0 or None.
        c: root_face
            The root face is the height of the part between the V-segments.
        h1: root_face2
            The root face is the height of the upper V-segment.
            Only c is needed.
        h2: root_face3
            The root face is the height of the lower V-segment.
            Only c is needed.

    .. _DUGroove:

    DUGroove:
        t: workpiece_thickness
            The workpiece thickness is a length Quantity, e.g.: "mm".
            It is assumed that both workpieces have the same thickness.
        beta: bevel_angle
            The bevel angle is the angle that emerges from the circle segment.
            Where 0 degree would be a parallel to the root face and 90 degree
            would be a parallel to the workpiece. The upper U-segment.
        beta2: bevel_angle2
            The bevel angle is the angle that emerges from the circle segment.
            Where 0 degree would be a parallel to the root face and 90 degree
            would be a parallel to the workpiece. The lower U-segment.
        R: bevel_radius
            The bevel radius defines the length of the radius of the
            upper U-segment. It is usually 6 millimeters.
        R2: bevel_radius2
            The bevel radius defines the length of the radius of the
            lower U-segment. It is usually 6 millimeters.
        b: root_gap
            The root gap is the distance of the 2 workpieces.
            It can be 0 or None.
        c: root_face
            The root face is the height of the part between the U-segments.
        h1: root_face2
            The root face is the height of the upper U-segment.
            Only c is needed.
        h2: root_face3
            The root face is the height of the lower U-segment.
            Only c is needed.

    .. _DHVGroove:

    DHVGroove:
        This is a special case of the DVGroove_. The values of the angles are
        interpreted here as bevel angel. So you have only half of the size.
        Accordingly the inputs beta1 (bevel angle) and beta2 (bevel angle 2)
        are used.

    .. _DHUGroove:

    DHUGroove:
        This is a special case of the DUGroove_.
        The parameters remain the same.

    .. _FFGroove:

    FFGroove:
        These grooves are identified by their code number. These correspond to the
        key figure numbers from the standard. For more information, see the
        documentation.

    """
    # get list of function parameters
    _loc = locals()

    groove_cls = _groove_name_to_type[groove_type]
    _mapping = groove_cls._mapping

    # convert function arguments to groove arguments
    args = {k: _loc[v] for k, v in _mapping.items() if _loc[v] is not None}
    if _loc["code_number"] is not None:
        args["code_number"] = _loc["code_number"]

    return groove_cls(**args)


def _create_test_grooves():
    """Create dictionary with examples for all groove variations."""
    v_groove = get_groove(
        groove_type="VGroove",
        workpiece_thickness=Q_(9, "mm"),
        groove_angle=Q_(50, "deg"),
        root_face=Q_(4, "mm"),
        root_gap=Q_(2, "mm"),
    )
    u_groove = get_groove(
        groove_type="UGroove",
        workpiece_thickness=Q_(15, "mm"),
        bevel_angle=Q_(9, "deg"),
        bevel_radius=Q_(6, "mm"),
        root_face=Q_(3, "mm"),
        root_gap=Q_(1, "mm"),
    )
    i_groove = get_groove(
        groove_type="IGroove", workpiece_thickness=Q_(4, "mm"), root_gap=Q_(4, "mm")
    )
    uv_groove = get_groove(
        groove_type="UVGroove",
        workpiece_thickness=Q_(12, "mm"),
        groove_angle=Q_(60, "deg"),
        bevel_angle=Q_(11, "deg"),
        bevel_radius=Q_(6, "mm"),
        root_face=Q_(4, "mm"),
        root_gap=Q_(2, "mm"),
    )
    vv_groove = get_groove(
        groove_type="VVGroove",
        workpiece_thickness=Q_(12, "mm"),
        groove_angle=Q_(70, "deg"),
        bevel_angle=Q_(13, "deg"),
        root_gap=Q_(3, "mm"),
        root_face=Q_(1, "mm"),
        root_face2=Q_(5, "mm"),
    )
    hv_groove = get_groove(
        groove_type="HVGroove",
        workpiece_thickness=Q_(9, "mm"),
        bevel_angle=Q_(55, "deg"),
        root_gap=Q_(2, "mm"),
        root_face=Q_(1, "mm"),
    )
    hu_groove = get_groove(
        groove_type="HUGroove",
        workpiece_thickness=Q_(18, "mm"),
        bevel_angle=Q_(15, "deg"),
        bevel_radius=Q_(8, "mm"),
        root_gap=Q_(2, "mm"),
        root_face=Q_(3, "mm"),
    )
    dv_groove = get_groove(
        groove_type="DVGroove",
        workpiece_thickness=Q_(19, "mm"),
        groove_angle=Q_(40, "deg"),
        groove_angle2=Q_(60, "deg"),
        root_gap=Q_(2, "mm"),
        root_face=Q_(5, "mm"),
        root_face2=Q_(7, "mm"),
        root_face3=Q_(7, "mm"),
    )
    dv_groove2 = get_groove(
        groove_type="DVGroove",
        workpiece_thickness=Q_(19, "mm"),
        groove_angle=Q_(40, "deg"),
        groove_angle2=Q_(60, "deg"),
        root_gap=Q_(2, "mm"),
        root_face=Q_(5, "mm"),
    )
    dv_groove3 = get_groove(
        groove_type="DVGroove",
        workpiece_thickness=Q_(19, "mm"),
        groove_angle=Q_(40, "deg"),
        groove_angle2=Q_(60, "deg"),
        root_gap=Q_(2, "mm"),
        root_face=Q_(5, "mm"),
        root_face3=Q_(7, "mm"),
    )
    # DU grooves
    du_groove = get_groove(
        groove_type="DUGroove",
        workpiece_thickness=Q_(33, "mm"),
        bevel_angle=Q_(8, "deg"),
        bevel_angle2=Q_(12, "deg"),
        bevel_radius=Q_(6, "mm"),
        bevel_radius2=Q_(6, "mm"),
        root_face=Q_(3, "mm"),
        root_face2=Q_(15, "mm"),
        root_gap=Q_(2, "mm"),
    )
    du_groove2 = get_groove(
        groove_type="DUGroove",
        workpiece_thickness=Q_(33, "mm"),
        bevel_angle=Q_(8, "deg"),
        bevel_angle2=Q_(12, "deg"),
        bevel_radius=Q_(6, "mm"),
        bevel_radius2=Q_(6, "mm"),
        root_face=Q_(3, "mm"),
        root_gap=Q_(2, "mm"),
    )
    du_groove3 = get_groove(
        groove_type="DUGroove",
        workpiece_thickness=Q_(33, "mm"),
        bevel_angle=Q_(8, "deg"),
        bevel_angle2=Q_(12, "deg"),
        bevel_radius=Q_(6, "mm"),
        bevel_radius2=Q_(6, "mm"),
        root_face=Q_(3, "mm"),
        root_face2=Q_(15, "mm"),
        root_face3=Q_(15, "mm"),
        root_gap=Q_(2, "mm"),
    )
    du_groove4 = get_groove(
        groove_type="DUGroove",
        workpiece_thickness=Q_(33, "mm"),
        bevel_angle=Q_(8, "deg"),
        bevel_angle2=Q_(12, "deg"),
        bevel_radius=Q_(6, "mm"),
        bevel_radius2=Q_(6, "mm"),
        root_face=Q_(3, "mm"),
        root_face3=Q_(15, "mm"),
        root_gap=Q_(2, "mm"),
    )
    dhv_groove = get_groove(
        groove_type="DHVGroove",
        workpiece_thickness=Q_(11, "mm"),
        bevel_angle=Q_(35, "deg"),
        bevel_angle2=Q_(60, "deg"),
        root_face2=Q_(5, "mm"),
        root_face=Q_(1, "mm"),
        root_gap=Q_(3, "mm"),
    )
    dhu_groove = get_groove(
        groove_type="DHUGroove",
        workpiece_thickness=Q_(32, "mm"),
        bevel_angle=Q_(10, "deg"),
        bevel_angle2=Q_(20, "deg"),
        bevel_radius=Q_(8, "mm"),
        bevel_radius2=Q_(8, "mm"),
        root_face2=Q_(15, "mm"),
        root_face=Q_(2, "mm"),
        root_gap=Q_(2, "mm"),
    )
    ff_groove0 = get_groove(
        groove_type="FFGroove", workpiece_thickness=Q_(5, "mm"), code_number="1.12",
    )
    ff_groove1 = get_groove(
        groove_type="FFGroove",
        workpiece_thickness=Q_(5, "mm"),
        workpiece_thickness2=Q_(7, "mm"),
        groove_angle=Q_(80, "deg"),
        root_gap=Q_(1, "mm"),
        code_number="3.1.1",
    )
    ff_groove2 = get_groove(
        groove_type="FFGroove",
        workpiece_thickness=Q_(2, "mm"),
        workpiece_thickness2=Q_(5, "mm"),
        root_gap=Q_(1, "mm"),
        code_number="3.1.2",
    )
    ff_groove3 = get_groove(
        groove_type="FFGroove",
        workpiece_thickness=Q_(2, "mm"),
        workpiece_thickness2=Q_(5, "mm"),
        groove_angle=Q_(80, "deg"),
        root_gap=Q_(1, "mm"),
        code_number="3.1.3",
    )
    ff_groove4 = get_groove(
        groove_type="FFGroove",
        workpiece_thickness=Q_(2, "mm"),
        workpiece_thickness2=Q_(5, "mm"),
        groove_angle=Q_(80, "deg"),
        special_depth=Q_(4, "mm"),
        code_number="4.1.2",
    )
    ff_groove5 = get_groove(
        groove_type="FFGroove",
        workpiece_thickness=Q_(2, "mm"),
        workpiece_thickness2=Q_(5, "mm"),
        root_gap=Q_(1, "mm"),
        code_number="4.1.3",
    )

    test_data = dict(
        v_groove=(v_groove, VGroove),
        u_groove=(u_groove, UGroove),
        i_groove=(i_groove, IGroove),
        uv_groove=(uv_groove, UVGroove),
        vv_groove=(vv_groove, VVGroove),
        hv_groove=(hv_groove, HVGroove),
        hu_groove=(hu_groove, HUGroove),
        dv_groove=(dv_groove, DVGroove),
        dv_groove2=(dv_groove2, DVGroove),
        dv_groove3=(dv_groove3, DVGroove),
        du_groove=(du_groove, DUGroove),
        du_groove2=(du_groove2, DUGroove),
        du_groove3=(du_groove3, DUGroove),
        du_groove4=(du_groove4, DUGroove),
        dhv_groove=(dhv_groove, DHVGroove),
        dhu_groove=(dhu_groove, DHUGroove),
        ff_groove0=(ff_groove0, FFGroove),
        ff_groove1=(ff_groove1, FFGroove),
        ff_groove2=(ff_groove2, FFGroove),
        ff_groove3=(ff_groove3, FFGroove),
        ff_groove4=(ff_groove4, FFGroove),
        ff_groove5=(ff_groove5, FFGroove),
    )

    return test_data
