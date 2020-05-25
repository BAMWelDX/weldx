"""<DOCSTRING>"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pint

import weldx.geometry as geo
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.asdf.types import WeldxType
from weldx.asdf.utils import dict_to_tagged_tree

_DEFAULT_LEN_UNIT = "mm"


def get_groove(
    groove_type,
    workpiece_thickness=None,
    workpiece_thickness2=None,
    root_gap=None,
    root_face=None,
    root_face2=None,
    root_face3=None,
    bevel_radius=None,
    bevel_radius2=None,
    bevel_angle=None,
    bevel_angle2=None,
    groove_angle=None,
    groove_angle2=None,
    special_depth=None,
    code_number=None,
):
    """Create a Groove from weldx.asdf.tags.weldx.core.groove.

    Make a selection from the given groove types.
    Groove Types:
        "VGroove"
        "UGroove"
        "IGroove"
        "UVGroove"
        "VVGroove"
        "HVGroove"
        "HUGroove"
        "DoubleVGroove"
        "DoubleUGroove"
        "DoubleHVGroove"
        "DoubleHUGroove"
        "FrontalFaceGroove"

    Each groove type has a different set of attributes which are required. Only
    required attributes are considered. All the required attributes for Grooves
    are in Quantity values from pint and related units are accepted.
    Required Groove attributes:
        "VGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            alpha: groove angle
                - The groove angle is the whole angle of the V-Groove.
                  It is a pint Quantity in degree or radian.
            b: root gap
                - The root gap is the distance of the 2 workpieces.
                  It can be 0 or None.
            c: root face
                - The root face is the length of the Y-Groove which is not
                  part of the V. It can be 0.

        "UGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            beta: bevel angle
                - The bevel angle is the angle that emerges from the circle segment.
                  Where 0 degree would be a parallel to the root face and 90 degree
                  would be a parallel to the workpiece.
            R: bevel radius
                - The bevel radius defines the length of the radius of the U-segment.
                  It is usually 6 millimeters.
            b: root gap
                - The root gap is the distance of the 2 workpieces.
                  It can be 0 or None.
            c: root face
                - The root face is the height of the part below the U-segment.

        "IGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            b: root gap
                - The root gap is the distance of the 2 workpieces.
                  It can be 0 or None.

        "UVGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            alpha: groove angle
                - The groove angle is the whole angle of the V-Groove part.
                  It is a pint Quantity in degree or radian.
            beta: bevel angle
                - The bevel angle is the angle that emerges from the circle segment.
                  Where 0 degree would be a parallel to the root face and 90 degree
                  would be a parallel to the workpiece.
            R: bevel radius
                - The bevel radius defines the length of the radius of the U-segment.
                  It is usually 6 millimeters.
            b: root gap
                - The root gap is the distance of the 2 workpieces.
                  It can be 0 or None.
            h: root face
                - The root face is the height of the V-segment.

        "VVGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            alpha: groove angle
                - The groove angle is the whole angle of the lower V-Groove part.
                  It is a pint Quantity in degree or radian.
            beta: bevel angle
                - The bevel angle is the angle of the upper V-Groove part.
                  It is a pint Quantity in degree or radian.
            b: root gap
                - The root gap is the distance of the 2 workpieces.
                  It can be 0 or None.
            c: root face
                - The root face is the height of the part below the lower V-segment.
                  It can be 0 or None.
            h: root face 2
                - This root face is the height of the part of the lower V-segment
                  and the root face c.

        "HVGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            beta: bevel angle
                - The bevel angle is the angle of the V-Groove part.
                  It is a pint Quantity in degree or radian.
            b: root gap
                - The root gap is the distance of the 2 workpieces.
                  It can be 0 or None.
            c: root face
                - The root face is the height of the part below the V-segment.

        "HUGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            beta: bevel angle
                - The bevel angle is the angle that emerges from the circle segment.
                  Where 0 degree would be a parallel to the root face and 90 degree
                  would be a parallel to the workpiece.
            R: bevel radius
                - The bevel radius defines the length of the radius of the U-segment.
                  It is usually 6 millimeters.
            b: root gap
                - The root gap is the distance of the 2 workpieces.
                  It can be 0 or None.
            c: root face
                - The root face is the height of the part below the U-segment.

        "DoubleVGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            alpha: groove angle
                - The groove angle is the whole angle of the upper V-Groove part.
                  It is a pint Quantity in degree or radian.
            alpha2: groove angle
                - The groove angle is the whole angle of the lower V-Groove part.
                  It is a pint Quantity in degree or radian.
            b: root gap
                - The root gap is the distance of the 2 workpieces.
                  It can be 0 or None.
            c: root face
                - The root face is the height of the part between the V-segments.
            h1: root face 2
                - The root face is the height of the upper V-segment.
                  Only c is needed.
            h2: root face 3
                - The root face is the height of the lower V-segment.
                  Only c is needed.

        "DoubleUGroove":
            t: workpiece thickness
                - The workpiece thickness is a length Quantity, e.g.: "mm".
                  It is assumed that both workpieces have the same thickness.
            beta: bevel angle
                - The bevel angle is the angle that emerges from the circle segment.
                  Where 0 degree would be a parallel to the root face and 90 degree
                  would be a parallel to the workpiece. The upper U-segment.
            beta2: bevel angle 2
                - The bevel angle is the angle that emerges from the circle segment.
                  Where 0 degree would be a parallel to the root face and 90 degree
                  would be a parallel to the workpiece. The lower U-segment.
            R: bevel radius
                - The bevel radius defines the length of the radius of the
                  upper U-segment. It is usually 6 millimeters.
            R2: bevel radius 2
                - The bevel radius defines the length of the radius of the
                  lower U-segment. It is usually 6 millimeters.
            b: root gap
                - The root gap is the distance of the 2 workpieces.
                  It can be 0 or None.
            c: root face
                - The root face is the height of the part between the U-segments.
            h1: root face 2
                - The root face is the height of the upper U-segment.
                  Only c is needed.
            h2: root face 3
                - The root face is the height of the lower U-segment.
                  Only c is needed.

        "DoubleHVGroove":
            This is a special case of the DoubleVGroove. The values of the angles are
            interpreted here as bevel angel. So you have only half of the size.
            Accordingly the inputs beta1 (bevel angle) and beta2 (bevel angle 2)
            are used.

        "DoubleHUGroove":
            This is a special case of the DoubleUGroove.
            The parameters remain the same.

        "FrontalFaceGroove":
            These grooves are identified by their code number. These correspond to the
            key figure numbers from the standard. For more information, see the
            documentation.


    Example:
        from weldx import Q_ as Quantity
        from weldx.asdf.tags.weldx.core.groove import get_groove

        get_groove(groove_type="VGroove",
                   workpiece_thickness=Quantity(9, "mm"),
                   groove_angle=Quantity(50, "deg"),
                   root_face=Quantity(4, "mm"),
                   root_gap=Quantity(2, "mm"))

        get_groove(groove_type="UGroove",
                   workpiece_thickness=Quantity(15, "mm"),
                   bevel_angle=Quantity(9, "deg"),
                   bevel_radius=Quantity(6, "mm"),
                   root_face=Quantity(3, "mm"),
                   root_gap=Quantity(1, "mm"))

    Parameters
    ----------
    groove_type :
        String specifying the Groove type
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

    """
    if groove_type == "VGroove":
        return VGroove(
            t=workpiece_thickness, alpha=groove_angle, b=root_gap, c=root_face
        )

    elif groove_type == "UGroove":
        return UGroove(
            t=workpiece_thickness,
            beta=bevel_angle,
            R=bevel_radius,
            b=root_gap,
            c=root_face,
        )

    elif groove_type == "IGroove":
        return IGroove(t=workpiece_thickness, b=root_gap)

    elif groove_type == "UVGroove":
        return UVGroove(
            t=workpiece_thickness,
            alpha=groove_angle,
            beta=bevel_angle,
            R=bevel_radius,
            b=root_gap,
            h=root_face,
        )

    elif groove_type == "VVGroove":
        return VVGroove(
            t=workpiece_thickness,
            alpha=groove_angle,
            beta=bevel_angle,
            b=root_gap,
            c=root_face,
            h=root_face2,
        )

    elif groove_type == "HVGroove":
        return HVGroove(
            t=workpiece_thickness, beta=bevel_angle, b=root_gap, c=root_face
        )

    elif groove_type == "HUGroove":
        return HUGroove(
            t=workpiece_thickness,
            beta=bevel_angle,
            R=bevel_radius,
            b=root_gap,
            c=root_face,
        )

    elif groove_type == "DoubleVGroove":
        return DVGroove(
            t=workpiece_thickness,
            alpha_1=groove_angle,
            alpha_2=groove_angle2,
            b=root_gap,
            c=root_face,
            h1=root_face2,
            h2=root_face3,
        )

    elif groove_type == "DoubleUGroove":
        return DUGroove(
            t=workpiece_thickness,
            beta_1=bevel_angle,
            beta_2=bevel_angle2,
            R=bevel_radius,
            R2=bevel_radius2,
            b=root_gap,
            c=root_face,
            h1=root_face2,
            h2=root_face3,
        )

    elif groove_type == "DoubleHVGroove":
        return DHVGroove(
            t=workpiece_thickness,
            beta_1=bevel_angle,
            beta_2=bevel_angle2,
            c=root_face,
            h1=root_face2,
            h2=root_face3,
            b=root_gap,
        )

    elif groove_type == "DoubleHUGroove":
        return DHUGroove(
            t=workpiece_thickness,
            beta_1=bevel_angle,
            beta_2=bevel_angle2,
            R=bevel_radius,
            R2=bevel_radius2,
            b=root_gap,
            c=root_face,
            h1=root_face2,
            h2=root_face3,
        )

    elif groove_type == "FrontalFaceGroove":
        return FFGroove(
            t_1=workpiece_thickness,
            t_2=workpiece_thickness2,
            alpha=groove_angle,
            b=root_gap,
            e=special_depth,
            code_number=code_number,
        )

    else:
        raise ValueError(f"{groove_type} is not a groove type or not yet implemented.")


class BaseGroove:
    """Generic base class for all groove types."""

    def parameters(self):
        """ """
        return {k: v for k, v in self.__dict__.items() if isinstance(v, pint.Quantity)}

    def param_strings(self):
        """Generate string representation of parameters."""
        return [f"{k}={v:~}" for k, v in self.parameters().items()]

    def _ipython_display_(self):
        """Display the Groove as plot in notebooks."""
        self.plot()

    def plot(
        self,
        title=None,
        axis_label=None,
        raster_width=0.5,
        show_params=True,
        axis="equal",
        grid=True,
        line_style=".-",
        ax=None,
    ):
        """Plot a 2D-Profile.

        Parameters
        ----------
        title :
             (Default value = None)
        axis_label :
            label string to pass onto matplotlib (Default value = None)
        raster_width :
             (Default value = 0.1)
        show_params :
             (Default value = True)
        axis :
             (Default value = "equal")
        grid :
             (Default value = True)
        line_style :
             (Default value = ".")
        ax :
             (Default value = None)

       """
        profile = self.to_profile()
        if title is None:
            title = _groove_type_to_name[self.__class__]

        if show_params:
            title = title + "\n" + ", ".join(self.param_strings())

        profile.plot(title, axis_label, raster_width, axis, grid, line_style, ax=ax)

    def to_profile(self, width_default: pint.Quantity = None) -> geo.Profile:
        """Implements profile generation.

        Parameters
        ----------
        width_default :
             optional width to extend each side of the profile (Default value = None)

       """
        raise NotImplementedError("to_profile() must be defined in subclass.")


@dataclass
class VGroove(BaseGroove):
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

    def to_profile(self, width_default=Q_(2, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(2, "mm"))


       """
        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        alpha = self.alpha.to("rad").magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        c = self.c.to(_DEFAULT_LEN_UNIT).magnitude
        width = width_default.to(_DEFAULT_LEN_UNIT).magnitude

        # Calculations:
        s = np.tan(alpha / 2) * (t - c)

        # Scaling
        edge = np.min([-s, 0])
        if width <= -edge + 1:
            width = width - edge

        # Bottom segment
        x_value = [-width, 0]
        y_value = [0, 0]
        segment_list = ["line"]

        # root face
        if c != 0:
            x_value.append(0)
            y_value.append(c)
            segment_list.append("line")

        # groove face
        x_value.append(-s)
        y_value.append(t)
        segment_list.append("line")

        # Top segment
        x_value.append(-width)
        y_value.append(t)
        segment_list.append("line")

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate([-b / 2, 0])
        # y-axis is mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)


@dataclass
class VVGroove(BaseGroove):
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

    def to_profile(self, width_default=Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))


       """
        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        alpha = self.alpha.to("rad").magnitude
        beta = self.beta.to("rad").magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        c = self.c.to(_DEFAULT_LEN_UNIT).magnitude
        h = self.h.to(_DEFAULT_LEN_UNIT).magnitude
        width = width_default.to(_DEFAULT_LEN_UNIT).magnitude

        # Calculations
        h_lower = h - c
        h_upper = t - h
        s_1 = np.tan(alpha / 2) * h_lower
        s_2 = np.tan(beta) * h_upper

        # Scaling
        edge = np.min([-(s_1 + s_2), 0])
        if width <= -edge + 1:
            # adjustment of the width
            width = width - edge

        # x-values
        x_value = [-width, 0]
        # y-values
        y_value = [0, 0]
        segment_list = ["line"]

        if c != 0:
            x_value.append(0)
            y_value.append(c)
            segment_list.append("line")

        x_value += [-s_1, -s_1 - s_2, -width]
        y_value += [h + c, t, t]
        segment_list += ["line", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate([-b / 2, 0])
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)


@dataclass
class UVGroove(BaseGroove):
    """An UV-Groove.

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

    def to_profile(self, width_default: pint.Quantity = Q_(2, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(2, "mm"))


       """
        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        alpha = self.alpha.to("rad").magnitude
        beta = self.beta.to("rad").magnitude
        R = self.R.to(_DEFAULT_LEN_UNIT).magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        h = self.h.to(_DEFAULT_LEN_UNIT).magnitude
        width = width_default.to(_DEFAULT_LEN_UNIT).magnitude

        # calculations:
        x_1 = np.tan(alpha / 2) * h
        # Center of the circle [0, y_m]
        y_circle = np.sqrt(R ** 2 - x_1 ** 2)
        y_m = h + y_circle
        # From next point to circle center is the vector (x,y)
        x = R * np.cos(beta)
        y = R * np.sin(beta)
        x_arc = -x
        y_arc = y_m - y
        # X-section of the upper edge
        x_end = x_arc - (t - y_arc) * np.tan(beta)

        # Scaling
        edge = np.max([-x_end, 0])
        if width <= edge + 1:
            # adjustment of the width
            width = width + edge

        # x-values
        x_value = [-width, 0, -x_1, 0, x_arc, x_end, -width]
        # y-values
        y_value = [0, 0, h, y_m, y_arc, t, t]
        segment_list = ["line", "line", "arc", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate([-b / 2, 0])
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)


@dataclass
class UGroove(BaseGroove):
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

    def to_profile(self, width_default: pint.Quantity = Q_(3, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(3, "mm"))


       """
        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        beta = self.beta.to("rad").magnitude
        R = self.R.to(_DEFAULT_LEN_UNIT).magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        c = self.c.to(_DEFAULT_LEN_UNIT).magnitude
        width = width_default.to(_DEFAULT_LEN_UNIT).magnitude

        # calculations:
        # From next point to circle center is the vector (x,y)
        x = R * np.cos(beta)
        y = R * np.sin(beta)
        # m = [0,c+R] circle center
        # => [-x,c+R-y] is the next point

        s = np.tan(beta) * (t - (c + R - y))

        # Scaling
        edge = np.max([x + s, 0])
        if width <= edge + 1:
            # adjustment of the width
            width = width + edge

        # x-values
        x_value = []
        # y-values
        y_value = []
        segment_list = []

        # bottom segment
        x_value.append(-width)
        y_value.append(0)
        x_value.append(0)
        y_value.append(0)
        segment_list.append("line")

        # root face
        if c != 0:
            x_value.append(0)
            y_value.append(c)
            segment_list.append("line")

        # groove face arc (circle center)
        x_value.append(0)
        y_value.append(c + R)

        # groove face arc
        x_value.append(-x)
        y_value.append(c + R - y)
        segment_list.append("arc")

        # groove face line
        x_value.append(-x - s)
        y_value.append(t)
        segment_list.append("line")

        # top segment
        x_value.append(-width)
        y_value.append(t)
        segment_list.append("line")

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate([-b / 2, 0])
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)


@dataclass
class IGroove(BaseGroove):
    """An I-Groove.

    For a detailed description of the execution look in get_groove.

    Parameters
    ----------
    t :
        workpiece thickness
    b :
        root gap
    code_number :
        Numbers of the standard

   """

    t: pint.Quantity
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.2.1", "1.2.2", "2.1"])

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))


       """
        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        width = width_default.to(_DEFAULT_LEN_UNIT).magnitude

        # x-values
        x_value = [-width, 0, 0, -width]
        # y-values
        y_value = [0, 0, t, t]
        segment_list = ["line", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate([-b / 2, 0])
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)


@dataclass
class HVGroove(BaseGroove):
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))


       """
        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        beta = self.beta.to("rad").magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        c = self.c.to(_DEFAULT_LEN_UNIT).magnitude
        width = width_default.to(_DEFAULT_LEN_UNIT).magnitude

        # Calculations
        s = np.tan(beta) * (t - c)

        # Scaling
        edge = np.min([-s, 0])
        if width <= -edge + 1:
            # adjustment of the width
            width = width - edge

        x_value = [-width, 0]
        y_value = [0, 0]
        segment_list = ["line"]

        if c != 0:
            x_value.append(0)
            y_value.append(c)
            segment_list.append("line")

        x_value += [-s, -width]
        y_value += [t, t]
        segment_list += ["line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])
        shape = shape.translate([-b / 2, 0])
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        shape_h = geo.Shape()
        shape_h.add_line_segments(
            [[-width - (b / 2), 0], [-b / 2, 0], [-b / 2, t], [-width - (b / 2), t]]
        )

        return geo.Profile([shape_h, shape_r], units=_DEFAULT_LEN_UNIT)


@dataclass
class HUGroove(BaseGroove):
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))


       """
        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        beta = self.beta.to("rad").magnitude
        R = self.R.to(_DEFAULT_LEN_UNIT).magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        c = self.c.to(_DEFAULT_LEN_UNIT).magnitude
        width = width_default.to(_DEFAULT_LEN_UNIT).magnitude

        # Calculations
        x = R * np.cos(beta)
        y = R * np.sin(beta)
        s = np.tan(beta) * (t - (c + R - y))

        # Scaling
        edge = np.max([x + s, 0])
        if width <= edge + 1:
            # adjustment of the width
            width = width + edge

        x_value = [-width, 0]
        y_value = [0, 0]
        segment_list = ["line"]

        if c != 0:
            x_value.append(0)
            y_value.append(c)
            segment_list.append("line")

        x_value += [0, -x, -x - s, -width]
        y_value += [c + R, c + R - y, t, t]
        segment_list += ["arc", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])
        shape = shape.translate([-b / 2, 0])
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        shape_h = geo.Shape()
        shape_h.add_line_segments(
            [[-width - (b / 2), 0], [-b / 2, 0], [-b / 2, t], [-width - (b / 2), t]]
        )

        return geo.Profile([shape_h, shape_r], units=_DEFAULT_LEN_UNIT)


# double Grooves
@dataclass
class DVGroove(BaseGroove):
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))


       """
        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        alpha_1 = self.alpha_1.to("rad").magnitude
        alpha_2 = self.alpha_2.to("rad").magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        c = self.c.to(_DEFAULT_LEN_UNIT).magnitude
        if self.h1 is None and self.h2 is None:
            h1 = (t - c) / 2
            h2 = (t - c) / 2
        elif self.h1 is not None and self.h2 is None:
            h1 = self.h1.to(_DEFAULT_LEN_UNIT).magnitude
            h2 = h1
        elif self.h1 is None and self.h2 is not None:
            h2 = self.h2.to(_DEFAULT_LEN_UNIT).magnitude
            h1 = h2
        else:
            h1 = self.h1.to(_DEFAULT_LEN_UNIT).magnitude
            h2 = self.h2.to(_DEFAULT_LEN_UNIT).magnitude
        width = width_default.to(_DEFAULT_LEN_UNIT).magnitude

        # Calculations
        s_upper = np.tan(alpha_1 / 2) * h1
        s_lower = np.tan(alpha_2 / 2) * h2

        # Scaling
        edge = np.min([-s_upper, -s_lower, 0])
        if width <= -edge + 1:
            # adjustment of the width
            width = width - edge

        x_value = [-width, -s_lower, 0]
        y_value = [0, 0, h2]
        segment_list = ["line", "line"]

        if c != 0:
            x_value.append(0)
            y_value.append(h2 + c)
            segment_list.append("line")

        x_value += [-s_upper, -width]
        y_value += [t, t]
        segment_list += ["line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])
        shape = shape.translate([-b / 2, 0])
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)


@dataclass
class DUGroove(BaseGroove):
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))


       """
        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        beta_1 = self.beta_1.to("rad").magnitude
        beta_2 = self.beta_2.to("rad").magnitude
        R = self.R.to(_DEFAULT_LEN_UNIT).magnitude
        R2 = self.R2.to(_DEFAULT_LEN_UNIT).magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        c = self.c.to(_DEFAULT_LEN_UNIT).magnitude
        if self.h1 is None and self.h2 is None:
            h1 = (t - c) / 2
            h2 = (t - c) / 2
        elif self.h1 is not None and self.h2 is None:
            h1 = self.h1.to(_DEFAULT_LEN_UNIT).magnitude
            h2 = h1
        elif self.h1 is None and self.h2 is not None:
            h2 = self.h2.to(_DEFAULT_LEN_UNIT).magnitude
            h1 = h2
        else:
            h1 = self.h1.to(_DEFAULT_LEN_UNIT).magnitude
            h2 = self.h2.to(_DEFAULT_LEN_UNIT).magnitude
        width = width_default.to(_DEFAULT_LEN_UNIT).magnitude

        # Calculations
        x_upper = R * np.cos(beta_1)
        y_upper = R * np.sin(beta_1)
        s_upper = np.tan(beta_1) * (h1 - (R - y_upper))
        x_lower = R2 * np.cos(beta_2)
        y_lower = R2 * np.sin(beta_2)
        s_lower = np.tan(beta_2) * (h2 - (R2 - y_lower))

        # Scaling
        edge = np.max([x_upper + s_upper, x_lower + s_lower, 0])
        if width <= edge + 1:
            # adjustment of the width
            width = width + edge

        x_value = [-width, -(s_lower + x_lower), -x_lower, 0, 0]
        y_value = [0, 0, h2 - (R2 - y_lower), h2 - R2, h2]
        segment_list = ["line", "line", "arc"]

        if c != 0:
            x_value.append(0)
            y_value.append(h1 + c)
            segment_list.append("line")

        x_value += [0, -x_upper, -(s_upper + x_upper), -width]
        y_value += [h2 + c + R, t - (h1 - (R - y_upper)), t, t]
        segment_list += ["arc", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])
        shape = shape.translate([-b / 2, 0])
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)


@dataclass
class DHVGroove(BaseGroove):
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))


       """
        dv_groove = DVGroove(
            self.t,
            self.beta_1 * 2,
            self.beta_2 * 2,
            self.c,
            self.h1,
            self.h2,
            self.b,
            self.code_number,
        )
        dv_profile = dv_groove.to_profile(width_default)
        right_shape = dv_profile.shapes[1]

        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        width_default = width_default.to(_DEFAULT_LEN_UNIT).magnitude
        left_shape = geo.Shape()
        left_shape.add_line_segments(
            [
                [-width_default - (b / 2), 0],
                [-b / 2, 0],
                [-b / 2, t],
                [-width_default - (b / 2), t],
            ]
        )

        return geo.Profile([left_shape, right_shape], units=_DEFAULT_LEN_UNIT)


@dataclass
class DHUGroove(BaseGroove):
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))


       """
        du_profile = DUGroove(
            self.t,
            self.beta_1,
            self.beta_2,
            self.R,
            self.R2,
            self.c,
            self.h1,
            self.h2,
            self.b,
            self.code_number,
        ).to_profile(width_default)
        right_shape = du_profile.shapes[1]

        t = self.t.to(_DEFAULT_LEN_UNIT).magnitude
        b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
        width_default = width_default.to(_DEFAULT_LEN_UNIT).magnitude
        left_shape = geo.Shape()
        left_shape.add_line_segments(
            [
                [-width_default - (b / 2), 0],
                [-b / 2, 0],
                [-b / 2, t],
                [-width_default - (b / 2), t],
            ]
        )

        return geo.Profile([left_shape, right_shape], units=_DEFAULT_LEN_UNIT)


# Frontal Face - Groove
@dataclass
class FFGroove(BaseGroove):
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))


       """
        if (
            self.code_number == "1.12"
            or self.code_number == "1.13"
            or self.code_number == "2.12"
        ):
            t_1 = self.t_1.to(_DEFAULT_LEN_UNIT).magnitude
            width_default = width_default.to(_DEFAULT_LEN_UNIT).magnitude
            shape1 = geo.Shape()
            shape1.add_line_segments(
                [
                    [0, 0],
                    [2 * width_default + t_1, 0],
                    [2 * width_default + t_1, t_1],
                    [0, t_1],
                    [0, 0],
                ]
            )
            shape2 = geo.Shape()
            shape2.add_line_segments(
                [
                    [width_default, 0],
                    [width_default + t_1, 0],
                    [width_default + t_1, -width_default],
                    [width_default, -width_default],
                    [width_default, 0],
                ]
            )
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "3.1.1":
            t_1 = self.t_1.to(_DEFAULT_LEN_UNIT).magnitude
            t_2 = self.t_2.to(_DEFAULT_LEN_UNIT).magnitude
            alpha = self.alpha.to("rad").magnitude
            b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
            width_default = width_default.to(_DEFAULT_LEN_UNIT).magnitude

            if width_default < t_1 + 1:
                width_default = t_1 + width_default

            # x = t_1
            # y = 0

            x_1 = np.cos(alpha) * width_default
            y_1 = np.sin(alpha) * width_default

            x_2 = np.cos(alpha + np.pi / 2) * t_1
            y_2 = np.sin(alpha + np.pi / 2) * t_1

            x_3 = x_1 + x_2
            y_3 = y_1 + y_2

            shape1 = geo.Shape()
            shape1.add_line_segments(
                [[t_1 + x_1, y_1], [t_1, 0], [t_1 + x_2, y_2], [t_1 + x_3, y_3]]
            )
            shape2 = geo.Shape()
            shape2.add_line_segments(
                [[width_default, -b], [0, -b], [0, -t_2 - b], [width_default, -t_2 - b]]
            )
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "3.1.2":
            t_1 = self.t_1.to(_DEFAULT_LEN_UNIT).magnitude
            t_2 = self.t_2.to(_DEFAULT_LEN_UNIT).magnitude
            b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
            width_default = width_default.to(_DEFAULT_LEN_UNIT).magnitude
            shape1 = geo.Shape()
            shape1.add_line_segments(
                [[0, 0], [width_default, 0], [width_default, t_1], [0, t_1]]
            )
            shape2 = geo.Shape()
            shape2.add_line_segments(
                [
                    [0, -b],
                    [2 * width_default, -b],
                    [2 * width_default, -t_2 - b],
                    [0, -t_2 - b],
                ]
            )
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "3.1.3" or self.code_number == "4.1.1":
            t_1 = self.t_1.to(_DEFAULT_LEN_UNIT).magnitude
            t_2 = self.t_2.to(_DEFAULT_LEN_UNIT).magnitude
            alpha = self.alpha.to("rad").magnitude
            b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
            width_default = width_default.to(_DEFAULT_LEN_UNIT).magnitude

            x = np.sin(alpha + np.pi / 2) * b + b
            y = np.cos(alpha + np.pi / 2) * b

            x_1 = np.sin(alpha) * t_2 + x
            y_1 = np.cos(alpha) * t_2 + y

            x_2 = np.sin(alpha + np.pi / 2) * (b + width_default) + b
            y_2 = np.cos(alpha + np.pi / 2) * (b + width_default)

            x_3 = x_1 + x_2 - x
            y_3 = y_1 + y_2 - y

            shape1 = geo.Shape()
            shape1.add_line_segments(
                [[-width_default, 0], [0, 0], [0, t_1], [-width_default, t_1]]
            )
            shape2 = geo.Shape()
            shape2.add_line_segments([[x_3, y_3], [x_1, y_1], [x, y], [x_2, y_2]])
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "4.1.2":
            t_1 = self.t_1.to(_DEFAULT_LEN_UNIT).magnitude
            t_2 = self.t_2.to(_DEFAULT_LEN_UNIT).magnitude
            alpha = self.alpha.to("rad").magnitude
            e = self.e.to(_DEFAULT_LEN_UNIT).magnitude
            width_default = width_default.to(_DEFAULT_LEN_UNIT).magnitude

            x_1 = np.sin(alpha) * e
            y_1 = np.cos(alpha) * e

            x_2 = np.sin(alpha + np.pi) * (t_2 - e)
            y_2 = np.cos(alpha + np.pi) * (t_2 - e)

            x_3 = x_2 + np.sin(alpha + np.pi / 2) * width_default
            y_3 = y_2 + np.cos(alpha + np.pi / 2) * width_default

            x_4 = x_1 + np.sin(alpha + np.pi / 2) * width_default
            y_4 = y_1 + np.cos(alpha + np.pi / 2) * width_default

            shape1 = geo.Shape()
            shape1.add_line_segments(
                [[-width_default, 0], [0, 0], [0, t_1], [-width_default, t_1]]
            )
            shape2 = geo.Shape()
            shape2.add_line_segments([[x_4, y_4], [x_1, y_1], [x_2, y_2], [x_3, y_3]])
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "4.1.3":
            t_1 = self.t_1.to(_DEFAULT_LEN_UNIT).magnitude
            t_2 = self.t_2.to(_DEFAULT_LEN_UNIT).magnitude
            b = self.b.to(_DEFAULT_LEN_UNIT).magnitude
            width_default = width_default.to(_DEFAULT_LEN_UNIT).magnitude
            shape1 = geo.Shape()
            shape1.add_line_segments(
                [[0, width_default], [0, 0], [t_1, 0], [t_1, width_default]]
            )
            shape2 = geo.Shape()
            shape2.add_line_segments(
                [
                    [-width_default, -b],
                    [t_1 + width_default, -b],
                    [t_1 + width_default, -t_2 - b],
                    [-width_default, -t_2 - b],
                    [-width_default, -b],
                ]
            )
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        else:
            raise ValueError(
                "Wrong code_number. The Code Number has to be"
                " one of the following strings: "
                '"1.12", "1.13", "2.12", "3.1.1", "3.1.2",'
                ' "3.1.3", "4.1.1", "4.1.2", "4.1.3"'
            )


def _helperfunction(segment, array):
    """Calculate a shape from input.
    Input segment of successive segments as strings.
    Input array of the points in the correct sequence. e.g.:
    array = [[x-values], [y-values]]

    Parameters
    ----------
    segment :
        list of String, segment names ("line", "arc")
    array :
        array of 2 array,
        first array are x-values
        second array are y-values

    Returns
    -------
    type
        geo.Shape

    """
    segment_list = []
    counter = 0
    for elem in segment:
        if elem == "line":
            seg = geo.LineSegment(
                [array[0][counter : counter + 2], array[1][counter : counter + 2]]
            )
            segment_list.append(seg)
            counter += 1
        if elem == "arc":
            arr0 = [
                # begin
                array[0][counter],
                # end
                array[0][counter + 2],
                # circle center
                array[0][counter + 1],
            ]
            arr1 = [
                # begin
                array[1][counter],
                # end
                array[1][counter + 2],
                # circle center
                array[1][counter + 1],
            ]
            seg = geo.ArcSegment([arr0, arr1], False)
            segment_list.append(seg)
            counter += 2

    return geo.Shape(segment_list)


_groove_type_to_name = {
    VGroove: "SingleVGroove",
    VVGroove: "VVGroove",
    UVGroove: "UVGroove",
    UGroove: "SingleUGroove",
    IGroove: "IGroove",
    HVGroove: "HVGroove",
    HUGroove: "HUGroove",
    DVGroove: "DoubleVGroove",
    DUGroove: "DoubleUGroove",
    DHVGroove: "DoubleHVGroove",
    DHUGroove: "DoubleHUGroove",
    FFGroove: "FrontalFaceGroove",
}

_groove_name_to_type = {v: k for k, v in _groove_type_to_name.items()}


class GrooveType(WeldxType):
    """ASDF Groove type."""

    name = "core/iso_groove"
    version = "1.0.0"
    types = [
        VGroove,
        VVGroove,
        UVGroove,
        UGroove,
        IGroove,
        HVGroove,
        HUGroove,
        DVGroove,
        DUGroove,
        DHVGroove,
        DHUGroove,
        FFGroove,
    ]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node, ctx):
        """Convert to tagged tree and remove all None entries from node dictionary."""
        if isinstance(node, tuple(_groove_type_to_name.keys())):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx),
                type=_groove_type_to_name[type(node)],
            )
            return tree
        else:  # pragma: no cover
            raise ValueError(
                f"Unknown groove type for object {node} with type {type(node)}"
            )

    @classmethod
    def from_tree(cls, tree, ctx):
        """Convert from tagged tree to a groove."""
        if tree["type"] in _groove_name_to_type:
            obj = _groove_name_to_type[tree["type"]](**tree["components"])
            return obj
        else:  # pragma: no cover
            raise ValueError(f"Unknown groove name {tree['type']}")
