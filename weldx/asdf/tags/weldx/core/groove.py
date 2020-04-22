"""<DOCSTRING>"""

import pint
from dataclasses import dataclass
from dataclasses import field
from typing import List
import numpy as np

from weldx import Q_
import weldx.geometry as geo
from weldx.asdf.types import WeldxType
from weldx.asdf.utils import dict_to_tagged_tree


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
    """
    Create a Groove from weldx.asdf.tags.weldx.core.groove.

    Make a selection from the given groove types.
    Groove Types:
        "VGroove"
        "UGroove"
        "IGroove"
        "UVGroove"
        "VVGroove"
        "HVGroove"
        "HUGroove"
        "DUGroove"
        "DHVGroove"
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
                - The root gap is the distance of the 2 workpieces. It can be 0 or None.
            c: root face
                - The root face is the length of the Y-Groove which is not part of the V.
                  It can be 0.

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
                - The root gap is the distance of the 2 workpieces. It can be 0 or None.
            c: root face
                - The root face is the height of the part below the U-segment.

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

    :param groove_type: String specifying the Groove type
    :param workpiece_thickness: workpiece thickness
    :param workpiece_thickness2: workpiece thickness if type needs 2 thicknesses
    :param root_gap: root gap, gap between work pieces
    :param root_face: root face, upper part when 2 root faces are needed, middle part
                      when 3 are needed
    :param root_face2: root face, the lower part when 2 root faces are needed. upper
                       part when 3 are needed - used when min. 2 parts are needed
    :param root_face3: root face, usually the lower part - used when 3 parts are needed
    :param bevel_radius: bevel radius
    :param bevel_radius2: bevel radius - lower radius for DU-Groove
    :param bevel_angle: bevel angle, usually the upper angle
    :param bevel_angle2: bevel angle, usually the lower angle
    :param groove_angle: groove angle, usually the upper angle
    :param groove_angle2: groove angle, usually the lower angle
    :param special_depth: special depth used for 4.1.2 Frontal-Face-Groove
    :param code_number: String, used to define the Frontal Face Groove
    :return: an Groove from weldx.asdf.tags.weldx.core.groove
    """
    if groove_type == "VGroove":
        return VGroove(
            t=workpiece_thickness, alpha=groove_angle, b=root_gap, c=root_face
        )

    if groove_type == "UGroove":
        return UGroove(
            t=workpiece_thickness,
            beta=bevel_angle,
            R=bevel_radius,
            b=root_gap,
            c=root_face,
        )

    if groove_type == "IGroove":
        return IGroove(t=workpiece_thickness, b=root_gap)

    if groove_type == "UVGroove":
        return UVGroove(
            t=workpiece_thickness,
            alpha=groove_angle,
            beta=bevel_angle,
            R=bevel_radius,
            b=root_gap,
            h=root_face,
        )

    if groove_type == "VVGroove":
        return VVGroove(
            t=workpiece_thickness,
            alpha=groove_angle,
            beta=bevel_angle,
            b=root_gap,
            c=root_face,
            h=root_face2,
        )

    if groove_type == "HVGroove":
        return HVGroove(
            t=workpiece_thickness, beta=bevel_angle, b=root_gap, c=root_face
        )

    if groove_type == "HUGroove":
        return HUGroove(
            t=workpiece_thickness,
            beta=bevel_angle,
            R=bevel_radius,
            b=root_gap,
            c=root_face,
        )

    if groove_type == "DoubleVGroove":
        return DVGroove(
            t=workpiece_thickness,
            alpha_1=groove_angle,
            alpha_2=groove_angle2,
            b=root_gap,
            c=root_face,
            h1=root_face2,
            h2=root_face3,
        )

    if groove_type == "DoubleUGroove":
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

    if groove_type == "DoubleHVGroove":
        return DHVGroove(
            t=workpiece_thickness,
            beta_1=bevel_angle,
            beta_2=bevel_angle2,
            c=root_face,
            h1=root_face2,
            h2=root_face3,
            b=root_gap,
        )

    if groove_type == "DoubleHUGroove":
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

    if groove_type == "FrontalFaceGroove":
        return FFGroove(
            t_1=workpiece_thickness,
            t_2=workpiece_thickness2,
            alpha=groove_angle,
            b=root_gap,
            e=special_depth,
            code_number=code_number,
        )


@dataclass
class VGroove:
    """<CLASS DOCSTRING>"""

    t: pint.Quantity
    alpha: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.3", "1.5"])

    def plot(
        self, title=None, raster_width=0.1, axis="equal", grid=True, line_style="."
    ):
        """Plot a 2D-Profile."""
        profile = self.to_profile()
        if title is None:
            title = self.__class__
        profile.plot(title, raster_width, axis, grid, line_style)

    def to_profile(self, width_default=Q_(2, "mm")):
        """Calculate a Profile."""
        t = self.t.to("mm").magnitude
        alpha = self.alpha.to("rad").magnitude
        b = self.b.to("mm").magnitude
        c = self.c.to("mm").magnitude
        width = width_default.to("mm").magnitude

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

        return geo.Profile([shape, shape_r])


@dataclass
class VVGroove:
    """<CLASS DOCSTRING>"""

    t: pint.Quantity
    alpha: pint.Quantity
    beta: pint.Quantity
    h: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.7"])

    def plot(
        self, title=None, raster_width=0.1, axis="equal", grid=True, line_style="."
    ):
        """Plot a 2D-Profile."""
        profile = self.to_profile()
        if title is None:
            title = self.__class__
        profile.plot(title, raster_width, axis, grid, line_style)

    def to_profile(self, width_default=Q_(5, "mm")):
        """Calculate a Profile."""
        t = self.t.to("mm").magnitude
        alpha = self.alpha.to("rad").magnitude
        beta = self.beta.to("rad").magnitude
        b = self.b.to("mm").magnitude
        c = self.c.to("mm").magnitude
        h = self.h.to("mm").magnitude
        width = width_default.to("mm").magnitude

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

        return geo.Profile([shape, shape_r])


@dataclass
class UVGroove:
    """<CLASS DOCSTRING>"""

    t: pint.Quantity
    alpha: pint.Quantity
    beta: pint.Quantity
    R: pint.Quantity
    h: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.6"])

    def plot(
        self, title=None, raster_width=0.1, axis="equal", grid=True, line_style="."
    ):
        """Plot a 2D-Profile."""
        profile = self.to_profile()
        if title is None:
            title = self.__class__
        profile.plot(title, raster_width, axis, grid, line_style)

    def to_profile(self, width_default=Q_(2, "mm")):
        """Calculate a Profile."""
        t = self.t.to("mm").magnitude
        alpha = self.alpha.to("rad").magnitude
        beta = self.beta.to("rad").magnitude
        R = self.R.to("mm").magnitude
        b = self.b.to("mm").magnitude
        h = self.h.to("mm").magnitude
        width = width_default.to("mm").magnitude

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

        return geo.Profile([shape, shape_r])


@dataclass
class UGroove:
    """<CLASS DOCSTRING>"""

    t: pint.Quantity
    beta: pint.Quantity
    R: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.8"])

    def plot(
        self, title=None, raster_width=0.1, axis="equal", grid=True, line_style="."
    ):
        """Plot a 2D-Profile."""
        profile = self.to_profile()
        if title is None:
            title = self.__class__
        profile.plot(title, raster_width, axis, grid, line_style)

    def to_profile(self, width_default=Q_(3, "mm")):
        """Calculate a Profile."""
        t = self.t.to("mm").magnitude
        beta = self.beta.to("rad").magnitude
        R = self.R.to("mm").magnitude
        b = self.b.to("mm").magnitude
        c = self.c.to("mm").magnitude
        width = width_default.to("mm").magnitude

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

        return geo.Profile([shape, shape_r])


@dataclass
class IGroove:
    """<CLASS DOCSTRING>"""

    t: pint.Quantity
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.2.1", "1.2.2", "2.1"])

    def plot(
        self, title=None, raster_width=0.1, axis="equal", grid=True, line_style="."
    ):
        """Plot a 2D-Profile."""
        profile = self.to_profile()
        if title is None:
            title = self.__class__
        profile.plot(title, raster_width, axis, grid, line_style)

    def to_profile(self, width_default=Q_(5, "mm")):
        """Calculate a Profile."""
        t = self.t.to("mm").magnitude
        b = self.b.to("mm").magnitude
        width = width_default.to("mm").magnitude

        # x-values
        x_value = [-width, 0, 0, -width]
        # y-values
        y_value = [0, 0, t, t]
        segment_list = ["line", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate([-b / 2, 0])
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r])


@dataclass
class HVGroove:
    """<CLASS DOCSTRING>"""

    t: pint.Quantity
    beta: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.9.1", "1.9.2", "2.8"])


@dataclass
class HUGroove:
    """<CLASS DOCSTRING>"""

    t: pint.Quantity
    beta: pint.Quantity
    R: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["1.11", "2.10"])


# double Grooves
@dataclass
class DVGroove:
    """<CLASS DOCSTRING>"""

    t: pint.Quantity
    alpha_1: pint.Quantity
    alpha_2: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    h1: pint.Quantity = None
    h2: pint.Quantity = None
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.4", "2.5.1", "2.5.2"])


@dataclass
class DUGroove:
    """<CLASS DOCSTRING>"""

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


@dataclass
class DHVGroove:
    """<CLASS DOCSTRING>"""

    t: pint.Quantity
    beta_1: pint.Quantity
    beta_2: pint.Quantity
    c: pint.Quantity = Q_(0, "mm")
    h1: pint.Quantity = None
    h2: pint.Quantity = None
    b: pint.Quantity = Q_(0, "mm")
    code_number: List[str] = field(default_factory=lambda: ["2.9.1", "2.9.2"])


@dataclass
class DHUGroove:
    """<CLASS DOCSTRING>"""

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


# Frontal Face - Groove
@dataclass
class FFGroove:
    """<CLASS DOCSTRING>"""

    t_1: pint.Quantity
    t_2: pint.Quantity = None
    alpha: pint.Quantity = None
    # ["1.12", "1.13", "2.12", "3.1.1", "3.1.2", "3.1.3", "4.1.1", "4.1.2", "4.1.3"]
    code_number: str = None
    b: pint.Quantity = Q_(0, "mm")
    e: pint.Quantity = None


def _helperfunction(segment, array):
    """
    Calculate a shape from input.
    Input segment of successive segments as strings.
    Input array of the points in the correct sequence. e.g.:
    array = [[x-values], [y-values]]

    :param segment: list of String, segment names ("line", "arc")
    :param array: array of 2 array,
        first array are x-values
        second array are y-values
    :return: geo.Shape
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


class GrooveType(WeldxType):
    """<ASDF TYPE DOCSTRING>"""

    name = "core/din_en_iso_9692-1_2013"
    version = "1.0.0"
    types = [
        VGroove,
        VVGroove,
        UVGroove,
        UGroove,
        IGroove,
        UVGroove,
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
        """<CLASS METHOD DOCSTRING>"""
        if isinstance(node, VGroove):
            tree = dict(components=dict_to_tagged_tree(node, ctx), type="SingleVGroove")
            return tree

        if isinstance(node, UGroove):
            tree = dict(components=dict_to_tagged_tree(node, ctx), type="SingleUGroove")
            return tree

        if isinstance(node, UVGroove):
            tree = dict(components=dict_to_tagged_tree(node, ctx), type="UVGroove")
            return tree

        if isinstance(node, IGroove):
            tree = dict(components=dict_to_tagged_tree(node, ctx), type="IGroove")
            return tree

        if isinstance(node, VVGroove):
            tree = dict(components=dict_to_tagged_tree(node, ctx), type="VVGroove")
            return tree

        if isinstance(node, HVGroove):
            tree = dict(components=dict_to_tagged_tree(node, ctx), type="HVGroove")
            return tree

        if isinstance(node, HUGroove):
            tree = dict(components=dict_to_tagged_tree(node, ctx), type="HUGroove")
            return tree

        if isinstance(node, DVGroove):
            tree = dict(components=dict_to_tagged_tree(node, ctx), type="DoubleVGroove")
            return tree

        if isinstance(node, DUGroove):
            tree = dict(components=dict_to_tagged_tree(node, ctx), type="DoubleUGroove")
            return tree

        if isinstance(node, DHVGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx), type="DoubleHVGroove"
            )
            return tree

        if isinstance(node, DHUGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx), type="DoubleHUGroove"
            )
            return tree

        if isinstance(node, FFGroove):
            tree = dict(
                components=dict_to_tagged_tree(node, ctx), type="FrontalFaceGroove"
            )
            return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """<CLASS METHOD DOCSTRING>"""
        if tree["type"] == "SingleVGroove":
            obj = VGroove(**tree["components"])
            return obj

        if tree["type"] == "SingleUGroove":
            obj = UGroove(**tree["components"])
            return obj

        if tree["type"] == "UVGroove":
            obj = UVGroove(**tree["components"])
            return obj

        if tree["type"] == "IGroove":
            obj = IGroove(**tree["components"])
            return obj

        if tree["type"] == "VVGroove":
            obj = VVGroove(**tree["components"])
            return obj

        if tree["type"] == "HVGroove":
            obj = HVGroove(**tree["components"])
            return obj

        if tree["type"] == "HUGroove":
            obj = HUGroove(**tree["components"])
            return obj

        if tree["type"] == "DoubleVGroove":
            obj = DVGroove(**tree["components"])
            return obj

        if tree["type"] == "DoubleUGroove":
            obj = DUGroove(**tree["components"])
            return obj

        if tree["type"] == "DoubleHVGroove":
            obj = DHVGroove(**tree["components"])
            return obj

        if tree["type"] == "DoubleHUGroove":
            obj = DHUGroove(**tree["components"])
            return obj

        if tree["type"] == "FrontalFaceGroove":
            obj = FFGroove(**tree["components"])
            return obj
