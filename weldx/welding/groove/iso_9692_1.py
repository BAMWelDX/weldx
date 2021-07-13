"""ISO 9692-1 welding groove type definitions."""
import abc
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
import pint
from sympy import Point2D, Polygon

import weldx.geometry as geo
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.util import inherit_docstrings, ureg_check_class

__all__ = [
    "IsoBaseGroove",
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
    "get_groove",
]


_DEFAULT_LEN_UNIT = "mm"


def _set_default_heights(groove):
    """Calculate default h1/h2 values."""
    if groove.h1 is None and groove.h2 is None:
        groove.h1 = (groove.t - groove.c) / 2
        groove.h2 = (groove.t - groove.c) / 2
    elif groove.h1 is not None and groove.h2 is None:
        groove.h2 = groove.h1
    elif groove.h1 is None and groove.h2 is not None:
        groove.h1 = groove.h2


def _get_bounds(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _compute_cross_sect_shape_points(
    points: List[List[Union[Point2D, Tuple]]]
) -> pint.Quantity:  # noqa
    # Assumes that we have two separate shapes for each workpiece
    # 1. compute the total area of all workpieces
    # 2. compute bounding box of all pieces (this includes the rift)
    # 3. compute area A = A_outer - A_workpieces

    area_workpiece = 0
    bounds = []

    for shape_points in points:
        p = Polygon(*shape_points, evaluate=False)
        area_workpiece += abs(p.area)
        x1, y1, x2, y2 = p.bounds
        bounds.append((x1, y1))
        bounds.append((x2, y2))

    # outer bbox
    x1, y1, x2, y2 = _get_bounds(bounds)

    bounding_box = Polygon((x1, y1), (x2, y1), (x2, y2), (x1, y2), evaluate=False)

    return Q_(float(bounding_box.area - area_workpiece), "mm²")


class IsoBaseGroove(metaclass=abc.ABCMeta):
    """Generic base class for all groove types."""

    _AREA_RASTER_WIDTH = 0.1
    """steers the area approximation of the groove in ~cross_sect_area."""

    def __post_init__(self):
        """Make sure all parameters are valid after class init."""
        for key, value in self.parameters().items():
            if value < 0.0:
                raise ValueError(f"Invalid value for parameter {key}={value:~}")

    def parameters(self):
        """Return groove parameters as dictionary of quantities."""
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
        show_area: bool = True,
    ):
        """Plot a 2D groove profile.

        Parameters
        ----------
        title :
             custom plot title
        axis_label :
            label string to pass onto matplotlib (Default value = None)
        raster_width :
             rasterization distance
        show_params :
             list groove parameters in plot title
        axis :
             axis scaling style
        grid :
             matplotlib grid setting
        line_style :
             matplotlib linestyle
        ax :
             Axis to plot to
        show_area
            Calculate and show the groove cross section area in the plot title.

        """
        profile = self.to_profile()
        if title is None:
            title = _groove_type_to_name[self.__class__]

        if show_area:
            try:
                ca = self.cross_sect_area
                title = title + f" ({np.around(ca,1):~.3P})"
            except NotImplementedError:
                pass
            except Exception as ex:
                raise ex

        if show_params:
            title = title + "\n" + ", ".join(self.param_strings())

        profile.plot(
            title, raster_width, None, axis, axis_label, grid, line_style, ax=ax
        )

    @abstractmethod
    def to_profile(self, width_default: pint.Quantity = None) -> geo.Profile:
        """Implement profile generation.

        Parameters
        ----------
        width_default : pint.Quantity
             optional width to extend each side of the profile (Default value = None)

        Returns
        -------
        profile: weldx.geometry.Profile
            The Profile object contains the shapes forming the groove.

        """

    # TODO: there is some effort going on to define dimensionality as type annotation.
    # https://github.com/hgrecco/pint/pull/1259 with this we can can annotate something
    # like -> Q(['length**2'])
    @property
    @abc.abstractmethod
    def cross_sect_area(self) -> pint.Quantity:
        """Area of the cross-section of the two work pieces.

        Returns
        -------
        area : pint.Quantity
            The computed area in mm².

        """

    def _compute_cross_sect_area_from_profile(self):
        points = []
        profile = self.to_profile()

        for shape in profile.shapes:
            shape_points = []
            points.append(shape_points)
            for seg in shape.segments:
                if isinstance(seg, geo.LineSegment):
                    shape_points.append(seg.point_start)
                    shape_points.append(seg.point_end)
                else:
                    raise RuntimeError("only for line segments!")
        return _compute_cross_sect_shape_points(points)

    def _compute_cross_sect_area_interpolated(self):
        # this method computes an approximation of the area by creating a big polygon
        # out of the rasterization points
        profile = self.to_profile()
        rasterization = profile.rasterize(self._AREA_RASTER_WIDTH, stack=False)
        points = [[(x, y) for x, y in shape.T] for shape in rasterization]

        return _compute_cross_sect_shape_points(points)


@ureg_check_class("[length]", "[length]", None)
@inherit_docstrings
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))

        """
        t, b, width = self.t, self.b, width_default

        # x-values
        x_value = np.stack((-width, 0, 0, -width))

        # y-values
        y_value = np.stack((0, 0, t, t))
        segment_list = ["line", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate(np.append(-b / 2, 0))
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_from_profile()


@ureg_check_class("[length]", "[]", "[length]", "[length]", None)
@inherit_docstrings
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
        t="workpiece_thickness",
        alpha="groove_angle",
        b="root_gap",
        c="root_face",
    )

    def to_profile(self, width_default=Q_(2, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(2, "mm"))

        """
        t, alpha, b, c, width = self.t, self.alpha, self.b, self.c, width_default

        # Calculations:
        s = np.tan(alpha / 2) * (t - c)

        # Scaling
        edge = np.append(-s, 0).min()
        if width <= -edge + Q_(1, "mm"):
            width = width - edge

        # Bottom segment
        x_value = np.append(-width, 0)
        y_value = Q_([0, 0], "mm")
        segment_list = ["line"]

        # root face
        if c != 0:
            x_value = np.append(x_value, 0)
            y_value = np.append(y_value, c)
            segment_list.append("line")

        # groove face
        x_value = np.append(x_value, -s)
        y_value = np.append(y_value, t)
        segment_list.append("line")

        # Top segment
        x_value = np.append(x_value, -width)
        y_value = np.append(y_value, t)
        segment_list.append("line")

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate(np.append(-b / 2, 0))
        # y-axis is mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_from_profile()


@ureg_check_class("[length]", "[]", "[]", "[length]", "[length]", "[length]", None)
@inherit_docstrings
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

    def to_profile(self, width_default=Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))

        """
        t, b, c, h, width = self.t, self.b, self.c, self.h, width_default
        alpha, beta = self.alpha, self.beta

        # Calculations
        h_lower = h - c
        h_upper = t - h
        s_1 = np.tan(alpha / 2) * h_lower
        s_2 = np.tan(beta) * h_upper

        # Scaling
        edge = np.append(-(s_1 + s_2), 0).min()
        if width <= -edge + Q_(1, "mm"):
            # adjustment of the width
            width = width - edge

        # x-values
        x_value = np.append(-width, 0)
        # y-values
        y_value = Q_([0, 0], "mm")
        segment_list = ["line"]

        if c != 0:
            x_value = np.append(x_value, 0)
            y_value = np.append(y_value, c)
            segment_list.append("line")

        x_value = np.append(x_value, (-s_1, -s_1 - s_2, -width))
        y_value = np.append(y_value, (h + c, t, t))
        segment_list += ["line", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate(np.append(-b / 2, 0))
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_from_profile()


@ureg_check_class("[length]", "[]", "[]", "[length]", "[length]", "[length]", None)
@inherit_docstrings
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

    def to_profile(self, width_default: pint.Quantity = Q_(2, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(2, "mm"))

        """
        t, alpha, beta, R, b, h, width = (
            self.t,
            self.alpha,
            self.beta,
            self.R,
            self.b,
            self.h,
            width_default,
        )

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
        edge = np.append(-x_end, 0).max()
        if width <= edge + Q_(1, "mm"):
            # adjustment of the width
            width = width + edge

        # x-values
        x_value = np.stack((-width, 0, -x_1, 0, x_arc, x_end, -width))
        # y-values
        y_value = np.stack((0, 0, h, y_m, y_arc, t, t))
        segment_list = ["line", "line", "arc", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate(np.append(-b / 2, 0))
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_interpolated()


@ureg_check_class("[length]", "[]", "[length]", "[length]", "[length]", None)
@inherit_docstrings
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

    def to_profile(self, width_default: pint.Quantity = Q_(3, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(3, "mm"))

        """
        t, beta, R, b, c, width = (
            self.t,
            self.beta,
            self.R,
            self.b,
            self.c,
            width_default,
        )

        # calculations:
        # From next point to circle center is the vector (x,y)
        x = R * np.cos(beta)
        y = R * np.sin(beta)
        # m = [0,c+R] circle center
        # => [-x,c+R-y] is the next point

        s = np.tan(beta) * (t - (c + R - y))

        # Scaling
        edge = np.append(x + s, 0).max()
        if width <= edge + Q_(1, "mm"):
            # adjustment of the width
            width = width + edge

        # x-values
        x_value = []
        # y-values
        y_value = []
        segment_list = []

        # bottom segment
        x_value = np.append(-width, 0)
        y_value = Q_([0, 0], "mm")
        segment_list.append("line")

        # root face
        if c != 0:
            x_value = np.append(x_value, 0)
            y_value = np.append(y_value, c)
            segment_list.append("line")

        # groove face arc (circle center)
        x_value = np.append(x_value, 0)
        y_value = np.append(y_value, c + R)

        # groove face arc
        x_value = np.append(x_value, -x)
        y_value = np.append(y_value, c + R - y)
        segment_list.append("arc")

        # groove face line
        x_value = np.append(x_value, -x - s)
        y_value = np.append(y_value, t)
        segment_list.append("line")

        # top segment
        x_value = np.append(x_value, -width)
        y_value = np.append(y_value, t)
        segment_list.append("line")

        shape = _helperfunction(segment_list, [x_value, y_value])

        shape = shape.translate(np.append(-b / 2, 0))
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_interpolated()


@ureg_check_class("[length]", "[]", "[length]", "[length]", None)
@inherit_docstrings
@dataclass
class HVGroove(IsoBaseGroove):
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))

        """
        t, beta, b, c, width = (
            self.t,
            self.beta,
            self.b,
            self.c,
            width_default,
        )

        # Calculations
        s = np.tan(beta) * (t - c)

        # Scaling
        edge = np.append(-s, 0).min()
        if width <= -edge + Q_(1, "mm"):
            # adjustment of the width
            width = width - edge

        x_value = np.append(-width, 0)
        y_value = Q_([0, 0], "mm")
        segment_list = ["line"]

        if c != 0:
            x_value = np.append(x_value, 0)
            y_value = np.append(y_value, c)
            segment_list.append("line")

        x_value = np.append(x_value, (-s, -width))
        y_value = np.append(y_value, (t, t))
        segment_list += ["line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])
        shape = shape.translate(np.append(-b / 2, 0))
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])
        shape_h = geo.Shape()
        arr = np.stack(
            (
                np.append(-width - (b / 2), 0),
                np.append(-b / 2, 0),
                np.append(-b / 2, t),
                np.append(-width - (b / 2), t),
            )
        )
        shape_h.add_line_segments(arr)

        return geo.Profile([shape_h, shape_r], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_from_profile()


@ureg_check_class("[length]", "[]", "[length]", "[length]", "[length]", None)
@inherit_docstrings
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

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))

        """
        t, beta, R, b, c, width = (
            self.t,
            self.beta,
            self.R,
            self.b,
            self.c,
            width_default,
        )

        # Calculations
        x = R * np.cos(beta)
        y = R * np.sin(beta)
        s = np.tan(beta) * (t - (c + R - y))

        # Scaling
        edge = np.append(x + s, 0).max()
        if width <= edge + Q_(1, "mm"):
            # adjustment of the width
            width = width + edge

        x_value = np.append(-width, 0)
        y_value = Q_([0, 0], "mm")
        segment_list = ["line"]

        if c != 0:
            x_value = np.append(x_value, 0)
            y_value = np.append(y_value, c)
            segment_list.append("line")

        x_value = np.append(x_value, (0, -x, -x - s, -width))
        y_value = np.append(y_value, (c + R, c + R - y, t, t))
        segment_list += ["arc", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])
        shape = shape.translate(np.append(-b / 2, 0))
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])
        shape_h = geo.Shape()
        arr = np.stack(
            (
                np.append(-width - (b / 2), 0),
                np.append(-b / 2, 0),
                np.append(-b / 2, t),
                np.append(-width - (b / 2), t),
            )
        )
        shape_h.add_line_segments(arr)

        return geo.Profile([shape_h, shape_r], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_interpolated()


@ureg_check_class("[length]", "[]", "[]", "[length]", None, None, "[length]", None)
@inherit_docstrings
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
        """Calculate missing values."""
        _set_default_heights(self)

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))

        """
        t, alpha_1, alpha_2, b, c, h1, h2, width = (
            self.t,
            self.alpha_1,
            self.alpha_2,
            self.b,
            self.c,
            self.h1,
            self.h2,
            width_default,
        )

        # Calculations
        s_upper = np.tan(alpha_1 / 2) * h1
        s_lower = np.tan(alpha_2 / 2) * h2

        # Scaling
        edge = np.stack((-s_upper, -s_lower, 0)).min()
        if width <= -edge + Q_(1, "mm"):
            # adjustment of the width
            width = width - edge

        x_value = np.stack((-width, -s_lower, 0))
        y_value = np.stack((0, 0, h2))
        segment_list = ["line", "line"]

        if c != 0:
            x_value = np.append(x_value, 0)
            y_value = np.append(y_value, h2 + c)
            segment_list.append("line")

        x_value = np.append(x_value, (-s_upper, -width))
        y_value = np.append(y_value, (t, t))
        segment_list += ["line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])
        shape = shape.translate(np.append(-b / 2, 0))
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_from_profile()


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
@inherit_docstrings
@dataclass
class DUGroove(IsoBaseGroove):
    # noinspection PyUnresolvedReferences
    """A DU-Groove.

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
        """Calculate missing values."""
        _set_default_heights(self)

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))

        """
        t, beta_1, beta_2, R, R2, b, c, h1, h2, width = (
            self.t,
            self.beta_1,
            self.beta_2,
            self.R,
            self.R2,
            self.b,
            self.c,
            self.h1,
            self.h2,
            width_default,
        )

        # Calculations
        x_upper = R * np.cos(beta_1)
        y_upper = R * np.sin(beta_1)
        s_upper = np.tan(beta_1) * (h1 - (R - y_upper))
        x_lower = R2 * np.cos(beta_2)
        y_lower = R2 * np.sin(beta_2)
        s_lower = np.tan(beta_2) * (h2 - (R2 - y_lower))

        # Scaling
        edge = np.stack((x_upper + s_upper, x_lower + s_lower, 0)).max()
        if width <= edge + Q_(1, "mm"):
            # adjustment of the width
            width = width + edge

        x_value = np.stack((-width, -(s_lower + x_lower), -x_lower, 0, 0))
        y_value = np.stack((0, 0, h2 - (R2 - y_lower), h2 - R2, h2))
        segment_list = ["line", "line", "arc"]

        if c != 0:
            x_value = np.append(x_value, 0)
            y_value = np.append(y_value, h1 + c)
            segment_list.append("line")

        x_value = np.append(x_value, (0, -x_upper, -(s_upper + x_upper), -width))
        y_value = np.append(y_value, (h2 + c + R, t - (h1 - (R - y_upper)), t, t))
        segment_list += ["arc", "line", "line"]

        shape = _helperfunction(segment_list, [x_value, y_value])
        shape = shape.translate(np.append(-b / 2, 0))
        # y-axis as mirror axis
        shape_r = shape.reflect_across_line([0, 0], [0, 1])

        return geo.Profile([shape, shape_r], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_interpolated()


@ureg_check_class("[length]", "[]", "[]", "[length]", None, None, "[length]", None)
@inherit_docstrings
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
        """Calculate missing values."""
        _set_default_heights(self)

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))

        """
        dv_groove = DVGroove(
            t=self.t,
            alpha_1=self.beta_1 * 2,
            alpha_2=self.beta_2 * 2,
            c=self.c,
            h1=self.h1,
            h2=self.h2,
            b=self.b,
            code_number=self.code_number,
        )
        dv_profile = dv_groove.to_profile(width_default)
        right_shape = dv_profile.shapes[1]
        left_shape = geo.Shape()
        arr = np.stack(
            (
                np.append(-width_default - (self.b / 2), 0),
                np.append(-self.b / 2, 0),
                np.append(-self.b / 2, self.t),
                np.append(-width_default - (self.b / 2), self.t),
            )
        )
        left_shape.add_line_segments(arr)

        return geo.Profile([left_shape, right_shape], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_from_profile()


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
@inherit_docstrings
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
        """Calculate missing values."""
        _set_default_heights(self)

    def to_profile(self, width_default: pint.Quantity = Q_(5, "mm")) -> geo.Profile:
        """Calculate a Profile.

        Parameters
        ----------
        width_default :
             pint.Quantity (Default value = Q_(5, "mm"))

        """
        du_profile = DUGroove(
            t=self.t,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            R=self.R,
            R2=self.R2,
            c=self.c,
            h1=self.h1,
            h2=self.h2,
            b=self.b,
            code_number=self.code_number,
        ).to_profile(width_default)
        right_shape = du_profile.shapes[1]
        left_shape = geo.Shape()
        arr = np.stack(
            (
                np.append(-width_default - (self.b / 2), 0),
                np.append(-self.b / 2, 0),
                np.append(-self.b / 2, self.t),
                np.append(-width_default - (self.b / 2), self.t),
            )
        )
        left_shape.add_line_segments(arr)

        return geo.Profile([left_shape, right_shape], units=_DEFAULT_LEN_UNIT)

    @property
    def cross_sect_area(self):  # noqa
        return self._compute_cross_sect_area_interpolated()


@ureg_check_class(
    "[length]",
    None,
    None,
    None,
    None,
    None,
)
@inherit_docstrings
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
            shape1 = geo.Shape()
            arr1 = np.stack(
                (
                    np.append(0, 0),
                    np.append(2 * width_default + self.t_1, 0),
                    np.append(2 * width_default + self.t_1, self.t_1),
                    np.append(0, self.t_1),
                    np.append(0, 0),
                )
            )
            shape1.add_line_segments(arr1)
            shape2 = geo.Shape()
            arr2 = np.stack(
                (
                    np.append(width_default, 0),
                    np.append(width_default + self.t_1, 0),
                    np.append(width_default + self.t_1, -width_default),
                    np.append(width_default, -width_default),
                    np.append(width_default, 0),
                )
            )
            shape2.add_line_segments(arr2)
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "3.1.1":
            t_1, t_2, alpha, b = self.t_1, self.t_2, self.alpha, self.b

            if width_default < t_1 + Q_(1, "mm"):
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
            arr1 = np.stack(
                (
                    np.append(t_1 + x_1, y_1),
                    np.append(t_1, 0),
                    np.append(t_1 + x_2, y_2),
                    np.append(t_1 + x_3, y_3),
                )
            )
            shape1.add_line_segments(arr1)
            shape2 = geo.Shape()
            arr2 = np.stack(
                (
                    np.append(width_default, -b),
                    np.append(0, -b),
                    np.append(0, -t_2 - b),
                    np.append(width_default, -t_2 - b),
                )
            )
            shape2.add_line_segments(arr2)
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "3.1.2":
            t_1, t_2, b = self.t_1, self.t_2, self.b
            shape1 = geo.Shape()
            arr1 = np.stack(
                (
                    np.append(0, 0),
                    np.append(width_default, 0),
                    np.append(width_default, t_1),
                    np.append(0, t_1),
                )
            )
            shape1.add_line_segments(arr1)
            shape2 = geo.Shape()
            arr2 = np.stack(
                (
                    np.append(0, -b),
                    np.append(2 * width_default, -b),
                    np.append(2 * width_default, -t_2 - b),
                    np.append(0, -t_2 - b),
                )
            )
            shape2.add_line_segments(arr2)
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "3.1.3" or self.code_number == "4.1.1":
            t_1, t_2, alpha, b = self.t_1, self.t_2, self.alpha, self.b

            x = np.sin(alpha + np.pi / 2) * b + b
            y = np.cos(alpha + np.pi / 2) * b

            x_1 = np.sin(alpha) * t_2 + x
            y_1 = np.cos(alpha) * t_2 + y

            x_2 = np.sin(alpha + np.pi / 2) * (b + width_default) + b
            y_2 = np.cos(alpha + np.pi / 2) * (b + width_default)

            x_3 = x_1 + x_2 - x
            y_3 = y_1 + y_2 - y

            shape1 = geo.Shape()
            arr1 = np.stack(
                (
                    np.append(-width_default, 0),
                    np.append(0, 0),
                    np.append(0, t_1),
                    np.append(-width_default, t_1),
                )
            )
            shape1.add_line_segments(arr1)
            shape2 = geo.Shape()
            arr2 = np.stack(
                (
                    np.append(x_3, y_3),
                    np.append(x_1, y_1),
                    np.append(x, y),
                    np.append(x_2, y_2),
                )
            )
            shape2.add_line_segments(arr2)
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "4.1.2":
            t_1, t_2, alpha, e = self.t_1, self.t_2, self.alpha, self.e

            x_1 = np.sin(alpha) * e
            y_1 = np.cos(alpha) * e

            x_2 = np.sin(alpha + np.pi) * (t_2 - e)
            y_2 = np.cos(alpha + np.pi) * (t_2 - e)

            x_3 = x_2 + np.sin(alpha + np.pi / 2) * width_default
            y_3 = y_2 + np.cos(alpha + np.pi / 2) * width_default

            x_4 = x_1 + np.sin(alpha + np.pi / 2) * width_default
            y_4 = y_1 + np.cos(alpha + np.pi / 2) * width_default

            shape1 = geo.Shape()
            arr1 = np.stack(
                (
                    np.append(-width_default, 0),
                    np.append(0, 0),
                    np.append(0, t_1),
                    np.append(-width_default, t_1),
                )
            )
            shape1.add_line_segments(arr1)
            shape2 = geo.Shape()
            arr2 = np.stack(
                (
                    np.append(x_4, y_4),
                    np.append(x_1, y_1),
                    np.append(x_2, y_2),
                    np.append(x_3, y_3),
                )
            )
            shape2.add_line_segments(arr2)
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        elif self.code_number == "4.1.3":
            t_1, t_2, b = self.t_1, self.t_2, self.b
            shape1 = geo.Shape()
            arr1 = np.stack(
                (
                    np.append(0, width_default),
                    np.append(0, 0),
                    np.append(t_1, 0),
                    np.append(t_1, width_default),
                )
            )
            shape1.add_line_segments(arr1)
            shape2 = geo.Shape()
            arr2 = np.stack(
                (
                    np.append(-width_default, -b),
                    np.append(t_1 + width_default, -b),
                    np.append(t_1 + width_default, -t_2 - b),
                    np.append(-width_default, -t_2 - b),
                    np.append(-width_default, -b),
                )
            )
            shape2.add_line_segments(arr2)
            return geo.Profile([shape1, shape2], units=_DEFAULT_LEN_UNIT)
        else:
            raise ValueError(
                "Wrong code_number. The Code Number has to be"
                " one of the following strings: "
                '"1.12", "1.13", "2.12", "3.1.1", "3.1.2",'
                ' "3.1.3", "4.1.1", "4.1.2", "4.1.3"'
            )

    @property
    def cross_sect_area(self):  # noqa
        raise NotImplementedError("Cannot determine FFGroove cross sectional area")


def _helperfunction(segment, array) -> geo.Shape:
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
                np.vstack(
                    [array[0][counter : counter + 2], array[1][counter : counter + 2]]
                )
            )
            segment_list.append(seg)
            counter += 1
        if elem == "arc":
            arr0 = array[0][[counter, counter + 2, counter + 1]]
            arr1 = array[1][[counter, counter + 2, counter + 1]]
            arr = np.vstack((arr0, arr1))
            seg = geo.ArcSegment(arr, False)
            segment_list.append(seg)
            counter += 2

    return geo.Shape(segment_list)


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
) -> IsoBaseGroove:
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

    Raises
    ------
    ValueError
        When passing negative parameter values.

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
    args = {k: Q_(_loc[v]) for k, v in _mapping.items() if _loc[v] is not None}
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
        groove_type="FFGroove",
        workpiece_thickness=Q_(5, "mm"),
        code_number="1.12",
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
