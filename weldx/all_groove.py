"""provides the calculation of all Groove-Types."""

from weldx import Q_
import numpy as np

import weldx.geometry as geo
from weldx.asdf.tags.weldx.core.groove import VGroove, UGroove, UVGroove, IGroove


def groove_to_profile(groove):
    """
    Calculate a Groove type.

    :param groove: groove type.
    :return: geo.Profile
    """
    if isinstance(groove, VGroove):
        return single_vgroovebuttweld(**groove.__dict__)

    if isinstance(groove, UGroove):
        return single_ugroovebuttweld(**groove.__dict__)

    if isinstance(groove, IGroove):
        return i_groove(**groove.__dict__)

    if isinstance(groove, UVGroove):
        return uv_groove(**groove.__dict__)


def single_vgroovebuttweld(t, alpha, b, c, code_number=None, width_default=Q_(2, "mm")):
    """
    Calculate a Single-V Groove Butt Weld.

    :param t: the workpiece thickness, as Pint unit
    :param alpha: the groove angle, as Pint unit
    :param b: the root opening, as Pint unit
    :param c: the root face, as Pint unit
    :param code_number: unused param
    :param width_default: the width of the workpiece, as Pint unit
    :return: geo.Profile
    """
    t = t.to("mm").magnitude
    alpha = alpha.to("rad").magnitude
    b = b.to("mm").magnitude
    c = c.to("mm").magnitude
    width = width_default.to("mm").magnitude

    # calculations:
    s = np.tan(alpha / 2) * (t - c)

    # Rand breite
    edge = np.min([-s, 0])
    if width <= -edge + 1:
        # zu Kleine Breite für die Naht wird angepasst
        width = width - edge

    x_value = []
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

    # groove face
    x_value.append(-s)
    y_value.append(t)
    segment_list.append("line")

    # top segment
    x_value.append(-width)
    y_value.append(t)
    segment_list.append("line")

    shape = _helperfunction(segment_list, [x_value, y_value])

    shape = shape.translate([-b / 2, 0])
    # y Achse als Spiegelachse
    shape_r = shape.reflect_across_line([0, 0], [0, 1])

    return geo.Profile([shape, shape_r])


def single_ugroovebuttweld(t, beta, R, b, c, code_number=None,
                           width_default=Q_(3, "mm")):
    """
    Calculate a Single-U Groove Butt Weld.

    :param t: the workpiece thickness, as Pint unit
    :param beta: the bevel angle, as Pint unit
    :param R: radius, as Pint unit
    :param b: the root opening, as Pint unit
    :param c: the root face, as Pint unit
    :param code_number: unused param
    :param width_default: the width of the workpiece, as Pint unit
    :return: geo.Profile
    """
    t = t.to("mm").magnitude
    beta = beta.to("rad").magnitude
    R = R.to("mm").magnitude
    b = b.to("mm").magnitude
    c = c.to("mm").magnitude
    width = width_default.to("mm").magnitude

    # calculations:
    # vom nächsten Punkt zum Kreismittelpunkt ist der Vektor (x,y)
    x = R * np.cos(beta)
    y = R * np.sin(beta)
    # m = [0,c+R] Kreismittelpunkt
    # => [-x,c+R-y] ist der nächste Punkt

    s = np.tan(beta) * (t - (c + R - y))

    # Rand breite
    edge = np.max([x + s, 0])
    if width <= edge + 1:
        # zu Kleine Breite für die Naht wird angepasst
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

    # groove face arc kreismittelpunkt
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
    # y Achse als Spiegelachse
    shape_r = shape.reflect_across_line([0, 0], [0, 1])

    return geo.Profile([shape, shape_r])


def uv_groove(t, alpha, beta, R, b, h, code_number=None, width_default=Q_(2, "mm")):
    """
    Calculate a U-Groove with V-Root.

    :param t: the workpiece thickness, as Pint unit
    :param alpha: the groove angle, as Pint unit
    :param beta: the bevel angle, as Pint unit
    :param R: radius, as Pint unit
    :param b: the root opening, as Pint unit
    :param h: the root face, as Pint unit
    :param code_number: unused param
    :param width_default: the width of the workpiece, as Pint unit
    :return: geo.Profile
    """
    t = t.to("mm").magnitude
    alpha = alpha.to("rad").magnitude
    beta = beta.to("rad").magnitude
    R = R.to("mm").magnitude
    b = b.to("mm").magnitude
    h = h.to("mm").magnitude
    width = width_default.to("mm").magnitude

    # calculations:
    x_1 = np.tan(alpha/2) * h
    # Kreismittelpunkt [0, y_m]
    y_circle = np.sqrt(R**2 - x_1**2)
    y_m = h + y_circle
    # vom nächsten Punkt zum Kreismittelpunkt ist der Vektor (x,y)
    x = R * np.cos(beta)
    y = R * np.sin(beta)
    x_arc = - x
    y_arc = y_m - y
    # x abschnit der oberen kante berechnen
    x_end = x_arc - (t - y_arc) * np.tan(beta)

    # Rand breite
    edge = np.max([-x_end, 0])
    if width <= edge + 1:
        # zu Kleine Breite für die Naht wird angepasst
        width = width + edge

    # x-values
    x_value = [-width, 0, -x_1, 0, x_arc, x_end, -width]
    # y-values
    y_value = [0, 0, h, y_m, y_arc, t, t]
    segment_list = ["line", "line", "arc", "line", "line"]

    shape = _helperfunction(segment_list, [x_value, y_value])

    shape = shape.translate([-b / 2, 0])
    # y Achse als Spiegelachse
    shape_r = shape.reflect_across_line([0, 0], [0, 1])

    return geo.Profile([shape, shape_r])


def i_groove(t, b, code_number=None, width_default=Q_(5, "mm")):
    """
    Calculate a I-Groove Butt Weld.

    :param t: the workpiece thickness, as Pint unit
    :param b: the root opening, as Pint unit
    :param code_number: unused param
    :param width_default: the width of the workpiece, as Pint unit
    :return: geo.Profile
    """
    t = t.to("mm").magnitude
    b = b.to("mm").magnitude
    width = width_default.to("mm").magnitude

    # x-values
    x_value = [-width, 0, 0, -width]
    # y-values
    y_value = [0, 0, t, t]
    segment_list = ["line", "line", "line"]

    shape = _helperfunction(segment_list, [x_value, y_value])

    shape = shape.translate([-b / 2, 0])
    # y Achse als Spiegelachse
    shape_r = shape.reflect_across_line([0, 0], [0, 1])

    return geo.Profile([shape, shape_r])


def _helperfunction(liste, array):
    """
    Calculate a shape from input.
    Input liste der aufeinanderfolgenden Segmente als strings.
    Input array der Punkte ich richtiger Reichenfolge. BSP:
    array = [[x-werte], [y-werte]]

    :param liste: list of String, segment names ("line", "arc")
    :param array: array of 2 array,
        first array are x-values
        second array are y-values
    :return: geo.Shape
    """
    segment_list = []
    counter = 0
    for elem in liste:
        if elem == "line":
            seg = geo.LineSegment(
                [array[0][counter: counter + 2],
                 array[1][counter: counter + 2]]
            )
            segment_list.append(seg)
            counter += 1
        if elem == "arc":
            arr0 = [
                # anfang
                array[0][counter],
                # ende
                array[0][counter + 2],
                # mittelpunkt
                array[0][counter + 1],
            ]
            arr1 = [
                # anfang
                array[1][counter],
                # ende
                array[1][counter + 2],
                # mittelpunkt
                array[1][counter + 1],
            ]
            seg = geo.ArcSegment([arr0, arr1], False)
            segment_list.append(seg)
            counter += 2

    return geo.Shape(segment_list)
