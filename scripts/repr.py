import weldx.geometry as geo
import weldx.transformations as tf
from weldx import Q_
from weldx.asdf.tags.weldx.core.iso_groove import get_groove

linesegment = geo.LineSegment([[0, 0], [2, 1]])
print(linesegment.__repr__())
print()

arcsegment = geo.ArcSegment([[0, 1, 0], [1, 0, 0]], True)
print(arcsegment.__repr__())
print()

shape = geo.Shape([linesegment, arcsegment])
print(shape.__repr__())
print()

profile = geo.Profile(shape)
print(profile.__repr__())
print()

LHTS = geo.LinearHorizontalTraceSegment(5)
print(LHTS.__repr__())
print()

RHTS = geo.RadialHorizontalTraceSegment(3, 90)
print(RHTS.__repr__())
print()


# how to get this one running?
# varProfile = geo.VariableProfile([profile], [0], [1])
# print(varProfile.__repr__())
# print()

# LCS
LCS = tf.LocalCoordinateSystem(coordinates=[2, 4, -1])
print(LCS.__repr__())
print()

# create a linear trace segment
trace_segment = geo.LinearHorizontalTraceSegment(300)
trace = geo.Trace(trace_segment)
print(trace.__repr__())
print()

groove = get_groove(
    groove_type="VGroove",
    workpiece_thickness=Q_(0.5, "cm"),
    groove_angle=Q_(50, "deg"),
    root_face=Q_(1, "mm"),
    root_gap=Q_(1, "mm"),
)

# create 3d workpiece geometry from the groove profile and trace objects
geometry = geo.Geometry(groove.to_profile(width_default=Q_(4, "mm")), trace)

# Geometry - trace is empty -> no repr?
# geometry = geo.Geometry(profile, trace)
print(geometry.__repr__())
print()

# crete a new coordinate system manager with default base coordinate system
csm = tf.CoordinateSystemManager("base")

# add the workpiece coordinate system
csm.add_coordinate_system("workpiece", "base", trace.coordinate_system)

# CSM - need to lookup how to initialize this in the tutorial...
# CSM = tf.CoordinateSystemManager()
print(csm.__repr__())
print()
