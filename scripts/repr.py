import weldx.geometry as geo
import weldx.transformations as tf

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

# does not show anything - Coordinate system missing?
trace = geo.Trace([linesegment, arcsegment], LCS)
print(trace.__repr__())
print()

# Geometry - trace is empty -> no repr?
geometry = geo.Geometry(profile, trace)
print(geometry.__repr__())
print()

# CSM - need to lookup how to initialize this in the tutorial...
CSM = tf.CoordinateSystemManager()
print(CSM.__repr__())
print()
