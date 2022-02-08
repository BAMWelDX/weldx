import matplotlib.pyplot as plt
import numpy as np
import sympy
from xarray import DataArray

from weldx import (
    Q_,
    U_,
    GenericSeries,
    LinearHorizontalTraceSegment,
    LocalCoordinateSystem,
    Trace,
)
from weldx.core import SpatialSeries
from weldx.geometry import RadialHorizontalTraceSegment


class SDTraceSegment:
    def __init__(self, series):
        self._series = series

    def _get_squared_derivative(self, i):
        me = self._series.data
        exp = me.expression
        # todo unit stripped -> how to proceed? how to cast all length units to mm?
        subs = [(k, v[i].data.to_base_units().m) for k, v in me.parameters.items()]
        return exp.subs(subs).diff("s") ** 2

    @property
    def length(self) -> float:

        der_sq = [self._get_squared_derivative(i) for i in range(3)]
        expr = sympy.sqrt(der_sq[0] + der_sq[1] + der_sq[2])
        mag = float(sympy.integrate(expr, ("s", 0, 1)).evalf())
        print("ohoh")
        return Q_(mag, Q_(1, "mm").to_base_units().u).to("mm")

    def local_coordinate_system(self, position: float) -> LocalCoordinateSystem:
        coords = self._series.evaluate(s=position).data.transpose()[0]
        return LocalCoordinateSystem(coordinates=coords)


expr = "a*s**2 + b*s + c"
params = dict(
    a=DataArray(Q_([0, 0, 1], "mm"), dims=["c"], coords=dict(c=["x", "y", "z"])),
    b=DataArray(Q_([1, 0, 0], "mm"), dims=["c"], coords=dict(c=["x", "y", "z"])),
    c=DataArray(Q_([0, 0, 0], "mm"), dims=["c"], coords=dict(c=["x", "y", "z"])),
)
series = SpatialSeries(expr, parameters=params)

segment = SDTraceSegment(series)
segment = LinearHorizontalTraceSegment("10mm")
segment2 = RadialHorizontalTraceSegment("1mm", Q_(np.pi, "rad"))
print(segment.length)
trace = Trace([segment, segment2])
print(trace.length)
trace.plot(Q_(0.1, "mm"))
plt.show()


# todo : check s=0 -> [0,0,0]
