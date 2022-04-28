from weldx.asdf.types import WeldxConverter
from weldx.constants import _DEFAULT_LEN_UNIT, Q_
from weldx.core import TimeSeries
from weldx.tags.core.common_types import Variable
from weldx.transformations import LocalCoordinateSystem


class LocalCoordinateSystemConverter(WeldxConverter):
    """Serialization class for weldx.transformations.LocalCoordinateSystem"""

    tags = [
        "asdf://weldx.bam.de/weldx/tags/"
        "core/transformations/local_coordinate_system-0.1.*"
    ]
    types = [LocalCoordinateSystem]

    def to_yaml_tree(self, obj: LocalCoordinateSystem, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = {}

        orientations = None
        if not obj.is_unity_rotation:
            orientations = Variable(
                "orientations", obj.orientation.dims, obj.orientation.data
            )
            # TODO: restore inlining
            # if "time" not in node.orientation.coords:
            #     ctx.set_array_storage(orientations.data, "inline")
            tree["orientations"] = orientations

        coordinates = None
        if isinstance(obj.coordinates, TimeSeries):
            tree["coordinates"] = obj.coordinates
        elif not obj.is_unity_translation:
            coordinates = Variable(
                "coordinates", obj.coordinates.dims, obj.coordinates.data
            )
            # TODO: restore inlining
            # if "time" not in node.coordinates.coords:
            #     if isinstance(coordinates.data, pint.Quantity):
            #         ctx.set_array_storage(coordinates.data.magnitude, "inline")
            #     else:
            #         ctx.set_array_storage(coordinates.data, "inline")
            tree["coordinates"] = coordinates

        if "time" in obj.dataset.coords:
            tree["time"] = obj.time.as_timedelta_index()

        if obj.reference_time is not None:
            tree["reference_time"] = obj.reference_time

        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        orientations = node.get("orientations")
        if orientations is not None:
            orientations = orientations.data

        coordinates = node.get("coordinates")
        if coordinates is not None and not isinstance(coordinates, TimeSeries):
            coordinates = node["coordinates"].data

            if (
                tag == "asdf://weldx.bam.de/weldx/tags/core/transformations"
                "/local_coordinate_system-0.1.0"
            ):  # legacy
                coordinates = Q_(coordinates, _DEFAULT_LEN_UNIT)

        return LocalCoordinateSystem(
            orientation=orientations,
            coordinates=coordinates,
            time=node.get("time"),
            time_ref=node.get("reference_time"),
        )
