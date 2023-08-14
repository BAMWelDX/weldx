"""Tests if quality standards are working as expected."""

import asdf
import fs
import pytest

from weldx import WeldxFile
from weldx.config import QualityStandard, add_quality_standard, enable_quality_standard
from weldx.measurement import MeasurementEquipment

manifest_file = """
id: http://weldx.bam.de/weldx/standards/manifests/test_standard-1.0.0
extension_uri: http://weldx.bam.de/weldx/standards/test_standard-1.0.0
asdf_standard_requirement: 1.0.0

tags:
  - uri: "asdf://weldx.bam.de/weldx/schemas/equipment/measurement_equipment-0.1.0"
    file: "test_schema-0.1.0"
"""

schema_file = """
%YAML 1.1
---
$schema: "http://stsci.edu/schemas/yaml-schema/draft-01"
id: "asdf://weldx.bam.de/weldx/schemas/equipment/measurement_equipment-0.1.0"

type: object
properties:
  name:
    type: string
  sources:
    type: array
    items:
      tag: "asdf://weldx.bam.de/weldx/tags/measurement/source-0.1.*"
  transformations:
    type: array
    items:
      tag: "asdf://weldx.bam.de/weldx/tags/measurement/signal_transformation-0.1.*"
  wx_metadata:
    type: object
    properties:
      required_field_for_test:
        type: number
    required: [required_field_for_test]


propertyOrder: [name, sources, transformations]
required: [name, wx_metadata]

flowStyle: block
...
"""


def test_quality_standards():
    # create file structure
    base_dir = "resources/some_organization"
    manifest_dir = f"{base_dir}/manifests"
    schema_dir = f"{base_dir}/schemas"

    vfs = fs.open_fs("mem://")
    vfs.makedirs(manifest_dir)
    vfs.makedirs(schema_dir)
    with vfs.open(f"{manifest_dir}/test_standard-1.0.0.yaml", "w") as file:
        file.write(manifest_file)
    with vfs.open(f"{schema_dir}/test_schema-0.1.0.yaml", "w") as file:
        file.write(schema_file)
    # print(vfs.tree())  # skipcq: PY-W0069

    # create and enable quality standard
    qs = QualityStandard(vfs.opendir(base_dir))
    add_quality_standard(qs)
    enable_quality_standard("test_standard")

    # run tests
    eq = MeasurementEquipment("some_equipment")
    with pytest.raises(asdf.exceptions.ValidationError):
        WeldxFile(tree={"equipment": eq}, mode="rw")

    eq.wx_metadata = {"required_field_for_test": 1234}
    WeldxFile(tree={"equipment": eq}, mode="rw")
