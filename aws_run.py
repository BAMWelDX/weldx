import asdf

from weldx.asdf.extension import WeldxExtension
from weldx.asdf.tags.weldx.aws.process.gas_component import GasComponent
from weldx.asdf.tags.weldx.aws.process.shielding_gas_type import ShieldingGasType

comp1 = GasComponent("Argon", 82.0)
comp2 = GasComponent("Carbon Dioxide", 18.0)

type1 = ShieldingGasType(
    gas_component=[comp1, comp2], common_name="NAME NEU", designation="ASD"
)

filename = "aws_demo.asdf"
tree = dict(gas_type=type1)

# Write the data to a new file
with asdf.AsdfFile(
    tree, extensions=[WeldxExtension()], ignore_version_mismatch=False
) as ff:
    ff.write_to(filename, all_array_storage="inline")

# read back data from ASDF file
with asdf.open(filename, copy_arrays=True, extensions=[WeldxExtension()]) as af:
    data = af.tree
    print(data["gas_type"])
