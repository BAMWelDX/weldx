"""Generate example implementation of AWS "A Welding Data Dictionary."""

from weldx.asdf.utils import create_asdf_dataclass

# design -----------------------------------------------------------------

"""
create_asdf_dataclass(
    asdf_name="aws/NAME",
    asdf_version="1.0.0",
    class_name="PythonClass",
    schema_description=""DESCRIPTION",
    properties=[
        "property1",
        "property2",
        "property3",
        "property4",
        "property5",
        "property6",
        "property7",
        "property8",
    ],
    required=[
        "property1",
        "property2",
        "property3",
        "property4",
        "property5",
        "property6",
        "property7",
        "property8",
    ],
    property_order=[
        "property1",
        "property2",
        "property3",
        "property4",
        "property5",
        "property6",
        "property7",
        "property8",
    ],
    property_types=[
        "property1",
        "property2",
        "property3",
        "property4",
        "property5",
        "property6",
        "property7",
        "property8",
    ],
    description=[
        "property1",
        "property2",
        "property3",
        "property4",
        "property5",
        "property6",
        "property7",
        "property8",
    ],
)
"""

# design -----------------------------------------------------------------

# design/weldment -----------------------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/design/weldment",
    asdf_version="1.0.0",
    class_name="Weldment",
    schema_description="An assembly whose component parts are joined by welding",
    properties=["sub_assembly"],
    required=["sub_assembly"],
    property_order=["sub_assembly"],
    property_types=["List[SubAssembly]"],
    description=["A list of assemblies composing the weldment"],
)

# design/sub_assembly -----------------------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/design/sub_assembly",
    asdf_version="1.0.0",
    class_name="SubAssembly",
    schema_description="Two workpieces joined by a weld. Identifies simple weldments that can be joined to form a largermore complicated weldment.",
    properties=["workpiece", "connection"],
    required=["workpiece", "connection"],
    property_order=["workpiece", "connection"],
    property_types=["List[Workpiece]", "Connection"],
    description=[
        "Describes a piece of metal, its materials and geometries.",
        "The joint formed by edge preparation of two workpieces and the weld that joins the workpieces together.",
    ],
)

# design/workpiece -----------------------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/design/workpiece",
    asdf_version="1.0.0",
    class_name="Workpiece",
    schema_description="""This element describes a piece of metal, its material, geometric properties and boundaries, including its edge shape for the welded joint. A workpiece can be a single piece of plate, or a previously assembled weldment. The scope of this data is beyond this welding data dictionary. This element represents an interface with CAD applications.""",
    properties=["geometry"],
    required=["geometry"],
    property_order=["geometry"],
    property_types=["str"],
    description=["The workpiece geometry."],
)


# design/connection -----------------------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/design/connection",
    asdf_version="1.0.0",
    class_name="Connection",
    schema_description="The joint formed by edge preparation of two workpieces and the weld that joins the workpieces together.",
    properties=["joint_type", "weld_type", "joint_penetration", "weld_details"],
    required=["joint_type", "weld_type", "joint_penetration", "weld_details"],
    property_order=["joint_type", "weld_type", "joint_penetration", "weld_details"],
    property_types=["str", "str", "JointPenetration", "WeldDetails"],
    description=[
        "A weld joint classification based the relative orientation of the members being joined. TODO ENUM",
        "A region of coalescence of materials produced by heating or pressure, that joins two pieces of metal. TODO ENUM",
        "joint_penetration",
        "weld_details",
    ],
)

# design/joint_penetration ------------------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/design/joint_penetration",
    asdf_version="1.0.0",
    class_name="JointPenetration",
    schema_description="Various dimensions of penetration of the weld into the base materials",
    properties=[
        "complete_or_partial",
        "root_penetration",
        "groove_weld_size",
        "incomplete_joint_penetration",
        "weld_size",
        "weld_size_E1",
        "weld_size_E2",
        "depth_of_fusion",
    ],
    required=["complete_or_partial", "units", "root_penetration"],
    property_order=[
        "complete_or_partial",
        "root_penetration",
        "groove_weld_size",
        "incomplete_joint_penetration",
        "weld_size",
        "weld_size_E1",
        "weld_size_E2",
        "depth_of_fusion",
    ],
    property_types=[
        "str",
        "pint.Quantity",
        "float",
        "float",
        "float",
        "float",
        "float",
        "float",
    ],
    description=[
        "The weld design calls for partial or complete penetration. TODO ENUM",
        "The distance the weld metal extends into the root joint",
        "The joint penetration of a groove weld",
        'See AWS A3.0 definition, and Figure 26 - "Joint Penetration...", Figure 30 - Fusion Welds',
        'See AWS A3.0 definition, and Figure 26 - "Joint Penetration...", Figure 30 - Fusion Welds',
        'See AWS A3.0 definition, and Figure 26 - "Joint Penetration...", Figure 30 - Fusion Welds',
        'See AWS A3.0 definition, and Figure 26 - "Joint Penetration...", Figure 30 - Fusion Welds',
        "The distance that fusion extends into the base metal or the previous bead from the surface melted during welding",
    ],
)

# design/weld_details -----------------------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/design/weld_details",
    asdf_version="1.0.0",
    class_name="WeldDetails",
    schema_description="Information describing the joint and the geometric parameters of the designed weld.",
    properties=["joint_design", "weld_sizes", "number_of_passes"],
    required=["joint_design", "weld_sizes", "number_of_passes"],
    property_order=["joint_design", "weld_sizes", "number_of_passes"],
    property_types=["str", "pint.Quantity", "int"],
    description=[
        'AWS 3.0, p. 19 : "Detailed Description of mating configuration of two components, sufficient information for shop preparation of the components." TODO CLASS',
        "Linear dimensions of cross sectional weld features. Sizes are of design and actual features. TODO CLASS",
        "Number of times needed to deposit weld metal to complete the weld.",
    ],
)

# design/base_metal -----------------------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/design/base_metal",
    asdf_version="1.0.0",
    class_name="BaseMetal",
    schema_description="The material that is being joined using a weld.",
    properties=[
        "common_name",
        "m_number",
        "group_number",
        "product_form",
        "thickness",
        "diameter",
        "specification_number",
        "specification_version",
        "specification_organization",
        "UNS_number",
        "CAS_number",
        "heat_lot_identification",
        "composition",
        "manufacturing_history",
        "service_history",
        "applied_coating_specification",
    ],
    required=["common_name", "product_form", "thickness"],
    property_order=[
        "common_name",
        "m_number",
        "group_number",
        "product_form",
        "thickness",
        "diameter",
        "specification_number",
        "specification_version",
        "specification_organization",
        "UNS_number",
        "CAS_number",
        "heat_lot_identification",
        "composition",
        "manufacturing_history",
        "service_history",
        "applied_coating_specification",
    ],
    property_types=[
        "str",
        "str",
        "str",
        "str",
        "float",
        "float",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
    ],
    description=[
        "The trade name, a name used without all the designations of the formal specification.",
        "A designation used to group base metals for procedure and performance qualifications.",
        "A classification system for metal by material properties.",
        "The form of the workpieces to be joined. TODO: ENUM",
        "Plate or sheet thickness, if tube, specifies wall thickness.",
        "Outside diameter, only used if material is tube.",
        "The standard designation of the formal material classification.",
        "Version of the specification used.",
        "The organization responsible for generating the specification.",
        "Unified Numbering System for Metals and Alloys, managed by ASTM and SAE.",
        "Chemical Abstracts Service Registry Number, a unique identifier for substances issued by the Chemical Abstracts Service.",
        "A unique identifier issued by a materials manufacturer assigned to manufacturing batches.",
        "Detailed chemical composition, by elements. This Type needs expansion.",
        "Mechanical manufacturing methods used to produce the welded material. TODO: ENUM",
        "The mechanical forming and heat treatment methods used to produce the stock material.",
        "Standard designation for the class of coating.",
    ],
)


# process -------------------------------------------------------------

# process/arc_welding_process --------------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/process/arc_welding_process",
    asdf_version="1.0.0",
    class_name="ArcWeldingProcess",
    schema_description="See AWS 3.0, Figure 54A - Master Chart of Welding and Allied Processes.",
    properties=["name"],
    required=["name"],
    property_order=["name"],
    property_types=["str"],
    description=["Process name and abbreviation . TODO ENUM"],
)

# process/shielding_gas_for_procedure -------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/process/shielding_gas_for_procedure",
    asdf_version="1.0.0",
    class_name="ShieldingGasForProcedure",
    schema_description="Description of applicable gas composition and flowrates, including torch gas shielding, backing gas, and trailing gas.",
    properties=[
        "use_torch_shielding_gas",
        "torch_shielding_gas",
        "torch_shielding_gas_flowrate",
        "use_backing_gas",
        "backing_gas",
        "backing_gas_flowrate",
        "use_trailing_gas",
        "trailing_shielding_gas",
        "trailing_shielding_gas_flowrate",
    ],
    required=[
        "use_torch_shielding_gas",
        "torch_shielding_gas",
        "torch_shielding_gas_flowrate",
    ],
    property_order=[
        "use_torch_shielding_gas",
        "torch_shielding_gas",
        "torch_shielding_gas_flowrate",
        "use_backing_gas",
        "backing_gas",
        "backing_gas_flowrate",
        "use_trailing_gas",
        "trailing_shielding_gas",
        "trailing_shielding_gas_flowrate",
    ],
    property_types=[
        "bool",
        "ShieldingGasType",
        "pint.Quantity",
        "bool",
        "ShieldingGasType",
        "pint.Quantity",
        "bool",
        "ShieldingGasType",
        "pint.Quantity",
    ],
    description=[
        "Torch shielding gas is/is not required or specified.",
        "Composition of shielding gas expelled from a nozzle in the welding torch.",
        "Flow rate of shielding gas required or specified.",
        "Backing gas is/is not required or specified.",
        "Specification of the component gases of the mixture.",
        "Flowrate of backing gas.",
        "Trailing shielding gas is/is not required or specified.",
        "Composition or identification of trailing gas.",
        "Flowrate of trailing gas during welding.",
    ],
)

# process/gas_component -------------------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/process/gas_component",
    asdf_version="1.0.0",
    class_name="GasComponent",
    schema_description="A single gas element of a mixture and its percentage of the mixture by weight.",
    properties=["gas_chemical_name", "gas_percentage"],
    required=["gas_chemical_name", "gas_percentage"],
    property_order=["gas_chemical_name", "gas_percentage"],
    property_types=["str", "pint.Quantity"],
    description=[
        "Name of a single element or compound of gas. TODO ENUM",
        "Percentage by weight this gas occupies of the total gas mixture.",
    ],
)

# process/shielding_gas_type -------------------------------------------------
create_asdf_dataclass(
    asdf_name="aws/process/shielding_gas_type",
    asdf_version="1.0.0",
    class_name="ShieldingGasType",
    schema_description="Description of a gas or gas mixture used for shielding in arc welding.",
    properties=["gas_component", "common_name", "designation"],
    required=["gas_component", "common_name"],
    property_order=["gas_component", "common_name", "designation"],
    property_types=["List[GasComponent]", "str", "str"],
    description=[
        "A single gas element.",
        "Trade name for the gas mixture.",
        "Specification according to AWS classification by chemical composition of the gas mixture.",
    ],
)
