"""single_pass_weld schema."""
import sys
from io import BytesIO
from typing import Optional, Tuple


def single_pass_weld_example(
    out_file: str = "single_pass_weld_example.asdf",
) -> Optional[Tuple[BytesIO, dict]]:
    """Create ASDF file containing all required fields of the single_pass_weld schema.

    Parameters
    ----------
    out_file :
        destination file, if None returns a BytesIO buffer.

    Returns
    -------
    buff, tree
        When writing to memory, return the buffer and the tree (as dictionary).

    """
    # Imports
    import asdf
    import numpy as np
    import pandas as pd
    from asdf.tags.core import Software

    # importing the weldx package with prevalent default abbreviations
    import weldx
    import weldx.geometry as geo
    import weldx.measurement as msm
    import weldx.util as ut
    from weldx import Q_, GmawProcess
    from weldx import LocalCoordinateSystem as lcs
    from weldx import TimeSeries, WXRotation, get_groove
    from weldx.asdf.tags.weldx.aws.process.gas_component import GasComponent
    from weldx.asdf.tags.weldx.aws.process.shielding_gas_for_procedure import (
        ShieldingGasForProcedure,
    )
    from weldx.asdf.tags.weldx.aws.process.shielding_gas_type import ShieldingGasType
    from weldx.asdf.util import get_schema_path, write_buffer
    from weldx.core import MathematicalExpression

    # Timestamp
    reference_timestamp = pd.Timestamp("2020-11-09 12:00:00")

    # Geometry
    # groove + trace = geometry
    groove = get_groove(
        groove_type="VGroove",
        workpiece_thickness=Q_(5, "mm"),
        groove_angle=Q_(50, "deg"),
        root_face=Q_(1, "mm"),
        root_gap=Q_(1, "mm"),
    )

    # define the weld seam length in mm
    seam_length = Q_(300, "mm")

    # create a linear trace segment a the complete weld seam trace
    trace_segment = geo.LinearHorizontalTraceSegment(seam_length)
    trace = geo.Trace(trace_segment)

    geometry = dict(groove_shape=groove, seam_length=seam_length)

    base_metal = dict(common_name="S355J2+N", standard="DIN EN 10225-2:2011")

    workpiece = dict(base_metal=base_metal, geometry=geometry)

    # Setup the Coordinate System Manager (CSM)
    # crete a new coordinate system manager with default base coordinate system
    csm = weldx.transformations.CoordinateSystemManager("base")

    # add the workpiece coordinate system
    csm.add_cs(
        coordinate_system_name="workpiece",
        reference_system_name="base",
        lcs=trace.coordinate_system,
    )

    tcp_start_point = Q_([5.0, 0.0, 2.0], "mm")
    tcp_end_point = Q_([-5.0, 0.0, 2.0], "mm") + np.append(
        seam_length, Q_([0, 0], "mm")
    )

    v_weld = Q_(10, "mm/s")
    s_weld = (tcp_end_point - tcp_start_point)[0]  # length of the weld
    t_weld = s_weld / v_weld

    t_start = pd.Timedelta("0s")
    t_end = pd.Timedelta(str(t_weld.to_base_units()))

    rot = WXRotation.from_euler(seq="x", angles=180, degrees=True)

    coords = [tcp_start_point.magnitude, tcp_end_point.magnitude]

    tcp_wire = lcs(coordinates=coords, orientation=rot, time=[t_start, t_end])

    # add the workpiece coordinate system
    csm.add_cs(
        coordinate_system_name="tcp_wire",
        reference_system_name="workpiece",
        lcs=tcp_wire,
    )

    tcp_contact = lcs(coordinates=[0, 0, -10])

    # add the workpiece coordinate system
    csm.add_cs(
        coordinate_system_name="tcp_contact",
        reference_system_name="tcp_wire",
        lcs=tcp_contact,
    )

    TCP_reference = csm.get_cs("tcp_contact", "workpiece")

    # Measurements
    # time
    time = pd.timedelta_range(start="0s", end="10s", freq="2s")

    # current data
    I_ts = ut.sine(f=Q_(10, "1/s"), amp=Q_(20, "A"), bias=Q_(300, "A"))
    current_data = TimeSeries(I_ts.interp_time(time).data, time)

    # voltage data
    U_ts = ut.sine(
        f=Q_(10, "1/s"), amp=Q_(3, "V"), bias=Q_(40, "V"), phase=Q_(0.1, "rad")
    )
    voltage_data = TimeSeries(U_ts.interp_time(time).data, time)

    # define current source and transformations

    src_current = msm.SignalSource(
        name="Current Sensor",
        output_signal=msm.Signal(signal_type="analog", unit="V", data=None),
        error=msm.Error(Q_(0.1, "percent")),
    )

    current_AD_transform = msm.SignalTransformation(
        name="AD conversion current measurement",
        error=msm.Error(Q_(0.01, "percent")),
        func=MathematicalExpression(
            "a * x + b", dict(a=Q_(32768.0 / 10.0, "1/V"), b=Q_(0.0, ""))
        ),
        type_transformation="AD",
    )

    current_calib_transform = msm.SignalTransformation(
        name="Calibration current measurement",
        error=msm.Error(0.0),
        func=MathematicalExpression(
            "a * x + b", dict(a=Q_(1000.0 / 32768.0, "A"), b=Q_(0.0, "A"))
        ),
    )

    # define voltage source and transformations

    src_voltage = msm.SignalSource(
        name="Voltage Sensor",
        output_signal=msm.Signal("analog", "V", data=None),
        error=msm.Error(Q_(0.1, "percent")),
    )

    voltage_AD_transform = msm.SignalTransformation(
        name="AD conversion voltage measurement",
        error=msm.Error(Q_(0.01, "percent")),
        func=MathematicalExpression(
            "a * x + b", dict(a=Q_(32768.0 / 10.0, "1/V"), b=Q_(0.0, ""))
        ),
        type_transformation="AD",
    )

    voltage_calib_transform = msm.SignalTransformation(
        name="Calibration voltage measurement",
        error=msm.Error(0.0),
        func=MathematicalExpression(
            "a * x + b", dict(a=Q_(100.0 / 32768.0, "V"), b=Q_(0.0, "V"))
        ),
    )

    # Define lab equipment

    HKS_sensor = msm.MeasurementEquipment(
        name="HKS P1000-S3",
        sources=[src_current, src_voltage],
    )
    BH_ELM = msm.MeasurementEquipment(
        name="Beckhoff ELM3002-0000",
        transformations=[current_AD_transform, voltage_AD_transform],
    )

    twincat_scope = Software(name="Beckhoff TwinCAT ScopeView", version="3.4.3143")
    current_calib_transform.wx_metadata = dict(software=twincat_scope)
    voltage_calib_transform.wx_metadata = dict(software=twincat_scope)

    # Define current measurement chain

    welding_current_chain = msm.MeasurementChain.from_equipment(
        name="welding current measurement chain",
        equipment=HKS_sensor,
        source_name="Current Sensor",
    )
    welding_current_chain.add_transformation_from_equipment(
        equipment=BH_ELM,
        transformation_name="AD conversion current measurement",
    )
    welding_current_chain.add_transformation(
        transformation=current_calib_transform,
        data=current_data,
    )

    # Define voltage measurement chain

    welding_voltage_chain = msm.MeasurementChain.from_equipment(
        name="welding voltage measurement chain",
        equipment=HKS_sensor,
        source_name="Voltage Sensor",
    )
    welding_voltage_chain.add_transformation_from_equipment(
        equipment=BH_ELM,
        transformation_name="AD conversion voltage measurement",
    )
    welding_voltage_chain.add_transformation(
        transformation=voltage_calib_transform,
        data=voltage_data,
    )

    # Define measurements

    welding_current = msm.Measurement(
        name="welding current measurement",
        data=[current_data],
        measurement_chain=welding_current_chain,
    )

    welding_voltage = msm.Measurement(
        name="welding voltage measurement",
        data=[voltage_data],
        measurement_chain=welding_voltage_chain,
    )

    # GMAW Process
    params_pulse = dict(
        wire_feedrate=Q_(10.0, "m/min"),
        pulse_voltage=Q_(40.0, "V"),
        pulse_duration=Q_(5.0, "ms"),
        pulse_frequency=Q_(100.0, "Hz"),
        base_current=Q_(60.0, "A"),
    )
    process_pulse = GmawProcess(
        "pulse",
        "CLOOS",
        "Quinto",
        params_pulse,
        tag="CLOOS/pulse",
        meta={"modulation": "UI"},
    )

    gas_comp = [
        GasComponent("argon", Q_(82, "percent")),
        GasComponent("carbon dioxide", Q_(18, "percent")),
    ]
    gas_type = ShieldingGasType(gas_component=gas_comp, common_name="SG")

    gas_for_procedure = ShieldingGasForProcedure(
        use_torch_shielding_gas=True,
        torch_shielding_gas=gas_type,
        torch_shielding_gas_flowrate=Q_(20, "l / min"),
    )

    process = dict(
        welding_process=process_pulse,
        shielding_gas=gas_for_procedure,
        weld_speed=TimeSeries(v_weld),
        welding_wire={"diameter": Q_(1.2, "mm")},
    )

    # ASDF file
    tree = dict(
        reference_timestamp=reference_timestamp,
        equipment=[HKS_sensor, BH_ELM],
        measurements=[welding_current, welding_voltage],
        welding_current=welding_current_chain.get_signal(
            "Calibration current measurement"
        ),
        welding_voltage=welding_voltage_chain.get_signal(
            "Calibration voltage measurement"
        ),
        coordinate_systems=csm,
        TCP=TCP_reference,
        workpiece=workpiece,
        process=process,
        wx_metadata={"welder": "A.W. Elder"},
    )

    model_path = get_schema_path("single_pass_weld-1.0.0.schema.yaml")

    # pre-validate?
    weldx.asdf.util.write_read_buffer(
        tree,
        asdffile_kwargs=dict(custom_schema=str(model_path)),
    )

    if out_file:
        with asdf.AsdfFile(
            tree,
            custom_schema=str(model_path),
        ) as ff:
            ff.write_to(out_file, all_array_storage="inline")
    else:
        return (
            write_buffer(
                tree,
                asdffile_kwargs=dict(custom_schema=str(model_path)),
                write_kwargs=dict(all_array_storage="inline"),
            ),
            tree,
        )


def parse_args(argv):
    """Parse args."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", default="single_pass_weld_example.asdf", type=Path
    )

    args = parser.parse_args(argv if argv is not None else sys.argv)
    return args


def main(argv=None):
    """Invoke single_pass_weld_example.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile(delete=True) as ntf:
    ...     main(['-o', ntf.name])

    """
    args = parse_args(argv)
    out = args.output
    if out.exists() and out.stat().st_size > 0:
        print(f"Destination {out} already exists. Quitting.")
        sys.exit(1)

    single_pass_weld_example(out)


if __name__ == "__main__":
    main()
