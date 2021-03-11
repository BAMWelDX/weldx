# Welding schema
if __name__ == "__main__":

    # Imports
    from pathlib import Path

    import asdf
    import numpy as np
    import pandas as pd
    import sympy
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
    time = pd.timedelta_range(start="0s", end="10s", freq="1ms")

    # current data
    I_ts = ut.sine(f=Q_(10, "1/s"), amp=Q_(20, "A"), bias=Q_(300, "A"))
    I = I_ts.interp_time(time)  # noqa: E741
    I["time"] = I["time"]

    current_data = msm.Data(name="Welding current", data=I)

    # voltage data
    U_ts = ut.sine(
        f=Q_(10, "1/s"), amp=Q_(3, "V"), bias=Q_(40, "V"), phase=Q_(0.1, "rad")
    )
    U = U_ts.interp_time(time)
    U["time"] = U["time"]

    voltage_data = msm.Data(name="Welding voltage", data=U)

    HKS_sensor = msm.GenericEquipment(name="HKS P1000-S3")
    BH_ELM = msm.GenericEquipment(name="Beckhoff ELM3002-0000")
    twincat_scope = Software(name="Beckhoff TwinCAT ScopeView", version="3.4.3143")

    src_current = msm.Source(
        name="Current Sensor",
        output_signal=msm.Signal(signal_type="analog", unit="V", data=None),
        error=msm.Error(Q_(0.1, "percent")),
    )

    HKS_sensor.sources = []
    HKS_sensor.sources.append(src_current)

    from weldx.core import MathematicalExpression

    [a, x, b] = sympy.symbols("a x b")
    current_AD_func = MathematicalExpression(a * x + b)
    current_AD_func.set_parameter("a", Q_(32768.0 / 10.0, "1/V"))
    current_AD_func.set_parameter("b", Q_(0.0, ""))

    current_AD_transform = msm.DataTransformation(
        name="AD conversion current measurement",
        input_signal=src_current.output_signal,
        output_signal=msm.Signal("digital", "", data=None),
        error=msm.Error(Q_(0.01, "percent")),
        func=current_AD_func,
    )

    BH_ELM.data_transformations = []
    BH_ELM.data_transformations.append(current_AD_transform)

    # define current output calibration expression and transformation
    current_calib_func = MathematicalExpression(a * x + b)
    current_calib_func.set_parameter("a", Q_(1000.0 / 32768.0, "A"))
    current_calib_func.set_parameter("b", Q_(0.0, "A"))

    current_calib_transform = msm.DataTransformation(
        name="Calibration current measurement",
        input_signal=current_AD_transform.output_signal,
        output_signal=msm.Signal("digital", "A", data=current_data),
        error=msm.Error(0.0),
        func=current_calib_func,
    )
    current_calib_transform.wx_metadata = dict(software=twincat_scope)

    welding_current_chain = msm.MeasurementChain(
        name="welding current measurement chain",
        data_source=src_current,
        data_processors=[current_AD_transform, current_calib_transform],
    )

    welding_current = msm.Measurement(
        name="welding current measurement",
        data=[current_data],
        measurement_chain=welding_current_chain,
    )

    src_voltage = msm.Source(
        name="Voltage Sensor",
        output_signal=msm.Signal("analog", "V", data=None),
        error=msm.Error(Q_(0.1, "percent")),
    )

    HKS_sensor.sources.append(src_voltage)

    # define AD conversion expression and transformation step
    [a, x, b] = sympy.symbols("a x b")
    voltage_ad_func = MathematicalExpression(a * x + b)
    voltage_ad_func.set_parameter("a", Q_(32768.0 / 10.0, "1/V"))
    voltage_ad_func.set_parameter("b", Q_(0.0, ""))

    voltage_AD_transform = msm.DataTransformation(
        name="AD conversion voltage measurement",
        input_signal=src_voltage.output_signal,
        output_signal=msm.Signal("digital", "", data=None),
        error=msm.Error(Q_(0.01, "percent")),
        func=voltage_ad_func,
    )

    HKS_sensor.data_transformations.append(voltage_AD_transform)

    # define voltage output calibration expression and transformation
    voltage_calib_func = MathematicalExpression(a * x + b)
    voltage_calib_func.set_parameter("a", Q_(100.0 / 32768.0, "V"))
    voltage_calib_func.set_parameter("b", Q_(0.0, "V"))

    voltage_calib_transform = msm.DataTransformation(
        name="Calibration voltage measurement",
        input_signal=voltage_AD_transform.output_signal,
        output_signal=msm.Signal("digital", "V", data=voltage_data),
        error=msm.Error(0.0),
        func=voltage_calib_func,
    )
    voltage_calib_transform.wx_metadata = dict(software=twincat_scope)

    welding_voltage_chain = msm.MeasurementChain(
        name="welding voltage measurement chain",
        data_source=src_voltage,
        data_processors=[voltage_AD_transform, voltage_calib_transform],
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
        welding_current=current_calib_transform.output_signal,
        welding_voltage=voltage_calib_transform.output_signal,
        coordinate_systems=csm,
        TCP=TCP_reference,
        workpiece=workpiece,
        process=process,
        wx_metadata={"welder": "A.W. Elder"},
    )

    model_path = Path(weldx.__path__[0]) / Path(
        "./asdf/schemas/weldx.bam.de/weldx/datamodels/"
        "single_pass_weld-1.0.0.schema.yaml"
    )
    model_path = model_path.as_posix()

    res = weldx.asdf.util._write_read_buffer(
        tree,
        asdffile_kwargs=dict(custom_schema=str(model_path)),
    )

    with asdf.AsdfFile(
        tree,
        custom_schema=str(model_path),
    ) as ff:
        ff.write_to("single_pass_weld_example.asdf")
