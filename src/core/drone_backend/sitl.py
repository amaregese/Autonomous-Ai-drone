import math
import threading

try:
    from pymavlink import mavutil
except ImportError:  # pragma: no cover - dependency may be absent during static checks
    mavutil = None


_master = None
_state_lock = threading.Lock()
_last_yaw_rate_rad_s = 0.0


def _require_mavlink():
    if mavutil is None:
        raise RuntimeError(
            "pymavlink is not installed in this Python environment. "
            "Install it before using --mode sitl."
        )


def _get_master():
    if _master is None:
        raise RuntimeError("SITL backend is not connected. Call connect_drone() first.")
    return _master


def _wait_command_ack(command_id, timeout=5.0):
    master = _get_master()
    end_time = time.time() + timeout
    while time.time() < end_time:
        msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
        if msg and msg.command == command_id:
            return msg
    return None


def _set_mode(mode_name):
    master = _get_master()
    mapping = master.mode_mapping()
    if not mapping or mode_name not in mapping:
        raise RuntimeError(f"Flight mode '{mode_name}' is not available on this vehicle.")

    mode_id = mapping[mode_name]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    master.recv_match(type="HEARTBEAT", blocking=True, timeout=2)


def connect_drone(connection_string, waitready=True, baud=57600):
    global _master
    _require_mavlink()

    print(f"Connecting to vehicle on {connection_string}")
    _master = mavutil.mavlink_connection(connection_string, baud=baud)
    try:
        _master.wait_heartbeat(timeout=10)
        print(
            f"Heartbeat received from system {_master.target_system} "
            f"component {_master.target_component}"
        )
    except Exception as e:
        print(f"Heartbeat timeout - {e}")
        raise RuntimeError(
            f"Failed to connect to SITL at {connection_string}. "
            "Make sure SITL is running with --out=tcp:127.0.0.1:5760"
        )
    return _master


def arm_and_takeoff(max_height):
    master = _get_master()
    _set_mode("GUIDED")
    print("Requesting arming...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0
    )
    ack = _wait_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
    if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print("Vehicle armed")
    else:
        print(f"Arm command result: {ack.result if ack else 'timeout'}")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        max_height,
    )
    _wait_command_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
    print(f"Mock: Takeoff requested to {max_height:.1f}m")


def land():
    master = _get_master()
    _set_mode("LAND")
    print("Landing")


def get_EKF_status():
    master = _get_master()
    msg = master.recv_match(type="EKF_STATUS_REPORT", blocking=False)
    if msg is None:
        return "EKF status unavailable"
    return f"EKF flags {msg.flags}"


def get_battery_info():
    master = _get_master()
    msg = master.recv_match(type="SYS_STATUS", blocking=False)
    if msg is None:
        return "Battery status unavailable"
    battery_remaining = getattr(msg, "battery_remaining", -1)
    return f"Battery {battery_remaining}%"


def get_version():
    master = _get_master()
    master.mav.autopilot_version_request_send(master.target_system, master.target_component)
    msg = master.recv_match(type="AUTOPILOT_VERSION", blocking=True, timeout=2)
    if msg is None:
        return "Version unavailable"
    return f"Flight software version {msg.flight_sw_version}"


def send_movement_command_YAW(angle):
    global _last_yaw_rate_rad_s
    with _state_lock:
        _last_yaw_rate_rad_s = math.radians(angle)
    direction = "RIGHT" if angle > 0 else "LEFT" if angle < 0 else "STOP"
    print(f"Yaw command {angle:.2f} deg/s -> Rotating {direction}")


def send_movement_command_XYA(x, y, altitude):
    master = _get_master()
    with _state_lock:
        yaw_rate = _last_yaw_rate_rad_s

    if y > 0:
        movement = f"FORWARD at {abs(y):.2f} m/s"
    elif y < 0:
        movement = f"BACKWARD at {abs(y):.2f} m/s"
    else:
        movement = "HOVERING"

    if x > 0:
        lateral = f"RIGHT at {abs(x):.2f} m/s"
    elif x < 0:
        lateral = f"LEFT at {abs(x):.2f} m/s"
    else:
        lateral = "CENTERED"

    print(f"Mock: Move -> {movement} | {lateral} | Alt: {altitude:.1f}m")

    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask,
        0,
        0,
        0,
        y,
        x,
        0,
        0,
        0,
        0,
        0,
        yaw_rate,
    )

