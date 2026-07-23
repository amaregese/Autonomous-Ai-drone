import math
import threading
import time

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

    print(f"SITL: Connecting to vehicle on {connection_string}")
    _master = mavutil.mavlink_connection(connection_string, baud=baud)
    _master.wait_heartbeat(timeout=15)
    print(
        f"SITL: Heartbeat received from system {_master.target_system} "
        f"component {_master.target_component}"
    )
    return _master


def arm_and_takeoff(max_height):
    master = _get_master()
    _set_mode("GUIDED")
    master.arducopter_arm()
    master.motors_armed_wait()
    print("SITL: Vehicle armed")

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
    print(f"SITL: Takeoff requested to {max_height:.1f}m")


def land():
    master = _get_master()
    _set_mode("LAND")
    print("SITL: Landing")


def get_EKF_status():
    master = _get_master()
    msg = master.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=1)
    if msg is None:
        return "SITL: EKF status unavailable"
    return f"SITL: EKF flags {msg.flags}"


def get_battery_info():
    master = _get_master()
    msg = master.recv_match(type="SYS_STATUS", blocking=True, timeout=1)
    if msg is None:
        return "SITL: Battery status unavailable"
    battery_remaining = getattr(msg, "battery_remaining", -1)
    return f"SITL: Battery {battery_remaining}%"


def get_version():
    master = _get_master()
    master.mav.autopilot_version_request_send(master.target_system, master.target_component)
    msg = master.recv_match(type="AUTOPILOT_VERSION", blocking=True, timeout=1)
    if msg is None:
        return "SITL: Version unavailable"
    return f"SITL: Flight software version {msg.flight_sw_version}"


def send_movement_command_YAW(angle):
    global _last_yaw_rate_rad_s
    with _state_lock:
        _last_yaw_rate_rad_s = math.radians(angle)


def send_movement_command_XYA(x, y, altitude):
    master = _get_master()
    with _state_lock:
        yaw_rate = _last_yaw_rate_rad_s

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

