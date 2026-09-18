import math
import subprocess
import sys
import threading
import time

try:
    from pymavlink import mavutil
except ImportError:  # pragma: no cover - dependency may be absent during static checks
    mavutil = None


_master = None
_state_lock = threading.Lock()
_last_yaw_rate_rad_s = 0.0
_message_listener_thread = None
_message_listener_running = False
_sitl_process = None

_cached_lat = 0.0
_cached_lon = 0.0
_cached_alt = 0.0
_cached_battery = -1
_cached_ekf_flags = 0
_cached_armed = False
_cached_mode = "UNKNOWN"
_cached_gps_fix_type = 0
_cached_armed = False
_cached_mode = "UNKNOWN"
_last_status_text = ""
_acked_commands = {}
_telemetry_lock = threading.Lock()


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
        with _telemetry_lock:
            msg = _acked_commands.get(command_id)
        if msg is not None:
            _acked_commands.pop(command_id, None)
            return msg
        time.sleep(0.1)
    return None


def _set_mode(mode_name):
    global _cached_mode
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
    with _telemetry_lock:
        _cached_mode = mode_name


def _message_listener():
    global _message_listener_running, _cached_lat, _cached_lon, _cached_alt
    global _cached_battery, _cached_ekf_flags
    global _cached_armed, _cached_mode, _cached_gps_fix_type, _last_status_text
    _message_listener_running = True
    _message_listener._last_mode = None
    types = ["STATUSTEXT", "HEARTBEAT", "GLOBAL_POSITION_INT", "SYS_STATUS", "EKF_STATUS_REPORT", "GPS_RAW_INT", "COMMAND_ACK"]
    while _message_listener_running:
        master = _master
        if master is None:
            time.sleep(0.1)
            continue
        try:
            msg = master.recv_match(type=types, blocking=True, timeout=1)
            if msg is None:
                continue
            mtype = msg.get_type()
            if mtype == "STATUSTEXT":
                severity = msg.severity
                prefix = ""
                if severity <= 4:
                    prefix = "AP: "
                _last_status_text = msg.text
                print(f"[VEHICLE] {prefix}{msg.text}")
            elif mtype == "HEARTBEAT":
                custom = msg.custom_mode
                mode_name = _mode_id_to_name(msg.type, custom)
                if mode_name and _message_listener._last_mode != mode_name:
                    print(f"[VEHICLE] Mode {mode_name}")
                _message_listener._last_mode = mode_name
                with _telemetry_lock:
                    _cached_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    _cached_mode = mode_name or "UNKNOWN"
            elif mtype == "GLOBAL_POSITION_INT":
                with _telemetry_lock:
                    _cached_lat = msg.lat / 1e7
                    _cached_lon = msg.lon / 1e7
                    _cached_alt = msg.alt / 1000.0
            elif mtype == "SYS_STATUS":
                with _telemetry_lock:
                    _cached_battery = getattr(msg, "battery_remaining", -1)
            elif mtype == "EKF_STATUS_REPORT":
                with _telemetry_lock:
                    _cached_ekf_flags = msg.flags
            elif mtype == "GPS_RAW_INT":
                with _telemetry_lock:
                    _cached_gps_fix_type = msg.fix_type
            elif mtype == "COMMAND_ACK":
                with _telemetry_lock:
                    _acked_commands[msg.command] = msg
        except Exception:
            time.sleep(0.1)


def _mode_id_to_name(type_id, custom_mode):
    table = mavutil.AP_MAV_TYPE_MODE_MAP.get(type_id, {})
    return table.get(custom_mode)


def _request_message_intervals(master):
    intervals = [
        (mavutil.mavlink.MAVLINK_MSG_ID_STATUSTEXT, 1000000),
        (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 1000000),
        (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 200000),
        (mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 1000000),
        (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 500000),
        (mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 500000),
    ]
    for msg_id, interval_us in intervals:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us,
            0, 0, 0, 0, 0,
        )


def _start_sitl_auto():
    global _sitl_process
    cmd = (
        'bash -c "'
        'source ~/.profile 2>/dev/null; '
        'cd ~/ardupilot/ArduCopter && '
        'sim_vehicle.py -v ArduCopter --out udp:127.0.0.1:14550 --no-mavproxy --console'
        '"'
    )
    print("[SITL] Launching ArduCopter SITL in WSL (this may take ~30s)...")
    _sitl_process = subprocess.Popen(
        ["wsl", "-e", "bash", "-c",
         "source ~/.profile 2>/dev/null; cd ~/ardupilot/ArduCopter && "
         "sim_vehicle.py -v ArduCopter --out udp:127.0.0.1:14550 --no-mavproxy --console"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    reader_thread = threading.Thread(target=_stream_sitl_output, args=(_sitl_process,), daemon=True)
    reader_thread.start()

    time.sleep(3)
    if _sitl_process.poll() is not None:
        raise RuntimeError("SITL process exited immediately. Check WSL ArduPilot installation.")
    print("[SITL] Waiting for vehicle to initialize...")


def _stream_sitl_output(proc):
    for line in iter(proc.stdout.readline, b""):
        try:
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                print(f"[SITL] {text}")
        except Exception:
            pass


def stop_sitl():
    global _sitl_process
    if _sitl_process is not None:
        _sitl_process.terminate()
        try:
            _sitl_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _sitl_process.kill()
        _sitl_process = None
        print("[SITL] SITL process stopped")


def connect_drone(connection_string, waitready=True, baud=57600, start_sitl=False):
    global _master
    _require_mavlink()

    if start_sitl:
        _start_sitl_auto()

    print(f"SITL: Connecting to vehicle on {connection_string}")
    try:
        _master = mavutil.mavlink_connection(connection_string, baud=baud)
        _master.mavlink20()
        _master.wait_heartbeat(timeout=30)
    except Exception as exc:
        _master = None
        raise RuntimeError(f"Failed to connect to vehicle at {connection_string}: {exc}") from exc

    if _master.target_system == 0 and _master.target_component == 0:
        _master = None
        raise RuntimeError(
            f"No vehicle found at {connection_string}. "
            "Start SITL (sim_vehicle.py) or connect a real vehicle first."
        )

    print(
        f"SITL: Heartbeat received from system {_master.target_system} "
        f"component {_master.target_component}"
    )

    _request_message_intervals(_master)

    global _message_listener_thread
    if _message_listener_thread is None or not _message_listener_thread.is_alive():
        _message_listener_thread = threading.Thread(target=_message_listener, daemon=True)
        _message_listener_thread.start()

    return _master


def arm_and_takeoff(max_height):
    global _cached_armed, _cached_mode
    master = _get_master()
    _set_mode("GUIDED")
    with _telemetry_lock:
        _cached_mode = "GUIDED"

    master.arducopter_arm()
    master.motors_armed_wait()
    with _telemetry_lock:
        _cached_armed = True
        _cached_mode = "GUIDED"
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
    ack = _wait_command_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
    if ack is None:
        raise RuntimeError(
            "Takeoff timed out waiting for COMMAND_ACK "
            f"(vehicle status: {_last_status_text or 'unknown'})"
        )
    if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        raise RuntimeError(
            f"Takeoff command rejected (result {ack.result}) "
            f"(vehicle status: {_last_status_text or 'unknown'})"
        )

    print(f"SITL: Takeoff requested to {max_height:.1f}m")

    with _telemetry_lock:
        start_alt = _cached_alt

    deadline = time.time() + 30.0
    poll_interval = 0.25
    while time.time() < deadline:
        time.sleep(poll_interval)
        with _telemetry_lock:
            armed = _cached_armed
            alt = _cached_alt

        if not armed:
            raise RuntimeError(
                f"Vehicle disarmed during takeoff (status: {_last_status_text or 'unknown'}) — "
                "check failsafe/arming settings"
            )

        climbed = alt - start_alt
        if climbed >= max_height * 0.95:
            print(f"SITL: Reached target altitude ({climbed:.1f}m climbed)")
            return

    with _telemetry_lock:
        alt = _cached_alt
    raise RuntimeError(
        f"Takeoff timeout: only climbed {alt - start_alt:.1f}m in 30s "
        f"(status: {_last_status_text or 'unknown'})"
    )


def land():
    master = _get_master()
    _set_mode("LAND")
    print("SITL: Landing")


def send_rtl():
    master = _get_master()
    _set_mode("RTL")
    print("SITL: RTL initiated")


def is_armed():
    with _telemetry_lock:
        return _cached_armed


def get_mode():
    with _telemetry_lock:
        return _cached_mode


def get_gps_fix_type():
    with _telemetry_lock:
        return _cached_gps_fix_type


def is_ekf_ok():
    with _telemetry_lock:
        flags = _cached_ekf_flags
    EKF_ATTITUDE = 1
    EKF_VELOCITY_HORIZ = 2
    EKF_POS_HORIZ_ABS = 16
    EKF_POS_VERT_ABS = 32
    required = EKF_ATTITUDE | EKF_VELOCITY_HORIZ | EKF_POS_HORIZ_ABS | EKF_POS_VERT_ABS
    return (flags & required) == required


def get_EKF_status():
    with _telemetry_lock:
        flags = _cached_ekf_flags
    return f"SITL: EKF flags {flags}"


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


def get_position():
    with _telemetry_lock:
        return _cached_lat, _cached_lon, _cached_alt


def get_battery_level():
    with _telemetry_lock:
        return _cached_battery


def send_movement_command_YAW(angle):
    global _last_yaw_rate_rad_s
    with _state_lock:
        _last_yaw_rate_rad_s = math.radians(angle)


def send_servo(channel=8, pulse=1500):
    master = _get_master()
    channel = int(channel)
    if channel < 1 or channel > 8:
        raise ValueError(f"Servo channel {channel} out of range 1-8 (RC_CHANNELS_OVERRIDE)")
    if not (1000 <= pulse <= 2000):
        raise ValueError(f"Servo pulse {pulse}us out of range 1000-2000")
    chans = [0] * 8
    chans[channel - 1] = pulse
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        chans[0],
        chans[1],
        chans[2],
        chans[3],
        chans[4],
        chans[5],
        chans[6],
        chans[7],
    )


def hold_position():
    master = _get_master()
    yaw_rate = 0.0
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
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0,
        yaw_rate,
    )


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

