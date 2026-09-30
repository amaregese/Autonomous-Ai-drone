import math
import os
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
_heartbeat_thread = None
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
_gps_seen = False
_ekf_seen = False
_cached_flight_sw_version = None
_last_status_text = ""
_status_history = []
_acked_commands = {}
_telemetry_lock = threading.Lock()
_command_lock = threading.Lock()


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


def _wait_for_arm_state(expected: bool, timeout: float = 20.0, poll: float = 0.1):
    """Wait until the FCU heartbeat reports the expected arm state.

    pymavlink's ``motors_armed_wait()`` / ``motors_disarmed_wait()`` are
    ``while True: wait_heartbeat()`` loops with no timeout, so a vehicle that
    never arms (or never disarms) hangs the caller forever. Poll the cached
    heartbeat instead and report the vehicle's own status text, which usually
    says why ("Pre-arm: ...").
    """
    end_time = time.time() + timeout
    while time.time() < end_time:
        with _telemetry_lock:
            armed = _cached_armed
        if armed == expected:
            return
        time.sleep(poll)
    state = "armed" if expected else "disarmed"
    raise RuntimeError(
        f"Vehicle did not report {state} within {timeout:.0f}s "
        f"(status: {_last_status_text or 'unknown'}) - check the pre-arm checks"
    )


def _status_tail(count=3):
    """The last few FCU status messages, newest last - far more useful than a
    single line when the autopilot fails something asynchronously."""
    with _telemetry_lock:
        return " | ".join(_status_history[-count:])


def _set_mode(mode_name, timeout=5.0):
    """Request a flight mode and confirm the vehicle actually switched.

    Uses ``MAV_CMD_DO_SET_MODE`` rather than the ``SET_MODE`` message: on the
    ArduPilot SITL instance this project is tested against, ``SET_MODE`` is
    silently ignored (no COMMAND_ACK, vehicle stays in RTL), while
    ``MAV_CMD_DO_SET_MODE`` is accepted and answers with COMMAND_ACK. Verified
    against SITL: SET_MODE -> still RTL; DO_SET_MODE -> GUIDED, ack 0.

    Also never calls ``recv_match`` here - the message listener already owns
    the socket, so racing it for the heartbeat steals the confirmation and then
    caches a mode the vehicle never entered.
    """
    master = _get_master()
    mapping = master.mode_mapping()
    if not mapping or mode_name not in mapping:
        raise RuntimeError(f"Flight mode '{mode_name}' is not available on this vehicle.")

    custom_mode = mapping[mode_name]
    base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED

    with _telemetry_lock:
        if _cached_mode == mode_name:
            return
        _acked_commands.pop(mavutil.mavlink.MAV_CMD_DO_SET_MODE, None)

    with _command_lock:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            base_mode,
            custom_mode,
            0,
            0,
            0,
            0,
            0,
        )

        ack = _wait_command_ack(mavutil.mavlink.MAV_CMD_DO_SET_MODE, timeout=timeout)
        if ack is None:
            raise RuntimeError(
                f"Vehicle did not acknowledge the {mode_name} mode request "
                f"within {timeout:.0f}s (still reporting {_cached_mode})"
            )
        if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(
                f"Vehicle rejected {mode_name} (result {ack.result}) - "
                f"check the mode is available and enabled"
            )

        if _wait_for_mode(mode_name, timeout=timeout):
            return

    with _telemetry_lock:
        seen = _cached_mode
    raise RuntimeError(
        f"Vehicle did not enter {mode_name} within {timeout:.0f}s "
        f"(still reporting {seen})"
    )


def _wait_for_mode(mode_name, timeout=10.0):
    """Block until the vehicle's own heartbeat reports ``mode_name``.

    A COMMAND_ACK for a mode change only means the request was accepted. This
    waits for the autopilot to actually report the mode, which is the only way
    to be certain the vehicle is where we believe it is before arming.
    """
    end_time = time.time() + timeout
    while time.time() < end_time:
        with _telemetry_lock:
            if _cached_mode == mode_name:
                return True
        time.sleep(0.1)
    return False


def _message_listener():
    global _message_listener_running, _cached_lat, _cached_lon, _cached_alt
    global _cached_battery, _cached_ekf_flags
    global _cached_armed, _cached_mode, _cached_gps_fix_type, _last_status_text
    global _gps_seen, _ekf_seen, _cached_flight_sw_version
    _message_listener_running = True
    _message_listener._last_mode = None
    types = ["STATUSTEXT", "HEARTBEAT", "GLOBAL_POSITION_INT", "SYS_STATUS", "EKF_STATUS_REPORT", "GPS_RAW_INT", "COMMAND_ACK", "AUTOPILOT_VERSION"]
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
                text = msg.text.strip()
                with _telemetry_lock:
                    _last_status_text = text
                    _status_history.append(text)
                    del _status_history[:-12]
                print(f"[VEHICLE] {prefix}{text}")
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
                    _ekf_seen = True
            elif mtype == "GPS_RAW_INT":
                with _telemetry_lock:
                    _cached_gps_fix_type = msg.fix_type
                    _gps_seen = True
            elif mtype == "COMMAND_ACK":
                with _telemetry_lock:
                    _acked_commands[msg.command] = msg
                # Log every ACK: an ACK for a command we did not send means
                # something else on the link is driving the vehicle.
                print(f"[VEHICLE] ACK cmd={msg.command} result={msg.result}")
            elif mtype == "AUTOPILOT_VERSION":
                with _telemetry_lock:
                    _cached_flight_sw_version = msg.flight_sw_version
                print(f"[VEHICLE] ArduCopter {msg.flight_sw_version.to_bytes(8, 'little', signed=False).decode(errors='replace').rstrip(chr(0))} "
                      f"({msg.flight_custom_version[0]:02x})")
        except Exception:
            time.sleep(0.1)


def _mode_id_to_name(type_id, custom_mode):
    table = mavutil.AP_MAV_TYPE_MODE_MAP.get(type_id, {})
    return table.get(custom_mode)


def _request_message_intervals(master):
    # STATUSTEXT is deliberately absent. ArduPilot broadcasts it on change but
    # does not accept a stream-rate request for it, and answering that request
    # makes it emit "No ap_message for mavlink id (253)" on every connect.
    intervals = [
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


def _heartbeat_loop():
    """Keep the GCS link alive so the FCU does not treat us as lost.

    pymavlink never sends heartbeats on its own. Without this the app goes
    completely silent whenever it is not issuing a command, and a vehicle
    left armed in GUIDED can decide its ground station vanished and disarm.
    """
    while True:
        master = _master
        if master is not None:
            try:
                with _command_lock:
                    master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0,
                    )
            except Exception:
                pass
        time.sleep(1.0)


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


def _check_safety_params(master):
    """Warn loudly about FCU parameters that make autonomous flight unsafe.

    The Python-side GPS/EKF checks in the UI are not a substitute for the
    autopilot's own arming checks - those are the last line of defence.

    Runs on a daemon thread: ``param_fetch_all`` blocks waiting for the vehicle
    to answer, and a vehicle that never does must not stall the connect.
    """
    def work():
        try:
            master.param_fetch_all()
        except Exception:
            return
        try:
            value = master.params.get("ARMING_CHECKS")
        except Exception:
            value = None
        if value is None:
            return
        try:
            checks = float(value)
        except (TypeError, ValueError):
            return
        if checks == 0.0:
            print(
                "[SAFETY] ARMING_CHECKS=0 - ALL arming checks are DISABLED. The "
                "autopilot will not verify throttle-at-zero, GPS, EKF or compass "
                "before arming. This is unsafe on a real vehicle; re-enable it."
            )
        elif checks == 1.0:
            print(
                "[SAFETY] ARMING_CHECKS=1 - only basic checks; GPS/EKF/compass are "
                "not verified by the autopilot. ARMING_CHECKS>=2 is recommended."
            )

    t = threading.Thread(target=work, daemon=True, name="param-check")
    t.start()
    return t


def connect_drone(connection_string, waitready=True, baud=57600, start_sitl=False):
    global _master
    _require_mavlink()

    if start_sitl:
        if os.name != "nt":
            raise RuntimeError("SITL auto-launch (--start-sitl) is only supported on Windows/WSL. "
                               "On Linux/Jetson, start SITL manually or connect to a running instance.")
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
    _check_safety_params(_master)

    global _message_listener_thread, _heartbeat_thread
    if _message_listener_thread is None or not _message_listener_thread.is_alive():
        _message_listener_thread = threading.Thread(target=_message_listener, daemon=True)
        _message_listener_thread.start()
    if _heartbeat_thread is None or not _heartbeat_thread.is_alive():
        _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        _heartbeat_thread.start()

    return _master


def arm():
    """Arm the vehicle in GUIDED and wait until the FCU reports it armed."""
    global _cached_armed, _cached_mode
    master = _get_master()
    _set_mode("GUIDED")
    # _set_mode returns early when the cache already claims GUIDED, and a mode
    # ACK only means the request was accepted. Confirm against a fresh
    # heartbeat before pulling the arming lever: the autopilot refuses to arm
    # in LAND ("Arm: Land mode not armable"), and arming in a mode we did not
    # intend is worse than refusing outright.
    if not _wait_for_mode("GUIDED", timeout=10.0):
        with _telemetry_lock:
            seen = _cached_mode
        raise RuntimeError(
            f"Vehicle is not in GUIDED (reporting {seen}); refusing to arm"
        )

    master.arducopter_arm()
    _wait_for_arm_state(True)
    with _telemetry_lock:
        _cached_armed = True
    print("SITL: Vehicle armed")


def _send_altitude_hold(height_above_launch, period=0.4):
    """Keep an explicit altitude setpoint alive for the autopilot.

    Nothing else in this file gives ArduPilot an altitude to hold: the follow
    controller sends SET_POSITION_TARGET_LOCAL_NED with Z_IGNORE, so after a
    NAV_TAKEOFF the vehicle has a climb command but no setpoint, and on the SITL
    instance tested here it simply kept climbing past the target forever.

    The frame is GLOBAL_RELATIVE_ALT (metres above the launch point) so the
    value is independent of the launch altitude.
    """
    master = _get_master()
    mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )
    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mask,
        0,
        0,
        height_above_launch,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    time.sleep(period)


def takeoff(max_height, timeout=60.0, hold_tolerance=1.0, stable_seconds=2.0):
    """Climb `max_height` above the launch point and hold it. Must be armed.

    MAV_CMD_NAV_TAKEOFF param7 is an **absolute AMSL altitude**, not a height
    above the launch point. Sending the relative height straight through (as
    this used to) asks a vehicle sitting at e.g. 584 m MSL to climb to 5 m MSL,
    i.e. to fly into the ground - which ArduPilot answers with a failsafe and a
    disarm. It "worked" in SITL only because SITL home sat near 0 m MSL.

    NAV_TAKEOFF alone is not enough to *hold* an altitude: it is a climb command,
    and with no altitude setpoint following it the vehicle keeps climbing. So
    this also streams an explicit hold setpoint and only reports success once
    the altitude is actually settled. A vehicle that is still climbing away when
    it hits the ceiling is reported as a failure, never as "holding".
    """
    master = _get_master()

    with _telemetry_lock:
        if not _cached_armed:
            raise RuntimeError("Vehicle is not armed")
        launch_amsl = _cached_alt
        current_mode = _cached_mode

    if not launch_amsl or launch_amsl <= 0.0:
        raise RuntimeError(
            f"No valid altitude reading (GLOBAL_POSITION_INT reports "
            f"{launch_amsl:.1f} m); refusing to send a takeoff altitude"
        )

    target_amsl = launch_amsl + max_height
    # A generous but real ceiling: a real overshoot is a few metres, an
    # unbounded climb is tens. Anything past this is a runaway, not a climb.
    ceiling_amsl = target_amsl + max(3.0, max_height)
    print(f"SITL: Takeoff {launch_amsl:.1f} -> {target_amsl:.1f} m AMSL "
          f"(+{max_height:.1f} m) in {current_mode}, ceiling {ceiling_amsl:.1f} m AMSL")

    with _telemetry_lock:
        _acked_commands.pop(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, None)
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
        target_amsl,
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

    start = time.time()
    stable_since = None
    last_alt = launch_amsl
    last_report = start
    while time.time() - start < timeout:
        with _telemetry_lock:
            armed = _cached_armed
            alt = _cached_alt

        if not armed:
            raise RuntimeError(
                f"Vehicle disarmed during takeoff "
                f"(status: {_status_tail()}) - check failsafe/arming settings"
            )

        climbed = alt - launch_amsl

        # Mis-framed takeoff: we asked to climb and the vehicle is sinking.
        if climbed < -0.5:
            raise RuntimeError(
                f"Takeoff commanded +{max_height:.1f} m to {target_amsl:.1f} m AMSL "
                f"but the vehicle is descending ({climbed:.1f} m, now {alt:.1f} m AMSL). "
                f"Check the NAV_TAKEOFF altitude frame and any altitude failsafe. "
                f"(status: {_status_tail()})"
            )

        # Do NOT stream a setpoint during the climb. In GUIDED a
        # SET_POSITION_TARGET overrides the NAV_TAKEOFF target, and a setpoint
        # near the current altitude is read as "stay here" - the vehicle then
        # never leaves the ground and ArduPilot disarms it as armed-but-idle
        # (ACK for cmd=22 arrives, altitude never changes). NAV_TAKEOFF alone
        # is what makes it climb.
        #
        # Only once it is at/past the target do we take over and hold, which
        # is what stops it climbing away past the target.
        # Only once the climb is clearly under way do we take over and hold,
        # and we always command the FULL target. Engaging late leaves the
        # autopilot no time to arrest, and it sails past the target; engaging
        # with a setpoint near the current altitude instead would read as
        # "stay here" and pin the vehicle on the ground.
        if climbed >= max_height * 0.4:
            try:
                _send_altitude_hold(max_height, period=0.0)
            except Exception as exc:
                print(f"SITL: altitude hold setpoint failed: {exc}")

        # Runaway: it blew through the ceiling and is still going up.
        if alt > ceiling_amsl and alt > last_alt:
            try:
                _send_altitude_hold(max_height, period=0.05)
            except Exception:
                pass
            raise RuntimeError(
                f"Vehicle is climbing away: {climbed:.1f} m above launch "
                f"({alt:.1f} m AMSL) and still rising, past the "
                f"{ceiling_amsl:.1f} m ceiling. The autopilot is not holding the "
                f"takeoff altitude. Check the altitude controller tuning and "
                f"whether anything is sending position targets. "
                f"(status: {_status_tail()})"
            )

        # Settled?
        if abs(alt - target_amsl) <= hold_tolerance:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= stable_seconds:
                print(f"SITL: Holding {climbed:.1f} m above launch "
                      f"({alt:.1f} m AMSL)")
                return
        else:
            stable_since = None

        last_alt = alt
        if time.time() - last_report >= 2.0:
            last_report = time.time()
            print(f"SITL: climbing {climbed:+.1f} m "
                  f"(target +{max_height:.1f} m, {alt:.1f} m AMSL)")
        time.sleep(0.2)

    with _telemetry_lock:
        alt = _cached_alt
    climbed = alt - launch_amsl
    if abs(alt - target_amsl) > hold_tolerance:
        raise RuntimeError(
            f"Takeoff did not settle at {max_height:.1f} m after "
            f"{timeout:.0f}s (at {climbed:.1f} m, {alt:.1f} m AMSL). "
            f"(status: {_status_tail()})"
        )
    print(f"SITL: Holding {climbed:.1f} m above launch ({alt:.1f} m AMSL)")


def arm_and_takeoff(max_height):
    arm()
    takeoff(max_height)


def land():
    master = _get_master()
    _set_mode("LAND")
    print("SITL: Landing")


def disarm():
    """Disarm the vehicle and wait for the FCU to report it disarmed.

    Only safe on the ground - the caller is responsible for that check.
    """
    global _cached_armed
    master = _get_master()
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0,   # param1: 0 = disarm
        0,   # param2: 0 = disarm (1 would arm)
        0,
        0,
        0,
        0,
        0,
    )
    ack = _wait_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
    if ack is None:
        raise RuntimeError(
            "Disarm timed out waiting for COMMAND_ACK "
            f"(vehicle status: {_last_status_text or 'unknown'})"
        )
    if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        raise RuntimeError(
            f"Disarm command rejected (result {ack.result}) "
            f"(vehicle status: {_last_status_text or 'unknown'})"
        )
    _wait_for_arm_state(False)
    with _telemetry_lock:
        _cached_armed = False
    print("SITL: Vehicle disarmed")


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
    # Read the listener's cache. A recv_match() here competes with
    # _message_listener() for the single shared socket and can swallow a
    # message the listener needed, and this is called by the pre-arm check.
    with _telemetry_lock:
        battery_remaining = _cached_battery
    if battery_remaining < 0:
        return "SITL: Battery status unavailable"
    return f"SITL: Battery {battery_remaining}%"


def get_version():
    with _telemetry_lock:
        version = _cached_flight_sw_version
    if version is None:
        return "SITL: Version unavailable"
    return f"SITL: Flight software version {version}"


def wait_for_navigation_telemetry(timeout=8.0, poll=0.1):
    """Block until the first GPS_RAW_INT and EKF_STATUS_REPORT have arrived.

    Returns True once both have been seen, False on timeout. The pre-arm gate
    needs this because a cold cache is indistinguishable from a bad one: a
    vehicle reporting fix 6 reads as "GPS fix 0/3" until its first report lands.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _telemetry_lock:
            if _gps_seen and _ekf_seen:
                return True
        time.sleep(poll)
    with _telemetry_lock:
        return _gps_seen and _ekf_seen


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

