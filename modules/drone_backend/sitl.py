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

from modules.drone_backend import servo_channels


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
# True once a GLOBAL_POSITION_INT has been cached. GPS/EKF reports arriving
# does not guarantee a position fix has been cached yet, and reading the home
# altitude before that lands records 0.0 for the whole run.
_position_seen = False
# monotonic stamp of the last message of any kind from the vehicle, so a dead
# USB/telemetry link can be told apart from a quiet one.
_last_telemetry_at = 0.0
_cached_flight_sw_version = None
_last_status_text = ""
_status_history = []
_acked_commands = {}
_telemetry_lock = threading.Lock()
_command_lock = threading.Lock()
# Armed by an RC override write, consumed by the next mode report. See
# `_arm_override_guard`: it is the after-the-fact safety net for an input that
# turned out to be flight control despite passing every static check.
_override_guard = None
OVERRIDE_GUARD_SECONDS = 2.5
# RC inputs currently held by an override this app wrote. ArduPilot latches
# RC_CHANNELS_OVERRIDE until it is told otherwise, so this set is what stands
# between a servo write and a control input pinned for the rest of the session.
# It is emptied by `_release_rc_overrides`, which every flight-state transition
# and the shutdown path call.
_rc_override_inputs = set()

# Servo discovery cache. `_servo_functions` is {channel: SERVOx_FUNCTION id},
# `_servo_pwm` and `_rc_channels` are {channel: pulse}, both read from the
# autopilot rather than assumed. The `_at` dicts stamp when each reading
# arrived, so a command can tell a fresh report from a leftover one.
# `_servo_limits` is {("SERVO"|"RC", channel): (MIN, TRIM, MAX)}, needed to
# predict what a k_rcinN_mapped output will actually report back.
_servo_functions = {}
_servo_pwm = {}
_servo_pwm_at = {}
_rc_channels = {}
_rc_channels_at = {}
_servo_limits = {}
_params_loaded = False
# Outcome of the last servo command, for the ground station. UDP is
# fire-and-forget, so this is the only way an operator learns that a command was
# refused or that the vehicle reported a different pulse than was commanded.
_servo_status = {
    "detected_channel": None,
    "source": None,
    "commanded": None,
    "expected": None,
    "reported": None,
    "verified": None,
    "refused": False,
    "refusal_reason": None,
    "limits_hit": False,
}
_SERVO_OUTPUT_FIELDS = servo_channels.build_servo_field_map("servo")
_RC_CHANNEL_FIELDS = servo_channels.build_servo_field_map("chan")


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

    # The lock covers the write only, never the waits. The heartbeat takes this
    # lock once a second to keep the GCS link alive, and holding it across the
    # ACK wait plus the mode-change wait - up to 2x timeout - starved the
    # heartbeat for seconds at a time, which is long enough for the FCU to
    # treat its ground station as gone mid-transition. Concurrent mode changes
    # therefore race their waits now; each still confirms only the mode it
    # asked for, and the loser reports "did not enter <mode>" instead of
    # silently succeeding.
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
    global _position_seen, _last_telemetry_at
    _message_listener_running = True
    _servo_pwm.clear()  # stale PWM from a previous vehicle would misreport state
    _message_listener._last_mode = None
    types = ["STATUSTEXT", "HEARTBEAT", "GLOBAL_POSITION_INT", "SYS_STATUS", "EKF_STATUS_REPORT", "GPS_RAW_INT", "COMMAND_ACK", "AUTOPILOT_VERSION", "SERVO_OUTPUT_RAW", "RC_CHANNELS_RAW"]
    while _message_listener_running:
        master = _master
        if master is None:
            time.sleep(0.1)
            continue
        try:
            msg = master.recv_match(type=types, blocking=True, timeout=1)
            if msg is None:
                continue
            _last_telemetry_at = time.monotonic()
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
                _check_override_guard(mode_name)
            elif mtype == "GLOBAL_POSITION_INT":
                with _telemetry_lock:
                    _cached_lat = msg.lat / 1e7
                    _cached_lon = msg.lon / 1e7
                    _cached_alt = msg.alt / 1000.0
                    _position_seen = True
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
            elif mtype == "SERVO_OUTPUT_RAW":
                _cache_pwm(msg, _SERVO_OUTPUT_FIELDS, "_servo_pwm")
            elif mtype == "RC_CHANNELS_RAW":
                _cache_pwm(msg, _RC_CHANNEL_FIELDS, "_rc_channels")
        except Exception:
            time.sleep(0.1)


def _mode_id_to_name(type_id, custom_mode):
    table = mavutil.AP_MAV_TYPE_MODE_MAP.get(type_id, {})
    return table.get(custom_mode)


def _cache_pwm(msg, fields, cache_name):
    """Merge a PWM telemetry report into one of the module caches.

    A named cache rather than an inline assignment because assigning to
    ``_servo_pwm`` inside the listener needs an explicit ``global``, and when
    that was missing the writes silently went to a function local: the caches
    stayed empty forever and servo discovery never saw a PWM report, with no
    error to show for it.
    """
    values = servo_channels.parse_pwm_fields(msg, fields)
    if not values:
        return
    target = globals()[cache_name]
    now = time.monotonic()
    with _telemetry_lock:
        target.update(values)
        # Stamped so a command can tell a fresh reading from a leftover one.
        stamps = _servo_pwm_at if cache_name == "_servo_pwm" else _rc_channels_at
        for channel in values:
            stamps[channel] = now


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
        # 5Hz is plenty to confirm a servo moved and to see which outputs are
        # live; faster just burns bandwidth the flight loop needs.
        (mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, 200000),
        (mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS_RAW, 200000),
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


def _wait_for_servo_params(master, timeout=15.0, settle=0.4):
    """Block until the parameter stream has settled enough to read servo data.

    ``param_fetch_all`` only *sends* ``PARAM_REQUEST_LIST``; the 1400-odd
    values stream back afterwards as ``PARAM_VALUE`` messages. Reading
    ``master.params`` straight after the call therefore sees an empty dict and
    discovery concludes "this vehicle has no servo outputs" - which is exactly
    the wrong conclusion on a vehicle that has them.

    There is no completion marker - pymavlink just accumulates - so this waits
    for the download to go *quiet* instead. Returning as soon as the first
    ``SERVOx_FUNCTION`` appeared was not enough: parameters arrive in
    alphabetical order, so ``SERVO1_FUNCTION`` turns up long before
    ``SERVO1_MAX``/``SERVO1_MIN``/``SERVO1_TRIM``, and discovery then found
    the function assignments with no ranges at all - silently degrading the
    read-back prediction to the raw commanded pulse.

    Timing out is normal on a vehicle that does not answer.
    """
    deadline = time.monotonic() + timeout
    last_count = -1
    last_change = time.monotonic()
    seen_function = False

    while time.monotonic() < deadline:
        try:
            params = master.params or {}
        except (AttributeError, TypeError):
            params = {}
        if "SERVO1_FUNCTION" in params:
            seen_function = True

        count = len(params)
        if count != last_count:
            last_count = count
            last_change = time.monotonic()
        elif seen_function and (time.monotonic() - last_change) >= settle:
            return True
        time.sleep(0.05)

    return seen_function


def _read_servo_limits(params, functions):
    """Read the MIN/TRIM/MAX ranges needed to predict mapped output PWM.

    A ``k_rcinN_mapped`` output rescales its RC input through ``pwm_from_angle()``,
    so without these the read-back check would compare a scaled value against an
    unscaled one and report a healthy output as broken. Only the RC inputs that
    are actually mapped are fetched, since ``param_fetch_one`` per input is not
    free.
    """
    limits = {}
    if not params:
        return limits

    for channel, function_id in functions.items():
        servo_range = servo_channels.limits_from_params(params, "SERVO", channel)
        if servo_range is not None:
            limits[("SERVO", channel)] = servo_range

        rc_input = servo_channels.rc_input_for_function(function_id)
        if rc_input is None:
            continue
        rc_range = servo_channels.limits_from_params(params, "RC", rc_input)
        if rc_range is not None:
            limits[("RC", rc_input)] = rc_range
    return limits


def _discover_servo_channels(master):
    """Ask the autopilot what its servo outputs are wired to.

    ``SERVOx_FUNCTION`` is the only source that names a function, so it is what
    identifies the gimbal. ``SERVO_OUTPUT_RAW`` telemetry confirms the output is
    actually moving PWM. Both are read from the parameters already fetched by
    the caller - no second parameter fetch, which would block on the shared
    socket.
    """
    global _servo_functions
    try:
        functions = servo_channels.parse_servo_functions(master.params)
    except Exception:
        functions = {}
    harvested = _read_servo_limits(master.params, functions)

    with _telemetry_lock:
        known = len(_servo_limits)
        _servo_functions = functions
        # Merge, never replace. Discovery runs both from the connect path and
        # from the background safety thread, and the two interleave with
        # separate parameter downloads. A read that caught only SERVOx_FUNCTION
        # but not the MIN/TRIM/MAX arriving later would otherwise publish an
        # empty dict over a complete one, silently downgrading every read-back
        # prediction to the raw commanded pulse.
        _servo_limits.update(harvested)
        limits = dict(_servo_limits)

    if known and not harvested:
        print(
            f"[SERVO] This parameter read returned no servo ranges; keeping the "
            f"{known} already known."
        )

    if not functions:
        print(
            "[SERVO] No SERVOx_FUNCTION parameters available - servo channel "
            "will have to be given explicitly (--servo-channel)"
        )
        return

    channel, label = servo_channels.pick_gimbal_channel(functions)
    # Sorted numerically: a plain sort puts SERVO10 before SERVO1.
    assigned = [
        f"SERVO{ch}=[{fn}] {servo_channels.function_name(fn) or 'unknown'}"
        for ch, fn in sorted(functions.items())
        if fn
    ]
    print(f"[SERVO] Output functions: {' '.join(assigned)}")
    if channel is None:
        print(
            "[SERVO] No mount/gimbal axis assigned to a servo output. "
            "Configure SERVOx_FUNCTION (e.g. SERVO9_FUNCTION=7 for mount "
            "pitch) or pass --servo-channel."
        )
    else:
        print(f"[SERVO] Camera gimbal detected on channel {channel} ({label})")
        _report_mapped_scaling(channel, functions, limits)


def _report_mapped_scaling(channel, functions, limits):
    """Warn when a mapped output rescales its RC input.

    Silence here would be misleading: the operator commands 1500us and the
    vehicle reports something else, which looks like a broken link when in fact
    the scaling is correct. Matching SERVOx_MIN/TRIM/MAX to the RC channel's own
    limits makes the mapping identity, which is the recommended setup.
    """
    rc_input = servo_channels.rc_input_for_function(functions.get(channel))
    if rc_input is None:
        return

    rc_range = limits.get(("RC", rc_input))
    servo_range = limits.get(("SERVO", channel))
    if servo_range is None:
        print(
            f"[SERVO] SERVO{channel}_MIN/TRIM/MAX unknown - cannot predict the "
            f"read-back PWM for the RC input {rc_input} mapping."
        )
        return
    if servo_channels.identity_scaling(rc_range, servo_range):
        print(
            f"[SERVO] SERVO{channel} follows RC input {rc_input} with matched "
            f"limits {servo_range} - commanded pulses pass through unchanged."
        )
        return
    print(
        f"[SERVO] SERVO{channel} follows RC input {rc_input}: RC range {rc_range} "
        f"is scaled to output range {servo_range}, so the vehicle reports a "
        f"different pulse than the one commanded. Set "
        f"SERVO{channel}_MIN/TRIM/MAX and RC{rc_input}_MIN/TRIM/MAX to the same "
        f"values for a 1:1 mapping."
    )


def _record_servo_status(**fields):
    """Merge one servo command's outcome into the reportable status block.

    ``limits_hit`` is derived rather than passed: the output landed somewhere
    other than the pulse the mapping predicts, which is what a servo hitting its
    SERVOx_MIN/MAX end stop looks like from here. A command whose pulse is read
    back exactly is not flagged, even though the two numbers differ, because for
    a rescaled output that is correct rather than clamped.
    """
    expected = fields.get("expected")
    reported = fields.get("reported")
    limits_hit = (
        expected is not None
        and reported is not None
        and int(reported) != int(expected)
    )
    with _telemetry_lock:
        _servo_status.update(fields)
        _servo_status["limits_hit"] = limits_hit


def get_servo_status():
    """What the vehicle did with the last servo command.

    ``reported_pwm`` comes from SERVO_OUTPUT_RAW, not from the commanded value,
    so a ground station can show a disagreement without anyone reading the drone
    console. Returns the same keys the API layer publishes.
    """
    with _telemetry_lock:
        status = dict(_servo_status)
    detected = detect_servo_channel()
    return {
        "detected_channel": status.get("detected_channel", detected),
        "source": status.get("source"),
        "refused": bool(status.get("refused")),
        "refusal_reason": status.get("refusal_reason"),
        "reported_pwm": status.get("reported"),
        "limits_hit": bool(status.get("limits_hit")),
    }


def get_servo_channels():
    """Discovered servo output assignments as ``{channel: function_name}``."""
    with _telemetry_lock:
        functions = dict(_servo_functions)
    return {
        channel: servo_channels.function_name(function_id) or f"function {function_id}"
        for channel, function_id in sorted(functions.items())
        if function_id
    }


def get_servo_channel_info():
    """Everything known about the servo outputs, for diagnostics."""
    with _telemetry_lock:
        return {
            "functions": dict(_servo_functions),
            "pwm": dict(_servo_pwm),
            "rc_channels": dict(_rc_channels),
            "params_loaded": _params_loaded,
            "limits": dict(_servo_limits),
        }


def detect_servo_channel(preferred_function=None):
    """The channel a servo command should address, or None if undetermined.

    Returns None rather than a guess. Falling back to "the first channel that
    looks alive" would happily select a motor or a control surface: channel 1 is
    usually a throttle, so an RC-driven gimbal test on channel 1 would spin a
    prop. Telling the operator that no gimbal output is configured is the only
    safe answer.
    """
    with _telemetry_lock:
        functions = dict(_servo_functions)

    channel, _label = servo_channels.pick_gimbal_channel(
        functions, preferred=preferred_function
    )
    return channel


def _clear_rc_override(rc_input):
    """Release an RC override by writing the unset value into its slot.

    ArduPilot holds an override until it is explicitly released or the vehicle
    reboots, so an override that turned out to land on a live control sticks
    around. Releasing it is how the app backs out of a bad guess.
    """
    master = _master
    if master is not None:
        override = [servo_channels.SERVO_CHANNEL_UNSET] * servo_channels.OVERRIDE_CHANNEL_MAX
        try:
            master.mav.rc_channels_override_send(
                master.target_system,
                master.target_component,
                *override,
            )
        except Exception:
            pass
    _rc_override_inputs.clear()


def _release_rc_overrides(trigger):
    """Release every RC input this app has overridden, and why.

    An override is a live RC input, not a one-shot command: left alone it
    survives LAND, RTL, disarm and the app's own exit, so a gimbal pulse (or a
    bad guess about which input is spare) keeps pinning the receiver for as
    long as the vehicle flies. Every transition out of normal operation -
    landing, returning home, disarming, shutting the link - releases first, and
    the override guard is disarmed at the same time so the mode change these
    transitions cause is never blamed on the write.

    Returns True when a release was actually sent; sends nothing when no
    override is outstanding.
    """
    global _override_guard
    if not _rc_override_inputs and _override_guard is None:
        return False
    _clear_rc_override(None)
    _override_guard = None
    print(f"SITL: RC override released ({trigger})")
    return True


def _arm_override_guard(channel, rc_input):
    """Watch for a flight-mode change in the window after an override write.

    The static checks in ``plan_drive`` cannot enumerate the RC inputs
    ``AP_Copter`` reads for flight control, so an operator-approved override can
    still be the mode channel on a vehicle we misread. The symptom is immediate
    and unmistakable on a real bench - a gimbal sweep that walks the aircraft
    through RTL, STABILIZE, AUTO, CIRCLE and LAND - so it is worth catching
    after the fact: if the mode moves within the window, release the override and
    say so, rather than leaving a live control pinned to a camera angle.
    """
    global _override_guard
    with _telemetry_lock:
        mode_now = _cached_mode
    _override_guard = {
        "channel": channel,
        "rc_input": rc_input,
        "mode_before": mode_now,
        "armed_at": time.monotonic(),
    }


def _check_override_guard(mode_name):
    """Called on every mode report; releases the override if the mode moved."""
    global _override_guard
    guard = _override_guard
    if guard is None or not mode_name:
        return
    if mode_name == guard["mode_before"]:
        return
    if time.monotonic() - guard["armed_at"] > OVERRIDE_GUARD_SECONDS:
        _override_guard = None  # too old to be ours
        return

    _override_guard = None
    rc_input = guard["rc_input"]
    channel = guard["channel"]
    _clear_rc_override(rc_input)
    reason = (
        f"Servo ch{channel} wrote RC input {rc_input} and the vehicle mode "
        f"changed {guard['mode_before']} -> {mode_name} straight afterwards, so "
        f"that input is flight control, not a spare. Override released. Drive "
        f"the output with MAV_CMD_DO_SET_SERVO (SERVO{channel}_FUNCTION = 0) "
        f"instead."
    )
    print(f"SITL: {reason}")
    _record_servo_status(refused=True, refusal_reason=reason)


def _reread_servo_assignment(master, channel, functions):
    """Re-read ``SERVOx_FUNCTION`` from the vehicle when the cache has no answer.

    ``_servo_functions`` is written by connect-time discovery, which can settle
    while the parameter download is still streaming. The cache then says
    "unknown" for a channel ``master.params`` already carries, and the refusal
    that follows is a stale read rather than anything the autopilot said - the
    race seen on the live bench with ``SERVO10_FUNCTION``.

    A miss here re-reads the live parameter dict once and merges it, so the
    caller plans against what the vehicle reports now. When the live read has
    no answer either, nothing changes and the caller still refuses.
    """
    global _servo_functions
    try:
        params = master.params or {}
        parsed = servo_channels.parse_servo_functions(params)
    except Exception:
        return None, functions
    function_id = parsed.get(channel)
    if function_id is None:
        return None, functions

    merged = dict(functions)
    merged.update(parsed)
    harvested = _read_servo_limits(params, merged)
    with _telemetry_lock:
        _servo_functions = merged
        _servo_limits.update(harvested)
    print(
        f"[SERVO] SERVO{channel}_FUNCTION = {function_id} read from the live "
        f"parameter stream - the discovery cache was stale."
    )
    return function_id, merged


def _send_servo_command(channel, pulse, allow_rc_override=False):
    """Command one output, by whatever mechanism its assignment allows.

    Which mechanism that is depends entirely on ``SERVOx_FUNCTION``, and the
    autopilot is unforgiving about it: ``MAV_CMD_DO_SET_SERVO`` refuses any
    channel assigned a mount function ("Channel N is already in use"), while a
    ``k_rcinN_mapped`` output is reachable only through
    ``RC_CHANNELS_OVERRIDE`` on the RC input it was mapped to. A plain
    ``k_rcinN`` pass-through (51-66) is whitelisted for ``DO_SET_SERVO`` as
    well, which is how an output whose RC input lies beyond the override's
    8 slots - e.g. ``SERVO10_FUNCTION = 61`` - is reached. Picking the wrong
    mechanism does nothing at all, silently.
    """
    master = _get_master()

    with _telemetry_lock:
        function_id = _servo_functions.get(channel)
        functions = dict(_servo_functions)

    if function_id is None:
        # Cache miss, not a verdict: discovery may have settled mid-download,
        # so ask the vehicle's own parameter dict before refusing anything.
        function_id, functions = _reread_servo_assignment(master, channel, functions)

    method, rc_input = servo_channels.plan_drive(
        channel, function_id, functions, allow_rc_override=allow_rc_override
    )

    if method == servo_channels.DRIVE_RC_OVERRIDE:
        # The override addresses RC *inputs*, not outputs: an output on channel
        # 12 mapped to RC input 5 is driven by writing slot 5. This is what
        # makes aux outputs reachable at all, since the message only carries 8
        # slots.
        override = [servo_channels.SERVO_CHANNEL_UNSET] * servo_channels.OVERRIDE_CHANNEL_MAX
        override[rc_input - 1] = pulse
        _arm_override_guard(channel, rc_input)
        master.mav.rc_channels_override_send(
            master.target_system,
            master.target_component,
            *override,
        )
        _rc_override_inputs.add(rc_input)
        return True

    if method == servo_channels.DRIVE_DO_SET_SERVO:
        with _telemetry_lock:
            _acked_commands.pop(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, None)
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            channel,
            pulse,
            0, 0, 0, 0, 0,
        )
        ack = _wait_command_ack(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, timeout=2.0)
        if ack is None:
            print(
                f"SITL: Servo ch{channel} sent but the vehicle did not acknowledge "
                f"within 2s; the output may be unmapped"
            )
            return False
        if ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print(f"SITL: Servo ch{channel} rejected (result {ack.result})")
            return False
        return True

    if function_id is None:
        # The live parameters have no SERVOx_FUNCTION for this output either,
        # so this is a real unknown and not the stale cache. Not the same as
        # "assigned to something undrivable" - the parameter download has not
        # finished or the vehicle did not answer. Guessing here is what broke
        # the original design, so say so instead.
        raise ValueError(
            f"SERVO{channel}_FUNCTION is unknown - the autopilot's parameter "
            f"list has not been read, so there is no safe way to know whether "
            f"this output is a gimbal or a motor. Wait for discovery to "
            f"finish, or pass a channel whose assignment is known."
        )

    if method == servo_channels.DRIVE_MOUNT_PROTOCOL:
        rc_input = servo_channels.rc_input_for_gimbal(channel)
        hint = ""
        if rc_input is not None:
            mapped_fn = servo_channels.RC_MAPPED_FUNCTION_MIN + rc_input - 1
            hint = (
                f" Assign SERVO{channel}_FUNCTION = {mapped_fn} to make it "
                f"follow RC input {rc_input}, or drive it through the mount "
                f"protocol."
            )
        raise ValueError(
            f"SERVO{channel}_FUNCTION is "
            f"{servo_channels.function_name(function_id)}, which the mount "
            f"backend owns - ArduPilot refuses direct PWM commands to it."
            f"{hint}"
        )

    if method == servo_channels.DRIVE_RESERVED_INPUT:
        # The refusal that matters most. RC_CHANNELS_OVERRIDE writes RC inputs,
        # and on inputs 1-5 ArduCopter is reading throttle/roll/pitch/yaw and the
        # flight-mode channel, so a gimbal sweep sent there walks the aircraft
        # through its mode slots instead of moving a camera.
        owner = servo_channels.function_name(functions.get(rc_input))
        detail = (
            f"which is assigned {owner}"
            if owner
            else "whose assignment is unknown"
        )
        if rc_input in servo_channels.PRIMARY_RC_INPUTS:
            because = (
                f"RC input {rc_input} is throttle/roll/pitch/yaw/flight-mode on "
                f"ArduCopter and is read straight from the RC stream"
            )
        else:
            because = (
                f"RC input {rc_input} already drives output {rc_input}, {detail}"
            )
        spare = [n for n in range(1, servo_channels.OVERRIDE_CHANNEL_MAX + 1)
                 if servo_channels.rc_input_is_spare(functions, n)]
        hint = ""
        if spare:
            hint = (
                f" Map it to a spare input instead: SERVO{channel}_FUNCTION = "
                f"{servo_channels.RC_MAPPED_FUNCTION_MIN + spare[-1] - 1} uses "
                f"RC input {spare[-1]}."
            )
        raise ValueError(
            f"Servo ch{channel} is {servo_channels.function_name(function_id)}, "
            f"so driving it needs RC_CHANNELS_OVERRIDE on RC input {rc_input} - "
            f"but {because}. Refusing rather than commanding the aircraft."
            f"{hint}"
        )

    if method == servo_channels.DRIVE_UNVERIFIED_INPUT:
        # Every static check passed, and that is still not enough.
        # SERVOx_FUNCTION says which input an output *follows*; it says nothing
        # about which inputs AP_Copter reads for flight control, and that set
        # comes from the channel mapping and the frame. A real bench mapped ch9
        # to an input outside 1-5, every check here passed, and sweeping it
        # walked the aircraft through RTL, STABILIZE, AUTO, CIRCLE and LAND - so
        # the input was the mode channel after all and nothing in the parameter
        # map admitted it. Only the operator knows their own wiring, so the
        # override is opt-in rather than inferred.
        raise ValueError(
            f"Servo ch{channel} is {servo_channels.function_name(function_id)}, "
            f"so driving it needs RC_CHANNELS_OVERRIDE on RC input {rc_input}. "
            f"That input passes every check this app can make statically, but "
            f"RC_CHANNELS_OVERRIDE writes RC inputs and ArduPilot reads some of "
            f"them straight into flight control - including the mode channel, "
            f"whose position depends on the channel mapping and frame, not on "
            f"SERVOx_FUNCTION. Refusing rather than sweeping the aircraft "
            f"through its flight modes. If you have confirmed RC input "
            f"{rc_input} is spare on this vehicle, re-run with "
            f"--allow-rc-override; the app also watches for a mode change after "
            f"each write and releases the override if it sees one. Driving the "
            f"output directly needs no override at all: set "
            f"SERVO{channel}_FUNCTION = 0."
        )

    raise ValueError(
        f"Servo ch{channel} is assigned "
        f"{servo_channels.function_name(function_id) or f'SERVO{channel}_FUNCTION={function_id}'}, "
        f"which is not a drivable output. Refusing rather than moving a "
        f"surface that is not a camera."
    )


def _verify_servo_pulse(channel, pulse, timeout=2.0, since=None):
    """Compare the commanded pulse against SERVO_OUTPUT_RAW.

    ``since`` is a ``time.monotonic()`` stamp taken before the command was sent.
    A reading older than that describes the *previous* command, so comparing
    against it reports a false failure: with a 1s timeout the cache still held
    the last value from the previous call and every command looked wrong. Only
    readings that arrived after the command count.

    The expected value is predicted rather than assumed. A ``k_rcinN_mapped``
    output rescales the RC pulse through ``pwm_from_angle()``, so its PWM is not
    the commanded pulse unless the output's MIN/TRIM/MAX match the RC channel's
    (see ``servo_channels.mapped_output_pulse``). When the prediction is
    unavailable - limits not read yet - the raw pulse is compared and the
    caveat is reported rather than hidden.

    A channel that has never reported is unverified, not a failure: the stream
    may simply not be flowing.
    """
    expected = _expected_servo_pulse(channel, pulse)
    deadline = time.monotonic() + timeout
    previous_fresh = None

    while time.monotonic() < deadline:
        with _telemetry_lock:
            reported = _servo_pwm.get(channel)
            arrived = _servo_pwm_at.get(channel)
        if reported is not None:
            fresh = since is None or arrived is None or arrived > since
            if fresh:
                if servo_channels.pulse_matches(expected, reported):
                    return True
                if servo_channels.pulse_matches(pulse, reported):
                    return True
                # No match, but the output can still be mid-travel. Only two
                # consecutive *fresh* identical readings mean it has settled on
                # the wrong value - a stale repeat is not evidence, since the
                # cache holds the previous command's value and would look
                # "stable" for as long as we waited.
                if previous_fresh == reported:
                    _report_pulse_mismatch(channel, pulse, expected, reported)
                    return False
                previous_fresh = reported
        time.sleep(0.05)
    return None


def _expected_servo_pulse(channel, pulse):
    """The PWM this output should report after being commanded ``pulse``."""
    with _telemetry_lock:
        function_id = _servo_functions.get(channel)
    rc_input = servo_channels.rc_input_for_function(function_id)
    if rc_input is None:
        return pulse  # k_none / manual output reports what we asked for

    predicted = servo_channels.mapped_output_pulse(
        pulse,
        _servo_limits.get(("RC", rc_input)),
        _servo_limits.get(("SERVO", channel)),
    )
    return predicted if predicted is not None else pulse


def _report_pulse_mismatch(channel, commanded, expected, reported):
    note = ""
    if expected != commanded:
        note = (
            f" (scaled: {expected}us is what SERVO{channel} should report given "
            f"the RC and output limits)"
        )
    print(
        f"SITL: Servo ch{channel} commanded {commanded}us but the vehicle "
        f"reports {reported}us - check SERVO{channel}_FUNCTION and "
        f"SERVO{channel}_MIN/MAX{note}"
    )


def _check_safety_params(master):
    """Warn loudly about FCU parameters that make autonomous flight unsafe.

    The Python-side GPS/EKF checks in the UI are not a substitute for the
    autopilot's own arming checks - those are the last line of defence.

    Runs on a daemon thread: the parameter download takes seconds and a vehicle
    that never answers must not stall the connect.
    """
    def work():
        global _params_loaded
        try:
            master.param_fetch_all()
        except Exception:
            return
        _wait_for_servo_params(master)
        _discover_servo_channels(master)
        with _telemetry_lock:
            _params_loaded = True
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


def _send_arm(master):
    """Send a force-arm request (MAV_CMD_COMPONENT_ARM_DISARM, param2=21196).

    The 21196 value is ArduPilot's documented "force arm": the command is
    accepted with the autopilot's pre-arm checks skipped entirely. The
    operator runs those checks separately in Mission Planner instead.
    """
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        21196,
        0,
        0,
        0,
        0,
        0,
    )


def arm():
    """Arm the vehicle in GUIDED and wait until the FCU reports it armed.

    Always force-armed: the autopilot's own pre-arm checks are skipped by
    design (the operator validates them in Mission Planner first).
    """
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

    _send_arm(master)
    _wait_for_arm_state(True, timeout=20.0)
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
    _release_rc_overrides("LAND")
    _set_mode("LAND")
    print("SITL: Landing")


def disarm():
    """Disarm the vehicle and wait for the FCU to report it disarmed.

    Only safe on the ground - the caller is responsible for that check.
    """
    global _cached_armed
    master = _get_master()
    _release_rc_overrides("disarm")
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
    _release_rc_overrides("RTL")
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


def wait_for_position(timeout=8.0, poll=0.1):
    """Block until the first GLOBAL_POSITION_INT has been cached.

    ``wait_for_navigation_telemetry`` only reports GPS_RAW_INT and
    EKF_STATUS_REPORT, which can land before a position fix is cached - so
    reading ``get_position()`` straight after it can still return the 0.0 cold
    cache. Recording that as the home altitude then poisons every relative
    altitude for the whole run: disarm is refused as "airborne at 584m" and the
    follow altitude gate passes on the ground.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _telemetry_lock:
            if _position_seen:
                return True
        time.sleep(poll)
    with _telemetry_lock:
        return _position_seen


def telemetry_age():
    """Seconds since any MAVLink message arrived from the vehicle, or None."""
    stamp = _last_telemetry_at
    if stamp == 0.0:
        return None
    return time.monotonic() - stamp


def disconnect():
    """Release the MAVLink link and stop the listener.

    The vehicle keeps flying (or landing) on its own after this - the point is
    to close the port deterministically on shutdown instead of leaving it to
    process teardown, so the port is free for another tool immediately. Any
    outstanding RC override goes first: after the link is gone nothing can
    release it, and the vehicle would keep that RC input pinned.
    """
    global _master, _message_listener_running
    _release_rc_overrides("shutdown")
    _message_listener_running = False
    master = _master
    _master = None
    if master is not None:
        try:
            master.close()
        except Exception:
            pass


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


def send_servo(channel=None, pulse=1500, verify=True, source=None, allow_rc_override=False):
    """Drive one servo output to a pulse width.

    ``channel=None`` means "the gimbal channel the autopilot reported", so a
    ground station that does not know its own wiring does not have to be told.

    Channels 1-16 are accepted, which the old 1-8 limit did not. How a given
    channel is reached depends on its ``SERVOx_FUNCTION`` - see
    ``_send_servo_command`` and ``servo_channels.plan_drive`` - so an output the
    autopilot will not let us drive is refused with the reason rather than
    silently ignored.

    Every outcome is recorded for ``get_servo_status``: a refusal is as
    interesting to the operator as a success, and UDP gives the ground station
    no other way to find out.
    """
    detected = None
    try:
        if channel is None:
            detected = detect_servo_channel()
            channel = detected
            if channel is None:
                raise ValueError(
                    "No servo channel given and the vehicle reported no gimbal "
                    "output; pass an explicit channel or set SERVOx_FUNCTION"
                )

        channel = int(channel)
        if not servo_channels.SERVO_CHANNEL_MIN <= channel <= servo_channels.SERVO_CHANNEL_MAX:
            raise ValueError(
                f"Servo channel {channel} out of range "
                f"{servo_channels.SERVO_CHANNEL_MIN}-{servo_channels.SERVO_CHANNEL_MAX}"
            )
        pulse = servo_channels.clamp_pulse(pulse)

        # Stamped before sending so verification can ignore PWM reports that were
        # already in the cache and describe the previous command.
        sent_at = time.monotonic()
        _send_servo_command(channel, pulse, allow_rc_override=allow_rc_override)

        verified = None
        if verify:
            verified = _verify_servo_pulse(channel, pulse, since=sent_at)

        with _telemetry_lock:
            reported = _servo_pwm.get(channel)
        expected = _expected_servo_pulse(channel, pulse)
        _record_servo_status(
            channel=channel,
            detected_channel=detected if detected is not None else channel,
            source=source,
            commanded=pulse,
            expected=expected,
            reported=reported,
            verified=verified,
            # Cleared explicitly: the status block is global, so a previous
            # refusal would otherwise still be reported against this command.
            refused=False,
            refusal_reason=None,
        )
        return channel
    except (TypeError, ValueError) as exc:
        _record_servo_status(
            channel=channel if isinstance(channel, int) else None,
            detected_channel=detected,
            source=source,
            refused=True,
            refusal_reason=str(exc),
        )
        raise


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

