from modules.drone_backend import servo_channels
from modules.drone_backend import sitl
from modules.drone_backend.mock_vehicle import MockVehicle

_backend_name = "mock"
_vehicle = None


def set_backend(backend_name):
    global _backend_name
    _backend_name = backend_name


def _get_backend():
    if _backend_name in ("sitl", "flight"):
        return sitl
    return None


def connect_drone(connection_string, waitready=True, baud=57600, start_sitl=False):
    global _vehicle
    backend = _get_backend()
    if backend is not None:
        try:
            _vehicle = backend.connect_drone(connection_string, waitready=waitready, baud=baud, start_sitl=start_sitl)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return False
        return True
    print(f"Mock: Connecting to drone at {connection_string}")
    _vehicle = MockVehicle()
    return True


def stop_sitl():
    backend = _get_backend()
    if backend is not None and hasattr(backend, 'stop_sitl'):
        backend.stop_sitl()


def arm():
    backend = _get_backend()
    if backend is not None and hasattr(backend, 'arm'):
        return backend.arm()
    print("Mock: Armed")


def takeoff(max_height):
    backend = _get_backend()
    if backend is not None and hasattr(backend, 'takeoff'):
        return backend.takeoff(max_height)
    print(f"Mock: Takeoff to {max_height}m")


def arm_and_takeoff(max_height):
    arm()
    takeoff(max_height)


def land():
    backend = _get_backend()
    if backend is not None:
        return backend.land()
    if _vehicle is not None:
        _vehicle.land()
    else:
        print("Mock: Landing")


def disarm():
    backend = _get_backend()
    if backend is not None and hasattr(backend, 'disarm'):
        return backend.disarm()
    print("Mock: Disarmed")


def is_armed():
    backend = _get_backend()
    if backend is not None:
        return backend.is_armed()
    if _vehicle is not None:
        return _vehicle.is_armed()
    return False


def get_mode():
    backend = _get_backend()
    if backend is not None:
        return backend.get_mode()
    if _vehicle is not None:
        return _vehicle.get_mode()
    return "UNKNOWN"


def get_gps_fix_type():
    backend = _get_backend()
    if backend is not None:
        return backend.get_gps_fix_type()
    if _vehicle is not None:
        return _vehicle.get_gps_fix_type()
    return 0


def wait_for_navigation_telemetry(timeout=8.0):
    """Wait for the first GPS/EKF reports; False if none arrive in time.

    A cold telemetry cache is indistinguishable from a broken one, so callers
    that gate on GPS/EKF quality must wait for real data first.
    """
    backend = _get_backend()
    if backend is not None and hasattr(backend, "wait_for_navigation_telemetry"):
        return backend.wait_for_navigation_telemetry(timeout=timeout)
    if _vehicle is not None and hasattr(_vehicle, "wait_for_navigation_telemetry"):
        return _vehicle.wait_for_navigation_telemetry(timeout=timeout)
    return True


def wait_for_position(timeout=8.0):
    """Wait for the first cached position fix; False if none arrives in time.

    Needed before anything records a home altitude: GPS/EKF telemetry can be
    reported while the position cache is still the 0.0 cold value.
    """
    backend = _get_backend()
    if backend is not None and hasattr(backend, "wait_for_position"):
        return backend.wait_for_position(timeout=timeout)
    return True


def telemetry_age():
    """Seconds since the last message from the vehicle, or None if unknown."""
    backend = _get_backend()
    if backend is not None and hasattr(backend, "telemetry_age"):
        return backend.telemetry_age()
    return None


def disconnect():
    """Close the MAVLink link; harmless when never opened."""
    backend = _get_backend()
    if backend is not None and hasattr(backend, "disconnect"):
        backend.disconnect()


def is_ekf_ok():
    backend = _get_backend()
    if backend is not None:
        return backend.is_ekf_ok()
    if _vehicle is not None:
        return _vehicle.is_ekf_ok()
    return False


def get_EKF_status():
    backend = _get_backend()
    if backend is not None:
        return backend.get_EKF_status()
    return "Mock: EKF status OK"


def get_battery_info():
    backend = _get_backend()
    if backend is not None:
        return backend.get_battery_info()
    if _vehicle is not None:
        return f"Mock: Battery {_vehicle.get_battery_level()}%"
    return "Mock: Battery 100%"


def get_version():
    backend = _get_backend()
    if backend is not None:
        return backend.get_version()
    return "Mock: Version 1.0"


def get_position():
    backend = _get_backend()
    if backend is not None:
        return backend.get_position()
    if _vehicle is not None:
        return _vehicle.get_gps_position()
    return 0.0, 0.0, 0.0


def get_battery_level():
    backend = _get_backend()
    if backend is not None:
        return backend.get_battery_level()
    if _vehicle is not None:
        return _vehicle.get_battery_level()
    return 100


def send_movement_command_YAW(angle):
    backend = _get_backend()
    if backend is not None:
        return backend.send_movement_command_YAW(angle)


def send_movement_command_XYA(x, y, altitude):
    backend = _get_backend()
    if backend is not None:
        return backend.send_movement_command_XYA(x, y, altitude)


def hold_position():
    backend = _get_backend()
    if backend is not None:
        return backend.hold_position()


def send_servo(channel=None, pulse=1500, verify=True, source=None,
               allow_rc_override=False):
    """Drive a servo output. ``channel=None`` uses the discovered gimbal channel.

    Validation lives here rather than in the backend so mock mode enforces the
    same 1-16 range as a real vehicle - otherwise the mock would happily report
    success for a command the FCU would reject.

    ``source`` records how the channel was chosen ("auto" or "explicit") so the
    outcome can be reported back to the ground station; see ``get_servo_status``.

    ``allow_rc_override`` is the operator's assertion that a mapped output's RC
    input is spare on this vehicle. It is False by default because no static
    check can identify the mode channel; see ``servo_channels.plan_drive``.
    """
    backend = _get_backend()
    if backend is not None and hasattr(backend, "send_servo"):
        return backend.send_servo(channel=channel, pulse=pulse, verify=verify,
                                  source=source,
                                  allow_rc_override=allow_rc_override)

    resolved = servo_channels.resolve_channel(channel, detected=None)
    pulse = servo_channels.clamp_pulse(pulse)
    if resolved is None:
        # Mock mode has no autopilot to detect from, so it falls back to the
        # historical default rather than refusing every command.
        resolved = 8
    print(f"Mock: Servo on channel {resolved} at {pulse}us")
    return resolved


def get_servo_channels():
    """Servo output assignments the autopilot reported, {channel: function}."""
    backend = _get_backend()
    if backend is not None and hasattr(backend, "get_servo_channels"):
        return backend.get_servo_channels()
    return {}


def get_servo_channel_info():
    """Full servo discovery state: functions, live PWM, RC channel count."""
    backend = _get_backend()
    if backend is not None and hasattr(backend, "get_servo_channel_info"):
        return backend.get_servo_channel_info()
    return {"functions": {}, "pwm": {}, "rc_channels": {}, "params_loaded": False}


def detect_servo_channel(preferred_function=None):
    """The gimbal channel the autopilot reported, or None if undetermined."""
    backend = _get_backend()
    if backend is not None and hasattr(backend, "detect_servo_channel"):
        return backend.detect_servo_channel(preferred_function=preferred_function)
    return None


# Keys the ground station expects under telemetry.servo. Kept here so the
# backend and the telemetry builder cannot drift apart.
SERVO_STATUS_KEYS = (
    "detected_channel",
    "source",
    "refused",
    "refusal_reason",
    "reported_pwm",
    "limits_hit",
)


def empty_servo_status():
    """A servo status block with every key present and nothing claimed."""
    return {
        "detected_channel": None,
        "source": None,
        "refused": False,
        "refusal_reason": None,
        "reported_pwm": None,
        "limits_hit": False,
    }


def get_servo_status():
    """Outcome of the last servo command, for the telemetry packet.

    Reports what the vehicle did, not what was asked: ``reported_pwm`` is read
    back from the FCU, so a ground station can see a disagreement without
    watching the drone console.
    """
    status = empty_servo_status()
    backend = _get_backend()
    if backend is not None and hasattr(backend, "get_servo_status"):
        status.update(backend.get_servo_status())
    return status


def send_rtl():
    backend = _get_backend()
    if backend is not None:
        return backend.send_rtl()
    print("Mock: RTL requested")
