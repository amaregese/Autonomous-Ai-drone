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


def arm_and_takeoff(max_height):
    backend = _get_backend()
    if backend is not None:
        return backend.arm_and_takeoff(max_height)
    print(f"Mock: Arm and takeoff to {max_height}m")


def land():
    backend = _get_backend()
    if backend is not None:
        return backend.land()
    if _vehicle is not None:
        _vehicle.land()
    else:
        print("Mock: Landing")


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
