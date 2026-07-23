from modules.drone_backend import sitl
from modules.drone_backend.mock_vehicle import MockVehicle

_backend_name = "mock"


def set_backend(backend_name):
    global _backend_name
    _backend_name = backend_name


def _get_backend():
    if _backend_name == "sitl":
        return sitl
    return None


def connect_drone(connection_string, waitready=True, baud=57600):
    backend = _get_backend()
    if backend is not None:
        return backend.connect_drone(connection_string, waitready=waitready, baud=baud)
    print(f"Mock: Connecting to drone at {connection_string}")
    return MockVehicle()


def arm_and_takeoff(max_height):
    backend = _get_backend()
    if backend is not None:
        return backend.arm_and_takeoff(max_height)
    print(f"Mock: Arm and takeoff to {max_height}m")


def land():
    backend = _get_backend()
    if backend is not None:
        return backend.land()
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
    return "Mock: Battery 100%"


def get_version():
    backend = _get_backend()
    if backend is not None:
        return backend.get_version()
    return "Mock: Version 1.0"


def send_movement_command_YAW(angle):
    backend = _get_backend()
    if backend is not None:
        return backend.send_movement_command_YAW(angle)
    direction = "RIGHT" if angle > 0 else "LEFT" if angle < 0 else "STOP"
    print(f"Mock: Yaw command {angle:.2f} deg/s -> Rotating {direction}")


def send_movement_command_XYA(x, y, altitude):
    backend = _get_backend()
    if backend is not None:
        return backend.send_movement_command_XYA(x, y, altitude)
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
