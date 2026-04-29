from src.core import drone
from src.core.control_system import state
from src.core.control_system.pid import configure_PID
from src.core.control_system.visualizer import (
    close_visualizer,
    draw_visualizer,
    initialize_debug_logs,
    update_telemetry_from_track,
    update_visualizer_target,
)


def connect_drone(drone_location):
    drone.connect_drone(drone_location)


def getMovementYawAngle():
    return state.movementYawAngle


def setXdelta(x_delta):
    state.inputValueYaw = x_delta


def getMovementVelocityXCommand():
    return state.movementRollAngle


def setZDelta(z_delta):
    state.inputValueVelocityX = z_delta


def set_system_state(current_state):
    state.state = current_state


def set_flight_altitude(alt):
    state.flight_altitude = alt


def arm_and_takeoff(max_height):
    drone.arm_and_takeoff(max_height)


def land():
    drone.land()


def print_drone_report():
    print(drone.get_EKF_status())
    print(drone.get_battery_info())
    print(drone.get_version())


def get_visualizer():
    return state.visualizer


def set_visualizer_status(message, color=(0, 255, 0), duration=0):
    if state.visualizer:
        state.visualizer.set_status(message, color, duration)


def control_drone():
    if state.inputValueYaw == 0:
        drone.send_movement_command_YAW(0)
        state.movementYawAngle = 0
    else:
        state.movementYawAngle = state.pidYaw(state.inputValueYaw) * -1
        drone.send_movement_command_YAW(state.movementYawAngle)
        if state.visualizer:
            state.visualizer.update(state.movementYawAngle * 0.1, 0)

    if state.inputValueVelocityX == 0:
        drone.send_movement_command_XYA(0, 0, state.flight_altitude)
        state.movementRollAngle = 0
    else:
        state.movementRollAngle = state.pidRoll(state.inputValueVelocityX) * -1
        drone.send_movement_command_XYA(state.movementRollAngle, 0, state.flight_altitude)
        if state.visualizer:
            state.visualizer.update(0, state.movementRollAngle * 0.1)

    if state.visualizer:
        state.visualizer.draw()


def stop_drone():
    drone.send_movement_command_YAW(0)
    drone.send_movement_command_XYA(0, 0, state.flight_altitude)
    if state.visualizer:
        state.visualizer.update(0, 0)
