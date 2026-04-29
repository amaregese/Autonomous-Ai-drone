from src.ui.drone_visualizer import DroneVisualizer

from src.core.control_system import state


def initialize_debug_logs(debug_filepath):
    import os

    os.makedirs(os.path.dirname(debug_filepath), exist_ok=True)
    try:
        state.visualizer = DroneVisualizer()
        print("✓ Drone visualizer initialized")
    except Exception as exc:
        print(f"Visualizer error: {exc}")


def update_visualizer_target(x, z, name="Target", confidence=0, distance=0):
    if state.visualizer:
        state.visualizer.set_target(x, z, name, confidence, distance)


def update_telemetry_from_track(fps, yaw, forward, lidar_on_target, x_delta, y_delta):
    if state.visualizer:
        state.visualizer.update_telemetry(fps, yaw, forward, lidar_on_target, x_delta, y_delta)


def draw_visualizer():
    if state.visualizer:
        state.visualizer.draw()
        return state.visualizer.should_quit()
    return False


def close_visualizer():
    if state.visualizer:
        state.visualizer.close()
        state.visualizer = None

