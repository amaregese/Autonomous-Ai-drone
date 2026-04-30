import sys
import time
import argparse
import cv2

sys.path.insert(0, '.')
sys.path.insert(1, 'src')

from src.sensors import lidar
from src.core import control
from src.perception.detectors import yolo11_detector as detector
from src.core import drone
from src.ui.app_config import MAX_ALT, TRACKING_LOST_THRESHOLD
from src.ui.display import (
    WINDOW_NAME,
    annotate_selection_overlay,
    annotate_tracking_overlay,
    draw_detection_window,
    update_drone_visualizer_status,
)
from src.navigation import FollowController
from src.perception.tracking import TrackingSession
import keyboard

# Args parser
parser = argparse.ArgumentParser(description='Drive autonomous with dual-model support')
parser.add_argument('--debug_path', type=str, default="debug/run1")
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--control', type=str, default='PID')
parser.add_argument('--drone_connection', type=str, default=None)
parser.add_argument('--input-source', type=str, default='camera', choices=['camera', 'video'], help='Input source (default: camera)')
parser.add_argument('--video-path', type=str, default=None, help='Path to video file (if input-source=video)')

# Model selection arguments
parser.add_argument('--general-model', type=str, default='models/yolo11n.pt',
                    help='Path to general YOLO model')
parser.add_argument('--laser-model', type=str, default='models/yolo11_laser_points.pt',
                    help='Path to custom laser detection model')
parser.add_argument('--model-mode', type=str, default='combined',
                    choices=['general', 'laser', 'combined'],
                    help='Detection mode: general, laser, or combined')

# Detection thresholds
parser.add_argument('--conf-threshold', type=float, default=None,
                    help='Confidence threshold for general objects')
parser.add_argument('--laser-conf-threshold', type=float, default=0.12,
                    help='Confidence threshold for laser points')
parser.add_argument('--iou-threshold', type=float, default=None)
parser.add_argument('--min-box-area-ratio', type=float, default=None)
parser.add_argument('--imgsz', type=int, default=None)

args = parser.parse_args()

STATE = "takeoff"
tracking_session = TrackingSession()
follow_controller = FollowController()
telemetry_log_counter = 0


def setup():
    """Initialize all systems with dual-model support"""
    drone.set_backend("sitl" if args.mode == "sitl" else "mock")
    connection_string = args.drone_connection

    print("=" * 70)
    print("AUTONOMOUS DRONE OBJECT FOLLOWING SYSTEM")
    print("=" * 70)
    print(f"General Model: {args.general_model}")
    print(f"Laser Model: {args.laser_model}")
    print(f"Detection Mode: {args.model_mode}")
    print("=" * 70 + "\n")

    print("Connecting LiDAR...")
    lidar.connect_lidar("/dev/ttyTHS1")

    print("Setting up detector...")

    # Configure thresholds for both models
    detector.configure_detector(
        confidence_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        min_box_area_ratio=args.min_box_area_ratio,
        inference_img_size=args.imgsz,
        laser_confidence_threshold=args.laser_conf_threshold,
    )

    # Initialize with both models
    success = detector.initialize_detector(
        model_path=args.general_model,
        laser_model_path=args.laser_model,
        source=args.input_source,
        video_path=args.video_path
    )

    if not success:
        print("❌ Failed to initialize detector!")
        sys.exit(1)

    # Set detection mode
    detector.set_model_mode(args.model_mode)

    # Print available models info
    models_info = detector.get_available_models()
    print(f"\n✅ General model loaded: {models_info['general']['classes']} classes")
    if models_info['laser']['loaded']:
        print(f"✅ Laser model loaded")
    else:
        print("⚠️ Laser model not loaded - running in general detection only mode")

    print("\nConnecting to drone...")
    if connection_string is None:
        if args.mode == "flight":
            connection_string = '/dev/ttyACM0'
        elif args.mode == "sitl":
            connection_string = 'udpin:0.0.0.0:14550'
        else:
            connection_string = '192.168.1.96:14550'

    control.connect_drone(connection_string)
    control.set_flight_altitude(MAX_ALT)


def main_loop():
    global telemetry_log_counter

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, tracking_session.handle_mouse_event)
    tracking_session.reset_loss_state()

    while True:
        if keyboard.is_pressed('q'):
            land()
            break

        # Get detections from dual models
        detections, fps, image = detector.get_detections()
        if image is None:
            time.sleep(0.01)
            continue

        height, width = image.shape[:2]

        # Draw detection window (clean, no mode indicators)
        image = draw_detection_window(image, detections, detector)

        # Handle mouse clicks for selection
        tracking_session.process_click(detections, detector, control)

        # Get current tracking state
        selected_obj = detector.get_selected_object()
        selected_class = detector.get_selected_class()
        tracking_lost = detector.get_tracking_status()
        tracking_conf = detector.get_tracking_confidence()

        # Process tracking and movement commands
        if selected_obj is not None:
            # Compute movement commands
            movement = follow_controller.compute_follow_command(selected_obj, image.shape)

            # Telemetry logging
            telemetry_log_counter += 1
            if telemetry_log_counter % 15 == 0:
                print(f"\n📊 Tracking Status:")
                print(f"  Class: {selected_obj.class_name} | Conf: {selected_obj.confidence:.3f}")
                print(f"  Object size: {movement['object_area']}px ({movement['normalized_size']:.3f})")
                print(f"  Distance: {movement['lidar_dist']:.2f}m | Error: {movement['distance_error']:.2f}m")
                print(f"  Speed: {movement['vel_z']:.2f}m/s | Yaw: {movement['yaw_cmd']:.2f} deg/s")

            # Update visualizer
            control.update_visualizer_target(
                movement["target_x"],
                movement["target_z"],
                selected_obj.class_name,
                selected_obj.confidence,
                movement["lidar_dist"],
            )

            # Send movement commands
            drone.send_movement_command_YAW(movement["yaw_cmd"])
            drone.send_movement_command_XYA(0, movement["vel_z"], MAX_ALT)

            # Update telemetry
            control.update_telemetry_from_track(
                fps,
                movement["yaw_cmd"],
                movement["vel_z"],
                movement["lidar_on_target"],
                movement["x_delta"],
                movement["y_delta"],
            )

            # Show loss status
            tracking_session.show_lost_status_once(
                selected_obj.class_name,
                tracking_conf,
                TRACKING_LOST_THRESHOLD,
                control,
            )

            # Annotate tracking overlay
            annotate_tracking_overlay(image, fps, selected_obj, movement)

        else:
            # No target selected
            control.update_visualizer_target(width // 2, height // 2, "No Target", 0, 0)
            control.update_telemetry_from_track(fps, 0, 0, False, 0, 0)

            if tracking_lost and selected_class:
                tracking_session.show_lost_status_once(
                    selected_class,
                    tracking_conf,
                    101,
                    control,
                )
            elif tracking_session.lost_shown and tracking_session.last_target_name:
                annotate_selection_overlay(image, detections)
            else:
                tracking_session.reset_loss_state()
                annotate_selection_overlay(image, detections)

        # Update visualizer status
        update_drone_visualizer_status(detector, control, TRACKING_LOST_THRESHOLD)

        # Show lost object status
        if selected_obj is None and tracking_session.lost_shown and tracking_session.last_target_name:
            control.set_visualizer_status(
                f"OBJECT LOST: {tracking_session.last_target_name}",
                (0, 0, 255),
                duration=0,
            )

        # Display frame (clean, no extra text)
        cv2.imshow(WINDOW_NAME, image)
        control.draw_visualizer()

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            land()
            break
        elif key == ord('g'):
            detector.set_model_mode('general')
            print("🟢 Switched to GENERAL detection mode")
        elif key == ord('l'):
            if detector.get_available_models()['laser']['loaded']:
                detector.set_model_mode('laser')
                print("🟢 Switched to LASER detection mode")
            else:
                print("❌ Laser model not available")
        elif key == ord('c'):
            detector.set_model_mode('combined')
            print("🟢 Switched to COMBINED detection mode")
        elif key == ord('+') or key == ord('='):
            new_conf = min(0.95, (args.conf_threshold or 0.3) + 0.05)
            detector.configure_detector(confidence_threshold=new_conf)
            print(f"📈 Confidence threshold: {new_conf:.2f}")
        elif key == ord('-') or key == ord('_'):
            new_conf = max(0.05, (args.conf_threshold or 0.3) - 0.05)
            detector.configure_detector(confidence_threshold=new_conf)
            print(f"📉 Confidence threshold: {new_conf:.2f}")

    return "land"


def takeoff():
    print("\n🚁 TAKEOFF")
    control.print_drone_report()
    control.arm_and_takeoff(MAX_ALT)
    return "main"


def land():
    print("\n🛬 LANDING...")
    control.land()
    detector.cleanup()
    control.close_visualizer()
    cv2.destroyAllWindows()
    print("Program ended")
    sys.exit(0)


# Main execution
setup()

detector.get_image_size()
control.configure_PID(args.control)
control.initialize_debug_logs(args.debug_path)

print("\n" + "=" * 70)
print("AUTONOMOUS DRONE OBJECT FOLLOWING SYSTEM")
print("=" * 70)
print("Features:")
print("  • YOLOv11n object detection (80 classes)")
print("  • Custom laser point detection")
print("  • Distance estimation from object size in frame")
print("  • Forward/Backward movement based on estimated distance")
print("  • Maintains 2m follow distance")
print("  • Clean detection window - shows only objects")
print("  • All telemetry and status on Drone Visualizer")
print("  • Click on any object to track it")
print("=" * 70)
print("\nControls:")
print("  • 'q' - Quit")
print("  • CLICK on any detected object to start tracking")
print("\nModel Switching (Console only):")
print("  • 'g' - Switch to GENERAL detection mode")
print("  • 'l' - Switch to LASER detection mode")
print("  • 'c' - Switch to COMBINED mode")
print("  • '+/-' - Adjust confidence threshold")
print("\nMovement Logic:")
print("  • Object too small (far) → Drone moves FORWARD")
print("  • Object too large (close) → Drone moves BACKWARD")
print("  • Object off-center → Drone rotates LEFT/RIGHT")
print("=" * 70 + "\n")

STATE = "takeoff" if args.mode in ("flight", "sitl") else "main"

try:
    while True:
        if STATE == "takeoff":
            STATE = takeoff()
        elif STATE == "main":
            STATE = main_loop()
        else:
            land()
except KeyboardInterrupt:
    land()