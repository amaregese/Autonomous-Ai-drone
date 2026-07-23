import sys
import time
import argparse
import cv2

sys.path.insert(1, 'modules')

from modules import lidar, control, detector_yolo11 as detector
from modules import drone
from modules.app_config import MAX_ALT, TRACKING_LOST_THRESHOLD
from modules.display import (
    annotate_selection_overlay,
    annotate_tracking_overlay,
    draw_detection_window,
    update_drone_visualizer_status,
)
from modules.navigation import FollowController
from modules.tracking import TrackingSession
import keyboard

parser = argparse.ArgumentParser(description='Drive autonomous')
parser.add_argument('--debug_path', type=str, default="debug/run1")
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--control', type=str, default='PID')
parser.add_argument('--drone_connection', type=str, default=None)
parser.add_argument('--model-path', type=str, default='YOLO/yolo11n.pt')
parser.add_argument('--conf-threshold', type=float, default=None)
parser.add_argument('--iou-threshold', type=float, default=None)
parser.add_argument('--min-box-area-ratio', type=float, default=None)
parser.add_argument('--imgsz', type=int, default=None)

args = parser.parse_args()

STATE = "takeoff"
tracking_session = TrackingSession()
follow_controller = FollowController()
telemetry_log_counter = 0


def setup():
    drone.set_backend("sitl" if args.mode == "sitl" else "mock")

    print("connecting lidar")
    lidar.connect_lidar("/dev/ttyTHS1")

    print("setting up detector")
    detector.configure_detector(
        confidence_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        min_box_area_ratio=args.min_box_area_ratio,
        inference_img_size=args.imgsz,
    )
    detector.initialize_detector(args.model_path)

    print("connecting to drone")

    if args.drone_connection is None:
        if args.mode == "flight":
            connection_string = '/dev/ttyACM0'
        elif args.mode == "sitl":
            connection_string = 'udpin:0.0.0.0:14550'
        else:
            connection_string = '192.168.1.96:14550'
    else:
        connection_string = args.drone_connection

    control.connect_drone(connection_string)
    control.set_flight_altitude(MAX_ALT)


def main_loop():
    global telemetry_log_counter

    tracking_session.reset_loss_state()

    while True:

        if keyboard.is_pressed('q'):
            land()
            break

        detections, fps, image = detector.get_detections()

        if image is None or image.size == 0:
            time.sleep(0.01)
            continue

        height, width = image.shape[:2]

        image = draw_detection_window(image, detections, detector)

        tracking_session.process_click(detections, detector, control)

        selected_obj = detector.get_selected_object()
        selected_class = detector.get_selected_class()
        tracking_lost = detector.get_tracking_status()
        tracking_conf = detector.get_tracking_confidence()

        if selected_obj is not None:

            movement = follow_controller.compute_follow_command(selected_obj, image.shape)

            telemetry_log_counter += 1
            if telemetry_log_counter % 15 == 0:
                print(
                    f"Distance: {movement['lidar_dist']:.2f}m | "
                    f"Error: {movement['distance_error']:.2f}m"
                )

            control.update_visualizer_target(
                movement["target_x"],
                movement["target_z"],
                selected_obj.class_name,
                selected_obj.confidence,
                movement["lidar_dist"],
            )

            drone.send_movement_command_YAW(movement["yaw_cmd"])
            drone.send_movement_command_XYA(0, movement["vel_z"], MAX_ALT)

            control.update_telemetry_from_track(
                fps,
                movement["yaw_cmd"],
                movement["vel_z"],
                movement["lidar_on_target"],
                movement["x_delta"],
                movement["y_delta"],
            )

            annotate_tracking_overlay(image, fps, selected_obj, movement)

        else:
            control.update_visualizer_target(width // 2, height // 2, "No Target", 0, 0)
            control.update_telemetry_from_track(fps, 0, 0, False, 0, 0)

            annotate_selection_overlay(image, detections)

        update_drone_visualizer_status(detector, control, TRACKING_LOST_THRESHOLD)

        # ONLY ONE DISPLAY HERE
        if image is not None:
            cv2.imshow("Detection", image)

        control.draw_visualizer()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            land()
            break

    return "land"


def takeoff():
    print("TAKEOFF")
    control.print_drone_report()
    control.arm_and_takeoff(MAX_ALT)
    return "main"


def land():
    print("LANDING...")
    control.land()
    detector.cleanup()
    control.close_visualizer()
    cv2.destroyAllWindows()
    sys.exit(0)


setup()

detector.get_image_size()
control.configure_PID(args.control)
control.initialize_debug_logs(args.debug_path)

STATE = "takeoff" if args.mode in ("flight", "sitl") else "main"

while True:
    if STATE == "takeoff":
        STATE = takeoff()
    elif STATE == "main":
        STATE = main_loop()
    else:
        land()