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
)
from modules.navigation import FollowController
from modules.tracking import TrackingSession
from jetson.communication.sgc_receiver import SGCCommandReceiver, _find_best_match
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
parser.add_argument('--rtsp-port', type=int, default=8554, help='RTSP server port')
parser.add_argument('--sgc-host', type=str, default='192.168.1.100', help='SGC IP address')
parser.add_argument('--sgc-port', type=int, default=9001, help='SGC detection UDP port')
parser.add_argument('--sgc-cmd-port', type=int, default=9002, help='SGC command UDP listen port')

args = parser.parse_args()

STATE = "takeoff"
tracking_session = TrackingSession()
follow_controller = FollowController()
streamer = None
sgc_receiver = None


def setup():
    global streamer

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

    from jetson.streaming.rtsp_server import RTSPServer
    from jetson.communication.detection_sender import Streamer
    from shared.detection_transport import UDPTransport

    rtsp = RTSPServer(width=640, height=480, fps=30, port=args.rtsp_port)
    transport = UDPTransport(host=args.sgc_host, port=args.sgc_port)
    streamer = Streamer(rtsp_server=rtsp, transport=transport)
    streamer.start()
    print(f"Streaming: {rtsp.stream_url} -> {args.sgc_host}:{args.sgc_port}")

    from jetson.communication.sgc_receiver import SGCCommandReceiver

    global sgc_receiver
    sgc_receiver = SGCCommandReceiver(port=args.sgc_cmd_port)
    sgc_receiver.start()
    print(f"SGC commands listening on UDP port {args.sgc_cmd_port}")


def _handle_sgc_command(cmd, detections):
    if cmd.command_type in ("select_target", "follow_start"):
        if cmd.bbox is None:
            return
        matched = _find_best_match(cmd.bbox, cmd.class_name, detections)
        if matched is not None:
            detector.select_object(matched)
            print(f"[SGC] Selected: {matched.class_name} ({matched.confidence * 100:.1f}%)")
        else:
            print(f"[SGC] No match for {cmd.class_name} bbox={cmd.bbox}")
    elif cmd.command_type in ("deselect_target", "follow_stop"):
        detector.clear_selection()
        print("[SGC] Selection cleared")


def main_loop():
    tracking_session.reset_loss_state()

    while True:

        if keyboard.is_pressed('q'):
            land()
            break

        detections, fps, image = detector.get_detections()

        if image is None or image.size == 0:
            time.sleep(0.01)
            continue

        if streamer is not None:
            streamer.push(image, detections, fps)

        if sgc_receiver is not None:
            cmd = sgc_receiver.pop_command()
            if cmd is not None:
                _handle_sgc_command(cmd, detections)

        height, width = image.shape[:2]

        image = draw_detection_window(image, detections, detector)

        tracking_session.process_click(detections, detector, control)

        selected_obj = detector.get_selected_object()

        if selected_obj is not None:

            movement = follow_controller.compute_follow_command(selected_obj, image.shape)

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
            control.update_telemetry_from_track(fps, 0, 0, False, 0, 0)

            annotate_selection_overlay(image, detections)

        if image is not None:
            cv2.imshow("Detection", image)
            cv2.setMouseCallback("Detection", tracking_session.handle_mouse_event)

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
    if sgc_receiver is not None:
        sgc_receiver.stop()
    if streamer is not None:
        streamer.stop()
    detector.cleanup()
    cv2.destroyAllWindows()
    sys.exit(0)


setup()

detector.get_image_size()
control.configure_PID(args.control)

STATE = "takeoff" if args.mode in ("flight", "sitl") else "main"

while True:
    if STATE == "takeoff":
        STATE = takeoff()
    elif STATE == "main":
        STATE = main_loop()
    else:
        land()