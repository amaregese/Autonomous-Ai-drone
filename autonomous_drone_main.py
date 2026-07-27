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
    draw_hud_background,
    draw_status_bar,
    draw_fps,
    draw_telemetry,
    draw_lost_banner,
    draw_shortcut_bar,
    hud,
    set_detector_ref,
)
from modules.navigation import FollowController
from modules.tracking import TrackingSession
from shared.detection_models import TelemetryData
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
_last_infer_time = 0.0


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
    set_detector_ref(detector)

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
    streamer.set_mode(args.mode)
    streamer.start()
    print(f"Streaming: {rtsp.stream_url} -> {args.sgc_host}:{args.sgc_port}")

    from jetson.communication.sgc_receiver import SGCCommandReceiver

    global sgc_receiver
    sgc_receiver = SGCCommandReceiver(port=args.sgc_cmd_port)
    sgc_receiver.start()
    print(f"SGC commands listening on UDP port {args.sgc_cmd_port}")

    hud.mode = args.mode
    hud.streaming = streamer is not None


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


def _handle_keyboard():
    if keyboard.is_pressed('q'):
        land()
        return "quit"

    if keyboard.is_pressed('escape'):
        detector.clear_selection()
        print("Selection cleared (ESC)")

    if keyboard.is_pressed('space'):
        sel = detector.get_selected_object()
        if sel is not None:
            detector.clear_selection()
            print("Follow stopped (Space)")
        else:
            print("No target to follow (Space)")

    if keyboard.is_pressed('r'):
        detector.clear_selection()
        tracking_session.reset_loss_state()
        print("Tracker reset (R)")

    if keyboard.is_pressed('h'):
        hud.hud_visible = not hud.hud_visible
        print(f"HUD: {'ON' if hud.hud_visible else 'OFF'}")
        time.sleep(0.2)

    return None


def _update_hud_state(fps, tracker_state, selected_obj, tracking_conf, movement):
    telemetry = _build_telemetry()
    hud.altitude = telemetry.altitude
    hud.battery = telemetry.battery
    hud.lat = telemetry.lat
    hud.lon = telemetry.lon
    hud.ekf_ok = telemetry.ekf_ok

    if tracker_state == "lost":
        hud.lost_flash_until = time.time() + 2.0


def _build_telemetry() -> TelemetryData:
    alt = MAX_ALT
    bat = 100
    lat, lon = 0.0, 0.0
    ekf = True

    try:
        lat, lon, alt = drone.get_position()
    except Exception:
        pass

    try:
        bat = drone.get_battery_level()
    except Exception:
        pass

    return TelemetryData(altitude=alt, battery=bat, lat=lat, lon=lon, ekf_ok=ekf)


def main_loop():
    global _last_infer_time

    tracking_session.reset_loss_state()

    while True:

        kb = _handle_keyboard()
        if kb == "quit":
            break

        t0 = time.perf_counter()
        detections, fps, image = detector.get_detections()
        _last_infer_time = (time.perf_counter() - t0) * 1000

        if image is None or image.size == 0:
            time.sleep(0.01)
            continue

        if sgc_receiver is not None:
            cmd = sgc_receiver.pop_command()
            if cmd is not None:
                _handle_sgc_command(cmd, detections)

        height, width = image.shape[:2]

        image = draw_detection_window(image, detections, detector)

        tracking_session.process_click(detections, detector, control)

        selected_obj = detector.get_selected_object()
        tracker_state = "idle"
        tracking_conf = 0.0
        movement = None

        if selected_obj is not None:
            tracker_state = "tracking" if not detector.get_tracking_status() else "lost"
            tracking_conf = detector.get_tracking_confidence()

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

        if streamer is not None:
            telemetry = _build_telemetry()
            streamer.push(image, detections, fps, movement=movement, telemetry=telemetry)

        if hud.hud_visible:
            _update_hud_state(fps, tracker_state, selected_obj, tracking_conf, movement)
            draw_hud_background(image)
            draw_status_bar(image, tracker_state,
                            selected_obj.class_name if selected_obj else None,
                            tracking_conf)
            draw_telemetry(image, hud.altitude, hud.battery, hud.lat, hud.lon, hud.ekf_ok)
            draw_fps(image, fps, _last_infer_time)
            draw_lost_banner(image)
            draw_shortcut_bar(image)

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
