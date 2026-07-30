import sys
import time
import argparse
import cv2
import numpy as np

import modules.app_config

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
    draw_lost_banner,
    draw_shortcut_bar,
    hud,
    set_detector_ref,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
)
from modules.navigation import FollowController
from modules.tracking import TrackingSession
from modules.auto_calibrate import AutoCalibrator
from shared.detection_models import TelemetryData
from jetson.communication.sgc_receiver import SGCCommandReceiver, _find_best_match
import keyboard
import shutil

parser = argparse.ArgumentParser(description='Drive autonomous')
parser.add_argument('--debug_path', type=str, default="debug/run1")
parser.add_argument('--mode', type=str, default='sitl', choices=['sitl', 'flight'], help='Run mode: sitl (SITL) or flight (real drone)')
parser.add_argument('--control', type=str, default='PID')
parser.add_argument('--drone_connection', type=str, default=None)
parser.add_argument('--model-path', type=str, default='YOLO/yolo11n.pt')
parser.add_argument('--conf-threshold', type=float, default=None)
parser.add_argument('--iou-threshold', type=float, default=None)
parser.add_argument('--min-box-area-ratio', type=float, default=None)
parser.add_argument('--imgsz', type=int, default=None)
parser.add_argument('--rtsp-port', type=int, default=8554, help='RTSP server port')
parser.add_argument('--jpeg-quality', type=int, default=50, help='JPEG quality for MJPEG stream (1-100, lower=faster)')
parser.add_argument('--sgc-host', type=str, default='192.168.1.100', help='SGC IP address')
parser.add_argument('--sgc-port', type=int, default=9001, help='SGC detection UDP port')
parser.add_argument('--sgc-cmd-port', type=int, default=9002, help='SGC command UDP listen port')
parser.add_argument('--start-sitl', action='store_true', help='Auto-launch SITL in WSL before connecting')
parser.add_argument('--intrinsics-fx', type=float, default=None, help='Camera focal length X (pixels)')
parser.add_argument('--intrinsics-fy', type=float, default=None, help='Camera focal length Y (pixels)')
parser.add_argument('--intrinsics-cx', type=float, default=None, help='Camera principal point X')
parser.add_argument('--intrinsics-cy', type=float, default=None, help='Camera principal point Y')
parser.add_argument('--auto-calibrate', action='store_true', help='Auto-detect chessboard and calibrate camera in background')
parser.add_argument('--chessboard', type=str, default='9x6', help='Chessboard inner corners WxH (default: 9x6)')
parser.add_argument('--object-height', type=float, default=0.25, help='Assumed object height in meters for monocular ranging (default: 0.25)')

args = parser.parse_args()
modules.app_config.OBJECT_HEIGHT = args.object_height

STATE = "takeoff"
tracking_session = TrackingSession()
follow_controller = FollowController()
streamer = None
sgc_receiver = None
_calibrator = None
_last_infer_time = 0.0
_fy = 1405.2


def setup():
    global streamer

    drone.set_backend(args.mode)

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
        else:
            connection_string = 'udpin:0.0.0.0:14550'
    else:
        connection_string = args.drone_connection

    if not control.connect_drone(connection_string, start_sitl=args.start_sitl):
        print("FATAL: Could not connect to vehicle. Ensure SITL is running or drone is connected.")
        sys.exit(1)
    control.set_flight_altitude(MAX_ALT)
    print("Vehicle connected")

    from jetson.streaming.rtsp_server import RTSPServer
    from jetson.communication.detection_sender import Streamer
    from shared.detection_transport import UDPTransport

    rtsp = RTSPServer(width=640, height=480, fps=30, port=args.rtsp_port, jpeg_quality=args.jpeg_quality)
    transport = UDPTransport(host=args.sgc_host, port=args.sgc_port)
    streamer = Streamer(rtsp_server=rtsp, transport=transport)
    streamer.set_mode(args.mode)

    if args.intrinsics_fx is not None and args.intrinsics_fy is not None:
        cx = args.intrinsics_cx if args.intrinsics_cx is not None else 320.0
        cy = args.intrinsics_cy if args.intrinsics_cy is not None else 240.0
        streamer.set_intrinsics(args.intrinsics_fx, args.intrinsics_fy, cx, cy)
        _fy = args.intrinsics_fy
        print(f"[INTRINSICS] Using CLI: fx={args.intrinsics_fx} fy={args.intrinsics_fy}")
    elif args.auto_calibrate:
        w, h = map(int, args.chessboard.lower().split("x"))
        _calibrator = AutoCalibrator(board_size=(w, h))
        if _calibrator.done and _calibrator.intrinsics:
            intr = _calibrator.intrinsics
            streamer.set_intrinsics(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                                    calib_w=intr.get("calib_w", 0), calib_h=intr.get("calib_h", 0))
            _fy = intr["fy"]
            print(f"[CALIB] Using saved calibration: fx={intr['fx']:.1f} fy={intr['fy']:.1f}")
        else:
            streamer.set_intrinsics(1406.4, 1405.2, 320.0, 240.0, calib_w=640, calib_h=480)
            _fy = 1405.2
            print(f"[CALIB] Auto-calibrating with {w}x{h} chessboard... show the board to the camera")
            print(f"[CALIB] Using default intrinsics until calibration completes")
    else:
        cx = 320.0
        cy = 240.0
        streamer.set_intrinsics(1406.4, 1405.2, cx, cy, calib_w=640, calib_h=480)
        _fy = 1405.2
        print("Using default intrinsics (fx=1406.4 fy=1405.2). Override with --intrinsics-fx/fy/cx/cy or --auto-calibrate")

    follow_controller.set_focal_length(_fy)
    print(f"[DIST] Monocular ranging: fy={_fy:.0f} object_height={args.object_height:.3f}m")

    streamer.start()
    print(f"Streaming: {rtsp.stream_url} -> {args.sgc_host}:{args.sgc_port}")

    from jetson.communication.sgc_receiver import SGCCommandReceiver

    global sgc_receiver
    sgc_receiver = SGCCommandReceiver(port=args.sgc_cmd_port)
    sgc_receiver.start()
    print(f"SGC commands listening on UDP port {args.sgc_cmd_port}")

    hud.mode = args.mode
    hud.streaming = streamer is not None

    cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracker", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.setMouseCallback("Tracker", tracking_session.handle_mouse_event)

    splash = _make_splash("Connecting to vehicle...")
    cv2.imshow("Tracker", splash)
    cv2.waitKey(1)


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


def _make_splash(text: str) -> np.ndarray:
    img = np.zeros((720, 960, 3), dtype=np.uint8)
    cv2.putText(img, text, (960 // 2 - 180, 720 // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def _console_status(mode: str, movement: dict | None, selected_obj) -> None:
    cols = shutil.get_terminal_size().columns
    parts = []
    parts.append(f"\033[1;36m{mode.upper():>7}\033[0m")

    if selected_obj and movement:
        dist = movement.get("fused_dist", 0)
        vel = movement.get("vel_z", 0)
        yaw = movement.get("yaw_cmd", 0)

        if abs(vel) < 0.05:
            move_txt = "\033[1;33mHOVER\033[0m"
        elif vel > 0:
            move_txt = f"\033[1;32mFORWARD\033[0m {vel:+.2f}m/s"
        else:
            move_txt = f"\033[1;31mBACKWARD\033[0m {vel:+.2f}m/s"

        parts.append(f"{selected_obj.class_name} \033[1m{dist:.2f}m\033[0m")
        parts.append(f"YAW {yaw:+.1f}\u00b0")
        parts.append(move_txt)
    else:
        parts.append("\033[2mno target\033[0m")
        parts.append("\033[2m--\033[0m")
        parts.append("\033[2mIDLE\033[0m")

    line = "  ".join(parts)
    if len(line) >= cols:
        line = line[:cols - 1]
    print(f"\r{line}\033[K", end="", flush=True)


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

        if _calibrator is not None and not _calibrator.done:
            _calibrator.feed(image)
            if _calibrator.done and _calibrator.intrinsics:
                intr = _calibrator.intrinsics
                streamer.set_intrinsics(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                                        calib_w=intr.get("calib_w", 0), calib_h=intr.get("calib_h", 0))
                _fy = intr["fy"]
                follow_controller.set_focal_length(_fy)
                print(f"[CALIB] Calibration complete! fy={_fy:.1f} — distance estimate updated.")

        if sgc_receiver is not None:
            cmd = sgc_receiver.pop_command()
            if cmd is not None:
                _handle_sgc_command(cmd, detections)

        height, width = image.shape[:2]
        tracking_session.set_frame_size(width, height, DISPLAY_WIDTH, DISPLAY_HEIGHT)

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

        else:
            control.update_telemetry_from_track(fps, 0, 0, False, 0, 0)
            drone.send_movement_command_YAW(0)
            drone.send_movement_command_XYA(0, 0, MAX_ALT)

        _console_status(args.mode, movement, selected_obj)

        if streamer is not None:
            telemetry = _build_telemetry()
            streamer.push(image, detections, fps, movement=movement, telemetry=telemetry)

        image = draw_detection_window(image, detections, detector)

        if selected_obj is not None:
            annotate_tracking_overlay(image, fps, selected_obj, movement)
        else:
            annotate_selection_overlay(image, detections)

        if hud.hud_visible:
            _update_hud_state(fps, tracker_state, selected_obj, tracking_conf, movement)
            draw_hud_background(image)
            draw_status_bar(image, tracker_state,
                            selected_obj.class_name if selected_obj else None,
                            tracking_conf)
            draw_fps(image, fps, _last_infer_time)
            draw_lost_banner(image)
            draw_shortcut_bar(image)

        if image is not None:
            display = cv2.resize(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Tracker", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            land()
            break

    return "land"


def takeoff():
    print("TAKEOFF")
    splash = _make_splash("Taking off...")
    cv2.imshow("Tracker", splash)
    cv2.waitKey(1)
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
    if args.start_sitl:
        control.stop_sitl()
    detector.cleanup()
    cv2.destroyAllWindows()
    sys.exit(0)


setup()

# Show window immediately so user isn't staring at a blank terminal
splash = _make_splash("Initializing...")
cv2.imshow("Tracker", splash)
cv2.waitKey(1)

detector.get_image_size()
control.configure_PID(args.control)

STATE = "takeoff"

try:
    while True:
        if STATE == "takeoff":
            STATE = takeoff()
        elif STATE == "main":
            STATE = main_loop()
        else:
            land()
except KeyboardInterrupt:
    print("\nShutting down...")
    land()
