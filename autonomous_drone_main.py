import sys
import time
import argparse
import glob
import ipaddress
import math
import os
import socket

os.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")

import cv2
import numpy as np

import modules.app_config

sys.path.insert(1, 'modules')

from modules import lidar, control, detector_yolo11 as detector
from modules import drone
from modules.run_config import config_source, get_run_defaults, save_default
from modules.app_config import (
    MAX_ALT,
    TRACKING_LOST_THRESHOLD,
    MIN_FOLLOW_ALT,
    MIN_FOLLOW_BATTERY,
    MIN_FOLLOW_GPS_FIX,
    MIN_FOLLOW_MODE,
)
from modules.display import (
    annotate_selection_overlay,
    annotate_tracking_overlay,
    draw_detection_window,
    draw_follow_prompt,
    draw_hud_background,
    draw_hud_notification,
    draw_fps,
    draw_lost_banner,
    compose_window,
    display_to_frame,
    get_lost_dismiss_rect,
    get_takeoff_button_rect,
    set_hud_status,
    hud,
    set_detector_ref,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    HEADER_FINAL,
)
from modules.person_follow import PersonFollowController
from modules.distance_estimator import (
    DEFAULT_CONFIGURED_INTRINSICS,
    DistanceEstimator,
    EstimatorConfig,
    VisionConfig,
    VisionDistanceEstimator,
    annotate_detections,
    load_configured_intrinsics,
)
from modules.distance_estimator.calibration import CameraIntrinsics
from modules.tracking import TrackingSession
from modules.auto_calibrate import AutoCalibrator
from shared.detection_models import TelemetryData
from jetson.communication.sgc_receiver import SGCCommandReceiver, _find_best_match
import shutil

parser = argparse.ArgumentParser(description='Drive autonomous')
# Persisted defaults (run_config.json) so an IDE run button behaves like the
# documented command line. Explicit CLI flags still win.
_RUN_DEFAULTS = get_run_defaults()


def _default(key, fallback):
    return _RUN_DEFAULTS.get(key, fallback)


def _add_flag(parser, enabled_flag, disabled_flag, key, fallback, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(enabled_flag, dest=key, action="store_true",
                       default=_default(key, fallback), help=help_text)
    group.add_argument(disabled_flag, dest=key, action="store_false",
                       help=argparse.SUPPRESS)


parser.add_argument('--debug_path', type=str, default=_default("debug_path", "debug/run1"))
parser.add_argument('--mode', type=str, default=_default('mode', 'sitl'), choices=['sitl', 'flight'], help='Run mode: sitl (SITL) or flight (real drone)')
parser.add_argument('--control', type=str, default=_default('control', 'PID'))
parser.add_argument('--drone_connection', type=str, default=_default('drone_connection', None))
parser.add_argument('--baud', type=int, default=_default('baud', 57600), help='Serial baud rate for a real FCU (default: 57600)')
_add_flag(parser, '--no-prompt', '--prompt', 'no_prompt', False,
          'Skip the connection prompt (headless): use --mode defaults')
parser.add_argument('--model-path', type=str, default=_default('model_path', 'YOLO/yolo11n.pt'))
parser.add_argument('--camera', type=int, default=_default('camera', None), help='Force webcam index (default: auto-detect)')
parser.add_argument('--conf-threshold', type=float, default=_default('conf_threshold', None))
parser.add_argument('--iou-threshold', type=float, default=_default('iou_threshold', None))
parser.add_argument('--min-box-area-ratio', type=float, default=_default('min_box_area_ratio', None))
parser.add_argument('--imgsz', type=int, default=_default('imgsz', 320))
parser.add_argument('--rtsp-port', type=int, default=_default('rtsp_port', 8554), help='RTSP server port')
parser.add_argument('--jpeg-quality', type=int, default=_default('jpeg_quality', 30), help='JPEG quality for MJPEG stream (1-100, lower=faster)')
parser.add_argument('--sgc-host', type=str, default=_default('sgc_host', None), help='SGC IP address (default: 127.0.0.1 in SITL, 192.168.1.100 in flight)')
_add_flag(parser, '--sgc-host-prompt', '--no-sgc-host-prompt', 'sgc_host_prompt', True,
          'Ask for the SGC IP on startup (skipped when --sgc-host is given)')
parser.add_argument('--sgc-port', type=int, default=_default('sgc_port', 9001), help='SGC detection UDP port')
parser.add_argument('--sgc-cmd-port', type=int, default=_default('sgc_cmd_port', 9002), help='SGC command UDP listen port')
_add_flag(parser, '--start-sitl', '--no-start-sitl', 'start_sitl', False,
          'Auto-launch SITL in WSL before connecting (Windows/WSL only)')
parser.add_argument('--intrinsics-fx', type=float, default=_default('intrinsics_fx', None), help='Camera focal length X (pixels)')
parser.add_argument('--intrinsics-fy', type=float, default=_default('intrinsics_fy', None), help='Camera focal length Y (pixels)')
parser.add_argument('--intrinsics-cx', type=float, default=_default('intrinsics_cx', None), help='Camera principal point X')
parser.add_argument('--intrinsics-cy', type=float, default=_default('intrinsics_cy', None), help='Camera principal point Y')
parser.add_argument('--intrinsics-width', type=int, default=_default('intrinsics_width', DEFAULT_CONFIGURED_INTRINSICS["frame_w"]), help='Reference calibration width (pixels)')
parser.add_argument('--intrinsics-height', type=int, default=_default('intrinsics_height', DEFAULT_CONFIGURED_INTRINSICS["frame_h"]), help='Reference calibration height (pixels)')

_add_flag(parser, '--auto-calibrate', '--no-auto-calibrate', 'auto_calibrate', False,
          'Auto-detect chessboard and calibrate camera in background')
parser.add_argument('--chessboard', type=str, default=_default('chessboard', '9x6'), help='Chessboard inner corners WxH (default: 9x6)')
parser.add_argument('--object-height', type=float, default=_default('object_height', 0.25), help='Assumed object height in meters for monocular ranging (default: 0.25)')
_add_flag(parser, '--headless', '--no-headless', 'headless', False,
          'Run without GUI (no cv2.imshow/waitKey). For headless Jetson deployment')
_add_flag(parser, '--no-flip-camera', '--flip-camera', 'no_flip_camera', False,
          'Disable horizontal camera flip (default: flip enabled for mirrored cameras)')
parser.add_argument('--lidar-port', type=str, default=_default('lidar_port', '/dev/ttyTHS1'), help='LiDAR serial port (default: /dev/ttyTHS1 on Jetson)')

args = parser.parse_args()
_SGC_HOST_FROM_CLI = any(
    entry == "--sgc-host" or entry.startswith("--sgc-host=") for entry in sys.argv[1:]
)
if _RUN_DEFAULTS:
    print(f"[CONFIG] defaults from {config_source()} "
          f"(command-line flags still win; --help lists every option)")
modules.app_config.OBJECT_HEIGHT = args.object_height


tracking_session = TrackingSession()
tracking_session.set_mapper(display_to_frame)
# The controller is deliberately only a decision layer.  It sends through the
# project's existing drone API below, so no additional MAVLink connection is
# created for person follow.
follow_controller = PersonFollowController()
streamer = None
sgc_receiver = None
sgc_host = None
_calibrator = None
_last_infer_time = 0.0
_DEFAULT_MANIFEST = os.path.join(os.path.dirname(__file__), "benchmarks", "distance", "manifest.json")
_CONFIGURED_INTRINSICS = load_configured_intrinsics(_DEFAULT_MANIFEST)
_fx = _CONFIGURED_INTRINSICS.fx if _CONFIGURED_INTRINSICS is not None else DEFAULT_CONFIGURED_INTRINSICS["fx"]
_fy = _CONFIGURED_INTRINSICS.fy if _CONFIGURED_INTRINSICS is not None else DEFAULT_CONFIGURED_INTRINSICS["fy"]
_cx = _CONFIGURED_INTRINSICS.cx if _CONFIGURED_INTRINSICS is not None else DEFAULT_CONFIGURED_INTRINSICS["cx"]
_cy = _CONFIGURED_INTRINSICS.cy if _CONFIGURED_INTRINSICS is not None else DEFAULT_CONFIGURED_INTRINSICS["cy"]
_calib_w = _CONFIGURED_INTRINSICS.frame_w if _CONFIGURED_INTRINSICS is not None else DEFAULT_CONFIGURED_INTRINSICS["frame_w"]
_calib_h = _CONFIGURED_INTRINSICS.frame_h if _CONFIGURED_INTRINSICS is not None else DEFAULT_CONFIGURED_INTRINSICS["frame_h"]
_distance_estimator = None
_follow_frame_size = None

_lost_start_time = None
_rtl_triggered = False
_following = False
_home_alt = 0.0
_sgc_peer_warned = False
_panic_rtl_active = False


def _reset_lost_state():
    global _lost_start_time, _rtl_triggered
    _lost_start_time = None
    _rtl_triggered = False


def _preflight_follow():
    global _home_alt
    problems = []

    try:
        if not drone.is_armed():
            problems.append("vehicle is not armed")
    except Exception:
        problems.append("cannot read armed state")

    try:
        mode = drone.get_mode()
        if mode != MIN_FOLLOW_MODE:
            problems.append(f"mode is {mode}, need {MIN_FOLLOW_MODE}")
    except Exception:
        problems.append("cannot read flight mode")

    try:
        fix = drone.get_gps_fix_type()
        if fix < MIN_FOLLOW_GPS_FIX:
            problems.append(f"GPS fix {fix}/3")
    except Exception:
        problems.append("cannot read GPS fix")

    try:
        if not drone.is_ekf_ok():
            problems.append("EKF not converged")
    except Exception:
        problems.append("cannot read EKF status")

    try:
        rel_alt = drone.get_position()[2] - _home_alt
        if rel_alt < MIN_FOLLOW_ALT:
            if rel_alt < 0.5:
                problems.append("vehicle is on the ground")
            else:
                problems.append(f"altitude {rel_alt:.1f}m below {MIN_FOLLOW_ALT}m")
    except Exception:
        problems.append("cannot read altitude")

    try:
        battery = drone.get_battery_level()
        if battery >= 0 and battery < MIN_FOLLOW_BATTERY:
            problems.append(f"battery {battery}% below {MIN_FOLLOW_BATTERY}%")
    except Exception:
        problems.append("cannot read battery")

    return problems


def _handle_takeoff_button():
    try:
        if drone.is_armed():
            msg = "Vehicle is already armed — land first or reset SITL"
            print(msg)
            set_hud_status(msg, (255, 255, 0), 3.0)
            return
    except Exception:
        pass

    problems = []
    try:
        if drone.get_gps_fix_type() < MIN_FOLLOW_GPS_FIX:
            problems.append(f"GPS fix {drone.get_gps_fix_type()}/3")
    except Exception:
        problems.append("cannot read GPS fix")
    try:
        if not drone.is_ekf_ok():
            problems.append("EKF not converged")
    except Exception:
        problems.append("cannot read EKF status")
    if problems:
        msg = "Takeoff refused: " + "; ".join(problems) + "."
        print(msg)
        set_hud_status(msg, (0, 0, 255), 4.0)
        return

    msg = f"Arming vehicle and taking off to {MAX_ALT:.0f}m..."
    print(msg)
    set_hud_status(msg, (0, 255, 255), 3.0)
    try:
        control.arm_and_takeoff(MAX_ALT)
    except Exception as exc:
        msg = f"Takeoff failed: {exc}"
        print(msg)
        set_hud_status(msg, (0, 0, 255), 4.0)
        return
    control.set_flight_altitude(MAX_ALT)
    msg = f"Vehicle armed — holding at {MAX_ALT:.0f}m"
    print(msg)
    set_hud_status(msg, (0, 200, 0), 3.0)


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rect = get_takeoff_button_rect()
        if rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
            _handle_takeoff_button()
            return
    tracking_session.handle_mouse_event(event, x, y, flags, param)


def _serial_heartbeat_ok(path, timeout=1.5):
    try:
        import serial
    except Exception:
        return False
    try:
        s = serial.Serial(path, 115200, timeout=0.3)
        s.reset_input_buffer()
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = s.read(256)
            if not data:
                continue
            if 0xFD in data or 0xFE in data:
                s.close()
                return True
        s.close()
    except Exception:
        pass
    return False


def _detect_serial_ports():
    ports = []
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            low = p.device.lower()
            base = low.rsplit("/", 1)[-1]
            if "usb" in low or base.startswith("ttyacm") or base.startswith("ttyusb") or base.startswith("com"):
                ports.append(p.device)
    except Exception:
        ports = []
    if not ports and os.name != "nt":
        for pattern in ["/dev/cu.usb*", "/dev/ttyACM*", "/dev/ttyUSB*", "/dev/serial/by-id/*"]:
            ports.extend(sorted(glob.glob(pattern)))
    ports = sorted(set(ports))
    live = [p for p in ports if _serial_heartbeat_ok(p)]
    return live if live else ports


def _default_fcu_serial():
    if os.name == "nt":
        ports = _detect_serial_ports()
        return ports[0] if ports else "COM3"
    return "/dev/ttyACM0"


def _pick_connection():
    if args.drone_connection is not None:
        return args.drone_connection, args.start_sitl, args.baud

    sitl_string = "udpin:0.0.0.0:14550"
    if args.no_prompt:
        if args.mode == "flight":
            return _default_fcu_serial(), args.start_sitl, args.baud
        return sitl_string, args.start_sitl, args.baud

    ports = _detect_serial_ports()
    print()
    print("Select drone connection:")
    print("  [1] SITL (connect to UDP 127.0.0.1:14550)")
    if os.name == "nt":
        print("  [2] SITL (auto-launch WSL ArduPilot)")
        menu_offset = 3
    else:
        menu_offset = 2
    for idx, port in enumerate(ports, start=menu_offset):
        print(f"  [{idx}] Real FCU serial/COM: {port}")
    if not ports:
        print("  (no serial/COM ports detected)")
    try:
        choice = input("Choice [1]: ").strip() or "1"
    except (EOFError, OSError):
        if args.mode == "flight":
            return _default_fcu_serial(), args.start_sitl, args.baud
        return sitl_string, args.start_sitl, args.baud

    if choice == "1":
        return sitl_string, False, args.baud
    if choice == "2" and os.name == "nt":
        return sitl_string, True, args.baud
    if choice == "2" and os.name != "nt":
        print("SITL auto-launch not available on this platform. Connect to existing SITL or real FCU.")
        return sitl_string, False, args.baud
    try:
        idx = int(choice)
        if menu_offset <= idx <= menu_offset - 1 + len(ports):
            return ports[idx - menu_offset], False, args.baud
    except ValueError:
        pass
    print(f"Invalid choice '{choice}', defaulting to SITL.")
    return sitl_string, False, args.baud


def _local_ipv4_addresses():
    """Best-effort list of this machine's non-loopback IPv4 addresses.

    The UDP connect is a routing-table lookup only - no packet is sent.
    """
    found = []
    for probe in ("192.0.2.1", "8.8.8.8"):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.settimeout(1.0)
                sock.connect((probe, 9))
                addr = sock.getsockname()[0]
            finally:
                sock.close()
            if addr and not addr.startswith("127.") and addr not in found:
                found.append(addr)
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if addr and not addr.startswith("127.") and addr not in found:
                found.append(addr)
    except OSError:
        pass
    return found


def _resolve_host(value):
    """Turn user input into an IPv4 string, or None when it is not usable."""
    text = value.strip().rstrip("/")
    if not text:
        return None
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        pass
    try:
        return socket.gethostbyname(text)
    except (OSError, UnicodeError):
        return None


def _foreign_networks(host):
    """Return this machine's addresses when `host` shares no /24 with them."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return []
    if addr.is_loopback:
        return []
    local = _local_ipv4_addresses()
    if not local:
        return []
    if any(candidate.rsplit(".", 1)[0] == host.rsplit(".", 1)[0] for candidate in local):
        return []
    return local


def _ask(prompt: str, fallback: str, attempts: int = 3) -> str:
    """Prompt until we get a usable answer, or fall back on no/closed input."""
    for _ in range(attempts):
        try:
            answer = input(prompt).strip()
        except (EOFError, OSError):
            return fallback
        if not answer:
            return fallback
        resolved = _resolve_host(answer)
        if resolved:
            return resolved
        print(f"  '{answer}' is not a valid IP address or hostname.")
    print(f"  Using {fallback}.")
    return fallback


def _default_sgc_host() -> str:
    configured = _default('sgc_host', None)
    if configured:
        return str(configured).strip()
    return '127.0.0.1' if args.mode == 'sitl' else '192.168.1.100'


def _remember_sgc_host(host: str) -> None:
    try:
        answer = input(f"  Remember {host} as the default for next time? [y/N] ").strip().lower()
    except (EOFError, OSError):
        return
    if answer not in ("y", "yes"):
        return
    if save_default("sgc_host", host):
        print(f"[CONFIG] saved sgc_host={host} to {config_source()}")
    else:
        print(f"[CONFIG] could not write {config_source()}")


def _resolve_sgc_host() -> str:
    """Return the SGC address, asking for it when it was not given on the CLI.

    The SGC laptop changes IP between setups, so the previous value (from
    run_config.json) is only a suggestion: press ENTER to reuse it, type a new
    address to use it just this once (optionally saving it for next time).
    """
    if _SGC_HOST_FROM_CLI:
        return _resolve_host(args.sgc_host) or args.sgc_host

    fallback = _default_sgc_host()
    if not args.sgc_host_prompt:
        return fallback

    local = _local_ipv4_addresses()
    print()
    print("SGC (ground station)")
    print(f"  this machine      : {', '.join(local) if local else 'no LAN address found'}")
    print(f"  last known SGC    : {fallback}")
    print("  Detections are sent there over UDP; commands arrive on "
          f"port {args.sgc_cmd_port}.")
    chosen = _ask(f"  SGC IP address [{fallback}]: ", fallback)

    foreign = _foreign_networks(chosen)
    if foreign:
        print(f"  WARNING: {chosen} is not on this machine's network "
              f"({', '.join(foreign)}) - the SGC may be unreachable.")
    if chosen != fallback:
        _remember_sgc_host(chosen)
    return chosen


def _describe_link(connection_string, baud):
    if connection_string.startswith("udpin"):
        return f"SITL over UDP ({connection_string})"
    return f"real FCU on {connection_string} @ {baud} baud"


def _print_startup_summary(sgc_host, connection_string, baud):
    print()
    print("Starting with")
    print(f"  mode          : {args.mode}")
    print(f"  drone link    : {_describe_link(connection_string, baud)}")
    print(f"  SGC target    : {sgc_host}:{args.sgc_port}   "
          f"(commands come in on UDP {args.sgc_cmd_port})")
    print(f"  video stream  : port {args.rtsp_port}, detection size {args.imgsz}px")
    print(f"  model         : {args.model_path}")
    print("  camera, lidar and the model load next (a few seconds).")
    print()


def setup():
    global streamer, _home_alt, _fx, _fy, _cx, _cy, _calib_w, _calib_h, sgc_host

    # Everything that needs an answer is asked first, so a mistake is caught
    # before the camera/model/simulator startup cost is paid.
    sgc_host = _resolve_sgc_host()
    connection_string, start_sitl, baud = _pick_connection()
    _print_startup_summary(sgc_host, connection_string, baud)

    drone.set_backend(args.mode)

    print("connecting lidar")
    lidar.connect_lidar(args.lidar_port)

    print("setting up detector")
    detector.configure_detector(
        confidence_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        min_box_area_ratio=args.min_box_area_ratio,
        inference_img_size=args.imgsz,
    )
    detector.initialize_detector(args.model_path, camera_index=args.camera, flip_camera=not args.no_flip_camera)
    set_detector_ref(detector)

    print("connecting to drone")
    print(f"drone link: {connection_string} (baud {baud})")

    if not control.connect_drone(connection_string, start_sitl=start_sitl, baud=baud):
        print("FATAL: Could not connect to vehicle. Ensure SITL is running or the FCU is connected.")
        sys.exit(1)
    control.set_flight_altitude(MAX_ALT)
    _, _, _home_alt = drone.get_position()
    modules.app_config.HOME_ALT = _home_alt
    print(f"Vehicle connected (home altitude {_home_alt:.1f}m)")

    from jetson.streaming.rtsp_server import RTSPServer
    from jetson.communication.detection_sender import Streamer
    from shared.detection_transport import UDPTransport

    rtsp = RTSPServer(width=640, height=480, fps=30, port=args.rtsp_port, jpeg_quality=args.jpeg_quality)
    transport = UDPTransport(host=sgc_host, port=args.sgc_port)
    streamer = Streamer(rtsp_server=rtsp, transport=transport)
    streamer.set_mode(args.mode)


    if args.intrinsics_fx is not None and args.intrinsics_fy is not None:
        _fx = args.intrinsics_fx
        _fy = args.intrinsics_fy
        _cx = args.intrinsics_cx if args.intrinsics_cx is not None else 320.0
        _cy = args.intrinsics_cy if args.intrinsics_cy is not None else 240.0
        _calib_w, _calib_h = args.intrinsics_width, args.intrinsics_height
        streamer.set_intrinsics(_fx, _fy, _cx, _cy, calib_w=_calib_w, calib_h=_calib_h)
        print(f"[INTRINSICS] Using CLI: fx={_fx} fy={_fy}")
    elif args.auto_calibrate:
        w, h = map(int, args.chessboard.lower().split("x"))
        _calibrator = AutoCalibrator(board_size=(w, h))
        if _calibrator.done and _calibrator.intrinsics:
            intr = _calibrator.intrinsics
            _fx = intr["fx"]
            _fy = intr["fy"]
            _cx = intr["cx"]
            _cy = intr["cy"]
            _calib_w = intr.get("calib_w") or DEFAULT_CONFIGURED_INTRINSICS["frame_w"]
            _calib_h = intr.get("calib_h") or DEFAULT_CONFIGURED_INTRINSICS["frame_h"]

            streamer.set_intrinsics(_fx, _fy, _cx, _cy,
                                    calib_w=_calib_w, calib_h=_calib_h)
            print(f"[CALIB] Using saved calibration: fx={_fx:.1f} fy={_fy:.1f}")
        else:
            _calib_w, _calib_h = (
                (_CONFIGURED_INTRINSICS.frame_w, _CONFIGURED_INTRINSICS.frame_h)
                if _CONFIGURED_INTRINSICS is not None
                else (
                    DEFAULT_CONFIGURED_INTRINSICS["frame_w"],
                    DEFAULT_CONFIGURED_INTRINSICS["frame_h"],
                )
            )
            streamer.set_intrinsics(_fx, _fy, _cx, _cy, calib_w=_calib_w, calib_h=_calib_h)

            print(f"[CALIB] Auto-calibrating with {w}x{h} chessboard... show the board to the camera")
            print(f"[CALIB] Using default intrinsics until calibration completes")
    else:
        _calib_w, _calib_h = (
            (_CONFIGURED_INTRINSICS.frame_w, _CONFIGURED_INTRINSICS.frame_h)
            if _CONFIGURED_INTRINSICS is not None
            else (640, 480)
        )
        streamer.set_intrinsics(_fx, _fy, _cx, _cy, calib_w=_calib_w, calib_h=_calib_h)

        print(f"Using configured-default intrinsics (fx={_fx:.1f} fy={_fy:.1f}). Override with --intrinsics-fx/fy/cx/cy or --auto-calibrate")

    follow_controller.set_focal_length(_fy)
    _refresh_follow_estimator()
    print(f"[DIST] Monocular ranging: fy={_fy:.0f} object_height={args.object_height:.3f}m")

    streamer.start()
    print(f"[STREAM] video:  {rtsp.stream_url}")
    print(f"[STREAM] detections -> {sgc_host}:{args.sgc_port} (UDP)")

    global sgc_receiver
    sgc_receiver = SGCCommandReceiver(port=args.sgc_cmd_port)
    sgc_receiver.start()
    print(f"[STREAM] commands <- UDP {args.sgc_cmd_port} on {', '.join(_local_ipv4_addresses()) or 'all interfaces'}")
    print("[READY] tracking is live — click an object in the window, SPACE to follow, Q to land")
    set_hud_status(f"Sending to SGC {sgc_host}", (0, 200, 0), 4.0)

    hud.mode = args.mode
    hud.streaming = streamer is not None

    if not args.headless:
        cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracker", DISPLAY_WIDTH, DISPLAY_HEIGHT)
        cv2.setMouseCallback("Tracker", _on_mouse)

        splash = _make_splash("Connecting to vehicle...")
        cv2.imshow("Tracker", splash)
        cv2.waitKey(1)


def _check_sgc_peer():
    """Warn once when commands arrive from a different host than we stream to.

    The SGC laptop changes address between setups, so the sender is the most
    reliable proof of where the ground station actually is.
    """
    global _sgc_peer_warned
    if sgc_receiver is None or _sgc_peer_warned or not sgc_host:
        return
    peer = sgc_receiver.last_peer()
    if not peer or peer == sgc_host:
        return
    _sgc_peer_warned = True
    msg = f"SGC is at {peer}, not {sgc_host}"
    print(f"[SGC] {msg} - detections are going to the wrong machine. "
          f"Restart with --sgc-host {peer}")
    set_hud_status(msg, (255, 200, 0), 6.0)


def _trigger_panic_rtl(source: str) -> None:
    """Panic RTL: one action shared by the 'P' key and the SGC `panic_rtl` command.

    Issues RTL immediately, stops streaming and the command receiver, and marks
    the run as finished so `main_loop` returns. The vehicle is NOT landed here -
    it flies itself home.
    """
    global _panic_rtl_active
    if _panic_rtl_active:
        return
    _panic_rtl_active = True
    print(f"[PANIC] RTL triggered by {source}")
    set_hud_status("PANIC RTL - returning to launch", (0, 0, 255), 6.0)
    _failsafe_rtl(f"Panic RTL from {source}")


def _handle_sgc_command(cmd, detections):
    global _following
    if cmd.command_type == "select_target":
        if cmd.bbox is None:
            return
        matched = _find_best_match(cmd.bbox, cmd.class_name, detections)
        if matched is not None:
            detector.select_object(matched)
            _following = False
            _reset_lost_state()
            print(f"[SGC] Selected: {matched.class_name} ({matched.confidence * 100:.1f}%)")
        else:
            print(f"[SGC] No match for {cmd.class_name} bbox={cmd.bbox}")
    elif cmd.command_type == "follow_start":
        problems = _preflight_follow()
        if problems:
            msg = "Cannot follow: " + "; ".join(problems) + "."
            print(f"[SGC] {msg}")
            set_hud_status(msg, (0, 0, 255), 4.0)
            return
        if cmd.bbox is None:
            if detector.get_selected_object() is not None:
                _following = True
                _reset_lost_state()
                follow_controller.reset()
                print("[SGC] Follow started")
            return
        matched = _find_best_match(cmd.bbox, cmd.class_name, detections)
        if matched is not None:
            detector.select_object(matched)
            _following = True
            _reset_lost_state()
            follow_controller.reset()
            print(f"[SGC] Following: {matched.class_name} ({matched.confidence * 100:.1f}%)")
        else:
            print(f"[SGC] No match for {cmd.class_name} bbox={cmd.bbox}")
    elif cmd.command_type == "deselect_target":
        if detector.get_tracking_status():
            detector.clear_selection(preserve_lost=True)
            _following = False
            print("[SGC] Selection cleared (lost state preserved for popup)")
        else:
            detector.clear_selection()
            _following = False
            _reset_lost_state()
            print("[SGC] Selection cleared")
    elif cmd.command_type == "follow_stop":
        _following = False
        _reset_lost_state()
        print("[SGC] Follow stopped")
    elif cmd.command_type == "takeoff":
        print("[SGC] Takeoff requested")
        _handle_takeoff_button()
    elif cmd.command_type == "panic_rtl":
        _trigger_panic_rtl("SGC command")
    elif cmd.command_type == "servo":
        try:
            if cmd.pulse is not None:
                pulse = int(cmd.pulse)
            elif cmd.angle is not None:
                pulse = int(1500 + cmd.angle * (500.0 / 45.0))
                pulse = max(1000, min(2000, pulse))
            else:
                pulse = 1500
            drone.send_servo(channel=cmd.channel, pulse=pulse)
            print(f"[SGC] Servo: ch{cmd.channel} -> {pulse}us")
        except ValueError as exc:
            msg = f"Servo refused: {exc}"
            print(f"[SGC] {msg}")
            set_hud_status(msg, (0, 0, 255), 4.0)


def _handle_keyboard():
    global _following

    if args.headless:
        return None

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        land()
        return "quit"

    if key == ord('p'):
        # Panic RTL - immediate return to launch
        _trigger_panic_rtl("keyboard 'P'")
        return "quit"

    if key == 27:  # escape
        detector.clear_selection()
        tracking_session.clear_click()
        _reset_lost_state()
        _following = False
        print("Selection cleared (ESC)")

    if key == ord('r'):
        detector.clear_selection()
        tracking_session.clear_click()
        tracking_session.reset_loss_state()
        _reset_lost_state()
        _following = False
        print("Tracker reset (R)")

    if key == ord('h'):
        hud.hud_visible = not hud.hud_visible
        print(f"HUD: {'ON' if hud.hud_visible else 'OFF'}")
        time.sleep(0.2)

    if key == ord(' '):
        sel = detector.get_selected_object()
        if sel is None:
            msg = "No target selected. Click an object first (SPACE to follow)"
            print(msg)
            set_hud_status(msg, (255, 255, 0), 3.0)
        elif _following:
            _following = False
            _reset_lost_state()
            print(f"Follow stopped ({sel.class_name})")
        else:
            problems = _preflight_follow()
            if problems:
                msg = f"Cannot follow {sel.class_name}: " + "; ".join(problems) + "."
                print(msg)
                set_hud_status(msg, (0, 0, 255), 4.0)
            else:
                _following = True
                _reset_lost_state()
                follow_controller.reset()
                print(f"Following: {sel.class_name}")

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
    armed = False

    try:
        lat, lon, alt = drone.get_position()
    except Exception:
        pass

    try:
        bat = drone.get_battery_level()
    except Exception:
        pass

    try:
        ekf = drone.is_ekf_ok()
    except Exception:
        pass

    try:
        armed = drone.is_armed()
    except Exception:
        pass

    return TelemetryData(altitude=alt, battery=bat, lat=lat, lon=lon, ekf_ok=ekf, armed=armed)


def _make_splash(text: str) -> np.ndarray:
    img = np.zeros((720, 960, 3), dtype=np.uint8)
    cv2.putText(img, text, (960 // 2 - 180, 720 // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def _selected_distance(selected_obj, movement):
    if selected_obj is not None:
        source = getattr(selected_obj, "distance_source", None)
        if source is not None:
            distance = getattr(selected_obj, "distance_m", None)
            if not getattr(selected_obj, "distance_valid", False):
                return None
            try:
                distance = float(distance)
            except (TypeError, ValueError):
                return None
            return distance if math.isfinite(distance) and distance > 0.0 else None
    if not movement or not movement.get("distance_valid"):
        return None
    distance = movement.get("distance_m")
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        return None
    return distance if math.isfinite(distance) and distance > 0.0 else None


def _console_status(mode: str, movement: dict | None, selected_obj, tracker_state: str = "idle", rtl_countdown: float = -1, following: bool = False) -> None:
    cols = shutil.get_terminal_size().columns
    parts = []
    parts.append(f"\033[1;36m{mode.upper():>7}\033[0m")

    if selected_obj and movement:
        dist = _selected_distance(selected_obj, movement)
        vel = movement.get("vel_z", 0)
        yaw = movement.get("yaw_cmd", 0)

        if tracker_state == "lost":
            if rtl_countdown > 0:
                status = f"\033[1;31mLOST\033[0m RTL in {rtl_countdown:.0f}s"
            elif rtl_countdown == 0:
                status = "\033[1;31mRTL\033[0m"
            else:
                status = "\033[1;31mLOST\033[0m"
        elif abs(vel) < 0.05:
            status = "\033[1;33mHOVER\033[0m"
        elif vel > 0:
            status = f"\033[1;32mFORWARD\033[0m {vel:+.2f}m/s"
        else:
            status = f"\033[1;31mBACKWARD\033[0m {vel:+.2f}m/s"

        distance_text = "N/A" if dist is None else f"{dist:.2f}m"
        parts.append(f"{selected_obj.class_name} \033[1m{distance_text}\033[0m")
        parts.append(f"YAW {yaw:+.1f}\u00b0")
        parts.append(status)
    elif selected_obj is not None:
        parts.append(f"{selected_obj.class_name} \033[1m--\033[0m")
        parts.append("YAW --")
        parts.append("\033[1;35mSELECTED\033[0m (SPACE to follow)")
    elif following and tracker_state == "lost":
        if rtl_countdown > 0:
            status = f"\033[1;31mLOST\033[0m RTL in {rtl_countdown:.0f}s"
        elif rtl_countdown == 0:
            status = "\033[1;31mRTL\033[0m"
        else:
            status = "\033[1;31mLOST\033[0m"
        parts.append("\033[2mtarget\033[0m \033[1m--\033[0m")
        parts.append("YAW --")
        parts.append(status)
    elif tracker_state == "lost":
        parts.append("\033[2mtarget\033[0m \033[1m--\033[0m")
        parts.append("YAW --")
        parts.append("\033[1;31mLOST\033[0m (re-acquire)")
    else:
        parts.append("\033[2mno target\033[0m")
        parts.append("\033[2m--\033[0m")
        parts.append("\033[2mIDLE\033[0m")

    line = "  ".join(parts)
    if len(line) >= cols:
        line = line[:cols - 1]
    print(f"\r{line}\033[K", end="", flush=True)


def _follow_movement_dict(cmd):
    """Expose a follow command through the legacy movement-dict interface."""
    if cmd is None:
        return None
    distance = cmd.range_m
    distance_valid = (
        distance is not None
        and cmd.active
        and not cmd.lost
        and cmd.lane != "no_range"
    )
    if not distance_valid:
        distance = None
    dist = distance if distance is not None else 0.0
    return {
        "lidar_dist": dist,
        "vision_dist": dist,
        "fused_dist": dist,
        "vel_z": cmd.vx,
        "yaw_cmd": cmd.yaw_cmd,
        "lidar_on_target": False,
        "x_delta": cmd.x_delta,
        "y_delta": 0.0,
        "vx": cmd.vx,
        "vy": cmd.vy,
        "distance_m": distance,
        "distance_valid": distance_valid,
        "distance_source": cmd.distance_source if distance_valid else "none",
        "distance_confidence": cmd.distance_confidence if distance_valid else 0.0,
        "target_detection_id": cmd.detection_id,
    }


def _refresh_follow_estimator(frame_w=None, frame_h=None, force=False):

    global _fx, _fy, _cx, _cy, _calib_w, _calib_h, _distance_estimator, _follow_frame_size

    base_intrinsics = CameraIntrinsics(
        fx=_fx, fy=_fy, cx=_cx, cy=_cy,
        distortion={"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        frame_w=_calib_w, frame_h=_calib_h,
    )
    if frame_w is None or frame_h is None:
        frame_w, frame_h = _calib_w, _calib_h
    frame_size = (frame_w, frame_h)
    if not force and _follow_frame_size == frame_size:
        return
    intrinsics = base_intrinsics.scaled_to_frame(frame_w, frame_h)
    if intrinsics.is_valid():
        _distance_estimator = DistanceEstimator(
            intrinsics=intrinsics,
            config=EstimatorConfig(vision=VisionConfig(default_object_height_m=modules.app_config.OBJECT_HEIGHT)),
        )
        follow_controller.set_distance_estimator(_distance_estimator)
        follow_controller.set_focal_length(intrinsics.fy)
        _follow_frame_size = frame_size



def main_loop():
    global _last_infer_time, _lost_start_time, _rtl_triggered, _following
    global _fx, _fy, _cx, _cy, _calib_w, _calib_h

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

        height, width = image.shape[:2]

        if _calibrator is not None and not _calibrator.done:
            _calibrator.feed(image)
            if _calibrator.done and _calibrator.intrinsics:
                intr = _calibrator.intrinsics
                _fx = intr["fx"]
                _fy = intr["fy"]
                _cx = intr["cx"]
                _cy = intr["cy"]
                _calib_w = intr.get("calib_w") or width
                _calib_h = intr.get("calib_h") or height
                streamer.set_intrinsics(_fx, _fy, _cx, _cy, calib_w=_calib_w, calib_h=_calib_h)
                _refresh_follow_estimator(width, height, force=True)
                print(f"[CALIB] Calibration complete! fy={_fy:.1f} — distance estimate updated.")

        if sgc_receiver is not None:
            cmd = sgc_receiver.pop_command()
            if cmd is not None:
                _check_sgc_peer()
                _handle_sgc_command(cmd, detections)
                if _panic_rtl_active:
                    break

        _refresh_follow_estimator(width, height)
        annotate_detections(_distance_estimator._vision if _distance_estimator else None, detections, width, height)
        tracking_session.set_frame_size(width, height, DISPLAY_WIDTH, DISPLAY_HEIGHT)


        if tracking_session.has_pending_click():
            fx = tracking_session.mouse_click_x
            fy = tracking_session.mouse_click_y
            dismiss_rect = get_lost_dismiss_rect()
            if dismiss_rect and dismiss_rect[0] <= fx <= dismiss_rect[2] and dismiss_rect[1] <= fy <= dismiss_rect[3]:
                _reset_lost_state()
                detector.reset_tracking_lost()
                tracking_session.clear_click()
            else:
                if tracking_session.process_click(detections, detector, control):
                    _following = False
                    _reset_lost_state()
        else:
            if tracking_session.process_click(detections, detector, control):
                _following = False
                _reset_lost_state()

        selected_obj = detector.get_selected_object()
        is_tracking_lost = detector.get_tracking_status()
        tracker_state = "idle"
        tracking_conf = 0.0
        movement = None

        if selected_obj is not None:
            tracker_state = "tracking" if not is_tracking_lost else "lost"
            tracking_conf = detector.get_tracking_confidence()
        elif is_tracking_lost:
            tracker_state = "lost"
            tracking_conf = detector.get_tracking_confidence()
        else:
            _reset_lost_state()

        if _following:
            follow_target = selected_obj if (selected_obj is not None and not is_tracking_lost) else None
            follow_cmd = follow_controller.update(follow_target, image.shape)
            movement = _follow_movement_dict(follow_cmd)

            drone.send_movement_command_YAW(follow_cmd.yaw_cmd)
            if abs(follow_cmd.vx) > 1e-9 or abs(follow_cmd.vy) > 1e-9:
                drone.send_movement_command_XYA(follow_cmd.vy, follow_cmd.vx, MAX_ALT)
            else:
                drone.send_movement_command_XYA(0, 0, MAX_ALT)
                drone.hold_position()
            control.update_telemetry_from_track(fps, follow_cmd.yaw_cmd, follow_cmd.vx, False, follow_cmd.x_delta, 0.0)
        else:
            follow_controller.reset()
            movement = None
            control.update_telemetry_from_track(fps, 0, 0, False, 0, 0)
            drone.send_movement_command_YAW(0)
            drone.hold_position()

        if selected_obj is not None and not is_tracking_lost:
            _lost_start_time = None
            _rtl_triggered = False

        if _following and follow_cmd.lost and _lost_start_time is None:
            _lost_start_time = time.time()
            _rtl_triggered = False
            print("[LOST] Target lost — holding position until re-acquired")

        rtl_countdown = -1
        if _following:
            loss_timeout = follow_controller.config.loss_timeout
            lost_elapsed = follow_controller.lost_time_s
            if lost_elapsed >= loss_timeout:
                if not _rtl_triggered:
                    _rtl_triggered = True
                    if follow_controller.config.rtl_on_loss:
                        print(f"[LOST] Target lost for {loss_timeout:.0f}s — initiating RTL")
                        drone.send_rtl()
                        detector.clear_selection()
                        _following = False
                    else:
                        print(f"[LOST] Target lost for {loss_timeout:.0f}s — hovering (RTL disabled)")
                        set_hud_status("Target lost - hovering", (255, 255, 0), 3.0)
                rtl_countdown = 0
            elif lost_elapsed > 0:
                rtl_countdown = max(0.0, loss_timeout - lost_elapsed)

        status_state = tracker_state
        if tracker_state == "tracking" and selected_obj is not None and not _following:
            status_state = "selected"

        _console_status(args.mode, movement, selected_obj, tracker_state, rtl_countdown, _following)

        if streamer is not None:
            telemetry = _build_telemetry()
            streamer.push(image, detections, fps, movement=movement, telemetry=telemetry)

        image = draw_detection_window(image, detections, detector, movement=movement)

        if selected_obj is not None and movement is not None:
            annotate_tracking_overlay(image, fps, selected_obj, movement)
        elif selected_obj is not None:
            draw_follow_prompt(image, selected_obj.class_name)
        else:
            annotate_selection_overlay(image, detections)

        if hud.hud_visible:
            _update_hud_state(fps, tracker_state, selected_obj, tracking_conf, movement)

        if image is not None:
            try:
                display = compose_window(
                    image,
                    status_state,
                    selected_obj.class_name if selected_obj else None,
                    tracking_conf,
                    fps,
                    _last_infer_time,
                    armed=drone.is_armed(),
                )
            except Exception:
                display = cv2.resize(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                                     interpolation=cv2.INTER_LINEAR)
            if hud.hud_visible:
                draw_hud_background(display, oy=HEADER_FINAL)
                draw_fps(display, fps, _last_infer_time, oy=HEADER_FINAL)
                draw_hud_notification(display, oy=HEADER_FINAL)
                if tracker_state == "lost" and _following and not _rtl_triggered:
                    draw_lost_banner(display, rtl_countdown, oy=HEADER_FINAL)
            if not args.headless:
                cv2.imshow("Tracker", display)

    return "land"


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
    if not args.headless:
        cv2.destroyAllWindows()
    sys.exit(0)


def _failsafe_rtl(reason):
    print(f"\n[FAILSAFE] {reason}")
    print("[FAILSAFE] Issuing RTL — vehicle will return to launch")
    try:
        drone.send_rtl()
    except Exception as exc:
        print(f"[FAILSAFE] RTL command failed: {exc}")
    try:
        if sgc_receiver is not None:
            sgc_receiver.stop()
    except Exception:
        pass
    try:
        if streamer is not None:
            streamer.stop()
    except Exception:
        pass
    try:
        detector.cleanup()
    except Exception:
        pass
    try:
        if not args.headless:
            cv2.destroyAllWindows()
    except Exception:
        pass


try:
    setup()
except KeyboardInterrupt:
    print("\nStartup cancelled.")
    sys.exit(130)
except SystemExit:
    raise
except Exception as exc:
    import traceback
    print(f"\nStartup failed: {exc}")
    print("Re-run with more detail:")
    traceback.print_exc()
    sys.exit(1)

# Show window immediately so user isn't staring at a blank terminal
if not args.headless:
    splash = _make_splash("Initializing...")
    cv2.imshow("Tracker", splash)
    cv2.waitKey(1)

detector.get_image_size()
control.configure_PID(args.control)

try:
    main_loop()
except KeyboardInterrupt:
    print("\nShutting down...")
    _failsafe_rtl("Keyboard interrupt received")
    sys.exit(130)
except Exception:
    import traceback
    traceback.print_exc()
    _failsafe_rtl("Unhandled exception")
    sys.exit(1)

if _panic_rtl_active:
    print("[PANIC] Vehicle is flying itself home — the app has stopped streaming and exited.")
    sys.exit(0)
