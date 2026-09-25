"""Task 5A - automatic USB camera checkerboard capture (image collection only).

Isolated, offline utility. It:

  1. opens a USB webcam (default index 0),
  2. shows the live feed with a countdown overlay,
  3. detects the checkerboard (configurable internal-corner size),
  4. saves the ORIGINAL frame approximately every ``--interval`` seconds when
     the board is detected, until ``--count`` successful images exist,
  5. writes per-image metadata to ``benchmarks/camera_calibration/manifest.json``.

Safety:
  * Does NOT calibrate the camera (no fx/fy/cx/cy computed here).
  * Does NOT import or touch flight code (``autonomous_drone_main``,
    ``modules.navigation``, ``modules.lidar_backend``, MAVLink, ArduPilot,
    SITL, SGC, RTSP, YOLO detector).
  * Frames are saved unmodified; corner drawings are display-only and never
    burned into the saved calibration images.
  * No distance to the board and no square size are requested or recorded -
    those are only needed by the later calibration step.

Run:
    python tools/capture_calibration_images.py
    python tools/capture_calibration_images.py --camera 0 --interval 5 --count 20
    python tools/capture_calibration_images.py --corners-x 9 --corners-y 6
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sys
import time
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_OUTPUT_DIR = os.path.join("benchmarks", "camera_calibration")
MANIFEST_VERSION = 1
_IMAGE_NAME_RE = re.compile(r"^calibration_(\d{3})\.jpg$")

EXAMPLES = """\
Command examples:

  Default 20 images, camera 0:
    python tools/capture_calibration_images.py

  Explicit camera, interval and count:
    python tools/capture_calibration_images.py --camera 0 --interval 5 --count 20

  Different checkerboard:
    python tools/capture_calibration_images.py \\
        --camera 0 \\
        --interval 5 \\
        --count 25 \\
        --corners-x 9 \\
        --corners-y 6
"""

GUIDANCE_TEXT = """\
Move the checkerboard between captures.
Try to cover:
  - center, left, right, top, bottom
  - different distances
  - different tilts
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically capture checkerboard images from a USB camera for "
            "later OpenCV intrinsic calibration. Image collection only - no "
            "calibration is performed and no flight code is touched."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=None,
                        help="requested frame width (falls back to actual)")
    parser.add_argument("--height", type=int, default=None,
                        help="requested frame height (falls back to actual)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="seconds between capture attempts (default: 5)")
    parser.add_argument("--count", type=int, default=20,
                        help="number of successful images to collect (default: 20)")
    parser.add_argument("--corners-x", type=int, default=9,
                        help="checkerboard inner corners per row (default: 9)")
    parser.add_argument("--corners-y", type=int, default=6,
                        help="checkerboard inner corners per column (default: 6)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help=f"output root directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--no-window", action="store_true",
                        help="do not open an OpenCV window (headless/CI only)")
    return parser


def validate_args(args) -> Optional[str]:
    """Return a human-readable problem, or None when configuration is OK."""
    if args.camera < 0:
        return f"invalid --camera: {args.camera} (must be >= 0)"
    if args.count < 1:
        return f"invalid --count: {args.count} (must be >= 1)"
    if not math.isfinite(args.interval) or args.interval <= 0:
        return f"invalid --interval: {args.interval} (must be > 0)"
    if args.corners_x < 2 or args.corners_y < 2:
        return (
            f"invalid checkerboard: {args.corners_x}x{args.corners_y} "
            "(inner corners must be >= 2 in both dimensions)"
        )
    if args.width is not None and args.width <= 0:
        return f"invalid --width: {args.width} (must be > 0)"
    if args.height is not None and args.height <= 0:
        return f"invalid --height: {args.height} (must be > 0)"
    return None


def open_camera(index: int, width: "Optional[int]" = None, height: "Optional[int]" = None):
    cap = cv2.VideoCapture(int(index))
    if not cap.isOpened():
        cap.release()
        return None
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return cap


_DOWNSCALE_MAX_DIM = 1024


def _find_chessboard_corners(gray, pattern, flags):
    try:
        return cv2.findChessboardCorners(gray, pattern, flags)
    except cv2.error:
        return False, None


def detect_checkerboard(frame, corners_x: int, corners_y: int):
    """Detect internal checkerboard corners; return (found, refined_corners).

    Full-resolution detection is attempted first. When the frame is very large
    (e.g. a 2560x1440 webcam) and the pattern is not found, detection is retried
    on a downscaled copy (corner coordinates are scaled back to the original
    frame) because flat-board detection is often more robust at a smaller size.
    """
    if frame is None or frame.size == 0:
        return False, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pattern = (int(corners_x), int(corners_y))
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    found, corners = _find_chessboard_corners(gray, pattern, flags)
    if not found and max(gray.shape[0], gray.shape[1]) > _DOWNSCALE_MAX_DIM:
        scale = _DOWNSCALE_MAX_DIM / float(max(gray.shape[0], gray.shape[1]))
        small = cv2.resize(
            gray,
            (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        found, corners = _find_chessboard_corners(small, pattern, flags)
        if found:
            corners = corners * (1.0 / scale)
    if not found:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    try:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    except cv2.error:
        pass
    return True, corners


def frame_is_valid(frame) -> bool:
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    return int(frame.shape[0]) > 0 and int(frame.shape[1]) > 0


class CalibrationStore:
    """Manages the output directory, unique filenames and the manifest."""

    def __init__(self, output_dir: str, camera_index: int, corners_x: int, corners_y: int) -> None:
        self.output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, output_dir))
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.output_dir, "manifest.json")
        self.camera_index = int(camera_index)
        self.corners_x = int(corners_x)
        self.corners_y = int(corners_y)
        self.records = self._load_existing_records()

    def _load_existing_records(self) -> list:
        if not os.path.exists(self.manifest_path):
            return []
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return []
        records = data.get("images") if isinstance(data, dict) else None
        return records if isinstance(records, list) else []

    def next_image_path(self) -> str:
        """Return the next available ``calibration_NNN.jpg`` (never overwrites)."""
        highest = 0
        for name in os.listdir(self.images_dir):
            match = _IMAGE_NAME_RE.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
        return os.path.join(self.images_dir, f"calibration_{highest + 1:03d}.jpg")

    def save(self, frame, timestamp: str) -> tuple:
        if not frame_is_valid(frame):
            raise ValueError("invalid frame (empty or zero-size)")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        path = self.next_image_path()
        if not cv2.imwrite(path, frame):
            raise OSError(f"failed to write image {path}")

        record = {
            "image": f"images/{os.path.basename(path)}",
            "camera_index": self.camera_index,
            "width": width,
            "height": height,
            "timestamp": timestamp,
            "checkerboard_corners_x": self.corners_x,
            "checkerboard_corners_y": self.corners_y,
        }
        self.records.append(record)
        self._write_manifest()
        return path, record

    def _write_manifest(self) -> None:
        payload = {
            "version": MANIFEST_VERSION,
            "camera_index": self.camera_index,
            "checkerboard_corners_x": self.corners_x,
            "checkerboard_corners_y": self.corners_y,
            "images": self.records,
        }
        tmp = self.manifest_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            os.replace(tmp, self.manifest_path)
        except OSError:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise


def _draw_text(img, text: str, pos, color=(255, 255, 255), scale=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = pos
    cv2.rectangle(img, (x, y - th), (x + tw, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def render_overlay(frame, args, captured: int, remaining: float, detected: bool, corners, status: str):
    """Annotated copy for LIVE DISPLAY only; the saved frame is untouched."""
    img = frame.copy()
    if detected and corners is not None:
        try:
            cv2.drawChessboardCorners(
                img, (args.corners_x, args.corners_y), corners, True
            )
        except cv2.error:
            pass

    lines = [
        f"Camera: {args.camera}",
        f"Resolution: {img.shape[1]}x{img.shape[0]}",
        f"Next capture: {remaining:.1f} s",
        f"Captured: {captured} / {args.count}",
        f"Checkerboard: {'DETECTED' if detected else 'NOT DETECTED'}",
        "Move/tilt the checkerboard between captures.",
        "s save debug | q / ESC stop",
    ]
    if status:
        lines.append(status)
    y = 20
    for line in lines:
        _draw_text(img, line, (6, y))
        y += 20
    return img


def run(
    args,
    open_fn=None,
    key_fn=None,
    show_fn=None,
    clock_fn=None,
    detect_fn=None,
    logger=print,
) -> int:
    """Main capture loop. Camera, keys, window, clock and detection are
    injectable so the same code runs headless in unit tests."""
    window = not args.no_window
    do_open = open_fn if open_fn is not None else open_camera
    do_key = (
        key_fn
        if key_fn is not None
        else (lambda: (cv2.waitKey(1) & 0xFF) if window else None)
    )
    do_show = show_fn if show_fn is not None else (cv2.imshow if window else (lambda *a, **k: None))
    clock = clock_fn if clock_fn is not None else time.monotonic

    def default_detect(frame):
        return detect_checkerboard(frame, args.corners_x, args.corners_y)

    do_detect = detect_fn if detect_fn is not None else default_detect

    cap = do_open(args.camera, args.width, args.height)
    if cap is None or not cap.isOpened():
        logger(f"Camera unavailable: {args.camera}")
        return 2

    store = CalibrationStore(args.output, args.camera, args.corners_x, args.corners_y)
    logger(f"Camera: {args.camera}")
    logger(f"Interval: {args.interval:g} s   Target: {args.count} images")
    logger(f"Output: {store.images_dir}")
    logger(GUIDANCE_TEXT)

    captured = 0
    announced_resolution = False
    status = ""
    next_due = clock() + float(args.interval)
    window_name = "capture_calibration_images"

    try:
        while captured < args.count:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger(
                    "Camera read failed (disconnected or end of source). "
                    f"Captured {captured}/{args.count}."
                )
                return 2

            if not announced_resolution and frame_is_valid(frame):
                logger(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                announced_resolution = True

            now = clock()
            detected, corners = do_detect(frame)
            remaining = next_due - now

            if remaining <= 0.0:
                if not detected:
                    logger("Checkerboard: NOT DETECTED - frame not saved.")
                elif not frame_is_valid(frame):
                    logger("Invalid frame - not saved.")
                else:
                    try:
                        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
                        path, record = store.save(frame, timestamp)
                    except (OSError, ValueError) as exc:
                        logger(f"Save failed: {exc}")
                    else:
                        captured += 1
                        status = f"Saved {os.path.basename(path)} ({captured}/{args.count})"
                        logger(
                            f"Checkerboard: DETECTED - saved {record['image']} "
                            f"({captured}/{args.count})"
                        )
                next_due = now + float(args.interval)

            if window:
                display = render_overlay(
                    frame, args, captured, max(0.0, next_due - now), detected, corners, status
                )
                do_show(window_name, display)

            key = do_key()
            if key in (ord("q"), ord("Q"), 27):
                logger(f"Capture stopped: {captured}/{args.count} images.")
                return 0
            if key in (ord("s"), ord("S")) and frame_is_valid(frame):
                debug_dir = os.path.join(store.output_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = os.path.join(debug_dir, f"debug_{stamp}_{captured:03d}.jpg")
                if cv2.imwrite(debug_path, frame):
                    status = f"Saved debug frame: {os.path.basename(debug_path)}"
                    logger(f"Debug frame saved: {debug_path}")
                else:
                    logger("Debug frame write failed.")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    logger("")
    logger("Calibration image capture complete.")
    logger("")
    logger(f"Successfully captured: {captured} images")
    logger("Output:")
    logger(store.images_dir)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    error = validate_args(args)
    if error:
        print(error, file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())