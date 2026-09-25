"""Temporary live-camera ground-truth distance capture tool (Task 4).

Completely isolated, offline/live-data-collection utility. It:

  1. opens a webcam / video file / RTSP source,
  2. runs the project's EXISTING YOLO11 detector
     (``modules/detector_yolo11``) on each frame,
  3. displays the live frame with boxes, class, confidence and selectable
     index,
  4. lets you select ONE detected target (keys 0-9),
  5. on ``c`` asks you to MANUALLY enter the independently measured distance,
  6. saves the current frame to ``benchmarks/distance/images/`` and appends a
     Task 3 manifest entry (ground truth is NEVER computed from the image),
  7. repeats at any number of distances without restarting.

This tool is NOT connected to autonomous flight. It does not import
``autonomous_drone_main``, ``modules.navigation``, ``modules.lidar_backend``,
MAVLink, ArduPilot, GUIDED controls, SGC or any flight logic. It never emits a
flight command. It exists only to build real-world validation data for the
Task 3 offline benchmark.

Ground-truth rule: the distance you type MUST be measured independently
(tape measure / laser rangefinder / measured rig position). It is never
calculated from bbox height, YOLO confidence, ZoeDepth, the distance estimator
or any prediction. Empty input cancels the capture; any invalid value
produces "Invalid distance. Sample was NOT saved." and nothing is recorded.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sys
import tempfile

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules import detector_yolo11 as detector  # noqa: E402
from modules.distance_estimator.dataset import (  # noqa: E402
    MIN_GT_DISTANCE_M,
    GroundTruthSample,
    create_empty_dataset,
    load_manifest,
)

DEFAULT_MANIFEST = os.path.join("benchmarks", "distance", "manifest.json")
DEFAULT_MODEL = os.path.join("YOLO", "yolo11n.pt")
INTRINSICS_SOURCE_OPTIONS = ("measured", "calibrated", "configured_default", "unknown")
_SAMPLE_ID_RE = re.compile(r"^live_(\d+)$")
_IMAGE_NAME_RE = re.compile(r"^live_\d{8}_\d{6}_(\d{3})\.jpg$")
_NAME_SEQUENCE = {"n": 0}

HEADER_TEXT = "LIVE DISTANCE DATA COLLECTION"


def parse_source(text) -> tuple:
    """Resolve ``--source`` into (kind, value).

    kind is one of ``camera``, ``video`` or ``rtsp``. A plain integer string is
    treated as a webcam index; a URL as RTSP; anything else as a video file.
    """
    s = str(text).strip()
    if s.isdigit():
        return "camera", int(s)
    if s.lower().startswith(("rtsp://", "rtsps://")):
        return "rtsp", s
    return "video", s


def source_label(parsed_source: tuple) -> str:
    kind, value = parsed_source
    if kind == "camera":
        return f"webcam {value}"
    if kind == "rtsp":
        return f"RTSP {value}"
    return f"video file {value}"


def open_capture(source_value) -> object:
    try:
        cap = cv2.VideoCapture(source_value)
    except Exception:
        return None
    if cap is not None and cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    if cap is not None:
        cap.release()
    return None


class FakeCamera:
    """Deterministic, in-memory camera used by the self-test and unit tests."""

    def __init__(self, frame):
        self._frame = frame
        self._released = False

    def read(self):
        return True, self._frame.copy()

    def isOpened(self):
        return True

    def release(self):
        self._released = True


def load_detector(model_path: str) -> bool:
    """Load the EXISTING YOLO11 weights and wire them into the existing detector.

    Reuses ``modules.yolo11_detector.model.load_model``; the model, prediction
    loop and Detection type all come from the existing pipeline.
    """
    from modules.yolo11_detector import api as _detector_api
    from modules.yolo11_detector.model import load_model

    model, classes = load_model(model_path)
    if model is None:
        return False
    _detector_api.model = model
    _detector_api.classes = classes
    return True


def detect_objects(frame) -> list:
    """Run the existing YOLO11 detector on an individual frame."""
    return detector.detect_objects(frame)


def detection_to_dict(detection) -> dict:
    return {
        "class_name": detection.class_name,
        "confidence": float(detection.confidence),
        "bbox": {
            "x": float(detection.Left),
            "y": float(detection.Top),
            "width": float(detection.width),
            "height": float(detection.height),
        },
        "center": detection.Center,
    }


def detection_label(index: int, detection) -> str:
    return f"[{index}] {detection.class_name} {detection.confidence:.2f}"


def selected_info_lines(detection) -> list:
    return [
        "Selected target:",
        f"  class = {detection.class_name}",
        f"  confidence = {detection.confidence:.3f}",
        f"  bbox = x={detection.Left:.0f}, y={detection.Top:.0f}, "
        f"w={detection.width:.0f}, h={detection.height:.0f}",
        f"  bbox_height_px = {detection.height:.0f}",
    ]


def hud_lines(source_text, frame_w, frame_h, n_detections, samples_captured) -> list:
    return [
        HEADER_TEXT,
        f"Source: {source_text}",
        f"Frame: {frame_w}x{frame_h}   Detections: {n_detections}   "
        f"Samples captured: {samples_captured}",
        "Keys: 0-9 select, r refresh, c capture, q quit",
    ]


def make_geometric_estimator(intrinsics_json):
    """Optional informational estimator. None when no calibration is supplied.

    The estimate is ONLY an overlay label; it never seeds ground truth and never
    reaches flight control.
    """
    if not intrinsics_json:
        return None
    from modules.distance_estimator.calibration import CameraIntrinsics
    from modules.distance_estimator.config import VisionConfig
    from modules.distance_estimator.vision import VisionDistanceEstimator

    data = json.loads(intrinsics_json)
    intrinsics = CameraIntrinsics.from_dict(data)
    return VisionDistanceEstimator(intrinsics, VisionConfig())


def estimate_distance_m(estimator, detection, frame_w, frame_h):
    if estimator is None or not estimator.geometry_available:
        return None
    measurement = estimator.estimate(
        bbox_height_px=detection.height,
        bbox_width_px=detection.width,
        class_name=detection.class_name,
        detection_confidence=detection.confidence,
        frame_w=frame_w,
        frame_h=frame_h,
        bbox_top_px=detection.Top,
        bbox_bottom_px=detection.Bottom,
    )
    return measurement.distance_m if measurement.valid else None


def estimate_line(estimator, detection, frame_w, frame_h) -> str:
    if estimator is None or not estimator.geometry_available:
        return "Estimated distance: NOT AVAILABLE (uncalibrated)"
    value = estimate_distance_m(estimator, detection, frame_w, frame_h)
    if value is None:
        return "Estimated distance: NOT AVAILABLE (geometry/class unsupported)"
    return f"Estimated distance: {value:.2f} m"


def parse_ground_truth(text):
    """Return None (cancel) for empty input, or a validated float.

    Rejects non-numeric, NaN, infinity, zero, negative and values below the
    shared Task 3 minimum (``MIN_GT_DISTANCE_M`` = 0.25 m). Raises ValueError
    with a human message otherwise.
    """
    t = "" if text is None else str(text).strip()
    if t == "":
        return None
    try:
        value = float(t)
    except (TypeError, ValueError):
        raise ValueError("distance must be a number")
    if not math.isfinite(value):
        raise ValueError("distance must be finite (got NaN/infinity)")
    if value < MIN_GT_DISTANCE_M:
        raise ValueError(
            f"distance must be >= {MIN_GT_DISTANCE_M:g} m "
            "(never 0, negative or effectively zero)"
        )
    return value


def next_image_name(images_dir: str) -> str:
    _NAME_SEQUENCE["n"] += 1
    base = datetime.datetime.now().strftime("live_%Y%m%d_%H%M%S")
    sequence = _NAME_SEQUENCE["n"]
    while os.path.exists(os.path.join(images_dir, f"{base}_{sequence:03d}.jpg")):
        sequence += 1
    return f"{base}_{sequence:03d}.jpg"


def next_sample_id(existing_ids) -> str:
    highest = 0
    for sample_id in existing_ids:
        match = _SAMPLE_ID_RE.fullmatch(str(sample_id))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"live_{highest + 1:03d}"


def _atomic_manifest_save(dataset) -> None:
    path = dataset.manifest_path
    target_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(target_dir, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except OSError as exc:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise OSError(f"failed to update manifest {path}: {exc}") from exc


class SampleRecorder:
    """Reads/writes the Task 3 manifest and image files without inventing data."""

    def __init__(
        self,
        manifest_path: str,
        camera_name: str = "live-camera",
        intrinsics_source: str = "unknown",
        intrinsics=None,
    ) -> None:
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        self.images_dir = os.path.join(self.manifest_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        if os.path.exists(self.manifest_path):
            self.dataset = load_manifest(self.manifest_path)
        else:
            self.dataset = create_empty_dataset(
                self.manifest_path,
                camera_name=camera_name,
                intrinsics_source=intrinsics_source,
                intrinsics=intrinsics,
            )
        if intrinsics is not None:
            self.dataset.camera.intrinsics = intrinsics
            self.dataset.camera.intrinsics_source = intrinsics_source
            self.dataset.camera_available = True

    def set_frame_dims(self, frame_w: int, frame_h: int) -> None:
        self.dataset.camera.frame_w = int(frame_w)
        self.dataset.camera.frame_h = int(frame_h)

    def save_sample(
        self,
        frame,
        detection_info: dict,
        distance_m: float,
        measurement_method: str,
        scene,
        lighting,
        pose,
        notes: str,
    ) -> tuple:
        problems = []
        if not isinstance(distance_m, (int, float)) or isinstance(distance_m, bool):
            problems.append("ground truth must be numeric")
        elif not math.isfinite(float(distance_m)):
            problems.append("ground truth must be finite")
        elif float(distance_m) < MIN_GT_DISTANCE_M:
            problems.append("ground truth must be >= 0.25 m")
        if problems:
            raise ValueError("; ".join(problems))

        sample_id = next_sample_id(set(self.dataset.sample_ids))
        image_name = next_image_name(self.images_dir)
        relative_image = "/".join(("images", image_name))
        absolute_image = os.path.join(self.images_dir, image_name)

        if frame is None or frame.size == 0:
            raise ValueError("cannot save an empty frame")

        written = cv2.imwrite(absolute_image, frame)
        if not written:
            raise OSError(f"failed to write image {absolute_image}")

        sample_dict = {
            "id": sample_id,
            "image": relative_image,
            "ground_truth_distance_m": float(distance_m),
            "class_name": detection_info["class_name"],
            "measurement_method": measurement_method,
            "bbox": detection_info["bbox"],
            "detection_confidence": detection_info["confidence"],
            "scene": scene or None,
            "lighting": lighting or None,
            "pose": pose or None,
            "notes": notes,
        }
        sample = GroundTruthSample.from_dict(sample_dict)
        self.dataset.samples.append(sample)
        try:
            _atomic_manifest_save(self.dataset)
        except OSError as exc:
            print(
                f"WARNING: manifest update failed after image was written. "
                f"Orphaned image (needs manual cleanup): {absolute_image}",
                file=sys.stderr,
            )
            raise
        return sample_id, relative_image, absolute_image


def capture_sequence(
    recorder: SampleRecorder,
    cap,
    start_frame,
    start_detections,
    selected_index,
    distance_m,
    detect_fn,
    args,
    notes_prefix: str,
    burst_note: str,
) -> int:
    burst = max(1, int(args.burst))
    saved = 0
    if selected_index < 0 or selected_index >= len(start_detections):
        return 0

    info = detection_to_dict(start_detections[selected_index])
    notes = f"source={args.source}; burst={burst_note}; {notes_prefix}".strip()

    for i in range(burst):
        if i == 0:
            frame = start_frame
        else:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed during burst; burst truncated.")
                break
            detections = detect_fn(frame)
            if 0 <= selected_index < len(detections):
                info = detection_to_dict(detections[selected_index])

        recorder.set_frame_dims(frame.shape[1], frame.shape[0])
        try:
            sample_id, rel_image, abs_image = recorder.save_sample(
                frame,
                info,
                distance_m,
                measurement_method=args.method,
                scene=args.scene,
                lighting=args.lighting,
                pose=args.pose,
                notes=notes,
            )
        except (ValueError, OSError) as exc:
            print(f"Sample NOT saved: {exc}", file=sys.stderr)
            break
        print(f"Saved {sample_id}: {abs_image} (ground truth {distance_m:.3f} m)")
        saved += 1
    return saved


def render_overlay(
    frame,
    detections,
    selected_index,
    status_lines,
    selected_lines,
    capture_summary,
):
    img = frame.copy()
    for idx, detection in enumerate(detections):
        is_selected = idx == selected_index
        color = (0, 0, 255) if is_selected else (0, 255, 0)
        thickness = 4 if is_selected else 2
        cv2.rectangle(
            img,
            (int(detection.Left), int(detection.Top)),
            (int(detection.Right), int(detection.Bottom)),
            color,
            thickness,
        )
        label = detection_label(idx, detection)
        _draw_text(img, label, (int(detection.Left), int(detection.Top) - 14), color)

    cursor_y = 6
    for line in status_lines:
        _draw_text(img, line, (6, cursor_y))
        cursor_y += 20

    if selected_lines:
        cursor_y = img.shape[0] - 12
        for line in reversed(selected_lines):
            _draw_text(img, line, (6, cursor_y - 16))
            cursor_y -= 19
    if capture_summary:
        _draw_text(img, capture_summary, (6, img.shape[0] - (58 if selected_lines else 34)))

    return img


def _draw_text(img, text, pos, color=(255, 255, 255), scale=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = pos
    cv2.rectangle(img, (x, y - th), (x + tw, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def run(
    args,
    key_fn=None,
    prompt_fn=None,
    open_fn=None,
    detect_fn=None,
    show_fn=None,
):
    """Main live loop. Injectable I/O so the same code runs headless in tests."""
    logger = print
    window = not args.no_window
    do_show = show_fn if show_fn is not None else (cv2.imshow if window else (lambda *a, **k: None))
    do_key = key_fn if key_fn is not None else (lambda: (cv2.waitKey(1) & 0xFF) if window else None)
    do_open = open_fn if open_fn is not None else open_capture
    do_detect = detect_fn if detect_fn is not None else detect_objects
    prompt = prompt_fn if prompt_fn is not None else input

    source_parsed = parse_source(args.source)
    cap = do_open(source_parsed[1])
    if cap is None or not cap.isOpened():
        logger(f"Camera/source unavailable: {args.source}")
        return 2

    if not load_detector(args.model_path):
        logger(f"Failed to load YOLO model: {args.model_path}")
        cap.release()
        return 2

    intrinsics = None
    if args.intrinsics:
        try:
            intrinsics = json.loads(args.intrinsics)
        except ValueError as exc:
            logger(f"Invalid --intrinsics JSON: {exc}")
            cap.release()
            return 2

    recorder = SampleRecorder(
        args.manifest,
        camera_name=args.camera_name,
        intrinsics_source=args.intrinsics_source,
        intrinsics=intrinsics,
    )
    estimator = None
    if intrinsics is not None:
        try:
            estimator = make_geometric_estimator(json.dumps(intrinsics))
        except Exception as exc:
            logger(f"Estimator not available (informational overlay off): {exc}")

    selected_index = None
    detections = []
    samples_captured = 0
    capture_summary = None
    last_capture = None
    window_name = "live_distance_capture"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger("Camera read failed (disconnected or end of source).")
                break
            recorder.set_frame_dims(frame.shape[1], frame.shape[0])
            detections = do_detect(frame)

            if selected_index is not None and (
                selected_index < 0 or selected_index >= len(detections)
            ):
                selected_index = None

            if window:
                status = hud_lines(
                    source_label(source_parsed),
                    frame.shape[1],
                    frame.shape[0],
                    len(detections),
                    samples_captured,
                )
                selected_lines = []
                if selected_index is not None and selected_index < len(detections):
                    det = detections[selected_index]
                    selected_lines = [
                        *selected_info_lines(det),
                        estimate_line(estimator, det, frame.shape[1], frame.shape[0]),
                        "Ground truth: NOT ENTERED",
                    ]
                annotated = render_overlay(
                    frame,
                    detections,
                    selected_index,
                    status,
                    selected_lines,
                    capture_summary,
                )
                do_show(window_name, annotated)

            key = do_key()
            if key in (ord("q"), ord("Q"), 27):
                break
            if ord("0") <= key <= ord("9"):
                index = key - ord("0")
                if index < len(detections):
                    selected_index = index
                    det = detections[index]
                    logger(f"Selected: {detection_label(index, det)}")
                    for line in selected_info_lines(det):
                        logger("  " + line)
                else:
                    logger(f"Invalid target index {index} (only {len(detections)} detection(s)).")
            elif key in (ord("r"), ord("R")):
                selected_index = None
                capture_summary = None
                logger("Selection cleared.")
            elif key in (ord("c"), ord("C")):
                if selected_index is None or selected_index >= len(detections):
                    logger("No target selected. Select a target (0-9) first.")
                    continue
                text = prompt("Enter independently measured distance in meters: ")
                try:
                    distance = parse_ground_truth(text)
                except ValueError as exc:
                    logger(f"Invalid distance. Sample was NOT saved ({exc}).")
                    continue
                if distance is None:
                    logger("Sample not saved.")
                    continue
                now = datetime.datetime.now()
                notes_prefix = now.isoformat(timespec="seconds")
                saved = capture_sequence(
                    recorder,
                    cap,
                    frame,
                    detections,
                    selected_index,
                    distance,
                    do_detect,
                    args,
                    notes_prefix=notes_prefix,
                    burst_note=str(max(1, int(args.burst))),
                )
                samples_captured += saved
                if saved:
                    det = detections[selected_index]
                    estimated = estimate_distance_m(
                        estimator, det, frame.shape[1], frame.shape[0]
                    )
                    if estimated is not None and estimator is not None:
                        last_capture = (
                            f"Last sample: GT {distance:.2f} m | "
                            f"Estimated {estimated:.2f} m | Error {abs(estimated - distance):.2f} m"
                        )
                    else:
                        last_capture = f"Last sample: ground truth {distance:.2f} m saved"
                    capture_summary = last_capture
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

    logger(f"Samples captured: {samples_captured}")
    return 0


def run_self_test() -> int:
    """Headless verification with a fake frame/camera; requires no hardware."""
    workdir = tempfile.mkdtemp(prefix="live_capture_selftest_")
    manifest = os.path.join(workdir, "manifest.json")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (300, 180), (460, 440), (0, 128, 255), -1)

    from modules.yolo11_detector.types import Detection

    def detect_fn(_frame):
        return [
            Detection(300, 180, 460, 440, 0, "person", 0.91),
            Detection(80, 120, 220, 220, 2, "car", 0.84),
        ]

    keys = iter([ord("0"), ord("c"), ord("c"), ord("c"), ord("q")])
    prompts = iter(["3.2", "2.1", "nope"])

    args = argparse.Namespace(
        source="0",
        model_path=DEFAULT_MODEL,
        manifest=manifest,
        camera_name="live-camera",
        intrinsics_source="unknown",
        intrinsics=None,
        method="manual_measurement",
        scene="live_camera",
        lighting="unknown",
        pose="unknown",
        notes="",
        burst=1,
        conf=None,
        no_window=True,
    )
    if args.conf is not None:
        detector.configure_detector(confidence_threshold=args.conf)

    rc = run(
        args,
        key_fn=lambda: next(keys, None),
        prompt_fn=lambda _prompt: next(prompts, None),
        open_fn=lambda _value: FakeCamera(frame),
        detect_fn=detect_fn,
    )

    dataset = load_manifest(manifest)
    image_files = [
        f
        for f in os.listdir(os.path.join(workdir, "images"))
        if _IMAGE_NAME_RE.match(f)
    ]
    ok = rc == 0 and len(dataset.samples) == 2 and len(image_files) == 2
    truths = sorted(s.ground_truth_distance_m for s in dataset.samples)
    ok = ok and truths == [2.1, 3.2]
    ok = ok and all(s.detection_confidence is not None for s in dataset.samples)
    ok = ok and all(s.bbox is not None for s in dataset.samples)

    print("SELF-TEST: tool started: PASS")
    print(f"SELF-TEST: argument parsing / run loop: {'PASS' if rc == 0 else 'FAIL'}")
    print("SELF-TEST: dataset directory + manifest validation: PASS")
    print(f"SELF-TEST: fake-frame sample creation: {'PASS' if len(dataset.samples) == 2 else 'FAIL'}")
    print(
        "SELF-TEST: invalid measurement rejected (2 of 3 prompts "
        f"saved): {'PASS' if len(dataset.samples) == 2 else 'FAIL'}"
    )
    print(f"SELF-TEST: overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Live YOLO11 + manual ground-truth distance capture (Task 4). "
        "NO flight integration."
    )
    parser.add_argument("--source", default="0",
                        help="webcam index (e.g. 0), video file path, or RTSP URL")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--camera-name", default="live-camera")
    parser.add_argument("--intrinsics-source", default="unknown",
                        choices=INTRINSICS_SOURCE_OPTIONS,
                        help="do NOT claim calibration you do not have; default unknown")
    parser.add_argument("--intrinsics",
                        help='camera intrinsics JSON \'{"fx":..,"fy":..,"cx":..,"cy":..}\' when '
                             "the camera IS calibrated (enables the optional informational estimate)")
    parser.add_argument("--method", default="manual_measurement")
    parser.add_argument("--scene", default="live_camera")
    parser.add_argument("--lighting", default="unknown")
    parser.add_argument("--pose", default="unknown")
    parser.add_argument("--notes", default="")
    parser.add_argument("--burst", type=int, default=1,
                        help="frames saved per measured distance (all share that same distance)")
    parser.add_argument("--conf", type=float, default=None,
                        help="override YOLO confidence threshold")
    parser.add_argument("--no-window", action="store_true",
                        help="do not open an OpenCV window (headless/CI only)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the headless fake-camera self test and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.conf is not None:
        detector.configure_detector(confidence_threshold=args.conf)

    if os.path.isabs(args.manifest):
        args.manifest = os.path.normpath(args.manifest)
    else:
        args.manifest = os.path.normpath(os.path.join(PROJECT_ROOT, args.manifest))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())