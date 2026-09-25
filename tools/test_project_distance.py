"""Tasks 6 & 7: observe-only YOLO11 -> person-distance diagnostic tool.

Reuses the project's EXISTING YOLO11 pipeline (``modules/yolo11_detector`` /
``modules/detector_yolo11``) and the existing isolated distance estimator
(``modules/distance_estimator``) with the current configured intrinsics
(``fy ~= 446.7 px``). It is strictly diagnostic / observe-only:

  * it never sends a MAVLink command,
  * it never sets velocity / altitude / mode,
  * it does not call ``modules.navigation``, ``modules.lidar_backend``,
    takeoff/landing, RTL, SGC or any flight control,
  * it does not recalibrate the camera or change any estimator parameter.

Task 7 adds a stability test: ``--stability`` records the person's estimated
distance over a fixed duration (default 30 s; use 30-60 s), prints summary
statistics (valid count / min / max / mean / std-dev) with a STABLE or
FLUCTUATING verdict for an approximately stationary person, and saves the
per-frame measurements + summary to a JSON or CSV file.

Run:

    python tools/test_project_distance.py --source 0
    python tools/test_project_distance.py --source 0 --frames 20 --no-window
    python tools/test_project_distance.py --source 0 --stability --duration 45
    python tools/test_project_distance.py --source 0 --stability --duration 60 --no-window --out-format csv
    python tools/test_project_distance.py --self-test        # headless

Per-frame output lines look like:

    person conf=0.86 bbox_height=182px distance=3.19m
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from typing import Callable, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.distance_estimator.calibration import (  # noqa: E402
    DEFAULT_CONFIGURED_INTRINSICS,
    CalibrationError,
    CameraIntrinsics,
    load_configured_intrinsics,
)
from modules.distance_estimator.config import VisionConfig  # noqa: E402
from modules.distance_estimator.vision import (  # noqa: E402
    VisionDistanceEstimator,
    annotate_detections,
    estimate_detection as measure_detection,
)

DEFAULT_MODEL = os.path.join("YOLO", "yolo11n.pt")
DEFAULT_MANIFEST = os.path.join(PROJECT_ROOT, "benchmarks", "distance", "manifest.json")

# Current configured intrinsics (source: configured_default). fy ~= 446.7 px was
# previously fitted from the 10-sample person-height baseline (Task 3/5); it is
# used as-is here and never refit by this tool.
DEFAULT_INTRINSICS = dict(DEFAULT_CONFIGURED_INTRINSICS)

_WINDOW_NAME = "project_distance_test"

# Task 7 stability verdict thresholds (diagnostic only): an approximately
# stationary person is STABLE when at least STABILITY_MIN_SAMPLES valid readings
# were recorded and the sample std-dev is at most STABILITY_MAX_RELATIVE_STD_PERCENT
# percent of the mean distance.
STABILITY_MAX_RELATIVE_STD_PERCENT = 15.0
STABILITY_MIN_SAMPLES = 3


def load_intrinsics(
    manifest_path: Optional[str] = None,
    intrinsics_json: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Optional[CameraIntrinsics]:
    """Return the current configured intrinsics for the diagnostic tool.

    Resolution order (no calibration is performed):
      1. ``intrinsics_json`` (explicit ``--intrinsics`` override),
      2. the ``camera.intrinsics`` block of ``manifest_path``
         (``benchmarks/distance/manifest.json`` -> configured_default),
      3. the documented configured-default intrinsics (``fy=446.7``).

    Returns ``None`` when a supplied JSON is invalid.
    """
    if intrinsics_json:
        try:
            data = json.loads(intrinsics_json)
        except ValueError as exc:
            log(f"Invalid --intrinsics JSON: {exc}")
            return None
        try:
            return CameraIntrinsics.from_dict(data)
        except Exception as exc:  # pragma: no cover - defensive
            log(f"Intrinsics invalid: {exc}")
            return None
    return load_configured_intrinsics(manifest_path, fallback=DEFAULT_INTRINSICS)


def make_estimator(intrinsics: Optional[CameraIntrinsics]) -> Optional[VisionDistanceEstimator]:
    """Build the geometric vision estimator with the current configuration."""
    if intrinsics is None:
        return None
    return VisionDistanceEstimator(intrinsics, VisionConfig())


def make_frame_estimator(
    intrinsics: Optional[CameraIntrinsics],
    frame_w: int,
    frame_h: int,
) -> Optional[VisionDistanceEstimator]:
    if intrinsics is None:
        return None
    try:
        scaled = intrinsics.scaled_to_frame(frame_w, frame_h)
    except CalibrationError:
        return make_estimator(intrinsics)
    return make_estimator(scaled)


def estimate_detection(
    estimator: Optional[VisionDistanceEstimator],
    detection,
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[float, float]]:
    """Estimate (distance_m, confidence) for one YOLO11 detection, or None."""
    source = getattr(detection, "distance_source", None)
    if source is not None:
        if not getattr(detection, "distance_valid", False):
            return None
        distance = getattr(detection, "distance_m", None)
        if distance is None:
            return None
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(distance) or distance <= 0.0:
            return None
        try:
            confidence = float(getattr(detection, "distance_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        return distance, max(0.0, min(1.0, confidence))
    measurement = measure_detection(estimator, detection, frame_w, frame_h)
    if measurement.valid and measurement.distance_m is not None:
        return float(measurement.distance_m), float(measurement.confidence)
    return None


def detection_line(
    estimator: Optional[VisionDistanceEstimator],
    detection,
    frame_w: int,
    frame_h: int,
) -> str:
    """Human-readable per-detection line (the Task 6 console format)."""
    estimate = estimate_detection(estimator, detection, frame_w, frame_h)
    distance_text = "N/A" if estimate is None else f"{estimate[0]:.2f}m"
    return (
        f"{detection.class_name} conf={detection.confidence:.2f} "
        f"bbox_height={round(detection.height)}px distance={distance_text}"
    )


def overlay_label(
    estimator: Optional[VisionDistanceEstimator],
    detection,
    frame_w: int,
    frame_h: int,
) -> str:
    """Short label drawn on the live diagnostic window."""
    estimate = estimate_detection(estimator, detection, frame_w, frame_h)
    distance_text = "N/A" if estimate is None else f"{estimate[0]:.2f}m"
    return f"{detection.class_name} {detection.confidence:.2f} | {distance_text}"


def render_frame(frame, detections, estimator) -> object:
    """Draw boxes + distance labels onto a copy of the frame (observe-only)."""
    import cv2  # noqa: PLC0415  (lazy: not needed by the headless self-test)

    img = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    for detection in detections:
        color = (0, 255, 0)
        cv2.rectangle(
            img,
            (int(detection.Left), int(detection.Top)),
            (int(detection.Right), int(detection.Bottom)),
            color,
            2,
        )
        text = overlay_label(estimator, detection, frame_w, frame_h)
        _draw_text(img, text, (int(detection.Left), int(detection.Top) - 12), color)
    return img


def _draw_text(img, text, pos, color=(255, 255, 255), scale=0.5):
    import cv2  # noqa: PLC0415  (lazy)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = pos
    cv2.rectangle(img, (x, y - th), (x + tw, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def run(
    args,
    acquire_fn: Callable[[], Optional[Tuple[List, float, object]]],
    show_fn: Optional[Callable[[object], None]] = None,
    key_fn: Optional[Callable[[], Optional[int]]] = None,
    log: Callable[[str], None] = print,
) -> int:
    """Observe-only live loop.

    ``acquire_fn()`` returns ``(detections, fps, frame)`` or ``None`` when the
    source is exhausted. Injected I/O keeps the loop fully testable headless.
    """
    intrinsics = load_intrinsics(getattr(args, "manifest", None), getattr(args, "intrinsics", None), log=log)
    if intrinsics is None:
        log("Distance estimator unavailable (no valid intrinsics); distances will print as N/A.")
    else:
        log(
            f"Distance intrinsics: fx={intrinsics.fx} fy={intrinsics.fy} "
            f"cx={intrinsics.cx} cy={intrinsics.cy} "
            f"(source: {getattr(args, 'intrinsics_source', 'configured_default')})"
        )
    log("Observe-only diagnostic. No flight control / MAVLink interaction.")
    log("Press Q or Esc to stop.")

    window = not getattr(args, "no_window", False)
    if window:
        import cv2  # noqa: PLC0415  (lazy: skipped by the headless self-test)

        default_show = lambda image: cv2.imshow(_WINDOW_NAME, image)  # noqa: E731
        default_key = lambda: cv2.waitKey(1) & 0xFF  # noqa: E731
    else:
        default_show = lambda _image: None  # noqa: E731
        default_key = lambda: None  # noqa: E731
    do_show = show_fn if show_fn is not None else default_show
    do_key = key_fn if key_fn is not None else default_key

    max_frames = int(getattr(args, "frames", 0) or 0)
    frame_index = 0
    try:
        while True:
            result = acquire_fn()
            if result is None:
                log("Source ended.")
                break
            detections, fps, frame = result
            if frame is None:
                log("Source ended.")
                break

            frame_index += 1
            frame_h, frame_w = frame.shape[:2]
            frame_estimator = make_frame_estimator(intrinsics, frame_w, frame_h)
            annotate_detections(frame_estimator, detections, frame_w, frame_h)
            log(f"--- frame {frame_index} | {fps:.1f} fps | {len(detections)} detection(s) ---")
            for detection in detections:
                log("  " + detection_line(frame_estimator, detection, frame_w, frame_h))

            if window:
                do_show(render_frame(frame, detections, frame_estimator))

            if max_frames and frame_index >= max_frames:
                log(f"Reached frame limit ({max_frames}).")
                break

            key = do_key()
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if window:
            try:
                import cv2  # noqa: PLC0415  (lazy)

                cv2.destroyAllWindows()
            except Exception:
                pass
    return 0


def default_output_path(fmt: str = "json") -> str:
    """Default stability-results path under ``benchmarks/distance/results/``."""
    ext = "json" if fmt != "csv" else "csv"
    return os.path.join(
        PROJECT_ROOT, "benchmarks", "distance", "results", f"stability_results.{ext}"
    )


def compute_stability_metrics(
    distances,
    max_relative_std_percent: float = STABILITY_MAX_RELATIVE_STD_PERCENT,
    min_samples: int = STABILITY_MIN_SAMPLES,
) -> dict:
    """Summary statistics for a sequence of valid distance measurements (m).

    std-dev is the sample standard deviation (``statistics.stdev``); single
    samples yield 0.0 and empty series yield ``None``. ``stable`` is True only
    when at least ``min_samples`` valid readings exist whose std-dev is at most
    ``max_relative_std_percent`` percent of the mean distance.
    """
    n = len(distances)
    if n == 0:
        return {
            "valid_count": 0,
            "minimum_m": None,
            "maximum_m": None,
            "mean_m": None,
            "std_dev_m": None,
            "relative_std_percent": None,
            "stable": False,
            "note": "no valid person measurements recorded",
        }
    mean = float(statistics.fmean(distances))
    std_dev = float(statistics.stdev(distances)) if n > 1 else 0.0
    relative_std_percent = (std_dev / mean * 100.0) if mean > 0.0 else None
    stable = (
        n >= min_samples
        and relative_std_percent is not None
        and relative_std_percent <= max_relative_std_percent
    )
    note = (
        "approximately stationary (low relative std-dev)"
        if stable
        else "excessive fluctuation or insufficient samples"
    )
    return {
        "valid_count": n,
        "minimum_m": float(min(distances)),
        "maximum_m": float(max(distances)),
        "mean_m": mean,
        "std_dev_m": std_dev,
        "relative_std_percent": relative_std_percent,
        "stable": stable,
        "note": note,
    }


def format_stability_summary(metrics: dict) -> str:
    """Human-readable stability summary block (the Task 7 console output)."""
    if metrics.get("valid_count", 0) == 0:
        return "No valid person measurements collected."
    lines = [
        f"Valid measurements: {metrics['valid_count']}",
        f"Minimum distance:   {metrics['minimum_m']:.2f} m",
        f"Maximum distance:   {metrics['maximum_m']:.2f} m",
        f"Mean distance:      {metrics['mean_m']:.2f} m",
        f"Std deviation:      {metrics['std_dev_m']:.3f} m",
    ]
    if metrics.get("relative_std_percent") is not None:
        lines.append(f"Relative std (of mean): {metrics['relative_std_percent']:.1f}%")
    verdict = "STABLE" if metrics.get("stable") else "FLUCTUATING"
    lines.append(f"Stability: {verdict} ({metrics.get('note', '')})")
    return "\n".join(lines)


def write_stability_json(
    path,
    samples,
    metrics,
    intrinsics_source: str = "configured_default",
) -> str:
    """Persist per-frame measurements + summary summary as JSON; returns path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "intrinsics_source": intrinsics_source,
        "metrics": metrics,
        "samples": samples,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def write_stability_csv(
    path,
    samples,
    metrics,
    intrinsics_source: str = "configured_default",
) -> str:
    """Persist per-frame measurements as CSV; returns path.

    The CSV holds the per-frame rows; the numeric summary is written to the JSON
    output and the printed report. ``metrics`` is accepted for a uniform write
    API but is not serialized here.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = ["frame", "confidence", "bbox_height_px", "distance_m"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({name: sample.get(name) for name in fieldnames})
    return path


def write_stability(
    path,
    samples,
    metrics,
    fmt: str = "json",
    intrinsics_source: str = "configured_default",
) -> str:
    """Dispatch to JSON or CSV; returns the written path."""
    if fmt == "csv":
        return write_stability_csv(path, samples, metrics, intrinsics_source)
    return write_stability_json(path, samples, metrics, intrinsics_source)


def run_stability(
    args,
    acquire_fn: Callable[[], Optional[Tuple[List, float, object]]],
    show_fn: Optional[Callable[[object], None]] = None,
    key_fn: Optional[Callable[[], Optional[int]]] = None,
    log: Callable[[str], None] = print,
) -> int:
    """Observe-only stability test: record person distance over a fixed duration.

    Uses the identical estimator pipeline as ``run`` (no flight control /
    MAVLink, no estimator or ``fy`` change). One sample is recorded per person
    detection frame, then count / min / max / mean / std-dev are printed with a
    STABLE or FLUCTUATING verdict and measurements + summary are saved to a
    JSON or CSV file.
    """
    intrinsics = load_intrinsics(
        getattr(args, "manifest", None),
        getattr(args, "intrinsics", None),
        log=log,
    )
    if intrinsics is None:
        log("Distance estimator unavailable (no valid intrinsics); distances will be N/A.")
    else:
        log(
            f"Distance intrinsics: fx={intrinsics.fx} fy={intrinsics.fy} "
            f"cx={intrinsics.cx} cy={intrinsics.cy} "
            f"(source: {getattr(args, 'intrinsics_source', 'configured_default')})"
        )
    log("Stability test (observe-only). No flight control / MAVLink interaction.")
    log("Stand in front of the camera and keep roughly stationary. Press Q or Esc to stop early.")

    window = not getattr(args, "no_window", False)
    if window:
        import cv2  # noqa: PLC0415  (lazy: skipped by the headless self-test)

        default_show = lambda image: cv2.imshow(_WINDOW_NAME, image)  # noqa: E731
        default_key = lambda: cv2.waitKey(1) & 0xFF  # noqa: E731
    else:
        default_show = lambda _image: None  # noqa: E731
        default_key = lambda: None  # noqa: E731
    do_show = show_fn if show_fn is not None else default_show
    do_key = key_fn if key_fn is not None else default_key

    duration_s = float(getattr(args, "duration", 30.0) or 30.0)
    max_frames = int(getattr(args, "frames", 0) or 0)
    max_relative_std = float(getattr(args, "stability_max_relative_std", STABILITY_MAX_RELATIVE_STD_PERCENT) or STABILITY_MAX_RELATIVE_STD_PERCENT)
    min_samples = int(getattr(args, "stability_min_samples", STABILITY_MIN_SAMPLES) or STABILITY_MIN_SAMPLES)
    out_format = getattr(args, "out_format", "json")
    out_path = getattr(args, "out", None) or default_output_path(out_format)
    intrinsics_source = getattr(args, "intrinsics_source", "configured_default")

    start = time.perf_counter()
    samples = []
    frame_index = 0
    try:
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= duration_s:
                log(f"Reached duration limit ({duration_s:g}s).")
                break
            if max_frames and frame_index >= max_frames:
                log(f"Reached frame limit ({max_frames}).")
                break

            result = acquire_fn()
            if result is None:
                log("Source ended.")
                break
            detections, fps, frame = result
            if frame is None:
                log("Source ended.")
                break

            frame_index += 1
            frame_h, frame_w = frame.shape[:2]
            frame_estimator = make_frame_estimator(intrinsics, frame_w, frame_h)
            annotate_detections(frame_estimator, detections, frame_w, frame_h)
            persons = [d for d in detections if d.class_name == "person"]
            best = max(persons, key=lambda d: d.confidence) if persons else None
            if best is not None:
                estimate = estimate_detection(frame_estimator, best, frame_w, frame_h)
                if estimate is not None:
                    distance_m, _ = estimate
                    samples.append(
                        {
                            "frame": frame_index,
                            "confidence": float(best.confidence),
                            "bbox_height_px": float(best.height),
                            "distance_m": distance_m,
                        }
                    )
                    log(
                        f"  frame {frame_index} | {fps:.1f} fps | person "
                        f"conf={best.confidence:.2f} bbox_height={round(best.height)}px "
                        f"distance={distance_m:.2f}m"
                    )
                else:
                    log(
                        f"  frame {frame_index} | {fps:.1f} fps | person "
                        f"conf={best.confidence:.2f} bbox_height={round(best.height)}px "
                        f"distance=N/A"
                    )
            else:
                log(f"  frame {frame_index} | {fps:.1f} fps | no person")

            if window:
                do_show(render_frame(frame, detections, frame_estimator))

            key = do_key()
            if key in (ord("q"), ord("Q"), 27):
                log("Stopped by user.")
                break
    finally:
        if window:
            try:
                import cv2  # noqa: PLC0415  (lazy)

                cv2.destroyAllWindows()
            except Exception:
                pass

    metrics = compute_stability_metrics(
        [s["distance_m"] for s in samples],
        max_relative_std_percent=max_relative_std,
        min_samples=min_samples,
    )
    log("")
    log("=== Distance stability summary ===")
    for line in format_stability_summary(metrics).splitlines():
        log(line)
    written = write_stability(
        out_path, samples, metrics, out_format, intrinsics_source
    )
    log(f"Measurements saved to: {written}")
    return 0


def parse_source(text) -> object:
    """Webcam index for a plain integer, raw string (video/RTSP) otherwise."""
    s = str(text).strip()
    if s.isdigit():
        return int(s)
    return s


def _make_live_acquire(model_path: str, source, log: Callable[[str], None] = print):
    """Wire the existing YOLO11 prediction path to a plain capture.

    Mirrors ``tools/live_distance_capture.py``: the model + ``Detection`` type
    come from the existing ``modules.yolo11_detector`` pipeline, but the capture
    is opened directly so the MSMF stream-probe in ``source.initialize_capture``
    is bypassed (it can corrupt camera streams that only expose a single mode).
    No flight / MAVLink code is involved.
    """
    import time  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    from modules import detector_yolo11 as detector  # noqa: PLC0415 (heavy: torch)
    from modules.yolo11_detector import api as _detector_api  # noqa: PLC0415 (heavy: torch)
    from modules.yolo11_detector import config  # noqa: PLC0415
    from modules.yolo11_detector.model import load_model  # noqa: PLC0415 (heavy: torch)

    model, classes = load_model(model_path)
    if model is None:
        log(f"Failed to load YOLO model: {model_path}")
        return None, None

    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        log(f"Camera/source unavailable: {source}")
        return None, None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    log(f"Opened source: {source}")

    _detector_api.model = model
    _detector_api.classes = classes

    last_t = {"t": None}

    def acquire():
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        if frame.shape[1] > config.DEFAULT_WIDTH:
            scale = config.DEFAULT_WIDTH / frame.shape[1]
            frame = cv2.resize(
                frame,
                (config.DEFAULT_WIDTH, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        detections = detector.detect_objects(frame)
        now = time.perf_counter()
        if last_t["t"] is not None:
            fps = 1.0 / max(now - last_t["t"], 1e-6)
        else:
            fps = 0.0
        last_t["t"] = now
        return detections, float(fps), frame

    def close():
        try:
            cap.release()
        except Exception:
            pass

    return acquire, close


def run_self_test() -> int:
    """Headless verification with mock detections; no camera, no model."""
    import io  # noqa: PLC0415
    from contextlib import redirect_stdout  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from modules.yolo11_detector.types import Detection  # noqa: PLC0415 (mock boxes only)

    intrinsics = CameraIntrinsics.from_dict(dict(DEFAULT_INTRINSICS))
    estimator = make_estimator(intrinsics)

    detections = [
        Detection(300, 180, 460, 440, 0, "person", 0.86),   # h=260 -> 2.23 m
        Detection(80, 120, 220, 220, 2, "car", 0.84),       # h=100 -> 6.70 m
        Detection(40, 60, 240, 260, 39, "snowboard", 0.60),  # h=200, default h=0.5 -> 1.12 m
    ]
    expected = {
        "person conf=0.86 bbox_height=260px distance=2.23m",
        "car conf=0.84 bbox_height=100px distance=6.70m",
        "snowboard conf=0.60 bbox_height=200px distance=1.12m",
    }
    lines = {detection_line(estimator, d, 640, 480) for d in detections}
    lines_ok = lines == expected

    args = argparse.Namespace(
        source="0",
        model_path=DEFAULT_MODEL,
        manifest=DEFAULT_MANIFEST,
        intrinsics=None,
        intrinsics_source="configured_default",
        frames=2,
        no_window=True,
    )
    frame_count = {"n": 0}

    def acquire():
        if frame_count["n"] >= 2:
            return None
        frame_count["n"] += 1
        return detections, 30.0, np.zeros((480, 640, 3), dtype=np.uint8)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run(args, acquire_fn=acquire, key_fn=lambda: None, log=print)
    output = buf.getvalue()

    loop_ok = rc == 0 and "person conf=0.86 bbox_height=260px distance=2.23m" in output

    import tempfile  # noqa: PLC0415

    stability_out = os.path.join(tempfile.mkdtemp(prefix="tpd_stability_"), "stability_results.json")
    args_st = argparse.Namespace(
        source="0",
        model_path=DEFAULT_MODEL,
        manifest=None,
        intrinsics=None,
        intrinsics_source="configured_default",
        frames=3,
        no_window=True,
        duration=30.0,
        out=stability_out,
        out_format="json",
        stability_max_relative_std=STABILITY_MAX_RELATIVE_STD_PERCENT,
        stability_min_samples=STABILITY_MIN_SAMPLES,
    )
    heights = [320.0, 316.0, 322.0]

    def acquire_stability():
        if acquire_stability.count >= 3:
            return None
        i = acquire_stability.count
        acquire_stability.count += 1
        return (
            [Detection(300, 180, 460, 180 + heights[i], 0, "person", 0.9)],
            30.0,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )

    acquire_stability.count = 0
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_stability = run_stability(
            args_st, acquire_fn=acquire_stability, key_fn=lambda: None, log=print
        )
    stability_text = buf.getvalue()
    stability_ok = (
        rc_stability == 0
        and os.path.exists(stability_out)
        and "Valid measurements: 3" in stability_text
        and "STABLE" in stability_text
    )

    print("SELF-TEST: estimator math (person/car/default-height): "
          f"{'PASS' if lines_ok else 'FAIL'}")
    print("SELF-TEST: detect -> distance loop (headless, mock detections): "
          f"{'PASS' if loop_ok else 'FAIL'}")
    print("SELF-TEST: stability collection + stats + save (headless, mock): "
          f"{'PASS' if stability_ok else 'FAIL'}")
    if not lines_ok:
        print("  unexpected lines:")
        for line in sorted(lines ^ expected):
            print(f"    {line}")
    overall = lines_ok and loop_ok and stability_ok
    print(f"SELF-TEST: overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe-only YOLO11 -> distance diagnostic + stability test "
        "(Tasks 6/7). NO flight/MAVLink integration."
    )
    parser.add_argument("--source", default="0",
                        help="USB webcam index (default 0)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL,
                        help="YOLO11 weights (default YOLO/yolo11n.pt)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="dataset manifest holding the current configured intrinsics")
    parser.add_argument("--intrinsics",
                        help='optional explicit intrinsics JSON \'{"fx":..,"fy":..,"cx":..,"cy":..}\' '
                             "(overrides the configured defaults; diagnostic only)")
    parser.add_argument("--intrinsics-source", default="configured_default",
                        help="provenance label printed with the intrinsics")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO11 confidence threshold (default 0.3)")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (default 0 = run until quit)")
    parser.add_argument("--no-window", action="store_true",
                        help="do not open an OpenCV window (headless/CI)")
    parser.add_argument("--stability", action="store_true",
                        help="run the distance-stability test (person, Task 7)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="stability collection duration in seconds (default 30; requirement 30-60)")
    parser.add_argument("--out", default=None,
                        help="path to write stability results (see --out-format)")
    parser.add_argument("--out-format", choices=("json", "csv"), default="json",
                        help="stability results file format (default json)")
    parser.add_argument("--stability-max-relative-std", type=float,
                        default=STABILITY_MAX_RELATIVE_STD_PERCENT,
                        help="%% of the mean distance at which person distance is still stable "
                             "(default 15)")
    parser.add_argument("--stability-min-samples", type=int,
                        default=STABILITY_MIN_SAMPLES,
                        help="minimum valid person frames required for a STABLE verdict "
                             "(default 3)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the headless mock-detection self test and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.conf is not None and args.conf > 0:
        from modules import detector_yolo11 as detector  # noqa: PLC0415 (heavy: torch)

        detector.configure_detector(confidence_threshold=args.conf)

    source_value = parse_source(args.source)
    acquire, close = _make_live_acquire(args.model_path, source_value, log=print)
    if acquire is None:
        return 2

    try:
        if args.stability:
            return run_stability(args, acquire_fn=acquire)
        return run(args, acquire_fn=acquire)
    finally:
        if close is not None:
            close()


if __name__ == "__main__":
    raise SystemExit(main())