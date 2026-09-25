"""Read-only camera-coordinate diagnostic for autonomous person follow.

This tool deliberately uses the same ``modules.detector_yolo11`` capture and
YOLO11 path as ``autonomous_drone_main.py``.  It never imports flight-control
modules, opens no MAVLink link, and sends no vehicle command.

Use a recognisable asymmetric object (for example, a sheet labelled ``LEFT``)
and place it physically left, then right, of the camera.  Compare that with the
on-screen LEFT/RIGHT labels and the printed target position.  A camera driver
or device may mirror an image before OpenCV receives it; this tool identifies
the coordinates used by YOLO, but it cannot infer that physical-camera setting
without this visual check.

Examples:
    python tools/camera_mirroring_diagnostic.py --camera 0
    python tools/camera_mirroring_diagnostic.py --camera 0 --no-window --frames 60
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Optional

import cv2


# ``python tools/camera_mirroring_diagnostic.py`` starts with ``tools`` as the
# import root.  Add the project root so this standalone tool can call the same
# ``modules.detector_yolo11`` facade as the autonomous application.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# The raw camera frame is normalized once in ``source.read_frame`` before this
# tool, YOLO11, tracking, display and person-follow receive it.
SOFTWARE_NORMALIZATION_FLIP = True
YOLO_FRAME_MIRRORED = False
DISPLAY_FRAME_MIRRORED = False


@dataclass(frozen=True)
class HorizontalReport:
    frame_width: int
    bbox_x: int
    bbox_width: int
    center_x: float
    image_center_x: float
    horizontal_error: float
    position: str
    displayed_frame_mirrored: bool = DISPLAY_FRAME_MIRRORED
    yolo_frame_mirrored: bool = YOLO_FRAME_MIRRORED


def horizontal_report(detection, frame_width: int, center_band_ratio: float = 0.05) -> HorizontalReport:
    """Report a detection's horizontal position in the YOLO input frame.

    ``horizontal_error`` is normalized to [-1, 1]: negative means image-left,
    positive means image-right.  This is the convention consumed by
    ``PersonFollowController._x_delta``.
    """
    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    if center_band_ratio < 0.0:
        raise ValueError("center_band_ratio must be non-negative")

    image_center = frame_width / 2.0
    center_x = float(detection.Center[0])
    error = (center_x - image_center) / image_center
    if abs(error) <= center_band_ratio:
        position = "CENTER"
    elif error < 0.0:
        position = "LEFT"
    else:
        position = "RIGHT"
    return HorizontalReport(
        frame_width=frame_width,
        bbox_x=int(detection.Left),
        bbox_width=int(detection.width),
        center_x=center_x,
        image_center_x=image_center,
        horizontal_error=error,
        position=position,
    )


def _choose_person(detections):
    people = [d for d in detections if d.class_name == "person"]
    return max(people, key=lambda d: d.confidence) if people else None


def _draw_diagnostic(frame, target, report: Optional[HorizontalReport]):
    image = frame.copy()
    h, w = image.shape[:2]
    center = int(w / 2)
    cv2.line(image, (center, 0), (center, h), (0, 255, 255), 2)
    cv2.putText(image, "IMAGE LEFT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, "IMAGE RIGHT", (max(10, w - 170), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    lines = [
        "Software normalization flip: YES (once in source.read_frame)",
        "YOLO input/display mirrored: NO (normalized frame)",
        "Raise an asymmetric object/hand. Move it physically LEFT and RIGHT.",
        "Compare physical position with IMAGE LEFT/RIGHT. Press Q or ESC to quit.",
    ]
    if report is not None:
        cv2.rectangle(image, (int(target.Left), int(target.Top)),
                      (int(target.Right), int(target.Bottom)), (0, 255, 0), 2)
        cv2.circle(image, (int(report.center_x), int(target.Center[1])), 6, (0, 0, 255), -1)
        cv2.putText(image, f"PERSON: {report.position}",
                    (int(target.Left), max(48, int(target.Top) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        lines.append(
            f"person {report.position}: x={report.bbox_x}, width={report.bbox_width}, cx={report.center_x:.1f}, "
            f"image_cx={report.image_center_x:.1f}, error={report.horizontal_error:+.3f}"
        )
    else:
        lines.append("No person detection: coordinates unavailable")
    y = max(58, h - 20 * len(lines) - 10)
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
        y += 20
    return image


def _print_report(frame_number: int, report: HorizontalReport) -> None:
    """Print the requested physical-verification values in an easy-to-read form."""
    print(
        f"Frame: {frame_number}\n"
        f"Frame width: {report.frame_width}\n"
        f"BBox x: {report.bbox_x}\n"
        f"BBox width: {report.bbox_width}\n"
        f"Target center_x: {report.center_x:.1f}\n"
        f"Image center_x: {report.image_center_x:.1f}\n"
        f"Normalized horizontal error: {report.horizontal_error:+.3f}\n"
        f"Position: {report.position}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only YOLO camera left/right diagnostic")
    parser.add_argument("--camera", type=int, default=None, help="camera index (default: detector auto-select)")
    parser.add_argument("--model-path", default="YOLO/yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames; 0 runs until q")
    parser.add_argument("--no-window", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Import here so geometry tests remain camera/model independent.  This is
    # the exact existing detector facade used by autonomous_drone_main.py.
    from modules import detector_yolo11 as detector  # noqa: PLC0415

    detector.configure_detector(confidence_threshold=args.confidence)
    if not detector.initialize_detector(args.model_path, camera_index=args.camera):
        print("Unable to initialize the existing YOLO11 camera pipeline.")
        return 2

    print("Read-only diagnostic: no MAVLink connection or flight command is used.")
    print("Software normalization flip applied: YES (once in source.read_frame)")
    print("Software mirror detected downstream: NO (YOLO and display use normalized pixels)")
    print("Physical camera/driver mirroring cannot be determined automatically")
    print("from the image coordinates alone. Verify using an asymmetric physical object.")
    print("Pipeline: raw capture -> source.read_frame() flip -> YOLO11 model.predict(normalized frame) -> display.")
    count = 0
    try:
        while args.frames <= 0 or count < args.frames:
            detections, _fps, frame = detector.get_detections()
            if frame is None or frame.size == 0:
                continue
            target = _choose_person(detections)
            report = horizontal_report(target, frame.shape[1]) if target is not None else None
            if report is not None:
                _print_report(count + 1, report)
            if not args.no_window:
                cv2.imshow("camera_mirroring_diagnostic", _draw_diagnostic(frame, target, report))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            count += 1
    finally:
        detector.cleanup()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
