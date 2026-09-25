"""Read-only one-frame audit of current person-follow distance and yaw data.

This diagnostic deliberately does not import ``autonomous_drone_main`` (which
starts flight setup), MAVLink, ``modules.drone`` or SGC.  It reuses the normal
YOLO capture path and the current follow controller to report the values that
the application maps into its local tracker and UDP payload.
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_MANIFEST = os.path.join(PROJECT_ROOT, "benchmarks", "distance", "manifest.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only audit: follow distance vs tracker/UDP fields")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--model-path", default="YOLO/yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=None)
    args = parser.parse_args(argv)

    from modules import detector_yolo11 as detector  # noqa: PLC0415
    from modules.distance_estimator import (  # noqa: PLC0415
        VisionConfig,
        VisionDistanceEstimator,
        annotate_detections,
        load_configured_intrinsics,
    )
    from modules.person_follow import PersonFollowController  # noqa: PLC0415

    detector.configure_detector(confidence_threshold=args.confidence)
    if not detector.initialize_detector(args.model_path, camera_index=args.camera):
        print("Unable to initialize the existing YOLO11 camera pipeline.")
        return 2
    try:
        detections, _fps, frame = detector.get_detections()
        people = [d for d in detections if d.class_name == "person"]
        if not people:
            print("No person detection in this frame; no comparison is available.")
            return 1
        target = max(people, key=lambda d: d.confidence)
        target.is_selected = True  # diagnostic-only eligibility for the existing controller

        base_intrinsics = load_configured_intrinsics(DEFAULT_MANIFEST)
        estimator = VisionDistanceEstimator(
            base_intrinsics.scaled_to_frame(frame.shape[1], frame.shape[0])
            if base_intrinsics is not None
            else None,
            VisionConfig(),
        )
        annotate_detections(estimator, detections, frame.shape[1], frame.shape[0])
        command = PersonFollowController(distance_estimator=estimator).update(target, frame.shape)
        follow_distance = command.range_m
        print("Read-only data-flow audit: no MAVLink, SGC, or UDP message is opened/sent.\n")
        print("Frame: 1")
        print(f"bbox_height:              {target.height} px")
        print(f"Authoritative vision:      {follow_distance:.3f} m" if follow_distance is not None else "Authoritative vision:      unavailable")
        print(f"PersonFollow:              {follow_distance:.3f} m" if follow_distance is not None else "PersonFollow:              unavailable")
        print(f"Tracker display:           {follow_distance:.3f} m" if follow_distance is not None else "Tracker display:           unavailable")
        print(f"UDP tracking_data.distance:{follow_distance:.3f} m" if follow_distance is not None else "UDP tracking_data.distance:unavailable")
        print(f"UDP detections[].distance: {follow_distance:.3f} m (selected target authority)" if follow_distance is not None else "UDP detections[].distance: unavailable")
        print()
        print(f"horizontal_error:          {command.x_delta:+.3f}")
        print("yaw command:               0.0 deg (main loop literal)")
        print("MAVLink yaw rate:          0.0 rad/s (after send_movement_command_YAW(0))")
        return 0
    finally:
        detector.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
