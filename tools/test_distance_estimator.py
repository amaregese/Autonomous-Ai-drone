"""
Offline demonstration of the isolated distance estimator.

Runs entirely on CPU with no drone hardware / network / camera / LiDAR:

    python tools/test_distance_estimator.py
    python -m tools.test_distance_estimator --frames 40 --movement approach

Demonstrates:
  * camera calibration loaded from a JSON file (sample generated in a temp dir),
  * geometric vision distance + confidence,
  * deterministic simulated LiDAR,
  * deterministic simulated metric depth (ROI extraction + depth confidence),
  * confidence-weighted fusion of vision + LiDAR + depth (no fixed 65/35 blend),
  * Kalman-filtered range and velocity.

This tool does NOT start the drone and is NOT part of autonomous flight.
"""
import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.distance_estimator import (  # noqa: E402
    CameraIntrinsics,
    DistanceEstimator,
    EstimatorConfig,
    SimulatedDepthSource,
    SimulatedLidar,
)


def _make_sample_calibration() -> CameraIntrinsics:
    """Write a sample calibration JSON to a temp dir and load it back."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calibration.json")
        cal = CameraIntrinsics(
            fx=1112.0,
            fy=1112.0,
            cx=320.0,
            cy=240.0,
            distortion={},
            frame_w=640,
            frame_h=480,
        )
        cal.save_to_json(path)
        print(f"[demo] calibration loaded from {path}")
        return CameraIntrinsics.from_json_file(path, frame_w=640, frame_h=480)


def _build_scenario(mode: str, frames: int):
    """Return a deterministic (lidar_m, bbox_h_px) plan for the demo."""
    bbox = 422.0
    if mode == "approach":
        plan = [(3.18 + 0.25, bbox) for _ in range(max(2, frames // 2))]
        plan += [(1.5 + (3.18 + 0.25 - 1.5) * i / max(1, frames // 2), bbox + 60 * i / max(1, frames // 2))
                 for i in range(frames - max(2, frames // 2))]
        return plan
    if mode == "recede":
        plan = [(1.5 + 0.35, bbox + 60) for _ in range(max(2, frames // 2))]
        plan += [(3.0 + 0.35 * i / max(1, frames // 2), bbox + 60 + 80 * i / max(1, frames // 2))
                 for i in range(frames - max(2, frames // 2))]
        return plan
    return [(3.18, bbox) for _ in range(frames)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline distance-estimator demo")
    parser.add_argument("--frames", type=int, default=30, help="number of frames")
    parser.add_argument(
        "--movement",
        choices=["stationary", "approach", "recede"],
        default="approach",
    )
    args = parser.parse_args()

    intrinsics = _make_sample_calibration()
    depth_source = SimulatedDepthSource(values=[3.0], confidence=0.9)
    est = DistanceEstimator(
        config=EstimatorConfig(),
        intrinsics=intrinsics,
        depth_source=depth_source,
    )
    lidar = SimulatedLidar(values=[], confidence=0.95)
    lidar.set_sequence([3.18])  # placeholder; overridden per frame below
    frame_img = np.zeros((480, 640, 3), dtype=np.uint8)

    plan = _build_scenario(args.movement, args.frames)
    dt = 0.05

    print()
    print(f"{'Vision':>8} {'LiDAR':>7} {'Depth':>7} {'Fused':>7} {'Filter':>7} {'vel':>8} {'src':>9}  conf")
    print("-" * 76)

    for frame, (lidar_m, bbox_h) in enumerate(plan, start=1):
        lidar.set_distance(lidar_m)
        bbox_top = max(0, (480 - bbox_h) / 2.0)
        bbox_bottom = min(480, (480 + bbox_h) / 2.0)
        vision = est.estimate_vision(
            bbox_height_px=bbox_h,
            bbox_width_px=bbox_h * 0.5,
            class_name="person",
            detection_confidence=0.9,
            frame_w=640,
            frame_h=480,
            bbox_top_px=bbox_top,
            bbox_bottom_px=bbox_bottom,
        )
        lidar_meas = lidar.read()
        depth_source.set_distance(lidar_m)  # keep depth near truth for the demo
        depth_meas = est.estimate_depth(
            frame_img,
            (int(320 - bbox_h * 0.25), int(bbox_top), int(320 + bbox_h * 0.25), int(bbox_bottom)),
            detection_confidence=0.9,
        )
        state = est.update(vision, lidar_meas, depth_meas, dt=dt)

        vision_m = "-" if not vision.valid else f"{vision.distance_m:.2f}"
        depth_m = "-" if not depth_meas.valid else f"{depth_meas.range_m:.2f}"
        fused_m = "-" if state.fused_range_m is None else f"{state.fused_range_m:.2f}"
        filt_m = "-" if state.range_m is None else f"{state.range_m:.2f}"
        vel = "-" if state.velocity_mps is None else f"{state.velocity_mps:+.2f}"
        print(
            f"{vision_m:>8} {lidar_m:>7.2f} {depth_m:>7} {fused_m:>7} {filt_m:>7} "
            f"{vel:>8} {state.source.value:>9}  {state.confidence:.2f}"
        )

    print("-" * 76)
    print("\nFinal state:")
    final = est.update(
        est.estimate_vision(
            bbox_height_px=plan[-1][1],
            class_name="person",
            detection_confidence=0.9,
        ),
        lidar.read(),
        dt=dt,
    )
    print(f"  range:      {final.range_m:.2f} m")
    print(f"  velocity:   {final.velocity_mps:+.2f} m/s")
    print(f"  confidence: {final.confidence:.2f}")
    print(f"  source:     {final.source.value}")
    print()
    print("Demo complete. The estimator is NOT connected to autonomous flight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())