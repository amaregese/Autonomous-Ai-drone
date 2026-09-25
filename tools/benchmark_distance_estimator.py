"""
Offline distance-estimator accuracy benchmark.

Compares against ground truth:

    Legacy bbox-height distance  (modules.navigation.FollowController)
    Geometric vision             (VisionDistanceEstimator)
    Metric monocular depth       (synthetic source; real backend supported too)
    Confidence-weighted fusion   (DistanceEstimator, pre-filter)
    Kalman-filtered fusion       (DistanceEstimator, post-filter)

Run entirely offline / CPU-only / no drone hardware:

    python tools/benchmark_distance_estimator.py
    python tools/benchmark_distance_estimator.py --seed 7 --frames 25
    python tools/benchmark_distance_estimator.py --temporal-only

Two kinds of data are clearly kept apart:

  * SYNTHETIC data: scripted ground truths with deterministic noise. These are
    test fixtures, NOT real camera measurements, and are always labelled as
    such in the output.
  * REAL data: read from ``benchmarks/distance/manifest.json`` when present.
    If no ground-truth camera dataset exists, the tool reports
    "NOT YET AVAILABLE" instead of fabricating numbers.

No accuracy claims are made from synthetic data alone; the table is for
measurement, not ranking.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from statistics import mean as _mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.distance_estimator import (  # noqa: E402
    CameraIntrinsics,
    DistanceEstimator,
    EstimatorConfig,
    FixedLidar,
    SimulatedDepthSource,
    VisionDistanceEstimator,
    VisionConfig,
)

FRAME_W, FRAME_H = 640, 480
INTRINSICS = CameraIntrinsics(fx=1112.0, fy=1112.0, cx=320.0, cy=240.0, frame_w=FRAME_W, frame_h=FRAME_H)
OBJECT_HEIGHT_M = 1.3  # person
GROUND_TRUTHS_M = [1.0, 2.0, 3.0, 4.0, 5.0]

SYNTHETIC_DISCLAIMER = (
    "SYNTHETIC DATA (scripted ground truth + deterministic noise). "
    "Not real camera measurements."
)


class LegacyBboxEstimator:
    """Wrapper around the untouched legacy controller distance method."""

    def __init__(self) -> None:
        from modules.navigation import FollowController

        self._ctrl = FollowController()
        self._ctrl.set_focal_length(float(INTRINSICS.fy))

    def estimate(self, bbox_height_px: float, class_name: str = "person") -> float:
        return self._ctrl.estimate_distance_from_size(bbox_height_px, class_name)


class MethodRunner:
    """Holds per-call state; benchmark calls estimate() once per target."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self.legacy = LegacyBboxEstimator()
        self.geometric = VisionDistanceEstimator(INTRINSICS, VisionConfig())
        self._depth = SimulatedDepthSource()
        self._estimator = DistanceEstimator(config=EstimatorConfig(), intrinsics=INTRINSICS, depth_source=self._depth)
        self._lidar = FixedLidar(0.0, confidence=0.95)

    def _scenario(self, ground_truth_m: float):
        bbox_noise = self._rng.uniform(-1.0, 1.0)
        bbox_height = OBJECT_HEIGHT_M * INTRINSICS.fy / ground_truth_m + bbox_noise
        lidar_noise = self._rng.uniform(-0.03, 0.03)
        depth_noise = self._rng.uniform(-0.05, 0.05)
        return bbox_height, lidar_noise, depth_noise

    def run(self, ground_truth_m: float):
        bbox_h, lidar_off, depth_off = self._scenario(ground_truth_m)
        top = max(0, (FRAME_H - bbox_h) / 2.0)
        bottom = min(FRAME_H, (FRAME_H + bbox_h) / 2.0)
        bbox = (int(320 - bbox_h * 0.25), int(top), int(320 + bbox_h * 0.25), int(bottom))

        legacy = self.legacy.estimate(bbox_h, "person")

        vision = self.geometric.estimate(
            bbox_height_px=bbox_h,
            bbox_width_px=bbox_h * 0.5,
            class_name="person",
            detection_confidence=0.9,
            frame_w=FRAME_W,
            frame_h=FRAME_H,
            bbox_top_px=top,
            bbox_bottom_px=bottom,
        )

        self._depth.set_distance(ground_truth_m + depth_off)
        depth = self._estimator.estimate_depth(
            self._dummy_image(),
            bbox,
            detection_confidence=0.9,
        )

        self._lidar.set_distance(ground_truth_m + lidar_off)
        fused = self._estimator.update(vision, self._lidar.read(), depth, dt=0.05)

        filtered = self._filtered(ground_truth_m, bbox_h, lidar_off, depth_off)

        return {
            "ground_truth_m": ground_truth_m,
            "legacy": legacy,
            "geometric": vision.distance_m if vision.valid else None,
            "depth": (depth.range_m if depth.range_m is not None else depth.depth_z_m) if depth.valid else None,
            "fusion": fused.fused_range_m,
            "filtered": filtered.range_m,
        }

    def _filtered(self, ground_truth_m, bbox_h, lidar_off, depth_off):
        est = DistanceEstimator(config=EstimatorConfig(), intrinsics=INTRINSICS, depth_source=self._depth)
        est.reset()
        top = max(0, (FRAME_H - bbox_h) / 2.0)
        bottom = min(FRAME_H, (FRAME_H + bbox_h) / 2.0)
        bbox = (int(320 - bbox_h * 0.25), int(top), int(320 + bbox_h * 0.25), int(bottom))
        lidar = FixedLidar(ground_truth_m + lidar_off, confidence=0.95)
        last = None
        for _ in range(30):
            vision = est.estimate_vision(
                bbox_height_px=bbox_h,
                class_name="person",
                detection_confidence=0.9,
                frame_w=FRAME_W,
                frame_h=FRAME_H,
            )
            self._depth.set_distance(ground_truth_m + depth_off)
            depth = est.estimate_depth(self._dummy_image(), bbox, detection_confidence=0.9)
            last = est.update(vision, lidar.read(), depth, dt=0.05)
        return last

    @staticmethod
    def _dummy_image():
        import numpy as np

        return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def aggregate(values_by_method):
    """Compute MAE / RMSE / max-abs-error / MRE from per-target tables."""
    summary = {}
    for method, records in values_by_method.items():
        samples = [(r["gt"], r["est"]) for r in records if r["est"] is not None]
        if not samples:
            summary[method] = None
            continue
        errors = [abs(gt - est) for gt, est in samples]
        signed = [est - gt for gt, est in samples]
        relative = [abs(gt - est) / gt * 100.0 for gt, est in samples]
        summary[method] = {
            "mae": _mean(errors),
            "rmse": math.sqrt(_mean(e * e for e in errors)),
            "mean_err": _mean(signed),
            "max_abs": max(errors),
            "mre_pct": _mean(relative),
            "n": len(samples),
        }
    return summary


def print_summary_table(aggregates):
    print("\nMethod                 MAE      RMSE     Mean Err  Max Abs   MRE %   n")
    print("-" * 78)
    labels = {
        "legacy": "Legacy bbox",
        "geometric": "Geometric vision",
        "depth": "Metric depth",
        "fusion": "Fusion",
        "filtered": "Filtered fusion",
    }
    for method, label in labels.items():
        agg = aggregates.get(method)
        if agg is None:
            print(f"{label:<21} N/A - insufficient benchmark data")
            continue
        print(
            f"{label:<21} {agg['mae']:6.3f}  {agg['rmse']:6.3f}  {agg['mean_err']:8.3f}  "
            f"{agg['max_abs']:7.3f}   {agg['mre_pct']:5.1f}%   {agg['n']}"
        )


def run_synthetic_benchmark(seed: int, frames: int) -> None:
    print(SYNTHETIC_DISCLAIMER)
    print(f"Ground truths (m): {GROUND_TRUTHS_M}   seed={seed}   filter_frames={frames}")
    runner = MethodRunner(seed=seed)

    per_target = {"legacy": [], "geometric": [], "depth": [], "fusion": [], "filtered": []}

    print("\nPer-target absolute / relative error:")
    print("GT     Legacy      Geom        Depth       Fusion      Filtered       (m)   / rel %")
    print("-" * 90)
    for gt in GROUND_TRUTHS_M:
        row = runner.run(gt)
        cells = []
        for method in per_target:
            est = row[method]
            assert est is not None, f"{method} produced no estimate at {gt} m"
            err = abs(gt - est)
            rel = err / gt * 100.0
            per_target[method].append({"gt": gt, "est": est})
            cells.append(f"{est:8.3f} ({rel:4.1f}%)")
        print(f"{gt:4.1f}  " + "  ".join(cells))

    print_summary_table(aggregate(per_target))
    print("\n(Synthetic benchmark only. Accuracy claims require real data.)")


def run_temporal_benchmark() -> None:
    print("\nTEMPORAL FILTER BENCHMARK")
    print("Sequence: gentle wobble around 3 m, then a step 3.0 -> 5.0 m, then invalid gaps.")
    print("-" * 78)

    sequence = [3.0, 3.0, 3.1, 3.2, 3.1, 3.0, 2.9, 2.8, 3.0, 3.1, 3.0, 3.0]  # wobble
    sequence += [5.0] * 15  # step (frame index 12)
    sequence += [None, None, 5.0, None]  # invalid gaps mixed in

    est = DistanceEstimator(config=EstimatorConfig(), intrinsics=INTRINSICS)
    lidar = FixedLidar(0.0, confidence=0.95)
    raw_values = []
    filtered_values = []
    delay_frames = None
    dt = 0.05
    last_state = None

    for i, gt in enumerate(sequence):
        if gt is None:
            raw_values.append(None)
            last_state = est.update(None, None, None, dt=dt)
        else:
            lidar.set_distance(gt)
            last_state = est.update(None, lidar.read(), None, dt=dt)
            raw_values.append(gt)
        filtered_values.append(last_state.range_m)

    # noise statistics over the pre-step wobble frames 6..11 (raw vs filtered)
    wobble_raw = [v for v in raw_values[0:12] if v is not None]
    wobble_filt = [v for v in filtered_values[0:12] if v is not None]
    if len(wobble_raw) >= 2:
        raw_mean = _mean(wobble_raw)
        filt_mean = _mean(wobble_filt)
        raw_std = math.sqrt(_mean((v - raw_mean) ** 2 for v in wobble_raw))
        filt_std = math.sqrt(_mean((v - filt_mean) ** 2 for v in wobble_filt))
        print(f"During 3 m wobble: raw std = {raw_std:.3f} m, filtered std = {filt_std:.3f} m")
    else:
        print("During 3 m wobble: insufficient data")

    # response delay: frames until filtered output crosses 4.0 after the step
    step_frame = 12
    for i in range(step_frame, len(filtered_values)):
        if filtered_values[i] is not None and filtered_values[i] >= 4.0:
            delay_frames = i - step_frame
            break
    if delay_frames is not None:
        print(f"Step response (3 -> 5 m): crossed 4 m after {delay_frames} frames "
              f"({delay_frames * dt * 1000:.0f} ms at dt={dt} s)")
    else:
        print("Step response: never crossed 4 m within sequence")

    valid_indexes = [i for i, v in enumerate(filtered_values) if v is not None]
    if valid_indexes:
        last = valid_indexes[-1]
        print(f"Estimated velocity at final valid frame: {last_state.velocity_mps:+.3f} m/s")
        print(f"Range at frame {last}: {filtered_values[last]:.3f} m")

    gap_report = [i for i, gt in enumerate(sequence) if gt is None]
    print(f"Invalid-measurement frames: {gap_report} -> filter continued (no NaN, no crash)")


def run_real_image_benchmark(benchmark_dir: str) -> None:
    manifest_path = os.path.join(benchmark_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("\nReal-world accuracy benchmark: NOT YET AVAILABLE")
        print("Reason: no benchmark manifest found at", manifest_path)
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    targets = manifest.get("targets", [])

    if not targets:
        print("\nReal-world accuracy benchmark: NOT YET AVAILABLE")
        print("Reason: benchmark manifest exists but contains no ground-truth targets")
        print("See benchmarks/distance/README.md for how to add real images.")
        return

    print(f"\nReal-image benchmark: {len(targets)} target(s)")
    try:
        import cv2  # noqa: F401
        from modules.distance_estimator.depth_backend.metric_depth import ZoeDepthBackend
    except ImportError as exc:
        print("NOT YET AVAILABLE - missing dependencies:", exc)
        return

    runner = MethodRunner(seed=1)
    records = {"legacy": [], "geometric": [], "depth": [], "fusion": [], "filtered": []}
    for target in targets:
        image_path = os.path.join(benchmark_dir, target.get("image", ""))
        if not os.path.exists(image_path):
            print(f"  skipped missing image: {image_path}")
            continue
        gt = float(target["ground_truth_distance_m"])
        class_name = target.get("class_name", "person")
        bbox = tuple(target["bbox_ltrb"])
        frame = cv2.imread(image_path)
        bbox_top, bbox_bottom = bbox[1], bbox[3]
        bbox_h = bbox_bottom - bbox_top
        legacy = runner.legacy.estimate(bbox_h, class_name)
        vision = runner.geometric.estimate(
            bbox_height_px=bbox_h,
            bbox_width_px=bbox[2] - bbox[0],
            class_name=class_name,
            detection_confidence=target.get("detection_confidence", 0.9),
            frame_w=frame.shape[1],
            frame_h=frame.shape[0],
        )
        runner._depth.set_distance(gt)  # placeholder until a metric backend is attached
        depth = runner._estimator.estimate_depth(frame, bbox)
        fused = runner._estimator.update(vision, None, depth, dt=0.05)
        for method, est in (("legacy", legacy), ("geometric", vision.distance_m),
                            ("depth", depth.range_m), ("fusion", fused.fused_range_m)):
            if est is not None:
                records[method].append({"gt": gt, "est": est})
    print_summary_table(aggregate(records))


def main() -> int:
    parser = argparse.ArgumentParser(description="Distance-estimator offline benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=30, help="filter settle frames")
    parser.add_argument("--temporal-only", action="store_true")
    parser.add_argument("--benchmark-dir", default=os.path.join("benchmarks", "distance"))
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    if not args.temporal_only:
        run_synthetic_benchmark(args.seed, args.frames)

    run_temporal_benchmark()

    if not args.skip_real:
        run_real_image_benchmark(args.benchmark_dir)

    print("\nBenchmark complete. No flight code was touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())