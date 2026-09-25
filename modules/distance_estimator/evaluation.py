"""
Real-image evaluation engine for the distance estimator.

This module turns a validated :class:`~.dataset.DistanceDataset` (real images +
independently measured ground truth) into per-method predictions and the full
set of Task 3 metrics/reports.

Methods evaluated per valid sample:

    A. Legacy bbox-height   (modules.navigation.FollowController)
    B. Geometric vision     (VisionDistanceEstimator, calibrated pinhole)
    C. Metric monocular depth (ZoeDepthBackend, optional)
    D. Fusion               (DistanceEstimator pre-filter)
    E. Filtered fusion      (DistanceEstimator post-filter)

Rules honored here:

  * Methods with no usable data report ``NOT_AVAILABLE`` with a reason; they
    are never replaced by synthetic values.
  * Samples with a missing image or no bounding box are rejected and counted,
    not silently skipped.
  * No ground truth is ever derived from the image/estimator.
  * LiDAR is never simulated: fusion simply uses whichever sources produced a
    usable measurement (vision and/or metric depth).
  * Importing this module does NOT import cv2/torch/transformers at module
    level; everything optional is resolved at evaluation time.
"""
from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .analysis import (
    compute_metrics,
    confidence_vs_error,
    flag_outliers,
    group_by_condition,
    group_by_distance_bin,
)
from .calibration import CameraIntrinsics
from .config import EstimatorConfig
from .dataset import CameraRecord, DistanceDataset, GroundTruthSample
from .estimator import DistanceEstimator
from .models import DepthMeasurement, LidarMeasurement, VisionMeasurement
from .vision import VisionDistanceEstimator

METHODS: Dict[str, str] = {
    "legacy": "Legacy bbox",
    "geometric": "Geometric vision",
    "depth": "Metric depth",
    "fusion": "Fusion",
    "filtered": "Filtered fusion",
}

_CONDITION_FIELDS = ("class_name", "scene", "lighting", "pose", "measurement_method")


@dataclass
class EvaluationOptions:
    enable_legacy: bool = True
    enable_depth: bool = False
    depth_backend: Optional[Any] = None  # forward-compatible injection point
    make_plots: bool = True
    outlier_threshold_m: float = 1.0
    outlier_relative_threshold_pct: Optional[float] = 1.0
    min_samples_condition: int = 3


def read_image(path: str) -> np.ndarray:
    """Read an image as an HxWx3 uint8 array. Prefer cv2, fall back to PIL."""
    try:
        import cv2  # noqa: PLC0415

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            return np.asarray(image)
    except Exception:  # pragma: no cover - optional dependency
        pass
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as im:
            rgb = im.convert("RGB")
            return np.asarray(rgb)
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ValueError(f"cannot read image {path} (cv2 and PIL unavailable: {exc})") from exc
    raise ValueError(f"cannot read image {path}: unsupported format")


class _LegacyBboxMethod:
    """Wrapper around the untouched legacy controller distance method."""

    def __init__(self, intrinsics: Optional[CameraIntrinsics]) -> None:
        from modules.navigation import FollowController  # noqa: PLC0415

        self._ctrl = FollowController()
        self.ready = intrinsics is not None and intrinsics.fy > 0.0
        if self.ready:
            self._ctrl.set_focal_length(float(intrinsics.fy))

    def estimate(self, bbox_height_px: float, class_name: str) -> Optional[float]:
        if not self.ready:
            return None
        try:
            return float(self._ctrl.estimate_distance_from_size(bbox_height_px, class_name))
        except Exception:  # pragma: no cover - defensive
            return None


def _probe_depth_availability(options: EvaluationOptions) -> tuple:
    """Return (available: bool, reason: str, backend: Optional[DepthSource])."""
    if not options.enable_depth:
        return (
            False,
            "depth backend disabled (pass --enable-depth; requires torch + transformers)",
            None,
        )
    if options.depth_backend is not None:
        return True, "", options.depth_backend
    try:
        from .depth_backend.metric_depth import ZoeDepthBackend  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"depth backend could not be imported: {exc}", None

    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return (
            False,
            "torch/transformers not installed — metric-depth model cannot run",
            None,
        )
    backend = ZoeDepthBackend()
    try:
        backend.load()  # may download weights on first run
    except Exception as exc:
        return False, f"ZoeDepth model failed to load: {exc}", None
    return True, "", backend


class RealImageEvaluator:
    """Evaluate a validated real-world dataset against all estimator methods."""

    def __init__(
        self,
        dataset: DistanceDataset,
        options: Optional[EvaluationOptions] = None,
    ) -> None:
        self._dataset = dataset
        self._options = options or EvaluationOptions()
        self._intrinsics, intrinsic_error = self._build_intrinsics(dataset.camera)
        self._intrinsic_error = intrinsic_error
        self._vision = VisionDistanceEstimator(self._intrinsics, EstimatorConfig().vision)
        self._legacy = _LegacyBboxMethod(self._intrinsics) if self._options.enable_legacy else None
        if self._legacy is not None and not self._legacy.ready:
            self._legacy = None
        self._depth_available, self._depth_reason, self._backend = _probe_depth_availability(self._options)
        self._config = EstimatorConfig()

    # ------------------------------------------------------------------ intrinsics

    @staticmethod
    def _build_intrinsics(camera: CameraRecord):
        if camera.intrinsics is None:
            return None, "camera has no intrinsic calibration in the manifest"
        kwargs: Dict[str, object] = dict(camera.intrinsics)
        if camera.frame_w is not None and camera.frame_h is not None:
            kwargs["frame_w"] = camera.frame_w
            kwargs["frame_h"] = camera.frame_h
        try:
            intrinsics = CameraIntrinsics.from_dict(kwargs)
            if not intrinsics.is_valid():
                return None, "camera intrinsics failed validation"
            return intrinsics, ""
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"camera intrinsics invalid: {exc}"

    # ------------------------------------------------------------------ per sample

    def _evaluate_sample(self, sample: GroundTruthSample) -> Dict[str, dict]:
        """Return {method_key: record_or_None} for one sample."""
        out: Dict[str, dict] = {}

        if sample.bbox is None:
            return out  # caller rejects sample before this point

        image_path = self._resolve_image(sample)
        if image_path is None:
            return out
        try:
            frame = read_image(image_path)
        except ValueError:
            return out

        frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        left, top, right, bottom = sample.bbox.ltrb
        bbox_height = bottom - top
        if bbox_height <= 0.0:
            return out
        bbox_width = right - left
        detection = sample.detection_confidence if sample.detection_confidence is not None else 0.9

        base = {
            "sample_id": sample.id,
            "ground_truth_m": sample.ground_truth_distance_m,
            "class_name": sample.class_name,
            "scene": sample.scene,
            "lighting": sample.lighting,
            "pose": sample.pose,
            "measurement_method": sample.measurement_method,
            "notes": sample.notes,
        }

        # --- legacy bbox-height -------------------------------------------
        if self._legacy is not None:
            legacy = self._legacy.estimate(bbox_height, sample.class_name)
            if legacy is not None:
                out["legacy"] = self._finish(dict(base, method="legacy", prediction_m=legacy, confidence=None))

        # --- geometric vision ---------------------------------------------
        if self._intrinsics is not None:
            vision = self._vision.estimate(
                bbox_height_px=bbox_height,
                bbox_width_px=bbox_width,
                class_name=sample.class_name,
                detection_confidence=detection,
                frame_w=frame_w,
                frame_h=frame_h,
                bbox_top_px=top,
                bbox_bottom_px=bottom,
            )
            if vision.valid and vision.distance_m is not None:
                out["geometric"] = self._finish(
                    dict(base, method="geometric", prediction_m=vision.distance_m, confidence=vision.confidence)
                )

        # --- metric depth --------------------------------------------------
        depth_measurement: Optional[DepthMeasurement] = None
        if self._depth_available:
            estimator = self._fresh_estimator()
            depth_measurement = estimator.estimate_depth(frame, (left, top, right, bottom), detection)
            if depth_measurement.valid and depth_measurement.confidence > 0.0:
                value = depth_measurement.range_m if depth_measurement.range_m is not None else depth_measurement.depth_z_m
                if value is not None and math.isfinite(value):
                    out["depth"] = self._finish(
                        dict(
                            base,
                            method="depth",
                            prediction_m=value,
                            confidence=depth_measurement.confidence,
                        )
                    )

        # --- fusion + filtered fusion --------------------------------------
        vision_measurement: Optional[VisionMeasurement] = None
        if "geometric" in out:
            vision_measurement = self._vision.estimate(
                bbox_height_px=bbox_height,
                bbox_width_px=bbox_width,
                class_name=sample.class_name,
                detection_confidence=detection,
                frame_w=frame_w,
                frame_h=frame_h,
                bbox_top_px=top,
                bbox_bottom_px=bottom,
            )
        if vision_measurement is None and depth_measurement is None:
            return out  # nothing usable to fuse

        estimator = self._fresh_estimator()
        state = estimator.update(vision_measurement, None, depth_measurement, dt=1.0 / 30.0)
        if state.fused_range_m is not None and state.valid:
            out["fusion"] = self._finish(
                dict(base, method="fusion", prediction_m=state.fused_range_m, confidence=state.confidence)
            )
        if state.range_m is not None and state.valid:
            out["filtered"] = self._finish(
                dict(base, method="filtered", prediction_m=state.range_m, confidence=state.confidence)
            )
        return out

    def _resolve_image(self, sample: GroundTruthSample) -> Optional[str]:
        path = sample.image_path
        if os.path.exists(path):
            return path
        return None

    def _fresh_estimator(self) -> DistanceEstimator:
        return DistanceEstimator(config=self._config, intrinsics=self._intrinsics, depth_source=self._backend)

    @staticmethod
    def _finish(record: dict) -> dict:
        gt = float(record["ground_truth_m"])
        pred = float(record["prediction_m"])
        abs_err = abs(pred - gt)
        record["absolute_error_m"] = abs_err
        record["signed_error_m"] = pred - gt
        record["relative_error"] = abs_err / gt
        return record

    # ------------------------------------------------------------------ run

    def _method_availability(self) -> Dict[str, Dict[str, object]]:
        availability: Dict[str, Dict[str, object]] = {}
        for key in METHODS:
            availability[key] = {"available": True, "reason": ""}
        if self._legacy is None:
            availability["legacy"] = {
                "available": False,
                "reason": self._intrinsic_error or "legacy method disabled",
            }
        if self._intrinsics is None:
            availability["geometric"]["available"] = False
            availability["geometric"]["reason"] = self._intrinsic_error or "no camera calibration"
            availability["fusion"]["available"] = False
            availability["fusion"]["reason"] = "no calibrated source available"
            availability["filtered"]["available"] = False
            availability["filtered"]["reason"] = "no calibrated source available"
        if not self._depth_available:
            availability["depth"]["available"] = False
            availability["depth"]["reason"] = self._depth_reason or "depth backend unavailable"
        return availability

    def run(self) -> Dict[str, object]:
        availability = self._method_availability()
        methods: Dict[str, Dict[str, object]] = {}
        for key in METHODS:
            methods[key] = {
                "available": availability[key]["available"],
                "reason": availability[key]["reason"],
                "records": [],
                "metrics": {"n": 0},
            }

        rejected: List[Dict[str, str]] = []
        ground_truths: List[float] = []
        usable = 0

        for sample in self._dataset.samples:
            ground_truths.append(sample.ground_truth_distance_m)

            if sample.bbox is None:
                rejected.append({"id": sample.id, "reason": "no bounding box"})
                continue
            if not os.path.exists(sample.image_path):
                rejected.append({"id": sample.id, "reason": "missing image"})
                continue
            try:
                frame = read_image(sample.image_path)
            except ValueError:
                rejected.append({"id": sample.id, "reason": "image load failed"})
                continue
            if frame is None or frame.size == 0:
                rejected.append({"id": sample.id, "reason": "image load failed"})
                continue

            usable += 1
            per_sample = self._evaluate_sample(sample)
            for key, record in per_sample.items():
                methods[key]["records"].append(record)

        for key in methods:
            records = methods[key]["records"]  # type: ignore[assignment]
            methods[key]["metrics"] = compute_metrics(records).to_dict()

        bin_by_method: Dict[str, object] = {}
        for key in methods:
            records = methods[key]["records"]
            binned: Dict[str, object] = {}
            for label, group in group_by_distance_bin(records).items():
                binned[label] = compute_metrics(group).to_dict()
            bin_by_method[key] = binned

        reference_key = self._pick_reference_method(
            {k: bool(m["records"]) for k, m in methods.items()}
        )

        conditions: Dict[str, object] = {}
        if reference_key is not None:
            records = methods[reference_key]["records"]
            for field_name in _CONDITION_FIELDS:
                buckets = group_by_condition(records, field_name, min_samples=1)
                conditions[field_name] = {
                    value: summary.to_dict()
                    for value, summary in buckets.items()
                }

        conf_analysis: Dict[str, object] = {}
        self._collect_confidence(reference_key, methods, conf_analysis)

        outliers: List[dict] = []
        for key in methods:
            flag_records, rule = flag_outliers(
                methods[key]["records"],
                threshold_m=self._options.outlier_threshold_m,
                relative_threshold_pct=self._options.outlier_relative_threshold_pct,
            )
            methods[key]["outliers"] = flag_records  # type: ignore[assignment]
            methods[key]["outlier_rule"] = rule  # type: ignore[assignment]
            outliers.extend(flag_records)

        classes = sorted({s.class_name for s in self._dataset.samples})
        measurement_methods = sorted({s.measurement_method for s in self._dataset.samples})

        payload = {
            "manifest": self._dataset.manifest_path,
            "version": self._dataset.version,
            "camera": {
                "name": self._dataset.camera.name,
                "frame_w": self._dataset.camera.frame_w,
                "frame_h": self._dataset.camera.frame_h,
                "intrinsics_source": self._dataset.camera.intrinsics_source,
                "intrinsics_present": self._dataset.camera.intrinsics is not None,
                "intrinsics": self._dataset.camera.intrinsics,
                "intrinsics_notes": self._dataset.camera.intrinsics_notes,
            },
            "options": {
                "enable_legacy": self._options.enable_legacy,
                "enable_depth": self._options.enable_depth,
                "outlier_threshold_m": self._options.outlier_threshold_m,
                "outlier_relative_threshold_pct": self._options.outlier_relative_threshold_pct,
            },
            "dataset": {
                "total_samples": len(self._dataset.samples),
                "usable_samples": usable,
                "rejected_samples": len(rejected),
                "rejections": rejected,
                "classes": classes,
                "measurement_methods": measurement_methods,
                "distance_min_m": round(min(ground_truths), 3) if ground_truths else None,
                "distance_max_m": round(max(ground_truths), 3) if ground_truths else None,
            },
            "methods": methods,
            "distance_bins": bin_by_method,
            "condition_analysis": conditions,
            "reference_method": reference_key,
            "confidence_analysis": conf_analysis,
            "outliers_total": len(outliers),
            "notes": [
                "Real-image evaluation. Synthetic data from Task 2 is never mixed in.",
                "LiDAR validation data: unavailable (no physical LiDAR returns in this dataset); fusion uses only usable vision/depth sources.",
            ],
        }
        return payload

    def _pick_reference_method(self, has_records: Dict[str, bool]) -> Optional[str]:
        for key in ("fusion", "geometric", "depth", "filtered", "legacy"):
            if has_records.get(key):
                return key
        return None

    def _collect_confidence(self, reference_key, methods, out) -> None:
        if reference_key is None:
            out["analyzed_method"] = None
            out["result"] = {"n": 0, "pearson_r": None, "note": "no predictions to analyze"}
            return
        records = methods[reference_key]["records"]
        out["analyzed_method"] = reference_key
        out["result"] = confidence_vs_error(records)


def make_plots(records_by_method: Dict[str, List[dict]], results_dir: str) -> List[str]:
    """Optional plots for real data only. Returns the list of saved files.

    Skips silently (returns []) when no real predictions or matplotlib are
    unavailable; no synthetic data is ever plotted as real-world results.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plots: List[str] = []
    records = records_by_method.get("fusion") or records_by_method.get("geometric") or records_by_method.get("depth")
    if not records:
        return []

    gts = [float(r["ground_truth_m"]) for r in records]
    preds = [float(r["prediction_m"]) for r in records]
    errors = [abs(p - g) for g, p in zip(gts, preds)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    ax = axes[0, 0]
    ax.scatter(gts, preds, alpha=0.7)
    bound = max(max(gts), max(preds), 1.0) * 1.1
    ax.plot([0, bound], [0, bound], "r--", label="ideal")
    ax.set_xlabel("ground truth (m)")
    ax.set_ylabel("prediction (m)")
    ax.set_title("Predicted vs ground truth (real data)")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(gts, errors, alpha=0.7)
    ax.set_xlabel("ground truth (m)")
    ax.set_ylabel("absolute error (m)")
    ax.set_title("Absolute error vs distance")

    ax = axes[1, 0]
    ax.hist(errors, bins=10)
    ax.set_xlabel("absolute error (m)")
    ax.set_ylabel("count")
    ax.set_title("Error distribution")

    ax = axes[1, 1]
    signed = [p - g for g, p in zip(gts, preds)]
    ax.hist(signed, bins=10)
    ax.set_xlabel("signed error (m)")
    ax.set_ylabel("count")
    ax.set_title("Signed error (bias) distribution")

    fig.tight_layout()
    path = os.path.join(results_dir, "real_world_plots.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    plots.append(path)
    return plots