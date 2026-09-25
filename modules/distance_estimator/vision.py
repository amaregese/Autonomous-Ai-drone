"""
Geometric (monocular) vision distance measurement with explicit validation and
a deterministic measurement-quality confidence.

Model:  distance = real_object_height_m * fy_px / object_height_px

Unlike the old application code, an invalid input never silently falls back to
an arbitrary "3.0 m": it yields an invalid, zero-confidence measurement. The
accumulated confidence is a 0..1 *measurement quality* score derived only from
observable inputs, not a claimed positional accuracy.
"""
from __future__ import annotations

import math
import time
from typing import Optional

from .calibration import CameraIntrinsics
from .config import VisionConfig
from .models import VisionMeasurement


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class VisionDistanceEstimator:
    """Validated pinhole-model ranging from a bounding-box height."""

    def __init__(self, intrinsics: Optional[CameraIntrinsics], config: VisionConfig) -> None:
        self._intrinsics = intrinsics
        self._config = config

    @property
    def geometry_available(self) -> bool:
        return self._intrinsics is not None and self._intrinsics.is_valid()

    @property
    def intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._intrinsics

    def estimate(
        self,
        bbox_height_px: Optional[float],
        bbox_width_px: Optional[float] = None,
        class_name: str = "",
        detection_confidence: Optional[float] = 1.0,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
        bbox_top_px: Optional[float] = None,
        bbox_bottom_px: Optional[float] = None,
        previous_distance_m: Optional[float] = None,
    ) -> VisionMeasurement:
        ts = time.time()

        if not self.geometry_available:
            return VisionMeasurement.invalid(class_name=class_name, timestamp=ts)

        if bbox_height_px is None or not math.isfinite(bbox_height_px) or bbox_height_px <= 0.0:
            return VisionMeasurement.invalid(class_name=class_name, timestamp=ts)

        if (
            bbox_width_px is not None
            and (not math.isfinite(bbox_width_px) or bbox_width_px <= 0.0)
        ):
            return VisionMeasurement.invalid(class_name=class_name, timestamp=ts)

        if detection_confidence is None or not math.isfinite(detection_confidence) or detection_confidence <= 0.0:
            return VisionMeasurement.invalid(class_name=class_name, timestamp=ts)

        obj_h = self._resolve_object_height(class_name)
        if obj_h is None or obj_h <= 0.0:
            return VisionMeasurement.invalid(class_name=class_name, timestamp=ts)

        distance_m = (obj_h * float(self._intrinsics.fy)) / float(bbox_height_px)

        cfg = self._config

        if not math.isfinite(distance_m) or distance_m <= 0.0:
            return VisionMeasurement.invalid(class_name=class_name, timestamp=ts)
        if distance_m < cfg.minimum_distance_m or distance_m > cfg.maximum_distance_m:
            return VisionMeasurement.invalid(class_name=class_name, timestamp=ts)

        size_score = self._size_score(bbox_height_px)
        detection_score = self._detection_score(detection_confidence)
        class_score = 1.0 if class_name in cfg.known_class_heights_m else 0.5
        edge_score, truncated = self._edge_score(frame_w, frame_h, bbox_top_px, bbox_bottom_px)

        confidence = (
            cfg.bbox_size_weight * size_score
            + cfg.detection_confidence_weight * detection_score
            + cfg.class_availability_weight * class_score
            + cfg.edge_proximity_weight * edge_score
        )

        if previous_distance_m is not None and math.isfinite(previous_distance_m) and previous_distance_m > 0.0:
            consistency = self._consistency_score(distance_m, previous_distance_m)
            confidence *= consistency

        confidence = _clamp01(confidence)

        return VisionMeasurement(
            distance_m=distance_m,
            confidence=confidence,
            bbox_height_px=bbox_height_px,
            bbox_width_px=bbox_width_px,
            class_name=class_name,
            truncated=truncated,
            valid=True,
            timestamp=ts,
        )

    def _resolve_object_height(self, class_name: str) -> Optional[float]:
        known = self._config.known_class_heights_m.get(class_name)
        if known is not None:
            return known
        default = self._config.default_object_height_m
        if default is None or default <= 0.0:
            return None
        return default

    def _size_score(self, bbox_height_px: float) -> float:
        cfg_min = self._config.minimum_bbox_height_px
        ref = max(self._config.reference_bbox_height_px, cfg_min + 1e-6)
        return _clamp01((bbox_height_px - cfg_min) / (ref - cfg_min))

    def _detection_score(self, detection_confidence: float) -> float:
        minimum = self._config.minimum_detection_confidence
        if detection_confidence <= minimum:
            return 0.0
        return _clamp01((detection_confidence - minimum) / (1.0 - minimum))

    @staticmethod
    def _edge_score(
        frame_w: Optional[int],
        frame_h: Optional[int],
        bbox_top_px: Optional[float],
        bbox_bottom_px: Optional[float],
    ):
        """0.0 (touching an edge / fully truncated) .. 1.0 (comfortably inside).

        Falls back to 1.0 (no truncation information) when the frame geometry or
        bbox position is not supplied.
        """
        if frame_w is None or frame_h is None or frame_w <= 0 or frame_h <= 0:
            return 1.0, None
        if bbox_top_px is None or bbox_bottom_px is None:
            return 1.0, None
        if not (math.isfinite(bbox_top_px) and math.isfinite(bbox_bottom_px)):
            return 1.0, None

        margin_top = bbox_top_px / frame_h
        margin_bottom = (frame_h - bbox_bottom_px) / frame_h
        margin = min(margin_top, margin_bottom)
        truncated = margin <= 1e-6 or bbox_top_px < 0.0 or bbox_bottom_px > frame_h
        if truncated:
            return 0.2, True
        return _clamp01(margin * 5.0), False

    @staticmethod
    def _consistency_score(distance_m: float, previous_distance_m: float) -> float:
        denominator = max(abs(previous_distance_m), 0.1)
        ratio = abs(distance_m - previous_distance_m) / denominator
        return 1.0 / (1.0 + ratio)


def estimate_detection(estimator, detection, frame_w: int, frame_h: int) -> VisionMeasurement:
    """Measure one detection with the same geometric estimator used by the diagnostic tool."""
    class_name = getattr(detection, "class_name", "")
    if estimator is None or not getattr(estimator, "geometry_available", False):
        return VisionMeasurement.invalid(class_name=class_name)
    return estimator.estimate(
        bbox_height_px=getattr(detection, "height", None),
        bbox_width_px=getattr(detection, "width", None),
        class_name=class_name,
        detection_confidence=getattr(detection, "confidence", None),
        frame_w=frame_w,
        frame_h=frame_h,
        bbox_top_px=getattr(detection, "Top", None),
        bbox_bottom_px=getattr(detection, "Bottom", None),
    )


def annotate_detection(detection, measurement, source: str = "vision") -> object:
    if detection is None:
        return None
    raw_distance = getattr(measurement, "distance_m", None)
    try:
        distance = float(raw_distance) if raw_distance is not None else None
    except (TypeError, ValueError):
        distance = None
    valid = (
        measurement is not None
        and bool(getattr(measurement, "valid", False))
        and distance is not None
        and math.isfinite(distance)
        and distance > 0.0
    )
    if not valid:
        distance = None
    try:
        confidence = float(getattr(measurement, "confidence", 0.0)) if valid else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    detection.distance_m = distance
    detection.distance_valid = valid
    detection.distance_source = source if valid else "none"
    detection.distance_confidence = max(0.0, min(1.0, confidence))
    return detection


def annotate_detections(estimator, detections, frame_w: int, frame_h: int) -> object:
    for detection in detections or ():
        annotate_detection(detection, estimate_detection(estimator, detection, frame_w, frame_h))
    return detections
