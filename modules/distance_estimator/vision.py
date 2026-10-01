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


# ---------------------------------------------------------------------------
# Optical-axis depth  ->  horizontal ground distance
# ---------------------------------------------------------------------------
# These helpers are pure geometry and are NOT used by ``estimate()``: the live
# follow path is unchanged. They are exposed for an explicit, opt-in horizontal
# measurement path (see ``VisionDistanceEstimator.estimate_horizontal``).
#
# They assume the camera optical axis is LEVEL (pitch == 0), the target stands
# upright on a flat ground plane, and roll == 0. Pitch is deliberately not
# modelled yet (no ATTITUDE telemetry); the resulting bias is documented in the
# README/report notes.

# Guards against floating-point noise pushing a physically-zero radicand
# slightly negative. Anything within this relative epsilon of zero is clamped
# to 0.0 rather than rejected.
_RADICAND_EPSILON_M2 = 1e-9


def axis_depth_to_slant_range(
    axis_depth_m: Optional[float],
    u_px: Optional[float],
    v_px: Optional[float],
    intrinsics: Optional[CameraIntrinsics],
) -> Optional[float]:
    """Convert optical-axis (camera-forward Z) depth into camera-to-target slant range.

    A pinhole projection gives ``Z`` (depth along the optical axis). The
    Euclidean camera-to-target range follows from the normalized image
    coordinates of the target pixel::

        x_n = (u_px - cx) / fx
        y_n = (v_px - cy) / fy
        r   = sqrt(1 + x_n**2 + y_n**2)
        slant = Z * r

    This is the inverse of the same transform used by the metric-depth path
    (``depth._z_to_range_m``), which converts ``Z`` -> range.

    Returns ``None`` when any input is missing, non-finite, non-positive, or the
    intrinsics are unusable. It never manufactures a distance.
    """
    if axis_depth_m is None or u_px is None or v_px is None or intrinsics is None:
        return None
    if not isinstance(axis_depth_m, (int, float)) or not math.isfinite(axis_depth_m):
        return None
    if axis_depth_m <= 0.0:
        return None
    if not math.isfinite(u_px) or not math.isfinite(v_px):
        return None
    if not intrinsics.is_valid():
        return None

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    if fx <= 0.0 or fy <= 0.0:
        return None

    x_n = (float(u_px) - float(intrinsics.cx)) / fx
    y_n = (float(v_px) - float(intrinsics.cy)) / fy

    scale = math.sqrt(1.0 + x_n * x_n + y_n * y_n)
    slant = float(axis_depth_m) * scale
    if not math.isfinite(slant) or slant <= 0.0:
        return None
    return slant


def slant_to_horizontal_distance(
    slant_range_m: Optional[float],
    altitude_m: Optional[float],
) -> Optional[float]:
    """Convert camera-to-target slant range plus camera altitude into horizontal range.

    With the optical axis level, the camera sits ``altitude_m`` above the
    target's ground plane, so the right triangle gives::

        slant**2 = horizontal**2 + altitude**2
        horizontal = sqrt(slant**2 - altitude**2)

    ``slant_range_m`` must be a true slant range (not optical-axis depth) —
    pass the output of :func:`axis_depth_to_slant_range`.

    Returns ``None`` for missing, non-finite or non-positive slant range, for
    missing/negative/non-finite altitude, and when the geometry is impossible
    (``slant <= altitude``, i.e. the target is not actually out on the ground in
    front of the vehicle). A radicand that is only negative through
    floating-point round-off is clamped to 0.0 instead of raising.
    """
    if slant_range_m is None or altitude_m is None:
        return None
    if not isinstance(slant_range_m, (int, float)) or not math.isfinite(slant_range_m):
        return None
    if not isinstance(altitude_m, (int, float)) or not math.isfinite(altitude_m):
        return None

    slant = float(slant_range_m)
    altitude = float(altitude_m)
    if slant <= 0.0:
        return None
    if altitude < 0.0:
        return None
    if slant <= altitude:
        # Impossible geometry: the target cannot be at or below the vehicle's
        # own altitude and still out in front of it on the ground plane.
        return None

    radicand = slant * slant - altitude * altitude
    if radicand < 0.0:
        if radicand > -_RADICAND_EPSILON_M2:
            radicand = 0.0
        else:
            return None

    horizontal = math.sqrt(radicand)
    if not math.isfinite(horizontal):
        return None
    return horizontal


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

    # ------------------------------------------- horizontal ground distance ----

    def estimate_horizontal(
        self,
        bbox_height_px: Optional[float],
        altitude_m: Optional[float],
        u_px: Optional[float] = None,
        v_px: Optional[float] = None,
        bbox_width_px: Optional[float] = None,
        class_name: str = "",
        detection_confidence: Optional[float] = 1.0,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
        bbox_left_px: Optional[float] = None,
        bbox_top_px: Optional[float] = None,
        bbox_bottom_px: Optional[float] = None,
        previous_distance_m: Optional[float] = None,
    ) -> VisionMeasurement:
        """Opt-in ground-plane horizontal distance, alongside the optical-axis depth.

        This is a NEW, SEPARATE path. ``estimate()`` is untouched and remains
        the authoritative measurement for the live follow loop.

        It reuses the identical pinhole optical-axis depth from ``estimate()``,
        then converts it to horizontal ground distance::

            axis_depth (Z)  --axis_depth_to_slant_range-->  slant
            slant, altitude --slant_to_horizontal_distance-->  horizontal

        ``altitude_m`` is the vehicle's height above the target's ground plane
        and is supplied by the caller (the application already has this from the
        existing SITL ``GLOBAL_POSITION_INT`` telemetry cache); this layer never
        creates a second altitude source.

        When ``u_px``/``v_px`` are omitted the bbox bottom-center is used, i.e.
        the target's ground contact point, derived from
        ``bbox_left_px + bbox_width_px / 2`` and ``bbox_bottom_px``.

        The returned measurement is invalid (``valid=False``,
        ``horizontal_distance_m=None``) whenever the geometry cannot be trusted.
        Nothing is invented and no fallback distance is manufactured. Confidence
        is inherited unchanged from the underlying optical-axis measurement.
        """
        base = self.estimate(
            bbox_height_px=bbox_height_px,
            bbox_width_px=bbox_width_px,
            class_name=class_name,
            detection_confidence=detection_confidence,
            frame_w=frame_w,
            frame_h=frame_h,
            bbox_top_px=bbox_top_px,
            bbox_bottom_px=bbox_bottom_px,
            previous_distance_m=previous_distance_m,
        )
        if not base.valid or base.distance_m is None:
            return VisionMeasurement.invalid(class_name=class_name)

        if u_px is None or v_px is None:
            if bbox_left_px is None or bbox_bottom_px is None:
                return VisionMeasurement.invalid(class_name=class_name)
            # Bottom-center pixel: the target's ground contact point.
            u_px = float(bbox_left_px) + (float(bbox_width_px) / 2.0 if bbox_width_px else 0.0)
            v_px = float(bbox_bottom_px)
        elif not math.isfinite(u_px) or not math.isfinite(v_px):
            return VisionMeasurement.invalid(class_name=class_name)

        slant = axis_depth_to_slant_range(base.distance_m, u_px, v_px, self._intrinsics)
        if slant is None:
            return VisionMeasurement.invalid(class_name=class_name)

        horizontal = slant_to_horizontal_distance(slant, altitude_m)
        if horizontal is None:
            return VisionMeasurement.invalid(class_name=class_name)

        return VisionMeasurement(
            distance_m=base.distance_m,
            confidence=base.confidence,
            bbox_height_px=base.bbox_height_px,
            bbox_width_px=base.bbox_width_px,
            class_name=class_name,
            truncated=base.truncated,
            valid=True,
            timestamp=base.timestamp,
            horizontal_distance_m=horizontal,
            slant_range_m=slant,
            altitude_m=float(altitude_m) if altitude_m is not None else None,
        )


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


def estimate_detection_horizontal(estimator, detection, frame_w: int, frame_h: int,
                                  altitude_m: Optional[float]) -> VisionMeasurement:
    """Measure ground-plane horizontal distance for one detection.

    Mirrors :func:`estimate_detection` but uses the opt-in
    :meth:`VisionDistanceEstimator.estimate_horizontal` path. ``altitude_m`` is
    supplied by the caller from the existing telemetry source; this function
    never creates its own altitude.
    """
    class_name = getattr(detection, "class_name", "")
    if estimator is None or not getattr(estimator, "geometry_available", False):
        return VisionMeasurement.invalid(class_name=class_name)
    if altitude_m is None:
        return VisionMeasurement.invalid(class_name=class_name)
    left = getattr(detection, "Left", None)
    width = getattr(detection, "width", None)
    bottom = getattr(detection, "Bottom", None)
    u_px = (float(left) + float(width) / 2.0) if left is not None and width is not None else None
    v_px = float(bottom) if bottom is not None else None
    return estimator.estimate_horizontal(
        bbox_height_px=getattr(detection, "height", None),
        altitude_m=altitude_m,
        u_px=u_px,
        v_px=v_px,
        bbox_width_px=width,
        class_name=class_name,
        detection_confidence=getattr(detection, "confidence", None),
        frame_w=frame_w,
        frame_h=frame_h,
        bbox_left_px=left,
        bbox_top_px=getattr(detection, "Top", None),
        bbox_bottom_px=bottom,
    )


def annotate_horizontal(detection, measurement, altitude_m: Optional[float] = None) -> object:
    """Attach ground-plane horizontal distance fields to a detection.

    Observational only: this never touches ``distance_m``/``distance_source``,
    so the authoritative optical-axis measurement and the follow loop are
    unaffected.
    """
    if detection is None:
        return None
    raw = getattr(measurement, "horizontal_distance_m", None)
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        value = None
    valid = (
        measurement is not None
        and bool(getattr(measurement, "valid", False))
        and value is not None
        and math.isfinite(value)
        and value > 0.0
    )
    if not valid:
        detection.horizontal_distance_m = None
        detection.horizontal_distance_valid = False
        detection.horizontal_distance_source = None
        detection.slant_range_m = None
        detection.horizontal_altitude_m = None
        return detection
    detection.horizontal_distance_m = value
    detection.horizontal_distance_valid = True
    detection.horizontal_distance_source = "vision-horizontal"
    slant = getattr(measurement, "slant_range_m", None)
    try:
        slant = float(slant) if slant is not None else None
    except (TypeError, ValueError):
        slant = None
    detection.slant_range_m = slant if slant is not None and math.isfinite(slant) else None
    alt = getattr(measurement, "altitude_m", altitude_m)
    try:
        alt = float(alt) if alt is not None else None
    except (TypeError, ValueError):
        alt = None
    detection.horizontal_altitude_m = alt if alt is not None and math.isfinite(alt) else None
    return detection


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


def annotate_detections(estimator, detections, frame_w: int, frame_h: int,
                        altitude_m: Optional[float] = None,
                        horizontal_classes=None) -> object:
    for detection in detections or ():
        annotate_detection(detection, estimate_detection(estimator, detection, frame_w, frame_h))
        annotate_horizontal(detection, None)
        if altitude_m is None or not getattr(detection, "is_selected", False):
            continue
        if horizontal_classes is not None and getattr(detection, "class_name", None) not in horizontal_classes:
            continue
        measurement = estimate_detection_horizontal(estimator, detection, frame_w, frame_h, altitude_m)
        annotate_horizontal(detection, measurement, altitude_m)
    return detections
