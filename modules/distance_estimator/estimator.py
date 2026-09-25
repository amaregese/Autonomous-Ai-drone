"""
Top-level isolated distance estimator.

``DistanceEstimator`` combines validated vision (monocular pinhole geometry),
metric monocular depth and LiDAR measurements into a fused range/velocity
estimate using a confidence-aware selection/fusion layer and a 1-D Kalman
filter.

It is deliberately decoupled from autonomous flight:
  * it sends no MAVLink commands,
  * it writes no drone velocity / altitude / mode,
  * it never calls the legacy ``modules.navigation.FollowController``.
"""
from __future__ import annotations

import math
import time
from typing import Any, Optional, Tuple

from .calibration import CameraIntrinsics
from .config import EstimatorConfig
from .depth import DepthBackendError, DepthSource, compute_depth_confidence, extract_depth_roi
from .filter import DistanceVelocityFilter
from .models import DepthMeasurement, LidarMeasurement, Source, TargetState, VisionMeasurement
from .vision import VisionDistanceEstimator


class DistanceEstimator:
    def __init__(
        self,
        config: Optional[EstimatorConfig] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        depth_source: Optional[DepthSource] = None,
    ) -> None:
        self._config = config or EstimatorConfig()
        self._intrinsics = intrinsics
        self._depth_source = depth_source
        self._vision = VisionDistanceEstimator(intrinsics, self._config.vision)
        self._filter = DistanceVelocityFilter(self._config.filter)
        self._last_call_t = None
        self.reset()

    def reset(self) -> None:
        """Return to the pristine initial state."""
        self._filter.reset()
        self._last_call_t = None
        self._last_fused_range_m: Optional[float] = None
        self._last_fused_confidence = 0.0
        self._last_disagreement_m = 0.0

    @property
    def vision_geometry_available(self) -> bool:
        """True only when valid camera intrinsics are installed."""
        return self._vision.geometry_available

    @property
    def depth_available(self) -> bool:
        """True when a metric-depth source is attached (may still be unloaded)."""
        return self._depth_source is not None

    @property
    def last_fused_range_m(self) -> Optional[float]:
        return self._last_fused_range_m

    def set_intrinsics(self, intrinsics: Optional[CameraIntrinsics]) -> None:
        """Install/replace camera calibration at runtime (e.g. after auto-cal)."""
        self._intrinsics = intrinsics
        self._vision = VisionDistanceEstimator(intrinsics, self._config.vision)

    def estimate_vision(
        self,
        bbox_height_px: Optional[float],
        bbox_width_px: Optional[float] = None,
        class_name: str = "",
        detection_confidence: Optional[float] = 1.0,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
        bbox_top_px: Optional[float] = None,
        bbox_bottom_px: Optional[float] = None,
    ) -> VisionMeasurement:
        """Convenience wrapper producing a vision measurement for ``update``."""
        previous = self._filter.distance if self._filter.initialized else None
        return self._vision.estimate(
            bbox_height_px=bbox_height_px,
            bbox_width_px=bbox_width_px,
            class_name=class_name,
            detection_confidence=detection_confidence,
            frame_w=frame_w,
            frame_h=frame_h,
            bbox_top_px=bbox_top_px,
            bbox_bottom_px=bbox_bottom_px,
            previous_distance_m=previous,
        )

    def estimate_depth(
        self,
        image: Any,
        bbox_ltrb: Tuple[float, float, float, float],
        detection_confidence: float = 1.0,
    ) -> DepthMeasurement:
        """Run metric depth at the target ROI and return a depth measurement.

        ``bbox_ltrb`` is ``(left, top, right, bottom)`` in pixels. The depth is
        read from an interior ROI (trimming borders), summarized with the
        median, and gated on coverage / range validity. Empty or low-coverage
        ROIs produce an invalid (zero-confidence) measurement.
        """
        if self._depth_source is None:
            return DepthMeasurement.invalid()

        try:
            frame = self._depth_source.estimate(image)
        except DepthBackendError as exc:
            return DepthMeasurement.invalid(source=getattr(self._depth_source, "name", "monocular"))

        left, top, right, bottom = bbox_ltrb
        cfg = self._config.depth

        stats = extract_depth_roi(
            frame,
            left,
            top,
            right,
            bottom,
            margin_fraction=cfg.roi_margin_fraction,
            min_valid_fraction=cfg.min_valid_fraction,
            intrinsics=self._intrinsics,
        )

        if not stats.valid or stats.z_median_m is None or not math.isfinite(stats.z_median_m):
            return DepthMeasurement.invalid(source=frame.source)

        if not (cfg.minimum_distance_m <= stats.z_median_m <= cfg.maximum_distance_m):
            return DepthMeasurement.invalid(source=frame.source)

        confidence = compute_depth_confidence(stats, detection_confidence, cfg)

        return DepthMeasurement(
            depth_z_m=stats.z_median_m,
            range_m=stats.range_median_m,
            confidence=confidence,
            valid=True,
            source=frame.source,
            n_valid_pixels=stats.n_valid_pixels,
            roi_size_px=stats.roi_size_px,
            depth_stdev_m=stats.stdev_m,
        )

    def update(
        self,
        vision_measurement: Optional[VisionMeasurement] = None,
        lidar_measurement: Optional[LidarMeasurement] = None,
        depth_measurement: Optional[DepthMeasurement] = None,
        dt: Optional[float] = None,
    ) -> TargetState:
        """Process the latest measurements and return the estimated state."""
        now = time.time()

        vision = self._sanitize_vision(vision_measurement)
        lidar = self._sanitize_lidar(lidar_measurement)
        depth = self._sanitize_depth(depth_measurement)

        dt = self._resolve_dt(dt, now)

        vision_usable = vision.valid and vision.confidence >= self._config.vision.vision_confidence_threshold
        lidar_usable = lidar.valid and lidar.confidence >= self._config.lidar.lidar_confidence_threshold
        depth_usable = depth.valid and depth.confidence >= self._config.depth.depth_confidence_threshold

        # --- apply fusion / selection -------------------------------------
        fused_range, fused_confidence, fused_source, disagreement_m, note = self._fuse(
            vision if vision_usable else None,
            lidar if lidar_usable else None,
            depth if depth_usable else None,
        )

        # --- feed the temporal filter -------------------------------------
        if fused_range is not None:
            self._filter.predict(dt)
            self._filter.update(
                fused_range,
                self._config.filter.measurement_variance_m2,
                measurement_confidence=fused_confidence,
            )
            self._last_fused_range_m = fused_range
            self._last_fused_confidence = fused_confidence
        else:
            self._filter.predict(dt)
            self._last_fused_range_m = None
            self._last_fused_confidence = 0.0

        self._last_disagreement_m = disagreement_m
        self._last_call_t = now

        if not self._filter.initialized:
            return TargetState(
                range_m=None,
                velocity_mps=None,
                confidence=0.0,
                source=Source.NONE,
                valid=False,
                fused_range_m=fused_range,
                disagreement_m=disagreement_m,
                note=note or "no usable measurement yet",
                timestamp=now,
            )

        prediction_only = fused_range is None
        source = Source.PREDICTED if prediction_only else fused_source
        valid = True
        note = note or ("prediction (no new usable measurement)" if prediction_only else "")

        return TargetState(
            range_m=self._filter.distance,
            velocity_mps=self._filter.velocity,
            confidence=self._filter.confidence,
            source=source,
            valid=valid,
            fused_range_m=fused_range,
            disagreement_m=disagreement_m,
            note=note,
            timestamp=now,
        )

    # ------------------------------------------------------------------ utils

    def _sanitize_vision(self, measurement: Optional[VisionMeasurement]) -> VisionMeasurement:
        if measurement is None or not isinstance(measurement, VisionMeasurement):
            return VisionMeasurement.invalid()
        if not measurement.valid:
            return measurement
        if measurement.distance_m is None or not math.isfinite(measurement.distance_m):
            return VisionMeasurement.invalid()
        cfg = self._config.vision
        if measurement.distance_m < cfg.minimum_distance_m or measurement.distance_m > cfg.maximum_distance_m:
            return VisionMeasurement.invalid()
        return measurement

    def _sanitize_lidar(self, measurement: Optional[LidarMeasurement]) -> LidarMeasurement:
        if measurement is None or not isinstance(measurement, LidarMeasurement):
            return LidarMeasurement.invalid()
        if not measurement.valid:
            return measurement
        if measurement.distance_m is None or not math.isfinite(measurement.distance_m):
            return LidarMeasurement.invalid()
        cfg = self._config.lidar
        if measurement.distance_m < cfg.minimum_distance_m or measurement.distance_m > cfg.maximum_distance_m:
            return LidarMeasurement.invalid()
        return measurement

    def _sanitize_depth(self, measurement: Optional[DepthMeasurement]) -> DepthMeasurement:
        if measurement is None or not isinstance(measurement, DepthMeasurement):
            return DepthMeasurement.invalid()
        if not measurement.valid:
            return measurement
        value = measurement.range_m if measurement.range_m is not None else measurement.depth_z_m
        if value is None or not math.isfinite(value):
            return DepthMeasurement.invalid(source=measurement.source)
        cfg = self._config.depth
        if value < cfg.minimum_distance_m or value > cfg.maximum_distance_m:
            return DepthMeasurement.invalid(source=measurement.source)
        return measurement

    def _resolve_dt(self, dt: Optional[float], now: float) -> float:
        if dt is not None and math.isfinite(dt):
            return max(dt, self._config.filter.minimum_dt_s)
        if self._last_call_t is not None and self._last_call_t > 0:
            return max(now - self._last_call_t, self._config.filter.minimum_dt_s)
        return self._config.default_dt_s

    def _fuse(self, vision, lidar, depth):
        """Confidence-aware selection/fusion. Returns a 5-tuple.

        Consensus (all usable sources agree within the disagreement threshold)
        yields a confidence-weighted average. Significant disagreement selects
        the highest-confidence source, applies a penalty, and reports the
        disagreement. No fixed source percentages are used.
        """
        cfg = self._config.fusion

        candidates = []
        if vision is not None:
            candidates.append((vision.distance_m, vision.confidence, Source.VISION, "vision"))
        if lidar is not None:
            candidates.append((lidar.distance_m, lidar.confidence, Source.LIDAR, "lidar"))
        if depth is not None:
            value = depth.range_m if depth.range_m is not None else depth.depth_z_m
            candidates.append((value, depth.confidence, Source.DEPTH, "depth"))

        if not candidates:
            return None, 0.0, Source.NONE, 0.0, "no usable measurements"

        if len(candidates) == 1:
            value, confidence, source, label = candidates[0]
            return value, confidence, source, 0.0, f"{label} only"

        values = [c[0] for c in candidates]
        disagreement = max(
            abs(a - b) for i, a in enumerate(values) for b in values[i + 1 :]
        )

        if disagreement > cfg.maximum_sensor_disagreement_m:
            value, picked_conf, source, label = max(candidates, key=lambda c: c[1])
            confidence = max(0.0, min(1.0, picked_conf * cfg.disagreement_confidence_penalty))
            note = f"sensors disagree ({disagreement:.2f} m); selected higher-confidence {label}"
            return value, confidence, source, disagreement, note

        total = sum(c[1] for c in candidates)
        if total <= 0.0:
            return values[0], 0.0, Source.FUSED, disagreement, ""
        weights = [c[1] / total for c in candidates]
        fused = sum(c[0] * w for c, w in zip(candidates, weights))
        confidence = sum(c[1] * c[1] for c in candidates) / total
        confidence = max(0.0, min(1.0, confidence))
        return fused, confidence, Source.FUSED, disagreement, ""