"""
Configuration for the isolated distance estimator.

These values are deliberately separate from ``modules.app_config`` flight
controller constants (``MAX_FOLLOW_DIST``, ``FORWARD_DEADBAND``,
``FORWARD_BRAKE_ZONE``, ``GAIN_FORWARD``...). The estimator does not control
flight, so it must not depend on flight constants.

All numbers are sane *starting* defaults. They require real-world calibration
and testing before use on hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Kept small on purpose: it is a *starting point*, not a copy of the old
# application table. Real per-class heights must be measured for the actual
# target set used in the field.
DEFAULT_KNOWN_CLASS_HEIGHTS_M: Dict[str, float] = {
    "person": 1.30,
    "bicycle": 1.00,
    "car": 1.50,
    "motorbike": 1.20,
    "truck": 2.50,
    "bus": 3.00,
    "cat": 0.25,
    "dog": 0.50,
    "bird": 0.15,
    "horse": 1.60,
    "sheep": 0.90,
    "cow": 1.40,
}


@dataclass(slots=True)
class VisionConfig:
    """Quality thresholds and weights for the geometric vision measurement."""

    minimum_distance_m: float = 0.25
    maximum_distance_m: float = 20.0
    minimum_bbox_height_px: float = 5.0
    reference_bbox_height_px: float = 100.0
    minimum_detection_confidence: float = 0.25
    default_object_height_m: float = 0.5
    known_class_heights_m: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_KNOWN_CLASS_HEIGHTS_M)
    )
    bbox_size_weight: float = 0.35
    detection_confidence_weight: float = 0.40
    class_availability_weight: float = 0.15
    edge_proximity_weight: float = 0.10
    vision_confidence_threshold: float = 0.30


@dataclass(slots=True)
class LidarConfig:
    """Quality thresholds for LiDAR measurements."""

    minimum_distance_m: float = 0.1
    maximum_distance_m: float = 50.0
    default_confidence: float = 0.90
    lidar_confidence_threshold: float = 0.30


@dataclass(slots=True)
class FusionConfig:
    """Measurement selection / fusion policy (no fixed blending ratio)."""

    maximum_sensor_disagreement_m: float = 2.0
    disagreement_confidence_penalty: float = 0.5
    minimum_source_confidence: float = 0.0


@dataclass(slots=True)
class DepthConfig:
    """ROI extraction and confidence tuning for the metric-depth source."""

    minimum_distance_m: float = 0.25
    maximum_distance_m: float = 20.0
    roi_margin_fraction: float = 0.2
    min_valid_fraction: float = 0.5
    coverage_weight: float = 0.35
    stability_weight: float = 0.35
    detection_weight: float = 0.20
    distance_band_weight: float = 0.10
    depth_confidence_threshold: float = 0.30


@dataclass(slots=True)
class FilterConfig:
    """Tunables for the 1-D distance/velocity Kalman filter."""

    initial_distance_variance_m2: float = 4.0
    initial_velocity_variance_m2_s2: float = 4.0
    process_variance_m2_s3: float = 0.05
    measurement_variance_m2: float = 0.25
    minimum_measurement_variance_m2: float = 0.01
    maximum_measurement_variance_m2: float = 4.0
    minimum_range_m: float = 0.0
    prediction_confidence_decay: float = 0.90
    minimum_dt_s: float = 1e-3


@dataclass(slots=True)
class EstimatorConfig:
    """Aggregate configuration for the whole estimator."""

    vision: VisionConfig = field(default_factory=VisionConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    default_dt_s: float = 1.0 / 30.0