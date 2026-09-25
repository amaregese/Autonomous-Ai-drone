"""
Isolated, testable distance estimator subsystem.

The active YOLO11 loop uses the geometric vision measurement as its
authoritative per-detection range. The legacy navigation and optional metric
depth paths remain available for compatibility and offline evaluation.

Public API:
    DistanceEstimator        main estimator (update / estimate_vision /
                             estimate_depth / reset)
    VisionDistanceEstimator  geometric vision measurement + confidence
    DistanceVelocityFilter   1-D Kalman filter (range + velocity)
    CameraIntrinsics         validated camera calibration
    SimulatedLidar           deterministic scripted LiDAR source
    DepthSource / DepthFrame / DepthMeasurement / ZoeDepthBackend /
    SimulatedDepthSource     metric-depth interface and backends
    VisionMeasurement, LidarMeasurement, TargetState, Source
    EstimatorConfig          aggregate estimator configuration

Note: importing this package does NOT load torch/transformers/cv2/ultralytics
(verified by tests). The metric-depth backend loads its model only on demand.
"""
from .calibration import (
    DEFAULT_CONFIGURED_INTRINSICS,
    CalibrationError,
    CameraIntrinsics,
    load_calibration,
    load_configured_intrinsics,
)
from .config import (
    DepthConfig,
    EstimatorConfig,
    FilterConfig,
    FusionConfig,
    LidarConfig,
    VisionConfig,
)
from .depth import (
    DepthBackendError,
    DepthFrame,
    DepthSource,
    RoiDepthStats,
    compute_depth_confidence,
    extract_depth_roi,
)
from .depth_backend import SimulatedDepthSource, ZoeDepthBackend
from .estimator import DistanceEstimator
from .filter import DistanceVelocityFilter
from .lidar import FixedLidar, LidarSource, SimulatedLidar
from .models import DepthMeasurement, LidarMeasurement, Source, TargetState, VisionMeasurement
from .vision import VisionDistanceEstimator, annotate_detection, annotate_detections, estimate_detection

__all__ = [
    "CalibrationError",
    "CameraIntrinsics",
    "DEFAULT_CONFIGURED_INTRINSICS",
    "DepthBackendError",
    "DepthConfig",
    "DepthFrame",
    "DepthMeasurement",
    "DepthSource",
    "DistanceEstimator",
    "DistanceVelocityFilter",
    "EstimatorConfig",
    "FilterConfig",
    "FixedLidar",
    "FusionConfig",
    "LidarConfig",
    "LidarMeasurement",
    "LidarSource",
    "RoiDepthStats",
    "SimulatedDepthSource",
    "SimulatedLidar",
    "Source",
    "TargetState",
    "VisionConfig",
    "VisionDistanceEstimator",
    "VisionMeasurement",
    "annotate_detection",
    "annotate_detections",
    "estimate_detection",
    "ZoeDepthBackend",
    "compute_depth_confidence",
    "extract_depth_roi",
    "load_calibration",
    "load_configured_intrinsics",
]