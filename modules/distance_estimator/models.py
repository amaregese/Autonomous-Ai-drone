"""
Strongly typed measurement / state models for the distance estimator subsystem.

These dataclasses replace ad-hoc dictionaries so that measurements, sources and
the final target state are unambiguous and unit-testable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Source(Enum):
    """Where the current target-state estimate originated."""

    NONE = "none"
    VISION = "vision"
    LIDAR = "lidar"
    DEPTH = "depth"
    FUSED = "fused"
    PREDICTED = "predicted"


@dataclass(slots=True)
class VisionMeasurement:
    """A single monocular vision distance candidate."""

    distance_m: Optional[float] = None
    confidence: float = 0.0
    bbox_height_px: Optional[float] = None
    bbox_width_px: Optional[float] = None
    class_name: str = ""
    truncated: Optional[bool] = None
    valid: bool = False
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()

    @classmethod
    def invalid(cls, class_name: str = "", timestamp: Optional[float] = None) -> "VisionMeasurement":
        return cls(
            distance_m=None,
            confidence=0.0,
            class_name=class_name,
            truncated=None,
            valid=False,
            timestamp=timestamp,
        )


@dataclass(slots=True)
class LidarMeasurement:
    """A single LiDAR range reading."""

    distance_m: Optional[float] = None
    confidence: float = 0.0
    valid: bool = False
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()

    @classmethod
    def invalid(cls, timestamp: Optional[float] = None) -> "LidarMeasurement":
        return cls(distance_m=None, confidence=0.0, valid=False, timestamp=timestamp)


@dataclass(slots=True)
class DepthMeasurement:
    """A metric monocular depth reading extracted at the target ROI.

    ``depth_z_m`` is the camera-axis (perpendicular) depth. ``range_m`` is the
    Euclidean distance from the camera to the target, computed geometrically
    from ``depth_z_m`` and the pixel location when camera intrinsics are
    available. ``range_m`` is left unset (None) rather than invented when only
    Z-depth is available. ``source`` names the backend/model that produced it.
    """

    depth_z_m: Optional[float] = None
    range_m: Optional[float] = None
    confidence: float = 0.0
    valid: bool = False
    source: str = "monocular"
    timestamp: Optional[float] = None
    n_valid_pixels: int = 0
    roi_size_px: int = 0
    depth_stdev_m: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()

    @classmethod
    def invalid(cls, source: str = "monocular", timestamp: Optional[float] = None) -> "DepthMeasurement":
        return cls(
            depth_z_m=None,
            range_m=None,
            confidence=0.0,
            valid=False,
            source=source,
            timestamp=timestamp,
            n_valid_pixels=0,
            roi_size_px=0,
            depth_stdev_m=0.0,
        )


@dataclass(slots=True)
class TargetState:
    """Final estimated state of the tracked target, produced by the estimator."""

    range_m: Optional[float] = None
    velocity_mps: Optional[float] = None
    confidence: float = 0.0
    source: Source = Source.NONE
    valid: bool = False
    fused_range_m: Optional[float] = None
    disagreement_m: float = 0.0
    note: str = ""
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()