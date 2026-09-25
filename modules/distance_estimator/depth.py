"""
Metric depth interface and utilities for the distance estimator.

This module defines:

  * ``DepthSource`` — the protocol any metric-depth backend implements.
  * ``DepthFrame`` — a full metric depth image (meters) plus a valid mask.
  * ``extract_depth_roi`` — extracts an interior ROI around a detection and
    computes robust statistics (median Z, median Euclidean range, MAD, stdev).
  * ``compute_depth_confidence`` — deterministic 0..1 quality score built only
    from measurable properties (usable without running any neural network).

Design rules followed here:

  * A relative-depth model output must NOT be treated as meters. Backends must
    expose *metric* depth (see ``depth_backend/metric_depth.py``).
  * Camera-axis depth (Z) is distinct from Euclidean range. The ROI extracts
    per-pixel Euclidean range only when camera intrinsics are supplied;
    otherwise ``range`` is left unavailable instead of being invented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from .calibration import CameraIntrinsics
from .config import DepthConfig


class DepthBackendError(Exception):
    """Raised when a depth backend cannot load or run."""


class DepthSource(Protocol):
    """Produces a metric depth image for one camera frame."""

    def estimate(self, image: Any) -> "DepthFrame":
        """Run metric depth on ``image`` (HxWx3 array) and return a DepthFrame."""
        ...

    @property
    def name(self) -> str:
        ...


@dataclass(slots=True)
class DepthFrame:
    """Metric depth for one frame, in meters."""

    depth_m: np.ndarray  # HxW, meters; NaN where invalid
    valid_mask: np.ndarray  # HxW bool
    source: str

    def __post_init__(self) -> None:
        if self.depth_m.shape != self.valid_mask.shape:
            raise DepthBackendError("depth_m and valid_mask must have the same shape")
        if self.depth_m.ndim != 2:
            raise DepthBackendError("depth_m must be a 2-D image")


@dataclass(slots=True)
class RoiDepthStats:
    """Robust statistics computed over the interior depth ROI."""

    z_median_m: Optional[float] = None  # camera-axis depth
    range_median_m: Optional[float] = None  # Euclidean range (nanable)
    mad_m: float = 0.0  # median absolute deviation from median depth
    stdev_m: float = 0.0
    n_valid_pixels: int = 0
    roi_size_px: int = 0
    valid: bool = False

    @property
    def fraction_valid(self) -> float:
        if self.roi_size_px <= 0:
            return 0.0
        return self.n_valid_pixels / self.roi_size_px


def _z_to_range_m(z: np.ndarray, u: np.ndarray, v: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    x_n = (u - float(intrinsics.cx)) / float(intrinsics.fx)
    y_n = (v - float(intrinsics.cy)) / float(intrinsics.fy)
    return z / np.sqrt(1.0 + x_n * x_n + y_n * y_n)


def extract_depth_roi(
    frame: DepthFrame,
    left: float,
    top: float,
    right: float,
    bottom: float,
    margin_fraction: float = 0.2,
    min_valid_fraction: float = 0.5,
    intrinsics: Optional[CameraIntrinsics] = None,
) -> RoiDepthStats:
    """Collect the valid metric-depth pixels inside the interior ROI of a bbox.

    The margin trims the bbox borders where background/sky/ground or neighbor
    objects usually live. ``min_valid_fraction`` gates how much of the ROI must
    contain usable depth before the statistic is trusted.
    """
    h, w = frame.depth_m.shape
    bw = right - left
    bh = bottom - top
    margin = max(0.0, margin_fraction)

    x0 = max(0, int(round(left + bw * margin)))
    x1 = min(w, int(round(right - bw * margin)))
    y0 = max(0, int(round(top + bh * margin)))
    y1 = min(h, int(round(bottom - bh * margin)))

    roi_size = max(0, x1 - x0) * max(0, y1 - y0)
    if roi_size <= 0:
        return RoiDepthStats(valid=False, roi_size_px=0)

    roi = frame.depth_m[y0:y1, x0:x1]
    mask = frame.valid_mask[y0:y1, x0:x1]

    values = roi[mask]
    n_valid = int(values.size)
    fraction = n_valid / roi_size if roi_size else 0.0

    if n_valid == 0 or fraction < min_valid_fraction:
        return RoiDepthStats(valid=False, n_valid_pixels=n_valid, roi_size_px=roi_size)

    z_median = float(np.median(values))
    mad = float(np.median(np.abs(values - z_median)))
    stdev = float(np.std(values))

    range_median: Optional[float] = None
    if (
        intrinsics is not None
        and intrinsics.is_valid()
        and math.isfinite(z_median)
        and z_median > 0.0
    ):
        yy, xx = np.nonzero(mask)
        u = x0 + xx.astype(np.float64) + 0.5
        v = y0 + yy.astype(np.float64) + 0.5
        z = roi[mask]
        ranges = _z_to_range_m(z, u, v, intrinsics)
        range_median = float(np.median(ranges))

    return RoiDepthStats(
        z_median_m=z_median,
        range_median_m=range_median,
        mad_m=mad,
        stdev_m=stdev,
        n_valid_pixels=n_valid,
        roi_size_px=roi_size,
        valid=True,
    )


def compute_depth_confidence(
    stats: RoiDepthStats,
    detection_confidence: float,
    config: DepthConfig,
) -> float:
    """Deterministic 0..1 measurement-quality score (not claimed accuracy)."""
    if not stats.valid:
        return 0.0

    coverage = min(1.0, stats.fraction_valid)

    denom = max(abs(stats.z_median_m or 0.0), 1e-6)
    relative_mad = math.sqrt(stats.mad_m * stats.mad_m + 1e-6) / denom
    stability = 1.0 / (1.0 + relative_mad)

    detection_score = max(0.0, min(1.0, detection_confidence))

    in_band = (
        stats.z_median_m is not None
        and config.minimum_distance_m <= stats.z_median_m <= config.maximum_distance_m
    )
    distance_score = 1.0 if in_band else 0.5

    confidence = (
        config.coverage_weight * coverage
        + config.stability_weight * stability
        + config.detection_weight * detection_score
        + config.distance_band_weight * distance_score
    )
    weights = (
        config.coverage_weight
        + config.stability_weight
        + config.detection_weight
        + config.distance_band_weight
    )
    if weights > 0.0:
        confidence /= weights
    return max(0.0, min(1.0, confidence))