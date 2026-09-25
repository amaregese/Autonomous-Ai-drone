"""
Synthetic (deterministic) metric-depth source for tests and offline benchmarks.

This backend does NOT run a neural network. It produces a metric depth image
filled with a programmed distance (meters) so the estimator's ROI extraction,
confidence and fusion can be exercised and unit-tested deterministically.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional

import numpy as np

from ..depth import DepthFrame, DepthSource


@dataclass
class SimulatedDepthSource:
    """Deterministic scripted metric-depth source (meters)."""

    values: List[float] = field(default_factory=lambda: [3.0, 3.0, 3.1])
    confidence: float = 0.90
    loop: bool = True
    noise_m: float = 0.0

    def __post_init__(self) -> None:
        self._queue: Deque[float] = deque(self.values)

    def set_distance(self, distance_m: float) -> None:
        """Program the source to always report ``distance_m`` meters."""
        self._queue.clear()
        self._queue.append(distance_m)
        self.values = [distance_m]

    def set_sequence(self, values: List[float]) -> None:
        self.values = list(values)
        self._queue = deque(self.values)

    @property
    def name(self) -> str:
        return "synthetic"

    def estimate(self, image: Any = None) -> DepthFrame:
        if image is None:
            height, width = 480, 640
        else:
            height, width = np.asarray(image).shape[:2]

        if not self._queue:
            if self.loop and self.values:
                self._queue = deque(self.values)
            else:
                depth = np.full((height, width), np.nan, dtype=np.float64)
                valid = np.zeros((height, width), dtype=bool)
                return DepthFrame(depth, valid, self.name)

        value = self._queue.popleft()
        if self.loop:
            self._queue.append(value)

        depth = np.full((height, width), float(value), dtype=np.float64)
        if self.noise_m > 0.0:
            yy, xx = np.mgrid[0:height, 0:width]
            depth += self.noise_m * np.sin(xx / 97.0) * np.sin(yy / 113.0)
        valid = np.isfinite(depth) & (depth > 0.0)
        depth[~valid] = np.nan
        return DepthFrame(depth, valid, self.name)