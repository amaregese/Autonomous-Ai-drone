"""
LiDAR input abstraction.

``LidarSource`` is the single interface the estimator consumes. A real LiDAR
driver implements :meth:`LidarSource.read`; no architectural change is needed
later. ``SimulatedLidar`` is a deterministic source for unit tests and the
offline demo (it never uses random numbers).
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Protocol

from .models import LidarMeasurement


class LidarSource(Protocol):
    """Anything that produces a LiDAR measurement on demand."""

    def read(self) -> LidarMeasurement:
        ...


@dataclass
class SimulatedLidar:
    """Deterministic scripted LiDAR source.

    ``values`` is cycled through on every ``read()``. A single value with
    ``loop=True`` behaves as a fixed source. A value <= 0 yields an invalid
    measurement, mirroring what a real driver would report for a bad return.
    """

    values: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 5.0])
    confidence: float = 0.95
    loop: bool = True

    def __post_init__(self) -> None:
        self._queue: Deque[float] = deque(self.values)

    def set_distance(self, distance_m: float) -> None:
        """Set a single fixed reading for all subsequent ``read()`` calls."""
        self._queue.clear()
        self._queue.append(distance_m)
        self.values = [distance_m]

    def set_sequence(self, values: List[float]) -> None:
        """Set an explicit scripted sequence (replayed each loop)."""
        self.values = list(values)
        self._queue = deque(self.values)

    def read(self) -> LidarMeasurement:
        if not self._queue:
            if self.loop and self.values:
                self._queue = deque(self.values)
            else:
                return LidarMeasurement.invalid()

        value = self._queue.popleft()
        if self.loop:
            self._queue.append(value)

        valid = value is not None and math.isfinite(value) and value > 0.0
        return LidarMeasurement(
            distance_m=value if valid else None,
            confidence=self.confidence if valid else 0.0,
            valid=valid,
            timestamp=time.time(),
        )


class FixedLidar(SimulatedLidar):
    """Convenience alias: a LiDAR that always returns one distance."""

    def __init__(self, distance_m: float, confidence: float = 0.95) -> None:
        super().__init__(values=[distance_m], confidence=confidence, loop=True)