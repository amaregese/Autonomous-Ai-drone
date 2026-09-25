"""
Self-contained 1-D constant-velocity Kalman filter for target range/velocity.

State vector  x = [distance, velocity]
Transition     F = [[1, dt], [0, 1]]          (constant velocity)
Control        H = [1, 0]                     (we measure distance only)

The implementation uses plain Python scalar arithmetic (2x2 matrices are a
handful of floats), so no external dependency is introduced. Discrete white
acceleration noise drives ``Q``.

Behaviour contract:
  * ``predict(dt)``  propagates the state and covariance; a no-op until the
    filter has been initialized by the first valid ``update``.
  * ``update(...)``  rejects non-finite measurements (returns unchanged) and
    always clamps the filtered distance to ``minimum_range_m``.
  * ``reset()``      returns to the pristine initial state.
"""
from __future__ import annotations

import math
from typing import Optional

from .config import FilterConfig


class DistanceVelocityFilter:
    def __init__(self, config: FilterConfig) -> None:
        self._config = config
        self.reset()

    def reset(self) -> None:
        cfg = self._config
        self._x = [0.0, 0.0]  # distance, velocity
        self._p00 = cfg.initial_distance_variance_m2
        self._p01 = 0.0
        self._p10 = 0.0
        self._p11 = cfg.initial_velocity_variance_m2_s2
        self._initialized = False
        self._confidence = 0.0
        self._last_innovation = 0.0

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def distance(self) -> float:
        return self._x[0]

    @property
    def velocity(self) -> float:
        return self._x[1]

    @property
    def confidence(self) -> float:
        return self._confidence

    def _validated_dt(self, dt: Optional[float]) -> float:
        if dt is None or not math.isfinite(dt):
            return self._config.minimum_dt_s
        return max(dt, self._config.minimum_dt_s)

    def predict(self, dt: Optional[float]) -> None:
        if not self._initialized:
            return
        dt = self._validated_dt(dt)

        # x' = F x
        x0, x1 = self._x
        self._x = [x0 + dt * x1, x1]

        # P' = F P F^T
        p00, p01, p10, p11 = self._p00, self._p01, self._p10, self._p11
        self._p00 = p00 + dt * (p10 + p01) + dt * dt * p11
        self._p01 = p01 + dt * p11
        self._p10 = p10 + dt * p11
        self._p11 = p11

        # Q (discrete white acceleration noise)
        q = self._config.process_variance_m2_s3
        dt2 = dt * dt
        dt3 = dt2 * dt
        self._p00 += q * dt3 / 3.0
        self._p01 += q * dt2 / 2.0
        self._p10 += q * dt2 / 2.0
        self._p11 += q * dt

        self._x[0] = max(self._x[0], self._config.minimum_range_m)
        self._confidence *= self._config.prediction_confidence_decay

    def update(
        self,
        measurement: Optional[float],
        measurement_variance: Optional[float],
        measurement_confidence: float = 1.0,
    ) -> None:
        """Correct the state with a scalar distance measurement.

        Non-finite measurements are rejected (no-op). ``measurement_variance``
        is scaled inversely by ``measurement_confidence`` so confident sources
        are trusted more, then clamped to the configured bounds.
        """
        if measurement is None or not math.isfinite(measurement):
            return

        R = self._resolved_variance(measurement_variance, measurement_confidence)

        if not self._initialized:
            self._x[0] = max(measurement, self._config.minimum_range_m)
            self._x[1] = 0.0
            self._initialized = True
            self._set_post_measurement_covariance(R)
            self._confidence = max(0.0, min(1.0, measurement_confidence))
            self._last_innovation = 0.0
            return

        p00, p01, p10, p11 = self._p00, self._p01, self._p10, self._p11
        s = p00 + R
        if s <= 0.0:
            return
        k0 = p00 / s
        k1 = p10 / s

        innovation = measurement - self._x[0]
        self._x[0] += k0 * innovation
        self._x[1] += k1 * innovation
        self._x[0] = max(self._x[0], self._config.minimum_range_m)

        self._p00 = (1.0 - k0) * p00
        self._p01 = (1.0 - k0) * p01
        self._p10 = p10 - k1 * p00
        self._p11 = p11 - k1 * p01

        innovation_ratio = abs(innovation) / math.sqrt(s) if s > 0.0 else 1.0
        quality = 1.0 / (1.0 + innovation_ratio)
        self._confidence = max(0.0, min(1.0, measurement_confidence * quality))
        self._last_innovation = innovation

    def _set_post_measurement_covariance(self, r: float) -> None:
        # First measurement: P = (I - K P H^T) after a direct injection.
        s = self._p00 + r
        if s <= 0.0:
            return
        k0 = self._p00 / s
        self._p00 = (1.0 - k0) * self._p00
        self._p01 = 0.0
        self._p10 = 0.0
        self._p11 = self._p11 * (1.0 - k0)

    def _resolved_variance(self, measurement_variance: Optional[float], confidence: float) -> float:
        cfg = self._config
        if measurement_variance is None or not math.isfinite(measurement_variance) or measurement_variance <= 0.0:
            measurement_variance = cfg.measurement_variance_m2
        effective = measurement_variance / max(min(confidence, 1.0), 0.05)
        return max(cfg.minimum_measurement_variance_m2, min(cfg.maximum_measurement_variance_m2, effective))