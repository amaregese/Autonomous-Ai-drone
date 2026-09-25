"""
Task 9: autonomous person-follow controller.

Reuses (never duplicates):
  * the existing YOLO11 ``Detection`` object
    (``modules.yolo11_detector.types.Detection``)
  * the existing distance estimator
    (``modules.distance_estimator.DistanceEstimator``)
  * the project's single MAVLink link via ``modules.drone`` using the
    existing ``SET_POSITION_TARGET_LOCAL_NED`` body-frame velocity mechanism
    (``drone.send_movement_command_XYA``).

This module is a pure decision/command layer: it turns a currently-tracked
detection into a body-frame velocity command (forward ``vx`` + lateral ``vy``)
using a configurable distance->speed profile, keeps the bbox horizontal center
centred, ramps accelerations, and never issues a command from a stale, missing,
low-confidence or unsafe target.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from modules import app_config
from modules.distance_estimator.vision import estimate_detection

DEFAULT_DT_S = 1.0 / 30.0
MAX_DT_S = 0.5


@dataclass(slots=True)
class FollowConfig:
    """All configurable follow/safety parameters (mirrors ``app_config``)."""

    speed_profile: Sequence[Tuple[float, float]] = field(
        default_factory=lambda: list(app_config.FOLLOW_SPEED_PROFILE)
    )
    max_speed: float = app_config.FOLLOW_MAX_SPEED_MPS
    min_speed: float = app_config.FOLLOW_MIN_SPEED_MPS
    min_distance: float = app_config.FOLLOW_MIN_DISTANCE_M
    max_lateral_speed: float = app_config.FOLLOW_MAX_LATERAL_SPEED_MPS
    accel_limit: float = app_config.FOLLOW_ACCEL_LIMIT_MPS2
    confidence_threshold: float = app_config.FOLLOW_CONFIDENCE_THRESHOLD
    loss_timeout: float = app_config.FOLLOW_TARGET_LOSS_TIMEOUT_S
    enabled: bool = app_config.FOLLOW_ENABLED
    rtl_on_loss: bool = app_config.FOLLOW_RTL_ON_LOSS
    lateral_gain: float = app_config.FOLLOW_LATERAL_GAIN
    yaw_gain: float = app_config.GAIN_YAW
    max_yaw: float = app_config.MAX_YAW
    target_class: str = app_config.FOLLOW_TARGET_CLASS
    max_detection_age: float = app_config.FOLLOW_MAX_DETECTION_AGE_S

    @classmethod
    def from_app_config(cls) -> "FollowConfig":
        """Build a config snapshot from ``modules.app_config`` at call time."""
        return cls(
            speed_profile=list(app_config.FOLLOW_SPEED_PROFILE),
            max_speed=app_config.FOLLOW_MAX_SPEED_MPS,
            min_speed=app_config.FOLLOW_MIN_SPEED_MPS,
            min_distance=app_config.FOLLOW_MIN_DISTANCE_M,
            max_lateral_speed=app_config.FOLLOW_MAX_LATERAL_SPEED_MPS,
            accel_limit=app_config.FOLLOW_ACCEL_LIMIT_MPS2,
            confidence_threshold=app_config.FOLLOW_CONFIDENCE_THRESHOLD,
            loss_timeout=app_config.FOLLOW_TARGET_LOSS_TIMEOUT_S,
            enabled=app_config.FOLLOW_ENABLED,
            rtl_on_loss=app_config.FOLLOW_RTL_ON_LOSS,
            lateral_gain=app_config.FOLLOW_LATERAL_GAIN,
            yaw_gain=app_config.GAIN_YAW,
            max_yaw=app_config.MAX_YAW,
            target_class=app_config.FOLLOW_TARGET_CLASS,
            max_detection_age=app_config.FOLLOW_MAX_DETECTION_AGE_S,
        )


@dataclass(slots=True)
class FollowCommand:
    """A single body-frame follow command. ``vx`` is forward, ``vy`` is lateral."""

    active: bool = False
    vx: float = 0.0
    vy: float = 0.0
    yaw_cmd: float = 0.0
    range_m: Optional[float] = None
    x_delta: float = 0.0
    lane: str = "idle"
    lost: bool = False
    lost_time_s: float = 0.0
    reason: str = ""
    distance_source: str = ""
    distance_confidence: float = 0.0
    distance_valid: bool = False
    detection_id: Optional[int] = None


class PersonFollowController:
    """Computes smooth forward/lateral velocity commands for the tracked target.

    ``measure`` is an optional test injection returning a range in metres from
    ``(detection, frame_shape)``. When it is omitted the controller uses the
    supplied ``DistanceEstimator`` (reused from the existing subsystem); without
    an estimator it falls back to monocular geometry.
    """

    def __init__(self, config=None, distance_estimator=None, measure=None):
        self.config = config or FollowConfig.from_app_config()
        self._estimator = distance_estimator
        self._measure_override = measure
        self._focal_length_y = None
        self.range_m = None
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_update_t = None
        self._lost_since_t = None
        self._distance_source = ""
        self._distance_confidence = 0.0

    # ------------------------------------------------------------ wiring ----

    def set_distance_estimator(self, estimator) -> None:
        self._estimator = estimator

    def set_focal_length(self, fy: float) -> None:
        """Focal length for the monocular-geometry fallback ranging."""
        self._focal_length_y = float(fy)

    def reset(self) -> None:
        """Reset ramp and loss state (called when a new follow session starts)."""
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_update_t = None
        self._lost_since_t = None
        self.range_m = None
        self._distance_source = ""
        self._distance_confidence = 0.0

    @property
    def lost_time_s(self) -> float:
        if self._lost_since_t is None:
            return 0.0
        return time.time() - self._lost_since_t

    # ------------------------------------------------------------ update ----

    def update(self, detection, frame_shape, now=None) -> FollowCommand:
        """Produce the next velocity command for the current frame."""
        now = now if now is not None else time.time()

        if not self.config.enabled:
            return FollowCommand(active=False, lane="disabled", reason="follow disabled")

        usable, reason = self._usable(detection, now)
        if not usable:
            return self._on_invalid(now, reason)

        self._mark_target_present()
        range_m = self.measure_range(detection, frame_shape)
        if range_m is None or not math.isfinite(range_m) or range_m <= 0.0:
            self.range_m = None
            return FollowCommand(
                active=True,
                vx=0.0,
                vy=0.0,
                range_m=None,
                x_delta=self._x_delta(detection, frame_shape),
                lane="no_range",
                reason="no usable range measurement",
                distance_source="none",
                distance_confidence=0.0,
                distance_valid=False,
                detection_id=getattr(detection, "detection_id", id(detection)),
            )

        self.range_m = float(range_m)
        dt = self._resolve_dt(now)
        self._last_update_t = now

        vx_target = self._target_forward_speed(range_m)
        vy_target = self._target_lateral_speed(detection, frame_shape)

        vx = self._ramp(self._last_vx, vx_target, dt)
        vy = self._ramp(self._last_vy, vy_target, dt)
        self._last_vx = vx
        self._last_vy = vy

        x_delta = self._x_delta(detection, frame_shape)
        yaw_cmd = max(-self.config.max_yaw, min(self.config.max_yaw, x_delta * self.config.max_yaw * self.config.yaw_gain))
        lane = "follow" if (abs(vx) > 1e-9 or abs(vy) > 1e-9) else "hover"
        return FollowCommand(
            active=True,
            vx=vx,
            vy=vy,
            yaw_cmd=yaw_cmd,
            range_m=self.range_m,
            x_delta=x_delta,
            lane=lane,
            reason="tracking",
            distance_source=self._distance_source,
            distance_confidence=self._distance_confidence,
            distance_valid=True,
            detection_id=getattr(detection, "detection_id", id(detection)),
        )

    # --------------------------------------------------------- decision ----

    def _usable(self, detection, now) -> Tuple[bool, str]:
        if detection is None:
            return False, "no target"
        if not getattr(detection, "is_selected", False):
            return False, "not selected"
        if self.config.target_class and detection.class_name != self.config.target_class:
            return False, "wrong class"
        if detection.confidence < self.config.confidence_threshold:
            return False, "low confidence"
        age = now - getattr(detection, "timestamp", now)
        if age < 0.0 or age > self.config.max_detection_age:
            return False, "stale detection"
        return True, "ok"

    def _on_invalid(self, now, reason) -> FollowCommand:
        self.range_m = None
        if self._lost_since_t is None:
            self._lost_since_t = now
        lost_time = max(0.0, now - self._lost_since_t)
        self._last_vx = 0.0
        self._last_vy = 0.0
        return FollowCommand(
            active=False,
            vx=0.0,
            vy=0.0,
            yaw_cmd=0.0,
            range_m=None,
            lane="lost",
            lost=True,
            lost_time_s=lost_time,
            reason=reason,
            distance_source="none",
            distance_confidence=0.0,
            distance_valid=False,
        )

    def _mark_target_present(self) -> None:
        self._lost_since_t = None

    # ------------------------------------------------------ measurement ----

    def measure_range(self, detection, frame_shape) -> Optional[float]:
        """Range (m) to the target using its attached measurement or an injected fallback."""
        source = getattr(detection, "distance_source", None)
        if source is not None:
            distance = getattr(detection, "distance_m", None)
            valid = bool(getattr(detection, "distance_valid", False))
            if distance is not None:
                try:
                    distance = float(distance)
                    valid = valid and math.isfinite(distance) and distance > 0.0
                except (TypeError, ValueError):
                    distance = None
                    valid = False
            else:
                valid = False
            if valid:
                self._distance_source = source
                self._distance_confidence = float(getattr(detection, "distance_confidence", 0.0))
                return distance
            self._distance_source, self._distance_confidence = "none", 0.0
            return None
        if self._measure_override is not None:
            self._distance_source, self._distance_confidence = "override", 1.0
            return self._measure_override(detection, frame_shape)
        if self._estimator is not None:
            if hasattr(self._estimator, "estimate_vision"):
                vision = self._estimator.estimate_vision(
                    bbox_height_px=float(detection.height),
                    bbox_width_px=float(detection.width),
                    class_name=detection.class_name,
                    detection_confidence=float(detection.confidence),
                    frame_w=int(frame_shape[1]),
                    frame_h=int(frame_shape[0]),
                    bbox_top_px=float(detection.Top),
                    bbox_bottom_px=float(detection.Bottom),
                )
                state = self._estimator.update(vision, None, None)
                if state.valid and state.range_m is not None and math.isfinite(state.range_m):
                    self._distance_source = state.source.value
                    self._distance_confidence = state.confidence
                    return float(state.range_m)
            else:
                measurement = estimate_detection(
                    self._estimator,
                    detection,
                    int(frame_shape[1]),
                    int(frame_shape[0]),
                )
                if measurement.valid and measurement.distance_m is not None:
                    self._distance_source = "vision"
                    self._distance_confidence = measurement.confidence
                    return float(measurement.distance_m)
            self._distance_source, self._distance_confidence = "none", 0.0
            return None
        self._distance_source, self._distance_confidence = "geometry", 1.0
        return self._geometry_range(detection)

    def _geometry_range(self, detection) -> Optional[float]:
        fy = self._focal_length_y
        if not fy or fy <= 0 or detection.height <= 0:
            return None
        known = app_config.OBJECT_HEIGHTS.get(detection.class_name)
        obj_h = known if known else app_config.OBJECT_HEIGHT
        if not obj_h or obj_h <= 0:
            return None
        return (obj_h * fy) / detection.height

    # ------------------------------------------------------------- speed ----

    def _target_forward_speed(self, distance_m: float) -> float:
        # This check must happen before applying ``min_speed``.  A configured
        # minimum cruise speed must never turn into forward motion at (or
        # inside) the follow safety distance.
        if distance_m <= self.config.min_distance:
            return 0.0
        max_speed = max(0.0, self.config.max_speed)
        min_speed = min(max(0.0, self.config.min_speed), max_speed)
        speed = max_speed
        profile = sorted((d, s) for d, s in self.config.speed_profile)
        for dist, val in profile:
            if distance_m <= dist:
                speed = val
                break
        return max(min_speed, min(speed, max_speed))

    def _target_lateral_speed(self, detection, frame_shape) -> float:
        target = self._x_delta(detection, frame_shape) * self.config.lateral_gain
        cap = max(0.0, self.config.max_lateral_speed)
        return max(-cap, min(cap, target))

    @staticmethod
    def _x_delta(detection, frame_shape) -> float:
        width = frame_shape[1]
        if width <= 0:
            return 0.0
        cx = detection.Center[0]
        return (cx - width / 2.0) / (width / 2.0)

    def _resolve_dt(self, now: float) -> float:
        if self._last_update_t is None:
            return DEFAULT_DT_S
        dt = now - self._last_update_t
        if dt <= 0:
            return DEFAULT_DT_S
        return min(max(dt, 1e-3), MAX_DT_S)

    def _ramp(self, current: float, target: float, dt: float) -> float:
        max_step = max(0.0, self.config.accel_limit) * max(dt, 1e-3)
        if target >= current:
            return min(target, current + max_step)
        return max(target, current - max_step)

    # ---------------------------------------------------- loss handling ----

    def loss_action(self) -> str:
        """Safe behaviour once the loss timeout is reached: 'hover' or 'rtl'."""
        return "rtl" if self.config.rtl_on_loss else "hover"

    def loss_timeout_reached(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        if self._lost_since_t is None:
            return False
        return (now - self._lost_since_t) >= self.config.loss_timeout
