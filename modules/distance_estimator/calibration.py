"""
Camera intrinsic calibration loading for the distance estimator.

The loader reads ``fx``, ``fy``, ``cx``, ``cy`` and distortion coefficients
from a JSON file and validates them. A repository manifest or the documented
configured default is used when no explicit calibration is supplied; callers
learn when vision geometry is unavailable instead of silently trusting an
arbitrary focal length.

The application's existing calibration path (``modules/auto_calibrate.py``,
``tools/calibrate_camera.py``) is intentionally not touched.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_DISTORTION_KEYS = ("k1", "k2", "p1", "p2", "k3")
DEFAULT_CONFIGURED_INTRINSICS = {
    "fx": 446.7,
    "fy": 446.7,
    "cx": 320.0,
    "cy": 240.0,
    "frame_w": 640,
    "frame_h": 480,
}


class CalibrationError(Exception):
    """Raised when calibration data is missing, malformed or invalid."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(slots=True)
class CameraIntrinsics:
    """Validated camera intrinsics required by the geometric vision model."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Dict[str, float] = field(default_factory=dict)
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None
    rms_error: Optional[float] = None

    def validate(self, frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> List[str]:
        """Return a list of human-readable problems. Empty list == valid."""
        errors: List[str] = []

        if not _is_number(self.fx) or self.fx <= 0.0:
            errors.append("fx must be a positive number")
        if not _is_number(self.fy) or self.fy <= 0.0:
            errors.append("fy must be a positive number")
        if not _is_number(self.cx) or self.cx < 0.0:
            errors.append("cx must be a non-negative number")
        if not _is_number(self.cy) or self.cy < 0.0:
            errors.append("cy must be a non-negative number")

        w = frame_w if frame_w is not None else self.frame_w
        h = frame_h if frame_h is not None else self.frame_h
        if (w is not None) != (h is not None):
            errors.append("frame_w and frame_h must either both be set or both be omitted")
        elif w is not None and h is not None:
            if not isinstance(w, int) or w <= 0:
                errors.append("frame_w must be a positive integer")
            if not isinstance(h, int) or h <= 0:
                errors.append("frame_h must be a positive integer")
            if w > 0 and h > 0:
                if not (0.0 <= self.cx < w):
                    errors.append(f"cx ({self.cx}) is outside frame width {w}")
                if not (0.0 <= self.cy < h):
                    errors.append(f"cy ({self.cy}) is outside frame height {h}")

        for key, value in self.distortion.items():
            if key not in _DISTORTION_KEYS:
                errors.append(f"unknown distortion key {key!r} (expected one of {_DISTORTION_KEYS})")
            elif not _is_number(value):
                errors.append(f"distortion {key!r} must be numeric")

        return errors

    def is_valid(self, frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> bool:
        return not self.validate(frame_w, frame_h)

    def scaled_to_frame(self, frame_w: int, frame_h: int) -> "CameraIntrinsics":
        """Scale this calibration to the exact pixel frame being processed."""
        if self.frame_w is None or self.frame_h is None:
            raise CalibrationError("calibration frame dimensions are required for runtime scaling")
        if frame_w <= 0 or frame_h <= 0:
            raise CalibrationError("target frame dimensions must be positive")
        sx, sy = frame_w / self.frame_w, frame_h / self.frame_h
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy,
            distortion=dict(self.distortion), frame_w=frame_w, frame_h=frame_h,
            rms_error=self.rms_error,
        )

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": {key: self.distortion.get(key, 0.0) for key in _DISTORTION_KEYS},
        }
        if self.frame_w is not None and self.frame_h is not None:
            data["frame_w"] = self.frame_w
            data["frame_h"] = self.frame_h
        if self.rms_error is not None:
            data["rms_error"] = self.rms_error
        return data

    def save_to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, object],
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
    ) -> "CameraIntrinsics":
        if not isinstance(data, dict):
            raise CalibrationError("calibration data must be a JSON object")

        missing = [k for k in ("fx", "fy", "cx", "cy") if k not in data]
        if missing:
            raise CalibrationError(f"calibration missing required key(s): {', '.join(missing)}")

        fx = data["fx"]
        fy = data["fy"]
        cx = data["cx"]
        cy = data["cy"]
        for name, value in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)):
            if not _is_number(value):
                raise CalibrationError(f"{name} must be numeric, got {value!r}")

        distortion_raw = data.get("distortion", {})
        if distortion_raw is None:
            distortion_raw = {}
        if not isinstance(distortion_raw, dict):
            raise CalibrationError("distortion must be a JSON object")

        distortion: Dict[str, float] = {}
        for key in _DISTORTION_KEYS:
            value = distortion_raw.get(key, 0.0)
            if not _is_number(value):
                raise CalibrationError(f"distortion {key!r} must be numeric")
            distortion[key] = float(value)

        resolved_w = data.get("frame_w")
        if resolved_w is None:
            resolved_w = frame_w
        resolved_h = data.get("frame_h")
        if resolved_h is None:
            resolved_h = frame_h

        rms_error: Optional[float] = None
        if "rms_error" in data and _is_number(data["rms_error"]):
            rms_error = float(data["rms_error"])

        intrinsics = cls(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            distortion=distortion,
            frame_w=int(resolved_w) if isinstance(resolved_w, int) else resolved_w,
            frame_h=int(resolved_h) if isinstance(resolved_h, int) else resolved_h,
            rms_error=rms_error,
        )

        errors = intrinsics.validate()
        if errors:
            raise CalibrationError("invalid calibration: " + "; ".join(errors))
        return intrinsics

    @classmethod
    def from_json_file(
        cls,
        path: str,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
    ) -> "CameraIntrinsics":
        if not os.path.exists(path):
            raise CalibrationError(f"calibration file not found: {path}")
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"failed to read calibration file {path}: {exc}") from exc
        return cls.from_dict(data, frame_w, frame_h)


def load_configured_intrinsics(
    manifest_path: Optional[str] = None,
    fallback: Optional[Dict[str, object]] = None,
) -> Optional[CameraIntrinsics]:
    data = dict(fallback or DEFAULT_CONFIGURED_INTRINSICS)
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            camera = manifest.get("camera") or {}
            configured = dict(camera.get("intrinsics") or {})
            if configured:
                data.update(configured)
            if "frame_w" not in configured and camera.get("frame_w"):
                data["frame_w"] = camera["frame_w"]
            if "frame_h" not in configured and camera.get("frame_h"):
                data["frame_h"] = camera["frame_h"]
        except (OSError, ValueError, TypeError):
            pass
    try:
        return CameraIntrinsics.from_dict(data)
    except CalibrationError:
        return None


def load_calibration(
    path: str,
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
) -> Optional[CameraIntrinsics]:
    """Best-effort loader. Returns ``None`` when calibration is unavailable."""
    try:
        return CameraIntrinsics.from_json_file(path, frame_w, frame_h)
    except CalibrationError:
        return None
