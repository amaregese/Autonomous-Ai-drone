"""Camera height above the target's ground plane.

The horizontal ground-distance geometry (``modules/distance_estimator/vision.py``)
needs the **camera's** height above the ground the target stands on, not the
vehicle's altitude above its launch point. Those differ by the camera's mounting
offset below the vehicle's altitude reference, and they only coincide when the
target is known to be standing at the same elevation as the vehicle's home.

``compute_camera_height`` is a pure helper with no telemetry dependency: the
caller supplies the vehicle AGL and the mount offset, and says whether assuming
the target stands at the home elevation is acceptable. It never manufactures a
height:

- ``assume_home=False`` means the target's elevation is unknown, so no camera
  height can be established -> ``None``;
- a non-positive or non-finite result is rejected -> ``None``.

It is deliberately not imported by the estimator; the application owns the
telemetry and passes the resulting camera height in.
"""
from __future__ import annotations

import math
from typing import Optional


def compute_camera_height(
    vehicle_agl: Optional[float],
    cam_offset_m: Optional[float],
    assume_home: bool,
) -> Optional[float]:
    """Camera height (m) above the target's ground plane, or ``None``.

    ``vehicle_agl`` is the vehicle's altitude above home; ``cam_offset_m`` is how
    far the camera sits *below* the vehicle's altitude reference (a downward
    gimbal or belly mount is positive). ``assume_home`` records the assumption
    that the target stands at the vehicle's home elevation.

    Returns ``vehicle_agl - cam_offset_m`` when that is a finite, strictly
    positive number and ``assume_home`` is true; otherwise ``None``.
    """
    if not assume_home:
        return None
    if vehicle_agl is None or cam_offset_m is None:
        return None
    if not isinstance(vehicle_agl, (int, float)) or not math.isfinite(vehicle_agl):
        return None
    if not isinstance(cam_offset_m, (int, float)) or not math.isfinite(cam_offset_m):
        return None

    height = float(vehicle_agl) - float(cam_offset_m)
    if not math.isfinite(height) or height <= 0.0:
        return None
    return height
