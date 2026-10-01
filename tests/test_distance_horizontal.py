"""
Unit tests for the opt-in horizontal ground-distance geometry.

Run with:
    python -m unittest discover -s tests -t . -p "test_*.py"

Scope: the pure helpers ``axis_depth_to_slant_range`` and
``slant_to_horizontal_distance``, plus the ``estimate_horizontal`` opt-in path.

These tests deliberately do NOT touch the person-follow controller, the speed
profile, or any MAVLink call. Nothing here changes vehicle movement: the live
follow path still uses ``estimate()`` and its optical-axis depth unchanged, and
this file also asserts that isolation.

Geometry under test (camera optical axis assumed LEVEL)::

    x_n       = (u_px - cx) / fx
    y_n       = (v_px - cy) / fy
    slant     = Z * sqrt(1 + x_n**2 + y_n**2)
    horizontal = sqrt(slant**2 - altitude**2)
"""
from __future__ import annotations

import math
import pathlib
import unittest

from modules.distance_estimator.calibration import CameraIntrinsics
from modules.distance_estimator.config import VisionConfig
from modules.distance_estimator.vision import (
    VisionDistanceEstimator,
    axis_depth_to_slant_range,
    slant_to_horizontal_distance,
)

FX = 446.7
FY = 446.7
CX = 320.0
CY = 240.0


def make_intrinsics(fx: float = FX, fy: float = FY, cx: float = CX, cy: float = CY) -> CameraIntrinsics:
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, frame_w=640, frame_h=480)


def true_slant_range(horizontal_m: float, altitude_m: float) -> float:
    return math.sqrt(horizontal_m * horizontal_m + altitude_m * altitude_m)


class TestSlantToHorizontalDistance(unittest.TestCase):
    """Requirements 10a/10b/10c: 4 m horizontal standoff at 1 m, 3 m and 5 m altitude."""

    def test_horizontal_4m_at_altitude_3m(self):
        """3 m altitude, 4 m horizontal: sqrt(5^2 - 3^2) == 4.0 m."""
        slant = true_slant_range(4.0, 3.0)
        result = slant_to_horizontal_distance(slant, 3.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 4.0, places=9)

    def test_horizontal_4m_at_altitude_1m(self):
        """1 m altitude, 4 m horizontal: sqrt(4.1231^2 - 1^2) == 4.0 m."""
        slant = true_slant_range(4.0, 1.0)
        result = slant_to_horizontal_distance(slant, 1.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 4.0, places=9)

    def test_horizontal_4m_at_altitude_5m(self):
        """5 m altitude, 4 m horizontal: sqrt(6.4031^2 - 5^2) == 4.0 m."""
        slant = true_slant_range(4.0, 5.0)
        result = slant_to_horizontal_distance(slant, 5.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 4.0, places=9)

    def test_zero_altitude_is_identity(self):
        """With the camera at ground level, horizontal == slant."""
        self.assertAlmostEqual(slant_to_horizontal_distance(4.0, 0.0), 4.0, places=12)


class TestSlantToHorizontalRejectsInvalid(unittest.TestCase):
    """Requirement 6: defensive rejection; never manufacture a distance."""

    def test_none_inputs_rejected(self):
        self.assertIsNone(slant_to_horizontal_distance(None, 3.0))
        self.assertIsNone(slant_to_horizontal_distance(5.0, None))
        self.assertIsNone(slant_to_horizontal_distance(None, None))

    def test_negative_altitude_rejected(self):
        """A negative altitude is invalid, not mirrored into a distance."""
        self.assertIsNone(slant_to_horizontal_distance(5.0, -1.0))
        self.assertIsNone(slant_to_horizontal_distance(5.0, -0.001))

    def test_zero_slant_rejected(self):
        """Zero slant range carries no information."""
        self.assertIsNone(slant_to_horizontal_distance(0.0, 3.0))

    def test_negative_slant_rejected(self):
        self.assertIsNone(slant_to_horizontal_distance(-5.0, 3.0))

    def test_nan_and_infinity_rejected(self):
        """Non-finite inputs must not propagate into a result."""
        self.assertIsNone(slant_to_horizontal_distance(float("nan"), 3.0))
        self.assertIsNone(slant_to_horizontal_distance(5.0, float("nan")))
        self.assertIsNone(slant_to_horizontal_distance(float("inf"), 3.0))
        self.assertIsNone(slant_to_horizontal_distance(5.0, float("inf")))

    def test_slant_smaller_than_altitude_rejected(self):
        """Requirement 10e: impossible geometry (target not out on the ground)."""
        self.assertIsNone(slant_to_horizontal_distance(2.0, 3.0))
        self.assertIsNone(slant_to_horizontal_distance(2.999, 3.0))

    def test_slant_equal_to_altitude_rejected(self):
        """slant == altitude implies zero horizontal range, which is not a measurement."""
        self.assertIsNone(slant_to_horizontal_distance(3.0, 3.0))

    def test_tiny_negative_radicand_does_not_raise(self):
        """Requirement 6: float round-off must not raise or produce NaN.

        ``slant`` is barely larger than ``altitude``; the radicand is a tiny
        positive number here, so a finite result is expected.
        """
        result = slant_to_horizontal_distance(3.0 + 1e-12, 3.0)
        self.assertIsNotNone(result)
        self.assertTrue(math.isfinite(result))
        self.assertGreaterEqual(result, 0.0)


class TestAxisDepthToSlantRange(unittest.TestCase):
    """Optical-axis depth -> slant range using the target pixel and intrinsics."""

    def test_optical_center_gives_identity(self):
        """Requirement 10f: at (cx, cy) the ray is the axis, so slant == Z."""
        intrinsics = make_intrinsics()
        for depth in (1.0, 3.0, 5.0, 12.5):
            with self.subTest(depth=depth):
                slant = axis_depth_to_slant_range(depth, CX, CY, intrinsics)
                self.assertIsNotNone(slant)
                self.assertAlmostEqual(slant, depth, places=12)

    def test_off_axis_pixel_lengthens_slant(self):
        """A pixel off the principal point yields slant > Z."""
        intrinsics = make_intrinsics()
        slant = axis_depth_to_slant_range(5.0, CX + 100.0, CY, intrinsics)
        self.assertIsNotNone(slant)
        self.assertGreater(slant, 5.0)
        expected = 5.0 * math.sqrt(1.0 + (100.0 / FX) ** 2)
        self.assertAlmostEqual(slant, expected, places=12)

    def test_matches_closed_form_in_both_axes(self):
        """x_n and y_n must both contribute."""
        intrinsics = make_intrinsics()
        u, v = CX + 60.0, CY + 40.0
        slant = axis_depth_to_slant_range(4.0, u, v, intrinsics)
        x_n = (u - CX) / FX
        y_n = (v - CY) / FY
        self.assertAlmostEqual(slant, 4.0 * math.sqrt(1.0 + x_n**2 + y_n**2), places=12)

    def test_symmetric_about_principal_point(self):
        """Opposite offsets give the same slant (distance is symmetric)."""
        intrinsics = make_intrinsics()
        left = axis_depth_to_slant_range(5.0, CX - 80.0, CY, intrinsics)
        right = axis_depth_to_slant_range(5.0, CX + 80.0, CY, intrinsics)
        self.assertAlmostEqual(left, right, places=12)

    def test_invalid_inputs_rejected(self):
        """Requirement 6: no invented distances."""
        intrinsics = make_intrinsics()
        self.assertIsNone(axis_depth_to_slant_range(None, CX, CY, intrinsics))
        self.assertIsNone(axis_depth_to_slant_range(5.0, None, CY, intrinsics))
        self.assertIsNone(axis_depth_to_slant_range(5.0, CX, None, intrinsics))
        self.assertIsNone(axis_depth_to_slant_range(5.0, CX, CY, None))
        self.assertIsNone(axis_depth_to_slant_range(0.0, CX, CY, intrinsics))
        self.assertIsNone(axis_depth_to_slant_range(-5.0, CX, CY, intrinsics))
        self.assertIsNone(axis_depth_to_slant_range(float("nan"), CX, CY, intrinsics))
        self.assertIsNone(axis_depth_to_slant_range(5.0, float("nan"), CY, intrinsics))
        self.assertIsNone(axis_depth_to_slant_range(5.0, CX, float("inf"), intrinsics))


class TestComposedHorizontalGeometry(unittest.TestCase):
    """The two helpers compose into the end-to-end horizontal conversion."""

    def _horizontal(self, horizontal_m: float, altitude_m: float) -> float:
        """Simulate the level-camera projection of a target at a known ground range."""
        intrinsics = make_intrinsics()
        # At the optical centre the axis depth equals the slant range, so the
        # pinhole model yields Z directly from the ground geometry.
        axis_depth = true_slant_range(horizontal_m, altitude_m)
        slant = axis_depth_to_slant_range(axis_depth, CX, CY, intrinsics)
        self.assertIsNotNone(slant)
        return slant_to_horizontal_distance(slant, altitude_m)

    def test_round_trip_4m_horizontal(self):
        """Requirement 10a: 4 m horizontal recovered at 1 m, 3 m and 5 m altitude."""
        for altitude in (1.0, 3.0, 5.0):
            with self.subTest(altitude=altitude):
                self.assertAlmostEqual(self._horizontal(4.0, altitude), 4.0, places=9)

    def test_round_trip_several_ranges(self):
        """A spread of standoffs all round-trip cleanly."""
        for horizontal in (2.0, 3.0, 4.0, 6.0, 8.0):
            with self.subTest(horizontal=horizontal):
                self.assertAlmostEqual(self._horizontal(horizontal, 3.0), horizontal, places=9)

    def test_altitude_reduces_reported_horizontal(self):
        """Higher altitude with the same slant must read a shorter horizontal range."""
        intrinsics = make_intrinsics()
        slant = axis_depth_to_slant_range(5.0, CX, CY, intrinsics)
        low = slant_to_horizontal_distance(slant, 1.0)
        high = slant_to_horizontal_distance(slant, 4.0)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertGreater(low, high)


class TestEstimateHorizontalOptIn(unittest.TestCase):
    """The opt-in estimator path: valid results, and defensive invalid results."""

    def setUp(self):
        self.intrinsics = make_intrinsics()
        self.estimator = VisionDistanceEstimator(self.intrinsics, VisionConfig())

    def _measure(self, altitude_m, **kwargs):
        params = dict(
            bbox_height_px=145.2,
            bbox_width_px=50.0,
            bbox_left_px=295.0,
            bbox_top_px=150.0,
            bbox_bottom_px=295.2,
            class_name="person",
            frame_w=640,
            frame_h=480,
        )
        params.update(kwargs)
        return self.estimator.estimate_horizontal(altitude_m=altitude_m, **params)

    def test_returns_horizontal_and_retains_axis_depth(self):
        """A valid measurement carries both values; distance_m is unchanged."""
        result = self._measure(3.0)
        self.assertTrue(result.valid)
        self.assertIsNotNone(result.distance_m)
        self.assertIsNotNone(result.horizontal_distance_m)
        self.assertIsNotNone(result.slant_range_m)
        self.assertAlmostEqual(result.altitude_m, 3.0, places=9)
        # horizontal < slant <= nothing manufactured beyond the inputs
        self.assertLess(result.horizontal_distance_m, result.slant_range_m)

    def test_explicit_pixel_used_when_supplied(self):
        """u_px/v_px override the bbox-derived contact point."""
        centred = self._measure(3.0, u_px=CX, v_px=CY)
        self.assertTrue(centred.valid)
        # At the principal point the slant equals the axis depth exactly.
        self.assertAlmostEqual(centred.slant_range_m, centred.distance_m, places=9)

    def test_invalid_altitude_yields_invalid_measurement(self):
        """Requirement 6/10d: bad altitude -> invalid, no manufactured distance."""
        for bad in (None, -1.0, float("nan"), float("inf")):
            with self.subTest(altitude=bad):
                result = self._measure(bad)
                self.assertFalse(result.valid)
                self.assertIsNone(result.horizontal_distance_m)

    def test_altitude_above_slant_yields_invalid_measurement(self):
        """Requirement 10e: impossible geometry -> invalid."""
        result = self._measure(50.0)
        self.assertFalse(result.valid)
        self.assertIsNone(result.horizontal_distance_m)

    def test_invalid_bbox_yields_invalid_measurement(self):
        """A bad bbox fails in estimate() first, so nothing is derived."""
        for bad in (None, 0.0, -10.0, float("nan")):
            with self.subTest(bbox_height_px=bad):
                result = self._measure(3.0, bbox_height_px=bad)
                self.assertFalse(result.valid)
                self.assertIsNone(result.horizontal_distance_m)

    def test_missing_pixel_information_yields_invalid(self):
        """No u/v and no bbox geometry -> refuse rather than guess."""
        result = self._measure(3.0, bbox_left_px=None, bbox_bottom_px=None)
        self.assertFalse(result.valid)
        self.assertIsNone(result.horizontal_distance_m)


class TestExistingBehaviourPreserved(unittest.TestCase):
    """Requirement 11: estimate() semantics and the live follow path are unchanged."""

    def setUp(self):
        self.intrinsics = make_intrinsics()
        self.estimator = VisionDistanceEstimator(self.intrinsics, VisionConfig())

    def test_estimate_unchanged_and_horizontal_is_none(self):
        """estimate() still returns the optical-axis depth, with no horizontal value."""
        result = self.estimator.estimate(
            bbox_height_px=145.2,
            class_name="person",
            frame_w=640,
            frame_h=480,
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.distance_m, 1.3 * FY / 145.2, places=9)
        self.assertIsNone(result.horizontal_distance_m)
        self.assertIsNone(result.slant_range_m)
        self.assertIsNone(result.altitude_m)

    def test_invalid_measurement_has_no_horizontal_fields(self):
        """VisionMeasurement.invalid() leaves the new fields None."""
        result = VisionDistanceEstimator(self.intrinsics, VisionConfig()).estimate(
            bbox_height_px=0.0, class_name="person"
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.horizontal_distance_m)
        self.assertIsNone(result.slant_range_m)
        self.assertIsNone(result.altitude_m)

    def test_intrinsics_defaults_unchanged(self):
        """Requirement 13: fx/fy/cx/cy are byte-identical to before."""
        from modules.distance_estimator.calibration import DEFAULT_CONFIGURED_INTRINSICS

        self.assertAlmostEqual(DEFAULT_CONFIGURED_INTRINSICS["fx"], 446.7, places=9)
        self.assertAlmostEqual(DEFAULT_CONFIGURED_INTRINSICS["fy"], 446.7, places=9)
        self.assertAlmostEqual(DEFAULT_CONFIGURED_INTRINSICS["cx"], 320.0, places=9)
        self.assertAlmostEqual(DEFAULT_CONFIGURED_INTRINSICS["cy"], 240.0, places=9)

    def test_horizontal_geometry_not_computed_in_follow_controller(self):
        """person_follow reads Detection.horizontal_distance_m; it never derives H."""
        source = (
            pathlib.Path(__file__).resolve().parents[1] / "modules" / "person_follow.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("estimate_horizontal", source)
        self.assertNotIn("axis_depth_to_slant_range", source)
        self.assertNotIn("slant_to_horizontal_distance", source)

    def test_no_horizontal_wiring_in_mavlink_or_follow_movement(self):
        """Requirement 8: MAVLink/follow movement consume only the axis depth.

        The live app annotates the selected target with the observational
        horizontal value for the HUD, so the value may appear in the annotation
        and console helpers. It must never reach the follow-movement dict or the
        MAVLink/telemetry dispatch.
        """
        root = pathlib.Path(__file__).resolve().parents[1]
        for relative in (
            "modules/control_system/api.py",
            "modules/drone_backend/sitl.py",
        ):
            with self.subTest(path=relative):
                source = (root / relative).read_text(encoding="utf-8")
                self.assertNotIn("estimate_horizontal", source)
                self.assertNotIn("horizontal_distance_m", source)

        main_source = (root / "autonomous_drone_main.py").read_text(encoding="utf-8")
        self.assertNotIn("estimate_horizontal(", main_source)
        movement_block = main_source.split("def _follow_movement_dict", 1)[1]
        movement_block = movement_block.split("def _refresh_follow_estimator", 1)[0]
        self.assertNotIn("horizontal", movement_block)

    def test_follow_constants_unchanged(self):
        """Requirements 8/9: profile and floor unchanged; new setpoints added."""
        from modules import app_config

        self.assertAlmostEqual(app_config.FOLLOW_MIN_DISTANCE_M, 2.0, places=9)
        self.assertEqual(
            app_config.FOLLOW_SPEED_PROFILE,
            [(2.0, 0.0), (3.0, 0.3), (5.0, 0.6), (8.0, 1.0)],
        )
        self.assertAlmostEqual(app_config.FOLLOW_DISTANCE, 4.0, places=9)
        self.assertAlmostEqual(app_config.DISTANCE_TOLERANCE, 0.5, places=9)


if __name__ == "__main__":
    unittest.main()
