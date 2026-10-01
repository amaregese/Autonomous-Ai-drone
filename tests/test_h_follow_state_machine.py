"""
Canonical consolidated tests for the horizontal (H) follow state machine.

This merges the unique assertions previously spread across
``test_follow_distance_state.py``, ``test_forward_follow_horizontal.py``,
``test_forward_h_independent_of_axis.py`` and ``test_reverse_when_too_close.py``.

State table (FOLLOW_DISTANCE=4.0 m, DISTANCE_TOLERANCE=0.5 m):

    H < 3.5          -> TOO_CLOSE (reverse, bounded by FOLLOW_REVERSE_SPEED)
    3.5 <= H <= 4.5  -> HOLD      (zero forward target)
    H > 4.5          -> TOO_FAR   (existing forward-speed profile, evaluated on H)
    invalid H        -> INVALID   (zero forward target, no axis-Z fallback)

Run with:
    python -m pytest tests/ -q
    python -m unittest discover -s tests -t . -p "test_*.py"
"""
from __future__ import annotations

import importlib.util
import pathlib
import unittest

from modules import app_config
from modules.distance_estimator.calibration import CameraIntrinsics
from modules.distance_estimator.config import VisionConfig
from modules.distance_estimator.vision import VisionDistanceEstimator, annotate_detections
from modules.person_follow import (
    DistanceState,
    FollowConfig,
    PersonFollowController,
    classify_horizontal_distance,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]

_types_path = ROOT / "modules" / "yolo11_detector" / "types.py"
_spec = importlib.util.spec_from_file_location("yolo11_types_hsm", _types_path)
_types = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_types)
Detection = _types.Detection

REVERSE = app_config.FOLLOW_REVERSE_SPEED


def make_detection(horizontal_distance_m=None, distance_m=None, selected=True,
                   class_name="person", confidence=0.9, cx=320,
                   timestamp=None) -> object:
    """Selected detection with explicit H / axis-Z measurements.

    ``distance_source`` is set to ``"vision"`` whenever an axis-Z value is
    present so ``measure_range`` resolves Z deterministically; ``distance_m=None``
    therefore means "Z invalid" and never triggers the geometry fallback.
    """
    detection = Detection(cx - 25, 150, cx + 25, 295, class_id=0,
                          class_name=class_name, confidence=confidence,
                          timestamp=timestamp)
    detection.is_selected = selected
    detection.horizontal_distance_m = horizontal_distance_m
    detection.horizontal_distance_valid = horizontal_distance_m is not None
    detection.distance_m = distance_m
    detection.distance_valid = distance_m is not None
    detection.distance_source = "vision" if distance_m is not None else None
    detection.distance_confidence = 0.9 if distance_m is not None else 0.0
    return detection


class TestFollowDistanceConfig(unittest.TestCase):
    """Requirement 1: setpoints exist, are mirrored, and the old profile is intact."""

    def test_setpoint_values(self):
        self.assertAlmostEqual(app_config.FOLLOW_DISTANCE, 4.0, places=9)
        self.assertAlmostEqual(app_config.DISTANCE_TOLERANCE, 0.5, places=9)
        self.assertAlmostEqual(app_config.FOLLOW_REVERSE_SPEED, 0.3, places=9)

    def test_follow_config_carries_setpoints(self):
        config = FollowConfig()
        self.assertAlmostEqual(config.follow_distance, 4.0, places=9)
        self.assertAlmostEqual(config.distance_tolerance, 0.5, places=9)
        self.assertAlmostEqual(config.reverse_speed, REVERSE, places=9)
        snapshot = FollowConfig.from_app_config()
        self.assertAlmostEqual(snapshot.follow_distance, 4.0, places=9)
        self.assertAlmostEqual(snapshot.distance_tolerance, 0.5, places=9)
        self.assertAlmostEqual(snapshot.reverse_speed, REVERSE, places=9)

    def test_existing_profile_and_floor_unchanged(self):
        self.assertEqual(
            app_config.FOLLOW_SPEED_PROFILE,
            [(2.0, 0.0), (3.0, 0.3), (5.0, 0.6), (8.0, 1.0)],
        )
        self.assertAlmostEqual(app_config.FOLLOW_MIN_DISTANCE_M, 2.0, places=9)


class TestDeadbandBoundaries(unittest.TestCase):
    """Requirement 2 and 7: classification boundaries, incl. exact 3.5 / 4.5 m."""

    CASES = [
        (2.0, DistanceState.TOO_CLOSE),
        (3.49, DistanceState.TOO_CLOSE),
        (3.50, DistanceState.HOLD),
        (4.00, DistanceState.HOLD),
        (4.49, DistanceState.HOLD),
        (4.50, DistanceState.HOLD),
        (4.51, DistanceState.TOO_FAR),
        (6.00, DistanceState.TOO_FAR),
        (None, DistanceState.INVALID),
        (float("nan"), DistanceState.INVALID),
        (float("inf"), DistanceState.INVALID),
        (-1.0, DistanceState.INVALID),
    ]

    def test_classification_boundaries(self):
        for value, expected in self.CASES:
            with self.subTest(horizontal_distance_m=value):
                self.assertEqual(classify_horizontal_distance(value), expected)

    def test_exact_deadband_boundaries(self):
        self.assertEqual(classify_horizontal_distance(3.5), DistanceState.HOLD)
        self.assertEqual(classify_horizontal_distance(4.5), DistanceState.HOLD)
        self.assertEqual(classify_horizontal_distance(3.499999), DistanceState.TOO_CLOSE)
        self.assertEqual(classify_horizontal_distance(4.500001), DistanceState.TOO_FAR)

    def test_zero_and_non_numeric_are_invalid(self):
        for value in (0.0, "oops", object()):
            with self.subTest(value=value):
                self.assertEqual(classify_horizontal_distance(value), DistanceState.INVALID)

    def test_state_reads_horizontal_distance(self):
        controller = PersonFollowController(config=FollowConfig())
        for value, expected in (
            (2.0, DistanceState.TOO_CLOSE),
            (4.0, DistanceState.HOLD),
            (6.0, DistanceState.TOO_FAR),
            (None, DistanceState.INVALID),
        ):
            with self.subTest(horizontal_distance_m=value):
                detection = make_detection(horizontal_distance_m=value)
                self.assertEqual(controller.horizontal_distance_state(detection), expected)

    def test_optical_axis_distance_does_not_decide_state(self):
        controller = PersonFollowController(config=FollowConfig())

        far_axis_close_h = make_detection(horizontal_distance_m=2.0, distance_m=100.0)
        self.assertEqual(controller.horizontal_distance_state(far_axis_close_h),
                         DistanceState.TOO_CLOSE)

        close_axis_far_h = make_detection(horizontal_distance_m=6.0, distance_m=1.0)
        self.assertEqual(controller.horizontal_distance_state(close_axis_far_h),
                         DistanceState.TOO_FAR)

        axis_only = make_detection(horizontal_distance_m=None, distance_m=4.0)
        self.assertEqual(controller.horizontal_distance_state(axis_only),
                         DistanceState.INVALID)

    def test_horizontal_distance_validated(self):
        controller = PersonFollowController(config=FollowConfig())
        self.assertEqual(
            controller.horizontal_distance(make_detection(horizontal_distance_m=4.0)), 4.0)
        self.assertIsNone(
            controller.horizontal_distance(make_detection(horizontal_distance_m=-1.0)))
        self.assertIsNone(
            controller.horizontal_distance(make_detection(horizontal_distance_m=float("nan"))))
        self.assertIsNone(controller.horizontal_distance(None))


class TestForwardAndHoldTargets(unittest.TestCase):
    """Requirement 3: TOO_FAR profile, HOLD/INVALID zero, profile bounds."""

    ZERO_CASES = [3.50, 4.00, 4.50, None, float("nan"), float("inf"), -1.0]
    PROFILE_CASES = [4.51, 6.0]

    def setUp(self):
        self.controller = PersonFollowController(config=FollowConfig())

    def test_zero_forward_cases(self):
        for h in self.ZERO_CASES:
            with self.subTest(horizontal_distance_m=h):
                detection = make_detection(h)
                self.assertEqual(self.controller.forward_speed_target(detection), 0.0)
                cmd = self.controller.update(detection, (480, 640), now=detection.timestamp)
                self.assertEqual(cmd.vx, 0.0)

    def test_profile_used_when_too_far(self):
        for h in self.PROFILE_CASES:
            with self.subTest(horizontal_distance_m=h):
                detection = make_detection(h)
                self.assertEqual(classify_horizontal_distance(h), DistanceState.TOO_FAR)
                expected = self.controller._target_forward_speed(h)
                self.assertGreater(expected, 0.0)
                self.assertEqual(self.controller.forward_speed_target(detection), expected)

    def test_profile_engages_via_update(self):
        detection = make_detection(6.0)
        cmd = self.controller.update(detection, (480, 640), now=detection.timestamp)
        self.assertGreater(cmd.vx, 0.0)

    def test_max_forward_speed_preserved(self):
        detection = make_detection(100.0)
        self.assertEqual(self.controller.forward_speed_target(detection),
                         FollowConfig().max_speed)

    def test_forward_speed_profile_non_negative(self):
        controller = PersonFollowController(config=FollowConfig())
        for distance in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 20.0):
            with self.subTest(distance=distance):
                self.assertGreaterEqual(controller._target_forward_speed(distance), 0.0)

    def test_forward_speed_target_semantics(self):
        controller = PersonFollowController(config=FollowConfig())
        self.assertEqual(controller.forward_speed_target(make_detection(2.0, None)),
                         -controller.config.reverse_speed)
        self.assertEqual(controller.forward_speed_target(make_detection(4.5, None)), 0.0)
        self.assertEqual(controller.forward_speed_target(make_detection(None, 6.0)), 0.0)
        self.assertGreater(controller.forward_speed_target(make_detection(4.51, None)), 0.0)

    def test_forward_speed_target_bounded(self):
        controller = PersonFollowController(config=FollowConfig())
        floor = -FollowConfig().reverse_speed - 1e-9
        for h in (None, float("nan"), -5.0, 2.0, 4.0, 6.0, 100.0):
            with self.subTest(horizontal_distance_m=h):
                self.assertGreaterEqual(
                    controller.forward_speed_target(make_detection(h)), floor)

    def test_update_populates_state_and_gates_forward(self):
        def update_with(detection):
            controller = PersonFollowController(config=FollowConfig(),
                                                measure=lambda d, s: 3.0)
            return controller, controller.update(detection, (480, 640),
                                                 now=detection.timestamp)

        no_h = make_detection(horizontal_distance_m=None)
        far_h = make_detection(horizontal_distance_m=6.0)
        ctrl_none, cmd_none = update_with(no_h)
        ctrl_far, cmd_far = update_with(far_h)

        self.assertEqual(ctrl_none.distance_state, DistanceState.INVALID)
        self.assertEqual(ctrl_far.distance_state, DistanceState.TOO_FAR)
        self.assertEqual(cmd_none.vx, 0.0)
        self.assertGreater(cmd_far.vx, 0.0)
        self.assertAlmostEqual(cmd_none.vy, cmd_far.vy, places=12)
        self.assertAlmostEqual(cmd_none.yaw_cmd, cmd_far.yaw_cmd, places=12)
        self.assertEqual(cmd_none.range_m, cmd_far.range_m)


class TestReverseBehavior(unittest.TestCase):
    """Requirement 4: TOO_CLOSE reverse, bounded, ramped, no forward leakage."""

    def test_too_close_commands_fixed_reverse(self):
        for h in (2.0, 3.49):
            with self.subTest(horizontal_distance_m=h):
                controller = PersonFollowController(config=FollowConfig())
                detection = make_detection(h)
                self.assertEqual(controller.horizontal_distance_state(detection),
                                 DistanceState.TOO_CLOSE)
                self.assertEqual(controller.forward_speed_target(detection), -REVERSE)

    def test_hold_and_invalid_command_zero(self):
        for h in (3.50, 4.00, 4.50, None):
            with self.subTest(horizontal_distance_m=h):
                controller = PersonFollowController(config=FollowConfig())
                self.assertEqual(
                    controller.forward_speed_target(make_detection(h)), 0.0)

    def test_too_far_commands_forward(self):
        for h in (4.51, 6.00):
            with self.subTest(horizontal_distance_m=h):
                controller = PersonFollowController(config=FollowConfig())
                detection = make_detection(h)
                self.assertEqual(controller.horizontal_distance_state(detection),
                                 DistanceState.TOO_FAR)
                self.assertGreater(controller.forward_speed_target(detection), 0.0)

    def test_reverse_cases(self):
        controller = PersonFollowController(config=FollowConfig())
        for h in (2.0, 3.49):
            with self.subTest(horizontal_distance_m=h):
                detection = make_detection(h)
                self.assertEqual(controller.forward_speed_target(detection),
                                 -FollowConfig().reverse_speed)
                cmd = controller.update(detection, (480, 640), now=detection.timestamp)
                self.assertLess(cmd.vx, 0.0)

    def test_no_unintended_forward_while_too_close(self):
        controller = PersonFollowController(config=FollowConfig())
        for h in (1.0, 2.0, 3.49):
            with self.subTest(horizontal_distance_m=h):
                detection = make_detection(h)
                cmd = controller.update(detection, (480, 640), now=detection.timestamp)
                self.assertLessEqual(cmd.vx, 0.0)

    def test_reverse_only_changes_vx(self):
        controller = PersonFollowController(config=FollowConfig())
        close = make_detection(2.0, distance_m=None, cx=200)
        hold = make_detection(4.0, distance_m=None, cx=200)
        close_cmd = controller.update(close, (480, 640), now=close.timestamp)
        controller.reset()
        hold_cmd = controller.update(hold, (480, 640), now=hold.timestamp)
        self.assertLess(close_cmd.vx, hold_cmd.vx)
        self.assertAlmostEqual(close_cmd.vy, hold_cmd.vy, places=9)
        self.assertAlmostEqual(close_cmd.yaw_cmd, hold_cmd.yaw_cmd, places=9)
        self.assertEqual(close_cmd.range_m, hold_cmd.range_m)

    def test_ramp_slows_forward_before_reversing(self):
        controller = PersonFollowController(config=FollowConfig())
        base = 1_000_000.0
        far = make_detection(8.0, distance_m=8.0, timestamp=base)
        for _ in range(60):
            controller.update(far, (480, 640), now=base)
        self.assertGreater(controller._last_vx, 0.0)

        dt = 1.0 / 30.0
        close = make_detection(2.0, distance_m=2.0, timestamp=base + dt)
        prev_vx = controller._last_vx
        first = controller.update(close, (480, 640), now=base + dt)
        max_step = app_config.FOLLOW_ACCEL_LIMIT_MPS2 * dt
        self.assertLess(first.vx, prev_vx)
        self.assertGreaterEqual(first.vx, -REVERSE)
        self.assertAlmostEqual(first.vx, prev_vx - max_step, places=9)


class TestVxBounds(unittest.TestCase):
    """Requirement 3/4: the signed vx never drops below -FOLLOW_REVERSE_SPEED."""

    def test_vx_bounded_by_reverse_speed(self):
        floor = -FollowConfig().reverse_speed - 1e-9
        for h in (None, float("nan"), float("inf"), -1.0, 0.0, 1.0, 2.0, 3.49,
                  3.50, 4.0, 4.50, 4.51, 6.0, 50.0):
            with self.subTest(horizontal_distance_m=h):
                controller = PersonFollowController(config=FollowConfig())
                detection = make_detection(h)
                cmd = controller.update(detection, (480, 640), now=detection.timestamp)
                self.assertGreaterEqual(cmd.vx, floor)
                self.assertGreaterEqual(controller._last_vx, floor)

    def test_vx_bounded_for_h_z_pairs(self):
        floor = -FollowConfig().reverse_speed - 1e-9
        for h, z in ((None, None), (6.0, None), (2.0, None), (4.0, None),
                     (None, 6.0), (6.0, 6.0), (50.0, None)):
            with self.subTest(horizontal_distance_m=h, distance_m=z):
                controller = PersonFollowController(config=FollowConfig())
                detection = make_detection(h, z, timestamp=0.0)
                cmd = controller.update(detection, (480, 640), now=0.0)
                self.assertGreaterEqual(cmd.vx, floor)
                self.assertGreaterEqual(controller._last_vx, floor)


class TestHIndependentOfAxis(unittest.TestCase):
    """Requirement 5/6: valid H drives vx even when axis Z is invalid, and vice versa."""

    def setUp(self):
        self.controller = PersonFollowController(config=FollowConfig())

    def _update(self, detection):
        return self.controller.update(detection, (480, 640), now=detection.timestamp)

    def test_valid_h_invalid_z_too_far_reaches_forward(self):
        detection = make_detection(6.0, None)
        self.assertEqual(self.controller.horizontal_distance_state(detection),
                         DistanceState.TOO_FAR)
        self.assertGreater(self.controller.forward_speed_target(detection), 0.0)
        cmd = self._update(detection)
        self.assertEqual(self.controller.distance_state, DistanceState.TOO_FAR)
        self.assertGreater(cmd.vx, 0.0)
        self.assertNotEqual(cmd.lane, "no_range")
        self.assertNotEqual(cmd.reason, "no usable range measurement")
        self.assertIsNone(cmd.range_m)
        self.assertFalse(cmd.distance_valid)
        self.assertEqual(cmd.distance_source, "none")

    def test_valid_h_invalid_z_too_close_reverses(self):
        detection = make_detection(2.0, None)
        self.assertEqual(self.controller.horizontal_distance_state(detection),
                         DistanceState.TOO_CLOSE)
        self.assertEqual(self.controller.forward_speed_target(detection),
                         -self.controller.config.reverse_speed)
        cmd = self._update(detection)
        self.assertLess(cmd.vx, 0.0)
        self.assertGreaterEqual(cmd.vx, -self.controller.config.reverse_speed)
        self.assertIsNone(cmd.range_m)

    def test_valid_h_invalid_z_hold_stops(self):
        detection = make_detection(4.0, None)
        self.assertEqual(self.controller.horizontal_distance_state(detection),
                         DistanceState.HOLD)
        self.assertEqual(self.controller.forward_speed_target(detection), 0.0)
        cmd = self._update(detection)
        self.assertEqual(cmd.vx, 0.0)
        self.assertIsNone(cmd.range_m)

    def test_invalid_h_valid_z_does_not_fall_back_to_z(self):
        detection = make_detection(None, 6.0)
        self.assertEqual(self.controller.horizontal_distance_state(detection),
                         DistanceState.INVALID)
        self.assertEqual(self.controller.forward_speed_target(detection), 0.0)
        cmd = self._update(detection)
        self.assertEqual(self.controller.distance_state, DistanceState.INVALID)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.range_m, 6.0)
        self.assertTrue(cmd.distance_valid)

    def test_invalid_h_with_valid_z_stays_zero(self):
        detection = make_detection(None, 6.0)
        self.assertEqual(self.controller.horizontal_distance_state(detection),
                         DistanceState.INVALID)
        self.assertEqual(self.controller.forward_speed_target(detection), 0.0)
        cmd = self._update(detection)
        self.assertEqual(cmd.vx, 0.0)

    def test_both_valid_preserves_existing_profile(self):
        detection = make_detection(6.0, 6.0)
        self.assertEqual(self.controller.horizontal_distance_state(detection),
                         DistanceState.TOO_FAR)
        self.assertEqual(self.controller.forward_speed_target(detection),
                         self.controller._target_forward_speed(6.0))
        cmd = self._update(detection)
        self.assertGreater(cmd.vx, 0.0)
        self.assertEqual(cmd.range_m, 6.0)

    def test_far_h_with_close_axis_uses_profile(self):
        detection = make_detection(horizontal_distance_m=6.0, distance_m=2.0)
        self.assertEqual(self.controller.horizontal_distance_state(detection),
                         DistanceState.TOO_FAR)
        self.assertEqual(self.controller.forward_speed_target(detection),
                         self.controller._target_forward_speed(6.0))
        self.assertGreater(self.controller.forward_speed_target(detection), 0.0)

    def test_close_h_with_far_axis_moves_backward(self):
        detection = make_detection(horizontal_distance_m=2.0, distance_m=6.0)
        self.assertEqual(self.controller.horizontal_distance_state(detection),
                         DistanceState.TOO_CLOSE)
        self.assertEqual(self.controller.forward_speed_target(detection),
                         -self.controller.config.reverse_speed)

    def test_axis_range_field_preserved_for_diagnostics(self):
        detection = make_detection(horizontal_distance_m=6.0, distance_m=3.0)
        cmd = self._update(detection)
        self.assertGreater(cmd.vx, 0.0)
        self.assertEqual(cmd.range_m, 3.0)

    def test_command_range_field_never_gets_h(self):
        far_h_no_z = make_detection(6.0, None)
        cmd = self._update(far_h_no_z)
        self.assertIsNone(cmd.range_m)

        close_h_far_z = make_detection(2.0, 6.0)
        cmd = self._update(close_h_far_z)
        self.assertEqual(cmd.range_m, 6.0)

    def test_reverse_keeps_axis_range_diagnostic(self):
        detection = make_detection(2.0, 5.5)
        cmd = self._update(detection)
        self.assertLess(cmd.vx, 0.0)
        self.assertAlmostEqual(cmd.range_m, 5.5, places=9)

    def test_both_invalid_keeps_legacy_no_range_path(self):
        detection = make_detection(None, None)
        self.assertEqual(self.controller.horizontal_distance_state(detection),
                         DistanceState.INVALID)
        cmd = self._update(detection)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.lane, "no_range")
        self.assertIsNone(cmd.range_m)


class TestLateralAndYawUnchanged(unittest.TestCase):
    """Requirement 6: changing H must not change vy or yaw."""

    def test_vy_and_yaw_independent_of_h(self):
        results = []
        for h in (None, 2.0, 4.0, 6.0):
            controller = PersonFollowController(config=FollowConfig())
            detection = make_detection(h, distance_m=3.0)
            cmd = controller.update(detection, (480, 640), now=detection.timestamp)
            results.append((cmd.vy, cmd.yaw_cmd))
        for vy, yaw in results[1:]:
            self.assertAlmostEqual(vy, results[0][0], places=12)
            self.assertAlmostEqual(yaw, results[0][1], places=12)

    def test_vy_changes_only_with_geometry(self):
        controller = PersonFollowController(config=FollowConfig())
        detection = make_detection(6.0)
        detection.Center = (240, 220)
        cmd = controller.update(detection, (480, 640), now=detection.timestamp)
        self.assertLess(cmd.vy, 0.0)


class TestMovementLayerGuards(unittest.TestCase):
    """Requirement 6: movement/MAVLink/calibration layers are not modified."""

    def test_reverse_is_fixed_and_bounded(self):
        source = (ROOT / "modules" / "person_follow.py").read_text(encoding="utf-8")
        lowered = source.lower()
        self.assertIn("reverse_speed", lowered)
        self.assertNotIn("brake", lowered)

    def test_no_pitch_or_attitude_added(self):
        source = (ROOT / "modules" / "person_follow.py").read_text(encoding="utf-8")
        lowered = source.lower()
        self.assertNotIn("brake", lowered)
        self.assertNotIn("pitch", lowered)
        self.assertNotIn("attitude", lowered)
        self.assertNotIn("ATTITUDE", source)

    def test_controller_never_calls_estimator_for_horizontal(self):
        source = (ROOT / "modules" / "person_follow.py").read_text(encoding="utf-8")
        self.assertNotIn("estimate_horizontal", source)
        self.assertNotIn("axis_depth_to_slant_range", source)
        self.assertNotIn("slant_to_horizontal_distance", source)

    def test_movement_schema_unchanged(self):
        source = (ROOT / "autonomous_drone_main.py").read_text(encoding="utf-8")
        block = source.split("def _follow_movement_dict", 1)[1]
        block = block.split("def _refresh_follow_estimator", 1)[0]
        self.assertNotIn("follow_distance", block)
        self.assertNotIn("distance_tolerance", block)
        self.assertNotIn("distance_state", block)
        self.assertNotIn("forward_speed_target", block)

    def test_send_movement_command_unchanged(self):
        source = (ROOT / "modules" / "drone_backend" / "sitl.py").read_text(encoding="utf-8")
        self.assertIn("def send_movement_command_XYA", source)
        lowered = source.lower()
        self.assertNotIn("follow_distance", lowered)
        self.assertNotIn("distance_tolerance", lowered)
        self.assertNotIn("reverse", lowered)

    def test_mavlink_layer_unchanged(self):
        source = (ROOT / "modules" / "control_system" / "api.py").read_text(encoding="utf-8")
        lowered = source.lower()
        self.assertNotIn("follow_distance", lowered)
        self.assertNotIn("distance_tolerance", lowered)
        self.assertNotIn("reverse", lowered)

    def test_calibration_values_unchanged(self):
        from modules.distance_estimator.calibration import DEFAULT_CONFIGURED_INTRINSICS

        self.assertAlmostEqual(DEFAULT_CONFIGURED_INTRINSICS["fx"], 446.7, places=9)
        self.assertAlmostEqual(DEFAULT_CONFIGURED_INTRINSICS["fy"], 446.7, places=9)
        self.assertAlmostEqual(DEFAULT_CONFIGURED_INTRINSICS["cx"], 320.0, places=9)
        self.assertAlmostEqual(DEFAULT_CONFIGURED_INTRINSICS["cy"], 240.0, places=9)


class TestLiveDataFlowIntoTarget(unittest.TestCase):
    """Requirement 5/6: H reaches the controller through Detection only."""

    def test_annotated_detection_drives_state(self):
        estimator = VisionDistanceEstimator(
            CameraIntrinsics(fx=446.7, fy=446.7, cx=320.0, cy=240.0,
                             frame_w=640, frame_h=480),
            VisionConfig(),
        )
        detection = make_detection(horizontal_distance_m=None)
        annotate_detections(estimator, [detection], 640, 480,
                            altitude_m=3.0, horizontal_classes={"person"})
        self.assertTrue(detection.horizontal_distance_valid)
        measured_h = detection.horizontal_distance_m

        controller = PersonFollowController(config=FollowConfig())
        self.assertAlmostEqual(controller.horizontal_distance(detection), measured_h,
                               places=12)
        self.assertEqual(controller.horizontal_distance_state(detection),
                         classify_horizontal_distance(measured_h))


if __name__ == "__main__":
    unittest.main()
