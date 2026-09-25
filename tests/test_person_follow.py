"""
Task 9: unit tests for the autonomous person-follow controller.

Run with:
    python -m unittest discover -s tests -t . -p "test_*.py"

The tests isolate the follow-control logic. Distance sensing is injected so the
speed profile / lateral steering / ramp / loss behaviour are deterministic;
integration with the real ``DistanceEstimator`` is covered by a dedicated test.
"""
from __future__ import annotations

import unittest

from modules.distance_estimator import (
    CameraIntrinsics,
    DistanceEstimator,
    EstimatorConfig,
)
from modules.person_follow import FollowConfig, PersonFollowController
from modules.yolo11_detector.types import Detection

SHAPE = (480, 640)


def make_det(
    center_x=320,
    top=120,
    bottom=480,
    conf=0.90,
    cls="person",
    timestamp=0.0,
    selected=True,
):
    half_w = 50
    det = Detection(
        int(center_x - half_w),
        top,
        int(center_x + half_w),
        bottom,
        0,
        cls,
        conf,
        timestamp=timestamp,
    )
    det.is_selected = selected
    return det


def run_offline(ctrl, det, frames=200, dt=1.0 / 30.0, start_time=0.0):
    """Drive updates with synthetic time so ramping settles deterministically."""
    now = start_time
    cmd = None
    for _ in range(frames):
        now += dt
        det.timestamp = now - 0.001
        cmd = ctrl.update(det, SHAPE, now=now)
    return cmd


def run_lost(ctrl, frames=400, dt=1.0 / 30.0, start_time=0.0):
    now = start_time
    cmd = None
    for _ in range(frames):
        now += dt
        cmd = ctrl.update(None, SHAPE, now=now)
    return cmd


def make_intrinsics(fx=1000.0, fy=1000.0) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=320.0,
        cy=240.0,
        distortion={"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        frame_w=640,
        frame_h=480,
    )


class TestHoverFollowSpeed(unittest.TestCase):
    """Tests 1-4: distance governs forward velocity."""

    def _run_with_distance(self, distance_m, frames=200):
        ctrl = PersonFollowController(measure=lambda det, shape: distance_m)
        return run_offline(ctrl, make_det())

    def test_close_target_hovers(self):
        """Distance <= 2.0 m -> zero forward + zero lateral."""
        cmd = self._run_with_distance(1.5)
        self.assertAlmostEqual(cmd.vx, 0.0, places=9)
        self.assertAlmostEqual(cmd.vy, 0.0, places=9)
        self.assertTrue(cmd.active)

    def test_minimum_distance_boundary_hovers_even_with_minimum_speed(self):
        """The exact safety boundary cannot be overridden by min cruise speed."""
        ctrl = PersonFollowController(
            config=FollowConfig(min_speed=0.2), measure=lambda det, shape: 2.0
        )
        cmd = run_offline(ctrl, make_det())
        self.assertEqual(cmd.vx, 0.0)

    def test_far_target_follows(self):
        """Distance > 2.0 m -> positive forward velocity (0.3 m/s at 2.5 m)."""
        cmd = self._run_with_distance(2.5)
        self.assertGreater(cmd.vx, 0.0)
        self.assertAlmostEqual(cmd.vx, 0.3, places=2)

    def test_farther_target_higher_speed(self):
        """4.0 m -> 0.6 m/s; 2.5 m -> 0.3 m/s."""
        near = self._run_with_distance(2.5)
        far = self._run_with_distance(4.0)
        self.assertAlmostEqual(far.vx, 0.6, places=2)
        self.assertGreater(far.vx, near.vx)

    def test_speed_capped_at_maximum(self):
        """Beyond the last profile breakpoint the speed is capped, never exceeded."""
        ctrl = PersonFollowController(measure=lambda det, shape: 20.0)
        cmd = run_offline(ctrl, make_det())
        self.assertAlmostEqual(cmd.vx, ctrl.config.max_speed, places=2)
        self.assertLessEqual(cmd.vx, ctrl.config.max_speed + 1e-9)


class TestLateralCentring(unittest.TestCase):
    """Tests 5-7: bbox horizontal centre steers laterally, centred = no lat."""

    def _lateral_steady(self, center_x):
        ctrl = PersonFollowController(measure=lambda det, shape: 4.0)
        det = make_det(center_x=center_x)
        cmd = run_offline(ctrl, det)
        return cmd

    def test_target_left_moves_left(self):
        cmd = self._lateral_steady(center_x=160)
        self.assertLess(cmd.vy, -0.1)

    def test_target_right_moves_right(self):
        cmd = self._lateral_steady(center_x=480)
        self.assertGreater(cmd.vy, 0.1)

    def test_centred_target_no_lateral(self):
        cmd = self._lateral_steady(center_x=320)
        self.assertAlmostEqual(cmd.vy, 0.0, places=6)


class TestTargetLossBehaviour(unittest.TestCase):
    """Tests 8-11: gating on confidence, loss and staleness."""

    def test_low_confidence_no_movement(self):
        ctrl = PersonFollowController(measure=lambda det, shape: 4.0)
        det = make_det(conf=0.3)
        now = 1.0
        det.timestamp = now - 0.001
        cmd = ctrl.update(det, SHAPE, now=now)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.vy, 0.0)
        self.assertIn("low confidence", cmd.reason)

    def test_lost_target_commands_hover(self):
        ctrl = PersonFollowController(measure=lambda det, shape: 2.5)
        cmd = ctrl.update(None, SHAPE, now=1.0)
        self.assertTrue(cmd.lost)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.vy, 0.0)

    def test_prolonged_loss_reaches_safe_behavior(self):
        ctrl = PersonFollowController(measure=lambda det, shape: 2.5)
        cmd = run_lost(ctrl, frames=400)
        self.assertGreaterEqual(cmd.lost_time_s, ctrl.config.loss_timeout)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.vy, 0.0)
        self.assertEqual(ctrl.loss_action(), "hover")
        self.assertTrue(ctrl.loss_timeout_reached(now=400.0 / 30.0))

    def test_rtl_option_is_configured_safe_behavior(self):
        ctrl = PersonFollowController(
            config=FollowConfig(rtl_on_loss=True), measure=lambda det, shape: 2.5
        )
        self.assertEqual(ctrl.loss_action(), "rtl")

    def test_stale_detection_no_movement(self):
        ctrl = PersonFollowController(measure=lambda det, shape: 4.0)
        det = make_det(timestamp=0.0)
        cmd = ctrl.update(det, SHAPE, now=5.0)
        self.assertFalse(cmd.active)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.vy, 0.0)
        self.assertIn("stale", cmd.reason)


class TestRampingAndSafety(unittest.TestCase):
    """Tests 12-13: ramp limits and the minimum safety distance."""

    def test_command_smoothing_ramp_limit(self):
        ctrl = PersonFollowController(measure=lambda det, shape: 20.0)
        det = make_det()
        now = 0.0
        dt = 1.0 / 30.0
        prev = None
        first_step = None
        for _ in range(10):
            now += dt
            det.timestamp = now - 0.001
            cmd = ctrl.update(det, SHAPE, now=now)
            if prev is not None:
                delta = abs(cmd.vx - prev)
                self.assertLessEqual(delta, ctrl.config.accel_limit * dt + 1e-9)
            else:
                first_step = cmd.vx
            prev = cmd.vx
        self.assertLess(first_step, 0.05)
        self.assertLess(prev, 0.5)  # still ramping well below the 1.0 m/s target

    def test_safety_distance_not_violated(self):
        """Forward velocity is zero while inside the configured minimum distance."""
        ctrl = PersonFollowController(
            config=FollowConfig(min_distance=3.0),
            measure=lambda det, shape: 2.5,
        )
        cmd = run_offline(ctrl, make_det())
        self.assertEqual(cmd.vx, 0.0)


class TestConfigGates(unittest.TestCase):
    """Follow enable/disable and target-class gating."""

    def test_follow_disabled_no_movement(self):
        ctrl = PersonFollowController(
            config=FollowConfig(enabled=False), measure=lambda det, shape: 4.0
        )
        cmd = ctrl.update(make_det(), SHAPE, now=1.0)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.vy, 0.0)
        self.assertIn("disabled", cmd.reason)

    def test_wrong_target_class_no_movement(self):
        ctrl = PersonFollowController(measure=lambda det, shape: 4.0)
        det = make_det(cls="car")
        cmd = ctrl.update(det, SHAPE, now=1.0)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.vy, 0.0)
        self.assertIn("wrong class", cmd.reason)

    def test_not_selected_detection_no_movement(self):
        ctrl = PersonFollowController(measure=lambda det, shape: 4.0)
        det = make_det(selected=False)
        cmd = ctrl.update(det, SHAPE, now=1.0)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.vy, 0.0)
        self.assertIn("not selected", cmd.reason)


class TestDistanceEstimatorIntegration(unittest.TestCase):
    """The controller reuses the existing DistanceEstimator end to end."""

    def test_real_estimator_drives_speed(self):
        estimator = DistanceEstimator(
            config=EstimatorConfig(), intrinsics=make_intrinsics()
        )
        ctrl = PersonFollowController(distance_estimator=estimator)
        # fy=1000, person 1.3 m tall -> 4.0 m needs a 325 px tall bbox
        det = make_det(top=10, bottom=335)
        cmd = run_offline(ctrl, det)
        self.assertIsNotNone(cmd.range_m)
        self.assertAlmostEqual(cmd.range_m, 4.0, delta=0.35)
        self.assertAlmostEqual(cmd.vx, 0.6, places=1)


if __name__ == "__main__":
    unittest.main()
