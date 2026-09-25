"""
Deterministic unit tests for the isolated distance estimator.

Run with:
    python -m unittest discover -s tests -t . -p "test_*.py"

No drone hardware, camera, network or GPU is required.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

from modules.distance_estimator import (
    CameraIntrinsics,
    DistanceEstimator,
    DistanceVelocityFilter,
    EstimatorConfig,
    FilterConfig,
    FixedLidar,
    LidarMeasurement,
    SimulatedLidar,
    Source,
    TargetState,
    VisionConfig,
    VisionMeasurement,
)
from modules.distance_estimator.calibration import CalibrationError, load_calibration
from modules.distance_estimator.vision import VisionDistanceEstimator


def make_intrinsics(fx: float = 1000.0, fy: float = 1000.0) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=320.0,
        cy=240.0,
        distortion={"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        frame_w=640,
        frame_h=480,
    )


class TestCalibration(unittest.TestCase):
    def test_valid_calibration_loads(self):
        intrinsics = make_intrinsics()
        self.assertTrue(intrinsics.is_valid(640, 480))
        self.assertEqual(intrinsics.fx, 1000.0)
        self.assertEqual(intrinsics.fy, 1000.0)

    def test_valid_calibration_round_trips_through_json(self):
        intrinsics = make_intrinsics()
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/calibration.json"
            intrinsics.save_to_json(path)
            loaded = CameraIntrinsics.from_json_file(path, frame_w=640, frame_h=480)
        self.assertEqual(loaded.fx, 1000.0)
        self.assertEqual(loaded.fy, 1000.0)
        self.assertEqual(loaded.cx, 320.0)
        self.assertEqual(loaded.cy, 240.0)
        self.assertTrue(loaded.is_valid())

    def test_invalid_focal_length_rejected(self):
        with self.assertRaises(CalibrationError):
            CameraIntrinsics.from_dict({"fx": 0.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0})

    def test_negative_focal_length_rejected(self):
        intrinsics = CameraIntrinsics(fx=-100.0, fy=1000.0, cx=320.0, cy=240.0)
        self.assertFalse(intrinsics.is_valid())

    def test_nonnumeric_fx_rejected(self):
        with self.assertRaises(CalibrationError):
            CameraIntrinsics.from_dict({"fx": "many", "fy": 1000.0, "cx": 320.0, "cy": 240.0})

    def test_missing_calibration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CalibrationError):
                CameraIntrinsics.from_json_file(f"{tmp}/does_not_exist.json")
            self.assertIsNone(load_calibration(f"{tmp}/does_not_exist.json"))

    def test_image_dimension_validation(self):
        intrinsics = make_intrinsics()
        self.assertTrue(intrinsics.is_valid(640, 480))
        self.assertFalse(intrinsics.is_valid(100, 100))

    def test_nonnumeric_distortion_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/calibration.json"
            with open(path, "w") as f:
                f.write('{"fx":1000,"fy":1000,"cx":320,"cy":240,"distortion":{"k1":"oops"}}')
            with self.assertRaises(CalibrationError):
                CameraIntrinsics.from_json_file(path)


class TestVision(unittest.TestCase):
    def setUp(self):
        self.est = VisionDistanceEstimator(make_intrinsics(), VisionConfig())

    def test_valid_bbox_produces_expected_distance(self):
        # 1.3 m person, fy=1000, 400 px tall -> 3.25 m
        meas = self.est.estimate(bbox_height_px=400.0, class_name="person", detection_confidence=1.0)
        self.assertTrue(meas.valid)
        self.assertAlmostEqual(meas.distance_m, 3.25, places=3)
        self.assertGreater(meas.confidence, 0.0)
        self.assertLessEqual(meas.confidence, 1.0)

    def test_zero_bbox_height_rejected(self):
        meas = self.est.estimate(bbox_height_px=0.0, class_name="person")
        self.assertFalse(meas.valid)
        self.assertEqual(meas.confidence, 0.0)
        self.assertIsNone(meas.distance_m)

    def test_negative_bbox_height_rejected(self):
        meas = self.est.estimate(bbox_height_px=-10.0, class_name="person")
        self.assertFalse(meas.valid)

    def test_none_bbox_height_rejected(self):
        meas = self.est.estimate(bbox_height_px=None, class_name="person")
        self.assertFalse(meas.valid)

    def test_invalid_focal_length_rejected(self):
        est = VisionDistanceEstimator(make_intrinsics(fy=0.0), VisionConfig())
        meas = est.estimate(bbox_height_px=400.0, class_name="person")
        self.assertFalse(est.geometry_available)
        self.assertFalse(meas.valid)

    def test_missing_calibration_rejected(self):
        est = VisionDistanceEstimator(None, VisionConfig())
        meas = est.estimate(bbox_height_px=400.0, class_name="person")
        self.assertFalse(est.geometry_available)
        self.assertFalse(meas.valid)

    def test_missing_object_height_rejected(self):
        config = VisionConfig(default_object_height_m=None)
        est = VisionDistanceEstimator(make_intrinsics(), config)
        meas = est.estimate(bbox_height_px=400.0, class_name="unknown_class")
        self.assertFalse(meas.valid)
        self.assertIsNone(meas.distance_m)

    def test_unreasonable_distance_rejected(self):
        # 60 px for a 1.3 m person with fy=1000 -> 21.7 m > max 20 m
        meas = self.est.estimate(bbox_height_px=60.0, class_name="person", detection_confidence=1.0)
        self.assertFalse(meas.valid)

    def test_poor_detection_confidence_reduces_confidence(self):
        good = self.est.estimate(bbox_height_px=400.0, class_name="person", detection_confidence=0.95)
        poor = self.est.estimate(bbox_height_px=400.0, class_name="person", detection_confidence=0.3)
        self.assertTrue(good.valid)
        self.assertTrue(poor.valid)
        self.assertLess(poor.confidence, good.confidence)

    def test_unknown_class_reduces_confidence(self):
        known = self.est.estimate(bbox_height_px=400.0, class_name="person", detection_confidence=1.0)
        unknown = self.est.estimate(bbox_height_px=400.0, class_name="not_in_table", detection_confidence=1.0)
        self.assertLess(unknown.confidence, known.confidence)

    def test_bbox_height_outside_reference_ramp_is_capped(self):
        large = self.est.estimate(bbox_height_px=5000.0, class_name="person", detection_confidence=1.0)
        self.assertTrue(large.valid)
        self.assertLessEqual(large.confidence, 1.0)


class TestLidar(unittest.TestCase):
    def test_valid_measurement_accepted(self):
        lidar = SimulatedLidar(values=[2.0], confidence=0.95)
        meas = lidar.read()
        self.assertTrue(meas.valid)
        self.assertEqual(meas.distance_m, 2.0)
        self.assertEqual(meas.confidence, 0.95)

    def test_zero_rejected(self):
        lidar = SimulatedLidar(values=[0.0])
        meas = lidar.read()
        self.assertFalse(meas.valid)
        self.assertIsNone(meas.distance_m)

    def test_negative_distance_rejected(self):
        lidar = SimulatedLidar(values=[-1.0])
        meas = lidar.read()
        self.assertFalse(meas.valid)

    def test_deterministic_sequence(self):
        lidar = SimulatedLidar(values=[1.0, 2.0, 3.0, 5.0])
        got = [lidar.read().distance_m for _ in range(4)]
        self.assertEqual(got, [1.0, 2.0, 3.0, 5.0])

    def test_fixed_lidar_always_returns_same(self):
        lidar = FixedLidar(3.0, confidence=0.9)
        for _ in range(5):
            meas = lidar.read()
            self.assertTrue(meas.valid)
            self.assertEqual(meas.distance_m, 3.0)


class TestFusion(unittest.TestCase):
    def _estimator(self, config=None):
        return DistanceEstimator(config=config or EstimatorConfig(), intrinsics=make_intrinsics())

    def _vision(self, distance_m, confidence=0.9):
        return VisionMeasurement(
            distance_m=distance_m,
            confidence=confidence,
            bbox_height_px=400.0,
            class_name="person",
            valid=True,
        )

    def _lidar(self, distance_m, confidence=0.95):
        return LidarMeasurement(distance_m=distance_m, confidence=confidence, valid=True)

    def test_lidar_and_vision_valid_fuse(self):
        est = self._estimator()
        state = est.update(self._vision(3.0), self._lidar(3.1), dt=0.05)
        self.assertTrue(state.valid)
        self.assertEqual(state.source, Source.FUSED)
        self.assertIsNotNone(state.fused_range_m)
        self.assertGreaterEqual(state.fused_range_m, 3.0)
        self.assertLessEqual(state.fused_range_m, 3.1)
        self.assertGreater(state.confidence, 0.8)

    def test_lidar_valid_vision_invalid_uses_lidar(self):
        est = self._estimator()
        state = est.update(
            VisionMeasurement(distance_m=None, confidence=0.0, valid=False),
            self._lidar(2.5),
            dt=0.05,
        )
        self.assertTrue(state.valid)
        self.assertEqual(state.source, Source.LIDAR)
        self.assertAlmostEqual(state.fused_range_m, 2.5, places=3)

    def test_lidar_invalid_vision_valid_uses_vision(self):
        est = self._estimator()
        state = est.update(self._vision(4.0), LidarMeasurement(valid=False), dt=0.05)
        self.assertTrue(state.valid)
        self.assertEqual(state.source, Source.VISION)
        self.assertAlmostEqual(state.fused_range_m, 4.0, places=3)

    def test_both_invalid_yields_invalid_state(self):
        est = self._estimator()
        state = est.update(VisionMeasurement(valid=False), LidarMeasurement(valid=False), dt=0.05)
        self.assertFalse(state.valid)
        self.assertEqual(state.source, Source.NONE)
        self.assertIsNone(state.range_m)

    def test_large_disagreement_not_averaged(self):
        est = self._estimator()
        state = est.update(self._vision(6.5), self._lidar(2.0), dt=0.05)
        self.assertTrue(state.valid)
        self.assertGreater(state.disagreement_m, 2.0)
        # picks the higher-confidence source (LiDAR 2.0), not the ~4.25 average
        self.assertIsNotNone(state.fused_range_m)
        self.assertLess(state.fused_range_m, 3.0)
        # disagreement reduces confidence vs the untouched source confidence
        self.assertLess(state.confidence, 0.95)

    def test_disagreement_flag_is_reported(self):
        est = self._estimator()
        state = est.update(self._vision(5.0), self._lidar(2.5), dt=0.05)
        self.assertGreater(state.disagreement_m, 2.0)


class TestFilter(unittest.TestCase):
    def _filter(self, measurement_variance=0.01, process_variance=0.01):
        config = FilterConfig(
            initial_distance_variance_m2=4.0,
            initial_velocity_variance_m2_s2=4.0,
            process_variance_m2_s3=process_variance,
            measurement_variance_m2=measurement_variance,
            minimum_measurement_variance_m2=0.001,
            maximum_measurement_variance_m2=4.0,
        )
        return DistanceVelocityFilter(config)

    def test_stable_3m_converges_near_3m(self):
        f = self._filter()
        for _ in range(80):
            f.predict(0.05)
            f.update(3.0, 0.01)
        self.assertLess(abs(f.distance - 3.0), 0.05)
        self.assertLess(abs(f.velocity), 0.1)

    def test_2_3_4_velocity_responds(self):
        f = self._filter()
        for z in (2.0, 3.0, 4.0):
            f.predict(1.0)
            f.update(z, 0.01)
        self.assertGreater(f.velocity, 0.3)
        self.assertGreater(f.distance, 2.0)
        self.assertLess(f.distance, 4.0)

    def test_invalid_measurement_prediction_continues(self):
        f = self._filter()
        f.update(3.0, 0.01)
        assert f.initialized
        before = f.distance
        before_conf = f.confidence
        f.update(None, 0.01)
        self.assertEqual(f.distance, before)
        self.assertEqual(f.confidence, before_conf)
        f.predict(1.0)
        self.assertGreaterEqual(f.distance, before)  # prediction propagated

    def test_negative_distance_clamped(self):
        f = self._filter()
        f.update(-2.0, 0.01)
        self.assertGreaterEqual(f.distance, 0.0)
        for _ in range(10):
            f.predict(0.05)
            f.update(-5.0, 0.01)
            self.assertGreaterEqual(f.distance, 0.0)

    def test_reset_returns_to_initial_state(self):
        f = self._filter()
        for _ in range(10):
            f.predict(0.05)
            f.update(3.0, 0.01)
        self.assertTrue(f.initialized)
        f.reset()
        self.assertFalse(f.initialized)
        self.assertEqual(f.distance, 0.0)
        self.assertEqual(f.velocity, 0.0)
        self.assertEqual(f.confidence, 0.0)

    def test_variable_dt(self):
        f = self._filter()
        # seed a positive velocity with 2 -> 4 measurements
        for z in (2.0, 4.0):
            f.predict(0.5)
            f.update(z, 0.01)
        f.predict(0.1)
        d_small = f.distance
        f.predict(2.5)
        d_large = f.distance
        self.assertGreater(d_large, d_small)


class TestEstimatorState(unittest.TestCase):
    def _estimator(self):
        return DistanceEstimator(config=EstimatorConfig(), intrinsics=make_intrinsics())

    def test_predicts_when_measurements_gap(self):
        est = self._estimator()
        lidar = LidarMeasurement(distance_m=3.0, confidence=0.95, valid=True)
        est.update(None, lidar, dt=0.05)
        self.assertTrue(est.update(None, None, dt=0.05).valid)
        state = est.update(None, None, dt=0.05)
        self.assertEqual(state.source, Source.PREDICTED)
        self.assertIsNotNone(state.range_m)

    def test_always_returns_state(self):
        est = self._estimator()
        state = est.update(None, None, dt=0.05)
        self.assertIsInstance(state, TargetState)

    def test_geometry_missing_reported(self):
        est = DistanceEstimator(config=EstimatorConfig(), intrinsics=None)
        self.assertFalse(est.vision_geometry_available)
        vision = est.estimate_vision(bbox_height_px=400.0, class_name="person")
        self.assertFalse(vision.valid)
        state = est.update(vision, None, dt=0.05)
        self.assertFalse(state.valid)

    def test_reset_clears_filter(self):
        est = self._estimator()
        est.update(_vision_meas(3.0), LidarMeasurement(valid=False), dt=0.05)
        est.reset()
        state = est.update(None, None, dt=0.05)
        self.assertFalse(state.valid)


class TestNoFlightSideEffects(unittest.TestCase):
    def test_import_does_not_load_hardware_or_network_modules(self):
        script = (
            "import sys; "
            "import modules.distance_estimator as de; "
            "forbidden = ['cv2','torch','ultralytics','pymavlink','serial','socket','pyserial']; "
            "loaded = sorted(m for m in sys.modules if m.split('.')[0] in forbidden); "
            "assert not loaded, loaded; "
            "assert 'modules.navigation' not in sys.modules; "
            "print('clean')"
        )
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        self.assertIn("clean", result.stdout)


def _vision_meas(distance_m: float, confidence: float = 0.9) -> VisionMeasurement:
    return VisionMeasurement(
        distance_m=distance_m,
        confidence=confidence,
        bbox_height_px=400.0,
        class_name="person",
        valid=True,
    )


if __name__ == "__main__":
    unittest.main()