"""
Task 2 unit tests: metric depth, depth confidence, three-source fusion, and
lazy backend loading. All deterministic; no neural network is run.

Run with:
    python -m unittest discover -s tests -t . -p "test_*.py"
"""
from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from modules.distance_estimator import (
    CameraIntrinsics,
    DepthConfig,
    DepthFrame,
    DepthMeasurement,
    DistanceEstimator,
    EstimatorConfig,
    LidarMeasurement,
    SimulatedDepthSource,
    SimulatedLidar,
    Source,
    VisionMeasurement,
    compute_depth_confidence,
    extract_depth_roi,
)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=1112.0, fy=1112.0, cx=320.0, cy=240.0, frame_w=640, frame_h=480)


def make_frame(depth_value=3.0, fill_all=True) -> DepthFrame:
    """Whole-frame constant metric depth (meters)."""
    h, w = 480, 640
    if fill_all:
        depth = np.full((h, w), float(depth_value), dtype=np.float64)
    else:
        depth = np.full((h, w), np.nan, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    depth[~valid] = np.nan
    return DepthFrame(depth, valid, "test-source")


BBOX = (140.0, 60.0, 500.0, 420.0)  # wide interior ROI


class TestDepthMeasurement(unittest.TestCase):
    def setUp(self):
        self.est = DistanceEstimator(
            config=EstimatorConfig(),
            intrinsics=intrinsics(),
            depth_source=SimulatedDepthSource(values=[3.0], confidence=0.9),
        )

    def test_valid_depth(self):
        meas = self.est.estimate_depth(FRAME, BBOX, detection_confidence=0.9)
        self.assertTrue(meas.valid)
        self.assertAlmostEqual(meas.depth_z_m, 3.0, places=5)
        self.assertIsNotNone(meas.range_m)
        self.assertGreater(meas.confidence, 0.0)

    def test_zero_depth_rejected(self):
        est = DistanceEstimator(
            config=EstimatorConfig(),
            intrinsics=intrinsics(),
            depth_source=SimulatedDepthSource(values=[0.0], confidence=0.9),
        )
        meas = est.estimate_depth(FRAME, BBOX, detection_confidence=0.9)
        self.assertFalse(meas.valid)
        self.assertEqual(meas.confidence, 0.0)

    def test_negative_depth_rejected(self):
        est = DistanceEstimator(
            config=EstimatorConfig(),
            intrinsics=intrinsics(),
            depth_source=SimulatedDepthSource(values=[-1.0], confidence=0.9),
        )
        meas = est.estimate_depth(FRAME, BBOX, detection_confidence=0.9)
        self.assertFalse(meas.valid)

    def test_nan_depth_rejected(self):
        est = DistanceEstimator(
            config=EstimatorConfig(),
            intrinsics=intrinsics(),
            depth_source=SimulatedDepthSource(values=[3.0], confidence=0.9),
        )
        est._depth_source.set_sequence([np.nan])
        meas = est.estimate_depth(FRAME, BBOX, detection_confidence=0.9)
        self.assertFalse(meas.valid)

    def test_infinity_depth_rejected(self):
        est = DistanceEstimator(
            config=EstimatorConfig(),
            intrinsics=intrinsics(),
            depth_source=SimulatedDepthSource(values=[np.inf], confidence=0.9),
        )
        est._depth_source.set_sequence([np.inf])
        meas = est.estimate_depth(FRAME, BBOX, detection_confidence=0.9)
        self.assertFalse(meas.valid)

    def test_empty_roi_rejected(self):
        frame = make_frame(3.0)
        stats = extract_depth_roi(frame, -50.0, -50.0, -10.0, -10.0)  # fully outside
        self.assertFalse(stats.valid)
        self.assertEqual(stats.roi_size_px, 0)

    def test_insufficient_valid_pixels_rejected(self):
        h, w = 480, 640
        depth = np.full((h, w), np.nan, dtype=np.float64)
        valid = np.zeros((h, w), dtype=bool)
        # a handful of valid pixels inside the enormous ROI -> below 50% coverage
        for x in range(250, 255):
            depth[200, x] = 3.0
            valid[200, x] = True
        stats = extract_depth_roi(DepthFrame(depth, valid, "t"), *BBOX)
        self.assertFalse(stats.valid)
        self.assertLess(stats.n_valid_pixels, 20)

    def test_range_is_not_invented_without_intrinsics(self):
        frame = make_frame(3.0)
        stats = extract_depth_roi(frame, *BBOX, intrinsics=None)
        self.assertTrue(stats.valid)
        self.assertIsNotNone(stats.z_median_m)
        self.assertIsNone(stats.range_median_m)

    def test_z_distinct_from_range_with_intrinsics(self):
        frame = make_frame(3.0)
        stats = extract_depth_roi(frame, *BBOX, intrinsics=intrinsics())
        self.assertTrue(stats.valid)
        self.assertIsNotNone(stats.range_median_m)
        self.assertAlmostEqual(stats.z_median_m, 3.0, places=5)


class TestDepthConfidence(unittest.TestCase):
    def test_high_quality_roi_better_than_poor_roi(self):
        cfg = DepthConfig()
        h, w = 480, 640
        stable = np.full((h, w), 3.0, dtype=np.float64)
        noisy = stable.copy()
        # deterministic high-variance interior: alternate 1 m / 5 m rows
        noisy[::2, :] = 1.0
        noisy[1::2, :] = 5.0
        valid = np.isfinite(stable) & (stable > 0)

        s1 = extract_depth_roi(
            DepthFrame(stable, valid, "t"), *BBOX, intrinsics=intrinsics()
        )
        s2 = extract_depth_roi(
            DepthFrame(noisy, valid, "t"), *BBOX, intrinsics=intrinsics()
        )
        c1 = compute_depth_confidence(s1, detection_confidence=0.9, config=cfg)
        c2 = compute_depth_confidence(s2, detection_confidence=0.9, config=cfg)
        self.assertGreater(c1, c2)

    def test_confidence_stays_in_unit_interval(self):
        cfg = DepthConfig()
        frame = make_frame(3.0)
        stats = extract_depth_roi(frame, *BBOX, intrinsics=intrinsics())
        for det_conf in (0.1, 0.5, 0.99):
            c = compute_depth_confidence(stats, detection_confidence=det_conf, config=cfg)
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_invalid_roi_has_zero_confidence(self):
        c = compute_depth_confidence(
            extract_depth_roi(make_frame(3.0), *BBOX, min_valid_fraction=1.5),
            detection_confidence=0.9,
            config=DepthConfig(),
        )
        self.assertEqual(c, 0.0)


class TestThreeSourceFusion(unittest.TestCase):
    def _est(self):
        return DistanceEstimator(config=EstimatorConfig(), intrinsics=intrinsics())

    def _vision(self, d, conf=0.9):
        return VisionMeasurement(distance_m=d, confidence=conf, bbox_height_px=400.0,
                                 class_name="person", valid=True)

    def _lidar(self, d, conf=0.95):
        return LidarMeasurement(distance_m=d, confidence=conf, valid=True)

    def _depth(self, d, conf=0.9):
        return DepthMeasurement(depth_z_m=d, range_m=d, confidence=conf, valid=True, source="t")

    def test_all_valid_fuse(self):
        est = self._est()
        state = est.update(self._vision(3.0), self._lidar(3.1), self._depth(3.0), dt=0.05)
        self.assertTrue(state.valid)
        self.assertEqual(state.source, Source.FUSED)
        self.assertIsNotNone(state.fused_range_m)
        self.assertGreaterEqual(state.fused_range_m, 2.95)
        self.assertLessEqual(state.fused_range_m, 3.15)

    def test_lidar_plus_depth_vision_invalid(self):
        est = self._est()
        state = est.update(
            VisionMeasurement(valid=False),
            self._lidar(2.5),
            self._depth(2.52),
            dt=0.05,
        )
        self.assertTrue(state.valid)
        self.assertEqual(state.source, Source.FUSED)
        self.assertAlmostEqual(state.fused_range_m, (2.5 + 2.52) / 2, delta=0.05)

    def test_vision_plus_depth_lidar_invalid(self):
        est = self._est()
        state = est.update(self._vision(4.0), LidarMeasurement(valid=False), self._depth(4.05), dt=0.05)
        self.assertTrue(state.valid)
        self.assertEqual(state.source, Source.FUSED)

    def test_lidar_only(self):
        est = self._est()
        state = est.update(
            VisionMeasurement(valid=False),
            self._lidar(3.2),
            DepthMeasurement(valid=False),
            dt=0.05,
        )
        self.assertEqual(state.source, Source.LIDAR)
        self.assertAlmostEqual(state.fused_range_m, 3.2, places=3)

    def test_vision_only(self):
        est = self._est()
        state = est.update(self._vision(3.4), LidarMeasurement(valid=False), DepthMeasurement(valid=False), dt=0.05)
        self.assertEqual(state.source, Source.VISION)

    def test_depth_only(self):
        est = self._est()
        state = est.update(
            VisionMeasurement(valid=False),
            LidarMeasurement(valid=False),
            self._depth(3.6),
            dt=0.05,
        )
        self.assertEqual(state.source, Source.DEPTH)
        self.assertAlmostEqual(state.fused_range_m, 3.6, places=3)

    def test_all_invalid(self):
        est = self._est()
        state = est.update(
            VisionMeasurement(valid=False),
            LidarMeasurement(valid=False),
            DepthMeasurement(valid=False),
            dt=0.05,
        )
        self.assertFalse(state.valid)
        self.assertEqual(state.source, Source.NONE)

    def test_disagreement_prefers_stronger_sources(self):
        est = self._est()
        # lidar+vision close, depth far off. Disagreement must be flagged,
        # the result must NOT be a blind average of all three.
        state = est.update(self._vision(3.0), self._lidar(3.05), self._depth(6.5), dt=0.05)
        self.assertTrue(state.valid)
        self.assertGreater(state.disagreement_m, 2.0)
        self.assertLess(state.fused_range_m, 4.5)

    def test_depth_source_runs_through_estimator(self):
        est = DistanceEstimator(
            config=EstimatorConfig(),
            intrinsics=intrinsics(),
            depth_source=SimulatedDepthSource(values=[3.25], confidence=0.9),
        )
        meas = est.estimate_depth(FRAME, BBOX, detection_confidence=0.9)
        self.assertTrue(meas.valid)
        self.assertAlmostEqual(meas.depth_z_m, 3.25, places=5)
        state = est.update(
            VisionMeasurement(valid=False),
            LidarMeasurement(valid=False),
            meas,
            dt=0.05,
        )
        self.assertEqual(state.source, Source.DEPTH)


class TestBackendLaziness(unittest.TestCase):
    def test_core_import_does_not_load_torch_or_transformers(self):
        import sys

        forbidden = {"torch", "transformers"}
        loaded = {m.split(".")[0] for m in sys.modules}
        self.assertTrue(forbidden.isdisjoint(loaded))

    def test_metric_depth_module_import_is_lazy(self):
        import sys

        from modules.distance_estimator.depth_backend.metric_depth import ZoeDepthBackend  # noqa: F401

        loaded = {m.split(".")[0] for m in sys.modules}
        self.assertNotIn("torch", loaded)
        self.assertNotIn("transformers", loaded)

    def test_zoe_backend_construction_is_cheap(self):
        from modules.distance_estimator.depth_backend.metric_depth import ZoeDepthBackend

        backend = ZoeDepthBackend()
        self.assertFalse(backend.loaded)
        self.assertIn("ZoeDepth", backend.name)

    def test_zoe_estimate_errors_cleanly_without_deps(self):
        from modules.distance_estimator.depth_backend.metric_depth import ZoeDepthBackend
        from modules.distance_estimator.depth import DepthBackendError

        if importlib.util.find_spec("transformers") is not None:
            self.skipTest("transformers installed; would trigger model download")
        with self.assertRaises(DepthBackendError):
            ZoeDepthBackend().estimate(FRAME)


class TestSimulatedDepthDeterminism(unittest.TestCase):
    def test_sequence_is_replayed(self):
        source = SimulatedDepthSource(values=[1.0, 2.0, 3.0, 5.0])
        got = [float(source.estimate(FRAME).depth_m[0, 0]) for _ in range(4)]
        self.assertEqual(got, [1.0, 2.0, 3.0, 5.0])

    def test_set_distance_is_fixed(self):
        source = SimulatedDepthSource()
        source.set_distance(4.0)
        for _ in range(3):
            self.assertEqual(float(source.estimate(FRAME).depth_m[0, 0]), 4.0)


if __name__ == "__main__":
    unittest.main()