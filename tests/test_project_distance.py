"""Task 6 tests: observe-only YOLO11 -> distance diagnostic tool.

These tests exercise ``tools/test_project_distance.py`` WITHOUT any physical
camera, YOLO model run or live window. Mock detections (``Detection`` objects)
are fed directly into the estimator/print helpers and the injectable live loop.

Verified here:

  * estimator math  d = class_height_m * fy / bbox_height_px  with fy = 446.7,
  * the exact Task 6 console line format
      ``person conf=0.86 bbox_height=182px distance=3.19m``,
  * unknown classes fall back to the default object height,
* invalid / missing bbox and a missing estimator produce ``None``,
    * intrinsics resolve from the benchmark manifest (configured_default, fy=446.7)
      and from an explicit ``--intrinsics`` JSON override,
    * the live loop runs headless on mock detections (no camera) with the correct
      output,
    * the built-in self-test passes headless,
    * importing the module loads no autonomous-flight component and no heavy
      dependency (torch/ultralytics/cv2 stay unloaded).

Task 7 additions ("distance stability"):

    * stability statistics (valid count / min / max / mean / std-dev) on known
      measurement series, including empty/single-sample edge cases,
    * the STABLE vs FLUCTUATING classification (relative std-dev + minimum
      sample gating),
    * JSON round-trip and CSV output of measurements + summary,
    * the ``run_stability`` live loop collects per-person samples headless,
      prints the summary and saves a results file,
    * frame-limit and duration-limit termination.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

_tpd_module = None


def _tpd():
    global _tpd_module
    if _tpd_module is None:
        import tools.test_project_distance as tpd  # noqa: PLC0415

        _tpd_module = tpd
    return _tpd_module


def _person(height_px=260.0, confidence=0.86):
    from modules.yolo11_detector.types import Detection  # noqa: PLC0415

    return Detection(300, 180, 460, 180 + height_px, 0, "person", confidence)


class TestEstimatorMath(unittest.TestCase):
    def setUp(self):
        tpd = _tpd()
        self.estimator = tpd.make_estimator(
            tpd.CameraIntrinsics.from_dict(dict(tpd.DEFAULT_INTRINSICS))
        )

    def test_person_distance_uses_pinhole_model(self):
        estimate = _tpd().estimate_detection(self.estimator, _person(260.0), 640, 480)
        self.assertIsNotNone(estimate)
        distance_m, confidence = estimate
        self.assertAlmostEqual(distance_m, 1.3 * 446.7 / 260.0, places=6)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_task_example_line_format(self):
        detection = _person(182.0)
        line = _tpd().detection_line(self.estimator, detection, 640, 480)
        self.assertEqual(line, "person conf=0.86 bbox_height=182px distance=3.19m")

    def test_unknown_class_uses_default_object_height(self):
        from modules.yolo11_detector.types import Detection  # noqa: PLC0415

        detection = Detection(40, 60, 240, 260, 80, "snowboard", 0.60)
        estimate = _tpd().estimate_detection(self.estimator, detection, 640, 480)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate[0], 0.5 * 446.7 / 200.0, places=6)

    def test_zero_height_bbox_is_invalid(self):
        from modules.yolo11_detector.types import Detection  # noqa: PLC0415

        detection = Detection(10, 50, 110, 50, 0, "person", 0.90)  # height 0
        self.assertIsNone(_tpd().estimate_detection(self.estimator, detection, 640, 480))

    def test_none_estimator_returns_none(self):
        tpd = _tpd()
        self.assertIsNone(tpd.estimate_detection(None, _person(260.0), 640, 480))
        self.assertIsNone(tpd.make_estimator(None))


class TestIntrinsicsResolution(unittest.TestCase):
    def test_configured_defaults_match_manifest(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        manifest = os.path.join(repo_root, "benchmarks", "distance", "manifest.json")
        intrinsics = _tpd().load_intrinsics(manifest)
        self.assertIsNotNone(intrinsics)
        self.assertAlmostEqual(intrinsics.fx, 446.7)
        self.assertAlmostEqual(intrinsics.fy, 446.7)
        self.assertAlmostEqual(intrinsics.cx, 320.0)
        self.assertAlmostEqual(intrinsics.cy, 240.0)

    def test_falls_back_to_documented_defaults_without_manifest(self):
        fake = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missing-manifest-xyz.json")
        intrinsics = _tpd().load_intrinsics(fake)
        self.assertIsNotNone(intrinsics)
        self.assertAlmostEqual(intrinsics.fy, 446.7)

    def test_explicit_json_override_wins(self):
        intrinsics = _tpd().load_intrinsics(
            intrinsics_json='{"fx":1000.0,"fy":1000.0,"cx":320.0,"cy":240.0}'
        )
        self.assertIsNotNone(intrinsics)
        self.assertAlmostEqual(intrinsics.fy, 1000.0)

    def test_invalid_json_returns_none(self):
        self.assertIsNone(_tpd().load_intrinsics(intrinsics_json="not json"))


class TestHeadlessLoop(unittest.TestCase):
    def test_run_loop_prints_distances_no_camera(self):
        tpd = _tpd()
        from modules.yolo11_detector.types import Detection  # noqa: PLC0415

        detections = [
            Detection(300, 180, 460, 440, 0, "person", 0.86),
            Detection(80, 120, 220, 220, 2, "car", 0.84),
        ]
        args = argparse.Namespace(
            manifest=None,
            intrinsics=None,
            intrinsics_source="configured_default",
            frames=0,
            no_window=True,
        )
        holder = {"n": 0}

        def acquire():
            if holder["n"] >= 1:
                return None
            holder["n"] += 1
            return detections, 30.0, np.zeros((480, 640, 3), dtype=np.uint8)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = tpd.run(args, acquire_fn=acquire, key_fn=lambda: ord("q"))
        output = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("person conf=0.86 bbox_height=260px distance=2.23m", output)
        self.assertIn("car conf=0.84 bbox_height=100px distance=6.70m", output)
        self.assertIn("Observe-only diagnostic. No flight control / MAVLink interaction.", output)

    def test_run_scales_configured_intrinsics_to_frame_height(self):
        tpd = _tpd()
        args = argparse.Namespace(
            manifest=None,
            intrinsics=None,
            intrinsics_source="configured_default",
            frames=1,
            no_window=True,
        )
        holder = {"n": 0}

        def acquire():
            if holder["n"] >= 1:
                return None
            holder["n"] += 1
            return [_person(180.0)], 30.0, np.zeros((360, 640, 3), dtype=np.uint8)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = tpd.run(args, acquire_fn=acquire, key_fn=lambda: None)
        self.assertEqual(rc, 0)
        self.assertIn("bbox_height=180px distance=2.42m", buf.getvalue())

    def test_explicit_intrinsics_without_frame_size_run_in_input_frame(self):
        tpd = _tpd()
        args = argparse.Namespace(
            manifest=None,
            intrinsics='{"fx":1000.0,"fy":1000.0,"cx":320.0,"cy":240.0}',
            intrinsics_source="cli",
            frames=1,
            no_window=True,
        )
        holder = {"n": 0}

        def acquire():
            if holder["n"] >= 1:
                return None
            holder["n"] += 1
            return [_person(180.0)], 30.0, np.zeros((480, 640, 3), dtype=np.uint8)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = tpd.run(args, acquire_fn=acquire, key_fn=lambda: None)
        self.assertEqual(rc, 0)
        self.assertIn("bbox_height=180px distance=7.22m", buf.getvalue())

    def test_frame_limit_stops_without_keys(self):
        tpd = _tpd()
        args = argparse.Namespace(
            manifest=None,
            intrinsics=None,
            intrinsics_source="configured_default",
            frames=2,
            no_window=True,
        )

        def acquire():
            if acquire.count >= 2:
                return None
            acquire.count += 1
            return [_person(260.0)], 30.0, np.zeros((480, 640, 3), dtype=np.uint8)

        acquire.count = 0
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = tpd.run(args, acquire_fn=acquire, key_fn=lambda: None)
        self.assertEqual(rc, 0)
        self.assertIn("Reached frame limit (2).", buf.getvalue())

    def test_self_test_passes_headless(self):
        with redirect_stdout(io.StringIO()):
            rc = _tpd().run_self_test()
        self.assertEqual(rc, 0)


class TestStabilityMetrics(unittest.TestCase):
    def test_known_series_statistics(self):
        distances = [1.81, 1.81, 1.84, 1.82]
        metrics = _tpd().compute_stability_metrics(distances)
        self.assertEqual(metrics["valid_count"], 4)
        self.assertAlmostEqual(metrics["minimum_m"], 1.81)
        self.assertAlmostEqual(metrics["maximum_m"], 1.84)
        self.assertAlmostEqual(metrics["mean_m"], statistics.fmean(distances))
        self.assertAlmostEqual(metrics["std_dev_m"], statistics.stdev(distances))

    def test_stable_stationary_series(self):
        metrics = _tpd().compute_stability_metrics([1.81, 1.82, 1.81, 1.83])
        self.assertTrue(metrics["stable"])
        self.assertLess(metrics["relative_std_percent"], 15.0)

    def test_fluctuating_series_not_stable(self):
        metrics = _tpd().compute_stability_metrics([1.5, 2.5, 1.5, 2.5])
        self.assertFalse(metrics["stable"])
        self.assertGreater(metrics["relative_std_percent"], 15.0)

    def test_min_samples_gating(self):
        metrics = _tpd().compute_stability_metrics([2.0, 2.0])
        self.assertEqual(metrics["valid_count"], 2)
        self.assertAlmostEqual(metrics["relative_std_percent"], 0.0)
        self.assertFalse(metrics["stable"])

    def test_empty_series(self):
        metrics = _tpd().compute_stability_metrics([])
        self.assertEqual(metrics["valid_count"], 0)
        self.assertIsNone(metrics["mean_m"])
        self.assertIsNone(metrics["std_dev_m"])
        self.assertFalse(metrics["stable"])

    def test_single_sample_std_is_zero(self):
        metrics = _tpd().compute_stability_metrics([3.2])
        self.assertEqual(metrics["valid_count"], 1)
        self.assertAlmostEqual(metrics["mean_m"], 3.2)
        self.assertAlmostEqual(metrics["std_dev_m"], 0.0)
        self.assertFalse(metrics["stable"])

    def test_custom_threshold_flips_verdict(self):
        tpd = _tpd()
        self.assertFalse(tpd.compute_stability_metrics([1.5, 2.5, 1.5, 2.5])["stable"])
        wide = tpd.compute_stability_metrics([1.5, 2.5, 1.5, 2.5], max_relative_std_percent=50.0)
        self.assertTrue(wide["stable"])


class TestStabilityWrites(unittest.TestCase):
    def setUp(self):
        self.tpd = _tpd()
        self.samples = [
            {"frame": 1, "confidence": 0.90, "bbox_height_px": 320.0, "distance_m": 1.815},
            {"frame": 2, "confidence": 0.91, "bbox_height_px": 316.0, "distance_m": 1.837},
            {"frame": 3, "confidence": 0.89, "bbox_height_px": 322.0, "distance_m": 1.801},
        ]
        self.metrics = self.tpd.compute_stability_metrics([s["distance_m"] for s in self.samples])

    def test_json_roundtrip(self):
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "stability_results.json")
            written = self.tpd.write_stability_json(path, self.samples, self.metrics, "configured_default")
            self.assertTrue(os.path.exists(written))
            with open(written, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["metrics"], self.metrics)
            self.assertEqual(data["samples"], self.samples)
            self.assertEqual(data["intrinsics_source"], "configured_default")

    def test_csv_write(self):
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stability_results.csv")
            written = self.tpd.write_stability_csv(path, self.samples, self.metrics, "configured_default")
            self.assertTrue(os.path.exists(written))
            with open(written, "r", encoding="utf-8", newline="") as f:
                rows = list(f.read().strip().splitlines())
            self.assertEqual(rows[0], "frame,confidence,bbox_height_px,distance_m")
            self.assertEqual(len(rows), 1 + len(self.samples))

    def test_write_dispatch_json_and_csv(self):
        with TemporaryDirectory() as tmp:
            json_path = self.tpd.write_stability(
                os.path.join(tmp, "a.json"), self.samples, self.metrics, "json"
            )
            csv_path = self.tpd.write_stability(
                os.path.join(tmp, "b.csv"), self.samples, self.metrics, "csv"
            )
            self.assertTrue(json_path.endswith(".json"))
            self.assertTrue(csv_path.endswith(".csv"))
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(csv_path))


class TestStabilityLoop(unittest.TestCase):
    def _stability_args(self, frames=0, duration=30.0, out=None, out_format="json"):
        return argparse.Namespace(
            manifest=None,
            intrinsics=None,
            intrinsics_source="configured_default",
            frames=frames,
            no_window=True,
            duration=duration,
            out=out,
            out_format=out_format,
            stability_max_relative_std=15.0,
            stability_min_samples=3,
        )

    def _person(self, height_px, confidence=0.90):
        from modules.yolo11_detector.types import Detection  # noqa: PLC0415

        return Detection(300, 180, 460, 180 + height_px, 0, "person", confidence)

    def test_collects_stationary_distances_and_saves(self):
        tpd = _tpd()
        with TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "stability.json")
            args = self._stability_args(frames=3, out=out)
            heights = [320.0, 316.0, 322.0]

            def acquire():
                if acquire.count >= 3:
                    return None
                i = acquire.count
                acquire.count += 1
                return (
                    [self._person(heights[i])],
                    30.0,
                    np.zeros((480, 640, 3), dtype=np.uint8),
                )

            acquire.count = 0
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = tpd.run_stability(args, acquire_fn=acquire, key_fn=lambda: None)
            output = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("Valid measurements: 3", output)
            self.assertIn("Mean distance:", output)
            self.assertIn("Std deviation:", output)
            self.assertIn("STABLE", output)
            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["metrics"]["valid_count"], 3)
            self.assertEqual(len(data["samples"]), 3)
            self.assertIn("distance_m", data["samples"][0])

    def test_frame_limit_terminates(self):
        tpd = _tpd()
        with TemporaryDirectory() as tmp:
            args = self._stability_args(frames=2, out=os.path.join(tmp, "s.json"))

            def acquire():
                if acquire.count >= 2:
                    return None
                acquire.count += 1
                return (
                    [self._person(320.0)],
                    30.0,
                    np.zeros((480, 640, 3), dtype=np.uint8),
                )

            acquire.count = 0
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = tpd.run_stability(args, acquire_fn=acquire, key_fn=lambda: None)
            output = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("Reached frame limit (2).", output)
            self.assertIn("Valid measurements: 2", output)

    def test_zero_person_frames_feeds_summary(self):
        tpd = _tpd()
        with TemporaryDirectory() as tmp:
            args = self._stability_args(frames=3, out=os.path.join(tmp, "s.json"))

            def acquire():
                if acquire.count >= 3:
                    return None
                acquire.count += 1
                return ([], 30.0, np.zeros((480, 640, 3), dtype=np.uint8))

            acquire.count = 0
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = tpd.run_stability(args, acquire_fn=acquire, key_fn=lambda: None)
            self.assertEqual(rc, 0)
            self.assertIn("No valid person measurements collected.", buf.getvalue())

    def test_duration_limit_terminates(self):
        tpd = _tpd()
        with TemporaryDirectory() as tmp:
            args = self._stability_args(frames=0, duration=0.05, out=os.path.join(tmp, "s.json"))
            high_frames = {"n": 0}

            def acquire():
                high_frames["n"] += 1
                return (
                    [self._person(320.0)],
                    60.0,
                    np.zeros((480, 640, 3), dtype=np.uint8),
                )

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = tpd.run_stability(args, acquire_fn=acquire, key_fn=lambda: None)
            self.assertEqual(rc, 0)
            self.assertIn("Reached duration limit", buf.getvalue())
            self.assertGreaterEqual(high_frames["n"], 1)

    def test_csv_loop_saves_measurements(self):
        tpd = _tpd()
        with TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "s.csv")
            args = self._stability_args(frames=2, out=out, out_format="csv")

            def acquire():
                if acquire.count >= 2:
                    return None
                acquire.count += 1
                return (
                    [self._person(320.0)],
                    30.0,
                    np.zeros((480, 640, 3), dtype=np.uint8),
                )

            acquire.count = 0
            with redirect_stdout(io.StringIO()):
                rc = tpd.run_stability(args, acquire_fn=acquire, key_fn=lambda: None)
            self.assertEqual(rc, 0)
            with open(out, "r", encoding="utf-8", newline="") as f:
                content = f.read()
            self.assertTrue(content.startswith("frame,confidence,bbox_height_px,distance_m"))


class TestNoFlightIntegration(unittest.TestCase):
    def test_module_imports_no_flight_components_and_no_heavy_deps(self):
        script = (
            "import sys;\n"
            "import tools.test_project_distance as tpd;\n"
            "top_forbidden = {'pymavlink', 'serial', 'autonomous_drone_main', 'jetson'};\n"
            "pkg_forbidden = ('modules.navigation', 'modules.lidar_backend',\n"
            "                 'modules.drone', 'modules.control');\n"
            "def is_forbidden(m):\n"
            "    if m.split('.')[0] in top_forbidden: return True\n"
            "    return any(m == p or m.startswith(p + '.') for p in pkg_forbidden)\n"
            "forbidden = sorted(m for m in sys.modules if is_forbidden(m));\n"
            "assert not forbidden, forbidden;\n"
            "for heavy in ('torch', 'ultralytics', 'cv2'):\n"
            "    assert heavy not in sys.modules, heavy;\n"
            "assert 'modules.distance_estimator' in sys.modules;\n"
            "print('clean')\n"
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


if __name__ == "__main__":
    unittest.main()