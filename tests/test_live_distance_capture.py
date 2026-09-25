"""Task 4 tests: live YOLO11 + manual ground-truth distance capture tool.

These tests exercise the capture tool WITHOUT any physical camera or live
window. The live loop runs against a deterministic in-memory fake camera, fake
frames and scripted keyboard/console input.

The capture module is imported lazily inside the test methods on purpose: it
imports ultralytics/torch (the existing YOLO11 pipeline), and the Task 2
backend-laziness tests assert that torch is not loaded in the shared discovery
process.

Verified here:

  * valid ground-truth parsing (and that it matches the Task 3 minimum rule),
  * invalid ground-truth rejection (empty, non-numeric, NaN, inf, 0, negative,
    below-minimum),
  * selected-detection metadata extraction,
  * sample creation (image + manifest entry) with fake frames,
  * schema compatibility (the produced manifest reloads with load_manifest),
  * unique image names / unique sample ids,
  * no sample saved when ground truth is missing or invalid,
  * ground truth is never overridden by any estimate,
  * the module does not import any autonomous-flight component.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from modules.distance_estimator.dataset import MIN_GT_DISTANCE_M, load_manifest

_ldc_module = None


def _ldc():
    global _ldc_module
    if _ldc_module is None:
        import tools.live_distance_capture as ldc  # noqa: PLC0415

        _ldc_module = ldc
    return _ldc_module


def _fake_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (300, 180), (460, 440), (0, 128, 255), -1)
    return frame


def _fake_detections():
    from modules.yolo11_detector.types import Detection  # noqa: PLC0415

    return [
        Detection(300, 180, 460, 440, 0, "person", 0.91),
        Detection(80, 120, 220, 220, 2, "car", 0.84),
    ]


def _make_args(tmpdir, ldc, **overrides):
    base = dict(
        source="0",
        model_path=ldc.DEFAULT_MODEL,
        manifest=os.path.join(tmpdir, "manifest.json"),
        camera_name="live-camera",
        intrinsics_source="unknown",
        intrinsics=None,
        method="manual_measurement",
        scene="live_camera",
        lighting="unknown",
        pose="unknown",
        notes="",
        burst=1,
        conf=None,
        no_window=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestGroundTruthParsing(unittest.TestCase):
    def test_valid_values_parse(self):
        ldc = _ldc()
        self.assertEqual(ldc.parse_ground_truth("3.2"), 3.2)
        self.assertEqual(ldc.parse_ground_truth("  4 "), 4.0)
        self.assertEqual(ldc.parse_ground_truth("0.25"), MIN_GT_DISTANCE_M)

    def test_value_never_transformed(self):
        self.assertEqual(_ldc().parse_ground_truth("3.147"), 3.147)

    def test_empty_is_cancel(self):
        ldc = _ldc()
        self.assertIsNone(ldc.parse_ground_truth(""))
        self.assertIsNone(ldc.parse_ground_truth("   "))
        self.assertIsNone(ldc.parse_ground_truth(None))

    def test_non_numeric_rejected(self):
        ldc = _ldc()
        for bad in ("abc", "3.2 m", "12,5"):
            with self.assertRaises(ValueError):
                ldc.parse_ground_truth(bad)

    def test_nan_infinity_rejected(self):
        ldc = _ldc()
        for bad in ("nan", "NaN", "inf", "infinity", "-inf"):
            with self.assertRaises(ValueError):
                ldc.parse_ground_truth(bad)

    def test_zero_negative_rejected(self):
        ldc = _ldc()
        for bad in ("0", "0.0", "-1", "-0.25", "0.249"):
            with self.assertRaises(ValueError):
                ldc.parse_ground_truth(bad)

    def test_minimum_matches_task3_rule(self):
        self.assertAlmostEqual(MIN_GT_DISTANCE_M, 0.25)


class TestDetectionMetadata(unittest.TestCase):
    def test_detection_to_dict(self):
        data = _ldc().detection_to_dict(_fake_detections()[0])
        self.assertEqual(data["class_name"], "person")
        self.assertEqual(data["confidence"], 0.91)
        self.assertEqual(
            data["bbox"], {"x": 300.0, "y": 180.0, "width": 160.0, "height": 260.0}
        )
        self.assertEqual(data["center"], (380, 310))

    def test_selected_info_lines_include_height(self):
        lines = _ldc().selected_info_lines(_fake_detections()[0])
        text = "\n".join(lines)
        self.assertIn("class = person", text)
        self.assertIn("confidence = 0.910", text)
        self.assertIn("h=260", text)
        self.assertIn("bbox_height_px = 260", text)

    def test_no_auto_selection_with_multiple(self):
        self.assertEqual(len(_fake_detections()), 2)


class TestUniqueNames(unittest.TestCase):
    def test_sequential_image_names_do_not_collide(self):
        ldc = _ldc()
        self.assertNotEqual(
            ldc.next_image_name("does-not-matter"),
            ldc.next_image_name("does-not-matter"),
        )

    def test_sample_ids_increment(self):
        ldc = _ldc()
        self.assertEqual(ldc.next_sample_id({"live_001", "live_003"}), "live_004")
        self.assertEqual(ldc.next_sample_id(set()), "live_001")


class TestSourceParsing(unittest.TestCase):
    def test_camera_index(self):
        self.assertEqual(_ldc().parse_source("0"), ("camera", 0))

    def test_video_file(self):
        self.assertEqual(
            _ldc().parse_source("videos/run.mp4"), ("video", "videos/run.mp4")
        )

    def test_rtsp_url(self):
        kind, value = _ldc().parse_source("rtsp://cam/stream")
        self.assertEqual(kind, "rtsp")


class TestSampleCreation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_sample_created_and_schema_compatible(self):
        ldc = _ldc()
        recorder = ldc.SampleRecorder(os.path.join(self._tmp, "manifest.json"))
        recorder.set_frame_dims(640, 480)
        info = ldc.detection_to_dict(_fake_detections()[0])
        sid, rel, abs_path = recorder.save_sample(
            _fake_frame(),
            info,
            3.2,
            "manual_measurement",
            "live_camera",
            "unknown",
            "unknown",
            "note",
        )
        self.assertTrue(os.path.exists(abs_path))
        self.assertEqual(sid, "live_001")
        self.assertTrue(rel.startswith("images/"))

        dataset = load_manifest(recorder.manifest_path)
        self.assertEqual(len(dataset.samples), 1)
        sample = dataset.samples[0]
        self.assertEqual(sample.ground_truth_distance_m, 3.2)
        self.assertEqual(sample.detection_confidence, 0.91)
        self.assertEqual(sample.class_name, "person")
        self.assertEqual(sample.measurement_method, "manual_measurement")
        self.assertIsNotNone(sample.bbox)
        self.assertEqual(sample.bbox.to_dict(), info["bbox"])
        self.assertEqual(sample.scene, "live_camera")
        self.assertEqual(sample.lighting, "unknown")
        self.assertEqual(sample.pose, "unknown")
        self.assertEqual(sample.notes, "note")

    def test_two_samples_unique(self):
        ldc = _ldc()
        recorder = ldc.SampleRecorder(os.path.join(self._tmp, "manifest.json"))
        for det in _fake_detections():
            recorder.save_sample(
                _fake_frame(),
                ldc.detection_to_dict(det),
                3.0,
                "manual_measurement",
                "live_camera",
                "unknown",
                "unknown",
                "",
            )
        dataset = load_manifest(recorder.manifest_path)
        self.assertEqual(len(dataset.samples), 2)
        self.assertEqual(len(set(s.id for s in dataset.samples)), 2)
        images = [
            os.path.join(recorder.images_dir, os.path.basename(s.image))
            for s in dataset.samples
        ]
        self.assertEqual(len(set(images)), 2)
        for path in images:
            self.assertTrue(os.path.exists(path))

    def test_ground_truth_never_overridden(self):
        ldc = _ldc()
        recorder = ldc.SampleRecorder(os.path.join(self._tmp, "manifest.json"))
        info = ldc.detection_to_dict(_fake_detections()[0])
        recorder.save_sample(
            _fake_frame(),
            info,
            2.75,
            "manual_measurement",
            "live_camera",
            "unknown",
            "unknown",
            "",
        )
        dataset = load_manifest(recorder.manifest_path)
        self.assertEqual(dataset.samples[0].ground_truth_distance_m, 2.75)

    def test_no_sample_when_gt_missing_or_invalid(self):
        ldc = _ldc()
        recorder = ldc.SampleRecorder(os.path.join(self._tmp, "manifest.json"))
        info = ldc.detection_to_dict(_fake_detections()[0])
        for bad_gt in (0.0, -1.0, float("nan"), float("inf"), 0.24):
            with self.assertRaises(ValueError):
                recorder.save_sample(
                    _fake_frame(),
                    info,
                    bad_gt,
                    "manual_measurement",
                    "live_camera",
                    "unknown",
                    "unknown",
                    "",
                )
        dataset = load_manifest(recorder.manifest_path)
        self.assertEqual(len(dataset.samples), 0)
        self.assertEqual(os.listdir(recorder.images_dir), [])

    def test_intrinsics_source_recorded(self):
        ldc = _ldc()
        recorder = ldc.SampleRecorder(
            os.path.join(self._tmp, "manifest.json"),
            intrinsics_source="unknown",
            intrinsics=None,
        )
        dataset = load_manifest(recorder.manifest_path)
        self.assertEqual(dataset.camera.intrinsics_source, "unknown")
        self.assertFalse(dataset.camera_available)


class TestOptionalEstimatorOverlay(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_estimate_line_uncalibrated_is_honest(self):
        ldc = _ldc()
        line = ldc.estimate_line(None, _fake_detections()[0], 640, 480)
        self.assertIn("NOT AVAILABLE", line)
        self.assertIn("uncalibrated", line)

    def test_estimator_from_intrinsics(self):
        ldc = _ldc()
        estimator = ldc.make_geometric_estimator(
            '{"fx":1112,"fy":1112,"cx":320,"cy":240}'
        )
        self.assertIsNotNone(estimator)
        self.assertTrue(estimator.geometry_available)
        det = _fake_detections()[0]
        estimated = ldc.estimate_distance_m(estimator, det, 640, 480)
        self.assertAlmostEqual(estimated, 1.30 * 1112.0 / 260.0, places=2)

    def test_computed_estimate_never_reaches_ground_truth(self):
        ldc = _ldc()
        recorder = ldc.SampleRecorder(os.path.join(self._tmp, "manifest.json"))
        info = ldc.detection_to_dict(_fake_detections()[0])
        recorder.save_sample(
            _fake_frame(),
            info,
            3.0,
            "manual_measurement",
            "live_camera",
            "unknown",
            "unknown",
            "",
        )
        dataset = load_manifest(recorder.manifest_path)
        self.assertEqual(dataset.samples[0].ground_truth_distance_m, 3.0)


class TestFullCaptureFlow(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_capture_flow_with_invalid_and_cancel(self):
        ldc = _ldc()
        manifest = os.path.join(self._tmp, "manifest.json")
        frame = _fake_frame()
        keys = iter([ord("0"), ord("c"), ord("c"), ord("c"), ord("c"), ord("q")])
        prompts = iter(["3.2", "2.1", "bad", "", "unused"])

        args = _make_args(self._tmp, ldc)
        with redirect_stdout(io.StringIO()) as buf:
            rc = ldc.run(
                args,
                key_fn=lambda: next(keys, None),
                prompt_fn=lambda _p: next(prompts, None),
                open_fn=lambda _v: ldc.FakeCamera(frame),
                detect_fn=lambda _f: _fake_detections(),
            )
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        dataset = load_manifest(manifest)
        self.assertEqual(len(dataset.samples), 2)
        truths = sorted(s.ground_truth_distance_m for s in dataset.samples)
        self.assertEqual(truths, [2.1, 3.2])
        images = os.listdir(os.path.join(self._tmp, "images"))
        self.assertEqual(len(images), 2)
        self.assertTrue(all(f.endswith(".jpg") for f in images))
        self.assertIn("Invalid distance. Sample was NOT saved", out)
        self.assertIn("Sample not saved.", out)

    def test_capture_requires_selection(self):
        ldc = _ldc()
        manifest = os.path.join(self._tmp, "manifest.json")
        keys = iter([ord("c"), ord("q")])
        prompts = iter(["9.0"])
        args = _make_args(self._tmp, ldc)
        with redirect_stdout(io.StringIO()) as buf:
            rc = ldc.run(
                args,
                key_fn=lambda: next(keys, None),
                prompt_fn=lambda _p: next(prompts, None),
                open_fn=lambda _v: ldc.FakeCamera(_fake_frame()),
                detect_fn=lambda _f: _fake_detections(),
            )
        self.assertEqual(rc, 0)
        dataset = load_manifest(manifest)
        self.assertEqual(len(dataset.samples), 0)
        self.assertIn("No target selected", buf.getvalue())

    def test_burst_keeps_same_ground_truth(self):
        ldc = _ldc()
        manifest = os.path.join(self._tmp, "manifest.json")
        args = _make_args(self._tmp, ldc, burst=3)
        keys = iter([ord("0"), ord("c"), ord("q")])
        prompts = iter(["4.0"])
        with redirect_stdout(io.StringIO()):
            rc = ldc.run(
                args,
                key_fn=lambda: next(keys, None),
                prompt_fn=lambda _p: next(prompts, None),
                open_fn=lambda _v: ldc.FakeCamera(_fake_frame()),
                detect_fn=lambda _f: _fake_detections(),
            )
        self.assertEqual(rc, 0)
        dataset = load_manifest(manifest)
        self.assertEqual(len(dataset.samples), 3)
        truths = {s.ground_truth_distance_m for s in dataset.samples}
        self.assertEqual(truths, {4.0})


class TestDetectorWiring(unittest.TestCase):
    def test_load_detector_wires_model_into_canonical_api_module(self):
        import unittest.mock as mock

        import modules.yolo11_detector.api as api

        ldc = _ldc()
        fake_model = object()
        fake_classes = {"0": "person"}
        original_model, original_classes = api.model, api.classes
        try:
            with mock.patch(
                "modules.yolo11_detector.model.load_model",
                return_value=(fake_model, fake_classes),
            ):
                self.assertTrue(ldc.load_detector("ignored.pt"))
            self.assertIs(api.model, fake_model)
            self.assertIs(api.classes, fake_classes)
        finally:
            api.model, api.classes = original_model, original_classes


class TestNoFlightIntegration(unittest.TestCase):
    def test_module_imports_no_flight_components(self):
        script = (
            "import sys;\n"
            "import tools.live_distance_capture as ldc;\n"
            "top_forbidden = {'pymavlink', 'serial', 'autonomous_drone_main', 'jetson'};\n"
            "pkg_forbidden = ('modules.navigation', 'modules.lidar_backend',\n"
            "'modules.drone', 'modules.control');\n"
            "def is_forbidden(m):\n"
            "    if m.split('.')[0] in top_forbidden: return True\n"
            "    return any(m == p or m.startswith(p + '.') for p in pkg_forbidden)\n"
            "loaded = sorted(m for m in sys.modules if is_forbidden(m));\n"
            "assert not loaded, loaded;\n"
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

    def test_self_test_passes_headless(self):
        with redirect_stdout(io.StringIO()):
            rc = _ldc().run_self_test()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()