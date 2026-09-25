"""
Task 3 tests: real-world dataset + benchmark tooling.

These tests verify manifest parsing/validation, metric math, evaluation
behavior (missing images, unavailable backends, empty datasets) and result
serialization WITHOUT requiring the real-world image dataset to exist. All
datasets used here are temporary directories and tiny generated images.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unittest

from modules.distance_estimator.analysis import (
    compute_metrics,
    confidence_vs_error,
    distance_bin_membership,
    flag_outliers,
    group_by_distance_bin,
    pearson_correlation,
)
from modules.distance_estimator.dataset import (
    DatasetError,
    create_empty_dataset,
    load_manifest,
)
from modules.distance_estimator.evaluation import EvaluationOptions, RealImageEvaluator
from modules.distance_estimator.report import write_all_results


def _write_manifest(dirpath, samples, camera=None):
    camera = camera or {
        "name": "testcam",
        "frame_w": 640,
        "frame_h": 480,
        "intrinsics_source": "calibrated",
        "intrinsics": {"fx": 1112.0, "fy": 1112.0, "cx": 320.0, "cy": 240.0},
    }
    manifest = {"version": 1, "camera": camera, "samples": samples}
    path = os.path.join(dirpath, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def _valid_sample(image="images/a.jpg"):
    return {
        "id": "sample_001",
        "image": image,
        "ground_truth_distance_m": 3.0,
        "class_name": "person",
        "measurement_method": "tape_measure",
        "bbox": {"x": 100, "y": 80, "width": 120, "height": 300},
    }


class TestManifestParsing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_valid_manifest_loads(self):
        path = _write_manifest(self._tmp, [_valid_sample()])
        dataset = load_manifest(path)
        self.assertEqual(dataset.version, 1)
        self.assertEqual(len(dataset.samples), 1)
        self.assertEqual(dataset.samples[0].id, "sample_001")
        self.assertEqual(dataset.samples[0].ground_truth_distance_m, 3.0)
        self.assertEqual(dataset.samples[0].bbox.width, 120)
        self.assertEqual(dataset.samples[0].measurement_method, "tape_measure")

    def test_missing_manifest_file(self):
        with self.assertRaises(DatasetError):
            load_manifest(os.path.join(self._tmp, "nope.json"))

    def test_empty_samples_and_zero_gt_are_valid(self):
        path = _write_manifest(self._tmp, [])
        dataset = load_manifest(path)
        self.assertEqual(dataset.samples, [])

    def test_required_fields_enforced(self):
        for missing in ("id", "image", "ground_truth_distance_m", "class_name", "measurement_method"):
            sample = _valid_sample()
            del sample[missing]
            path = _write_manifest(self._tmp, [sample])
            with self.assertRaises(DatasetError, msg=missing):
                load_manifest(path)

    def test_invalid_ground_truth_rejected(self):
        for bad in (0.0, -1.0, 1e-8, float("inf"), "abc", None):
            sample = _valid_sample()
            sample["ground_truth_distance_m"] = bad
            path = _write_manifest(self._tmp, [sample])
            with self.assertRaises(DatasetError, msg=str(bad)):
                load_manifest(path)

    def test_duplicate_ids_rejected(self):
        path = _write_manifest(self._tmp, [_valid_sample(), _valid_sample()])
        with self.assertRaises(DatasetError):
            load_manifest(path)

    def test_empty_id_rejected(self):
        sample = _valid_sample()
        sample["id"] = ""
        path = _write_manifest(self._tmp, [sample])
        with self.assertRaises(DatasetError):
            load_manifest(path)

    def test_invalid_bbox_rejected(self):
        for bad_bbox in (
            {"x": 10, "y": 10, "width": 0, "height": 50},
            {"x": 10, "y": 10, "width": -5, "height": 50},
            {"x": -1, "y": 10, "width": 20, "height": 50},
            {"x": 10, "y": 10},  # missing keys
        ):
            sample = _valid_sample()
            sample["bbox"] = bad_bbox
            path = _write_manifest(self._tmp, [sample])
            with self.assertRaises(DatasetError, msg=str(bad_bbox)):
                load_manifest(path)

    def test_detection_confidence_out_of_range_rejected(self):
        for bad in (0.0, 1.5, -0.1):
            sample = _valid_sample()
            sample["detection_confidence"] = bad
            path = _write_manifest(self._tmp, [sample])
            with self.assertRaises(DatasetError, msg=str(bad)):
                load_manifest(path)

    def test_camera_intrinsics_source_validated(self):
        camera = {
            "name": "c",
            "frame_w": 640,
            "frame_h": 480,
            "intrinsics_source": "unknown",
            "intrinsics": {"fx": 1112.0, "fy": 1112.0, "cx": 320.0, "cy": 240.0},
        }
        path = _write_manifest(self._tmp, [], camera=camera)
        dataset = load_manifest(path)
        self.assertTrue(dataset.camera_available)
        self.assertEqual(dataset.camera.intrinsics["fx"], 1112.0)


class TestMetricCalculations(unittest.TestCase):
    def test_known_metrics(self):
        pairs = [(3.0, 3.2), (5.0, 4.9), (2.0, 2.0)]
        summary = compute_metrics(pairs)
        self.assertEqual(summary.n, 3)
        self.assertAlmostEqual(summary.mae, (0.2 + 0.1 + 0.0) / 3.0)
        self.assertAlmostEqual(summary.bias, (0.2 - 0.1 + 0.0) / 3.0)
        self.assertAlmostEqual(summary.max_ae, 0.2)
        self.assertAlmostEqual(summary.median_ae, 0.1)
        expected_rmse = math.sqrt((0.2**2 + 0.1**2 + 0.0**2) / 3.0)
        self.assertAlmostEqual(summary.rmse, expected_rmse)
        self.assertAlmostEqual(summary.mre_pct, (0.2 / 3.0 + 0.1 / 5.0 + 0.0) / 3.0 * 100.0)

    def test_zero_ground_truth_never_used(self):
        summary = compute_metrics([(0.0, 3.0), (-1.0, 3.0), (4.0, 3.5), (None, 3.0)])
        self.assertEqual(summary.n, 1)
        self.assertAlmostEqual(summary.mae, 0.5)

    def test_no_pairs_is_empty(self):
        summary = compute_metrics([])
        self.assertEqual(summary.n, 0)
        self.assertFalse(summary.available())
        self.assertTrue(math.isnan(summary.mae))

    def test_distance_bin_membership(self):
        self.assertEqual(distance_bin_membership(1.0), "0-2 m")
        self.assertEqual(distance_bin_membership(3.0), "2-4 m")
        self.assertEqual(distance_bin_membership(9.0), "8-10 m")
        self.assertEqual(distance_bin_membership(50.0), "10+ m")

    def test_group_by_distance_bin_only_populated(self):
        records = [
            {"ground_truth_m": 1.0, "prediction_m": 1.1},
            {"ground_truth_m": 3.0, "prediction_m": 3.0},
        ]
        grouped = group_by_distance_bin(records)
        self.assertIn("0-2 m", grouped)
        self.assertIn("2-4 m", grouped)
        self.assertEqual(sum(len(v) for v in grouped.values()), 2)

    def test_pearson_correlation(self):
        xs = [1, 2, 3, 4, 5]
        ys = [2, 4, 6, 8, 10]
        self.assertAlmostEqual(pearson_correlation(xs, ys), 1.0, places=6)
        self.assertIsNone(pearson_correlation([1, 2], [1, 2]))  # too few samples

    def test_flag_outliers(self):
        records = [
            {"ground_truth_m": 3.0, "prediction_m": 3.1},
            {"ground_truth_m": 3.0, "prediction_m": 5.0},
        ]
        flagged, rule = flag_outliers(records, threshold_m=1.0)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["prediction_m"], 5.0)
        self.assertIn("1", rule)

    def test_confidence_vs_error_insufficient(self):
        result = confidence_vs_error([{"confidence": 0.9, "prediction_m": 3.1, "ground_truth_m": 3.0}])
        self.assertEqual(result["n"], 1)
        self.assertIsNone(result["pearson_r"])


class TestEvaluationBehavior(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._images = os.path.join(self._tmp, "images")
        os.makedirs(self._images, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_sample(self, image, gt, sample_id="s1"):
        return {
            "id": sample_id,
            "image": os.path.join("images", image),
            "ground_truth_distance_m": gt,
            "class_name": "person",
            "measurement_method": "tape_measure",
            "bbox": {"x": 100, "y": 100, "width": 120, "height": 280},
        }

    def _write_png(self, name, w=640, h=480):
        path = os.path.join(self._images, name)
        try:
            import cv2

            import numpy as np

            cv2.imwrite(path, np.zeros((h, w, 3), dtype=np.uint8))
        except Exception:
            try:
                from PIL import Image

                import numpy as np

                Image.fromarray(np.zeros((h, w, 3), dtype="uint8")).save(path)
            except Exception:
                return False
        return True

    def test_empty_dataset_does_not_crash(self):
        path = _write_manifest(self._tmp, [])
        dataset = load_manifest(path)
        payload = RealImageEvaluator(dataset).run()
        self.assertEqual(payload["dataset"]["total_samples"], 0)
        self.assertEqual(payload["dataset"]["usable_samples"], 0)
        self.assertEqual(payload["dataset"]["rejected_samples"], 0)

    def test_missing_image_handled(self):
        path = _write_manifest(self._tmp, [self._make_sample("missing.jpg", 3.0)])
        dataset = load_manifest(path)
        payload = RealImageEvaluator(dataset).run()
        self.assertEqual(payload["dataset"]["total_samples"], 1)
        self.assertEqual(payload["dataset"]["usable_samples"], 0)
        self.assertEqual(payload["dataset"]["rejected_samples"], 1)
        self.assertEqual(payload["dataset"]["rejections"][0]["reason"], "missing image")

    def test_missing_bbox_handled(self):
        sample = self._make_sample("a.jpg", 3.0)
        if not self._write_png("a.jpg"):
            self.skipTest("no image writer available")
        del sample["bbox"]
        path = _write_manifest(self._tmp, [sample])
        dataset = load_manifest(path)
        self.assertIsNone(dataset.samples[0].bbox)
        payload = RealImageEvaluator(dataset).run()
        self.assertEqual(payload["dataset"]["rejected_samples"], 1)
        self.assertEqual(payload["dataset"]["rejections"][0]["reason"], "no bounding box")

    def test_unavailable_backend_reported_honestly(self):
        if not self._write_png("a.jpg"):
            self.skipTest("no image writer available")
        path = _write_manifest(self._tmp, [self._make_sample("a.jpg", 3.0)])
        dataset = load_manifest(path)
        options = EvaluationOptions(enable_depth=False)
        payload = RealImageEvaluator(dataset, options).run()
        self.assertFalse(payload["methods"]["depth"]["available"])
        self.assertIn("depth backend disabled", payload["methods"]["depth"]["reason"])

    def test_no_calibration_marks_geometric_unavailable(self):
        if not self._write_png("a.jpg"):
            self.skipTest("no image writer available")
        camera = {
            "name": "uncalibrated",
            "frame_w": None,
            "frame_h": None,
            "intrinsics_source": "unknown",
            "intrinsics": None,
        }
        path = _write_manifest(self._tmp, [self._make_sample("a.jpg", 3.0)], camera=camera)
        dataset = load_manifest(path)
        payload = RealImageEvaluator(dataset).run()
        self.assertFalse(payload["camera"]["intrinsics_present"])
        self.assertFalse(payload["methods"]["geometric"]["available"])
        self.assertFalse(payload["methods"]["legacy"]["available"])

    def test_full_pipeline_with_generated_image(self):
        if not self._write_png("a.jpg"):
            self.skipTest("no image writer available")
        path = _write_manifest(self._tmp, [self._make_sample("a.jpg", 3.0)])
        dataset = load_manifest(path)
        payload = RealImageEvaluator(dataset).run()
        self.assertEqual(payload["dataset"]["usable_samples"], 1)
        for method in ("legacy", "geometric"):
            self.assertEqual(len(payload["methods"][method]["records"]), 1)
        # fusion uses vision when depth is disabled
        self.assertEqual(len(payload["methods"]["fusion"]["records"]), 1)
        self.assertEqual(payload["methods"]["fusion"]["metrics"]["n"], 1)

    def test_create_empty_dataset_writes_manifest(self):
        path = os.path.join(self._tmp, "created", "manifest.json")
        dataset = create_empty_dataset(path, camera_name="c", intrinsics_source="unknown")
        self.assertTrue(os.path.exists(path))
        reloaded = load_manifest(path)
        self.assertEqual(reloaded.camera.name, "c")
        self.assertEqual(dataset.camera.intrinsics_source, "unknown")


class TestResultFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._images = os.path.join(self._tmp, "images")
        os.makedirs(self._images, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _sample(self, image, gt, sample_id="s1"):
        return {
            "id": sample_id,
            "image": os.path.join("images", image),
            "ground_truth_distance_m": gt,
            "class_name": "person",
            "measurement_method": "tape_measure",
            "bbox": {"x": 100, "y": 100, "width": 120, "height": 280},
            "scene": "outdoor",
            "lighting": "daylight",
            "pose": "standing",
            "notes": "test",
        }

    def test_json_csv_report_generated(self):
        try:
            import cv2  # noqa: F401

            import numpy as np  # noqa: F401

            cv2.imwrite(os.path.join(self._images, "a.jpg"), np.zeros((480, 640, 3), dtype=np.uint8))
        except Exception:
            self.skipTest("no image writer available")

        path = _write_manifest(self._tmp, [self._sample("a.jpg", 3.0)])
        dataset = load_manifest(path)
        payload = RealImageEvaluator(dataset).run()
        results_dir = os.path.join(self._tmp, "results")
        paths = write_all_results(payload, results_dir)

        for key in ("json", "csv", "report"):
            self.assertTrue(os.path.exists(paths[key]), key)

        with open(paths["csv"], newline="") as f:
            import csv as _csv

            rows = list(_csv.DictReader(f))
        self.assertEqual(rows[0]["sample_id"], "s1")
        self.assertEqual(rows[0]["method"], "legacy")
        self.assertIn(rows[0]["ground_truth_m"], ("3.0", "3"))
        self.assertTrue(all(r["method"] in ("legacy", "geometric", "fusion", "filtered") for r in rows))

        with open(paths["json"]) as f:
            data = json.load(f)
        self.assertEqual(data["dataset"]["total_samples"], 1)
        self.assertIn("methods", data)

        with open(paths["report"]) as f:
            text = f.read()
        self.assertIn("REAL-WORLD", text.upper())
        self.assertIn("Connected to autonomous flight: NO", text)


if __name__ == "__main__":
    unittest.main()