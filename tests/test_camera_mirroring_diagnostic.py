"""Coordinate-only tests for the read-only camera mirroring diagnostic."""
import unittest

import numpy as np

from modules.yolo11_detector.source import normalize_horizontal_frame
from modules.yolo11_detector.types import Detection
from tools.camera_mirroring_diagnostic import build_parser, horizontal_report


def detection_at(center_x):
    return Detection(center_x - 10, 10, center_x + 10, 50, 0, "person", 0.9)


class TestHorizontalCoordinates(unittest.TestCase):
    def test_normalization_flips_horizontally_without_mutating_source(self):
        source = np.array(
            [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]],
            dtype=np.uint8,
        )
        original = source.copy()
        normalized = normalize_horizontal_frame(source)
        np.testing.assert_array_equal(normalized, np.array(
            [[[3, 3, 3], [2, 2, 2], [1, 1, 1]], [[6, 6, 6], [5, 5, 5], [4, 4, 4]]],
            dtype=np.uint8,
        ))
        np.testing.assert_array_equal(source, original)

    def test_normalization_does_not_flip_vertically(self):
        source = np.array(
            [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.uint8
        )
        normalized = normalize_horizontal_frame(source)
        self.assertEqual(normalized[0, 0, 0], 2)
        self.assertEqual(normalized[1, 0, 0], 4)

    def test_physical_image_left_has_negative_error(self):
        report = horizontal_report(detection_at(160), 640)
        self.assertEqual(report.position, "LEFT")
        self.assertLess(report.horizontal_error, 0.0)
        self.assertFalse(report.yolo_frame_mirrored)

    def test_center_has_zero_error(self):
        report = horizontal_report(detection_at(320), 640)
        self.assertEqual(report.position, "CENTER")
        self.assertEqual(report.horizontal_error, 0.0)

    def test_bbox_center_and_width_use_existing_detection_values(self):
        detection = Detection(100, 10, 200, 50, 0, "person", 0.9)
        report = horizontal_report(detection, 640)
        self.assertEqual(detection.Center[0], 150)
        self.assertEqual(report.bbox_x, 100)
        self.assertEqual(report.bbox_width, 100)
        self.assertEqual(report.center_x, 150.0)

    def test_physical_image_right_has_positive_error(self):
        report = horizontal_report(detection_at(480), 640)
        self.assertEqual(report.position, "RIGHT")
        self.assertGreater(report.horizontal_error, 0.0)
        self.assertFalse(report.displayed_frame_mirrored)

    def test_rejects_invalid_frame_width(self):
        with self.assertRaises(ValueError):
            horizontal_report(detection_at(10), 0)

    def test_cli_defaults_and_overrides(self):
        args = build_parser().parse_args(["--camera", "0", "--confidence", "0.5", "--frames", "0", "--no-window"])
        self.assertEqual(args.camera, 0)
        self.assertEqual(args.confidence, 0.5)
        self.assertEqual(args.frames, 0)
        self.assertTrue(args.no_window)


if __name__ == "__main__":
    unittest.main()
