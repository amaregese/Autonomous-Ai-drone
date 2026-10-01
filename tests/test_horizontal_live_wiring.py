"""
Tests for wiring the observational horizontal ground distance into the live
selected-target measurement path.

Run with:
    python -m unittest discover -s tests -t . -p "test_*.py"

Scope: the live annotation seam (``annotate_detections`` / ``annotate_horizontal``)
now attaches ``horizontal_distance_m`` to the *selected* person detection using
the caller-supplied AGL altitude. The authoritative optical-axis ``distance_m``
and the entire follow/movement path are unchanged.

These tests deliberately avoid importing the YOLO11 package (which pulls torch);
``types.py`` is loaded directly so the real ``Detection`` class is exercised.
"""
from __future__ import annotations

import importlib.util
import pathlib
import unittest

from modules.distance_estimator.calibration import CameraIntrinsics
from modules.distance_estimator.config import VisionConfig
from modules.distance_estimator.vision import (
    VisionDistanceEstimator,
    annotate_detection,
    annotate_detections,
    annotate_horizontal,
    estimate_detection_horizontal,
)
from modules.person_follow import FollowConfig, PersonFollowController

FX = 446.7
FY = 446.7
CX = 320.0
CY = 240.0

ROOT = pathlib.Path(__file__).resolve().parents[1]

_types_path = ROOT / "modules" / "yolo11_detector" / "types.py"
_spec = importlib.util.spec_from_file_location("yolo11_types_under_test", _types_path)
_types = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_types)
Detection = _types.Detection


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY, frame_w=640, frame_h=480)


def make_detection(selected=True, class_name="person", confidence=0.9,
                   left=295, top=150, right=345, bottom=295) -> object:
    detection = Detection(left, top, right, bottom, class_id=0,
                          class_name=class_name, confidence=confidence)
    detection.is_selected = selected
    return detection


class TestDetectionFields(unittest.TestCase):
    """The real Detection carries the new observational fields, defaulted off."""

    def test_horizontal_fields_default_off(self):
        detection = make_detection()
        self.assertIsNone(detection.horizontal_distance_m)
        self.assertFalse(detection.horizontal_distance_valid)
        self.assertIsNone(detection.slant_range_m)
        self.assertIsNone(detection.horizontal_altitude_m)
        self.assertIsNone(detection.horizontal_distance_source)

    def test_existing_distance_fields_untouched_by_construction(self):
        detection = make_detection()
        self.assertIsNone(detection.distance_m)
        self.assertFalse(detection.distance_valid)


class TestLiveAnnotation(unittest.TestCase):
    """Requirement 3: the live selected-target point populates horizontal distance."""

    def setUp(self):
        self.estimator = VisionDistanceEstimator(make_intrinsics(), VisionConfig())
        self.expected_distance = self.estimator.estimate(
            bbox_height_px=145, class_name="person", frame_w=640, frame_h=480
        ).distance_m

    def test_selected_person_gets_horizontal(self):
        detection = make_detection(selected=True)
        annotate_detections(self.estimator, [detection], 640, 480,
                            altitude_m=3.0, horizontal_classes={"person"})
        self.assertTrue(detection.distance_valid)
        self.assertAlmostEqual(detection.distance_m, self.expected_distance, places=9)
        self.assertTrue(detection.horizontal_distance_valid)
        self.assertIsNotNone(detection.horizontal_distance_m)
        self.assertGreater(detection.horizontal_distance_m, 0.0)
        self.assertLess(detection.horizontal_distance_m, detection.distance_m)
        self.assertAlmostEqual(detection.horizontal_altitude_m, 3.0, places=9)
        self.assertEqual(detection.horizontal_distance_source, "vision-horizontal")

    def test_distance_m_identical_with_and_without_altitude(self):
        """Requirement 5: the authoritative distance is byte-identical."""
        with_alt = make_detection(selected=True)
        without_alt = make_detection(selected=True)
        annotate_detections(self.estimator, [with_alt], 640, 480,
                            altitude_m=3.0, horizontal_classes={"person"})
        annotate_detections(self.estimator, [without_alt], 640, 480)
        self.assertEqual(with_alt.distance_m, without_alt.distance_m)
        self.assertEqual(with_alt.distance_source, without_alt.distance_source)
        self.assertIsNotNone(with_alt.horizontal_distance_m)
        self.assertIsNone(without_alt.horizontal_distance_m)

    def test_missing_altitude_leaves_horizontal_unset(self):
        detection = make_detection(selected=True)
        annotate_detections(self.estimator, [detection], 640, 480,
                            altitude_m=None, horizontal_classes={"person"})
        self.assertTrue(detection.distance_valid)
        self.assertFalse(detection.horizontal_distance_valid)
        self.assertIsNone(detection.horizontal_distance_m)

    def test_invalid_altitude_leaves_horizontal_unset(self):
        for bad in (-1.0, 50.0, float("nan"), float("inf")):
            with self.subTest(altitude=bad):
                detection = make_detection(selected=True)
                annotate_detections(self.estimator, [detection], 640, 480,
                                    altitude_m=bad, horizontal_classes={"person"})
                self.assertTrue(detection.distance_valid)
                self.assertFalse(detection.horizontal_distance_valid)
                self.assertIsNone(detection.horizontal_distance_m)

    def test_unselected_target_is_not_measured(self):
        detection = make_detection(selected=False)
        annotate_detections(self.estimator, [detection], 640, 480,
                            altitude_m=3.0, horizontal_classes={"person"})
        self.assertTrue(detection.distance_valid)
        self.assertFalse(detection.horizontal_distance_valid)
        self.assertIsNone(detection.horizontal_distance_m)

    def test_non_target_class_is_not_measured(self):
        detection = make_detection(selected=True, class_name="car")
        annotate_detections(self.estimator, [detection], 640, 480,
                            altitude_m=3.0, horizontal_classes={"person"})
        self.assertTrue(detection.distance_valid)
        self.assertFalse(detection.horizontal_distance_valid)
        self.assertIsNone(detection.horizontal_distance_m)

    def test_empty_detections_is_safe(self):
        result = annotate_detections(self.estimator, [], 640, 480, altitude_m=3.0)
        self.assertEqual(result, [])


class TestAnnotateHorizontalObservational(unittest.TestCase):
    """annotate_horizontal only ever touches the horizontal fields."""

    def setUp(self):
        self.estimator = VisionDistanceEstimator(make_intrinsics(), VisionConfig())

    def test_does_not_touch_axis_depth(self):
        detection = make_detection(selected=True)
        base = self.estimator.estimate(
            bbox_height_px=detection.height, class_name="person",
            frame_w=640, frame_h=480,
        )
        annotate_detection(detection, base)
        before = (detection.distance_m, detection.distance_valid,
                  detection.distance_source, detection.distance_confidence)
        measurement = estimate_detection_horizontal(
            self.estimator, detection, 640, 480, 3.0
        )
        annotate_horizontal(detection, measurement, 3.0)
        after = (detection.distance_m, detection.distance_valid,
                 detection.distance_source, detection.distance_confidence)
        self.assertEqual(before, after)
        self.assertTrue(detection.horizontal_distance_valid)

    def test_reset_to_none(self):
        detection = make_detection(selected=True)
        measurement = estimate_detection_horizontal(
            self.estimator, detection, 640, 480, 3.0
        )
        annotate_horizontal(detection, measurement, 3.0)
        self.assertTrue(detection.horizontal_distance_valid)
        annotate_horizontal(detection, None)
        self.assertFalse(detection.horizontal_distance_valid)
        self.assertIsNone(detection.horizontal_distance_m)
        self.assertIsNone(detection.slant_range_m)

    def test_none_detection_is_safe(self):
        self.assertIsNone(annotate_horizontal(None, None))


class TestAltitudeSourceReuse(unittest.TestCase):
    """Requirement 2: the app reuses the existing AGL altitude, adds no source."""

    def test_main_passes_ground_altitude_into_annotation(self):
        source = (ROOT / "autonomous_drone_main.py").read_text(encoding="utf-8")
        self.assertIn("altitude_m=_ground_altitude()", source)
        self.assertIn("horizontal_classes={modules.app_config.FOLLOW_TARGET_CLASS}", source)
        self.assertIn("def _ground_altitude(", source)

    def test_no_second_altitude_source_added(self):
        source = (ROOT / "autonomous_drone_main.py").read_text(encoding="utf-8")
        self.assertNotIn("ATTITUDE", source)
        self.assertNotIn("horizontal_distance_m =", source)


class TestMovementUnaffected(unittest.TestCase):
    """Requirements 5/6: horizontal measurement never drives movement."""

    def test_target_forward_speed_is_distance_only(self):
        controller = PersonFollowController(config=FollowConfig())
        self.assertEqual(controller._target_forward_speed(2.0), 0.0)
        self.assertGreater(controller._target_forward_speed(3.0), 0.0)
        self.assertGreater(
            controller._target_forward_speed(4.0),
            controller._target_forward_speed(3.0),
        )
        self.assertEqual(
            controller._target_forward_speed(3.0),
            controller._target_forward_speed(3.0),
        )

    def test_person_follow_reads_horizontal_without_computing_it(self):
        source = (ROOT / "modules" / "person_follow.py").read_text(encoding="utf-8")
        self.assertIn("horizontal_distance_m", source)
        self.assertNotIn("estimate_horizontal", source)
        self.assertNotIn("axis_depth_to_slant_range", source)
        self.assertNotIn("slant_to_horizontal_distance", source)

    def test_follow_movement_dict_block_has_no_horizontal(self):
        source = (ROOT / "autonomous_drone_main.py").read_text(encoding="utf-8")
        block = source.split("def _follow_movement_dict", 1)[1]
        block = block.split("def _refresh_follow_estimator", 1)[0]
        self.assertNotIn("horizontal", block)

    def test_mavlink_and_control_layers_have_no_horizontal(self):
        for relative in (
            "modules/control_system/api.py",
            "modules/drone_backend/sitl.py",
        ):
            with self.subTest(path=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                self.assertNotIn("horizontal", source)
                self.assertNotIn("estimate_horizontal", source)


if __name__ == "__main__":
    unittest.main()
