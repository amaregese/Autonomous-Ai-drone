import unittest

from jetson.communication.detection_sender import _convert_detection, apply_authoritative_target_distance
from modules.distance_estimator import (
    CameraIntrinsics,
    DistanceEstimator,
    VisionConfig,
    VisionDistanceEstimator,
    annotate_detections,
)
from modules.person_follow import PersonFollowController
from modules.yolo11_detector.types import Detection
from shared.detection_models import Detection as SharedDetection


def calibrated():
    return CameraIntrinsics(
        fx=1406.4, fy=1405.2, cx=320.0, cy=240.0,
        distortion={"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        frame_w=640, frame_h=480,
    )


class TestRuntimeIntrinsics(unittest.TestCase):
    def test_same_resolution_does_not_change_intrinsics(self):
        scaled = calibrated().scaled_to_frame(640, 480)
        self.assertEqual((scaled.fx, scaled.fy, scaled.cx, scaled.cy), (1406.4, 1405.2, 320.0, 240.0))

    def test_640x480_to_640x360_scales_y_terms(self):
        scaled = calibrated().scaled_to_frame(640, 360)
        self.assertEqual(scaled.fx, 1406.4)
        self.assertAlmostEqual(scaled.fy, 1053.9)
        self.assertEqual(scaled.cx, 320.0)
        self.assertEqual(scaled.cy, 180.0)

    def test_1280x960_to_640x480_halves_all_pixel_intrinsics(self):
        base = CameraIntrinsics(2000, 1800, 640, 480, frame_w=1280, frame_h=960)
        scaled = base.scaled_to_frame(640, 480)
        self.assertEqual((scaled.fx, scaled.fy, scaled.cx, scaled.cy), (1000, 900, 320, 240))

    def test_follow_uses_corrected_intrinsics(self):
        intrinsics = calibrated().scaled_to_frame(640, 360)
        det = Detection(270, 20, 370, 302, 0, "person", 0.9, timestamp=1.0)
        det.is_selected = True
        cmd = PersonFollowController(distance_estimator=DistanceEstimator(intrinsics=intrinsics)).update(
            det, (360, 640), now=1.01
        )
        self.assertAlmostEqual(cmd.range_m, 1.3 * 1053.9 / 282.0, places=3)


    def test_annotation_keeps_each_detection_value(self):
        intrinsics = CameraIntrinsics(
            fx=446.7, fy=446.7, cx=320.0, cy=240.0,
            distortion={"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
            frame_w=640, frame_h=480,
        )
        estimator = VisionDistanceEstimator(intrinsics, VisionConfig())
        person = Detection(300, 100, 400, 360, 0, "person", 0.9)
        car = Detection(100, 200, 300, 300, 2, "car", 0.8)
        annotate_detections(estimator, [person, car], 640, 480)
        self.assertAlmostEqual(person.distance_m, 1.3 * 446.7 / 260.0, places=6)
        self.assertAlmostEqual(car.distance_m, 1.5 * 446.7 / 100.0, places=6)
        self.assertTrue(person.distance_valid)
        self.assertTrue(car.distance_valid)
        self.assertEqual(person.distance_source, "vision")
        self.assertEqual(car.distance_source, "vision")
        self.assertNotEqual(person.detection_id, car.detection_id)

    def test_invalid_annotation_is_explicitly_unavailable(self):
        estimator = VisionDistanceEstimator(calibrated().scaled_to_frame(640, 480), VisionConfig())
        detection = Detection(10, 50, 110, 50, 0, "person", 0.9)
        annotate_detections(estimator, [detection], 640, 480)
        self.assertIsNone(detection.distance_m)
        self.assertFalse(detection.distance_valid)
        self.assertEqual(detection.distance_source, "none")
        self.assertEqual(detection.distance_confidence, 0.0)

    def test_controller_uses_attached_value_without_recomputing(self):
        class FailingEstimator:
            def estimate_vision(self, *args, **kwargs):
                raise AssertionError("distance was recomputed")

            def update(self, *args, **kwargs):
                raise AssertionError("distance was recomputed")

        detection = Detection(10, 10, 110, 210, 0, "person", 0.9, timestamp=1.0)
        detection.is_selected = True
        detection.distance_m = 4.25
        detection.distance_valid = True
        detection.distance_source = "vision"
        detection.distance_confidence = 0.8
        command = PersonFollowController(distance_estimator=FailingEstimator()).update(
            detection, (480, 640), now=1.01
        )
        self.assertEqual(command.range_m, 4.25)
        self.assertTrue(command.distance_valid)
        self.assertEqual(command.detection_id, detection.detection_id)

    def test_controller_does_not_fallback_from_invalid_annotation(self):
        class FailingEstimator:
            def estimate_vision(self, *args, **kwargs):
                raise AssertionError("invalid annotation was recomputed")

            def update(self, *args, **kwargs):
                raise AssertionError("invalid annotation was recomputed")

        detection = Detection(10, 10, 110, 210, 0, "person", 0.9, timestamp=1.0)
        detection.is_selected = True
        detection.distance_m = 4.25
        detection.distance_valid = False
        detection.distance_source = "none"
        command = PersonFollowController(distance_estimator=FailingEstimator()).update(
            detection, (480, 640), now=1.01
        )
        self.assertIsNone(command.range_m)
        self.assertEqual(command.lane, "no_range")
        self.assertFalse(command.distance_valid)


class TestAuthoritativeStreamDistance(unittest.TestCase):
    def test_selected_target_receives_authoritative_distance_only(self):
        selected = Detection(10, 10, 110, 210, 0, "person", 0.9)
        other = Detection(200, 10, 300, 210, 2, "car", 0.9)
        shared = [SharedDetection(), SharedDetection()]
        apply_authoritative_target_distance(
            shared, [selected, other], selected,
            {"distance_valid": True, "distance_m": 4.25, "target_detection_id": id(selected)},
        )
        self.assertEqual(shared[0].distance, 4.25)
        self.assertIsNone(shared[1].distance)

    def test_no_selected_target_does_not_assign_distance(self):
        source = Detection(10, 10, 110, 210, 0, "person", 0.9)
        shared = [SharedDetection()]
        apply_authoritative_target_distance(shared, [source], None, {"distance_valid": True, "distance_m": 4.25})
        self.assertIsNone(shared[0].distance)

    def test_shared_conversion_copies_per_detection_fields(self):
        source = Detection(10, 10, 110, 210, 0, "person", 0.9)
        source.distance_m = 4.25
        source.distance_valid = True
        source.distance_source = "vision"
        source.distance_confidence = 0.75
        converted = _convert_detection(source, True, "tracking")
        self.assertEqual(converted.distance, 4.25)
        self.assertTrue(converted.distance_valid)
        self.assertEqual(converted.distance_source, "vision")
        self.assertEqual(converted.distance_confidence, 0.75)
        self.assertEqual(converted.detection_id, source.detection_id)


if __name__ == "__main__":
    unittest.main()
