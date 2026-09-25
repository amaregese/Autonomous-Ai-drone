import unittest
from types import SimpleNamespace

from modules.display import _detection_label, _follow_banner_text
from modules.yolo11_detector.types import Detection


class TestDetectionLabels(unittest.TestCase):
    def setUp(self):
        self.person = Detection(10, 10, 110, 210, 0, "person", 0.92)
        self.car = Detection(200, 20, 300, 180, 2, "car", 0.87)

    def test_selected_target_shows_confidence_and_authoritative_distance(self):
        movement = {"distance_valid": True, "distance_m": 4.405}
        self.assertEqual(_detection_label(self.person, True, movement), "person 0.92 | 4.41 m")

    def test_detection_with_no_distance_shows_na(self):
        self.assertEqual(_detection_label(self.car, False), "car 0.87 | N/A")

    def test_selected_target_with_unavailable_distance_shows_na(self):
        movement = {"distance_valid": False, "distance_m": 4.405}
        self.assertEqual(_detection_label(self.person, True, movement), "person 0.92 | N/A")

    def test_non_selected_detection_does_not_inherit_target_distance(self):
        movement = {"distance_valid": True, "distance_m": 4.405}
        self.assertEqual(_detection_label(self.car, False, movement), "car 0.87 | N/A")

    def test_follow_banner_uses_selected_authoritative_distance(self):
        movement = {"distance_valid": True, "distance_m": 4.405, "vel_z": 0.12, "yaw_cmd": -1.0}
        self.assertEqual(
            _follow_banner_text(self.person, movement),
            "FOLLOWING person | 0.92 | 4.41 m   +0.12m/s   YAW -1.0",
        )

    def test_follow_banner_shows_na_when_distance_is_unavailable(self):
        movement = {"distance_valid": False, "distance_m": 4.405}
        self.assertEqual(
            _follow_banner_text(self.person, movement),
            "FOLLOWING person | 0.92 | N/A   +0.00m/s   YAW +0.0",
        )

    def test_bbox_and_follow_banner_share_the_same_distance(self):
        movement = {"distance_valid": True, "distance_m": 4.405}
        bbox_label = _detection_label(self.person, True, movement)
        banner = _follow_banner_text(self.person, movement)
        self.assertIn("4.41 m", bbox_label)
        self.assertIn("4.41 m", banner)

    def test_follow_command_range_m_is_accepted_as_the_result_distance(self):
        command = SimpleNamespace(range_m=4.405)
        self.assertEqual(
            _detection_label(self.person, True, command),
            "person 0.92 | 4.41 m",
        )
        self.assertIn("4.41 m", _follow_banner_text(self.person, command))

    def test_no_range_follow_command_does_not_reuse_last_range(self):
        command = SimpleNamespace(range_m=4.405, active=True, lost=False, lane="no_range")
        self.assertEqual(
            _detection_label(self.person, True, command),
            "person 0.92 | N/A",
        )
        self.assertIn("N/A", _follow_banner_text(self.person, command))

    def test_distance_is_not_applied_to_a_different_target(self):
        movement = {
            "distance_valid": True,
            "distance_m": 4.405,
            "target_detection_id": id(self.car),
        }
        self.assertEqual(_detection_label(self.person, True, movement), "person 0.92 | N/A")
        self.assertIn("N/A", _follow_banner_text(self.person, movement))


if __name__ == "__main__":
    unittest.main()
