"""
Focused tests for the read-only selected-target Z/R/H/AGL console diagnostic.

Run with:
    python -m unittest discover -s tests -t . -p "test_*.py"

The formatter lives in ``modules.follow_diagnostics`` so it can be tested
without importing ``autonomous_drone_main`` (which runs the flight app on
import). These tests prove the diagnostic is observational only: it reports the
existing optical-axis distance unchanged and never fabricates a horizontal value.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import unittest

from modules.follow_diagnostics import diagnostic_text, metric_text, selected_diagnostic

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")

_types_path = ROOT / "modules" / "yolo11_detector" / "types.py"
_spec = importlib.util.spec_from_file_location("yolo11_types_diag", _types_path)
_types = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_types)
Detection = _types.Detection


def make_detection(**distance_fields) -> object:
    detection = Detection(295, 150, 345, 295, class_id=0,
                          class_name="person", confidence=0.9)
    detection.is_selected = True
    for key, value in distance_fields.items():
        setattr(detection, key, value)
    return detection


def plain(text: str) -> str:
    return ANSI.sub("", text)


class TestMetricText(unittest.TestCase):
    def test_valid_value(self):
        self.assertEqual(metric_text(5.0), "5.00m")

    def test_unusable_values_render_dashes(self):
        for bad in (None, 0.0, -1.0, float("nan"), float("inf"), "oops"):
            with self.subTest(value=bad):
                self.assertEqual(metric_text(bad), "--")


class TestValidExposesAllFour(unittest.TestCase):
    """Requirement 1: a valid measurement exposes Z, R, H and AGL."""

    def setUp(self):
        self.detection = make_detection(
            distance_m=5.0,
            horizontal_distance_valid=True,
            slant_range_m=5.1,
            horizontal_distance_m=4.1,
            horizontal_altitude_m=3.0,
        )

    def test_labels_and_values_present(self):
        out = selected_diagnostic(self.detection, z=self.detection.distance_m)
        self.assertEqual(plain(out), "Z=5.00m R=5.10m H=4.10m AGL=3.00m")

    def test_each_label_present(self):
        out = plain(selected_diagnostic(self.detection, z=self.detection.distance_m))
        for token in ("Z=5.00m", "R=5.10m", "H=4.10m", "AGL=3.00m"):
            with self.subTest(token=token):
                self.assertIn(token, out)

    def test_z_is_not_replaced_by_horizontal(self):
        """Requirement 2/3: Z stays the optical-axis value, not H."""
        out = plain(selected_diagnostic(self.detection, z=self.detection.distance_m))
        self.assertIn("Z=5.00m", out)
        self.assertIn("H=4.10m", out)
        self.assertNotIn("Z=4.10m", out)


class TestExistingDistanceUnchanged(unittest.TestCase):
    """Requirement 2: the diagnostic never mutates the authoritative distance."""

    def test_detection_distance_untouched(self):
        detection = make_detection(
            distance_m=5.0,
            horizontal_distance_valid=True,
            slant_range_m=5.1,
            horizontal_distance_m=4.1,
            horizontal_altitude_m=3.0,
        )
        before = (detection.distance_m, detection.distance_valid, detection.distance_source)
        selected_diagnostic(detection, z=detection.distance_m)
        after = (detection.distance_m, detection.distance_valid, detection.distance_source)
        self.assertEqual(before, after)

    def test_z_uses_passed_axis_distance(self):
        detection = make_detection(
            distance_m=2.0,
            horizontal_distance_valid=True,
            slant_range_m=2.2,
            horizontal_distance_m=1.1,
            horizontal_altitude_m=1.9,
        )
        out = plain(selected_diagnostic(detection, z=7.5))
        self.assertIn("Z=7.50m", out)


class TestInvalidAltitudeNoFakeHorizontal(unittest.TestCase):
    """Requirement 3: invalid altitude must not produce a fabricated H."""

    def test_invalid_horizontal_renders_dashes(self):
        detection = make_detection(
            distance_m=5.0,
            horizontal_distance_valid=False,
            slant_range_m=None,
            horizontal_distance_m=None,
            horizontal_altitude_m=None,
        )
        out = plain(selected_diagnostic(detection, z=detection.distance_m))
        self.assertEqual(out, "Z=5.00m R=-- H=-- AGL=--")
        self.assertNotIn("H=0", out)
        self.assertIn("Z=5.00m", out)


class TestMissingHorizontalDoesNotBreakOutput(unittest.TestCase):
    """Requirement 4: absent horizontal measurement still yields a status segment."""

    def test_no_selected_object(self):
        self.assertEqual(plain(selected_diagnostic(None, z=None)),
                         "Z=-- R=-- H=-- AGL=--")

    def test_object_without_fields(self):
        self.assertEqual(plain(selected_diagnostic(object(), z=3.0)),
                         "Z=3.00m R=-- H=-- AGL=--")

    def test_diagnostic_text_all_none(self):
        self.assertEqual(plain(diagnostic_text()),
                         "Z=-- R=-- H=-- AGL=--")

    def test_partial_values(self):
        out = plain(diagnostic_text(z=5.0, r=5.1, h=4.1, agl=3.0))
        self.assertEqual(out, "Z=5.00m R=5.10m H=4.10m AGL=3.00m")


class TestNoConsumerOfHorizontal(unittest.TestCase):
    """Requirements 5/6: neither movement nor MAVLink reads H/R."""

    def test_person_follow_reads_horizontal_without_geometry(self):
        source = (ROOT / "modules" / "person_follow.py").read_text(encoding="utf-8")
        self.assertIn("horizontal_distance_m", source)
        self.assertNotIn("estimate_horizontal", source)
        self.assertNotIn("axis_depth_to_slant_range", source)
        self.assertNotIn("slant_to_horizontal_distance", source)

    def test_main_movement_block_has_no_horizontal(self):
        source = (ROOT / "autonomous_drone_main.py").read_text(encoding="utf-8")
        block = source.split("def _follow_movement_dict", 1)[1]
        block = block.split("def _refresh_follow_estimator", 1)[0]
        self.assertNotIn("horizontal", block)
        self.assertNotIn("slant_range", block)

    def test_mavlink_and_control_have_no_horizontal(self):
        for relative in (
            "modules/drone_backend/sitl.py",
            "modules/control_system/api.py",
        ):
            with self.subTest(path=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                self.assertNotIn("horizontal", source)
                self.assertNotIn("slant_range", source)


class TestConsoleWiring(unittest.TestCase):
    """The existing console status uses the diagnostic and keeps the Z display."""

    def test_console_status_calls_diagnostic(self):
        source = (ROOT / "autonomous_drone_main.py").read_text(encoding="utf-8")
        status_block = source.split("def _console_status", 1)[1]
        status_block = status_block.split("def _follow_movement_dict", 1)[0]
        self.assertIn("selected_diagnostic(", status_block)
        self.assertIn("distance_text", status_block)

    def test_import_present(self):
        source = (ROOT / "autonomous_drone_main.py").read_text(encoding="utf-8")
        self.assertIn("from modules.follow_diagnostics import selected_diagnostic", source)


if __name__ == "__main__":
    unittest.main()
