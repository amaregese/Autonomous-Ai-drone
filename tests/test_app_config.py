"""
Unit tests for the vehicle altitude setpoints in ``modules.app_config``.

Run with:
    python -m unittest discover -s tests -t . -p "test_*.py"

These values are declarative configuration only: nothing consumes them yet, so
the tests assert they are defined, numerically sane and reachable both as module
attributes and by direct import (the two styles used across this project).
"""
from __future__ import annotations

import pathlib
import unittest

from modules import app_config
from modules.app_config import ALTITUDE_TOLERANCE, FOLLOW_ALTITUDE, TAKEOFF_ALTITUDE


class TestAltitudeSetpoints(unittest.TestCase):
    """TAKEOFF_ALTITUDE / FOLLOW_ALTITUDE / ALTITUDE_TOLERANCE are configured."""

    def test_configured_values(self):
        """The three altitude constants hold the documented default values."""
        self.assertAlmostEqual(TAKEOFF_ALTITUDE, 3.0, places=9)
        self.assertAlmostEqual(FOLLOW_ALTITUDE, 3.0, places=9)
        self.assertAlmostEqual(ALTITUDE_TOLERANCE, 0.5, places=9)

    def test_accessible_via_module_namespace(self):
        """They are reachable as ``app_config.<NAME>`` (late-bound style)."""
        self.assertAlmostEqual(app_config.TAKEOFF_ALTITUDE, 3.0, places=9)
        self.assertAlmostEqual(app_config.FOLLOW_ALTITUDE, 3.0, places=9)
        self.assertAlmostEqual(app_config.ALTITUDE_TOLERANCE, 0.5, places=9)

    def test_values_are_positive_floats(self):
        """Altitudes and the tolerance are positive finite floats (metres)."""
        for value in (TAKEOFF_ALTITUDE, FOLLOW_ALTITUDE, ALTITUDE_TOLERANCE):
            with self.subTest(value=value):
                self.assertIsInstance(value, float)
                self.assertGreater(value, 0.0)


class TestExistingFollowConfigUnchanged(unittest.TestCase):
    """Adding the altitude setpoints must not disturb the existing constants."""

    def test_existing_follow_constants(self):
        """Pre-existing follow / speed / safety constants are untouched."""
        self.assertAlmostEqual(app_config.FOLLOW_MIN_DISTANCE_M, 2.0, places=9)
        self.assertAlmostEqual(app_config.FOLLOW_MAX_SPEED_MPS, 1.0, places=9)
        self.assertAlmostEqual(app_config.FOLLOW_MAX_LATERAL_SPEED_MPS, 0.6, places=9)
        self.assertAlmostEqual(app_config.FOLLOW_LATERAL_GAIN, 1.2, places=9)
        self.assertAlmostEqual(app_config.GAIN_YAW, 2.0, places=9)
        self.assertAlmostEqual(app_config.MAX_ALT, 5.0, places=9)
        self.assertEqual(app_config.FOLLOW_TARGET_CLASS, "person")

    def test_altitude_setpoints_are_not_wired_into_follow_config(self):
        """``FollowConfig`` still mirrors the same fields as before the change."""
        from modules.person_follow import FollowConfig

        fields = {f.name for f in FollowConfig.__dataclass_fields__.values()}
        self.assertNotIn("follow_altitude", fields)
        self.assertNotIn("altitude_tolerance", fields)
        self.assertNotIn("takeoff_altitude", fields)


class TestTakeoffAltitudeWiring(unittest.TestCase):
    """The takeoff path must command TAKEOFF_ALTITUDE, keeping MAX_ALT as the ceiling."""

    def setUp(self):
        import pathlib

        self.source = (
            pathlib.Path(__file__).resolve().parents[1] / "autonomous_drone_main.py"
        ).read_text(encoding="utf-8")

    def test_takeoff_commands_takeoff_altitude(self):
        """``climb()`` commands ``control.takeoff`` with TAKEOFF_ALTITUDE, not MAX_ALT."""
        self.assertIn("control.takeoff(TAKEOFF_ALTITUDE)", self.source)
        self.assertNotIn("control.takeoff(MAX_ALT)", self.source)

    def test_post_takeoff_flight_altitude_uses_takeoff_altitude(self):
        """The local flight-altitude state matches the altitude just climbed to."""
        self.assertIn("control.set_flight_altitude(TAKEOFF_ALTITUDE)", self.source)

    def test_takeoff_uses_takeoff_altitude_value(self):
        """The commanded value is 3.0 m, the configured TAKEOFF_ALTITUDE."""
        self.assertAlmostEqual(TAKEOFF_ALTITUDE, 3.0, places=9)
        self.assertLess(TAKEOFF_ALTITUDE, app_config.MAX_ALT)

    def test_max_alt_retained_as_ceiling(self):
        """MAX_ALT is still defined and still guards the non-takeoff call sites."""
        self.assertAlmostEqual(app_config.MAX_ALT, 5.0, places=9)
        self.assertIn("control.set_flight_altitude(MAX_ALT)", self.source)
        self.assertIn("alt = MAX_ALT", self.source)
        self.assertIn(
            "drone.send_movement_command_XYA(follow_cmd.vy, follow_cmd.vx, MAX_ALT)",
            self.source,
        )
        self.assertIn("drone.send_movement_command_XYA(0, 0, MAX_ALT)", self.source)

    def test_takeoff_messages_report_configured_altitude(self):
        """No hardcoded takeoff altitude is left in the arm/takeoff messages."""
        self.assertIn("TAKEOFF {TAKEOFF_ALTITUDE:.0f}m is now available", self.source)
        self.assertIn("Taking off to {TAKEOFF_ALTITUDE:.0f}m", self.source)
        self.assertIn("Airborne — holding at {TAKEOFF_ALTITUDE:.0f}m", self.source)
        self.assertNotIn("TAKEOFF 5m is now available", self.source)

    def test_follow_and_altitude_hold_not_implemented(self):
        """FOLLOW_ALTITUDE / ALTITUDE_TOLERANCE remain configuration only."""
        self.assertNotIn("app_config.FOLLOW_ALTITUDE", self.source)
        self.assertNotIn("app_config.ALTITUDE_TOLERANCE", self.source)
        self.assertNotIn("FOLLOW_ALTITUDE,", self.source)
        self.assertNotIn("ALTITUDE_TOLERANCE,", self.source)


class TestTakeoffButtonLabel(unittest.TestCase):
    """The on-screen TAKEOFF button must render the configured altitude."""

    def setUp(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        self.source = (root / "modules" / "display.py").read_text(encoding="utf-8")

    def test_button_label_uses_configured_takeoff_altitude(self):
        """The pill label is built from ``app_config.TAKEOFF_ALTITUDE``."""
        self.assertIn('f"TAKEOFF {app_config.TAKEOFF_ALTITUDE:.0f}m"', self.source)
        self.assertNotIn('"TAKEOFF 5m"', self.source)

    def test_button_geometry_and_enabling_unchanged(self):
        """Only the label text changed; size, position and arm gating are intact."""
        for fragment in (
            "_TAKEOFF_BUTTON_W",
            "_TAKEOFF_BUTTON_H",
            "_BUTTON_GAP",
            "_LAND_BUTTON_W",
            "_BUTTON_MARGIN",
            "enabled=armed",
            "get_takeoff_button_rect",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_button_label_renders_configured_value(self):
        """With the shipped config the label reads 'TAKEOFF 3m'."""
        label = f"TAKEOFF {app_config.TAKEOFF_ALTITUDE:.0f}m"
        self.assertEqual(label, "TAKEOFF 3m")

    def test_display_import_does_not_create_a_cycle(self):
        """``display`` imports ``app_config``; ``app_config`` imports nothing."""
        app_config_source = (
            pathlib.Path(app_config.__file__).read_text(encoding="utf-8")
        )
        for line in app_config_source.splitlines():
            with self.subTest(line=line):
                self.assertFalse(
                    line.startswith("import ") or line.startswith("from "),
                    "app_config must stay import-free to avoid a cycle with display",
                )


if __name__ == "__main__":
    unittest.main()
