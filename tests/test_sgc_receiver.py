"""The SGC camera-servo UDP protocol: parse, validate, queue.

Path under test:  SGC Camera UI -> UDP 9002 -> ``SGCCommandReceiver`` ->
``SGCCommand`` -> main's servo handler -> (later) MAVLink.  These tests stop at
the queue, so they need no Pixhawk, no MAVLink link and no real servo; the
handler wiring that turns ``channel=None`` into automatic channel selection is
pinned by reading the source, the same way ``test_app_config`` pins the
takeoff wiring.

Protocol rules pinned here:

* ``type`` must be present and a non-empty string; the camera-servo feature is
  ``type == "servo"``.
* ``pulse`` is an integer 1000-2000 when present - never coerced, never
  clamped.
* ``channel`` is optional; absent means automatic/default selection downstream.
* ``channel`` is an integer 1-16 when present - ``0``, ``17`` and ``"8"`` are
  rejected, not repaired.
"""

import json
import pathlib
import socket
import time
import unittest

from jetson.communication import sgc_receiver
from jetson.communication.sgc_receiver import SGCCommandReceiver, parse_sgc_command

LOGGER = "jetson.communication.sgc_receiver"
MAIN_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "autonomous_drone_main.py"
).read_text(encoding="utf-8")
RECEIVER_SOURCE = pathlib.Path(sgc_receiver.__file__).read_text(encoding="utf-8")


def payload(**fields):
    """One SGC datagram, exactly as the camera UI sends it."""
    return json.dumps(fields).encode("utf-8")


class TestServoPayloadParsing(unittest.TestCase):
    """`parse_sgc_command`: what the standardized SGC payload may contain."""

    def test_valid_servo_with_channel(self):
        cmd = parse_sgc_command(payload(type="servo", pulse=1500, channel=8))
        self.assertEqual(cmd.command_type, "servo")
        self.assertEqual(cmd.pulse, 1500)
        self.assertEqual(cmd.channel, 8)

    def test_valid_servo_without_channel(self):
        """Missing `channel` is valid: it means automatic selection."""
        cmd = parse_sgc_command(payload(type="servo", pulse=1500))
        self.assertEqual(cmd.command_type, "servo")
        self.assertEqual(cmd.pulse, 1500)
        self.assertIsNone(cmd.channel)

    def test_channel_boundaries_are_valid(self):
        self.assertEqual(
            parse_sgc_command(payload(type="servo", pulse=1500, channel=1)).channel, 1)
        self.assertEqual(
            parse_sgc_command(payload(type="servo", pulse=1500, channel=16)).channel, 16)

    def test_pulse_boundaries_are_valid(self):
        self.assertEqual(
            parse_sgc_command(payload(type="servo", pulse=1000)).pulse, 1000)
        self.assertEqual(
            parse_sgc_command(payload(type="servo", pulse=2000)).pulse, 2000)

    def test_invalid_channel_values_are_rejected(self):
        for channel in (0, 17, -1, "8", 8.0, True):
            with self.subTest(channel=channel):
                with self.assertRaises(ValueError):
                    parse_sgc_command(
                        payload(type="servo", pulse=1500, channel=channel))

    def test_invalid_pulse_values_are_rejected(self):
        # Out of range, wrong JSON type, and booleans (bool is an int subclass)
        # must all be refused - nothing is clamped into range or coerced.
        for pulse in (999, 2001, 500, "1500", 1500.0, True):
            with self.subTest(pulse=pulse):
                with self.assertRaises(ValueError):
                    parse_sgc_command(payload(type="servo", pulse=pulse))

    def test_missing_pulse_keeps_the_handlers_default(self):
        """Absence is not a malformed value: the handler still defaults to 1500.

        The standardized payload always carries `pulse`; this only pins that a
        pulse-less servo packet is not turned into a *new* rejection, so the
        legacy `angle` form keeps working.
        """
        cmd = parse_sgc_command(payload(type="servo"))
        self.assertEqual(cmd.command_type, "servo")
        self.assertIsNone(cmd.pulse)

    def test_missing_type_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_sgc_command(payload(pulse=1500))

    def test_wrong_type_field_is_rejected(self):
        """`type` present but not a non-empty string never reaches the handler."""
        for command_type in (42, None, True, ""):
            with self.subTest(type=command_type):
                with self.assertRaises(ValueError):
                    parse_sgc_command(payload(type=command_type, pulse=1500))

    def test_malformed_json_is_rejected(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_sgc_command(b'{"type": "servo", "pulse": ')

    def test_invalid_utf8_is_rejected(self):
        with self.assertRaises(UnicodeDecodeError):
            parse_sgc_command(b"\xff\xfe\x00")

    def test_non_object_json_is_rejected(self):
        for raw in (b"[1, 2, 3]", b'"servo"', b"42", b"null"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_sgc_command(raw)

    def test_non_servo_commands_keep_their_free_form_fields(self):
        """Only `servo` payloads are range-checked; other types pass through."""
        cmd = parse_sgc_command(
            payload(type="select_target", bbox=[1, 2, 30, 40],
                    class_name="person", confidence=0.9))
        self.assertEqual(cmd.command_type, "select_target")
        self.assertEqual(cmd.bbox, [1, 2, 30, 40])

    def test_unknown_type_is_still_queued(self):
        """Forward compatibility: an unrecognised `type` is not a parse error.

        The handler reports unknown commands as such and sends nothing; only a
        missing/non-string `type` is malformed.
        """
        cmd = parse_sgc_command(payload(type="something_new"))
        self.assertEqual(cmd.command_type, "something_new")


class TestServoHandlerWiring(unittest.TestCase):
    """How a queued servo command becomes a channel - read from the source.

    Importing ``autonomous_drone_main`` would load the camera, the detector and
    the argument parser, so the wiring is asserted textually, following the
    pattern established by ``test_app_config``.
    """

    def _servo_branch(self):
        """The `elif cmd.command_type == "servo":` block of the SGC handler."""
        head, _, rest = MAIN_SOURCE.partition('elif cmd.command_type == "servo":')
        self.assertIn("servo", head + rest, "servo branch not found")
        branch, _, _ = rest.partition("elif cmd.command_type:")
        return branch

    def test_missing_channel_falls_through_to_automatic_selection(self):
        """channel absent -> cmd.channel is None -> resolve_servo_channel(None).

        That resolves to the gimbal channel discovered from SERVOx_FUNCTION
        (or the pinned --servo-channel); there is deliberately no hardcoded
        fallback channel.
        """
        branch = self._servo_branch()
        self.assertIn("requested = cmd.channel", branch)
        self.assertIn("resolve_servo_channel(requested)", branch)
        self.assertNotIn("= 8", branch)

    def test_explicit_channel_is_passed_through(self):
        branch = self._servo_branch()
        self.assertIn('source = "explicit" if requested is not None else "auto"',
                      branch)

    def test_servo_branch_sends_no_rc_override_and_writes_no_params(self):
        """The camera-servo path is DO_SET_SERVO only - no RC override, no PARAM_SET."""
        branch = self._servo_branch()
        self.assertNotIn("allow_rc_override", branch)
        self.assertNotIn("rc_channels", branch)
        self.assertNotIn("param", branch.lower())

    def test_receiver_ships_no_override_or_param_writes(self):
        self.assertNotIn("rc_channels_override", RECEIVER_SOURCE)
        self.assertNotIn("param_set", RECEIVER_SOURCE)
        self.assertNotIn("PARAM_SET", RECEIVER_SOURCE)

    def test_command_port_stays_9002(self):
        self.assertIn("port: int = 9002", RECEIVER_SOURCE)
        self.assertIn("default=_default('sgc_cmd_port', 9002)", MAIN_SOURCE)


class TestReceiverOverUDP(unittest.TestCase):
    """One datagram in on a real (ephemeral) UDP socket, one command queued.

    The socket uses port 0 so the OS picks a free port - the production
    default (9002) is asserted separately in ``TestServoHandlerWiring`` and is
    never bound here, so the suite cannot collide with a running system.
    """

    @classmethod
    def setUpClass(cls):
        cls.receiver = SGCCommandReceiver(port=0)
        cls.receiver.start()
        # port=0 lets the OS choose; the bound port comes back off the socket.
        cls.port = cls.receiver._sock.getsockname()[1]
        cls.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @classmethod
    def tearDownClass(cls):
        cls.sock.close()
        cls.receiver.stop()

    def _send_expect_queued(self, raw):
        self.receiver.drain_commands()
        self.sock.sendto(raw, ("127.0.0.1", self.port))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            cmd = self.receiver.pop_command()
            if cmd is not None:
                return cmd
            time.sleep(0.02)
        self.fail("valid SGC command was never queued")

    def _send_expect_rejected(self, raw):
        self.receiver.drain_commands()
        # assertLogs fails when *no* warning arrives, which is exactly the
        # failure mode we care about: a bad packet getting queued silently.
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self.sock.sendto(raw, ("127.0.0.1", self.port))
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not logs.records:
                time.sleep(0.02)
            self.assertTrue(logs.records, "packet was not rejected by the listener")
        self.assertIsNone(self.receiver.pop_command())

    def test_servo_with_channel_is_queued(self):
        cmd = self._send_expect_queued(payload(type="servo", pulse=1700, channel=8))
        self.assertEqual((cmd.command_type, cmd.pulse, cmd.channel),
                         ("servo", 1700, 8))

    def test_servo_without_channel_is_queued_as_automatic(self):
        cmd = self._send_expect_queued(payload(type="servo", pulse=1500))
        self.assertEqual(cmd.command_type, "servo")
        self.assertIsNone(cmd.channel)

    def test_channel_zero_is_dropped(self):
        self._send_expect_rejected(payload(type="servo", pulse=1500, channel=0))

    def test_channel_out_of_range_is_dropped(self):
        self._send_expect_rejected(payload(type="servo", pulse=1500, channel=17))

    def test_string_channel_is_dropped(self):
        self._send_expect_rejected(payload(type="servo", pulse=1500, channel="8"))

    def test_out_of_range_pulse_is_dropped_not_clamped(self):
        self._send_expect_rejected(payload(type="servo", pulse=3000))

    def test_missing_type_is_dropped(self):
        self._send_expect_rejected(payload(pulse=1500))

    def test_non_string_type_is_dropped(self):
        self._send_expect_rejected(payload(type=7, pulse=1500))

    def test_malformed_json_is_dropped(self):
        self._send_expect_rejected(b'{"type": "servo", "pulse": ')


if __name__ == "__main__":
    unittest.main()
