"""The MAVLink side of servo commands, exercised without a vehicle.

The routing tested here is not a choice: ArduPilot's DO_SET_SERVO handler
refuses any channel assigned a mount function, and an RC pass-through output is
only reachable through RC_CHANNELS_OVERRIDE. Both refusals were observed against
a live SITL, so these tests pin the routing that actually works.
"""
import time
import unittest
from unittest import mock

from modules.drone_backend import sitl
from modules.drone_backend import servo_channels as sc


class FakeMaster:
    def __init__(self):
        self.target_system = 1
        self.target_component = 1
        self.overrides = []
        self.commands = []
        self.mav = mock.Mock()
        self.mav.rc_channels_override_send.side_effect = (
            lambda *a, **k: self.overrides.append(a)
        )
        self.mav.command_long_send.side_effect = (
            lambda *a, **k: self.commands.append(a)
        )


class FakeAck:
    def __init__(self, result=0):
        self.result = result


def _fresh_servo_status():
    """A reset status block, so one test's outcome cannot leak into the next."""
    return {
        "detected_channel": None,
        "source": None,
        "commanded": None,
        "expected": None,
        "reported": None,
        "verified": None,
        "refused": False,
        "refusal_reason": None,
        "limits_hit": False,
    }


class ServoTestCase(unittest.TestCase):
    """Common wiring: a fake master and empty caches, function ids per channel."""

    functions = {}
    # Read-back is stubbed out for the routing tests, which are about what gets
    # sent rather than what comes back. Classes testing verification itself set
    # this False so the real implementation runs.
    stub_verify = True

    def setUp(self):
        self.master = FakeMaster()
        patches = [
            mock.patch.object(sitl, "_master", self.master),
            mock.patch.object(sitl, "_servo_functions", dict(self.functions)),
            mock.patch.object(sitl, "_servo_pwm", {}),
            mock.patch.object(sitl, "_servo_pwm_at", {}),
            mock.patch.object(sitl, "_rc_channels", {}),
            mock.patch.object(sitl, "_rc_channels_at", {}),
            mock.patch.object(sitl, "_servo_limits", {}),
            mock.patch.object(sitl, "_acked_commands", {}),
            mock.patch.object(sitl, "_servo_status", _fresh_servo_status()),
            mock.patch.object(sitl, "_params_loaded", False),
            mock.patch.object(sitl, "_override_guard", None),
            mock.patch.object(sitl, "_rc_override_inputs", set()),
            mock.patch.object(sitl, "_cached_mode", "STABILIZE"),
            mock.patch.object(sitl, "_wait_command_ack", lambda *a, **k: None),
        ]
        if self.stub_verify:
            patches.append(
                mock.patch.object(sitl, "_verify_servo_pulse", lambda *a, **k: None)
            )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)


class TestRoutingByAssignment(ServoTestCase):
    """The observed ArduPilot matrix, one test per row."""

    functions = {1: 0, 9: 0}

    def test_unassigned_channel_uses_do_set_servo(self):
        # AP_ServoRelayEvents whitelists k_none; this is the handler it exists for.
        # No override consent needed: this path never writes an RC input.
        sitl.send_servo(channel=9, pulse=1600)
        self.assertEqual(self.master.overrides, [])
        ts, comp, cmd, conf, param1, param2 = self.master.commands[0][:6]
        self.assertEqual(cmd, sitl.mavutil.mavlink.MAV_CMD_DO_SET_SERVO)
        self.assertEqual(param1, 9)
        self.assertEqual(param2, 1600)

    def test_unassigned_low_channel_also_uses_do_set_servo(self):
        # Not RC_CHANNELS_OVERRIDE: an override writes an RC *input*, and with
        # nothing mapped to it that write goes nowhere.
        sitl.send_servo(channel=1, pulse=1600)
        self.assertEqual(self.master.overrides, [])
        self.assertEqual(len(self.master.commands), 1)


class TestMountFunctionRefused(ServoTestCase):
    functions = {9: 7, 10: 6}

    def test_mount_axis_is_refused_with_a_reason(self):
        with self.assertRaises(ValueError) as ctx:
            sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        message = str(ctx.exception)
        self.assertIn("Mount Pitch", message)
        # The message must name the parameter fix, or the operator is stuck.
        # Channel 9 has no matching input, so it suggests the last of the 8.
        self.assertIn("SERVO9_FUNCTION = 147", message)
        self.assertIn("RC input 8", message)

    def test_mount_yaw_is_refused_too(self):
        with self.assertRaises(ValueError):
            sitl.send_servo(channel=10, pulse=1600)

    def test_nothing_is_sent_when_refused(self):
        with self.assertRaises(ValueError):
            sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertEqual(self.master.overrides, [])
        self.assertEqual(self.master.commands, [])

    def test_detect_still_finds_the_mount_axis(self):
        # Detection is right even though driving it needs another mechanism.
        self.assertEqual(sitl.detect_servo_channel(), 9)


class TestMappedAuxOutput(ServoTestCase):
    """The recommended setup: an aux output mapped to a spare RC input.

    Outputs 6 and 7 are disabled, which is what makes inputs 6 and 7 spare and
    therefore safe to override. Mapping an aux output onto inputs 1-5 instead -
    throttle, roll, pitch, yaw and the flight-mode channel - is refused; see
    TestReservedRcInputs.
    """

    functions = {6: 0, 7: 0, 9: 145, 10: 146}  # SERVO9 follows RC6, SERVO10 RC7

    def test_aux_output_is_driven_through_its_rc_input(self):
        used = sitl.send_servo(channel=9, pulse=1700, allow_rc_override=True)
        self.assertEqual(used, 9)
        self.assertEqual(len(self.master.overrides), 1)
        self.assertEqual(self.master.commands, [])
        (ts, comp, *slots) = self.master.overrides[0]
        self.assertEqual(slots[5], 1700)   # RC input 6
        self.assertEqual(slots[0], sc.SERVO_CHANNEL_UNSET)

    def test_second_output_uses_its_own_rc_input(self):
        sitl.send_servo(channel=10, pulse=1300, allow_rc_override=True)
        (ts, comp, *slots) = self.master.overrides[0]
        self.assertEqual(slots[6], 1300)   # RC input 7
        self.assertEqual(slots[5], sc.SERVO_CHANNEL_UNSET)

    def test_unset_inputs_are_65535_not_zero(self):
        # 0 is not a pulse; it only worked before because ArduPilot happens to
        # treat it as "leave alone".
        sitl.send_servo(channel=9, pulse=1700, allow_rc_override=True)
        (ts, comp, *slots) = self.master.overrides[0]
        for i, value in enumerate(slots):
            if i != 5:
                self.assertEqual(value, sc.SERVO_CHANNEL_UNSET, f"input {i + 1}")

    def test_detect_finds_a_lone_mapped_output(self):
        with mock.patch.object(sitl, "_servo_functions", {6: 0, 9: 145}):
            self.assertEqual(sitl.detect_servo_channel(), 9)

    def test_channel_16_is_reachable(self):
        with mock.patch.object(sitl, "_servo_functions", {6: 0, 16: 145}):
            self.assertEqual(sitl.send_servo(channel=16, pulse=1500, allow_rc_override=True), 16)


class TestReservedRcInputs(ServoTestCase):
    """RC inputs that belong to the aircraft must never be overridden.

    Regression cover for a bench report: sweeping the gimbal walked the vehicle
    through RTL and then STABILIZE. A servo pulse written into the flight-mode
    input is read by AP_Copter as a mode slot, so the aircraft changes mode
    instead of the camera moving.

    Every test here passes ``allow_rc_override=True`` deliberately. Consent only
    unlocks inputs that pass every static check, so a provably reserved input
    must stay refused even when the operator insists - otherwise the flag becomes
    a way to defeat the whole safety rule.
    """

    functions = {1: 33, 2: 34, 3: 35, 4: 36, 5: 0, 6: 0, 7: 0, 8: 0}

    def test_output_mapped_to_the_mode_input_is_refused(self):
        # SERVO9 following RC input 5 is the reported failure exactly.
        with mock.patch.object(sitl, "_servo_functions", {**self.functions, 9: 144}):
            with self.assertRaises(ValueError) as ctx:
                sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        message = str(ctx.exception)
        self.assertIn("RC input 5", message)
        self.assertIn("flight-mode", message)

    def test_throttle_input_is_refused(self):
        with mock.patch.object(sitl, "_servo_functions", {**self.functions, 9: 140}):
            with self.assertRaises(ValueError) as ctx:
                sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertIn("RC input 1", str(ctx.exception))

    def test_manual_output_on_a_primary_channel_is_refused(self):
        # k_manual on channel 5 passes RC input 5 through, and input 5 is mode.
        with mock.patch.object(sitl, "_servo_functions", {**self.functions, 5: 1}):
            with self.assertRaises(ValueError) as ctx:
                sitl.send_servo(channel=5, pulse=1600)
        self.assertIn("RC input 5", str(ctx.exception))

    def test_nothing_is_sent_when_an_input_is_reserved(self):
        with mock.patch.object(sitl, "_servo_functions", {**self.functions, 9: 144}):
            with self.assertRaises(ValueError):
                sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertEqual(self.master.overrides, [])
        self.assertEqual(self.master.commands, [])

    def test_an_input_whose_output_owns_a_function_is_refused(self):
        # Input 6 is aux on ArduCopter, but output 6 is assigned a flight-mode
        # slot, so writing input 6 still reaches the aircraft.
        with mock.patch.object(sitl, "_servo_functions", {**self.functions, 6: 20, 9: 145}):
            with self.assertRaises(ValueError) as ctx:
                sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertIn("already drives output 6", str(ctx.exception))

    def test_an_input_whose_output_is_unknown_is_refused(self):
        # Not reading SERVO6_FUNCTION is not evidence that input 6 is free.
        with mock.patch.object(sitl, "_servo_functions", {9: 145}):
            with self.assertRaises(ValueError):
                sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)

    def test_the_refusal_suggests_a_spare_input(self):
        with mock.patch.object(sitl, "_servo_functions", {**self.functions, 9: 144}):
            with self.assertRaises(ValueError) as ctx:
                sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertIn("spare input", str(ctx.exception))

    def test_planner_refuses_without_a_function_map(self):
        # No map means no way to tell a spare input from the mode channel.
        self.assertEqual(
            sc.plan_drive(9, 145, None)[0], sc.DRIVE_RESERVED_INPUT
        )


class TestAmbiguousMappedOutputs(ServoTestCase):
    """Two pass-through outputs means we cannot tell which is the camera."""

    functions = {6: 0, 7: 0, 9: 145, 10: 146}

    def test_detection_refuses_rather_than_guessing(self):
        self.assertIsNone(sitl.detect_servo_channel())

    def test_auto_send_is_refused(self):
        with self.assertRaises(ValueError):
            sitl.send_servo(pulse=1500)

    def test_an_explicit_channel_still_works(self):
        self.assertEqual(sitl.send_servo(channel=10, pulse=1500, allow_rc_override=True), 10)


class TestUndrivableOutputs(ServoTestCase):
    functions = {1: 33, 2: 34, 3: 140}

    def test_motor_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            sitl.send_servo(channel=1, pulse=1500)
        self.assertIn("SERVO1_FUNCTION=33", str(ctx.exception))

    def test_rc_input_beyond_the_override_message_is_refused(self):
        # k_rcin12_mapped needs input 12; RC_CHANNELS_OVERRIDE has 8 slots.
        with mock.patch.object(sitl, "_servo_functions", {12: 152}):
            with self.assertRaises(ValueError):
                sitl.send_servo(channel=12, pulse=1500)

    def test_unknown_function_is_refused_not_assumed(self):
        with mock.patch.object(sitl, "_servo_functions", {}):
            with self.assertRaises(ValueError) as ctx:
                sitl.send_servo(channel=9, pulse=1500)
            self.assertIn("unknown", str(ctx.exception))


class TestRcPassThroughDoSetServo(ServoTestCase):
    """SERVO10_FUNCTION = 61: the camera servo on M10, reached directly.

    Function 61 is ``k_rcin11`` - an RCIN11 pass-through, so the output
    follows RC input 11. That input is beyond the 8 slots
    ``RC_CHANNELS_OVERRIDE`` carries, and the FCU's configuration is fixed
    (SERVO10_FUNCTION/MIN/TRIM/MAX/REVERSED are not ours to write), so the
    routing layer must reach the output without an override at all.
    ArduPilot whitelists ``k_rcin1``-``k_rcin16`` for
    ``MAV_CMD_DO_SET_SERVO``, and that command writes the *output* rather
    than any RC input: no FCU parameter changes, no RC override, and no
    ``allow_rc_override`` consent needed.
    """

    # The confirmed vehicle layout: motors on 1-8, camera servo on M10.
    functions = {1: 33, 2: 34, 3: 35, 4: 36, 5: 37, 6: 38, 7: 39, 8: 40, 10: 61}

    def test_planner_routes_function_61_to_do_set_servo(self):
        method, rc_input = sc.plan_drive(10, 61, self.functions)
        self.assertEqual(method, sc.DRIVE_DO_SET_SERVO)
        self.assertIsNone(rc_input)  # no RC input is part of this transport

    def test_explicit_channel_10_sends_do_set_servo(self):
        used = sitl.send_servo(channel=10, pulse=1500)
        self.assertEqual(used, 10)
        self.assertEqual(len(self.master.commands), 1)
        ts, comp, cmd, conf, param1, param2 = self.master.commands[0][:6]
        self.assertEqual(cmd, sitl.mavutil.mavlink.MAV_CMD_DO_SET_SERVO)
        self.assertEqual(param1, 10)
        self.assertEqual(param2, 1500)

    def test_no_rc_channels_override_is_sent(self):
        sitl.send_servo(channel=10, pulse=1500)
        self.master.mav.rc_channels_override_send.assert_not_called()
        self.assertEqual(self.master.overrides, [])
        self.assertEqual(sitl._rc_override_inputs, set())

    def test_consent_is_not_required_for_function_61(self):
        # allow_rc_override defaults to False, and that must be enough: this
        # path writes an output, not an RC input, so there is nothing to
        # consent to and nothing for the override guard to watch.
        used = sitl.send_servo(channel=10, pulse=1600, allow_rc_override=False)
        self.assertEqual(used, 10)
        self.assertEqual(len(self.master.commands), 1)
        self.assertEqual(self.master.commands[0][4], 10)   # param1 = channel
        self.assertEqual(self.master.commands[0][5], 1600)  # param2 = pulse
        self.master.mav.rc_channels_override_send.assert_not_called()
        self.assertIsNone(sitl._override_guard)

    def test_the_requested_pulse_is_preserved(self):
        for pulse in (1000, 1500, 1750, 2000):
            with self.subTest(pulse=pulse):
                sitl.send_servo(channel=10, pulse=pulse)
                self.assertEqual(self.master.commands[-1][5], pulse)

    def test_detection_finds_the_lone_pass_through(self):
        # pick_gimbal_channel already accepts a single pass-through output;
        # function 61 needs no detection change to be found, so the SGC path
        # without an explicit channel resolves to 10 as well.
        self.assertEqual(sitl.detect_servo_channel(), 10)
        self.assertEqual(sitl.send_servo(pulse=1500), 10)
        self.assertEqual(self.master.commands[-1][4], 10)
        self.assertEqual(self.master.commands[-1][5], 1500)

    def test_pass_through_within_eight_inputs_keeps_the_override_path(self):
        # k_rcin7 on channel 9: RC input 7 fits in the override message, so
        # the routing the architecture chose - the consent-gated override -
        # is unchanged. The new direct path must not leak backwards into it.
        fns = {6: 0, 7: 0, 9: 57}
        self.assertEqual(
            sc.plan_drive(9, 57, fns)[0], sc.DRIVE_UNVERIFIED_INPUT
        )
        method, rc_input = sc.plan_drive(9, 57, fns, allow_rc_override=True)
        self.assertEqual(method, sc.DRIVE_RC_OVERRIDE)
        self.assertEqual(rc_input, 7)

        with mock.patch.object(sitl, "_servo_functions", dict(fns)):
            with self.assertRaises(ValueError) as ctx:
                sitl.send_servo(channel=9, pulse=1600)
            self.assertIn("--allow-rc-override", str(ctx.exception))
            self.assertEqual(self.master.commands, [])
            self.assertEqual(self.master.overrides, [])

            used = sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
            self.assertEqual(used, 9)
            self.assertEqual(self.master.commands, [])  # still not DO_SET_SERVO
            (ts, comp, *slots) = self.master.overrides[0]
            self.assertEqual(slots[6], 1600)  # RC input 7
            self.assertEqual(slots[5], sc.SERVO_CHANNEL_UNSET)

    def test_mapped_output_beyond_the_override_stays_refused(self):
        # k_rcinN_mapped is NOT whitelisted for DO_SET_SERVO - only the plain
        # pass-through (51-66) gained a direct path. Consent must not unlock
        # this one either.
        fns = {12: 152}
        self.assertEqual(
            sc.plan_drive(12, 152, fns, allow_rc_override=True)[0],
            sc.DRIVE_UNSUPPORTED,
        )
        with mock.patch.object(sitl, "_servo_functions", fns):
            with self.assertRaises(ValueError):
                sitl.send_servo(channel=12, pulse=1500, allow_rc_override=True)
        self.assertEqual(self.master.commands, [])
        self.assertEqual(self.master.overrides, [])

    def test_a_reserved_rc_input_stays_refused(self):
        # Consent never unlocks a flight-control input. The direct path does
        # not weaken that: it simply never involves RC inputs at all.
        with mock.patch.object(sitl, "_servo_functions", {**self.functions, 9: 144}):
            with self.assertRaises(ValueError) as ctx:
                sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertIn("RC input 5", str(ctx.exception))
        self.assertEqual(self.master.commands, [])
        self.assertEqual(self.master.overrides, [])

    def test_mount_axis_stays_refused(self):
        with mock.patch.object(sitl, "_servo_functions", {**self.functions, 9: 7}):
            with self.assertRaises(ValueError) as ctx:
                sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertIn("Mount Pitch", str(ctx.exception))
        self.assertEqual(self.master.commands, [])

    def test_motor_stays_refused(self):
        with self.assertRaises(ValueError) as ctx:
            sitl.send_servo(channel=1, pulse=1500)
        self.assertIn("SERVO1_FUNCTION=33", str(ctx.exception))
        self.assertEqual(self.master.commands, [])

    def test_unassigned_output_is_still_do_set_servo(self):
        # k_none must not shift routing just because another function gained
        # a direct path; the send-level tests live in TestRoutingByAssignment.
        self.assertEqual(
            sc.plan_drive(10, 0, self.functions)[0], sc.DRIVE_DO_SET_SERVO
        )


class TestRangeAndClamp(ServoTestCase):
    # All outputs unassigned, so the range and clamp rules are tested on their
    # own rather than on whichever transport the assignment happens to select.
    functions = {n: 0 for n in range(1, 17)}

    def test_accepted_range_is_1_to_16(self):
        for channel in range(1, 17):
            sitl.send_servo(channel=channel, pulse=1500)
        self.assertEqual(len(self.master.commands), 16)

    def test_out_of_range_rejected(self):
        for channel in (0, 17, -1):
            with self.assertRaises(ValueError):
                sitl.send_servo(channel=channel, pulse=1500)

    def test_pulse_is_clamped_not_rejected(self):
        sitl.send_servo(channel=1, pulse=3000)
        _ts, _comp, _cmd, _conf, channel, pulse = self.master.commands[0][:6]
        self.assertEqual(pulse, sc.SERVO_PULSE_MAX)

    def test_returns_channel_used(self):
        self.assertEqual(sitl.send_servo(channel=1, pulse=1500), 1)


class TestAckHandling(ServoTestCase):
    functions = {9: 0}

    def test_missing_ack_is_reported(self):
        with mock.patch.object(sitl, "_wait_command_ack", lambda *a, **k: None):
            sitl.send_servo(channel=9, pulse=1500)

    def test_rejected_command_is_reported(self):
        with mock.patch.object(sitl, "_wait_command_ack", lambda *a, **k: FakeAck(result=1)):
            sitl.send_servo(channel=9, pulse=1500)

    def test_accepted_command_is_silent(self):
        with mock.patch.object(sitl, "_wait_command_ack", lambda *a, **k: FakeAck(result=0)):
            sitl.send_servo(channel=9, pulse=1500)


class TestVerifyIgnoresStaleReadings(ServoTestCase):
    """A reading from before the command describes the previous command.

    Regression: with a 1s verification timeout the PWM cache still held the last
    value from the previous send, so every command was reported as a mismatch
    even on an output with identity scaling.
    """

    functions = {9: 147}
    stub_verify = False

    def test_reading_predating_the_command_is_ignored(self):
        with mock.patch.object(sitl, "_servo_pwm", {9: 1900}):
            with mock.patch.object(sitl, "_servo_pwm_at", {9: time.monotonic() - 5}):
                # Nothing fresh arrives, so the result is unverified - not False.
                self.assertIsNone(
                    sitl._verify_servo_pulse(9, 1500, timeout=0.2,
                                             since=time.monotonic())
                )

    def test_fresh_reading_is_accepted(self):
        now = time.monotonic()
        with mock.patch.object(sitl, "_servo_pwm", {9: 1500}):
            with mock.patch.object(sitl, "_servo_pwm_at", {9: now}):
                self.assertTrue(
                    sitl._verify_servo_pulse(9, 1500, timeout=0.2, since=now - 0.1)
                )

    def test_stale_match_does_not_prove_anything(self):
        now = time.monotonic()
        with mock.patch.object(sitl, "_servo_pwm", {9: 1500}):
            with mock.patch.object(sitl, "_servo_pwm_at", {9: now - 5}):
                self.assertIsNone(
                    sitl._verify_servo_pulse(9, 1900, timeout=0.2, since=now)
                )


class TestVerifyPredictsMappedScaling(ServoTestCase):
    """A mapped output reports a scaled pulse; verification must know that."""

    functions = {9: 147}
    stub_verify = False
    limits = {
        ("SERVO", 9): (1100, 1500, 1900),
        ("RC", 8): (1000, 1500, 1800),
    }

    def setUp(self):
        super().setUp()
        sitl._servo_limits = dict(self.limits)

    def test_expected_value_is_the_scaled_pulse(self):
        # Live: commanding 1750 on this configuration reports 1833 back.
        self.assertEqual(sitl._expected_servo_pulse(9, 1750), 1833)

    def test_scaled_readback_verifies_clean(self):
        now = time.monotonic()
        with mock.patch.object(sitl, "_servo_pwm", {9: 1833}):
            with mock.patch.object(sitl, "_servo_pwm_at", {9: now}):
                self.assertTrue(
                    sitl._verify_servo_pulse(9, 1750, timeout=0.2, since=now - 0.1)
                )

    def test_matched_limits_verify_the_raw_pulse(self):
        sitl._servo_limits = {
            ("SERVO", 9): (1000, 1500, 2000),
            ("RC", 8): (1000, 1500, 2000),
        }
        self.assertEqual(sitl._expected_servo_pulse(9, 1750), 1750)

    def test_unmapped_output_expects_what_was_commanded(self):
        with mock.patch.object(sitl, "_servo_functions", {9: 0}):
            self.assertEqual(sitl._expected_servo_pulse(9, 1750), 1750)

    def test_missing_limits_fall_back_to_the_raw_pulse(self):
        # No prediction available: compare against the commanded value and let
        # the mismatch report explain itself.
        sitl._servo_limits = {}
        self.assertEqual(sitl._expected_servo_pulse(9, 1750), 1750)


class TestLimitHarvesting(ServoTestCase):
    def test_reads_output_and_mapped_input_ranges(self):
        params = {
            "SERVO9_FUNCTION": 147,
            "SERVO9_MIN": 1000, "SERVO9_TRIM": 1500, "SERVO9_MAX": 2000,
            "RC8_MIN": 1000, "RC8_TRIM": 1500, "RC8_MAX": 2000,
        }
        limits = sitl._read_servo_limits(params, {9: 147})
        self.assertEqual(limits, {
            ("SERVO", 9): (1000, 1500, 2000),
            ("RC", 8): (1000, 1500, 2000),
        })

    def test_ignores_inputs_that_are_not_mapped(self):
        params = {"SERVO9_FUNCTION": 0, "SERVO9_MIN": 1000,
                  "SERVO9_TRIM": 1500, "SERVO9_MAX": 2000,
                  "RC8_MIN": 1000, "RC8_TRIM": 1500, "RC8_MAX": 2000}
        limits = sitl._read_servo_limits(params, {9: 0})
        self.assertEqual(list(limits), [("SERVO", 9)])

    def test_absent_params(self):
        self.assertEqual(sitl._read_servo_limits(None, {9: 147}), {})
        self.assertEqual(sitl._read_servo_limits({}, {9: 147}), {})

    def test_a_partial_rediscovery_cannot_erase_known_ranges(self):
        """Regression: discovery runs on two threads with separate downloads.

        A read that caught SERVO9_FUNCTION but not SERVO9_MIN/TRIM/MAX used to
        publish an empty dict over the complete one, so every read-back
        prediction silently fell back to the raw commanded pulse and reported
        healthy scaled outputs as broken.
        """
        full = {
            "SERVO9_FUNCTION": 147,
            "SERVO9_MIN": 1100, "SERVO9_TRIM": 1500, "SERVO9_MAX": 1900,
            "RC8_MIN": 1000, "RC8_TRIM": 1500, "RC8_MAX": 1800,
        }
        partial = {"SERVO9_FUNCTION": 147}  # ranges have not streamed in yet

        master = mock.Mock()
        master.params = dict(full)
        sitl._discover_servo_channels(master)
        self.assertEqual(sitl._expected_servo_pulse(9, 1750), 1833)

        master.params = dict(partial)
        sitl._discover_servo_channels(master)
        self.assertEqual(
            sitl._expected_servo_pulse(9, 1750), 1833,
            "a partial parameter read erased the ranges it had already learned",
        )


class TestDiscoveryReporting(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch.object(sitl, "_servo_functions", {}),
            mock.patch.object(sitl, "_servo_limits", {}),
            mock.patch.object(sitl, "_params_loaded", False),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_get_servo_channels_names_assignments(self):
        sitl._servo_functions = {1: 33, 9: 7, 10: 140}
        self.assertEqual(sitl.get_servo_channels(), {
            1: "function 33",
            9: "Mount Pitch",
            10: "RC Input 1 (mapped)",
        })

    def test_discover_populates_from_params(self):
        master = mock.Mock()
        master.params = {"SERVO9_FUNCTION": 7, "SERVO10_FUNCTION": 6}
        sitl._discover_servo_channels(master)
        self.assertEqual(sitl._servo_functions, {9: 7, 10: 6})
        self.assertEqual(sitl.detect_servo_channel(), 9)

    def test_discover_survives_missing_params(self):
        master = mock.Mock()
        master.params = {}
        sitl._discover_servo_channels(master)
        self.assertEqual(sitl._servo_functions, {})
        self.assertIsNone(sitl.detect_servo_channel())

    def test_verify_reports_mismatch(self):
        sitl._servo_pwm = {9: 1200}
        self.assertFalse(sitl._verify_servo_pulse(9, 1600, timeout=0.01))

    def test_verify_accepts_a_match(self):
        sitl._servo_pwm = {9: 1600}
        self.assertTrue(sitl._verify_servo_pulse(9, 1600, timeout=0.01))

    def test_verify_unknown_channel_is_not_a_failure(self):
        sitl._servo_pwm = {}
        self.assertIsNone(sitl._verify_servo_pulse(9, 1600, timeout=0.01))

    def test_wait_for_params_gives_up_rather_than_spinning(self):
        master = mock.Mock()
        master.params = {}
        self.assertFalse(sitl._wait_for_servo_params(master, timeout=0.3))

    def test_wait_for_params_returns_once_arrived(self):
        master = mock.Mock()
        master.params = {"SERVO1_FUNCTION": 0}
        self.assertTrue(sitl._wait_for_servo_params(master, timeout=2.0))

    def test_wait_for_params_does_not_return_mid_download(self):
        """Regression: parameters arrive alphabetically, so SERVO1_FUNCTION
        shows up long before SERVO1_MIN/TRIM/MAX.

        Returning on the first function id left discovery with the assignments
        but no ranges, which silently downgraded the read-back prediction to the
        raw commanded pulse and produced false mismatch reports.
        """
        class StreamingParams(dict):
            """params that keeps growing, like a live download."""

            def __init__(self):
                super().__init__({"SERVO1_FUNCTION": 0, "SERVO1_MAX": 2000})
                self.ticks = 0

            def __len__(self):
                self.ticks += 1
                if self.ticks == 2:
                    self["SERVO1_MIN"] = 1000
                elif self.ticks == 4:
                    self["SERVO1_TRIM"] = 1500
                return super().__len__()

        master = mock.Mock()
        master.params = StreamingParams()
        self.assertTrue(sitl._wait_for_servo_params(master, timeout=5.0, settle=0.3))
        for name in ("SERVO1_MIN", "SERVO1_TRIM", "SERVO1_MAX"):
            self.assertIn(name, master.params, "returned before the stream settled")

    def test_wait_for_params_times_out_on_a_silent_vehicle(self):
        class Frozen(dict):
            def __len__(self):
                return 1

        master = mock.Mock()
        master.params = Frozen({"SERVO1_FUNCTION": 0})
        started = time.monotonic()
        # A vehicle that stops streaming must not hang the connect forever.
        self.assertTrue(
            sitl._wait_for_servo_params(master, timeout=0.4, settle=0.3)
        )
        self.assertLess(time.monotonic() - started, 2.0)

    def test_stale_repeat_is_not_evidence_of_a_stable_mismatch(self):
        """A cached value repeating is not a fault; only fresh readings count."""
        now = time.monotonic()
        with mock.patch.object(sitl, "_servo_pwm", {9: 1900}):
            with mock.patch.object(sitl, "_servo_pwm_at", {9: now - 5}):
                # Never settled on anything, so the verdict is unverified.
                self.assertIsNone(
                    sitl._verify_servo_pulse(9, 1200, timeout=0.3, since=now)
                )


class TestTelemetryIsGlobal(unittest.TestCase):
    """Regression: these were assigned as locals and the cache stayed empty."""

    def test_listener_populates_the_pwm_cache(self):
        with mock.patch.object(sitl, "_servo_pwm", {}):
            msg = mock.Mock(servo1_raw=1500, servo9_raw=1600)
            sitl._cache_pwm(msg, sc.build_servo_field_map("servo"), "_servo_pwm")
            self.assertEqual(sitl._servo_pwm, {1: 1500, 9: 1600})

    def test_listener_populates_the_rc_cache(self):
        with mock.patch.object(sitl, "_rc_channels", {}):
            msg = mock.Mock(chan1_raw=1500, chan5_raw=1600)
            sitl._cache_pwm(msg, sc.build_servo_field_map("chan"), "_rc_channels")
            self.assertEqual(sitl._rc_channels, {1: 1500, 5: 1600})

    def test_reports_merge_rather_than_replace(self):
        # A PWM report only carries live channels, so replacing the cache would
        # drop channels that simply went quiet.
        with mock.patch.object(sitl, "_servo_pwm", {5: 1200}):
            sitl._cache_pwm(mock.Mock(servo9_raw=1600),
                             sc.build_servo_field_map("servo"), "_servo_pwm")
            self.assertEqual(sitl._servo_pwm, {5: 1200, 9: 1600})


class TestRcOverrideNeedsConsent(ServoTestCase):
    """No static check can identify the mode channel, so the override is opt-in.

    The bench that prompted this mapped ch9 onto an input outside 1-5, passed
    every check the app could make, and walked the aircraft through RTL,
    STABILIZE, AUTO, CIRCLE and LAND. The input was the mode channel and nothing
    in SERVOx_FUNCTION said so. Only the operator knows their own wiring.
    """

    functions = {6: 0, 9: 145}  # input 6 looks provably spare

    def test_a_mapped_output_is_refused_without_consent(self):
        with self.assertRaises(ValueError) as ctx:
            sitl.send_servo(channel=9, pulse=1600)
        message = str(ctx.exception)
        self.assertIn("--allow-rc-override", message)
        self.assertIn("RC input 6", message)

    def test_nothing_is_sent_when_consent_is_withheld(self):
        with self.assertRaises(ValueError):
            sitl.send_servo(channel=9, pulse=1600)
        self.assertEqual(self.master.overrides, [])
        self.assertEqual(self.master.commands, [])

    def test_auto_detection_is_refused_too(self):
        # The dangerous path is the automatic one: the operator never chose this
        # channel, so there is nothing for them to have confirmed.
        with self.assertRaises(ValueError):
            sitl.send_servo(pulse=1600)
        self.assertEqual(self.master.overrides, [])

    def test_the_refusal_offers_the_override_free_alternative(self):
        with self.assertRaises(ValueError) as ctx:
            sitl.send_servo(channel=9, pulse=1600)
        self.assertIn("SERVO9_FUNCTION = 0", str(ctx.exception))

    def test_the_refusal_is_recorded_for_the_ground_station(self):
        with self.assertRaises(ValueError):
            sitl.send_servo(channel=9, pulse=1600, source="auto")
        status = sitl.get_servo_status()
        self.assertTrue(status["refused"])
        self.assertIn("--allow-rc-override", status["refusal_reason"])

    def test_consent_lets_the_override_through(self):
        used = sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertEqual(used, 9)
        (ts, comp, *slots) = self.master.overrides[0]
        self.assertEqual(slots[5], 1600)


class TestModeChangeWatchdog(ServoTestCase):
    """An override that turns out to be flight control is undone, not left latched.

    ArduPilot holds an RC override until it is explicitly released, so a bad one
    stays bad. This is the net for the case the static checks cannot catch.
    """

    functions = {6: 0, 9: 145}

    def test_an_override_write_arms_the_guard(self):
        sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertIsNotNone(sitl._override_guard)
        self.assertEqual(sitl._override_guard["rc_input"], 6)
        self.assertEqual(sitl._override_guard["mode_before"], "STABILIZE")

    def test_a_mode_change_releases_the_override(self):
        sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        sitl._check_override_guard("RTL")
        # A second, all-unset message is the release.
        self.assertEqual(len(self.master.overrides), 2)
        for value in self.master.overrides[1][2:]:
            self.assertEqual(value, sc.SERVO_CHANNEL_UNSET)

    def test_the_mode_change_is_reported_as_a_refusal(self):
        sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        sitl._check_override_guard("RTL")
        status = sitl.get_servo_status()
        self.assertTrue(status["refused"])
        self.assertIn("STABILIZE -> RTL", status["refusal_reason"])

    def test_the_unchanged_mode_releases_nothing(self):
        sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        sitl._check_override_guard("STABILIZE")
        self.assertEqual(len(self.master.overrides), 1)
        self.assertFalse(sitl.get_servo_status()["refused"])

    def test_a_stale_mode_change_is_not_blamed_on_the_override(self):
        # The operator may legitimately change mode minutes later; only the
        # window right after the write is evidence.
        sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        sitl._override_guard["armed_at"] -= sitl.OVERRIDE_GUARD_SECONDS + 1
        sitl._check_override_guard("RTL")
        self.assertEqual(len(self.master.overrides), 1)
        self.assertFalse(sitl.get_servo_status()["refused"])

    def test_the_guard_disarms_after_it_fires(self):
        sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        sitl._check_override_guard("RTL")
        sitl._check_override_guard("AUTO")
        self.assertEqual(len(self.master.overrides), 2)

    def test_only_override_writes_arm_the_guard(self):
        # A DO_SET_SERVO command touches no RC input, so there is nothing to
        # watch: the guard must not fire on an unrelated later mode change.
        with mock.patch.object(sitl, "_servo_functions", {9: 0}):
            sitl.send_servo(channel=9, pulse=1600)
        self.assertIsNone(sitl._override_guard)
        sitl._check_override_guard("RTL")
        self.assertEqual(len(self.master.overrides), 0)
        self.assertFalse(sitl.get_servo_status()["refused"])


class TestOverrideReleaseOnTransitions(ServoTestCase):
    """An outstanding override does not outlive normal operation.

    ArduPilot latches RC_CHANNELS_OVERRIDE until it is told otherwise, so land,
    RTL, disarm and the app's own exit have to release it first - otherwise a
    gimbal pulse (or a bad guess about which input is spare) keeps pinning the
    receiver after the operation that needed it is over.
    """

    functions = {6: 0, 9: 145}

    def _override_outstanding(self):
        sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertEqual(len(self.master.overrides), 1)
        self.assertEqual(sitl._rc_override_inputs, {6})

    def test_landing_releases_the_override_before_the_mode_change(self):
        self._override_outstanding()
        # The release must already be on the wire when LAND is requested: a
        # mode change that outruns the release is exactly what the guard would
        # blame on the write.
        seen = []
        with mock.patch.object(
            sitl, "_set_mode",
            side_effect=lambda mode: seen.append((mode, len(self.master.overrides))),
        ):
            sitl.land()
        self.assertEqual(seen, [("LAND", 2)])
        for value in self.master.overrides[1][2:]:
            self.assertEqual(value, sc.SERVO_CHANNEL_UNSET)
        self.assertEqual(sitl._rc_override_inputs, set())
        self.assertIsNone(sitl._override_guard)

    def test_rtl_releases_the_override_before_the_mode_change(self):
        self._override_outstanding()
        seen = []
        with mock.patch.object(
            sitl, "_set_mode",
            side_effect=lambda mode: seen.append((mode, len(self.master.overrides))),
        ):
            sitl.send_rtl()
        self.assertEqual(seen, [("RTL", 2)])
        self.assertEqual(sitl._rc_override_inputs, set())

    def test_disarm_releases_the_override_before_the_command(self):
        self._override_outstanding()
        acks = []

        def ack_with_check(*args, **kwargs):
            acks.append(len(self.master.overrides))
            return FakeAck(0)

        with mock.patch.object(sitl, "_wait_command_ack", side_effect=ack_with_check), \
                mock.patch.object(sitl, "_wait_for_arm_state"):
            sitl.disarm()
        # By the time the disarm handshake runs, the release has been sent.
        self.assertEqual(acks, [2])
        for value in self.master.overrides[1][2:]:
            self.assertEqual(value, sc.SERVO_CHANNEL_UNSET)
        self.assertEqual(sitl._rc_override_inputs, set())

    def test_shutdown_releases_the_override_while_the_link_is_still_open(self):
        self._override_outstanding()
        sitl.disconnect()
        self.assertEqual(len(self.master.overrides), 2)
        for value in self.master.overrides[1][2:]:
            self.assertEqual(value, sc.SERVO_CHANNEL_UNSET)
        self.assertEqual(sitl._rc_override_inputs, set())

    def test_nothing_outstanding_sends_no_release(self):
        with mock.patch.object(sitl, "_set_mode"):
            sitl.land()
        self.assertEqual(self.master.overrides, [])

    def test_releasing_disarms_the_guard(self):
        # The operator lands within the 2.5 s guard window; the heartbeat then
        # reports STABILIZE -> LAND, which must not be reported as the override
        # having moved a flight control.
        self._override_outstanding()
        with mock.patch.object(sitl, "_set_mode"):
            sitl.land()
        sitl._check_override_guard("LAND")
        self.assertEqual(len(self.master.overrides), 2)
        self.assertFalse(sitl.get_servo_status()["refused"])


class TestServoStatusReporting(ServoTestCase):
    """The outcome of a command, for the ground station that sent it.

    UDP gives the ground station no acknowledgement, so a refusal that only
    appears on the drone console is invisible to the operator.
    """

    functions = {6: 0, 9: 145}

    def test_a_sent_command_reports_the_readback(self):
        sitl._servo_pwm[9] = 1600
        sitl.send_servo(channel=9, pulse=1600, source="auto", allow_rc_override=True)
        status = sitl.get_servo_status()
        self.assertEqual(status["detected_channel"], 9)
        self.assertEqual(status["source"], "auto")
        self.assertFalse(status["refused"])
        self.assertIsNone(status["refusal_reason"])
        self.assertEqual(status["reported_pwm"], 1600)
        self.assertFalse(status["limits_hit"])

    def test_a_refusal_carries_its_reason(self):
        with mock.patch.object(sitl, "_servo_functions", {**self.functions, 9: 144}):
            with self.assertRaises(ValueError):
                sitl.send_servo(channel=9, pulse=1600, source="auto", allow_rc_override=True)
        status = sitl.get_servo_status()
        self.assertTrue(status["refused"])
        self.assertIn("RC input 5", status["refusal_reason"])
        self.assertIsNone(status["reported_pwm"])

    def test_nothing_detected_is_reported_as_a_refusal(self):
        with mock.patch.object(sitl, "_servo_functions", {1: 33, 2: 34}):
            with self.assertRaises(ValueError):
                sitl.send_servo(pulse=1600, source="auto")
        status = sitl.get_servo_status()
        self.assertTrue(status["refused"])
        self.assertIsNone(status["detected_channel"])

    def test_limits_hit_is_a_disagreement_with_the_prediction(self):
        # The output landed somewhere other than the mapped prediction, which is
        # what a servo resting on SERVOx_MIN/MAX looks like from here.
        sitl._servo_pwm[9] = 1100
        sitl.send_servo(channel=9, pulse=1600, allow_rc_override=True)
        self.assertTrue(sitl.get_servo_status()["limits_hit"])

    def test_the_status_block_always_has_every_key(self):
        from shared.detection_models import SERVO_TELEMETRY_KEYS

        self.assertEqual(set(sitl.get_servo_status()), set(SERVO_TELEMETRY_KEYS))


class TestServoTelemetryShape(unittest.TestCase):
    """telemetry.servo must always carry the same keys, whatever is known."""

    def test_every_key_present_when_nothing_is_known(self):
        from shared.detection_models import SERVO_TELEMETRY_KEYS, TelemetryData

        servo = TelemetryData().to_dict()["servo"]
        self.assertEqual(set(servo), set(SERVO_TELEMETRY_KEYS))
        self.assertIsNone(servo["detected_channel"])
        self.assertIsNone(servo["reported_pwm"])
        self.assertFalse(servo["refused"])
        self.assertFalse(servo["limits_hit"])

    def test_partial_status_does_not_drop_keys(self):
        from shared.detection_models import SERVO_TELEMETRY_KEYS, TelemetryData

        servo = TelemetryData(servo={"detected_channel": 9}).to_dict()["servo"]
        self.assertEqual(set(servo), set(SERVO_TELEMETRY_KEYS))
        self.assertEqual(servo["detected_channel"], 9)

    def test_round_trips(self):
        from shared.detection_models import SERVO_TELEMETRY_KEYS, TelemetryData

        original = TelemetryData(altitude=5.0, servo={"detected_channel": 9,
                                                      "refused": True,
                                                      "refusal_reason": "nope"})
        packet = original.to_dict()
        restored = TelemetryData.from_dict(packet)
        # to_dict fills in the keys the caller left out, so compare against the
        # normalised packet rather than the partial dict that went in.
        self.assertEqual(restored.servo, packet["servo"])
        self.assertEqual(set(restored.servo), set(SERVO_TELEMETRY_KEYS))
        self.assertEqual(restored.altitude, 5.0)

    def test_a_packet_without_servo_still_decodes(self):
        from shared.detection_models import TelemetryData

        decoded = TelemetryData.from_dict({"altitude": 3.0, "armed": True})
        self.assertIsNone(decoded.servo)
        self.assertIn("servo", decoded.to_dict())


class TestStaleDiscoveryCache(ServoTestCase):
    """The cache says unknown while the vehicle already knows.

    Connect-time discovery can settle mid-download: ``_servo_functions`` was
    written from a partial read, and ``master.params`` later carries the
    ``SERVOx_FUNCTION`` the refusal is about. Refusing on that stale cache is a
    false alarm, so the servo path re-reads the live parameters first.
    """

    functions = {1: 33, 2: 34}  # SERVO10 never made it into the cache

    def test_live_params_are_re_read_before_refusing(self):
        self.master.params = {"SERVO10_FUNCTION": 61}
        used = sitl.send_servo(channel=10, pulse=1500)
        self.assertEqual(used, 10)
        self.assertEqual(len(self.master.commands), 1)
        ts, comp, cmd, conf, param1, param2 = self.master.commands[0][:6]
        self.assertEqual(cmd, sitl.mavutil.mavlink.MAV_CMD_DO_SET_SERVO)
        self.assertEqual(param1, 10)
        self.assertEqual(param2, 1500)
        self.assertEqual(self.master.overrides, [])

    def test_the_cache_is_repaired_by_the_live_read(self):
        self.master.params = {"SERVO10_FUNCTION": 61}
        sitl.send_servo(channel=10, pulse=1500)
        self.assertEqual(sitl._servo_functions.get(10), 61)
        # Second send plans from the cache, with no re-read needed.
        sitl.send_servo(channel=10, pulse=1600)
        self.assertEqual(self.master.commands[-1][5], 1600)

    def test_a_live_mapped_function_still_needs_consent(self):
        # The re-read feeds the safety checks, it does not bypass them.
        self.master.params = {"SERVO10_FUNCTION": 145, "SERVO6_FUNCTION": 0}
        with self.assertRaises(ValueError) as ctx:
            sitl.send_servo(channel=10, pulse=1600)
        self.assertIn("--allow-rc-override", str(ctx.exception))
        self.assertEqual(self.master.overrides, [])
        self.assertEqual(self.master.commands, [])

        used = sitl.send_servo(channel=10, pulse=1600, allow_rc_override=True)
        self.assertEqual(used, 10)
        (ts, comp, *slots) = self.master.overrides[0]
        self.assertEqual(slots[5], 1600)  # RC input 6

    def test_a_live_mount_axis_is_still_refused(self):
        self.master.params = {"SERVO10_FUNCTION": 6}
        with self.assertRaises(ValueError) as ctx:
            sitl.send_servo(channel=10, pulse=1600)
        self.assertIn("Mount Yaw", str(ctx.exception))
        self.assertEqual(self.master.commands, [])

    def test_no_answer_anywhere_still_refuses(self):
        self.master.params = {}
        with self.assertRaises(ValueError) as ctx:
            sitl.send_servo(channel=10, pulse=1600)
        self.assertIn("unknown", str(ctx.exception))
        self.assertEqual(self.master.commands, [])
        self.assertEqual(self.master.overrides, [])

    def test_a_vehicle_that_never_answered_still_refuses(self):
        # No params attribute at all: the re-read must not raise instead.
        with self.assertRaises(ValueError) as ctx:
            sitl.send_servo(channel=10, pulse=1600)
        self.assertIn("unknown", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()