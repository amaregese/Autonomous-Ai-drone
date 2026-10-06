"""Servo channel discovery rules.

Covers the mapping from the autopilot's `SERVOx_FUNCTION` parameters to a
channel, plus the range/clamp behaviour that used to disagree between the SGC
handler (1-16) and the MAVLink backend (1-8).
"""
import unittest

from modules.drone_backend import servo_channels as sc


class TestParseServoFunctions(unittest.TestCase):
    def test_reads_all_sixteen_channels(self):
        params = {f"SERVO{n}_FUNCTION": n for n in range(1, 17)}
        functions = sc.parse_servo_functions(params)
        self.assertEqual(len(functions), 16)
        self.assertEqual(functions[9], 9)
        self.assertEqual(functions[16], 16)

    def test_accepts_float_valued_params(self):
        # ArduPilot returns ints here, but a param store that widened to float
        # must not silently drop every channel.
        self.assertEqual(
            sc.parse_servo_functions({"SERVO9_FUNCTION": 7.0}), {9: 7}
        )

    def test_ignores_missing_and_unparsable_params(self):
        params = {"SERVO1_FUNCTION": 33, "SERVO2_FUNCTION": "bogus"}
        self.assertEqual(sc.parse_servo_functions(params), {1: 33})

    def test_absent_params_yield_nothing(self):
        self.assertEqual(sc.parse_servo_functions({}), {})
        self.assertEqual(sc.parse_servo_functions(None), {})


class TestPickGimbalChannel(unittest.TestCase):
    def test_prefers_pitch(self):
        functions = {9: 8, 10: 6, 11: 7}
        self.assertEqual(sc.pick_gimbal_channel(functions), (11, "Mount Pitch"))

    def test_falls_back_to_yaw_then_roll(self):
        self.assertEqual(
            sc.pick_gimbal_channel({9: 8, 10: 6}), (10, "Mount Yaw")
        )
        self.assertEqual(
            sc.pick_gimbal_channel({9: 8}), (9, "Mount Roll")
        )

    def test_ignores_non_gimbal_outputs(self):
        # Motors, ground steering, and the camera trigger are all real servo
        # outputs but none of them aims the camera.
        functions = {1: 33, 2: 34, 3: 70, 9: 10, 10: 90, 11: 92}
        self.assertEqual(sc.pick_gimbal_channel(functions), (None, None))

    def test_lowest_channel_wins_a_tie(self):
        self.assertEqual(
            sc.pick_gimbal_channel({12: 7, 9: 7}), (9, "Mount Pitch")
        )

    def test_honours_explicit_function_preference(self):
        functions = {9: 7, 13: 13}
        self.assertEqual(
            sc.pick_gimbal_channel(functions, preferred=13), (13, "Mount2 Pitch")
        )

    def test_no_functions_at_all(self):
        self.assertEqual(sc.pick_gimbal_channel({}), (None, None))


class TestReservedRcInputs(unittest.TestCase):
    """Which RC inputs may be overridden without disturbing the aircraft."""

    # Outputs 6-8 disabled, so those inputs are free. Outputs 1-4 are motors.
    spare = {1: 33, 2: 34, 3: 35, 4: 36, 5: 0, 6: 0, 7: 0, 8: 0}

    def test_primary_inputs_are_never_spare(self):
        # Throttle/roll/pitch/yaw and the mode channel are read from the RC
        # stream directly, so no output assignment can release them.
        for rc_input in sorted(sc.PRIMARY_RC_INPUTS):
            with self.subTest(rc_input=rc_input):
                self.assertFalse(
                    sc.rc_input_is_spare({n: 0 for n in range(1, 17)}, rc_input)
                )

    def test_disabled_output_leaves_its_input_spare(self):
        self.assertTrue(sc.rc_input_is_spare(self.spare, 6))
        self.assertTrue(sc.rc_input_is_spare(self.spare, 7))

    def test_an_output_owning_a_function_reserves_its_input(self):
        self.assertFalse(sc.rc_input_is_spare({**self.spare, 6: 20}, 6))

    def test_a_pass_through_of_the_same_input_stays_spare(self):
        # Output 6 following input 6 is a loop, not a flight function.
        functions = {**self.spare, 6: 140 + 5}
        self.assertTrue(sc.rc_input_is_spare(functions, 6))

    def test_manual_output_leaves_its_own_input_spare(self):
        self.assertTrue(sc.rc_input_is_spare({**self.spare, 7: 1}, 7))

    def test_unknown_output_is_not_proof_of_a_spare_input(self):
        self.assertFalse(sc.rc_input_is_spare({1: 33}, 6))
        self.assertFalse(sc.rc_input_is_spare({}, 6))
        self.assertFalse(sc.rc_input_is_spare(None, 6))

    def test_only_inputs_inside_the_override_message_qualify(self):
        self.assertFalse(sc.rc_input_is_spare(self.spare, 9))
        self.assertFalse(sc.rc_input_is_spare(self.spare, 0))


class TestPlanDrive(unittest.TestCase):
    """Transport selection, which is safety-critical rather than a preference."""

    spare = {6: 0, 7: 0, 8: 0}

    def test_spare_aux_input_needs_operator_consent(self):
        # Everything static passes, and the override is still withheld: the
        # parameter map cannot identify the mode channel.
        self.assertEqual(
            sc.plan_drive(9, 145, self.spare)[0], sc.DRIVE_UNVERIFIED_INPUT
        )
        method, rc_input = sc.plan_drive(9, 145, self.spare, allow_rc_override=True)
        self.assertEqual(method, sc.DRIVE_RC_OVERRIDE)
        self.assertEqual(rc_input, 6)

    def test_consent_does_not_unlock_a_reserved_input(self):
        self.assertEqual(
            sc.plan_drive(9, 144, self.spare, allow_rc_override=True)[0],
            sc.DRIVE_RESERVED_INPUT,
        )
        self.assertEqual(
            sc.plan_drive(9, 140, self.spare, allow_rc_override=True)[0],
            sc.DRIVE_RESERVED_INPUT,
        )
        self.assertEqual(
            sc.plan_drive(9, 145, None, allow_rc_override=True)[0],
            sc.DRIVE_RESERVED_INPUT,
        )

    def test_consent_does_not_unlock_a_mount_axis(self):
        self.assertEqual(
            sc.plan_drive(9, 7, self.spare, allow_rc_override=True)[0],
            sc.DRIVE_MOUNT_PROTOCOL,
        )

    def test_consent_never_reaches_an_unassigned_output(self):
        # k_none is driven directly; consent is irrelevant, so the method must not
        # change with it.
        self.assertEqual(
            sc.plan_drive(9, 0, self.spare)[0], sc.DRIVE_DO_SET_SERVO
        )
        self.assertEqual(
            sc.plan_drive(9, 0, self.spare, allow_rc_override=True)[0],
            sc.DRIVE_DO_SET_SERVO,
        )

    def test_primary_input_is_refused_rather_than_sent(self):
        self.assertEqual(
            sc.plan_drive(9, 144, self.spare)[0], sc.DRIVE_RESERVED_INPUT
        )

    def test_missing_function_map_refuses_every_override(self):
        self.assertEqual(
            sc.plan_drive(9, 145, None)[0], sc.DRIVE_RESERVED_INPUT
        )

    def test_mount_axis_still_goes_to_the_mount_protocol(self):
        self.assertEqual(sc.plan_drive(9, 7, self.spare)[0], sc.DRIVE_MOUNT_PROTOCOL)

    def test_unassigned_output_still_uses_do_set_servo(self):
        self.assertEqual(sc.plan_drive(9, 0, self.spare)[0], sc.DRIVE_DO_SET_SERVO)

    def test_input_beyond_the_override_message_is_unsupported(self):
        self.assertEqual(sc.plan_drive(12, 152, self.spare)[0], sc.DRIVE_UNSUPPORTED)

    def test_rc_pass_through_beyond_the_override_message_uses_do_set_servo(self):
        # k_rcin11 (61): RC input 11 does not fit in the override message, and
        # ArduPilot whitelists k_rcin1-k_rcin16 for MAV_CMD_DO_SET_SERVO, which
        # writes the output rather than any RC input. Boundaries of the range:
        self.assertEqual(sc.plan_drive(10, 61, self.spare)[0], sc.DRIVE_DO_SET_SERVO)
        self.assertEqual(sc.plan_drive(9, 59, self.spare)[0], sc.DRIVE_DO_SET_SERVO)
        self.assertEqual(sc.plan_drive(9, 66, self.spare)[0], sc.DRIVE_DO_SET_SERVO)

    def test_rc_pass_through_within_the_override_message_keeps_it(self):
        # k_rcin7: input 7 fits in the message, so the override path - and
        # its consent gate - is unchanged by the direct path above.
        fns = {**self.spare, 9: 57}
        self.assertEqual(sc.plan_drive(9, 57, fns)[0], sc.DRIVE_UNVERIFIED_INPUT)
        method, rc_input = sc.plan_drive(9, 57, fns, allow_rc_override=True)
        self.assertEqual(method, sc.DRIVE_RC_OVERRIDE)
        self.assertEqual(rc_input, 7)

    def test_mapped_output_beyond_the_override_stays_unsupported(self):
        # Only the plain pass-through gained DO_SET_SERVO; k_rcinN_mapped is
        # not whitelisted for it.
        self.assertEqual(sc.plan_drive(9, 148, self.spare)[0], sc.DRIVE_UNSUPPORTED)
        self.assertEqual(sc.plan_drive(9, 155, self.spare)[0], sc.DRIVE_UNSUPPORTED)

    def test_motor_is_unsupported(self):
        self.assertEqual(sc.plan_drive(1, 33, self.spare)[0], sc.DRIVE_UNSUPPORTED)


class TestResolveChannel(unittest.TestCase):
    def test_explicit_request_wins_over_detection(self):
        self.assertEqual(sc.resolve_channel(11, detected=9), 11)

    def test_falls_back_to_detected(self):
        self.assertEqual(sc.resolve_channel(None, detected=9), 9)

    def test_no_live_channel_guessing(self):
        # There is deliberately no "any channel that looks alive" fallback: the
        # lowest reporting output on a copter is a throttle, and feeding the
        # planner a channel like that turns a gimbal command into an RC override
        # on a flight-control input.
        self.assertIsNone(sc.resolve_channel(None))

    def test_detected_channel_is_used_when_known(self):
        self.assertEqual(sc.resolve_channel(None, detected=11), 11)

    def test_unknown_returns_none_rather_than_guessing(self):
        # Guessing here would mean driving whatever channel happens to be first.
        self.assertIsNone(sc.resolve_channel(None))

    def test_out_of_range_is_rejected(self):
        for bad in (0, 17, -1):
            with self.assertRaises(ValueError):
                sc.resolve_channel(bad)

    def test_full_range_is_accepted(self):
        # The gimbal aux range 9-16 was previously rejected outright.
        for channel in range(1, 17):
            self.assertEqual(sc.resolve_channel(channel), channel)


class TestPulseHandling(unittest.TestCase):
    def test_clamps_to_travel(self):
        self.assertEqual(sc.clamp_pulse(500), sc.SERVO_PULSE_MIN)
        self.assertEqual(sc.clamp_pulse(3000), sc.SERVO_PULSE_MAX)
        self.assertEqual(sc.clamp_pulse(1600), 1600)

    def test_non_numeric_pulse_rejected(self):
        with self.assertRaises(ValueError):
            sc.clamp_pulse("full left")

    def test_angle_matches_documented_conversion(self):
        # 1500 + angle * 500/45, the formula the SGC protocol documents.
        self.assertEqual(sc.pulse_from_angle(0), 1500)
        self.assertEqual(sc.pulse_from_angle(45), 2000)
        self.assertEqual(sc.pulse_from_angle(-45), 1000)
        self.assertEqual(sc.pulse_from_angle(9), 1600)

    def test_non_numeric_angle_rejected(self):
        with self.assertRaises(ValueError):
            sc.pulse_from_angle(None)


class TestReadBack(unittest.TestCase):
    class _Msg:
        def __init__(self, **fields):
            for name, value in fields.items():
                setattr(self, name, value)

    def test_parses_present_channels(self):
        fields = sc.build_servo_field_map("servo")
        msg = self._Msg(servo1_raw=1500, servo9_raw=1600)
        self.assertEqual(sc.parse_pwm_fields(msg, fields), {1: 1500, 9: 1600})

    def test_drops_unused_outputs_reporting_zero(self):
        # ArduPilot reports 0 for an unmapped output, and 0 is not a legal PWM.
        fields = sc.build_servo_field_map("servo")
        msg = self._Msg(servo1_raw=1500, servo10_raw=0)
        self.assertEqual(sc.parse_pwm_fields(msg, fields), {1: 1500})

    def test_missing_fields_are_skipped(self):
        fields = sc.build_servo_field_map("servo")
        msg = self._Msg(servo1_raw=1500)
        self.assertEqual(sc.parse_pwm_fields(msg, fields), {1: 1500})

    def test_rc_channel_field_map_is_distinct(self):
        rc_fields = sc.build_servo_field_map("chan")
        self.assertEqual(rc_fields[1], "chan1_raw")

    def test_tolerance_accepts_servo_travel_error(self):
        self.assertTrue(sc.pulse_matches(1500, 1540))
        self.assertFalse(sc.pulse_matches(1500, 1200))

    def test_absent_readback_does_not_match(self):
        self.assertFalse(sc.pulse_matches(1500, None))


class TestMappedOutputPulse(unittest.TestCase):
    """Predicting what a k_rcinN_mapped output will report back.

    Every expectation in this class was measured against a live SITL
    ArduCopter with RC8 at 1000/1500/1800 and SERVO9 at 1100/1500/1900, then
    re-measured with both at 1000/1500/2000. It reproduces
    ``SRV_Channel::output_ch``'s ``pwm_from_angle(norm_input_dz() * 4500)``.
    """

    # measured live: cmd -> servo9_raw
    MISMATCHED = {
        1000: 1100,
        1100: 1180,
        1250: 1300,
        1500: 1500,
        1750: 1833,
        1900: 1900,
        2000: 1900,
    }
    RC_RANGE = (1000, 1500, 1800)
    SERVO_RANGE = (1100, 1500, 1900)

    def test_matches_live_mismatched_limits(self):
        for commanded, reported in self.MISMATCHED.items():
            self.assertEqual(
                sc.mapped_output_pulse(commanded, self.RC_RANGE, self.SERVO_RANGE),
                reported,
                f"cmd {commanded}",
            )

    def test_matched_limits_are_identity(self):
        matched = (1000, 1500, 2000)
        for pulse in range(1000, 2001, 50):
            self.assertEqual(
                sc.mapped_output_pulse(pulse, matched, matched), pulse, f"cmd {pulse}"
            )

    def test_scaling_is_not_a_plain_offset(self):
        # 1100 -> 1180 but 1900 -> 1900: the relationship is a ratio, which is
        # why the pulse we command is not the pulse we read back.
        self.assertEqual(
            sc.mapped_output_pulse(1100, self.RC_RANGE, self.SERVO_RANGE), 1180
        )
        self.assertEqual(
            sc.mapped_output_pulse(1900, self.RC_RANGE, self.SERVO_RANGE), 1900
        )

    def test_output_beyond_the_rc_range_is_clamped_to_the_output_range(self):
        for pulse in (0, 500, 2100, 3000):
            result = sc.mapped_output_pulse(pulse, self.RC_RANGE, self.SERVO_RANGE)
            self.assertGreaterEqual(result, self.SERVO_RANGE[0])
            self.assertLessEqual(result, self.SERVO_RANGE[2])

    def test_unknown_limits_yield_no_prediction(self):
        # Better no prediction than a guessed one: a wrong "expected" value
        # would report a healthy output as broken.
        self.assertIsNone(sc.mapped_output_pulse(1500, None, self.SERVO_RANGE))
        self.assertIsNone(sc.mapped_output_pulse(1500, self.RC_RANGE, None))
        self.assertIsNone(sc.mapped_output_pulse(1500, None, None))

    def test_degenerate_ranges_yield_no_prediction(self):
        for limits in ((1500, 1500, 1500), (1800, 1500, 1000), (1500, 1000, 1500)):
            self.assertIsNone(
                sc.mapped_output_pulse(1500, limits, self.SERVO_RANGE), limits
            )

    def test_non_numeric_pulse_yields_no_prediction(self):
        self.assertIsNone(
            sc.mapped_output_pulse("over there", self.RC_RANGE, self.SERVO_RANGE)
        )


class TestLimitsFromParams(unittest.TestCase):
    def test_reads_min_trim_max(self):
        params = {"SERVO9_MIN": 1100, "SERVO9_TRIM": 1500, "SERVO9_MAX": 1900}
        self.assertEqual(sc.limits_from_params(params, "SERVO", 9), (1100, 1500, 1900))

    def test_float_params_are_converted(self):
        params = {"RC8_MIN": 1000.0, "RC8_TRIM": 1500.0, "RC8_MAX": 1800.0}
        self.assertEqual(sc.limits_from_params(params, "RC", 8), (1000, 1500, 1800))

    def test_partial_range_is_not_usable(self):
        self.assertIsNone(sc.limits_from_params({"SERVO9_MIN": 1100}, "SERVO", 9))

    def test_absent_params(self):
        self.assertIsNone(sc.limits_from_params({}, "SERVO", 9))
        self.assertIsNone(sc.limits_from_params(None, "SERVO", 9))


class TestIdentityScaling(unittest.TestCase):
    def test_matched_ranges_are_identity(self):
        self.assertTrue(sc.identity_scaling((1000, 1500, 2000), (1000, 1500, 2000)))

    def test_mismatched_ranges_are_not(self):
        self.assertFalse(sc.identity_scaling((1000, 1500, 1800), (1100, 1500, 1900)))

    def test_unknown_input_range_is_not_identity(self):
        self.assertFalse(sc.identity_scaling(None, (1100, 1500, 1900)))
        self.assertFalse(sc.identity_scaling((1000, 1500, 2000), None))
        self.assertFalse(sc.identity_scaling(None, None))


class TestFunctionNames(unittest.TestCase):
    def test_known_functions(self):
        self.assertEqual(sc.function_name(7), "Mount Pitch")
        self.assertEqual(sc.function_name(10), "Camera Trigger")

    def test_unknown_and_none(self):
        self.assertIsNone(sc.function_name(None))
        self.assertIsNone(sc.function_name(4242))

    def test_mavlink_gimbal_numbers_are_not_treated_as_mounts(self):
        # MAV_SERVO_FUNCTION_GIMBAL_ROLL is 26, which is ArduPilot's ground
        # steering. Mapping it as a gimbal would drive a steering servo.
        self.assertNotIn(26, sc.SERVO_FUNCTIONS)


if __name__ == "__main__":
    unittest.main()