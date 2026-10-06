"""Which physical servo output the camera gimbal is wired to.

The servo channel was previously hardcoded to 8 in four places, and nothing in
the MAVLink link ever asked the autopilot what it had. This module holds the
discovery rules so they can be unit tested without a vehicle connected; the
MAVLink side lives in ``sitl.py``.

Three sources of truth, in descending order of authority:

1. ``SERVOx_FUNCTION`` parameters. ArduPilot's ``SERVO9_FUNCTION=7`` and
   friends declare what every output *is* (mount pitch, mount yaw, ...). This
   is the only source that names the function, so it is the one that identifies
   a gimbal axis. See ``SERVO_FUNCTIONS`` below for the value table.
2. ``SERVO_OUTPUT_RAW`` telemetry. Reports the live PWM of outputs 1-16, so it
   proves the channel exists and lets us confirm a command actually landed.
3. ``RC_CHANNELS_RAW`` telemetry. Reports the receiver's channel count, which
   bounds what an RC override can even address.

Note on the function numbers: these are ArduPilot ``SRV_Channel::Function``
values, *not* MAVLink ``MAV_SERVO_FUNCTION``. The two numbering schemes
collide - MAVLink's ``MAV_SERVO_FUNCTION_GIMBAL_ROLL`` is 26, which is
ArduPilot's *ground steering*. Mixing them up means driving a rover's steering
servo while believing a gimbal is moving, so only the ArduPilot table is used
here.

Why a gimbal output is normally *not* directly drivable
-------------------------------------------------------
ArduPilot's ``MAV_CMD_DO_SET_SERVO`` handler whitelists a small set of
functions (``AP_ServoRelayEvents::do_set_servo``): ``k_none``, ``k_manual``,
the sprayer and gripper outputs, and RC pass-through. A channel assigned
``k_mount_pan``/``k_mount_tilt``/``k_mount_roll`` falls through to ``default:``
and is refused with "Channel N is already in use" - because the mount backend
is supposed to own it.

That is a trap for automatic discovery: the channel is identified *because* it
is assigned mount pitch, and then the autopilot refuses every direct command to
it. The way out is ``k_rcinN_mapped`` (140-155), which means "this output
follows RC input N" and is mapped through ``SERVOx_MIN``/``SERVOx_MAX``. The
output can be any channel up to 16 while the RC input it listens to must be
within the 8 channels ``RC_CHANNELS_OVERRIDE`` carries - which is what makes an
aux output drivable at all.

A plain ``k_rcinN`` pass-through (51-66) is whitelisted for ``DO_SET_SERVO``
too, so it can be driven either through its RC input - when that input fits in
``RC_CHANNELS_OVERRIDE``'s 8 slots - or directly by the command when it does
not. That second route is what makes an output such as
``SERVO10_FUNCTION = 61`` (the camera servo on an RCIN11 pass-through, RC
input 11) reachable without reconfiguring the FCU.

The cost is real and worth stating plainly: a ``k_rcinN_mapped`` output is just
a pass-through servo to the autopilot. It gets no stabilisation, and
``SERVOx_FUNCTION`` no longer says "gimbal", so detection has to infer it from
the wiring (see ``pick_gimbal_channel``).
"""

# ArduPilot SRV_Channel::Function values that belong to a camera gimbal.
SERVO_FUNCTIONS = {
    6: "Mount Yaw",
    7: "Mount Pitch",
    8: "Mount Roll",
    9: "Mount Deploy/Retract",
    10: "Camera Trigger",
    12: "Mount2 Yaw",
    13: "Mount2 Pitch",
    14: "Mount2 Roll",
    15: "Mount2 Deploy/Retract",
    90: "Camera ISO",
    91: "Camera Aperture",
    92: "Camera Focus",
    93: "Camera Shutter Speed",
}

# The axes that can point the camera. Deploy/Retract and the lens controls are
# excluded: they are real gimbal outputs but steering one of them does not aim
# the camera.
GIMBAL_AXIS_FUNCTIONS = (7, 6, 8, 13, 12, 14)

# ArduPilot RC pass-through outputs. ``k_rcinN`` (51-66) passes the input
# straight through; ``k_rcinN_mapped`` (140-155) remaps it through
# SERVOx_MIN/MAX/TRIM. Both are drivable by RC_CHANNELS_OVERRIDE while the
# input they listen to fits in its 8 slots, unlike the mount functions above.
# Past those 8 slots only the plain pass-through still has a path: ArduPilot
# whitelists k_rcin1-k_rcin16 for MAV_CMD_DO_SET_SERVO, so plan_drive drives
# such an output directly instead of refusing it.
RC_PASSTHROUGH_FUNCTION_MIN = 51  # k_rcin1
RC_PASSTHROUGH_FUNCTION_MAX = 66  # k_rcin16
RC_MAPPED_FUNCTION_MIN = 140       # k_rcin1_mapped
RC_MAPPED_FUNCTION_MAX = 155       # k_rcin16_mapped

# The axis a follower needs when it has to pick one on its own: pitch is the
# elevation axis, which is what tracks a target's vertical position.
PREFERRED_GIMBAL_FUNCTION = 7

SERVO_CHANNEL_MIN = 1
SERVO_CHANNEL_MAX = 16
SERVO_PULSE_MIN = 1000
SERVO_PULSE_MAX = 2000

# RC_CHANNELS_OVERRIDE carries exactly 8 *input* channels. That limit is why an
# aux output needs a k_rcinN_mapped function with N <= 8 to be addressable at
# all; the output channel itself may still be 9-16.
OVERRIDE_CHANNEL_MAX = 8

# RC inputs ArduCopter reads directly, whatever the servo outputs are assigned
# to. Throttle/roll/pitch/yaw and the flight-mode channel are taken from the RC
# *input* stream by AP_Copter::read_channels(), not from an output, so no
# SERVOx_FUNCTION assignment can free them up.
#
# This matters because RC_CHANNELS_OVERRIDE writes inputs. An override aimed at
# one of these does not move a gimbal - it commands the aircraft, and the
# flight-mode input turns a servo sweep into a walk through RTL, STABILIZE and
# everything between them. Inputs 6-8 are aux on ArduCopter and are only spoken
# for if an output says so, which is what rc_input_is_spare checks.
PRIMARY_RC_INPUTS = frozenset({1, 2, 3, 4, 5})

# MAVLink's "do not override this channel" value for RC_CHANNELS_OVERRIDE.
# The spec says UINT16_MAX; ArduPilot also accepts 0. The old code sent 0,
# which works only by accident - 65535 is what the message actually means.
SERVO_CHANNEL_UNSET = 65535

# Acceptable error when reading a servo back, in microseconds. Servo travel is
# usually ±50us around the commanded value, so anything tighter reports noise.
PULSE_TOLERANCE = 60

# Ways to reach an output, in the order they should be preferred.
DRIVE_MOUNT_PROTOCOL = "mount_protocol"
DRIVE_RC_OVERRIDE = "rc_override"
DRIVE_DO_SET_SERVO = "do_set_servo"
DRIVE_RESERVED_INPUT = "reserved_input"
DRIVE_UNVERIFIED_INPUT = "unverified_input"
DRIVE_UNSUPPORTED = "unsupported"


def function_name(function_id):
    """Human name for a SERVOx_FUNCTION value, or None if not recognised."""
    if function_id is None:
        return None
    try:
        value = int(function_id)
    except (TypeError, ValueError):
        return None
    name = SERVO_FUNCTIONS.get(value)
    if name is not None:
        return name
    rc_input = rc_input_for_function(value)
    if rc_input is not None:
        suffix = "mapped" if value >= RC_MAPPED_FUNCTION_MIN else "pass-through"
        return f"RC Input {rc_input} ({suffix})"
    return None


def rc_input_for_function(function_id):
    """RC input a pass-through function listens to, or None.

    ``k_rcinN`` -> N, and ``k_rcinN_mapped`` -> N. Both let
    ``RC_CHANNELS_OVERRIDE`` reach the output, provided N <= 8.
    """
    try:
        value = int(function_id)
    except (TypeError, ValueError):
        return None
    if RC_PASSTHROUGH_FUNCTION_MIN <= value <= RC_PASSTHROUGH_FUNCTION_MAX:
        return value - RC_PASSTHROUGH_FUNCTION_MIN + 1
    if RC_MAPPED_FUNCTION_MIN <= value <= RC_MAPPED_FUNCTION_MAX:
        return value - RC_MAPPED_FUNCTION_MIN + 1
    return None


def mapped_output_pulse(input_pulse, input_limits, output_limits):
    """Predict the PWM a ``k_rcinN_mapped`` output will produce, or None.

    Mirrors ``SRV_Channel::output_ch`` in
    ``libraries/SRV_Channel/SRV_Channel_aux.cpp``, which for a mapped output
    does *not* copy the RC pulse across. It normalises the input against the RC
    channel's own MIN/TRIM/MAX, scales that to ±4500, and converts back through
    the *output* channel's MIN/TRIM/MAX::

        radio_in = pwm_from_angle(c->norm_input_dz() * 4500)

    So an input of 1750us with RC8 at 1000/1500/1800 and SERVO9 at
    1100/1500/1900 comes out at 1833us, not 1750us. Verified against live SITL
    across both matched and mismatched limit sets.

    Returns None when either limit set is unknown, because a prediction built
    from guesses would produce a wrong "expected" value and report a healthy
    output as broken.
    """
    if not input_limits or not output_limits:
        return None
    in_min, in_trim, in_max = input_limits
    out_min, out_trim, out_max = output_limits
    if None in (in_min, in_trim, in_max, out_min, out_trim, out_max):
        return None
    if in_trim <= in_min or in_max <= in_trim:
        return None  # degenerate RC range; nothing sensible to normalise by
    if out_trim <= out_min or out_max <= out_trim:
        return None

    try:
        pulse = int(input_pulse)
    except (TypeError, ValueError):
        return None

    if pulse >= in_trim:
        normalised = (pulse - in_trim) / float(in_max - in_trim)
    else:
        normalised = (pulse - in_trim) / float(in_trim - in_min)
    normalised = max(-1.0, min(1.0, normalised))

    if normalised >= 0:
        return int(round(out_trim + normalised * (out_max - out_trim)))
    return int(round(out_trim + normalised * (out_trim - out_min)))


def limits_from_params(params, prefix, channel):
    """``(MIN, TRIM, MAX)`` for one channel's range parameters, or None."""
    values = []
    for suffix in ("MIN", "TRIM", "MAX"):
        name = f"{prefix}{channel}_{suffix}"
        try:
            values.append(int(float(params[name])))
        except (KeyError, TypeError, ValueError):
            return None
    return tuple(values)


def identity_scaling(input_limits, output_limits):
    """Whether a mapped output passes its RC pulse through unchanged.

    Worth reporting explicitly: matching the output's MIN/TRIM/MAX to the RC
    channel's makes ``input_pulse == reported_pulse``, which turns read-back
    verification into a real check instead of a scaled prediction.
    """
    return bool(input_limits) and input_limits == output_limits


def parse_servo_functions(params, max_channel=SERVO_CHANNEL_MAX):
    """Extract ``{channel: function_id}`` from a fetched parameter dict.

    Returns an empty dict when the parameters are unavailable rather than
    raising: a vehicle that never answered the parameter request must not stop
    the app from connecting, it just leaves servo discovery to the fallback.
    """
    functions = {}
    if not params:
        return functions
    for channel in range(SERVO_CHANNEL_MIN, max_channel + 1):
        name = f"SERVO{channel}_FUNCTION"
        try:
            value = params[name]
        except (KeyError, TypeError):
            continue
        try:
            functions[channel] = int(float(value))
        except (TypeError, ValueError):
            continue
    return functions


def pick_gimbal_channel(functions, preferred=None):
    """Choose the output that aims the camera, or None if none is identifiable.

    ``preferred`` is a function id to look for first, which lets a caller ask
    for a specific axis (e.g. mount2 pitch) without changing the fallback
    order. Ties between channels assigned the same function go to the lowest
    channel, matching ArduPilot's own "first one wins" convention.

    A channel assigned a mount function is authoritative - the operator said
    "this is a gimbal" - and is returned even though it needs the mount protocol
    to actually move. A ``k_rcinN_mapped`` channel is only accepted when it is
    the *only* candidate: those functions carry no hint that the output is a
    gimbal, so picking one out of several would be guessing, and guessing here
    means pointing a camera at whatever the wiring happened to map.
    """
    if not functions:
        return None, None

    ordered = ([preferred] if preferred is not None else []) + [
        f for f in GIMBAL_AXIS_FUNCTIONS if f != preferred
    ]
    for function_id in ordered:
        candidates = sorted(ch for ch, fn in functions.items() if fn == function_id)
        if candidates:
            return candidates[0], SERVO_FUNCTIONS.get(function_id)

    mapped = sorted(
        ch for ch, fn in functions.items()
        if rc_input_for_function(fn) is not None
    )
    if len(mapped) == 1:
        return mapped[0], function_name(functions[mapped[0]])
    return None, None


def rc_input_for_gimbal(channel):
    """The RC input an aux gimbal output should be mapped to, or None.

    ``RC_CHANNELS_OVERRIDE`` only carries 8 slots, so an output above 8 has to
    be mapped to an input inside that range. Channel 9 is the natural first
    choice; past channel 16 there is nothing left to map to.

    Reusing the output's own index where possible keeps the wiring obvious when
    reading the parameters: SERVO5_FUNCTION = 144 means "output 5 follows RC
    input 5".
    """
    if not SERVO_CHANNEL_MIN <= channel <= SERVO_CHANNEL_MAX:
        return None
    return channel if channel <= OVERRIDE_CHANNEL_MAX else OVERRIDE_CHANNEL_MAX


def rc_input_is_spare(functions, rc_input):
    """Whether overriding RC input ``rc_input`` cannot disturb flight control.

    ``RC_CHANNELS_OVERRIDE`` addresses RC *inputs*, and ArduPilot's default
    mapping is input N to output N, so input N is free exactly when output N is
    not already doing something the aircraft depends on.

    Two independent reasons to refuse:

    - inputs 1-5 are ArduCopter's throttle/roll/pitch/yaw/mode inputs, read from
      the RC stream directly. No output assignment can release them.
    - the output that input N feeds is assigned some other function, so writing
      the input moves that function. A flight-mode slot assigned to output 6
      means input 6 changes the flight mode.

    An output whose function we have not read counts as reserved. The autopilot
    always has *something* assigned there, and the whole point of this check is
    that an unknown is not a safe.

    Only ``functions`` covering output N is consulted; callers pass the full
    ``{channel: function_id}`` map from ``parse_servo_functions``.
    """
    try:
        input_number = int(rc_input)
    except (TypeError, ValueError):
        return False

    if not 1 <= input_number <= OVERRIDE_CHANNEL_MAX:
        return False
    if input_number in PRIMARY_RC_INPUTS:
        return False

    output_function = (functions or {}).get(input_number)
    if output_function is None:
        return False  # never read, so never proven safe
    try:
        value = int(output_function)
    except (TypeError, ValueError):
        return False

    # k_none (disabled) and k_manual are both just the input passing straight
    # through, so overriding the input cannot reach anything else.
    if value in (0, 1):
        return True
    return rc_input_for_function(value) == input_number


def plan_drive(channel, function_id, functions=None, allow_rc_override=False):
    """How to reach this output, given what it is assigned to.

    Returns ``(method, rc_input_or_None)`` where method is one of
    ``DRIVE_*``. Getting this wrong is not a no-op: the autopilot silently
    refuses commands to a mount-function output, so the caller needs to be told
    which mechanism applies before it sends anything.

    - mount axis -> the mount protocol owns it. Direct PWM is refused by
      ``AP_ServoRelayEvents``, so we must not pretend otherwise.
    - RC pass-through / mapped -> RC_CHANNELS_OVERRIDE, when the input it
      listens to is within the 8 channels that message carries *and* that input
      is spare. ``functions`` is the full ``{channel: function_id}`` map, needed
      because an override writes an RC input and inputs 1-5 belong to the
      aircraft (see ``rc_input_is_spare``).
    - RC pass-through whose input is beyond those 8 slots -> the override
      message cannot address it, but ArduPilot whitelists ``k_rcin1``-``k_rcin16``
      (51-66) for MAV_CMD_DO_SET_SERVO, and that command writes the *output*
      rather than any RC input - so no input is overridden, nothing needs to be
      spare, and no consent applies. ``k_rcinN_mapped`` (140-155) is not in the
      whitelist and has no path at all, so it stays refused.
    - k_none -> MAV_CMD_DO_SET_SERVO, which is the handler this output exists
      for. Not an RC path at all, so an override would silently do nothing.
    - anything else (a motor, an airframe surface) -> not ours to drive.

    Every RC path additionally needs ``allow_rc_override``. ``SERVOx_FUNCTION``
    only records which input an output *follows*; it says nothing about which
    inputs ``AP_Copter`` reads for flight control, and that set depends on the
    channel mapping and on the frame. So even an input that passes every check
    here can still be the mode channel on a vehicle we have not surveyed, which
    is exactly what a real bench did: ch9 was mapped to an input outside 1-5,
    every static check passed, and sweeping it walked the aircraft through
    RTL, STABILIZE, AUTO, CIRCLE and LAND. Only the operator knows their own
    wiring, so the override is opt-in.

    Passing ``functions=None`` refuses every RC override, because without the
    map there is no way to tell a spare input from the mode channel. Guessing
    here is what walks a vehicle through its flight modes.

    ``allow_rc_override`` is the operator's assertion that the input is spare on
    *this* vehicle. Without it an otherwise-spare input yields
    ``DRIVE_UNVERIFIED_INPUT``, so a mapped output is refused rather than driven.
    """
    try:
        value = int(function_id)
    except (TypeError, ValueError):
        return DRIVE_UNSUPPORTED, None

    if value in GIMBAL_AXIS_FUNCTIONS or value in (9, 15):
        return DRIVE_MOUNT_PROTOCOL, None

    rc_input = rc_input_for_function(value)
    if rc_input is not None:
        if rc_input > OVERRIDE_CHANNEL_MAX:
            # A plain k_rcinN pass-through (51-66) is whitelisted by
            # ArduPilot's MAV_CMD_DO_SET_SERVO handler, so once the RC input
            # it listens to is beyond the 8 slots RC_CHANNELS_OVERRIDE
            # carries there is no reason to refuse: DO_SET_SERVO writes the
            # *output* - never an RC input - so no input is overridden, no
            # spare-input check applies and no consent is needed. This is how
            # SERVO10_FUNCTION=61 (k_rcin11, an RCIN11 camera servo) is
            # reached without touching the FCU's configuration.
            # k_rcinN_mapped (140-155) is not in that whitelist, so it keeps
            # its refusal: its only path is the override message, which
            # cannot carry the input it follows.
            if RC_PASSTHROUGH_FUNCTION_MIN <= value <= RC_PASSTHROUGH_FUNCTION_MAX:
                return DRIVE_DO_SET_SERVO, None
            return DRIVE_UNSUPPORTED, rc_input
        if not rc_input_is_spare(functions, rc_input):
            return DRIVE_RESERVED_INPUT, rc_input
        if not allow_rc_override:
            return DRIVE_UNVERIFIED_INPUT, rc_input
        return DRIVE_RC_OVERRIDE, rc_input

    if value == 0:
        return DRIVE_DO_SET_SERVO, None

    if value == 1:  # k_manual: passes through the RC input of the same index
        if channel > OVERRIDE_CHANNEL_MAX:
            return DRIVE_UNSUPPORTED, channel
        if not rc_input_is_spare(functions, channel):
            return DRIVE_RESERVED_INPUT, channel
        if not allow_rc_override:
            return DRIVE_UNVERIFIED_INPUT, channel
        return DRIVE_RC_OVERRIDE, channel

    return DRIVE_UNSUPPORTED, None


def clamp_pulse(pulse):
    """Clamp a pulse into the servo's physical travel.

    A raw pulse from the ground station must not be able to drive a real servo
    past its end stops.
    """
    try:
        value = int(pulse)
    except (TypeError, ValueError):
        raise ValueError(f"pulse {pulse!r} is not a number")
    return max(SERVO_PULSE_MIN, min(SERVO_PULSE_MAX, value))


def pulse_from_angle(angle, center=1500, span_us=500, full_scale_deg=45.0):
    """Convert a servo angle in degrees to a pulse width.

    ``center`` is the neutral pulse, ``span_us`` the total travel either side of
    it, ``full_scale_deg`` the angle at that end stop. The defaults describe the
    usual ±45 degree / ±500us servo, and match what the SGC protocol documents.
    """
    try:
        degrees = float(angle)
    except (TypeError, ValueError):
        raise ValueError(f"angle {angle!r} is not a number")
    if full_scale_deg == 0:
        raise ValueError("full_scale_deg must not be zero")
    return center + degrees * (span_us / full_scale_deg)


def resolve_channel(requested, detected=None):
    """Work out which channel a servo command should address.

    ``requested`` wins when given - the ground station knows its own hardware.
    Otherwise the channel discovered from ``SERVOx_FUNCTION`` is used. Returns
    None only when nothing is known, so the caller can report it instead of
    silently picking a channel that may be a motor.

    There is deliberately no "any channel that looks alive" fallback. On an
    ArduCopter the lowest reporting output is a throttle, so that heuristic both
    spins a prop and, once the planner turns the channel into an RC override,
    writes into the flight-control inputs. Refusing is the only safe answer.
    """
    if requested is not None:
        channel = int(requested)
        if not SERVO_CHANNEL_MIN <= channel <= SERVO_CHANNEL_MAX:
            raise ValueError(
                f"channel {channel} outside {SERVO_CHANNEL_MIN}-{SERVO_CHANNEL_MAX}"
            )
        return channel

    if detected:
        return int(detected)

    return None


def parse_pwm_fields(msg, fields):
    """Pull ``{channel: pulse}`` out of a SERVO_OUTPUT_RAW / RC_CHANNELS_RAW message.

    ``fields`` is the prefix to read (``"servo"`` or ``"chan"``); pymavlink names
    the fields ``servo1_raw``..``servo16_raw``. An unused output reports 0, and
    0 is not a legal PWM value, so those are dropped rather than stored.
    """
    values = {}
    for channel, field in fields.items():
        pulse = getattr(msg, field, None)
        if pulse is None:
            continue
        try:
            pulse = int(pulse)
        except (TypeError, ValueError):
            continue
        if pulse <= 0:
            continue
        values[channel] = pulse
    return values


def build_servo_field_map(prefix, max_channel=SERVO_CHANNEL_MAX):
    """``{channel: "<prefix><n>_raw"}`` for MAVLink messages that report PWM."""
    return {n: f"{prefix}{n}_raw" for n in range(1, max_channel + 1)}


def pulse_matches(commanded, reported, tolerance=PULSE_TOLERANCE):
    """Whether a read-back PWM confirms the commanded pulse."""
    if reported is None:
        return False
    return abs(int(reported) - int(commanded)) <= tolerance