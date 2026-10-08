"""Send one SGC command to a running drone, from a laptop on the same network.

The drone listens for plain JSON commands on UDP ``--sgc-cmd-port`` (9002 by
default); this tool is the same socket send the SGC performs, so the SGC team
can verify each command - including the panic RTL - without running the ground
station.

Examples:
    python tools/send_sgc_command.py --host 192.168.1.42 --type panic_rtl
    python tools/send_sgc_command.py --host 192.168.1.42 --type follow_start
    python tools/send_sgc_command.py --host 192.168.1.42 --type land
    python tools/send_sgc_command.py --host 192.168.1.42 --type disarm
    python tools/send_sgc_command.py --host 192.168.1.42 --type select_target \\
        --bbox 320 240 120 260 --class-name person
    python tools/send_sgc_command.py --host 192.168.1.42 --type servo --pulse 1600

``servo`` without ``--channel`` uses the gimbal channel the drone detected from
the autopilot, so the channel does not have to be repeated here.

``takeoff`` no longer arms the vehicle: send ``arm`` first, then ``takeoff`` -
or skip the pair with ``arm_takeoff``, which arms and climbs to the takeoff
altitude as one command (refused if the vehicle is already armed).

``land`` lands but keeps the app running, so ``disarm`` can follow once the
vehicle is on the ground - ``disarm`` is refused above 0.5 m altitude.

There is no acknowledgement: UDP is fire-and-forget. The drone echoes every
command it accepts on its own console, and after a panic RTL it stops streaming
and exits - the detection stream going quiet is the acknowledgement.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys

COMMAND_TYPES = (
    "select_target",
    "deselect_target",
    "follow_start",
    "follow_stop",
    "arm",
    "arm_takeoff",
    "takeoff",
    "land",
    "disarm",
    "panic_rtl",
    "servo",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one SGC command to the drone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="command types: " + ", ".join(COMMAND_TYPES),
    )
    parser.add_argument("--host", required=True, help="drone IP address")
    parser.add_argument("--type", required=True, choices=COMMAND_TYPES, dest="cmd_type",
                        help="command to send")
    parser.add_argument("--port", type=int, default=9002, help="SGC command port (default: 9002)")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                        default=None, help="x y w h, for select_target / follow_start")
    parser.add_argument("--class-name", default="", help="class name, for select_target / follow_start")
    parser.add_argument("--confidence", type=float, default=0.0, help="confidence hint (optional)")
    parser.add_argument("--frame-w", type=int, default=0, help="frame width (optional)")
    parser.add_argument("--frame-h", type=int, default=0, help="frame height (optional)")
    parser.add_argument("--channel", type=int, default=None,
                        help="servo channel 1-16 (default: the gimbal channel the "
                             "drone detected from the autopilot's SERVOx_FUNCTION)")
    parser.add_argument("--pulse", type=int, default=None, help="servo pulse in microseconds")
    parser.add_argument("--angle", type=float, default=None, help="servo angle in degrees")
    parser.add_argument("--dry-run", action="store_true", help="print the payload without sending")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    payload = {"type": args.cmd_type}
    if args.bbox is not None:
        payload["bbox"] = [float(v) for v in args.bbox]
        payload["class_name"] = args.class_name
        payload["confidence"] = args.confidence
        payload["frame_w"] = args.frame_w
        payload["frame_h"] = args.frame_h
    if args.cmd_type == "servo":
        # Omit channel entirely when not given, so the drone uses its detected
        # gimbal channel rather than being told a hardcoded one.
        if args.channel is not None:
            payload["channel"] = args.channel
        # Send only the field that was given, so the payload matches the
        # documented minimum instead of carrying explicit nulls.
        if args.pulse is not None:
            payload["pulse"] = args.pulse
        if args.angle is not None:
            payload["angle"] = args.angle

    body = json.dumps(payload).encode("utf-8")
    if args.dry_run:
        print(body.decode("utf-8"))
        return 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(body, (args.host, args.port))
    except OSError as exc:
        print(f"Failed to send to {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    finally:
        sock.close()

    print(f"Sent '{args.cmd_type}' to {args.host}:{args.port} "
          f"({len(body)} bytes) - check the drone console for the result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
