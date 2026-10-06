"""Flight-controller link selection and the SITL fallback.

Covers the rule that a run prefers the real drone and only substitutes the
simulator when no flight controller answers. The important regressions here are
the two that used to break a bench run:

- ``_default_fcu_serial`` returned a hardcoded ``COM3`` that did not exist, so
  the run died on the port instead of falling through to SITL.
- a bare ``udp:`` endpoint was classified as a *real* FCU, which put the run in
  flight mode while it was talking to the simulator.

``autonomous_drone_main.py`` runs ``setup()`` at import time and pulls in cv2 and
the detector, so these tests compile only the module-level function definitions
plus the two literal constants. Nothing heavy is imported.
"""
import argparse
import ast
import contextlib
import glob
import io
import ipaddress
import math
import os
import socket
import sys
import unittest
from pathlib import Path

import __future__ as _future

MAIN = Path(__file__).resolve().parents[1] / "autonomous_drone_main.py"

LITERAL_CONSTANTS = {"_SITL_PREFIXES", "_SITL_ENDPOINT_DEFAULT"}


def _compile_link_section():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in LITERAL_CONSTANTS:
                    body.append(node)
                    break
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    # PEP 563 so annotations such as `-> TelemetryData` are never evaluated:
    # that type comes from a module the harness deliberately does not import.
    return compile(
        module,
        str(MAIN),
        "exec",
        flags=_future.annotations.compiler_flag,
        dont_inherit=True,
    )


_LINK_CODE = _compile_link_section()


class _FakeOs:
    def __init__(self, name):
        self.name = name


def _make_args(**overrides):
    base = dict(
        drone_link="auto",
        drone_connection=None,
        start_sitl=False,
        sitl_connection=None,
        baud=57600,
        mode="flight",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class _FakeControl:
    """Stands in for modules.control; answers per endpoint."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def connect_drone(self, connection_string, start_sitl=False, baud=None):
        self.calls.append((connection_string, start_sitl, baud))
        return self.answers.get(connection_string, False)


class _LinkHarness(unittest.TestCase):
    def harness(self, args=None, answers=None, ports=None, os_name=None,
                sgc_from_cli=True, local_ips=None):
        ns = {
            "os": os,
            "sys": sys,
            "glob": glob,
            "socket": socket,
            "ipaddress": ipaddress,
            "math": math,
            "argparse": argparse,
            "__name__": "autonomous_drone_main_link_under_test",
        }
        exec(_LINK_CODE, ns)

        ns["args"] = args if args is not None else _make_args()
        ns["control"] = _FakeControl(answers)
        ns["_sitl_started"] = False
        ns["_SGC_HOST_FROM_CLI"] = sgc_from_cli
        ns["_detect_serial_ports"] = lambda: list(ports or [])
        ns["_serial_heartbeat_ok"] = lambda *a, **k: False
        ns["_local_ipv4_addresses"] = lambda: list(local_ips or [])
        ns["saved"] = []
        ns["save_default"] = lambda key, value: ns["saved"].append((key, value))

        if os_name is not None:
            ns["os"] = _FakeOs(os_name)

        self.ns = ns
        return ns

    def run_silently(self, fn, *fn_args, **fn_kwargs):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = fn(*fn_args, **fn_kwargs)
        return result, buffer.getvalue()


class TestDefaultFcuSerial(_LinkHarness):
    def test_returns_detected_port(self):
        ns = self.harness(ports=["COM7", "COM3"], os_name="nt")
        self.assertEqual(ns["_default_fcu_serial"](), "COM7")

    def test_windows_without_a_port_returns_none(self):
        ns = self.harness(ports=[], os_name="nt")
        self.assertIsNone(ns["_default_fcu_serial"]())

    def test_windows_never_invents_com3(self):
        ns = self.harness(ports=[], os_name="nt")
        self.assertNotEqual(ns["_default_fcu_serial"](), "COM3")

    def test_posix_without_enumeration_keeps_conventional_path(self):
        ns = self.harness(ports=[], os_name="posix")
        self.assertEqual(ns["_default_fcu_serial"](), "/dev/ttyACM0")


class TestIsSitlLink(_LinkHarness):
    def test_udp_forms_are_simulator(self):
        ns = self.harness()
        for endpoint in ("udpin:0.0.0.0:14550", "udpout:127.0.0.1:14550",
                         "udp:172.20.176.1:14550", "tcp:127.0.0.1:5760"):
            with self.subTest(endpoint=endpoint):
                self.assertTrue(ns["_is_sitl_link"](endpoint))

    def test_serial_ports_are_not_simulator(self):
        ns = self.harness()
        for port in ("COM3", "COM7", "/dev/ttyACM0", "/dev/ttyUSB0"):
            with self.subTest(port=port):
                self.assertFalse(ns["_is_sitl_link"](port))


class TestLinkCandidates(_LinkHarness):
    def test_auto_puts_the_real_fcu_first(self):
        ns = self.harness(ports=["COM7"], os_name="nt")
        candidates = ns["_link_candidates"]()
        self.assertEqual(
            candidates,
            [("COM7", False, "real"),
             ("udpin:0.0.0.0:14550", False, "sitl")],
        )

    def test_auto_skips_the_missing_fcu(self):
        ns = self.harness(ports=[], os_name="nt")
        self.assertEqual(
            ns["_link_candidates"](),
            [("udpin:0.0.0.0:14550", False, "sitl")],
        )

    def test_auto_honours_a_configured_endpoint(self):
        ns = self.harness(
            args=_make_args(sitl_connection="udp:127.0.0.1:14540"),
            ports=[], os_name="nt",
        )
        self.assertEqual(
            ns["_link_candidates"](),
            [("udp:127.0.0.1:14540", False, "sitl")],
        )

    def test_explicit_real_never_falls_back(self):
        ns = self.harness(
            args=_make_args(drone_link="real"), ports=["COM7"], os_name="nt"
        )
        self.assertEqual(ns["_link_candidates"](), [("COM7", False, "real")])

    def test_explicit_real_without_a_controller_is_fatal(self):
        ns = self.harness(
            args=_make_args(drone_link="real"), ports=[], os_name="nt"
        )
        with self.assertRaises(SystemExit) as raised:
            ns["_link_candidates"]()
        self.assertIn("--drone-link auto", str(raised.exception))

    def test_explicit_sitl_stays_on_the_simulator(self):
        ns = self.harness(args=_make_args(drone_link="sitl"), ports=["COM7"])
        self.assertEqual(
            ns["_link_candidates"](),
            [("udpin:0.0.0.0:14550", False, "sitl")],
        )

    def test_explicit_endpoint_is_used_verbatim(self):
        ns = self.harness(
            args=_make_args(drone_connection="COM9"), ports=["COM7"]
        )
        self.assertEqual(ns["_link_candidates"](), [("COM9", False, "real")])

    def test_explicit_udp_endpoint_is_classified_as_simulator(self):
        ns = self.harness(
            args=_make_args(drone_connection="udp:172.20.176.1:14550")
        )
        self.assertEqual(
            ns["_link_candidates"](),
            [("udp:172.20.176.1:14550", False, "sitl")],
        )

    def test_start_sitl_flag_selects_only_the_simulator(self):
        ns = self.harness(args=_make_args(start_sitl=True), ports=["COM7"])
        candidates = ns["_link_candidates"]()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][2], "sitl")


class TestPickConnection(_LinkHarness):
    def test_returns_the_first_candidate(self):
        ns = self.harness(
            args=_make_args(baud=115200), ports=["COM7"], os_name="nt"
        )
        connection_string, start_sitl, baud = ns["_pick_connection"]()
        self.assertEqual(connection_string, "COM7")
        self.assertFalse(start_sitl)
        self.assertEqual(baud, 115200)

    def test_announces_the_fallback_only_when_one_is_left(self):
        ns = self.harness(ports=["COM7"], os_name="nt")
        _, output = self.run_silently(ns["_pick_connection"])
        self.assertIn("fall back to SITL", output)

    def test_sitl_only_run_does_not_claim_a_fallback(self):
        ns = self.harness(ports=[], os_name="nt")
        _, output = self.run_silently(ns["_pick_connection"])
        self.assertNotIn("fall back to SITL", output)


class TestDescribeLink(_LinkHarness):
    def test_real_kind_names_the_fcu(self):
        ns = self.harness()
        self.assertEqual(
            ns["_describe_link"]("COM7", 115200, False, "real"),
            "real FCU on COM7 @ 115200 baud",
        )

    def test_sitl_kind_names_the_simulator(self):
        ns = self.harness()
        self.assertEqual(
            ns["_describe_link"]("udpin:0.0.0.0:14550", 57600, False, "sitl"),
            "SITL over UDP (udpin:0.0.0.0:14550)",
        )

    def test_bare_udp_is_not_reported_as_a_real_fcu(self):
        ns = self.harness()
        label = ns["_describe_link"]("udp:172.20.176.1:14550", 57600)
        self.assertIn("SITL", label)
        self.assertNotIn("real FCU", label)

    def test_launched_sitl_says_so(self):
        ns = self.harness()
        self.assertIn(
            "launched in WSL",
            ns["_describe_link"]("udpin:0.0.0.0:14550", 57600, True, "sitl"),
        )


class TestConnectWithFallback(_LinkHarness):
    def test_real_fcu_wins_and_sitl_is_never_touched(self):
        ns = self.harness(
            ports=["COM7"], os_name="nt",
            answers={"COM7": True, "udpin:0.0.0.0:14550": True},
        )
        connection_string, output = self.run_silently(
            ns["_connect_with_fallback"], "192.168.1.160"
        )
        self.assertEqual(connection_string, "COM7")
        self.assertEqual(ns["args"].mode, "flight")
        self.assertNotIn("falling back", output)
        self.assertEqual([c[0] for c in ns["control"].calls], ["COM7"])

    def test_missing_fcu_falls_back_to_sitl_and_switches_mode(self):
        ns = self.harness(
            ports=["COM7"], os_name="nt",
            answers={"udpin:0.0.0.0:14550": True},
        )
        connection_string, output = self.run_silently(
            ns["_connect_with_fallback"], "192.168.1.160"
        )
        self.assertEqual(connection_string, "udpin:0.0.0.0:14550")
        self.assertEqual(ns["args"].mode, "sitl")
        self.assertIn("falling back", output)
        self.assertIn("running in sitl mode", output)
        self.assertEqual(
            [c[0] for c in ns["control"].calls],
            ["COM7", "udpin:0.0.0.0:14550"],
        )

    def test_explicit_real_does_not_quietly_substitute_the_simulator(self):
        ns = self.harness(
            args=_make_args(drone_link="real"), ports=[], os_name="nt",
            answers={"udpin:0.0.0.0:14550": True},
        )
        with self.assertRaises(SystemExit):
            self.run_silently(
                ns["_connect_with_fallback"], "192.168.1.160"
            )
        self.assertEqual(ns["control"].calls, [])

    def test_total_failure_lists_every_attempt(self):
        ns = self.harness(ports=["COM7"], os_name="nt", answers={})
        with self.assertRaises(SystemExit) as raised:
            self.run_silently(ns["_connect_with_fallback"], "192.168.1.160")
        self.assertEqual(raised.exception.code, 1)

    def test_flight_mode_is_restored_when_a_real_fcu_answers(self):
        ns = self.harness(
            args=_make_args(mode="sitl"), ports=["COM7"], os_name="nt",
            answers={"COM7": True},
        )
        self.run_silently(ns["_connect_with_fallback"], "192.168.1.160")
        self.assertEqual(ns["args"].mode, "flight")


class TestSitlSgcRemap(_LinkHarness):
    def test_cli_host_is_left_alone(self):
        ns = self.harness(sgc_from_cli=True, local_ips=["10.0.0.5"])
        ns["_remap_sgc_host_for_sitl"]("192.168.1.160")
        self.assertEqual(ns["saved"], [])

    def test_remembered_host_moves_to_this_machine(self):
        ns = self.harness(sgc_from_cli=False, local_ips=["10.0.0.5"])
        ns["_remap_sgc_host_for_sitl"]("192.168.1.160")
        self.assertEqual(ns["saved"], [("sgc_host", "10.0.0.5")])


if __name__ == "__main__":
    unittest.main()