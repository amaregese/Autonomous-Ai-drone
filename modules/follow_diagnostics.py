"""
Read-only selected-target distance diagnostics for the console status line.

This module only formats already-computed measurements for display. It never
computes a distance, commands movement, or talks to MAVLink, so it cannot affect
flight behaviour. Missing or invalid values render as ``--`` and are never
fabricated.

Labels:
    Z   = existing optical-axis ``distance_m`` (authoritative follow range)
    R   = calculated camera-to-target slant range
    H   = calculated horizontal ground distance
    AGL = altitude supplied to the horizontal calculation
"""
from __future__ import annotations

import math


def metric_text(value) -> str:
    """Format a metre value for the status line, or ``--`` when unusable."""
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(value) or value <= 0.0:
        return "--"
    return f"{value:.2f}m"


def diagnostic_text(z=None, r=None, h=None, agl=None) -> str:
    """Label the four observational values on one status segment."""
    return (
        f"\033[2mZ\033[0m={metric_text(z)} "
        f"\033[2mR\033[0m={metric_text(r)} "
        f"\033[2mH\033[0m={metric_text(h)} "
        f"\033[2mAGL\033[0m={metric_text(agl)}"
    )


def selected_diagnostic(selected_obj, z=None) -> str:
    """Build the Z/R/H/AGL readout for the selected target.

    ``z`` is the existing optical-axis ``distance_m`` supplied by the caller, so
    this module never re-derives the authoritative range. R/H/AGL are shown only
    when the horizontal measurement actually succeeded; otherwise they render
    as ``--``.
    """
    r = h = agl = None
    if selected_obj is not None and bool(getattr(selected_obj, "horizontal_distance_valid", False)):
        r = getattr(selected_obj, "slant_range_m", None)
        h = getattr(selected_obj, "horizontal_distance_m", None)
        agl = getattr(selected_obj, "horizontal_altitude_m", None)
    return diagnostic_text(z=z, r=r, h=h, agl=agl)
