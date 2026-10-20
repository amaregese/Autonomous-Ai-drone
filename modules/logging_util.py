"""
Shared console logging that coexists with the live one-line status.

The app repaints a single ``\\r``-updated status line every frame
(``_console_status``). A plain ``print`` from another thread while that line is
up writes over it, which is what produced run-together output such as
``... IDLESITL: Vehicle armed``. Every console log in the app and in the drone
backend goes through :func:`log`, which clears the status line first and forces
the next frame to repaint it, so each log keeps its own row.

The status line itself is announced with :func:`set_status_active`; only the
code that owns that line needs to call it. This module holds no state beyond
that one flag and never blocks, so it is safe to call from any thread.
"""
from __future__ import annotations

import shutil
import sys

_status_active = False


def set_status_active(active: bool) -> None:
    """Record whether a ``\\r`` status line is currently on screen."""
    global _status_active
    _status_active = active


def clear_status() -> None:
    """Erase the live status line, if one is on screen.

    The erase is done by overwriting with spaces rather than an ANSI
    ``\\033[K``: the status line is drawn in PyCharm's Run console, which
    renders the colour codes but leaves a bare ``\\033[K`` as literal
    ``[K`` text, which is exactly the garbage this is meant to prevent.
    """
    global _status_active
    if _status_active:
        cols = shutil.get_terminal_size().columns
        sys.stdout.write("\r" + " " * max(cols - 1, 0) + "\r")
        sys.stdout.flush()
        _status_active = False


def log(tag: str, msg: str) -> None:
    """Print one standard ``[TAG] message`` line without clobbering the status line."""
    clear_status()
    print(f"[{tag}] {msg}")
