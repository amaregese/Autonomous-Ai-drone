from __future__ import annotations

import json
import logging
import socket
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)

# How many commands may queue up while the main loop is busy. A burst from the
# SGC must not be silently dropped (an overwritten `panic_rtl` would be fatal),
# but an unbounded queue would let a flood grow without limit.
MAX_PENDING_COMMANDS = 32

# The SGC camera-servo protocol bounds: pulse in microseconds, channel is a
# physical output. These mirror the vehicle-side limits in
# modules/drone_backend/servo_channels.py (SERVO_PULSE_MIN/MAX,
# SERVO_CHANNEL_MIN/MAX), but are stated here as protocol numbers so the
# receiver stays stdlib-only and validates the UDP payload before anything
# vehicle-side is involved.
SERVO_PULSE_MIN = 1000
SERVO_PULSE_MAX = 2000
SERVO_CHANNEL_MIN = 1
SERVO_CHANNEL_MAX = 16


@dataclass(frozen=False, slots=True)
class SGCCommand:
    command_type: str = ""
    bbox: Optional[List[float]] = None
    class_name: str = ""
    confidence: float = 0.0
    frame_w: int = 0
    frame_h: int = 0
    # None means "the channel the drone detected", not a hardcoded default:
    # every SGC that omitted `channel` used to be silently sent to channel 8.
    channel: Optional[int] = None
    pulse: Optional[int] = None
    angle: Optional[float] = None


def _xywh_to_corners(bbox) -> Optional[List[float]]:
    """Normalise an SGC bbox to ``[x1, y1, x2, y2]``.

    The drone *sends* detections as ``BBox(x, y, width, height)``
    (``detection_sender._convert_detection``), so the ground station echoes the
    same ``[x, y, w, h]`` form back. Converting here keeps that round trip
    working. Returns None for anything malformed rather than raising, because a
    bad packet must never take the flight loop down.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return [x, y, x + w, y + h]


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _find_best_match(
    sgc_bbox,
    sgc_class: str,
    detections,
    min_iou: float = 0.3,
):
    """Best detection matching the SGC's ``[x, y, w, h]`` box, or None."""
    corners = _xywh_to_corners(sgc_bbox)
    if corners is None:
        logger.warning("SGC bbox is not [x, y, w, h]: %r", sgc_bbox)
        return None
    best_iou = 0.0
    best_det = None
    for det in detections:
        if sgc_class and det.class_name != sgc_class:
            continue
        det_bbox = [det.Left, det.Top, det.Right, det.Bottom]
        iou = _bbox_iou(corners, det_bbox)
        if iou > best_iou:
            best_iou = iou
            best_det = det
    if best_iou >= min_iou:
        return best_det
    return None


def parse_sgc_command(data: bytes) -> SGCCommand:
    """Decode one SGC datagram into an ``SGCCommand``.

    Raises ``UnicodeDecodeError``/``json.JSONDecodeError`` for a packet that is
    not UTF-8 JSON, and ``ValueError`` for a well-formed object that breaks the
    protocol: anything without a non-empty string ``type``, or a ``servo``
    command whose ``pulse``/``channel`` is outside the documented ranges.

    Validation is deliberately strict rather than coercing or clamping: the
    sender asked for a specific servo position, and silently turning
    ``channel: "8"`` into 8 or ``pulse: 3000`` into 2000 would move hardware in
    a way the SGC never requested. A rejected packet is dropped and logged by
    the caller - it must never take the flight loop down.
    """
    msg = json.loads(data.decode("utf-8"))
    if not isinstance(msg, dict):
        raise ValueError(
            f"SGC payload must be a JSON object, got {type(msg).__name__}")

    command_type = msg.get("type")
    if not isinstance(command_type, str) or not command_type:
        raise ValueError(
            f"SGC payload needs a non-empty string 'type', got {command_type!r}")

    channel = msg.get("channel")
    pulse = msg.get("pulse")
    if command_type == "servo":
        # `bool` is an int subclass, so True would pass an isinstance(int)
        # check; JSON `true` is not a channel or a pulse.
        if channel is not None and (
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not SERVO_CHANNEL_MIN <= channel <= SERVO_CHANNEL_MAX
        ):
            raise ValueError(
                "servo 'channel' must be an integer "
                f"{SERVO_CHANNEL_MIN}-{SERVO_CHANNEL_MAX}, got {channel!r}")
        if pulse is not None and (
            isinstance(pulse, bool)
            or not isinstance(pulse, int)
            or not SERVO_PULSE_MIN <= pulse <= SERVO_PULSE_MAX
        ):
            raise ValueError(
                "servo 'pulse' must be an integer "
                f"{SERVO_PULSE_MIN}-{SERVO_PULSE_MAX}, got {pulse!r}")

    return SGCCommand(
        command_type=command_type,
        bbox=msg.get("bbox"),
        class_name=msg.get("class_name", ""),
        confidence=msg.get("confidence", 0.0),
        frame_w=msg.get("frame_w", 0),
        frame_h=msg.get("frame_h", 0),
        # None (missing `channel`) means automatic/default channel selection
        # downstream - it is not an error, so it is passed through as-is.
        channel=channel,
        pulse=pulse,
        angle=msg.get("angle"),
    )


class SGCCommandReceiver:
    def __init__(self, port: int = 9002) -> None:
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        # A queue, not a single slot: a single slot silently drops every command
        # that arrives while the previous one is still pending, which would let a
        # `panic_rtl` be overwritten by a stray servo command.
        self._pending: Deque[SGCCommand] = deque(maxlen=MAX_PENDING_COMMANDS)
        self._last_peer: Optional[str] = None
        self._dropped = 0

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="sgc-cmd-rx")
        self._thread.start()
        logger.info("SGC command receiver listening on UDP port %d", self._port)

    def _listen_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                cmd = parse_sgc_command(data)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("Bad SGC command from %s: %s", addr, exc)
                continue
            except ValueError as exc:
                logger.warning("Rejected SGC command from %s: %s", addr, exc)
                continue
            with self._lock:
                if len(self._pending) == self._pending.maxlen:
                    self._dropped += 1
                    logger.warning(
                        "SGC command queue full (%d) - dropping oldest command",
                        self._pending.maxlen)
                self._pending.append(cmd)
                self._last_peer = addr[0] if addr else None
            logger.debug("SGC command: %s", cmd.command_type)

    def pop_command(self) -> Optional[SGCCommand]:
        """Oldest pending command, or None. FIFO - order of sending is kept."""
        with self._lock:
            if not self._pending:
                return None
            return self._pending.popleft()

    def drain_commands(self) -> List[SGCCommand]:
        """Every pending command, oldest first, leaving the queue empty."""
        with self._lock:
            cmds = list(self._pending)
            self._pending.clear()
            return cmds

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def dropped_count(self) -> int:
        """Commands discarded because the queue was full."""
        with self._lock:
            return self._dropped

    def last_peer(self) -> Optional[str]:
        """Address of the most recent SGC that sent us a command."""
        with self._lock:
            return self._last_peer

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        logger.info("SGC command receiver stopped")
