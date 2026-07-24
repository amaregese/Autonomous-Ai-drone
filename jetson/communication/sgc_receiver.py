from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class SGCCommand:
    command_type: str = ""
    bbox: Optional[List[float]] = None
    class_name: str = ""
    confidence: float = 0.0
    frame_w: int = 0
    frame_h: int = 0


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
    sgc_bbox: List[float],
    sgc_class: str,
    detections,
    min_iou: float = 0.3,
):
    best_iou = 0.0
    best_det = None
    for det in detections:
        if sgc_class and det.class_name != sgc_class:
            continue
        det_bbox = [det.Left, det.Top, det.Right, det.Bottom]
        iou = _bbox_iou(sgc_bbox, det_bbox)
        if iou > best_iou:
            best_iou = iou
            best_det = det
    if best_iou >= min_iou:
        return best_det
    return None


class SGCCommandReceiver:
    def __init__(self, port: int = 9002) -> None:
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._pending: Optional[SGCCommand] = None

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
                msg = json.loads(data.decode("utf-8"))
                cmd = SGCCommand(
                    command_type=msg.get("type", ""),
                    bbox=msg.get("bbox"),
                    class_name=msg.get("class_name", ""),
                    confidence=msg.get("confidence", 0.0),
                    frame_w=msg.get("frame_w", 0),
                    frame_h=msg.get("frame_h", 0),
                )
                with self._lock:
                    self._pending = cmd
                logger.debug("SGC command: %s", cmd.command_type)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("Bad SGC command from %s: %s", addr, exc)

    def pop_command(self) -> Optional[SGCCommand]:
        with self._lock:
            cmd = self._pending
            self._pending = None
            return cmd

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        logger.info("SGC command receiver stopped")
