from __future__ import annotations

import json
import logging
import socket
import struct
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

from shared.detection_models import FrameDetections

logger = logging.getLogger(__name__)


class DetectionTransport(ABC):
    @abstractmethod
    def send(self, frame_detections: FrameDetections) -> None: ...

    @abstractmethod
    def start_receiving(
        self, callback: Callable[[FrameDetections], None]
    ) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class UDPTransport(DetectionTransport):
    MAX_DGRAM = 65507

    def __init__(self, host: str, port: int, bind: bool = False) -> None:
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._receiver_thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[FrameDetections], None]] = None
        self._bind = bind
        self._lock = threading.Lock()
        self._create_socket()

    def _create_socket(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self._bind:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self._host, self._port))
        self._sock.settimeout(0.5)

    def send(self, frame_detections: FrameDetections) -> None:
        if self._sock is None:
            print("[UDP] ERROR: socket is None, cannot send")
            return
        try:
            payload = json.dumps(frame_detections.to_dict()).encode("utf-8")
            header = struct.pack("!I", len(payload))
            with self._lock:
                self._sock.sendto(header + payload, (self._host, self._port))
        except Exception as e:
            print(f"[UDP] SEND FAILED: {e}")
            logger.exception("UDP send failed")

    def start_receiving(
        self, callback: Callable[[FrameDetections], None]
    ) -> None:
        self._callback = callback
        self._running = True
        self._receiver_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="udp-receiver"
        )
        self._receiver_thread.start()
        logger.info("UDP receiver started on %s:%d", self._host, self._port)

    def _receive_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(self.MAX_DGRAM + 4)
                if len(data) < 4:
                    continue
                length = struct.unpack("!I", data[:4])[0]
                payload = data[4 : 4 + length]
                frame_det = FrameDetections.from_dict(json.loads(payload.decode("utf-8")))
                if self._callback:
                    self._callback(frame_det)
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    logger.exception("UDP receive error")

    def stop(self) -> None:
        self._running = False
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=2.0)

    def close(self) -> None:
        self.stop()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


class TCPTransport(DetectionTransport):
    def __init__(self, host: str, port: int, server: bool = False) -> None:
        self._host = host
        self._port = port
        self._server = server
        self._sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._running = False
        self._receiver_thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[FrameDetections], None]] = None
        self._lock = threading.Lock()

    def _ensure_connected(self) -> None:
        if self._conn is not None:
            return
        if self._server:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._host, self._port))
            srv.listen(1)
            srv.settimeout(0.5)
            try:
                self._conn, addr = srv.accept()
                logger.info("TCP connection from %s", addr)
            except socket.timeout:
                pass
            finally:
                srv.close()
        else:
            try:
                self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._conn.connect((self._host, self._port))
            except Exception:
                logger.exception("TCP connect failed")
                self._conn = None

    def send(self, frame_detections: FrameDetections) -> None:
        self._ensure_connected()
        if self._conn is None:
            return
        try:
            payload = json.dumps(frame_detections.to_dict()).encode("utf-8")
            header = struct.pack("!I", len(payload))
            with self._lock:
                self._conn.sendall(header + payload)
        except Exception:
            logger.exception("TCP send failed")
            self._conn = None

    def start_receiving(
        self, callback: Callable[[FrameDetections], None]
    ) -> None:
        self._callback = callback
        self._running = True
        self._receiver_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="tcp-receiver"
        )
        self._receiver_thread.start()

    def _receive_loop(self) -> None:
        while self._running:
            self._ensure_connected()
            if self._conn is None:
                try:
                    import time as _time
                    _time.sleep(0.5)
                except Exception:
                    pass
                continue
            try:
                header = self._recv_exact(4)
                if header is None:
                    self._conn = None
                    continue
                length = struct.unpack("!I", header)[0]
                payload = self._recv_exact(length)
                if payload is None:
                    self._conn = None
                    continue
                frame_det = FrameDetections.from_dict(
                    json.loads(payload.decode("utf-8"))
                )
                if self._callback:
                    self._callback(frame_det)
            except Exception:
                if self._running:
                    logger.exception("TCP receive error")
                    self._conn = None

    def _recv_exact(self, n: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < n and self._running:
            try:
                chunk = self._conn.recv(n - len(buf))
                if not chunk:
                    return None
                buf.extend(chunk)
            except socket.timeout:
                continue
            except Exception:
                return None
        return bytes(buf) if len(buf) == n else None

    def stop(self) -> None:
        self._running = False
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=2.0)

    def close(self) -> None:
        self.stop()
        for sock in (self._conn, self._sock):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        self._conn = None
        self._sock = None
