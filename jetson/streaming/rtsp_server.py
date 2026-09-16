from __future__ import annotations

import io
import logging
import socket
import struct
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _get_lan_ip() -> str:
    """Auto-detect the LAN IP of the active network interface."""
    # Method 1: Connect to external route — OS picks the right interface
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # Method 2:hostname resolution
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # Method 3: Enumerate all interfaces, pick the first non-loopback IPv4
    try:
        hostname = socket.gethostname()
        for addr_info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = addr_info[4][0]
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return "127.0.0.1"


class _FrameHolder:
    def __init__(self, jpeg_quality: int = 50) -> None:
        self._frames: list[Optional[np.ndarray]] = [None, None]
        self._write_idx: int = 0
        self._frame_id: int = 0
        self.jpeg_quality: int = jpeg_quality
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._write_idx ^= 1
            self._frames[self._write_idx] = frame.copy()
            self._frame_id += 1
        self._new_frame_event.set()

    def wait_for_frame(self, last_frame_id: int, timeout: float = 0.05) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._frame_id != last_frame_id:
                return self._frames[self._write_idx], self._frame_id
        self._new_frame_event.wait(timeout=timeout)
        self._new_frame_event.clear()
        with self._lock:
            return self._frames[self._write_idx], self._frame_id

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            frame = self._frames[self._write_idx]
        return frame.copy() if frame is not None else None

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frames[self._write_idx]
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buf.tobytes() if ok else None


class _MJPEGHandler(BaseHTTPRequestHandler):
    frame_holder: _FrameHolder

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        last_frame_id = 0
        try:
            while True:
                frame, last_frame_id = self.frame_holder.wait_for_frame(last_frame_id, timeout=0.05)
                if frame is None:
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.frame_holder.jpeg_quality])
                if not ok:
                    continue
                jpeg = buf.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except Exception:
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _HTTPServerThread(threading.Thread):
    def __init__(self, holder: _FrameHolder, port: int) -> None:
        super().__init__(daemon=True, name="mjpeg-server")
        self._holder = holder
        self._port = port
        self._server: Optional[_ThreadedHTTPServer] = None

    def run(self) -> None:
        handler_class = type(
            "Handler",
            (_MJPEGHandler,),
            {"frame_holder": self._holder},
        )
        self._server = _ThreadedHTTPServer(("0.0.0.0", self._port), handler_class)
        logger.info("MJPEG server listening on http://0.0.0.0:%d/stream", self._port)
        self._server.serve_forever()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()


class RTSPServer:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        port: int = 8554,
        stream_name: str = "drone",
        jpeg_quality: int = 50,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._port = port
        self._stream_name = stream_name
        self._holder = _FrameHolder(jpeg_quality=jpeg_quality)
        self._http_thread: Optional[_HTTPServerThread] = None
        self._gstreamer_writer: Optional[cv2.VideoWriter] = None
        self._running = False
        self._backend = "none"

    @property
    def stream_url(self) -> str:
        ip = _get_lan_ip()
        if self._backend == "gstreamer":
            return f"rtsp://{ip}:{self._port}/{self._stream_name}"
        return f"http://{ip}:{self._port}/stream"

    def push_frame(self, frame: np.ndarray) -> None:
        self._holder.push(frame)

    def start(self, camera_index: int = 0) -> None:
        if self._running:
            return
        self._running = True

        if self._try_gstreamer():
            self._backend = "gstreamer"
            logger.info("Streaming via GStreamer RTSP")
        elif self._try_ffmpeg():
            self._backend = "ffmpeg"
            logger.info("Streaming via FFmpeg RTSP")
        else:
            self._backend = "mjpeg"
            self._http_thread = _HTTPServerThread(self._holder, self._port)
            self._http_thread.start()
            logger.info("Streaming via MJPEG HTTP (fallback)")

    def _try_gstreamer(self) -> bool:
        try:
            pipeline = (
                f"gst-launch-1.0 -v udpsink host=127.0.0.1 port={self._port} "
                f"sync=false"
            )
            gst_pipe = (
                "appsrc ! videoconvert ! x264enc tune=zerolatency "
                "bitrate=800 speed-preset=ultrafast ! rtph264pay ! udpsink "
                f"host=127.0.0.1 port={self._port}"
            )
            self._gstreamer_writer = cv2.VideoWriter(
                gst_pipe, cv2.CAP_GSTREAMER, 0, self._fps, (self._width, self._height), True
            )
            if not self._gstreamer_writer.isOpened():
                self._gstreamer_writer = None
                return False

            self._writer_thread = threading.Thread(
                target=self._gstreamer_write_loop, daemon=True, name="gst-writer"
            )
            self._writer_thread.start()
            return True
        except Exception:
            self._gstreamer_writer = None
            return False

    def _gstreamer_write_loop(self) -> None:
        while self._running and self._gstreamer_writer:
            frame = self._holder.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            if frame.shape[:2] != (self._height, self._width):
                frame = cv2.resize(frame, (self._width, self._height))
            try:
                self._gstreamer_writer.write(frame)
            except Exception:
                logger.exception("GStreamer write error")
                time.sleep(0.1)

    def _try_ffmpeg(self) -> bool:
        try:
            url = f"rtsp://0.0.0.0:{self._port}/{self._stream_name}"
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self._width}x{self._height}", "-r", str(self._fps),
                "-i", "-",
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-f", "rtsp", "-rtsp_transport", "tcp", url,
            ]
            self._pipe = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self._pipe.stdin.write(b"\x00")
            self._pipe.stdin.flush()
            self._writer_thread = threading.Thread(
                target=self._ffmpeg_write_loop, daemon=True, name="ffmpeg-writer"
            )
            self._writer_thread.start()
            return True
        except (FileNotFoundError, Exception):
            return False

    def _cleanup_ffmpeg(self) -> None:
        pipe = getattr(self, "_pipe", None)
        if pipe is None:
            return
        try:
            if pipe.stdin:
                pipe.stdin.close()
        except Exception:
            pass
        if pipe.poll() is None:
            try:
                pipe.terminate()
                pipe.wait(timeout=3.0)
            except Exception:
                try:
                    pipe.kill()
                except Exception:
                    pass
        self._pipe = None

    def _ffmpeg_write_loop(self) -> None:
        while self._running:
            frame = self._holder.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            if frame.shape[:2] != (self._height, self._width):
                frame = cv2.resize(frame, (self._width, self._height))
            if self._pipe is None or self._pipe.stdin is None or self._pipe.poll() is not None:
                break
            try:
                self._pipe.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                logger.warning("FFmpeg pipe closed (client disconnected) — stopping FFmpeg backend")
                break
        self._cleanup_ffmpeg()
        if self._running and self._http_thread is None:
            try:
                self._http_thread = _HTTPServerThread(self._holder, self._port)
                self._http_thread.start()
                self._backend = "mjpeg"
                logger.info("Streaming via MJPEG HTTP (fallback after FFmpeg exited)")
            except Exception:
                logger.exception("Failed to start MJPEG fallback")

    def stop(self) -> None:
        self._running = False
        if self._backend == "gstreamer" and self._gstreamer_writer:
            self._gstreamer_writer.release()
            self._gstreamer_writer = None
        elif self._backend == "ffmpeg":
            self._cleanup_ffmpeg()
        if self._http_thread:
            self._http_thread.stop()
        logger.info("Stream server stopped")
