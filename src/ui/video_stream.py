import cv2
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


class MJPEGStreamer:
    def __init__(self, port=8080):
        self.port = port
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        self.server = None

    def start(self):
        self.running = True

        def run_server():
            self.server = HTTPServer(('127.0.0.1', self.port), self._create_handler())
            self.server.streamer = self
            self.server.serve_forever()

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        print(f"MJPEG stream: http://127.0.0.1:{self.port}/video")

    def _create_handler(self):
        streamer = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/video':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    try:
                        while streamer.running:
                            with streamer.lock:
                                frame = streamer.latest_frame
                            if frame is not None:
                                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                data = jpeg.tobytes()
                                self.wfile.write(b'--frame\r\n')
                                self.send_header('Content-Type', 'image/jpeg')
                                self.send_header('Content-Length', str(len(data)))
                                self.end_headers()
                                self.wfile.write(data)
                                self.wfile.write(b'\r\n')
                            time.sleep(0.03)
                    except Exception:
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass
        return Handler

    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame

    def stop(self):
        self.running = False
        if self.server:
            self.server.shutdown()
