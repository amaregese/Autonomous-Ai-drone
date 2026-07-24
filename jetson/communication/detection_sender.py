from __future__ import annotations

import logging
import time
from typing import Optional

import sys
sys.path.insert(0, "modules")

from modules import detector_yolo11 as detector
from shared.detection_models import (
    BBox,
    Detection as SharedDetection,
    FrameDetections,
    OverlayConfig,
    Point,
    _IDLE_COLOR,
    _LOST_COLOR,
    _SELECTED_COLOR,
    _TRACKING_COLOR,
    _bbox_color_for_state,
)
from shared.detection_transport import DetectionTransport
from jetson.streaming.rtsp_server import RTSPServer

logger = logging.getLogger(__name__)


def _convert_detection(det, is_selected: bool, tracker_state: str) -> SharedDetection:
    bbox = BBox(x=det.Left, y=det.Top, width=det.width, height=det.height)
    center = Point(x=det.Center[0], y=det.Center[1])
    if is_selected:
        color = _SELECTED_COLOR
    else:
        color = _bbox_color_for_state(tracker_state)
    return SharedDetection(
        class_name=det.class_name,
        confidence=det.confidence,
        bbox=bbox,
        center=center,
        timestamp=time.time(),
        detection_id=id(det),
        is_selected=is_selected,
        bbox_color=color,
    )


def _get_tracker_state() -> str:
    selected = detector.get_selected_object()
    if selected is None:
        return "idle"
    if detector.get_tracking_status():
        return "lost"
    return "tracking"


def _build_overlay(tracker_state: str, num_detections: int) -> OverlayConfig:
    if tracker_state == "tracking":
        bbox_color = _TRACKING_COLOR
    elif tracker_state == "lost":
        bbox_color = _LOST_COLOR
    else:
        bbox_color = _IDLE_COLOR

    if tracker_state == "idle":
        prompt_text = "Click on any object to track"
    elif tracker_state == "lost":
        prompt_text = "Object lost — click to re-acquire"
    else:
        prompt_text = None

    counter_text = f"Objects detected: {num_detections}" if num_detections > 0 else None

    return OverlayConfig(
        bbox_color=bbox_color,
        bbox_line_width=2,
        label_format="class_only",
        show_center_dot=True,
        center_dot_color="#ffffff",
        center_dot_radius=3,
        counter_text=counter_text,
        counter_position="top_left",
        counter_bg="rgba(0,0,0,0.6)",
        counter_color="#ffffff",
        counter_font="bold 11px monospace",
        prompt_text=prompt_text,
        prompt_position="bottom_center",
        prompt_bg="rgba(0,0,0,0.6)",
        prompt_color="#ffffff",
        prompt_font="bold 11px monospace",
        show_fps=True,
        fps_position="bottom_left",
        fps_color="#4ade80",
        fps_bg="rgba(0,0,0,0.6)",
        fps_font="bold 11px monospace",
    )


class Streamer:
    def __init__(
        self,
        rtsp_server: RTSPServer,
        transport: DetectionTransport,
        source_id: str = "drone_0",
    ) -> None:
        self._rtsp = rtsp_server
        self._transport = transport
        self._source_id = source_id
        self._frame_id = 0
        self._active = False

    def start(self) -> None:
        self._rtsp.start()
        self._active = True
        logger.info("Streaming started: %s", self._rtsp.stream_url)

    def push(self, frame, detections, fps: float) -> None:
        if not self._active:
            return
        self._frame_id += 1
        self._rtsp.push_frame(frame)

        selected_obj = detector.get_selected_object()
        tracker_state = _get_tracker_state()

        h, w = frame.shape[:2]
        shared = [
            _convert_detection(
                d,
                is_selected=(selected_obj is not None and d is selected_obj),
                tracker_state=tracker_state,
            )
            for d in detections
        ]
        overlay = _build_overlay(tracker_state, len(shared))
        fd = FrameDetections(
            frame_id=self._frame_id,
            detections=shared,
            timestamp=time.time(),
            source_id=self._source_id,
            fps=fps,
            frame_w=w,
            frame_h=h,
            tracker_state=tracker_state,
            overlay=overlay,
        )
        self._transport.send(fd)

    def stop(self) -> None:
        self._active = False
        self._rtsp.stop()
        self._transport.close()
        logger.info("Streaming stopped")
