from __future__ import annotations

import logging
import math
import time
from typing import Optional

import sys
sys.path.insert(0, "modules")

from modules import detector_yolo11 as detector
from modules.display import hud
from shared.detection_models import (
    BBox,
    CameraIntrinsics,
    Detection as SharedDetection,
    FrameDetections,
    OverlayConfig,
    Point,
    TelemetryData,
    TrackingData,
    _IDLE_COLOR,
    _LOST_COLOR,
    _SELECTED_COLOR,
    _TRACKING_COLOR,
    _bbox_color_for_state,
)
from shared.detection_transport import DetectionTransport
from jetson.streaming.rtsp_server import RTSPServer

logger = logging.getLogger(__name__)


def _bgr_to_hex(color: tuple) -> str:
    try:
        r, g, b = int(color[2]), int(color[1]), int(color[0])
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#ffffff"


def _active_notification() -> Optional[tuple]:
    if not hud.notification or hud.notification_until <= time.time():
        return None
    if len(hud.notification_color) >= 3:
        text_color = _bgr_to_hex(hud.notification_color)
    else:
        text_color = "#ff4444"
    return hud.notification, text_color


def _detection_distance(det) -> tuple[Optional[float], bool, Optional[str], float]:
    source = getattr(det, "distance_source", None)
    if source is None:
        return None, False, None, 0.0
    distance = getattr(det, "distance_m", None)
    valid = bool(getattr(det, "distance_valid", False))
    if distance is not None:
        try:
            distance = float(distance)
            valid = valid and math.isfinite(distance) and distance > 0.0
        except (TypeError, ValueError):
            distance = None
            valid = False
    else:
        valid = False
    if not valid:
        return None, False, "none", 0.0
    return distance, True, source, float(getattr(det, "distance_confidence", 0.0))


def _convert_detection(det, is_selected: bool, tracker_state: str) -> SharedDetection:
    bbox = BBox(x=det.Left, y=det.Top, width=det.width, height=det.height)
    center = Point(x=det.Center[0], y=det.Center[1])
    distance, distance_valid, distance_source, distance_confidence = _detection_distance(det)
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
        detection_id=getattr(det, "detection_id", id(det)),
        distance=distance,
        distance_valid=distance_valid,
        distance_source=distance_source,
        distance_confidence=distance_confidence,
        is_selected=is_selected,
        bbox_color=color,
    )


def _get_tracker_state(following: bool = False) -> str:
    if detector.get_tracking_status():
        return "lost"
    if detector.get_selected_object() is not None:
        return "tracking" if following else "selected"
    return "idle"


def apply_authoritative_target_distance(shared, source_detections, selected_obj, movement) -> None:
    """Attach the one authoritative result only to its source detection."""
    if selected_obj is None:
        return
    source_distance, source_valid, source, source_confidence = _detection_distance(selected_obj)
    for raw, converted in zip(source_detections, shared):
        if raw is not selected_obj:
            continue
        target_id = (movement or {}).get("target_detection_id")
        if target_id is not None and getattr(raw, "detection_id", id(raw)) != target_id:
            return
        if source is not None:
            converted.distance = source_distance
            converted.distance_valid = source_valid
            converted.distance_source = source
            converted.distance_confidence = source_confidence
            return
        if movement and movement.get("distance_valid"):
            distance = movement.get("distance_m")
            try:
                distance = float(distance)
            except (TypeError, ValueError):
                return
            if math.isfinite(distance) and distance > 0.0:
                converted.distance = distance
                converted.distance_valid = True
                converted.distance_source = movement.get("distance_source", "follow")
                converted.distance_confidence = float(movement.get("distance_confidence", 0.0))
        return



def _build_overlay(
    tracker_state: str,
    num_detections: int,
    mode: str,
    streaming: bool,
    movement: Optional[dict],
    selected_class: Optional[str],
    tracking_conf: float,
) -> OverlayConfig:
    if tracker_state == "tracking":
        bbox_color = _TRACKING_COLOR
    elif tracker_state == "selected":
        bbox_color = _SELECTED_COLOR
    elif tracker_state == "lost":
        bbox_color = _LOST_COLOR
    else:
        bbox_color = _IDLE_COLOR

    if tracker_state == "idle":
        prompt_text = "Click on any object to track"
    elif tracker_state == "lost":
        prompt_text = "Object lost — click to re-acquire"
    elif tracker_state == "selected":
        prompt_text = "Selected — press SPACE to follow"
    else:
        prompt_text = None

    counter_text = f"Objects detected: {num_detections}" if num_detections > 0 else None

    mode_colors = {
        "TEST": ("rgba(60,60,60,0.8)", "#c8c8c8"),
        "sitl": ("rgba(0,100,180,0.8)", "#ffffff"),
        "flight": ("rgba(0,140,0,0.8)", "#ffffff"),
    }
    mode_bg, mode_color = mode_colors.get(mode, ("rgba(60,60,60,0.8)", "#c8c8c8"))

    show_tracking_bar = tracker_state == "tracking" and movement is not None
    tracking_bar_text = None
    if show_tracking_bar and selected_class:
        distance = movement.get("distance_m")
        distance_valid = bool(movement.get("distance_valid")) and distance is not None
        if distance_valid:
            try:
                distance = float(distance)
                distance_valid = math.isfinite(distance) and distance > 0.0
            except (TypeError, ValueError):
                distance_valid = False
        distance_text = f"{distance:.1f}m" if distance_valid else "N/A"
        speed = movement.get("vel_z", 0.0)
        yaw = movement.get("yaw_cmd", 0.0)
        tracking_bar_text = (
            f"Following: {selected_class}  |  "
            f"Dist: {distance_text}  |  Speed: {speed:.1f}m/s  |  "
            f"Yaw: {yaw:.1f}  |  Conf: {tracking_conf:.0f}%"
        )

    show_lost_banner = tracker_state == "lost" and selected_class is not None

    notification = _active_notification()
    show_notification = notification is not None
    notification_text = notification[0] if notification else ""
    notification_color = notification[1] if notification else "#ffffff"

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
        mode_text=mode.upper(),
        mode_position="top_center",
        mode_bg=mode_bg,
        mode_color=mode_color,
        mode_font="bold 12px monospace",
        show_stream_indicator=True,
        stream_indicator_color="#00c800" if streaming else "#c80000",
        show_tracking_bar=show_tracking_bar,
        tracking_bar_text=tracking_bar_text,
        tracking_bar_position="top_below_status",
        tracking_bar_bg="rgba(30,80,30,0.8)",
        tracking_bar_color="#b4ffb4",
        tracking_bar_font="bold 10px monospace",
        show_lost_banner=show_lost_banner,
        lost_banner_text="TARGET LOST" if show_lost_banner else "",
        lost_banner_color="#ffffff",
        lost_banner_bg="rgba(180,0,0,0.4)",
        lost_banner_font="bold 18px monospace",
        show_shortcut_bar=True,
        shortcut_bar_text="[ESC] deselect  [SPACE] follow  [R] reset  [H] hud  [Q] quit",
        shortcut_bar_position="bottom_center",
        shortcut_bar_color="#828282",
        shortcut_bar_font="9px monospace",
        show_notification=show_notification,
        notification_text=notification_text,
        notification_color=notification_color,
        notification_position="center_top",
        notification_bg="rgba(0,0,0,0.75)",
        notification_font="bold 14px monospace",
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
        self._mode: str = "test"
        self._hud_visible: bool = True
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._last_selected_class: Optional[str] = None
        self._last_state_sent: Optional[str] = None

    def start(self) -> None:
        self._rtsp.start()
        self._active = True
        logger.info("Streaming started: %s", self._rtsp.stream_url)

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_hud_visible(self, visible: bool) -> None:
        self._hud_visible = visible

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float, calib_w: int = 0, calib_h: int = 0) -> None:
        self._intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, calib_w=calib_w, calib_h=calib_h)
        logger.info("Camera intrinsics set: fx=%.1f fy=%.1f cx=%.1f cy=%.1f (calib %dx%d)", fx, fy, cx, cy, calib_w, calib_h)

    def _scaled_intrinsics(self, frame_w: int, frame_h: int) -> Optional[CameraIntrinsics]:
        if self._intrinsics is None or not self._intrinsics.is_valid:
            return None
        if self._intrinsics.calib_w > 0 and self._intrinsics.calib_h > 0:
            if frame_w != self._intrinsics.calib_w or frame_h != self._intrinsics.calib_h:
                return self._intrinsics.scaled(frame_w, frame_h)
        return self._intrinsics

    def push(
        self,
        frame,
        detections,
        fps: float,
        movement: Optional[dict] = None,
        telemetry: Optional[TelemetryData] = None,
    ) -> None:
        if not self._active:
            return
        self._frame_id += 1
        self._rtsp.push_frame(frame)

        selected_obj = detector.get_selected_object()
        if selected_obj is not None:
            self._last_selected_class = selected_obj.class_name
        tracker_state = _get_tracker_state(following=(movement is not None))
        if tracker_state != self._last_state_sent:
            print(
                f"[SEND] tracker_state: '{self._last_state_sent}' -> '{tracker_state}' "
                f"(class={self._last_selected_class})"
            )
            self._last_state_sent = tracker_state
        tracking_conf = detector.get_tracking_confidence() if selected_obj else 0.0
        if tracker_state == "lost":
            selected_class = self._last_selected_class
        else:
            selected_class = selected_obj.class_name if selected_obj else None

        h, w = frame.shape[:2]
        shared = [
            _convert_detection(
                d,
                is_selected=(selected_obj is not None and d is selected_obj),
                tracker_state=tracker_state,
            )
            for d in detections
        ]
        apply_authoritative_target_distance(shared, detections, selected_obj, movement)

        overlay = _build_overlay(
            tracker_state,
            len(shared),
            self._mode,
            True,
            movement,
            selected_class,
            tracking_conf,
        )

        tracking_data = None
        if selected_obj and movement:
            distance, distance_valid, distance_source, distance_confidence = _detection_distance(selected_obj)
            if distance_source is None:
                distance = movement.get("distance_m")
                distance_valid = bool(movement.get("distance_valid"))
                distance_source = movement.get("distance_source") if distance_valid else "none"
                distance_confidence = movement.get("distance_confidence", 0.0) if distance_valid else 0.0
                try:
                    distance = float(distance) if distance is not None else None
                    distance_valid = distance_valid and distance is not None and math.isfinite(distance) and distance > 0.0
                except (TypeError, ValueError):
                    distance = None
                    distance_valid = False
            tracking_data = TrackingData(
                target_class=selected_class,
                distance=distance if distance_valid else None,
                distance_valid=distance_valid,
                distance_source=distance_source,
                distance_confidence=distance_confidence if distance_valid else 0.0,
                speed=movement.get("vel_z", 0.0),
                yaw_rate=movement.get("yaw_cmd", 0.0),
                confidence=tracking_conf,
            )

        intr = self._scaled_intrinsics(w, h)
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
            mode=self._mode,
            hud_visible=self._hud_visible,
            telemetry=telemetry,
            tracking_data=tracking_data,
            intrinsics=intr,
        )

        self._transport.send(fd)

    def stop(self) -> None:
        self._active = False
        self._rtsp.stop()
        self._transport.close()
        logger.info("Streaming stopped")
