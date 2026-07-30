from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrackingState(Enum):
    IDLE = "idle"
    ACQUIRED = "acquired"
    TRACKING = "tracking"
    LOST = "lost"


_IDLE_COLOR = "#3b82f6"
_TRACKING_COLOR = "#22c55e"
_LOST_COLOR = "#ef4444"
_SELECTED_COLOR = "#facc15"

_DEFAULT_BG = "rgba(0,0,0,0.6)"
_DEFAULT_FONT = "bold 11px monospace"
_DEFAULT_TEXT_COLOR = "#ffffff"
_DEFAULT_FPS_COLOR = "#4ade80"


def _bbox_color_for_state(tracker_state: str) -> str:
    if tracker_state == "tracking":
        return _TRACKING_COLOR
    if tracker_state == "lost":
        return _LOST_COLOR
    return _IDLE_COLOR


@dataclass(frozen=False, slots=True)
class OverlayConfig:
    bbox_color: str = _IDLE_COLOR
    bbox_line_width: int = 2
    label_format: str = "class_only"
    show_center_dot: bool = True
    center_dot_color: str = "#ffffff"
    center_dot_radius: int = 3
    counter_text: Optional[str] = None
    counter_position: str = "top_left"
    counter_bg: str = _DEFAULT_BG
    counter_color: str = _DEFAULT_TEXT_COLOR
    counter_font: str = _DEFAULT_FONT
    prompt_text: Optional[str] = None
    prompt_position: str = "bottom_center"
    prompt_bg: str = _DEFAULT_BG
    prompt_color: str = _DEFAULT_TEXT_COLOR
    prompt_font: str = _DEFAULT_FONT
    show_fps: bool = True
    fps_position: str = "bottom_left"
    fps_color: str = _DEFAULT_FPS_COLOR
    fps_bg: str = _DEFAULT_BG
    fps_font: str = _DEFAULT_FONT
    mode_text: str = "TEST"
    mode_position: str = "top_center"
    mode_bg: str = "rgba(60,60,60,0.8)"
    mode_color: str = "#c8c8c8"
    mode_font: str = "bold 12px monospace"
    show_stream_indicator: bool = True
    stream_indicator_color: str = "#00c800"
    show_tracking_bar: bool = False
    tracking_bar_text: Optional[str] = None
    tracking_bar_position: str = "top_below_status"
    tracking_bar_bg: str = "rgba(30,80,30,0.8)"
    tracking_bar_color: str = "#b4ffb4"
    tracking_bar_font: str = "bold 10px monospace"
    show_lost_banner: bool = False
    lost_banner_text: str = "TARGET LOST"
    lost_banner_color: str = "#ffffff"
    lost_banner_bg: str = "rgba(180,0,0,0.4)"
    lost_banner_font: str = "bold 18px monospace"
    show_shortcut_bar: bool = True
    shortcut_bar_text: str = "[ESC] deselect  [SPACE] follow  [R] reset  [H] hud  [Q] quit"
    shortcut_bar_position: str = "bottom_center"
    shortcut_bar_color: str = "#828282"
    shortcut_bar_font: str = "9px monospace"

    def to_dict(self) -> dict:
        return {
            "bbox_color": self.bbox_color,
            "bbox_line_width": self.bbox_line_width,
            "label_format": self.label_format,
            "show_center_dot": self.show_center_dot,
            "center_dot_color": self.center_dot_color,
            "center_dot_radius": self.center_dot_radius,
            "counter_text": self.counter_text,
            "counter_position": self.counter_position,
            "counter_bg": self.counter_bg,
            "counter_color": self.counter_color,
            "counter_font": self.counter_font,
            "prompt_text": self.prompt_text,
            "prompt_position": self.prompt_position,
            "prompt_bg": self.prompt_bg,
            "prompt_color": self.prompt_color,
            "prompt_font": self.prompt_font,
            "show_fps": self.show_fps,
            "fps_position": self.fps_position,
            "fps_color": self.fps_color,
            "fps_bg": self.fps_bg,
            "fps_font": self.fps_font,
            "mode_text": self.mode_text,
            "mode_position": self.mode_position,
            "mode_bg": self.mode_bg,
            "mode_color": self.mode_color,
            "mode_font": self.mode_font,
            "show_stream_indicator": self.show_stream_indicator,
            "stream_indicator_color": self.stream_indicator_color,
            "show_tracking_bar": self.show_tracking_bar,
            "tracking_bar_text": self.tracking_bar_text,
            "tracking_bar_position": self.tracking_bar_position,
            "tracking_bar_bg": self.tracking_bar_bg,
            "tracking_bar_color": self.tracking_bar_color,
            "tracking_bar_font": self.tracking_bar_font,
            "show_lost_banner": self.show_lost_banner,
            "lost_banner_text": self.lost_banner_text,
            "lost_banner_color": self.lost_banner_color,
            "lost_banner_bg": self.lost_banner_bg,
            "lost_banner_font": self.lost_banner_font,
            "show_shortcut_bar": self.show_shortcut_bar,
            "shortcut_bar_text": self.shortcut_bar_text,
            "shortcut_bar_position": self.shortcut_bar_position,
            "shortcut_bar_color": self.shortcut_bar_color,
            "shortcut_bar_font": self.shortcut_bar_font,
        }


@dataclass(frozen=False, slots=True)
class BBox:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains_point(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict) -> BBox:
        return cls(x=d["x"], y=d["y"], width=d["width"], height=d["height"])


@dataclass(frozen=False, slots=True)
class Point:
    x: int = 0
    y: int = 0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> Point:
        return cls(x=d["x"], y=d["y"])


@dataclass(frozen=False, slots=True)
class Detection:
    class_name: str = ""
    confidence: float = 0.0
    bbox: BBox = field(default_factory=BBox)
    center: Point = field(default_factory=Point)
    timestamp: float = field(default_factory=time.time)
    detection_id: int = field(default_factory=lambda: uuid.uuid4().int >> 96)
    distance: Optional[float] = None
    selected: bool = False
    is_selected: bool = False
    tracking_state: TrackingState = TrackingState.IDLE
    bbox_color: str = _IDLE_COLOR

    def to_dict(self) -> dict:
        d: dict = {
            "detection_id": self.detection_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
            "center": self.center.to_dict(),
            "timestamp": self.timestamp,
            "is_selected": self.is_selected,
            "bbox_color": self.bbox_color,
        }
        if self.distance is not None:
            d["distance"] = round(self.distance, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Detection:
        return cls(
            detection_id=d.get("detection_id", 0),
            class_name=d.get("class_name", ""),
            confidence=d.get("confidence", 0.0),
            bbox=BBox.from_dict(d.get("bbox", {})),
            center=Point.from_dict(d.get("center", {})),
            timestamp=d.get("timestamp", 0.0),
            distance=d.get("distance"),
            is_selected=d.get("is_selected", False),
        )


@dataclass(frozen=False, slots=True)
class TelemetryData:
    altitude: float = 0.0
    battery: int = 100
    lat: float = 0.0
    lon: float = 0.0
    ekf_ok: bool = True

    def to_dict(self) -> dict:
        return {
            "altitude": round(self.altitude, 2),
            "battery": self.battery,
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "ekf_ok": self.ekf_ok,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TelemetryData:
        return cls(
            altitude=d.get("altitude", 0.0),
            battery=d.get("battery", 100),
            lat=d.get("lat", 0.0),
            lon=d.get("lon", 0.0),
            ekf_ok=d.get("ekf_ok", True),
        )


@dataclass(frozen=False, slots=True)
class TrackingData:
    target_class: Optional[str] = None
    distance: float = 0.0
    speed: float = 0.0
    yaw_rate: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "target_class": self.target_class,
            "distance": round(self.distance, 2),
            "speed": round(self.speed, 2),
            "yaw_rate": round(self.yaw_rate, 2),
            "confidence": round(self.confidence, 1),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrackingData:
        return cls(
            target_class=d.get("target_class"),
            distance=d.get("distance", 0.0),
            speed=d.get("speed", 0.0),
            yaw_rate=d.get("yaw_rate", 0.0),
            confidence=d.get("confidence", 0.0),
        )


@dataclass(frozen=False, slots=True)
class CameraIntrinsics:
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    calib_w: int = 0
    calib_h: int = 0

    @property
    def is_valid(self) -> bool:
        return self.fx > 0 and self.fy > 0

    def scaled(self, target_w: int, target_h: int) -> CameraIntrinsics:
        if self.calib_w <= 0 or self.calib_h <= 0 or target_w <= 0 or target_h <= 0:
            return self
        sx = target_w / self.calib_w
        sy = target_h / self.calib_h
        return CameraIntrinsics(
            fx=round(self.fx * sx, 1),
            fy=round(self.fy * sy, 1),
            cx=round(self.cx * sx, 1),
            cy=round(self.cy * sy, 1),
            calib_w=target_w,
            calib_h=target_h,
        )

    def to_dict(self) -> dict:
        return {
            "fx": round(self.fx, 1),
            "fy": round(self.fy, 1),
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "calib_w": self.calib_w,
            "calib_h": self.calib_h,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CameraIntrinsics:
        return cls(
            fx=d.get("fx", 0.0),
            fy=d.get("fy", 0.0),
            cx=d.get("cx", 0.0),
            cy=d.get("cy", 0.0),
            calib_w=d.get("calib_w", 0),
            calib_h=d.get("calib_h", 0),
        )


@dataclass(frozen=False, slots=True)
class FrameDetections:
    frame_id: int = 0
    detections: list[Detection] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source_id: str = "drone_0"
    fps: float = 0.0
    frame_w: int = 0
    frame_h: int = 0
    tracker_state: str = "idle"
    overlay: Optional[OverlayConfig] = None
    mode: str = "test"
    hud_visible: bool = True
    telemetry: Optional[TelemetryData] = None
    tracking_data: Optional[TrackingData] = None
    intrinsics: Optional[CameraIntrinsics] = None

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "fps": round(self.fps, 1),
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "tracker_state": self.tracker_state,
            "mode": self.mode,
            "hud_visible": self.hud_visible,
            "overlay": self.overlay.to_dict() if self.overlay else None,
            "telemetry": self.telemetry.to_dict() if self.telemetry else None,
            "tracking_data": self.tracking_data.to_dict() if self.tracking_data else None,
            "intrinsics": self.intrinsics.to_dict() if self.intrinsics and self.intrinsics.is_valid else None,
            "detections": [d.to_dict() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, d: dict) -> FrameDetections:
        overlay_d = d.get("overlay")
        telemetry_d = d.get("telemetry")
        tracking_d = d.get("tracking_data")
        intrinsics_d = d.get("intrinsics")
        return cls(
            frame_id=d.get("frame_id", 0),
            source_id=d.get("source_id", "drone_0"),
            timestamp=d.get("timestamp", 0.0),
            fps=d.get("fps", 0.0),
            frame_w=d.get("frame_w", 0),
            frame_h=d.get("frame_h", 0),
            tracker_state=d.get("tracker_state", "idle"),
            mode=d.get("mode", "test"),
            hud_visible=d.get("hud_visible", True),
            overlay=OverlayConfig(**overlay_d) if overlay_d else None,
            telemetry=TelemetryData(**telemetry_d) if telemetry_d else None,
            tracking_data=TrackingData(**tracking_d) if tracking_d else None,
            intrinsics=CameraIntrinsics(**intrinsics_d) if intrinsics_d else None,
            detections=[Detection.from_dict(det) for det in d.get("detections", [])],
        )
