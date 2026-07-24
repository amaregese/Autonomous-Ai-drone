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

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "fps": round(self.fps, 1),
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "tracker_state": self.tracker_state,
            "overlay": self.overlay.to_dict() if self.overlay else None,
            "detections": [d.to_dict() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, d: dict) -> FrameDetections:
        overlay_d = d.get("overlay")
        return cls(
            frame_id=d.get("frame_id", 0),
            source_id=d.get("source_id", "drone_0"),
            timestamp=d.get("timestamp", 0.0),
            fps=d.get("fps", 0.0),
            frame_w=d.get("frame_w", 0),
            frame_h=d.get("frame_h", 0),
            tracker_state=d.get("tracker_state", "idle"),
            overlay=OverlayConfig(**overlay_d) if overlay_d else None,
            detections=[Detection.from_dict(det) for det in d.get("detections", [])],
        )
