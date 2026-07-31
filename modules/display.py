from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720


@dataclass
class HUDState:
    hud_visible: bool = True
    mode: str = "TEST"
    streaming: bool = True
    altitude: float = 0.0
    battery: int = 100
    lat: float = 0.0
    lon: float = 0.0
    ekf_ok: bool = True
    lost_flash_until: float = 0.0
    notification: str = ""
    notification_color: tuple = (255, 255, 255)
    notification_until: float = 0.0


hud = HUDState()


def set_hud_status(message: str, color=(255, 255, 255), duration: float = 2.0):
    hud.notification = message
    hud.notification_color = color
    hud.notification_until = time.time() + duration


def draw_hud_notification(img):
    remaining = hud.notification_until - time.time()
    if not hud.notification or remaining <= 0:
        hud.notification = ""
        return
    fade = min(1.0, remaining / 0.5)
    color = tuple(int(c * fade) for c in hud.notification_color)
    h, w = img.shape[:2]
    font_scale = 0.5
    tw, th = _text_size(hud.notification, font_scale, 2)
    x = w // 2 - tw // 2
    y = 50
    pad = 8
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6 * fade, img, 1 - 0.6 * fade, 0, img)
    _put_text(img, hud.notification, (x, y), font_scale, color, 2)


def _put_text(img, text, org, font_scale=0.45, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _text_size(text, font_scale=0.45, thickness=1):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]


def _draw_pill(img, x, y, text, bg_color, text_color, font_scale=0.4, thickness=1, padding_h=8, padding_v=4):
    tw, th = _text_size(text, font_scale, thickness)
    rx1 = x
    ry1 = y
    rx2 = x + tw + padding_h * 2
    ry2 = y + th + padding_v * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    _put_text(img, text, (x + padding_h, y + th + padding_v), font_scale, text_color, thickness)
    return rx2


def _draw_panel(img, x, y, w, h, bg_color=(20, 20, 20)):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)


def draw_hud_background(img):
    h, w = img.shape[:2]
    bar_h = 36
    footer_h = 32
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 10), -1)
    cv2.rectangle(overlay, (0, h - footer_h), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.line(img, (0, bar_h), (w, bar_h), (40, 40, 40), 1)
    cv2.line(img, (0, h - footer_h), (w, h - footer_h), (40, 40, 40), 1)


def draw_status_bar(img, tracker_state, target_class, tracking_conf):
    h, w = img.shape[:2]

    state_colors = {
        "idle": ((70, 70, 70), "IDLE"),
        "selected": ((0, 100, 170), "SELECTED"),
        "tracking": ((0, 140, 0), "TRACKING"),
        "lost": ((0, 0, 170), "LOST"),
    }
    bg, label = state_colors.get(tracker_state, ((70, 70, 70), "UNKNOWN"))
    _draw_pill(img, 10, 6, label, bg, (255, 255, 255), 0.42, 1)

    if target_class:
        conf_text = f"{target_class} {tracking_conf:.0f}%" if tracking_conf > 0 else target_class
        _draw_pill(img, 90, 6, conf_text, (40, 40, 40), (255, 255, 255), 0.42, 1)

    mode_colors = {
        "TEST": ((50, 50, 50), (200, 200, 200)),
        "sitl": ((0, 90, 170), (255, 255, 255)),
        "flight": ((0, 130, 0), (255, 255, 255)),
    }
    mc, mt = mode_colors.get(hud.mode, ((50, 50, 50), (200, 200, 200)))
    mode_text = f" {hud.mode.upper()} "
    tw, _ = _text_size(mode_text, 0.45, 2)
    _draw_pill(img, w // 2 - tw // 2 - 8, 6, mode_text, mc, mt, 0.45, 2)

    dot_color = (0, 200, 0) if hud.streaming else (0, 0, 200)
    cv2.circle(img, (w - 12, 18), 4, dot_color, -1)
    cv2.circle(img, (w - 12, 18), 4, (255, 255, 255), 1)


def draw_fps(img, fps, infer_ms=0.0):
    h, w = img.shape[:2]
    fps_text = f"FPS {fps:.0f}"
    if infer_ms > 0:
        fps_text += f"  |  {infer_ms:.0f}ms"
    _draw_pill(img, 10, h - 32, fps_text, (30, 30, 30), (120, 220, 120), 0.38, 1)


def draw_telemetry(img, altitude, battery, lat, lon, ekf_ok):
    h, w = img.shape[:2]
    panel_w = 95
    panel_h = 82
    px = w - panel_w - 6
    py = 42
    _draw_panel(img, px, py, panel_w, panel_h)

    y = py + 13
    _put_text(img, f"{altitude:.1f}m", (px + 6, y), 0.28, (200, 200, 200))
    y += 13

    bat_color = (0, 200, 0) if battery > 50 else (0, 180, 180) if battery > 20 else (0, 0, 200)
    _put_text(img, f"{battery}%", (px + 6, y), 0.28, bat_color)
    y += 13

    _put_text(img, f"{lat:.4f}", (px + 6, y), 0.25, (160, 160, 160))
    y += 11
    _put_text(img, f"{lon:.4f}", (px + 6, y), 0.25, (140, 140, 140))
    y += 14

    ekf_color = (0, 180, 0) if ekf_ok else (0, 0, 200)
    cv2.circle(img, (px + 8, y - 3), 2, ekf_color, -1)
    _put_text(img, "EKF", (px + 14, y), 0.28, ekf_color)


_lost_dismiss_rect = None


def draw_lost_banner(img, rtl_countdown: float = -1):
    global _lost_dismiss_rect
    h, w = img.shape[:2]
    bar_h = 36

    alpha = 0.3 + 0.2 * abs(((time.time() * 4) % 2) - 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, bar_h), (w, bar_h + 36), (0, 0, 160), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.line(img, (0, bar_h), (w, bar_h), (0, 0, 255), 2)
    cv2.line(img, (0, bar_h + 36), (w, bar_h + 36), (0, 0, 255), 2)

    text = "TARGET LOST"
    tw, th = _text_size(text, 0.6, 2)
    _put_text(img, text, (w // 2 - tw // 2, bar_h + 24), 0.6, (255, 255, 255), 2)

    if rtl_countdown > 0:
        cd = f"RTL in {rtl_countdown:.0f}s"
        _draw_pill(img, w // 2 + tw // 2 + 10, bar_h + 4, cd, (160, 0, 0), (255, 200, 200), 0.4, 1)
    elif rtl_countdown == 0:
        cd = "RTL"
        _draw_pill(img, w // 2 + tw // 2 + 10, bar_h + 4, cd, (200, 0, 0), (255, 150, 150), 0.4, 1)

    btn_text = "[DISMISS]"
    btw, _ = _text_size(btn_text, 0.4, 1)
    bx = w - btw - 16
    by = bar_h + 6
    _lost_dismiss_rect = (bx, by, bx + btw + 10, by + 26)
    _draw_pill(img, bx - 4, by - 2, btn_text, (60, 60, 60), (200, 200, 200), 0.4, 1)


def get_lost_dismiss_rect():
    return _lost_dismiss_rect


def draw_shortcut_bar(img):
    h, w = img.shape[:2]
    shortcuts = [
        ("ESC", "deselect"),
        ("SPACE", "follow"),
        ("R", "reset"),
        ("H", "hud"),
        ("Q", "quit"),
    ]
    parts = [f"[{k}] {v}" for k, v in shortcuts]
    bar_text = "   ".join(parts)
    tw, _ = _text_size(bar_text, 0.35, 1)
    x = w // 2 - tw // 2
    _put_text(img, bar_text, (x, h - 10), 0.35, (120, 120, 120))


def draw_target_tracking(img, selected_obj, movement, fps):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    tx, ty = selected_obj.Center
    dist = movement.get("lidar_dist", 0.0) or movement.get("vision_dist", 0.0)
    speed = movement.get("vel_z", 0.0)
    yaw = movement.get("yaw_cmd", 0.0)
    conf = detector_get_confidence()

    cv2.line(img, (cx, cy), (tx, ty), (0, 200, 0), 1, cv2.LINE_AA)
    cv2.circle(img, (tx, ty), 8, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.circle(img, (tx, ty), 2, (0, 200, 0), -1)

    bar_text = (
        f"Following: {selected_obj.class_name}  |  "
        f"Dist: {dist:.1f}m  |  Speed: {speed:.1f}m/s  |  "
        f"Yaw: {yaw:.1f}  |  Conf: {conf:.0f}%"
    )
    tw, _ = _text_size(bar_text, 0.4, 1)
    _draw_pill(img, w // 2 - tw // 2 - 10, 40, bar_text, (20, 70, 20), (170, 255, 170), 0.4)


def draw_selection_prompt(img, detections):
    h, w = img.shape[:2]
    prompt = "Click on any object to track"
    count = f"{len(detections)} detected"
    tw1, _ = _text_size(prompt, 0.55, 1)
    tw2, _ = _text_size(count, 0.42, 1)
    _put_text(img, prompt, (w // 2 - tw1 // 2, h - 44), 0.55, (0, 200, 255), 1)
    _put_text(img, count, (w // 2 - tw2 // 2, h - 60), 0.42, (100, 220, 100))


def draw_follow_prompt(img, class_name):
    h, w = img.shape[:2]
    prompt = f"Selected: {class_name} — press SPACE to follow"
    tw, _ = _text_size(prompt, 0.5, 1)
    _put_text(img, prompt, (w // 2 - tw // 2, h - 44), 0.5, (230, 230, 0), 1)


_detector_ref = None


def set_detector_ref(det):
    global _detector_ref
    _detector_ref = det


def detector_get_confidence():
    if _detector_ref is not None:
        return _detector_ref.get_tracking_confidence()
    return 0.0


def _draw_corner_brackets(img, x1, y1, x2, y2, color, length=12, thickness=2):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


def _draw_label(img, text, x, y, bg_color, text_color, font_scale=0.42, thickness=1):
    tw, th = _text_size(text, font_scale, thickness)
    pad = 4
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - th - pad * 2), (x + tw + pad * 2, y + pad), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    _put_text(img, text, (x + pad, y - pad), font_scale, text_color, thickness)


def draw_detection_window(image, detections, detector):
    selected_obj = detector.get_selected_object()
    h, w = image.shape[:2]

    for obj in detections:
        is_sel = obj is selected_obj
        x1, y1, x2, y2 = obj.Left, obj.Top, obj.Right, obj.Bottom

        if is_sel:
            box_color = (0, 230, 255)
            _draw_corner_brackets(image, x1, y1, x2, y2, box_color, length=16, thickness=2)
            label = f"{obj.class_name} {obj.confidence:.0f}%"
            _draw_label(image, label, x1, y1 - 4, (0, 180, 220), (255, 255, 255), 0.45, 1)

            cx_d = (x1 + x2) // 2
            cy_d = (y1 + y2) // 2
            cv2.circle(image, (cx_d, cy_d), 4, box_color, 1, cv2.LINE_AA)
            cv2.line(image, (cx_d - 7, cy_d), (cx_d + 7, cy_d), box_color, 1, cv2.LINE_AA)
            cv2.line(image, (cx_d, cy_d - 7), (cx_d, cy_d + 7), box_color, 1, cv2.LINE_AA)
        elif selected_obj is None:
            box_color = (180, 120, 40)
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
            label = obj.class_name
            _draw_label(image, label, x1, y1 - 4, (80, 70, 30), (220, 220, 220), 0.38, 1)

    cx, cy = w // 2, h // 2
    cross_size = 20
    cv2.line(image, (cx - cross_size, cy), (cx - 6, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (cx + 6, cy), (cx + cross_size, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (cx, cy - cross_size), (cx, cy - 6), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (cx, cy + 6), (cx, cy + cross_size), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 3, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def annotate_tracking_overlay(image, fps, selected_obj, movement):
    draw_target_tracking(image, selected_obj, movement, fps)


def annotate_selection_overlay(image, detections):
    draw_selection_prompt(image, detections)


def update_drone_visualizer_status(detector, control, lost_threshold):
    selected_obj = detector.get_selected_object()
    selected_class = detector.get_selected_class()
    tracking_lost = detector.get_tracking_status()
    tracking_conf = detector.get_tracking_confidence()

    if tracking_lost and selected_class:
        status = f"OBJECT LOST: {selected_class}"
        status_color = (0, 0, 255)
    elif selected_obj is None:
        status = "OBJECT NOT SELECTED"
        status_color = (100, 100, 100)
    elif tracking_conf < lost_threshold:
        status = f"OBJECT LOST: {selected_obj.class_name}"
        status_color = (0, 0, 255)
    else:
        status = f"TRACKING: {selected_obj.class_name} ({tracking_conf:.0f}%)"
        status_color = (0, 255, 0)

    control.set_visualizer_status(status, status_color, duration=0)
