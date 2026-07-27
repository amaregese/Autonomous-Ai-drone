from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


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


hud = HUDState()


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
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (10, 10, 10), -1)
    cv2.rectangle(overlay, (0, h - 28), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)


def draw_status_bar(img, tracker_state, target_class, tracking_conf):
    h, w = img.shape[:2]

    state_colors = {
        "idle": ((80, 80, 80), "IDLE"),
        "tracking": ((0, 160, 0), "TRACKING"),
        "lost": ((0, 0, 180), "LOST"),
    }
    bg, label = state_colors.get(tracker_state, ((80, 80, 80), "UNKNOWN"))
    _draw_pill(img, 8, 4, label, bg, (255, 255, 255))

    if target_class:
        conf_text = f"{target_class} {tracking_conf:.0f}%" if tracking_conf > 0 else target_class
        _draw_pill(img, 80, 4, conf_text, (50, 50, 50), (255, 255, 255))

    mode_colors = {
        "TEST": ((60, 60, 60), (200, 200, 200)),
        "sitl": ((0, 100, 180), (255, 255, 255)),
        "flight": ((0, 140, 0), (255, 255, 255)),
    }
    mc, mt = mode_colors.get(hud.mode, ((60, 60, 60), (200, 200, 200)))
    mode_text = f" {hud.mode.upper()} "
    tw, _ = _text_size(mode_text, 0.45, 2)
    _draw_pill(img, w // 2 - tw // 2 - 8, 4, mode_text, mc, mt, 0.45, 2)

    dot_color = (0, 200, 0) if hud.streaming else (0, 0, 200)
    cv2.circle(img, (w - 18, 18), 5, dot_color, -1)
    cv2.circle(img, (w - 18, 18), 5, (255, 255, 255), 1)


def draw_fps(img, fps, infer_ms=0.0):
    h, w = img.shape[:2]
    fps_text = f"FPS {fps:.0f}"
    if infer_ms > 0:
        fps_text += f"  {infer_ms:.0f}ms"
    tw, _ = _text_size(fps_text, 0.4, 1)
    _draw_pill(img, 8, h - 28, fps_text, (40, 40, 40), (120, 220, 120))


def draw_telemetry(img, altitude, battery, lat, lon, ekf_ok):
    h, w = img.shape[:2]
    panel_w = 160
    panel_h = 110
    px = w - panel_w - 8
    py = 40
    _draw_panel(img, px, py, panel_w, panel_h)

    y = py + 18
    _put_text(img, f"ALT  {altitude:.1f}m", (px + 8, y), 0.38, (200, 200, 200))
    y += 20

    bat_color = (0, 200, 0) if battery > 50 else (0, 180, 180) if battery > 20 else (0, 0, 200)
    _put_text(img, f"BAT  {battery}%", (px + 8, y), 0.38, bat_color)
    y += 20

    _put_text(img, f"POS  {lat:.5f}", (px + 8, y), 0.38, (200, 200, 200))
    y += 16
    _put_text(img, f"     {lon:.5f}", (px + 8, y), 0.38, (160, 160, 160))
    y += 20

    ekf_text = "EKF OK" if ekf_ok else "EKF FAIL"
    ekf_color = (0, 180, 0) if ekf_ok else (0, 0, 200)
    _put_text(img, ekf_text, (px + 8, y), 0.38, ekf_color)


def draw_lost_banner(img):
    now = time.time()
    if now > hud.lost_flash_until:
        return
    h, w = img.shape[:2]
    alpha = 0.3 + 0.2 * abs(((now * 4) % 2) - 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h // 2 - 18), (w, h // 2 + 18), (0, 0, 180), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    text = "TARGET LOST"
    tw, th = _text_size(text, 0.7, 2)
    _put_text(img, text, (w // 2 - tw // 2, h // 2 + th // 2 + 2), 0.7, (255, 255, 255), 2)


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
    tw, _ = _text_size(bar_text, 0.33, 1)
    x = w // 2 - tw // 2
    _put_text(img, bar_text, (x, h - 8), 0.33, (130, 130, 130))


def draw_target_tracking(img, selected_obj, movement, fps):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    cv2.line(img, center, selected_obj.Center, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.circle(img, selected_obj.Center, 7, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.circle(img, selected_obj.Center, 2, (0, 200, 0), -1)

    dist = movement.get("lidar_dist", 0.0) or movement.get("vision_dist", 0.0)
    speed = movement.get("vel_z", 0.0)
    yaw = movement.get("yaw_cmd", 0.0)
    conf = detector_get_confidence()

    bar_text = (
        f"Following: {selected_obj.class_name}  |  "
        f"Dist: {dist:.1f}m  |  Speed: {speed:.1f}m/s  |  "
        f"Yaw: {yaw:.1f}  |  Conf: {conf:.0f}%"
    )
    h, w = img.shape[:2]
    tw, _ = _text_size(bar_text, 0.38, 1)
    _draw_pill(img, w // 2 - tw // 2 - 8, 36, bar_text, (30, 80, 30), (180, 255, 180), 0.38)


def draw_selection_prompt(img, detections):
    h, w = img.shape[:2]
    prompt = "Click on any object to track"
    count = f"{len(detections)} detected"
    tw1, _ = _text_size(prompt, 0.5, 1)
    tw2, _ = _text_size(count, 0.4, 1)
    _put_text(img, prompt, (w // 2 - tw1 // 2, h - 40), 0.5, (0, 200, 255), 1)
    _put_text(img, count, (w // 2 - tw2 // 2, h - 55), 0.4, (100, 220, 100))


_detector_ref = None


def set_detector_ref(det):
    global _detector_ref
    _detector_ref = det


def detector_get_confidence():
    if _detector_ref is not None:
        return _detector_ref.get_tracking_confidence()
    return 0.0


def draw_detection_window(image, detections, detector):
    selected_obj = detector.get_selected_object()

    for obj in detections:
        is_sel = obj is selected_obj
        box_color = (0, 200, 255) if is_sel else (200, 120, 0)
        thickness = 2 if is_sel else 1

        cv2.rectangle(image, (obj.Left, obj.Top), (obj.Right, obj.Bottom), box_color, thickness, cv2.LINE_AA)

        if is_sel:
            cv2.putText(
                image,
                f"{obj.class_name} ({obj.confidence:.0f}%)",
                (obj.Left, obj.Top - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                obj.class_name,
                (obj.Left, obj.Top - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                box_color,
                1,
                cv2.LINE_AA,
            )

    cx, cy = image.shape[1] // 2, image.shape[0] // 2
    cv2.circle(image, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (cx - 8, cy), (cx + 8, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (cx, cy - 8), (cx, cy + 8), (255, 255, 255), 1, cv2.LINE_AA)

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
