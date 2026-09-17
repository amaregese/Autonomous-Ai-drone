from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

DISPLAY_WIDTH = 960
VIDEO_H = 720
HEADER_FINAL = 84
FOOTER_FINAL = 60
DISPLAY_HEIGHT = HEADER_FINAL + VIDEO_H + FOOTER_FINAL

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

HUD_TEXT = (235, 238, 245)
HUD_TEXT_DIM = (150, 158, 178)
HUD_PANEL = (20, 24, 34)
HUD_PANEL_BORDER = (48, 56, 78)
HUD_ACCENT = (0, 190, 235)

GREEN = (45, 230, 130)
CYAN = (70, 205, 255)
AMBER = (255, 190, 60)
RED = (255, 70, 70)

STATE_THEME = {
    "idle": {
        "bg": (16, 34, 52),
        "fg": (150, 215, 245),
        "accent": (70, 190, 255),
    },
    "selected": {
        "bg": (48, 33, 8),
        "fg": (255, 216, 130),
        "accent": (255, 178, 40),
    },
    "tracking": {
        "bg": (8, 42, 26),
        "fg": (140, 245, 165),
        "accent": (40, 225, 125),
    },
    "lost": {
        "bg": (46, 10, 12),
        "fg": (255, 150, 150),
        "accent": (255, 40, 40),
    },
}

MODE_COLORS = {
    "TEST": ((42, 42, 50), HUD_TEXT_DIM),
    "sitl": ((10, 50, 80), CYAN),
    "flight": ((14, 55, 24), GREEN),
}


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


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def _put_text(img, text, org, font_scale=0.45, color=(255, 255, 255), thickness=1,
              font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def _text_size(text, font_scale=0.45, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    return cv2.getTextSize(text, font, font_scale, thickness)[0]


def _wrap_text(text, font_scale, thickness, max_width, font=cv2.FONT_HERSHEY_SIMPLEX):
    words = text.split()
    if not words:
        return []
    lines = []
    cur = ""
    for word in words:
        test = (cur + " " + word).strip()
        if _text_size(test, font_scale, thickness, font)[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [text]
    return lines


def _pulse(t) -> float:
    return 0.5 + 0.5 * math.sin(time.time() * t)


def _rounded_rect_fill(img, pt1, pt2, radius, color, line_type=cv2.LINE_AA):
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if radius <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, line_type)
        return
    r = min(int(radius), (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, line_type)
        return
    pts = np.array([
        (x1 + r, y1), (x2 - r, y1), (x2, y1 + r), (x2, y2 - r),
        (x2 - r, y2), (x1 + r, y2), (x1, y2 - r), (x1, y1 + r),
    ], np.int32)
    cv2.fillPoly(img, [pts], color, line_type)
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1, line_type)
    cv2.circle(img, (x2 - r, y1 + r), r, color, -1, line_type)
    cv2.circle(img, (x1 + r, y2 - r), r, color, -1, line_type)
    cv2.circle(img, (x2 - r, y2 - r), r, color, -1, line_type)


def _rounded_rect_bordered(img, pt1, pt2, radius, fill, border, border_thick=1,
                           line_type=cv2.LINE_AA):
    x1, y1 = pt1
    x2, y2 = pt2
    _rounded_rect_fill(img, (x1, y1), (x2, y2), radius, border, line_type)
    inset = max(1, border_thick)
    _rounded_rect_fill(img, (x1 + inset, y1 + inset),
                       (x2 - inset, y2 - inset), max(0, radius - inset), fill, line_type)


def _rounded_rect_glass(img, pt1, pt2, radius, color, alpha):
    overlay = img.copy()
    _rounded_rect_fill(overlay, pt1, pt2, radius, color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _pill(img, x, y, text, bg_color, text_color, font_scale=0.4, thickness=1,
          padding_h=8, padding_v=4, radius=None, outline=None, dot=None,
          font=cv2.FONT_HERSHEY_SIMPLEX):
    tw, th = _text_size(text, font_scale, thickness, font)
    dot_pad = 14 if dot is not None else 0
    tx = x + padding_h + dot_pad
    rx1 = x
    ry1 = y
    rx2 = x + tw + padding_h * 2 + dot_pad
    ry2 = y + th + padding_v * 2
    r = (ry2 - ry1) // 2
    if outline is not None:
        _rounded_rect_bordered(img, (rx1, ry1), (rx2, ry2), r, bg_color, outline, 1)
    else:
        _rounded_rect_glass(img, (rx1, ry1), (rx2, ry2), r, bg_color, 0.8)
    if dot is not None:
        cy_dot = ry1 + (ry2 - ry1) // 2
        cv2.circle(img, (x + padding_h + 4, cy_dot), 4, dot, -1, cv2.LINE_AA)
        cv2.circle(img, (x + padding_h + 4, cy_dot), 4, HUD_TEXT, 1, cv2.LINE_AA)
    _put_text(img, text, (tx, y + th + padding_v), font_scale, text_color, thickness, font)
    return rx2


def _draw_panel(img, x, y, w, h, bg_color=None, radius=10, title=None,
                accent=None, border=None):
    if bg_color is None:
        bg_color = HUD_PANEL
    if border is None:
        border = HUD_PANEL_BORDER
    _rounded_rect_glass(img, (x, y), (x + w, y + h), radius, bg_color, 0.62)
    if border is not None:
        _rounded_rect_fill(img, (x, y), (x + w, y + h), radius, border, cv2.LINE_AA)
        _rounded_rect_glass(img, (x + 1, y + 1), (x + w - 1, y + h - 1),
                            max(0, radius - 1), bg_color, 0.9)
    if accent is not None:
        cv2.rectangle(img, (x + 6, y + 6), (x + 6, y + h - 6), accent, 2, cv2.LINE_AA)
    if title is not None:
        tw, th = _text_size(title, 0.3, 1)
        cv2.rectangle(img, (x + 5, y + 1), (x + 5 + tw + 10, y + 12), bg_color, -1, cv2.LINE_AA)
        _put_text(img, title, (x + 10, y + th + 1), 0.3, HUD_TEXT_DIM, 1)


def _draw_corner_brackets(img, x1, y1, x2, y2, color, length=12, thickness=2):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


def _draw_scan_ring(img, cx, cy, radius, color, thickness=1, speed=2.0, offset=0.0,
                    segments=36, arc_frac=0.75):
    now = time.time() * speed + offset
    for i in range(segments):
        ang = now + i * (math.pi * 2) / segments
        if i >= segments * arc_frac:
            break
        x = int(cx + radius * math.cos(ang))
        y = int(cy + radius * math.sin(ang))
        cv2.circle(img, (x, y), 1, color, -1, cv2.LINE_AA)


def _gradient_h(img, y0, y1, top_color, bottom_color):
    h, w = img.shape[:2]
    rows = max(1, y1 - y0)
    grad = np.zeros((rows, w, 3), np.uint8)
    top = np.array(top_color, np.float32)
    bottom = np.array(bottom_color, np.float32)
    for i in range(rows):
        t = i / (rows - 1)
        grad[i, :] = np.round(top + (bottom - top) * t)
    overlay = img.copy()
    overlay[y0:y1, :] = grad
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)


# ---------------------------------------------------------------------------
# Background chrome
# ---------------------------------------------------------------------------
def draw_hud_background(img):
    h, w = img.shape[:2]
    accent = (50, 90, 140)
    _draw_corner_brackets(img, 3, 3, w - 4, h - 4, accent, length=22, thickness=1)


# ---------------------------------------------------------------------------
# Window composition (header band + video band + footer band)
# ---------------------------------------------------------------------------
_video_geom = None  # (sx, sy, cw, ch, vw, vh, fw, fh) cover-scaled size + crop offsets + band geometry


def set_video_geom(frame_w, frame_h):
    global _video_geom
    if frame_w <= 0 or frame_h <= 0:
        _video_geom = None
        return
    scale = max(DISPLAY_WIDTH / frame_w, VIDEO_H / frame_h)
    rw = int(frame_w * scale)
    rh = int(frame_h * scale)
    sx = max(0, (rw - DISPLAY_WIDTH) // 2)
    sy = max(0, (rh - VIDEO_H) // 2)
    _video_geom = (sx, sy, rw, rh, DISPLAY_WIDTH, VIDEO_H, frame_w, frame_h)


def get_video_rect():
    if _video_geom is None:
        return (0, HEADER_FINAL, DISPLAY_WIDTH, HEADER_FINAL + VIDEO_H)
    _, _, _, _, _, _, _, _ = _video_geom
    return (0, HEADER_FINAL, DISPLAY_WIDTH, HEADER_FINAL + VIDEO_H)


def display_to_frame(dx, dy):
    if _video_geom is None:
        return None
    sx, sy, cw, ch, vw, vh, fw, fh = _video_geom
    if not (0 <= dx < vw and HEADER_FINAL <= dy < HEADER_FINAL + vh):
        return None
    frame_x = max(0.0, min(1.0, (dx + sx) / cw))
    frame_y = max(0.0, min(1.0, (dy - HEADER_FINAL + sy) / ch))
    return int(frame_x * fw), int(frame_y * fh)


def _fill_band_gradients(canvas):
    _gradient_h(canvas, 0, HEADER_FINAL, (14, 20, 32), (24, 28, 44))
    _gradient_h(canvas, DISPLAY_HEIGHT - FOOTER_FINAL, DISPLAY_HEIGHT,
                (24, 28, 44), (14, 20, 32))
    _draw_band_separators(canvas)


def _draw_band_separators(canvas):
    cv2.line(canvas, (0, HEADER_FINAL), (DISPLAY_WIDTH, HEADER_FINAL),
             (58, 72, 104), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, HEADER_FINAL + 1), (DISPLAY_WIDTH, HEADER_FINAL + 1),
             (10, 14, 22), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, DISPLAY_HEIGHT - FOOTER_FINAL),
             (DISPLAY_WIDTH, DISPLAY_HEIGHT - FOOTER_FINAL), (58, 72, 104), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, DISPLAY_HEIGHT - FOOTER_FINAL - 1),
             (DISPLAY_WIDTH, DISPLAY_HEIGHT - FOOTER_FINAL - 1), (10, 14, 22), 1, cv2.LINE_AA)


def compose_window(native, tracker_state, target_class, tracking_conf, fps,
                   infer_ms=0.0, armed=False):
    if native is None or native.size == 0:
        canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
        canvas[:] = (10, 12, 18)
        return canvas
    fh, fw = native.shape[:2]
    if not native.flags['C_CONTIGUOUS']:
        native = np.ascontiguousarray(native)

    set_video_geom(fw, fh)
    sx, sy, cw, ch, vw, vh, _, _ = _video_geom

    canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
    canvas[:] = (10, 12, 18)
    _fill_band_gradients(canvas)

    video = cv2.resize(native, (cw, ch), interpolation=cv2.INTER_LINEAR)
    if cw > vw or ch > vh:
        video = video[sy:sy + vh, sx:sx + vw]
    try:
        canvas[HEADER_FINAL:HEADER_FINAL + vh, 0:vw] = video
    except ValueError:
        pass
    _draw_band_separators(canvas)

    if hud.hud_visible:
        draw_status_bar(canvas, tracker_state, target_class, tracking_conf)
        draw_shortcut_bar(canvas)
    draw_takeoff_button(canvas, armed)
    return canvas


# ---------------------------------------------------------------------------
# Status header
# ---------------------------------------------------------------------------
def draw_status_bar(img, tracker_state, target_class, tracking_conf):
    h, w = img.shape[:2]
    theme = STATE_THEME.get(tracker_state, STATE_THEME["idle"])

    y = (HEADER_FINAL - 38) // 2
    mode_text = f"  {hud.mode.upper()}  "
    mc, mt = MODE_COLORS.get(hud.mode, MODE_COLORS["TEST"])

    mode_pill_w = _text_size(mode_text, 0.55, 1)[0] + 9 * 2
    mode_start = max(mode_pill_w // 2, (w - mode_pill_w) // 2)
    max_x = min(w - 170, mode_start - 40)

    pills = []
    pills.append((tracker_state.upper(), theme["bg"], theme["fg"], 0.58, 12, theme["accent"], theme["accent"]))
    if target_class:
        conf_text = f"{target_class}  {tracking_conf:.0f}%" if tracking_conf > 0 else target_class
        pills.append((conf_text, (28, 34, 48), HUD_TEXT, 0.55, 11, (64, 76, 106), None))
    if hud.streaming:
        pills.append(("LIVE", (10, 54, 32), (120, 255, 160), 0.44, 7, (30, 160, 80), GREEN))
    else:
        pills.append(("OFFLINE", (44, 16, 16), (255, 150, 150), 0.44, 7, (200, 40, 40), RED))

    gap = 10
    x = 10
    for _text, bg, fg, fs, padh, outline, dot in pills:
        dot_pad = 14 if dot is not None else 0
        if x > 10 and x + _text_size(_text, fs, 1)[0] + padh * 2 + dot_pad > max_x:
            break
        _pill(img, x, y, _text, bg, fg, fs, 1,
              padding_h=padh, padding_v=7, outline=outline, dot=dot)
        x += _text_size(_text, fs, 1)[0] + padh * 2 + dot_pad + gap

    _pill(img, mode_start, y, mode_text, mc, mt, 0.55, 1,
          padding_h=9, padding_v=7, outline=(70, 80, 120), dot=None)


# ---------------------------------------------------------------------------
# FPS / telemetry
# ---------------------------------------------------------------------------
def draw_fps(img, fps, infer_ms=0.0):
    h, w = img.shape[:2]
    y = 4
    x = 10
    x = _pill(img, x, y, "FPS", (30, 38, 50), (150, 165, 195), 0.32, 1,
              padding_h=6, padding_v=4, radius=10, outline=(60, 78, 108), dot=GREEN if fps >= 15 else AMBER)
    x += 4
    val = f"{fps:.0f}"
    x = _pill(img, x, y, val, (10, 44, 30), (130, 255, 170), 0.36, 1,
              padding_h=8, padding_v=4, radius=10, outline=(30, 150, 80))
    if infer_ms > 0:
        x += 4
        _pill(img, x, y, f"{infer_ms:.0f}ms", (30, 38, 50), (HUD_TEXT_DIM), 0.32, 1,
              padding_h=6, padding_v=4, radius=10, outline=(60, 78, 108))


def draw_telemetry(img, altitude, battery, lat, lon, ekf_ok):
    h, w = img.shape[:2]
    panel_w = 118
    panel_h = 88
    px = w - panel_w - 6
    py = 6
    _draw_panel(img, px, py, panel_w, panel_h, radius=10, title="TELEMETRY", accent=(60, 120, 220))

    y = py + 20
    _put_text(img, f"{altitude:.1f}m", (px + 24, y), 0.3, HUD_TEXT)
    _put_text(img, "ALT", (px + 8, y), 0.22, HUD_TEXT_DIM)
    y += 17

    if battery > 50:
        bat_color = GREEN
    elif battery > 20:
        bat_color = AMBER
    else:
        bat_color = RED
    _put_text(img, f"{battery}%", (px + 24, y), 0.3, bat_color)
    _put_text(img, "BAT", (px + 8, y), 0.22, HUD_TEXT_DIM)
    _bar = max(6, int((panel_w - 52) * battery / 100))
    cv2.rectangle(img, (px + 42, y - 9), (px + 42 + _bar, y - 6), bat_color, 2, cv2.LINE_AA)
    y += 18

    _put_text(img, f"{lat:.4f}", (px + 8, y), 0.22, (160, 172, 200))
    y += 11
    _put_text(img, f"{lon:.4f}", (px + 8, y), 0.22, (140, 152, 185))
    y += 13

    ekf_color = GREEN if ekf_ok else RED
    cv2.circle(img, (px + 10, y - 4), 3, ekf_color, -1, cv2.LINE_AA)
    cv2.circle(img, (px + 10, y - 4), 3, HUD_TEXT, 1, cv2.LINE_AA)
    _put_text(img, "EKF " + ("OK" if ekf_ok else "BAD"), (px + 18, y), 0.26, ekf_color)


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------
def draw_hud_notification(img):
    remaining = hud.notification_until - time.time()
    if not hud.notification or remaining <= 0:
        hud.notification = ""
        return
    fade = min(1.0, remaining / 0.5)
    color = tuple(int(c * fade) for c in hud.notification_color)
    if all(c < 40 for c in color):
        color = (255, 255, 255)
    accent = color if any(c > 60 for c in color) else (200, 200, 200)

    h, w = img.shape[:2]
    max_w = w - 24

    font_scale = 0.46
    thickness = 1
    lines = _wrap_text(hud.notification, font_scale, thickness, max_w - 44)
    if len(lines) > 2:
        font_scale = 0.40
        lines = _wrap_text(hud.notification, font_scale, thickness, max_w - 44)

    cap_h = _text_size("Mg", font_scale, thickness)[1]
    line_h = cap_h + 5
    pad_x, pad_y = 14, 8
    tw_line = max(_text_size(ln, font_scale, thickness)[0] for ln in lines)
    bw = min(tw_line + pad_x * 2 + 8, max_w)
    bh = line_h * len(lines) + pad_y * 2
    bx = max(4, w // 2 - bw // 2)
    by = 8

    overlay = img.copy()
    _rounded_rect_fill(overlay, (bx, by), (bx + bw, by + bh), 8, (12, 14, 20))
    cv2.addWeighted(overlay, 0.84 * fade, img, 1 - 0.84 * fade, 0, img)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (54, 60, 84), 1, cv2.LINE_AA)
    cv2.rectangle(img, (bx + 3, by + 4), (bx + 5, by + bh - 4), accent, -1, cv2.LINE_AA)

    shadow = (0, 0, 0)
    y_cursor = by + cap_h + pad_y
    for ln in lines:
        lw = _text_size(ln, font_scale, thickness)[0]
        tx = bx + max(pad_x + 5, (bw - lw) // 2)
        _put_text(img, ln, (tx + 1, y_cursor + 1), font_scale, shadow, thickness)
        _put_text(img, ln, (tx, y_cursor), font_scale, color, thickness)
        y_cursor += line_h


# ---------------------------------------------------------------------------
# Takeoff button
# ---------------------------------------------------------------------------
_lost_dismiss_rect = None
_takeoff_button_rect = None
_TAKEOFF_BUTTON_W = 150
_TAKEOFF_BUTTON_H = 30


def draw_takeoff_button(img, armed: bool = False):
    global _takeoff_button_rect
    h, w = img.shape[:2]
    bx = w - _TAKEOFF_BUTTON_W - 8
    by = (HEADER_FINAL - 38) // 2
    _takeoff_button_rect = (bx, by, bx + _TAKEOFF_BUTTON_W, by + _TAKEOFF_BUTTON_H)

    if armed:
        fill, border, dot_c = (12, 52, 30), (60, 200, 120), GREEN
        text = "ARMED"
    else:
        fill, border, dot_c = (10, 40, 68), (70, 180, 255), CYAN
        text = "TAKEOFF 5m"

    _rounded_rect_bordered(img, (bx, by), (bx + _TAKEOFF_BUTTON_W, by + _TAKEOFF_BUTTON_H),
                           15, fill, border, 1)
    cv2.circle(img, (bx + 16, by + _TAKEOFF_BUTTON_H // 2), 3, dot_c, -1, cv2.LINE_AA)
    cv2.circle(img, (bx + 16, by + _TAKEOFF_BUTTON_H // 2), 3, HUD_TEXT, 1, cv2.LINE_AA)

    tw, th = _text_size(text, 0.42, 1)
    step = 26
    cx = bx + step + ((_TAKEOFF_BUTTON_W - 2 * step) - tw) // 2
    cy = by + (_TAKEOFF_BUTTON_H + th) // 2
    _put_text(img, text, (cx, cy), 0.42, HUD_TEXT, 1)
    return img


def get_takeoff_button_rect():
    return _takeoff_button_rect


# ---------------------------------------------------------------------------
# Lost banner
# ---------------------------------------------------------------------------
def draw_lost_banner(img, rtl_countdown: float = -1):
    global _lost_dismiss_rect
    h, w = img.shape[:2]
    bar_h = 2
    banner_h = 40

    alpha = 0.42 + 0.18 * abs(((time.time() * 4) % 2) - 1)
    overlay = img.copy()
    _rounded_rect_fill(overlay, (2, bar_h + 2), (w - 2, bar_h + banner_h + 2), 6, (36, 4, 10))
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    stripes = img.copy()
    t = time.time() * 3
    off = int((t * 22) % 36)
    for x in range(-36 + off, w + 36, 36):
        cv2.line(stripes, (x, bar_h + 2), (x + 20, bar_h + banner_h + 2), (90, 8, 10), 2, cv2.LINE_AA)
    cv2.addWeighted(stripes, 0.35, img, 0.65, 0, img)
    cv2.rectangle(img, (2, bar_h + 2), (w - 2, bar_h + banner_h + 2), RED, 1, cv2.LINE_AA)
    cv2.rectangle(img, (2, bar_h + banner_h - 3), (w - 2, bar_h + banner_h + 2), (60, 8, 10), -1, cv2.LINE_AA)

    text = "TARGET LOST"
    tw, th = _text_size(text, 0.58, 2)
    tx = w // 2 - tw // 2
    if rtl_countdown > 0:
        cd = f"RTL {rtl_countdown:.0f}s"
        tx -= _text_size(cd, 0.42, 1)[0] // 2 + 4
    _put_text(img, text, (tx, bar_h + th + 6), 0.58, (255, 235, 235), 2)
    _put_text(img, text, (tx, bar_h + th + 6), 0.58, (255, 120, 120), 1)

    cx = w // 2 + tw // 2 + 10
    if rtl_countdown > 0:
        _pill(img, cx, bar_h + 8, cd, (70, 8, 8), (255, 190, 190), 0.42, 1,
              padding_h=10, padding_v=6, radius=12, outline=RED)
    elif rtl_countdown == 0:
        _pill(img, cx, bar_h + 8, "RTL ACTIVE", (70, 8, 8), (255, 190, 190), 0.42, 1,
              padding_h=10, padding_v=6, radius=12, outline=RED)

    btn_text = "DISMISS"
    btw, _ = _text_size(btn_text, 0.4, 1)
    bw = btw + 22
    bx = w - bw - 12
    by = bar_h + 8
    _lost_dismiss_rect = (bx, by, bx + bw, by + 24)
    _rounded_rect_bordered(img, (bx, by), (bx + bw, by + 24), 12, (52, 52, 62), (130, 130, 150), 1)
    _put_text(img, btn_text, (bx + 11, by + 15), 0.4, (220, 220, 230), 1)


def get_lost_dismiss_rect():
    return _lost_dismiss_rect


# ---------------------------------------------------------------------------
# Footer shortcuts
# ---------------------------------------------------------------------------
def draw_shortcut_bar(img):
    h, w = img.shape[:2]
    shortcuts = [
        ("ESC", "deselect"),
        ("SPACE", "follow"),
        ("R", "reset"),
        ("H", "hud"),
        ("Q", "quit"),
    ]
    chip_w = 0
    for k, v in shortcuts:
        kw, kh = _text_size(k, 0.5, 1)
        vw, _ = _text_size(v, 0.4, 1)
        chip_w += kw + vw + 24 + 8
    cx = w // 2 - chip_w // 2
    ch = 24
    y = h - FOOTER_FINAL + (FOOTER_FINAL - ch) // 2
    for k, v in shortcuts:
        kw, kh = _text_size(k, 0.5, 1)
        vw, vh = _text_size(v, 0.4, 1)
        cw = kw + vw + 24
        _rounded_rect_fill(img, (cx, y), (cx + cw, y + ch), ch // 2, (30, 36, 50))
        _rounded_rect_fill(img, (cx + 1, y + 1), (cx + 1 + kw + 12, y + ch - 1), (ch - 2) // 2, (52, 62, 88))
        _put_text(img, k, (cx + 7, y + kh + 4), 0.5, (220, 235, 255), 1)
        _put_text(img, v, (cx + kw + 16, y + vh + 4), 0.4, HUD_TEXT_DIM, 1)
        cx += cw + 8


# ---------------------------------------------------------------------------
# Detection overlay
# ---------------------------------------------------------------------------
def _draw_label(img, text, x, y, bg_color, text_color, font_scale=0.42, thickness=1,
                border=None, bar=None, bar_w=None):
    tw, th = _text_size(text, font_scale, thickness)
    pad = 4
    bx1 = x
    by1 = y - th - pad * 2
    by2 = y + pad
    bw = tw + pad * 2 + (bar_w if bar is not None else 0)
    hbox = by2 - by1
    radius = hbox // 2
    if border is not None:
        _rounded_rect_bordered(img, (bx1, by1), (bx1 + bw, by2), radius, bg_color, border, 1)
    else:
        _rounded_rect_glass(img, (bx1, by1), (bx1 + bw, by2), radius, bg_color, 0.8)
    _put_text(img, text, (x + pad, y - pad), font_scale, text_color, thickness)
    if bar is not None:
        bw_bar = bar_w if bar_w is not None else max(14, int(tw * bar))
        by_bar = by2 - 3
        cv2.rectangle(img, (bx1 + pad, by_bar), (bx1 + pad + max(14, bw - pad * 2), by_bar + 2),
                      (60, 66, 82), 1, cv2.LINE_AA)
        cv2.rectangle(img, (bx1 + pad, by_bar), (bx1 + pad + bw_bar, by_bar + 2), (40, 200, 120), -1, cv2.LINE_AA)


def _draw_reticle(img, cx, cy, accent, radius=26):
    gap = 5
    size = 11
    color = accent
    cv2.line(img, (cx - size, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx + gap, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + gap), (cx, cy + size), color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, color, 1, cv2.LINE_AA)


def _label_anchor(y1, y2, th, img_h):
    above = y1 - 4
    if above - th - 8 >= 3:
        return above
    below = min(y2 + 14, img_h - 14)
    if below - th - 8 >= 3:
        return below
    return max(th + 12, y1 + 8)


def draw_detection_window(image, detections, detector):
    selected_obj = detector.get_selected_object()
    h, w = image.shape[:2]

    for obj in detections:
        is_sel = obj is selected_obj
        x1, y1, x2, y2 = obj.Left, obj.Top, obj.Right, obj.Bottom

        if is_sel:
            accent = STATE_THEME["selected"]["accent"]
            glow = 1.0 + 0.5 * _pulse(5.0)

            box_color = tuple(min(255, int(c * glow)) for c in accent)
            _draw_corner_brackets(image, x1, y1, x2, y2, box_color, length=18, thickness=3)
            _draw_corner_brackets(image, x1 + 3, y1 + 3, x2 - 3, y2 - 3, box_color,
                                  length=10, thickness=1)

            label = f"{obj.class_name}  {obj.confidence:.0f}%"
            lth = _text_size(label, 0.45, 1)[1]
            ly = _label_anchor(y1, y2, lth, h)
            _draw_label(image, label, x1, ly, (38, 26, 8), (255, 224, 150), 0.45, 1,
                        border=accent, bar=obj.confidence, bar_w=max(14, int(_text_size(label, 0.45, 1)[0] * 0.6)))

            cx_d = (x1 + x2) // 2
            cy_d = (y1 + y2) // 2
            _draw_reticle(image, cx_d, cy_d, accent, radius=12)
        else:
            edge = (140, 110, 70)
            dim = 0.30
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 60, 30), -1)
            cv2.addWeighted(overlay, dim, image, 1 - dim, 0, image)
            cv2.rectangle(image, (x1, y1), (x2, y2), edge, 1, cv2.LINE_AA)
            _draw_corner_brackets(image, x1, y1, x2, y2, edge, length=8, thickness=1)
            label = obj.class_name
            lth = _text_size(label, 0.38, 1)[1]
            ly = _label_anchor(y1, y2, lth, h)
            _draw_label(image, label, x1, ly, (44, 38, 22), (210, 200, 180), 0.38, 1)

    cx, cy = w // 2, h // 2
    accent = (70, 130, 175)
    _draw_reticle(image, cx, cy, accent, radius=22)

    return image


# ---------------------------------------------------------------------------
# Tracking / selection prompts
# ---------------------------------------------------------------------------
def draw_selection_prompt(img, detections):
    h, w = img.shape[:2]
    prompt = f"CLICK - SELECT TARGET     {len(detections)} OBJECTS"
    tw, th = _text_size(prompt, 0.46, 1)
    total = tw + 24 + 14
    x = max(4, w // 2 - total // 2)
    y = h - 64
    _pill(img, x, y, prompt, (12, 36, 52), CYAN, 0.46, 1, padding_h=12, padding_v=7,
          radius=14, outline=(40, 120, 170), dot=CYAN)


def draw_follow_prompt(img, class_name):
    h, w = img.shape[:2]
    prompt = f"SELECTED - {class_name}    [SPACE] FOLLOW"
    tw, th = _text_size(prompt, 0.46, 1)
    total = tw + 24 + 14
    x = max(4, w // 2 - total // 2)
    y = h - 64
    _pill(img, x, y, prompt, (48, 33, 8), (255, 218, 130), 0.46, 1, padding_h=12, padding_v=7,
          radius=14, outline=AMBER, dot=AMBER)


def draw_target_tracking(img, selected_obj, movement, fps):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    tx, ty = selected_obj.Center
    dist = movement.get("lidar_dist", 0.0) or movement.get("vision_dist", 0.0)
    speed = movement.get("vel_z", 0.0)
    yaw = movement.get("yaw_cmd", 0.0)
    conf = detector_get_confidence()

    accent = STATE_THEME["tracking"]["accent"]

    orig = img.copy()
    steps = 22
    for i in range(steps):
        t = i / steps
        x = int(cx + (tx - cx) * t)
        y = int(cy + (ty - cy) * t)
        thk = max(1, int(3 - t * 2))
        col = (int(accent[0] * (1 - t) + 60 * t),
               int(accent[1] * (1 - t) + 200 * t),
               int(accent[2] * (1 - t) + 120 * t))
        if i % 2 == 0:
            cv2.circle(img, (x, y), thk, col, -1, cv2.LINE_AA)

    r = 8 + int(4 * _pulse(4.0))
    cv2.circle(img, (tx, ty), r, accent, 2, cv2.LINE_AA)
    cv2.circle(img, (tx, ty), r + 8, accent, 1, cv2.LINE_AA)
    cv2.circle(img, (tx, ty), 3, accent, -1, cv2.LINE_AA)

    bar_text = (
        f"FOLLOW {selected_obj.class_name}   "
        f"{dist:.1f}m   {speed:+.2f}m/s   "
        f"YAW {yaw:+.1f}   {conf:.0f}%"
    )
    scale = 0.34
    max_bar_w = w - 40
    tw, _ = _text_size(bar_text, scale, 1)
    if tw > max_bar_w:
        scale = 0.30
        tw, _ = _text_size(bar_text, scale, 1)
    total = tw + 24 + 14
    x = max(4, w // 2 - total // 2)
    y = h - 64
    _pill(img, x, y, bar_text, (8, 40, 26), (160, 255, 180), scale, 1,
          padding_h=12, padding_v=7, radius=14, outline=accent, dot=GREEN)


def annotate_tracking_overlay(image, fps, selected_obj, movement):
    draw_target_tracking(image, selected_obj, movement, fps)


def annotate_selection_overlay(image, detections):
    draw_selection_prompt(image, detections)


# ---------------------------------------------------------------------------
# Detector reference
# ---------------------------------------------------------------------------
_detector_ref = None


def set_detector_ref(det):
    global _detector_ref
    _detector_ref = det


def detector_get_confidence():
    if _detector_ref is not None:
        return _detector_ref.get_tracking_confidence()
    return 0.0


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
        status_color = (0, 200, 90)

    control.set_visualizer_status(status, status_color, duration=0)