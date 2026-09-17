# Tracker Window Overlay Spec (for SGC camera view)

Reference implementation: `modules/display.py` (OpenCV, BGR colors).
This doc defines the exact look SGC must replicate on the streamed camera view
so it matches the on-drone "Tracker" window.

> Colors are given as **BGR** (as in the source) and **CSS hex** (for web).
> All drawing uses anti-aliasing (`cv2.LINE_AA`) and font
> `cv2.FONT_HERSHEY_SIMPLEX` unless noted.

---

## 1. Where to draw (coordinate space)

The streamed video is the **raw native frame** (frame_w × frame_h, from the
`FrameDetections` payload). All bbox coordinates in the payload are already in
native frame coordinates and can be drawn 1:1 on the same-sized canvas.

**Edge-to-edge (cover) — REQUIRED, never shrink/fit with padding.** The video
must fill the entire video band on ALL sides, exactly like the on-drone
Tracker window. Do NOT letterbox, center-in-canvas, or anchor the frame
top-left of a larger canvas — any of these creates a black bar (most commonly
a thick bar on the right edge). Draw 1:1 when `frame_w/frame_h === band ratio`
(960/720); otherwise **scale-to-cover + center-crop** exactly as
`modules/display.py::set_video_geom`:

```
scale = max(band_w/frame_w, band_h/frame_h)
cw, ch = round(frame_w*scale), round(frame_h*scale)
sx = max(0, (cw - band_w)//2); sy = max(0, (ch - band_h)//2)
canvas_video = resized[sy:sy+band_h, sx:sx+band_w]  # fills every column
```

**Layout / container — MUST dynamically resize in ALL directions (no voids).**
The widget that holds/canvases the video is the top-level, size-bounded
container of the app window. It must be **fully dynamic on both axes**
(`X_AXIS | Y_AXIS` expansion in SWT; `flex: 1 1 auto` + `width:100%;
height:100%` in web), pinned to all four window edges with **zero fixed
size** on the video canvas or its parent. Window resize (any edge or corner,
in any direction) must re-layout the container so the canvas takes exactly the
remaining client area — it must **shrink** when the window shrinks (no dead
space appears on the right/bottom) and **grow** when the window grows.

The full resize contract:
1. Video canvas width = container width; video canvas height = remaining
   height after the header/footer chrome (see §5), i.e. the canvas fills
   `w_client × (h_client − chrome)`.
2. Never pin the canvas to a fixed pixel size, never `fill` it inside an
   automatically-sized parent, and never anchor it top-left of a larger,
   non-resizing parent (anchoring top-left is the cause of the empty void on
   the right and bottom when the window shrinks).
3. Repaint the whole canvas on every `Resize`/`resize` event — don't scale a
   stale buffer; redraw video + overlays at the new size.
4. The video picture inside the canvas always uses cover-crop (§1), so the
   picture is always flush with all four edges of the canvas — the canvas
   itself must never be smaller than the video picture.

**Window resize — video must track the window on EVERY resize direction.** The
video element must be stretchable on *all* sides (`left/right` and `top/bottom`
handles), exactly like the on-drone window. The video band width is always
`100%` of the container and the band height is proportional to the desk ratio;
the video fills the band cover-mode. Dragging the RIGHT edge (or top/bottom
edges) wider MUST stretch the video to remain flush with the band edge — never
leave a black gap between the chrome edge and the video edge, and never keep
the video at a fixed pixel width while only the border grows. Use
`object-fit: cover` (`width:100%; height:100%`) on an edge-to-edge canvas; do
not center or pad the canvas inside the chrome.

To match the on-drone look, reproduce the overlays from section 2–4 on the
video canvas, plus the header/footer chrome from section 5 if the SGC feed is
desk-size (960×864). If the SGC canvas is a different size, scale all
dimensions proportionally; the video band is always edge-to-edge.

---

## 2. Detection bounding boxes (`draw_detection_window`)

Draw **all** detections every frame. The selected target is styled
completely differently from the others; non-selected boxes are never removed.

### 2.1 Selected target box

- `accent = selected.accent` → BGR `(255,178,40)`, CSS `#28b2ff`
- Pulsing brightness: `glow = 1.0 + 0.5 * pulse(5.0)`
  where `pulse(t) = 0.5 + 0.5*sin(now*t)` (1 px loop ≈ 0.63 s). Clamp each
  channel to 255: `box_color = accent * glow`.
- **Outer corner brackets** exactly on the bbox rect: length `18`, thickness `3`.
- **Inner corner brackets** inset by 3 px on all sides
  (`x1+3,y1+3,x2-3,y2-3`): length `10`, thickness `1`.
- **Label** `"{class}  {conf:.0f}%"` above the box (inside if no headroom):
  - font scale `0.45`, thickness `1`
  - bg `(38,26,8)` (`#081a26`), text `(255,224,150)` (`#96e0ff`)
  - **border** = accent (1 px), **capsule shape** (radius = `card_height/2`,
    fully curved ends)
  - padding: text offset `(x1+4, baseline-4)`
  - **confidence bar** under the label, height 2 px:
    - track `(60,66,82)` (`#52423c`), fill `(40,200,120)` (`#78c828`)
    - `bar_w = max(14, int(text_width * 0.6))`
- **Static reticle at bbox center** (`radius=12`, see §3).

Label placement rule (`_label_anchor`):
1. Preferred baseline `y1 - 4` (above top edge) if room
   (`above - label_h - 8 >= 3`);
2. else below the box (`min(y2+14, img_h-14)`) if room;
3. else inside the box (`y1+8`).

### 2.2 Non-selected boxes

- **Dim interior**: fill the bbox with `(70,60,30)` (`#1e3c46`) at alpha `0.30`
  (blend over the video).
- **Edge rectangle**: color `(140,110,70)` (`#466e8c`), thickness `1`.
- **Corner brackets**: same edge color, length `8`, thickness `1`.
- **Label**: `class_name` only (no confidence), font scale `0.38`, thickness `1`,
  bg `(44,38,22)` (`#16262c`), text `(210,200,180)` (`#b4c8d2`),
  **capsule shape** (radius = `card_height/2`).

### 2.3 Fixed center reticle

Drawn always at video center `(w/2, h/2)`:
accent BGR `(70,130,175)` (`#af8246`), radius `22` (see §3).

---

## 3. Reticle (static — NO rotating scan ring)

The old rotating dotted "scan ring" was **removed**. Reticles are now static
crosshair + circle:

```
gap=5, size=11, thickness=1
left:   (cx-11,cy) -> (cx-5,cy)
right:  (cx+5,cy)  -> (cx+11,cy)
top:    (cx,cy-11) -> (cx,cy-5)
bottom: (cx,cy+5)  -> (cx,cy+11)
center dot: filled circle r=2
ring:       stroked circle, radius per caller:
            - fixed center reticle: 22
            - selected-object reticle: 12
```

---

## 4. Tracking overlay (`draw_target_tracking`, while following)

Shown when a target is being followed.

- **Leader line** from frame center to target center: 22 dashes.
  - `t = i/22`, thickness `max(1, 3 - 2t)` (tapers 3→1).
  - color interpolates from `tracking.accent` BGR `(40,225,125)` (`#7de128`)
    to `(60,200,120)` (`#78c83c`):
    `c = accent*(1-t) + (60,200,120)*t`.
  - draw filled circles only on even `i`.
- **Target ring** at target center (pulsing):
  - radius `8 + int(4*pulse(4.0))` (8…12), thickness `2`, color accent.
  - **static outer ring** at `radius + 8`, thickness `1`, same color.
  - static center dot, filled `r=3`.
- **Follow bar** bottom-center (`y = h-64`), pill:
  - text `"FOLLOW {class}   {dist:.1f}m   {speed:+.2f}m/s   YAW {yaw:+.1f}   {conf:.0f}%"`
  - font scale `0.34` (auto-drop to `0.30` if too wide), thickness `1`
  - bg `(8,40,26)` (`#1a2808`), text `(160,255,180)` (`#b4ffb4`)
  - padding h12/v7, radius `14`, outline = tracking accent `(40,225,125)`,
    dot GREEN `(45,230,130)`.
  - Data source: `tracking_data` in the payload; `dist = lidar_dist or vision_dist`.

---

## 5. Window chrome (header + footer bands)

Only needed if SGC renders the full desk window (960×864 =
84 header + 720 video + 60 footer).

- **Header band** rows `0…83`: vertical gradient `(14,20,32)` → `(24,28,44)`.
- **Footer band** rows `804…863`: gradient `(24,28,44)` → `(14,20,32)`.
- **Separator lines** at `y=84` and `y=804`:
  - top line `(58,72,104)` (`#68483a`), then 1 px below `(10,14,22)`.
- **Canvas default fill** (no frame): `(10,12,18)`.

### 5.1 Status header pills (top band, `y = (84-38)//2 = 23`)

All pills and chips are **true capsules/stadiums** (radius = `height/2`,
semicircular ends, flat top/bottom removed) — e.g. the IDLE/LIVE state pills
and the footer ESC/SPACE chips. Rounded-rect drawing must use a
cut-corner octagon + 4 corner circles (a plain `rect` does NOT produce
curved corners).

All header pills share the same top edge `y = 23` (vertical padding 7 for
every pill, so heights differ only by font size).

| Pill | Text | bg | fg | font | pad | outline | dot |
|---|---|---|---|---|---|---|---|
| State | `tracker_state.upper()` (IDLE/SELECTED/TRACKING/LOST) | STATE_THEME bg (below) | theme fg | 0.58/1 | 12,7 | theme accent | theme accent |
| Target | `"{class}  {conf:.0f}%"` (tracking only) | `(28,34,48)` `#30221c` | `(235,238,245)` | 0.55/1 | 11,7 | `(64,76,106)` `#6a4c40` | — |
| LIVE | `LIVE` | `(10,54,32)` `#20360a` | `(120,255,160)` `#a0ff78` | 0.44/1 | 7,6 | `(30,160,80)` `#50a01e` | GREEN |
| OFFLINE | `OFFLINE` | `(44,16,16)` `#10102c` | `(255,150,150)` `#9696ff` | 0.44/1 | 7,6 | `(200,40,40)` `#2828c8` | RED |
| Mode | `"  {mode.upper()}  "` centered | MODE_COLORS bg | MODE_COLORS fg | 0.55/1 | 9,7 | `(70,80,120)` `#785046` | — |

Layout: state pill (IDLE/SELECTED/TRACKING/LOST) is anchored at the left
(`x=10`), followed right by the Target pill (tracking only) then the
LIVE/OFFLINE pill, all top-aligned at `y=23`, `gap=10`. The **Mode pill is
centered separately** at the window middle (`x ≈ (w-pill_w)/2`), independent
of the left group. The takeoff button stays anchored at the top-right.

**STATE_THEME** (bg / fg / accent):

| State | bg | fg | accent |
|---|---|---|---|
| idle | `#342210` (16,34,52) | `#f5d796` (150,215,245) | `#ffbe46` (70,190,255) |
| selected | `#082130` (48,33,8) | `#82d8ff` (255,216,130) | `#28b2ff` (255,178,40) |
| tracking | `#1a2a08` (8,42,26) | `#a5f58c` (140,245,165) | `#7de128` (40,225,125) |
| lost | `#0c0a2e` (46,10,12) | `#9696ff` (255,150,150) | `#2828ff` (255,40,40) |

**MODE_COLORS** (bg / fg):

| Mode | bg | fg |
|---|---|---|
| TEST | `#322a2a` (42,42,50) | HUD_TEXT_DIM `#b29e96` |
| sitl | `#50320a` (10,50,80) | CYAN `#ffcd46` (70,205,255) |
| flight | `#18370e` (14,55,24) | GREEN `#82e62d` (45,230,130) |

### 5.2 Takeoff button (top-right of header)

- Size `150 × 30`, `bx = w-158`, `by = 23`, rounded radius `15`.
- Armed: fill `(12,52,30)` `#1e340c`, border `(60,200,120)` `#78c83c`,
  dot GREEN, text `"ARMED"`.
- Disarmed: fill `(10,40,68)` `#44280a`, border `(70,180,255)` `#ffb446`,
  dot CYAN, text `"TAKEOFF 5m"`.
- Dot at `(bx+16, by+15)` r3; text font `0.42/1` `(235,238,245)`.

### 5.3 Footer shortcut bar (bottom-center)

`y = h-60 + (60-24)//2 = h-42`, chip height `24`, gap `8`, centered line.
Chips: `ESC/deselect`, `SPACE/follow`, `R/reset`, `H/hud`, `Q/quit`.

- chip bg `(30,36,50)` `#32241e`, **capsule** (radius `ch/2`)
- key box bg `(52,62,88)` `#583e34`, **capsule** (radius `(ch-2)/2`)
- key text `(220,235,255)` `#ffebdc`, font `0.50/1`
- value text HUD_TEXT_DIM `(150,158,178)`, font `0.40/1`

---

## 6. Prompts & banners

### 6.1 Selection prompt (nothing selected)

Center-bottom `y = h-64`, pill:
- `"CLICK - SELECT TARGET     {N} OBJECTS"`
- bg `(12,36,52)`, text CYAN `(70,205,255)`, font `0.46/1`,
  pad 12/7, radius `14`, outline `(40,120,170)` (`#aa7828`), dot CYAN.

### 6.2 Follow prompt (target selected, not following)

`"SELECTED - {class}    [SPACE] FOLLOW"`, same placement:
- bg `(48,33,8)` `#082130`, text `(255,218,130)` `#82d8ff`, font `0.46/1`,
  pad 12/7, radius `14`, outline AMBER `(255,190,60)`, dot AMBER.

### 6.3 Lost banner (target lost)

Full-width strip at top of video (`y ≈ 4…44`):
- animated: `alpha = 0.42 + 0.18*|((now*4) % 2) - 1|` over bg
  `(36,4,10)` `#0a0424`, rounded `6`.
- diagonal stripes `(90,8,10)` scrolling, alpha `0.35` over the whole band.
- border **RED** `(255,70,70)`, bottom edge `(60,8,10)`.
- centered text `"TARGET LOST"` font `0.58/2`: fill `(255,235,235)`
  `#ebebff`, stroke `(255,120,120)` `#7878ff`.
- optional RTL pill `"RTL {s}s"` / `"RTL ACTIVE"`: bg `(70,8,8)`,
  text `(255,190,190)` `#bebeff`, outline RED.
- `DISMISS` button right: rounded `12`, bg `(52,52,62)`, border
  `(130,130,150)` `#968282`, text `(220,220,230)` `#e6dcdc`.

---

## 7. Color constants (single source of truth)

| Name | BGR | CSS hex | Use |
|---|---|---|---|
| HUD_TEXT | `(235,238,245)` | `#f5eeeb` | main text |
| HUD_TEXT_DIM | `(150,158,178)` | `#b29e96` | secondary text |
| HUD_ACCENT | `(0,190,235)` | `#ebbe00` | accent |
| GREEN | `(45,230,130)` | `#82e62d` | ok / live dots |
| CYAN | `(70,205,255)` | `#ffcd46` | info dots |
| AMBER | `(255,190,60)` | `#3cbeff` | warn dots |
| RED | `(255,70,70)` | `#4646ff` | alarms |
| selected accent | `(255,178,40)` | `#28b2ff` | selected box/reticle |
| other-edge | `(140,110,70)` | `#466e8c` | non-selected box |
| center-reticle | `(70,130,175)` | `#af8246` | fixed center reticle |
| tracking accent | `(40,225,125)` | `#7de128` | follow target |

---

## 8. Streamed data → how to render (mapping)

From each `FrameDetections` frame:

- `detections[]` — one entry per object; use `is_selected` to pick the
  selected style (§2.1) vs the dim style (§2.2).
  - Per-detection `bbox_color` is provided in the payload, but to match the
    on-drone look use the SGC-side constants in §7 (selected accent for
    `is_selected=true`, `other-edge` for the rest). Treat the payload color as
    an override only.
- `fps`, `mode`, `tracker_state` (`idle/selected/tracking/lost`) feed the
  header pills (§5.1) and state theming.
- `overlay.counter_text` = object count (top-left is optional; on-drone the
  count only appears in the selection prompt).
- `overlay.prompt_text` = selection / follow prompt (§6.1/§6.2).
- `overlay.show_tracking_bar` + `overlay.tracking_bar_text` = follow bar (§4).
- `overlay.show_lost_banner` = lost banner (§6.3).
- `frame_w`/`frame_h` = canvas size (draw overlay 1:1).
- `hud_visible` = global show/hide of all overlays.

**Thickness/label summary (quick reference):**
selected: brackets 18/3 + 10/1, label `class conf%` (0.45) with border +
conf bar; others: 1 px rect + brackets 8/1, label `class` (0.38), dim 30%.
Reticle: gap 5, size 11, center r22, selected-object r12, no animation.