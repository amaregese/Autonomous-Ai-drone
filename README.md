# Autonomous-AI-Drone

Onboard Jetson target-tracking and person-following pipeline. YOLO11 runs on the live
camera feed, a user-selected object is tracked, and the drone flies toward it via
ArduPilot/MAVLink commands in `GUIDED` mode while keeping the target centred. The system
also provides an OpenCV HUD overlay, RTSP video and UDP detection streaming to a ground
station (SGC), command-and-control back from the SGC, monocular distance estimation,
automatic chessboard camera calibration, and a Return-to-Launch fallback if the target is
lost.

> **This file is the single source of documentation for the repository.** It was produced by
> merging 34 prior Markdown documents (project reports, dependency maps, verification
> audits, deployment checklists, benchmark reports and cleanup plans). Where those
> documents disagreed, the disagreement is preserved and annotated in
> [Known contradictions and defects](#known-contradictions-and-defects) rather than silently
> resolved. Sections marked *(historical)* describe audits that were performed at a point
> in time and are not current statements about the tree.

---

## Table of contents

- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Architecture](#architecture)
- [Detection and tracking](#detection-and-tracking)
- [Distance estimation](#distance-estimation)
- [Person-follow control](#person-follow-control)
- [SGC ground-station link](#sgc-ground-station-link)
- [Tracker overlay specification](#tracker-overlay-specification)
- [Camera calibration](#camera-calibration)
- [Video streaming](#video-streaming)
- [Testing](#testing)
- [Jetson deployment](#jetson-deployment)
- [Target platform: Jetson Orin Nano](#target-platform-jetson-orin-nano-verified-hardware-facts)
- [JetPack and CUDA matrix](#jetpack-and-cuda-matrix)
- [Install torch on the Jetson](#install-torch-on-the-jetson-the-part-that-breaks)
- [OpenCV and JetPack](#opencv-and-jetpack)
- [Distance benchmark](#distance-benchmark)
  - [Circular-fit warning: the 0.133 m baseline is not valid accuracy](#circular-fit-warning-the-0133-m-baseline-is-not-valid-accuracy)
- [Repository hygiene and Git policy](#repository-hygiene-and-git-policy)
- [Known contradictions and defects](#known-contradictions-and-defects)
- [Historical audit record](#historical-audit-record)
- [Consolidation provenance](#consolidation-provenance)

---

## Quick start

### Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Jetson
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` is deliberately unpinned and minimal:

```
opencv-python
numpy
torch
ultralytics
simple-pid
pymavlink
pyserial
```

The YOLO11n weights ship in the repository at `YOLO/yolo11n.pt` (5,613,764 bytes, blob
`45b273b46165267f2d4c90df94468b547f43cf67`). **The file is tracked in Git on purpose** — see
[Repository hygiene and Git policy](#repository-hygiene-and-git-policy). It is loaded from a
local path; there is no auto-download anywhere in the code.

### Run

Pick the flight controller interactively (the default — no flags needed):

```bash
python autonomous_drone_main.py
```

```
Flight controller to connect to
  [1] SITL - ArduPilot simulator on udpin:0.0.0.0:14550
  [2] SITL - launch ArduPilot SITL in WSL, then connect
  [3] Real FCU serial/COM: /dev/ttyACM0
  [4] Real FCU serial/COM: COM7
Choice [3]:
```

The preselected entry is the first real FCU port when `--mode flight`, otherwise SITL. Skip the
menu entirely by pinning the link with `--drone-link`:

```bash
python autonomous_drone_main.py --drone-link sitl-launch   # start ArduPilot SITL in WSL, then connect
python autonomous_drone_main.py --drone-link sitl          # connect to an already-running SITL
python autonomous_drone_main.py --drone-link real          # real FCU on the first detected port
```

Or name the endpoint directly, keeping the old behaviour:

```bash
python autonomous_drone_main.py --mode sitl --start-sitl
python autonomous_drone_main.py --mode flight --drone_connection COM7
# or on Jetson
python autonomous_drone_main.py --mode flight --drone_connection /dev/ttyTHS1
```

See [`--drone-link`](#choosing-the-flight-controller) for the precedence rules.

Common options:

```bash
--camera 0                     # force webcam index (default: auto-detect)
--imgsz 320                    # inference resolution (default 320)
--model-path YOLO/yolo11n.pt   # detector weights
--auto-calibrate               # background chessboard calibration
--chessboard 9x6               # inner corners, default 9x6
--intrinsics-fx 446.7 --intrinsics-fy 446.7 --intrinsics-cx 320.0 --intrinsics-cy 240.0
--sgc-host 192.168.1.100 --sgc-port 9001 --sgc-cmd-port 9002
--rtsp-port 8554 --jpeg-quality 30
--no-prompt                    # skip the interactive prompt
--no-sgc-host-prompt           # don't ask for the SGC IP (headless)
```

The IDE **Run** button (no arguments) works too: defaults are read from `run_config.json` in the
repository root, so a bare launch behaves like the documented command line. Edit that file to
change the mode, flight controller, SGC address, or any other option — see
[`run_config.json`](#run_configjson-ide--run-button-defaults). The shipped file keeps
`"drone_link": "auto"` and no `drone_connection` key, so the **Run** button opens the
[flight-controller menu](#choosing-the-flight-controller) instead of always attaching to a real
FCU. Because the SGC laptop's IP changes between setups, the app also asks for it once at startup —
press ENTER to keep the last known address.

Flight-safety preconditions are enforced before takeoff in `_preflight_follow`
(`autonomous_drone_main.py:195`): minimum altitude 2.5 m (below the 3 m cruise/takeoff
altitude, so a follow is never refused for being at its own configured height), minimum
battery 20 %, minimum GPS fix type 3, and FCU mode must be `GUIDED`.

### Test

```bash
python -m pytest -q
```

Current result: **355 passed, 315 subtests passed** (see [Testing](#testing)).

---

## Configuration

### `modules/app_config.py`

The central configuration module. Two groups of constants live here.

**Legacy follow constants**

| Constant | Value |
| --- | --- |
| `MAX_FOLLOW_DIST` | 1.0 |
| `MAX_ALT` | 5.0 |
| `MAX_SPEED` | 2.0 |
| `MAX_YAW` | 15.0 |
| `GAIN_FORWARD` | 0.8 |
| `GAIN_YAW` | 2.0 |
| `FORWARD_DEADBAND` | 0.18 |
| `FORWARD_BRAKE_ZONE` | 0.75 |
| `LIDAR_BLEND_WEIGHT` | 0.65 |
| `TRACKING_LOST_THRESHOLD` | 30.0 |
| `OBJECT_HEIGHT` | 0.25 |

**Task 9 — YOLO11 + distance-estimator person-follow**

| Constant | Value | Meaning |
| --- | --- | --- |
| `FOLLOW_SPEED_PROFILE` | `[(2.0,0.0),(3.0,0.3),(5.0,0.6),(8.0,1.0)]` | distance (m) → forward speed (m/s) breakpoints |
| `FOLLOW_MAX_SPEED_MPS` | 1.0 | cap beyond the last breakpoint |
| `FOLLOW_MIN_SPEED_MPS` | 0.0 | |
| `FOLLOW_MIN_DISTANCE_M` | 2.0 | closer than this → hover |
| `FOLLOW_MAX_LATERAL_SPEED_MPS` | 0.6 | |
| `FOLLOW_ACCEL_LIMIT_MPS2` | 0.5 | acceleration ramp limit |
| `FOLLOW_CONFIDENCE_THRESHOLD` | 0.5 | below this the measurement is rejected |
| `FOLLOW_TARGET_LOSS_TIMEOUT_S` | 10.0 | seconds before the loss action fires |
| `FOLLOW_ENABLED` | `True` | |
| `FOLLOW_RTL_ON_LOSS` | `False` | RTL on loss is **off** by default |
| `FOLLOW_LATERAL_GAIN` | 1.2 | |
| `FOLLOW_TARGET_CLASS` | `"person"` | |
| `FOLLOW_MAX_DETECTION_AGE_S` | 0.5 | stale-detection rejection |
| `MIN_FOLLOW_ALT` / `MIN_FOLLOW_BATTERY` / `MIN_FOLLOW_GPS_FIX` / `MIN_FOLLOW_MODE` | 2.5 / 20 / 3 / `"GUIDED"` | preflight gates (`MIN_FOLLOW_ALT` = `FOLLOW_ALTITUDE - ALTITUDE_TOLERANCE`) |
| `DISARM_MAX_ALT` | 0.5 | how far above home altitude the vehicle may be when it is disarmed |

`OBJECT_HEIGHTS` maps each of the 80 COCO classes to an assumed real-world height in metres
(`person` 1.3, `bicycle` 1.0, `car` 1.5, `giraffe` 4.5, …). These are **approximations** used
by the geometric-vision ranging method and are a documented source of ranging error.

`FollowConfig.from_app_config()` (`modules/person_follow.py:56`) reads all of the above at
runtime, so editing `app_config.py` is sufficient to retune behaviour.

### Command-line arguments

Defined in `autonomous_drone_main.py`. Complete set:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--debug_path` | `debug/run1` | debug output directory |
| `--mode` | `sitl` | `sitl` or `flight` |
| `--control` | `PID` | control scheme |
| `--drone-link` | `auto` | `auto` (ask), `sitl` (running SITL), `sitl-launch` (auto-start SITL in WSL), `real` (serial FCU) |
| `--drone_connection` | `None` | explicit MAVLink endpoint; overrides `--drone-link auto` |
| `--sitl-connection` | `udpin:0.0.0.0:14550` | MAVLink endpoint for SITL |
| `--baud` | `57600` (overridden by `run_config.json`, which currently sets `115200`) | serial baud for a real FCU |
| `--no-prompt` | flag | skip interactive prompt |
| `--model-path` | `YOLO/yolo11n.pt` | detector weights |
| `--camera` | `None` | force webcam index |
| `--conf-threshold` | `None` | detector confidence threshold |
| `--iou-threshold` | `None` | detector IoU threshold |
| `--min-box-area-ratio` | `None` | reject boxes smaller than this fraction of frame |
| `--imgsz` | 320 | inference size |
| `--rtsp-port` | 8554 | RTSP server port |
| `--jpeg-quality` | 30 | MJPEG quality, 1–100 (lower = faster) |
| `--sgc-host` | `None` | SGC address (default `127.0.0.1` in SITL, `192.168.1.100` in flight) |
| `--sgc-host-prompt` | flag | ask for the SGC IP on startup unless `--sgc-host` is given |
| `--sgc-port` | 9001 | SGC detection UDP port (drone → SGC) |
| `--sgc-cmd-port` | 9002 | SGC command UDP listen port (SGC → drone) |
| `--start-sitl` | flag | auto-launch SITL in WSL before connecting |
| `--intrinsics-fx/fy/cx/cy` | `None` | camera intrinsics in pixels |
| `--intrinsics-width/height` | 640 / 480 | reference calibration resolution |
| `--auto-calibrate` | flag | run chessboard calibration in a background thread |
| `--chessboard` | `9x6` | chessboard inner corners |
| `--object-height` | 0.25 | assumed object height for monocular ranging |

### `run_config.json` (IDE / Run button defaults)

`modules/run_config.py` loads `run_config.json` from the repository root and uses its values as
argparse **defaults**, so launching from PyCharm/VS Code with no arguments behaves like the
documented command line. CLI flags always win, and a missing or malformed file is ignored (the
built-in defaults are used). Keys may be dashed or underscored (`"sgc-host"` == `"sgc_host"`);
boolean flags are `true`/`false`.

```json
{
  "mode": "flight",
  "drone_link": "auto",
  "sitl_connection": "udpin:0.0.0.0:14550",
  "baud": 115200,
  "sgc_host": "192.168.1.160"
}
```

Set the `AAD_RUN_CONFIG` environment variable to read the file from another path. On startup the
file that was used is echoed as `[CONFIG] defaults from ...`.

Flags whose default can be `true` in this file also accept an inverse switch so they stay
overridable from the command line: `--prompt`, `--no-start-sitl`, `--no-auto-calibrate`,
`--no-headless`, `--flip-camera`, `--no-sgc-host-prompt`.

#### Startup flow

Everything that needs an answer is asked **first**, before the camera, model and simulator are
touched, so a mistake costs a keystroke instead of a 10-second startup. The flight controller is
chosen first, because a SITL endpoint switches the run to `sitl` mode and that in turn decides the
SGC default:

```
Flight controller to connect to
  [1] SITL - ArduPilot simulator on udpin:0.0.0.0:14550
  [2] SITL - launch ArduPilot SITL in WSL, then connect
  [3] Real FCU serial/COM: /dev/ttyACM0
Choice [3]:

SGC (ground station)
  this machine      : 192.168.1.160, 192.168.1.137
  last known SGC    : 192.168.1.247
  Detections are sent there over UDP; commands arrive on port 9002.
  SGC IP address [192.168.1.247]:

Starting with
  mode          : flight
  drone link    : real FCU on /dev/ttyACM0 @ 115200 baud
  SGC target    : 192.168.1.247:9001   (commands come in on UDP 9002)
  video stream  : port 8554, detection size 320px
  model         : YOLO/yolo11n.pt
  camera, lidar and the model load next (a few seconds).

[STREAM] video:  http://192.168.1.160:8554/stream
[STREAM] detections -> 192.168.1.247:9001 (UDP)
[STREAM] commands <- UDP 9002 on 192.168.1.160, 192.168.1.137
[READY] tracking is live — click an object in the window, SPACE to follow, Q to land
```

#### Choosing the flight controller

`--drone-link` decides what to connect to, and is resolved in this order:

| `--drone-link` | Result |
| --- | --- |
| `real` | first detected serial/COM port, or `--drone_connection` when given; never starts SITL |
| `sitl` | `--sitl-connection` (default `udpin:0.0.0.0:14550`) |
| `sitl-launch` | same endpoint, and ArduPilot SITL is started in WSL first |
| `auto` (default) | the real FCU first, then SITL as a fallback; `--drone_connection` overrides both |

Notes:

- **Real FCU first.** In `auto` the app connects to the first detected serial/COM port and only
  falls back to the SITL endpoint if no vehicle answers there. The fallback is always announced
  (`[LINK] no flight controller on the previous link - falling back`) and switches the run to `sitl`
  mode, because quietly attaching to a simulator instead of the drone is the kind of surprise that
  ends badly. If nothing answers anywhere, every link that was tried is listed before it exits.
- **An explicit choice is never overridden.** `--drone-link real` with no FCU present exits with a
  clear message instead of falling back to SITL — you asserted a drone exists, so a silent simulator
  would hide the real problem. `--drone-link sitl` likewise connects to SITL only and will not grab a
  real FCU that happens to be plugged in.
- **No fabricated ports.** With no FCU attached, the real-FCU candidate is simply absent from the
  list. This used to invent a `COM3` that does not exist and then die on
  `could not open port 'COM3'` rather than trying the fallback.
- `auto` is what makes the prompt reachable. A `drone_connection` key in `run_config.json` is an
  explicit endpoint and therefore short-circuits it — delete the key (or set `"drone_link": "auto"`)
  to get the menu back.
- `sitl-launch` needs WSL on Windows. Elsewhere the app says so and connects to an already-running
  SITL instead of failing.
- `udp`/`tcp` endpoints are recognised as SITL, including the bare `udp:host:port` form. Selecting one
  while `--mode flight` prints `[LINK] SITL endpoint selected - running in sitl mode`, so the HUD, the
  SGC default and `streamer.set_mode` all follow the link you actually picked.
- A SITL in WSL is reached by listening locally on `0.0.0.0:14550` and letting `mavproxy --out
  <this-host>:14550` send to it. Nothing is listening on the WSL side, so pointing `sitl_connection`
  at the WSL IP will not work.
- Whichever option auto-launches SITL is also the one that is stopped on exit (Q), not just
  `--start-sitl`.
- `baud` only applies to a serial FCU; it is ignored for UDP endpoints.

#### Choosing the SGC IP

The SGC laptop gets a different address in every setup, so `sgc_host` is treated as a *suggestion*
rather than a fixed value. Unless `--sgc-host` is passed on the command line, the app asks once
before it opens the UDP transport:

- press ENTER to reuse the last known address;
- type an IPv4 address **or a hostname** (`my-sgc.local`) — it is resolved before use, and invalid
  input is reported and re-asked (3 attempts) instead of failing later inside a socket;
- a new address is used for this run only, and you are then offered
  `Remember <ip> as the default for next time? [y/N]`, which rewrites only the `sgc_host` key;
- if the address shares no `/24` with this machine the app prints a warning, since the SGC is then
  probably unreachable from here;
- if the last known value came from neither the CLI nor the file, the prompt falls back to
  `127.0.0.1` in SITL and `192.168.1.100` in flight mode.

Skip the prompt with `--sgc-host <ip>` (one-shot) or `--no-sgc-host-prompt` /
`"sgc_host_prompt": false` in `run_config.json` (headless Jetson deployment). A closed or
non-interactive stdin falls back to the last known address instead of failing.

#### If the SGC is on a different address than configured

`SGCCommandReceiver` records the sender of every command. The first time a command arrives from a
host other than the configured `sgc_host`, the app says so once in the console and in the HUD
banner, including the exact flag to use:

```
[SGC] SGC is at 192.168.1.51, not 192.168.1.247 - detections are going to the wrong machine.
      Restart with --sgc-host 192.168.1.51
```

Pressing `Ctrl+C` during startup now exits cleanly with `Startup cancelled.` instead of a
traceback, and any other startup failure prints `Startup failed: <reason>` before the traceback.

---

## Repository layout

```
Autonomous-AI-Drone/
├── README.md                    # this file — the only documentation
├── autonomous_drone_main.py     # entry point / main loop
├── run_config.json              # persisted defaults for IDE / Run button launches
├── requirements.txt
├── YOLO/
│   └── yolo11n.pt               # tracked model weights (5.6 MB)
├── modules/                     # core production code
│   ├── app_config.py            # central configuration
│   ├── run_config.py            # run_config.json loader (argparse defaults)
│   ├── auto_calibrate.py        # chessboard camera calibration
│   ├── control.py               # 42-byte abstraction shim
│   ├── detector_yolo11.py       # 42-byte abstraction shim
│   ├── drone.py                 # 41-byte abstraction shim
│   ├── lidar.py                 # 42-byte abstraction shim
│   ├── display.py               # OpenCV HUD / overlay rendering (~34 KB)
│   ├── person_follow.py         # Task 9 follow controller
│   ├── tracking.py              # click-to-select target session
│   ├── drone_visualizer.py      # 53-byte wrapper → visualizer_ui
│   ├── navigation.py            # LEGACY FollowController
│   ├── vision.py                # LEGACY
│   ├── vision_utils/            # LEGACY geometry helpers
│   ├── control_system/          # PID + state + visualizer façade
│   ├── distance_estimator/      # monocular distance estimation
│   │   ├── estimator.py         # fusion + Kalman
│   │   ├── vision.py            # geometric vision ranging
│   │   ├── calibration.py       # CameraIntrinsics
│   │   ├── config.py  filter.py  models.py  lidar.py  depth.py
│   │   └── depth_backend/       # metric_depth (ZoeDepth), synthetic
│   ├── drone_backend/           # api.py, sitl.py, mock_vehicle.py
│   ├── lidar_backend/           # mock LiDAR
│   ├── visualizer_ui/           # ACTIVE visualizer implementation
│   └── yolo11_detector/         # model, source, matching, config, api, types
├── jetson/                      # Jetson-specific
│   ├── communication/           # detection_sender.py, sgc_receiver.py
│   └── streaming/               # rtsp_server.py
├── shared/                      # detection_models.py, detection_transport.py
├── tools/                       # 10 calibration / benchmark / diagnostic scripts
├── benchmarks/
│   ├── camera_calibration/      # checkerboard target
│   └── distance/                # manifest.json (calibration + empty dataset)
```

> `benchmarks/distance/images/` and `benchmarks/distance/results/` have been **deleted** and
> are now gitignored. They held a 10-image desktop-camera ground-truth set and its results.
> The images are unusable on the Orin Nano (different camera and resolution) and both are
> recoverable at `cd766f91` if you want to re-measure on desktop. `manifest.json` is kept and
> its `samples` array is now empty, so `tools/run_distance_benchmark.py` correctly reports
> `READY FOR DATA COLLECTION` / `NOT AVAILABLE` until you collect Jetson data. See
> [the circular-fit warning](#circular-fit-warning-the-0133-m-baseline-is-not-valid-accuracy).

The five "abstraction shim" files (`control.py`, `drone.py`, `lidar.py`,
`detector_yolo11.py`, `drone_visualizer.py`) are one-line star-import files of 41–53 bytes.
They exist so that `from modules import drone` keeps working while the real implementation
lives in a subpackage. See [Historical audit record](#historical-audit-record) for the YOLO
wrapper in particular.

### What is not in the tree

The repository has been trimmed to a deployment shape. Removed: the two duplicate
`tools/test_*.py` runners, 16 root audit scripts, `AUTHORITATIVE_FILE_INVENTORY.csv`, all 33
non-`README.md` Markdown files, 12 `__pycache__` directories, the stale
`shared/__pycache__/sgc_config.cpython-312.pyc` bytecode, and the `.idea/` + `.vtcode/`
tooling. The only tracked file still matched by an ignore rule is the model, by design.

`tests/` was removed in that trim and has since been **restored and extended**: it now holds
10 tracked files covering servo channel planning, servo MAVLink routing, link selection, the
SGC receiver and app config — see [Testing](#testing).

**`modules/navigation.py`, `modules/vision.py` and `modules/vision_utils/` were kept even
though nothing on the flight path imports them.** They are the reference implementation behind
the "Legacy bbox" benchmark column — `distance_estimator/evaluation.py:99` imports
`FollowController` from them — so deleting them would make the distance benchmark
unreproducible. See [Historical audit record](#historical-audit-record).

**Automated regression coverage lives in `tests/`** — 355 tests, all passing
(`python -m pytest -q`). The older 227/2 baseline and the two depth-laziness failures refer to
a deleted suite; see [Testing](#testing).

---

## Architecture

### Entry point

`autonomous_drone_main.py` is the single entry point. Notable structure:

| Line | Symbol | Role |
| --- | --- | --- |
| 177 | `_reset_lost_state` | clears tracking-loss state after a successful follow |
| 183 | `_preflight_follow` | battery / GPS / mode / altitude gates before takeoff |
| 233 | `_airframe_readiness_problems` | GPS-fix / EKF gates shared by ARM and TAKEOFF |
| 253 | `_run_flight_command` | runs a blocking vehicle call on the `flight-cmd` worker thread |
| 295 | `_handle_arm_button` | on-screen ARM button |
| 317 | `_handle_takeoff_button` | on-screen takeoff button (requires an armed vehicle) |
| 344 | `_handle_land_button` | on-screen LAND button (stops follow, keeps the app running) |
| 360 | `_ground_altitude` | altitude above home, used by the disarm guard |
| 368 | `_handle_disarm_action` | disarm (refused above `DISARM_MAX_ALT`) — `D` key / SGC `disarm` |
| 407 | `_on_mouse` | click-to-select target, ARM, TAKEOFF and LAND buttons |
| 418 | `_serial_heartbeat_ok` | serial heartbeat watchdog |
| 440 | `_detect_serial_ports` | COM-port enumeration |
| 459 | `_default_fcu_serial` | default FCU port |
| 542 | `_pick_connection` | connection-string selection |
| 723 | `setup` | camera, detector, estimator, SGC, streaming initialisation |
| 885 | `_handle_sgc_command` | inbound SGC command handling |
| 975 | `_handle_keyboard` | `ESC` deselect, `SPACE` follow, `R` reset, `H` HUD, `L` land, `D` disarm, `P` panic RTL, `Q` quit |
| 1045 | `_update_hud_state` | |
| 1057 | `_build_telemetry` | builds `TelemetryData` for the SGC payload |
| 1094 | `_selected_distance` | current target distance for the HUD |
| 1116 | `_console_status` | console status line |
| 1173 | `_follow_movement_dict` | follow command → dict for the HUD |
| 1206 | `_refresh_follow_estimator` | re-creates the estimator when intrinsics change |
| 1232 | `main_loop` | the per-frame loop (drains the SGC queue, `panic_rtl` first) |
| 1422 | `land` | |
| 1437 | `_failsafe_rtl` | RTL failsafe |

### Data flow

```
camera (cv2.VideoCapture)
  └─ modules/yolo11_detector/source.py        capture + horizontal flip
      └─ modules/yolo11_detector/api.py       frame-skip, inference, selection
          ├─ modules/yolo11_detector/matching.py   IoU + feature matching, reacquisition
          └─ modules/distance_estimator/
                 ├─ vision.py   geometric ranging (bbox height + intrinsics)
                 ├─ lidar.py    mock/ranged source
                 ├─ depth.py    → depth_backend/metric_depth.py (ZoeDepth, optional)
                 └─ estimator.py fusion + Kalman filter
                     └─ modules/person_follow.py  PersonFollowController
                          └─ FollowCommand
                              └─ modules/drone_backend/api.py
                                   └─ sitl.py | mock_vehicle.py  → MAVLink
```

### Control flow

The visualizer/HUD chain is production-reachable, not debug-only. It is reached through a
star-import shim:

```
autonomous_drone_main.py
  → modules/control.py:1                from modules.control_system.api import *
      → modules/control_system/api.py:1-4
          → modules/control_system/visualizer.py:1
              → modules/drone_visualizer.py:1
                  → modules/visualizer_ui/drone_visualizer.py   (9,880 bytes — the real implementation)
```

`modules/control_system/api.py` also reaches the drone backend at line 1
(`from modules import drone`). Verified call sites in `api.py`:

- `get_visualizer()` / `set_visualizer_status()` — lines 59–65
- `state.visualizer.update(state.movementYawAngle * 0.1, 0)` — lines 75–76
- `state.visualizer.update(0, state.movementRollAngle * 0.1)` — lines 84–85
- `state.visualizer.draw()` — lines 87–88

> An earlier audit doc claimed `autonomous_drone_main.py` imports ten symbols directly from
> `modules.control_system.api` and cited line numbers 421 / 833 / 838 / 889 for the PID
> configuration, telemetry update, `control_drone()` call and HUD block. **None of that is
> accurate.** The entry point has no `control_system` import; the real lines are 973
> (`configure_PID`), 841 (telemetry update), and 913–916 (HUD block), and
> `control_drone` has zero call sites in the entry point. See
> [Known contradictions and defects](#known-contradictions-and-defects).

---

## Detection and tracking

**Model** — YOLO11n via Ultralytics, weights at `YOLO/yolo11n.pt`. 80 COCO classes.

**`modules/yolo11_detector/`**

| File | Role |
| --- | --- |
| `model.py` | loads the local `.pt` path; guarded by `os.path.exists` at lines 128–129, raises inside a `try` at lines 6–17, and **never** downloads (lines 150–157) |
| `source.py` | `cv2.VideoCapture`; MSMF backend on Windows; applies `cv2.flip(frame, 1)` |
| `api.py` | global model/camera/tracking state, `FRAME_SKIP_CAMERA = 2`, selected-object persistence |
| `matching.py` | IoU + visual-feature matching, target memory for reacquisition (up to 90 frames), velocity prediction through occlusion |
| `config.py`, `types.py` | thresholds and the `Detection` dataclass |

Failure behaviour matters for deployment: if the weights are missing, `model.py` returns
`(None, [])`, `api.py:57-62` returns `False`, and `autonomous_drone_main.py:752` aborts
**before** flight setup. A missing model file is therefore a hard startup failure, not a
degraded mode.

**Target selection** — `modules/tracking.py` provides `TrackingSession`; a mouse click
selects a detection (`_on_mouse` at `autonomous_drone_main.py:407`), and the selection
persists across frames.

---

## Distance estimation

`modules/distance_estimator/` implements monocular distance estimation with optional
metric depth, a LiDAR-style ranged source, sensor fusion and a Kalman filter.

### File classification

Fifteen modules. Classification letters here are **per-document**; see
[Known contradictions and defects](#known-contradictions-and-defects) for the fact that
`D` means two different things across the source documents.

**Class A — production (11 files)**

`estimator.py`, `vision.py`, `calibration.py`, `config.py`, `filter.py`, `models.py`,
`lidar.py`, `depth.py`, `depth_backend/__init__.py`, `depth_backend/metric_depth.py`,
`depth_backend/synthetic.py`

**Class D — development-only (4 files)**

`report.py`, `analysis.py`, `dataset.py`, `evaluation.py`

`__init__.py` re-exports only the A-class names. `report`, `analysis`, `dataset` and
`evaluation` are **not** exported from the package API. No production module imports any
D-class file.

D-class importers:

| File | Imported by |
| --- | --- |
| `report.py` | `tools/run_distance_benchmark.py` (and the removed `tests/test_distance_dataset_benchmark.py`, restorable from history) |
| `analysis.py` | `tools/run_distance_benchmark.py` |
| `dataset.py` | `tools/create_distance_dataset.py`, `tools/live_distance_capture.py`, `tools/run_distance_benchmark.py` (and the removed `tests/test_distance_dataset_benchmark.py`) |
| `evaluation.py` | `tools/run_distance_benchmark.py` (and the removed `tests/test_distance_dataset_benchmark.py`) |

No circular dependencies are reported.

### Optional metric depth

`depth_backend/metric_depth.py` is optional and requires **torch + transformers** (ZoeDepth).
It is not imported by the core path; that laziness was checked by the removed
`tests/test_distance_estimator_depth.py` (restorable from history at `cd766f91`) and is not
covered by the current suite. To enable it in benchmarks, pass `--enable-depth`; the model
downloads once on first use.

### Camera intrinsics

`calibration.py` provides `CameraIntrinsics`. Supply values via `--intrinsics-*` flags, or
let `--auto-calibrate` fill them from a chessboard, or fall back to the configured default
`fx = fy = 446.7`, `cx = 320.0`, `cy = 240.0` at 640×480 (see
[Distance benchmark](#distance-benchmark) for exactly where that number came from and why it
is not a calibration).

---

## Person-follow control

`modules/person_follow.py` implements the Task 9 follow controller.

**Classes**

| Line | Symbol |
| --- | --- |
| 34 | `FollowConfig` (56: `from_app_config`) |
| 78 | `FollowCommand` |
| 97 | `PersonFollowController` |

**`PersonFollowController` methods**

| Line | Method | Role |
| --- | --- | --- |
| 106 | `__init__` | |
| 121 | `set_distance_estimator` | |
| 124 | `set_focal_length` | |
| 128 | `reset` | |
| 139 | `lost_time_s` | |
| 146 | `update` | main per-frame entry → `FollowCommand` |
| 207 | `_usable` | confidence / age / class gating |
| 221 | `_on_invalid` | |
| 243 | `_mark_target_present` | |
| 248 | `measure_range` | LiDAR distance preferred, geometry fallback |
| 305 | `_geometry_range` | |
| 317 | `_target_forward_speed` | applies `FOLLOW_SPEED_PROFILE` |
| 333 | `_target_lateral_speed` | |
| 339 | `_x_delta` | |
| 346 | `_resolve_dt` | |
| 354 | `_ramp` | enforces `FOLLOW_ACCEL_LIMIT_MPS2` |
| 362 | `loss_action` | |
| 366 | `loss_timeout_reached` | honours `FOLLOW_TARGET_LOSS_TIMEOUT_S` |

**Behaviour.** Forward control uses the horizontal ground distance `H` from the vision
distance-estimation pipeline:

| `H` | Action |
| --- | --- |
| `H > 4.5 m` | forward (existing distance→speed breakpoints) |
| `3.5–4.5 m` | hold |
| `H < 3.5 m` | backward at `FOLLOW_REVERSE_SPEED` (0.3 m/s) |
| invalid `H` | stop |

The setpoint is `FOLLOW_DISTANCE` (4.0 m) with `DISTANCE_TOLERANCE` (0.5 m); the backward speed
is fixed at `FOLLOW_REVERSE_SPEED` (0.3 m/s). Lateral speed is driven by the horizontal
image-space delta scaled by `FOLLOW_LATERAL_GAIN` and capped at 0.6 m/s. All velocity changes
remain rate-limited to 0.5 m/s². Measurements below `FOLLOW_CONFIDENCE_THRESHOLD` or older than
`FOLLOW_MAX_DETECTION_AGE_S` are rejected.

**Geometry limitation.** The horizontal conversion assumes an approximately level camera
orientation; pitch compensation / ATTITUDE integration is not yet implemented. The existing
camera calibration values have not been changed.

**Loss handling.** After `FOLLOW_TARGET_LOSS_TIMEOUT_S` (10 s) the configured loss action
fires. `FOLLOW_RTL_ON_LOSS` is `False` by default, so the default is a non-RTL loss action;
`_failsafe_rtl` at `autonomous_drone_main.py:1437` is the separate RTL path.

---

## SGC ground-station link

The SGC (station ground control) link is bidirectional and carries detections outbound and
commands inbound over UDP.

### Ports

| Port | Direction | Constant |
| --- | --- | --- |
| 9001 | drone → SGC (detections) | `--sgc-port` |
| 9002 | SGC → drone (commands) | `--sgc-cmd-port` |
| 8554 | RTSP video | `--rtsp-port` |

Default SGC host: `127.0.0.1` in SITL, `192.168.1.100` in flight.

### Transports

`shared/detection_transport.py` defines `DetectionTransport` (ABC, line 16), `UDPTransport`
(line 32) and `TCPTransport` (line 109). `UDPTransport` sends
`json.dumps(frame.to_dict())` as a length-prefixed datagram (lines 58–61); the TCP variant
is at line 150.

`shared/detection_models.py` holds the schemas. **This file, not any specification
document, is the authoritative schema.** Classes:

| Line | Class |
| --- | --- |
| 10 | `TrackingState` (Enum) |
| 37 | `OverlayConfig` |
| 145 | `BBox` |
| 183 | `Point` |
| 196 | `Detection` |
| 247 | `TelemetryData` |
| 278 | `TrackingData` |
| 315 | `CameraIntrinsics` |
| 364 | `FrameDetections` |

`FrameDetections` fields with defaults:

| Field | Type | Default |
| --- | --- | --- |
| `frame_id` | `int` | 0 |
| `detections` | `list[Detection]` | |
| `timestamp` | `float` | |
| `source_id` | `str` | `"drone_0"` |
| `fps` | `float` | 0.0 |
| `frame_w` | `int` | 0 |
| `frame_h` | `int` | 0 |
| `tracker_state` | `str` | `"idle"` |
| `overlay` | `OverlayConfig \| None` | `None` |
| `mode` | `str` | `"test"` |
| `hud_visible` | `bool` | `True` |
| `telemetry` | `TelemetryData \| None` | `None` |
| `tracking_data` | `TrackingData \| None` | `None` |
| `intrinsics` | `CameraIntrinsics \| None` | `None` |

`to_dict()` rounds `fps` to one decimal place. `from_dict()` applies defaults via
`.get()`. `intrinsics` is **omitted** from the payload when it is invalid (line 394).
`TrackingState` is the four-value enum `idle`, `selected`, `tracking`, `lost`.

Inbound commands are handled by `SGCCommandReceiver` in
`jetson/communication/sgc_receiver.py`, with matching logic in `_find_best_match`, both
imported at `autonomous_drone_main.py:67` and dispatched at
`_handle_sgc_command` (`autonomous_drone_main.py:885`).

### Inbound command protocol (SGC → drone)

One JSON object per UDP datagram to port 9002. The only required field is `type`; the drone
parses unknown types without complaint, and every accepted command is echoed on the drone console
as `[SGC] ...`.

| `type` | Extra fields | What the drone does |
| --- | --- | --- |
| `select_target` | `bbox` `[x, y, w, h]`, `class_name`, `confidence`, `frame_w`, `frame_h` | matches the box against live detections (IoU ≥ 0.3) and selects it |
| `deselect_target` | — | clears the selection (keeps the lost banner state) |
| `follow_start` | same as `select_target`, `bbox` optional | runs the takeoff preflight, then starts person-follow |
| `follow_stop` | — | stops following and holds position |
| `arm` | — | arms the vehicle in GUIDED (checks GPS fix ≥ 3 and EKF); same as the on-screen ARM button |
| `takeoff` | — | climbs to `MAX_ALT` — **refused unless the vehicle is already armed**, so send `arm` first |
| `land` | — | stops following, clears the target and lands; the app keeps running |
| `disarm` | — | disarms — **refused above 0.5 m altitude**, so send `land` first |
| **`panic_rtl`** | — | **panic RTL: immediate return to launch, then the app stops streaming and exits** |
| `servo` | `channel` (optional, 1–16), `pulse` in µs **or** `angle` in degrees | one servo command; `angle` is converted with `1500 + angle * 500/45`; both forms are clamped to 1000–2000 µs. Omit `channel` to use the gimbal channel detected from the autopilot (see [Servo channel detection](#servo-channel-detection)) |

Minimum payloads:

```json
{"type": "arm"}
{"type": "takeoff"}
{"type": "land"}
{"type": "disarm"}
{"type": "panic_rtl"}
{"type": "follow_start", "bbox": [320, 240, 120, 260], "class_name": "person"}
{"type": "servo", "pulse": 1600}
{"type": "servo", "channel": 9, "angle": 20}
```

> **SGC team: `takeoff` no longer arms.** Arming and taking off are separate controls now, matching
> the drone UI. Send `{"type": "arm"}`, wait for the vehicle to report armed (the drone console
> prints `Vehicle armed`), then send `{"type": "takeoff"}`. A `takeoff` on a disarmed vehicle is
> refused with `Takeoff refused: vehicle is not armed — press ARM first` and nothing flies.

> **SGC team: landing needs two commands.** There is no auto-disarm — send `{"type": "land"}`, wait
> for touchdown, then send `{"type": "disarm"}`. `disarm` is refused while the vehicle is more than
> `DISARM_MAX_ALT` (0.5 m) above its launch altitude, and refused outright when the altitude cannot
> be read, because a disarmed drone falls. `land` deliberately does **not** stop the app: telemetry
> keeps streaming so the SGC sees the vehicle settle, and the receiver stays open for `disarm`.

`panic_rtl` is exactly what the on-screen `P` key does — it shares one code path,
`_trigger_panic_rtl` (`autonomous_drone_main.py:869`), so both sources issue `drone.send_rtl()`,
stop the detection stream, the command receiver and the camera, and exit. There is **no
acknowledgement**: the drone console shows `[PANIC] RTL triggered by SGC command`, and on the SGC
side the detection stream simply goes quiet — that is the acknowledgement.

Any host that can reach UDP 9002 can send these commands, including `arm`, `takeoff`, `land`,
`disarm` and `panic_rtl`.
Keep the port on the trusted bench network only.

### Delivery semantics

- **Queued, not dropped.** Commands go into a FIFO of up to `MAX_PENDING_COMMANDS` (32) and the
  main loop drains the whole queue every frame, so a burst is never partially lost — no command
  is overwritten before it runs. (An earlier single-slot mailbox silently overwrote unconsumed
  commands; a `panic_rtl` could be lost behind a stray servo command.)
- **Queued does not mean concurrent.** The queue guarantees *delivery*, not *sequencing*, and the
  arm/takeoff/land/disarm handlers all go through `_run_flight_command`, which allows one vehicle
  command at a time. A second one in the same batch is refused with
  `Another command is still running` rather than interleaved with the first. SGC must therefore
  **wait for the state to change in telemetry** before sending the dependent action: `arm` →
  poll for `armed == true` → `takeoff`, and `land` → poll for `height < 0.5 m` → `disarm`
  (`DISARM_MAX_ALT`). Sending `land` and `disarm` back to back always loses the disarm, because
  disarming while airborne is refused by design.
- **`panic_rtl` jumps the queue.** A drained batch is sorted so `panic_rtl` runs first, and the
  loop stops processing the rest and exits — a safety command is never stuck behind cosmetic ones.
- **A malformed packet can never take the loop down.** Each command is dispatched inside
  `try`/`except`; a bad packet is logged as `Bad SGC command '<type>': <reason>`, the HUD shows
  `SGC command rejected`, and the remaining commands in the batch still run.
- **Unknown or empty `type`** is reported as `Unknown SGC command: <type>` rather than being
  ignored, so an SGC typo shows up on the drone console instead of looking like a dead link.
- **`bbox` is `[x, y, w, h]`** — the same form the drone sends in `detections[].bbox`
  (`BBox(x, y, width, height)`), so a box can be echoed straight back. It is converted to corners
  before matching. A malformed bbox (wrong length, non-numeric) logs a warning and simply does not
  match.
- **Servo values are validated and clamped** to 1000–2000 µs whether they arrive as `pulse` or
  `angle`, and `channel` must be an integer in 1–16 — otherwise the command is refused with
  `Servo refused: <reason>`.
- **Servo commands are verified.** After sending, the expected pulse is compared against the vehicle's
  own `SERVO_OUTPUT_RAW` report. A mismatch logs `Servo chN commanded Xus but the vehicle reports Yus`
  and names `SERVOx_FUNCTION` and `SERVOx_MIN/MAX` as the things to check. Only readings that arrived
  *after* the command count, so a leftover report from the previous command cannot fake a fault. For
  a mapped output the expected value is predicted through the same scaling ArduPilot applies (see
  below) rather than assuming the pulse comes back unchanged.
- **Outputs the autopilot will not drive are refused, not ignored.** Assigning `SERVOx_FUNCTION` to a
  mount axis makes ArduPilot's own `MAV_CMD_DO_SET_SERVO` handler reject the channel
  (`Channel N is already in use`), because the mount backend owns it. The command is refused with the
  reason and the parameter change that would fix it. A mapped output whose RC input belongs to the
  aircraft is refused for the same reason — see
  [the mapped input must be spare](#the-mapped-input-must-be-spare-and-that-is-your-call-to-make).

#### Servo state in the telemetry packet

Commands arrive over UDP and are never acknowledged, so the detection packet on port 9001 carries the
outcome of the last one under `telemetry.servo`. No new socket, no new message type:

```json
"servo": {
  "detected_channel": 9,
  "source": "auto",
  "refused": false,
  "refusal_reason": null,
  "reported_pwm": 1600,
  "limits_hit": false
}
```

| Field | Meaning |
| --- | --- |
| `detected_channel` | channel detection resolved to, or `null` when nothing was detected or the command was refused |
| `source` | `"auto"` (detected) or `"explicit"` (operator/SGC/CLI supplied) |
| `refused` | the autopilot or the app would not accept the command |
| `refusal_reason` | why, as the app would print it — `null` when accepted |
| `reported_pwm` | pulse **read back from the FCU**, not the commanded value; `null` before anything is reported |
| `limits_hit` | the output landed somewhere other than the mapped prediction, which is what a servo on its `SERVOx_MIN`/`SERVOx_MAX` end stop looks like |

Every key is always present, so a display can bind to the shape once; unknown values are `null` or
`false` rather than absent. The block updates with the existing telemetry, so it arrives at whatever
rate the detection stream already runs.

### Servo channel detection

The gimbal channel is **detected from the flight controller**, not hardcoded. At connect time the
app requests the `SERVO_OUTPUT_RAW` and `RC_CHANNELS_RAW` streams and reads the `SERVOx_FUNCTION`
parameters, then prints what it found:

```
[SERVO] Output functions: SERVO9=[147] RC Input 8 (mapped)
[SERVO] Camera gimbal detected on channel 9 (RC Input 8 (mapped))
[SERVO] SERVO9 follows RC input 8 with matched limits (1000, 1500, 2000) - commanded pulses pass through unchanged.
```

Detection picks the lowest-numbered output assigned a mount axis, preferring pitch, then yaw, then
roll. Mount deploy/retract and the lens controls (ISO, aperture, focus, shutter) are recognised but
never selected — steering those does not aim the camera.

A detected mount axis is reported but **refused** for driving, because `MAV_CMD_DO_SET_SERVO` cannot
reach it; assigning `SERVOx_FUNCTION = 0` is what makes a gimbal output drivable.

Three ways to influence the result, in priority order:

1. `channel` in the SGC `servo` command, for when the SGC knows its own wiring.
2. `--servo-channel N`, which pins the channel for the whole run and overrides the SGC. Also
   settable in `run_config.json` as `"servo_channel": 9`.
3. Otherwise, whatever was detected. If nothing was detected the command is **refused** rather than
   sent to a guessed channel — driving an arbitrary output risks moving a motor or a control surface.

One further flag applies to *how* the channel is driven, not which: `--allow-rc-override`, also
settable as `"allow_rc_override": true` in `run_config.json`. Off by default; see
[the mapped input must be spare](#the-mapped-input-must-be-spare-and-that-is-your-call-to-make).

### Configuring the autopilot: `k_rcinN_mapped`

A gimbal output on an aux channel is **not** drivable by direct PWM, and this is not a limitation of
this app. ArduPilot's `MAV_CMD_DO_SET_SERVO` handler only accepts a whitelist of functions
(`AP_ServoRelayEvents::do_set_servo`): unassigned outputs, manual pass-through, sprayer, gripper. An
output assigned a mount function falls through to `default:` and is refused, because the mount
backend is expected to own it.

The way to make an aux output drivable is to assign it to an RC input instead:

| `SERVOx_FUNCTION` | Meaning | Driven by |
| --- | --- | --- |
| `0` | unassigned | `MAV_CMD_DO_SET_SERVO` — no RC input, no consent needed |
| `7` / `6` / `8` | mount pitch / yaw / roll | mount protocol only — **not** direct PWM |
| `51`–`66` (`k_rcinN`) | follows RC input N, unchanged | `RC_CHANNELS_OVERRIDE` on input N — needs `--allow-rc-override` |
| `140`–`155` (`k_rcinN_mapped`) | follows RC input N, rescaled | `RC_CHANNELS_OVERRIDE` on input N — needs `--allow-rc-override` |

An aux output is reachable either way, but the two routes are not equivalent. `SERVO9_FUNCTION = 0`
makes the app drive the output directly and touch no RC input at all — that is the recommended setup.
`SERVO9_FUNCTION = 147` means "output 9 follows RC input 8", which requires an override on input 8:
`RC_CHANNELS_OVERRIDE` addresses RC **inputs** and carries only 8 slots, so an output above channel 8
must be mapped to an input inside that range. That input has to be genuinely spare, which the app
cannot establish for you.

#### The mapped input must be spare, and that is your call to make

`RC_CHANNELS_OVERRIDE` writes RC **inputs**, and ArduPilot reads some of them straight into flight
control. Writing a servo pulse into one of those does not move a camera — it commands the aircraft.

The app refuses inputs it can prove are unsafe:

- inputs 1–5 are always refused (throttle/roll/pitch/yaw/mode on ArduCopter, read from the RC stream
  directly; no output assignment can free them);
- inputs 6–8 are refused when output N — which input N feeds under the default input→output mapping —
  is assigned anything else;
- an output whose `SERVOx_FUNCTION` has not been read counts as reserved. An unknown is not a safe.

**But passing those checks is not sufficient, so every RC override requires `--allow-rc-override`.**
`SERVOx_FUNCTION` records which input an output *follows*; it says nothing about which inputs
`AP_Copter` reads for flight control, and that depends on the channel mapping and the airframe. A
bench once mapped `SERVO9_FUNCTION = 145` (RC input 6), passed every static check the app had, and
sweeping it walked the aircraft through RTL, STABILIZE, AUTO, CIRCLE and LAND — input 6 was the mode
channel on that vehicle and nothing in the parameter map admitted it. Only you know your own wiring,
so the override is opt-in:

```
[SGC] Servo refused: Servo ch9 is k_rcin6_mapped, so driving it needs
RC_CHANNELS_OVERRIDE on RC input 6. That input passes every check this app can make statically, but
RC_CHANNELS_OVERRIDE writes RC inputs and ArduPilot reads some of them straight into flight control
- including the mode channel, whose position depends on the channel mapping and frame, not on
SERVOx_FUNCTION. Refusing rather than sweeping the aircraft through its flight modes. If you have
confirmed RC input 6 is spare on this vehicle, re-run with --allow-rc-override; ...
```

When you do opt in, the app also watches the mode channel after **every** write. ArduPilot holds an
override until it is explicitly released, so a bad one stays bad; if the flight mode moves inside the
2.5 s window following a write, the app releases the override and reports the refusal:

```
[SGC] Servo refused: Servo ch9 wrote RC input 6 and the vehicle mode changed STABILIZE -> RTL
straight afterwards, so that input is flight control, not a spare. Override released. ...
```

The watchdog is not the only release. Any transition out of normal operation — `land`, `disarm`,
RTL, and the shutdown path — releases every outstanding override first
(`SITL: RC override released (LAND)`), because an override left latched survives the app's own exit
and would keep pinning that RC input for the rest of the flight.

To confirm which inputs your vehicle actually reads, look at the mode slots ArduPilot prints at
startup — `[SERVO] Output functions` — or run `mode_channel()`'s inputs: 1–8, and the mode channel is
the one carrying the FLTMODE assignment.

**The override-free alternative.** A gimbal does not need an RC path at all. Set
`SERVO9_FUNCTION = 0` and the app drives it with `MAV_CMD_DO_SET_SERVO`, which addresses the output
directly and touches no RC input. That is the recommended setup for a camera.

Two further consequences worth planning for:

- **A mapped output is a pass-through, not a gimbal controller.** ArduPilot provides no stabilisation
  through it, and `SERVOx_FUNCTION` no longer says "gimbal", so detection has to infer the channel
  from the wiring. A single mapped output is accepted; several are ambiguous and refused rather than
  guessed at.
- **A mapped output rescales its input.** ArduPilot normalises the RC value against that channel's
  own MIN/TRIM/MAX, scales to ±4500, then converts back through the *output*'s MIN/TRIM/MAX
  (`SRV_Channel::output_ch`). So a pulse is not echoed back unchanged: with RC8 at 1000/1500/1800 and
  SERVO9 at 1100/1500/1900, commanding 1750 µs makes the vehicle report 1833 µs. Set
  `SERVOx_MIN/TRIM/MAX` and `RC{N}_MIN/TRIM/MAX` to the **same** values for a 1:1 mapping, which is
  what the app assumes when it checks the read-back. Connect logs a warning when they differ.

The function numbers are ArduPilot `SRV_Channel::Function` values, **not** MAVLink
`MAV_SERVO_FUNCTION` — the two schemes collide (MAVLink's `MAV_SERVO_FUNCTION_GIMBAL_ROLL` is 26,
which is ArduPilot's *ground steering*), so the code only ever interprets the ArduPilot table.

To try a command from a laptop without running the SGC:

```bash
python tools/send_sgc_command.py --host <drone-ip> --type panic_rtl
python tools/send_sgc_command.py --host <drone-ip> --type follow_start --bbox 320 240 120 260 --class-name person
```

---

## Tracker overlay specification

This section reproduces the drawing contract implemented in `modules/display.py`. All drawing
uses `cv2.LINE_AA` and `cv2.FONT_HERSHEY_SIMPLEX`. Colours are given as **BGR** triples with
the equivalent CSS hex; the hex column is the reverse conversion of the BGR triple read as
RGB, which is why `#28b2ff` and `(255,178,40)` are the same colour. Do not "correct" one to
match the other.

### 1. Geometry and cover-crop

The video is the raw native frame; bounding boxes are already in native frame coordinates
and are drawn 1:1. When the display band and the frame aspect ratios differ, a **cover crop**
is required — never letterbox or fit-with-padding. A black bar on the right side of the
picture is the specific failure this rule exists to prevent.

`set_video_geom` in `modules/display.py`:

```
scale = max(band_w / frame_w, band_h / frame_h)
cw, ch = round(frame_w * scale), round(frame_h * scale)
sx = max(0, (cw - band_w) // 2)
sy = max(0, (ch - band_h) // 2)
canvas_video = resized[sy:sy + band_h, sx:sx + band_w]
```

When `frame_w/frame_h` equals the band ratio (960/720) the frame is drawn 1:1. The container
must resize on **both** axes (`X_AXIS|Y_AXIS` in SWT; `flex:1 1 auto` plus `width/height:100%`
on the web) with zero fixed size, pinned to all four edges. Resize contract:

1. Canvas width = container width; canvas height = client height minus chrome.
2. Never a fixed pixel size; never `fill` inside an auto-sized parent; never anchor the
   top-left of a non-resizing parent.
3. Repaint the whole canvas on every resize — never scale a stale buffer.
4. The canvas is never smaller than the video picture.

### 2. Detection boxes — `draw_detection_window`

All detections are drawn every frame; non-selected boxes are never removed.

**Selected** — accent BGR `(255,178,40)` / `#28b2ff`:

- Glow `1.0 + 0.5 * pulse(5.0)` where `pulse(t) = 0.5 + 0.5 * sin(now * t)`; one 1-px loop
  is ≈ 0.63 s. Clamp per channel at 255.
- Outer corner brackets on the bbox rect: length 18, thickness 3.
- Inner brackets inset 3 px (`x1+3, y1+3, x2-3, y2-3`): length 10, thickness 1.
- Label `"{class}  {conf:.0f}%"`, font 0.45/1, background `(38,26,8)` / `#081a26`, text
  `(255,224,150)` / `#96e0ff`, 1-px accent border, capsule radius `card_height / 2`, text
  offset `(x1+4, baseline-4)`.
- Confidence bar under the label, 2 px: track `(60,66,82)` / `#52423c`, fill
  `(40,200,120)` / `#78c828`, `bar_w = max(14, int(text_width * 0.6))`.
- Static reticle at the bbox centre, radius 12.

`_label_anchor` resolution order:

1. Above — baseline `y1 - 4`, accepted only if `above - label_h - 8 >= 3`.
2. Below — `min(y2 + 14, img_h - 14)`.
3. Inside — `y1 + 8`.

**Non-selected** — dimmed:

- Fill `(70,60,30)` / `#1e3c46` blended over the video at alpha **0.30**.
- Edge rect `(140,110,70)` / `#466e8c`, thickness 1.
- Corner brackets length 8, thickness 1.
- Label = class name only, font 0.38/1, background `(44,38,22)` / `#16262c`, text
  `(210,200,180)` / `#b4c8d2`, capsule.

**Fixed centre reticle** — always at `(w/2, h/2)`, BGR `(70,130,175)` / `#af8246`, radius
**22**.

### 3. Reticle

Static; there is **no** rotating scan ring (an earlier dotted scan ring was removed).
`gap = 5`, `size = 11`, `thickness = 1`. Four arms: left `(cx-11,cy)→(cx-5,cy)`, right
`(cx+5,cy)→(cx+11,cy)`, top `(cx,cy-11)→(cx,cy-5)`, bottom `(cx,cy+5)→(cx,cy+11)`. Filled
centre dot radius 2. Ring stroke radius is **22** for the fixed centre reticle and **12**
for a selected object.

> Spec/implementation drift: `modules/display.py:829` calls `radius=22`, but the function
> signature default at `modules/display.py:694` is `radius=26`. Any other caller silently
> gets the wrong radius.

### 4. Tracking overlay — `draw_target_tracking`

Drawn only while following.

**Leader line** from the centre to the target: **22 dashes**, `t = i / 22`, thickness
`max(1, 3 - 2t)` (tapering 3→1), colour lerped from `tracking.accent (40,225,125)` /
`#7de128` toward `(60,200,120)` / `#78c83c` as `c = accent*(1-t) + (60,200,120)*t`. Filled
circles only on **even** `i`.

**Target ring** — pulsing radius `8 + int(4 * pulse(4.0))` (8…12), thickness 2, accent
colour; a static outer ring at `radius + 8`, thickness 1, same colour; static filled centre
dot radius 3.

**Follow bar** — bottom centre, `y = h - 64`, pill shape, text:

```
FOLLOW {class}   {dist:.1f}m   {speed:+.2f}m/s   YAW {yaw:+.1f}   {conf:.0f}%
```

Font 0.34, auto-dropping to 0.30 if too wide, thickness 1. Background `(8,40,26)` /
`#1a2808`, text `(160,255,180)` / `#b4ffb4`, pad h12/v7, radius 14, accent outline
`(40,225,125)`, dot GREEN `(45,230,130)`. `dist` is `lidar_dist or vision_dist` taken from
`tracking_data`.

### 5. Chrome

Drawn only for the full desk window 960×864 = 84 header + 720 video + 60 footer. Header rows
0…83 gradient `(14,20,32)`→`(24,28,44)`; footer rows 804…863 `(24,28,44)`→`(14,20,32)`;
separator at `y = 84` in `(58,72,104)` / `#68483a` with a 1-px `(10,14,22)` line below;
default canvas fill `(10,12,18)`.

**5.1 Header pills** — all at `y = (84-38)//2 = 23` (vpad 7, so heights vary only by font).
Every pill and chip is a **true capsule** (radius = height/2); a rounded rect must be drawn
as a cut-corner octagon plus four corner circles. Layout: state pill left-anchored at
`x = 10`, then the Target pill (tracking only), then LIVE/OFFLINE, `gap = 10`; the **Mode
pill is centred separately** at `x ≈ (w - pill_w)/2`; the takeoff button stays top-right.

| Pill | Background | Foreground | Font | Pad | Outline |
| --- | --- | --- | --- | --- | --- |
| State | `STATE_THEME` bg | `STATE_THEME` fg | 0.58/1 | 12,7 | theme accent |
| Target | `(28,34,48)` / `#30221c` | `(235,238,245)` | 0.55/1 | 11,7 | `(64,76,106)` / `#6a4c40` |
| LIVE | `(10,54,32)` / `#20360a` | `(120,255,160)` / `#a0ff78` | 0.44/1 | 7,6 | `(30,160,80)` / `#50a01e` |
| OFFLINE | `(44,16,16)` / `#10102c` | `(255,150,150)` / `#9696ff` | 0.44/1 | 7,6 | `(200,40,40)` / `#2828c8` |
| Mode | `MODE_COLORS` bg | `MODE_COLORS` fg | 0.55/1 | 9,7 | `(70,80,120)` / `#785046` |

State text is `tracker_state.upper()` ∈ IDLE / SELECTED / TRACKING / LOST. Target text is
`"{class}  {conf:.0f}%"` and appears only while tracking.

`STATE_THEME` (bg hex, fg hex, accent hex):

| State | bg | fg | accent |
| --- | --- | --- | --- |
| `idle` | `#342210` → `(16,34,52)` | `#f5d796` → `(150,215,245)` | `#ffbe46` → `(70,190,255)` |
| `selected` | `#082130` → `(48,33,8)` | `#82d8ff` → `(255,216,130)` | `#28b2ff` → `(255,178,40)` |
| `tracking` | `#1a2a08` → `(8,42,26)` | `#a5f58c` → `(140,245,165)` | `#7de128` → `(40,225,125)` |
| `lost` | `#0c0a2e` → `(46,10,12)` | `#9696ff` → `(255,150,150)` | `#2828ff` → `(255,40,40)` |

`MODE_COLORS` (bg hex → BGR, fg):

| Mode | bg | fg |
| --- | --- | --- |
| TEST | `#322a2a` → `(42,42,50)` | `HUD_TEXT_DIM` `#b29e96` |
| sitl | `#50320a` → `(10,50,80)` | CYAN `#ffcd46` → `(70,205,255)` |
| flight | `#18370e` → `(14,55,24)` | GREEN `#82e62d` → `(45,230,130)` |

**5.2 Arm button** — top-right, `110×30`, `bx = w - 398`, `by = 23`, radius 15. Disarmed
(clickable): fill `(14,34,56)` / `#38220e`, border `(70,180,255)` / `#ffb446`, dot CYAN, text
`"ARM"`. Armed (not clickable): fill `(12,52,30)` / `#1e340c`, border `(60,200,120)` / `#78c83c`,
dot GREEN, text `"ARMED"` and `HUD_TEXT_DIM` label. Dot at `(bx+16, by+15)` radius 3; text font
0.42/1 in `(235,238,245)`.

**5.3 Takeoff button** — top-right, `150×30`, `bx = w - 278`, `by = 23`, radius 15. Armed
(clickable): fill `(12,46,32)` / `#202e0c`, border `(60,200,120)` / `#78c83c`, dot GREEN. Text
is always `"TAKEOFF 5m"`. Disarmed: fill `(26,28,34)` / `#221c1a`, border `(70,78,92)` / `#5c4e46`,
dot and label `HUD_TEXT_DIM` — i.e. the control is visibly disabled. Dot at `(bx+16, by+15)`
radius 3; text font 0.42/1 in `(235,238,245)`.

**5.4 Land button** — top-right, rightmost, `110×30`, `bx = w - 118`, `by = 23`, radius 15.
Armed (clickable): fill `(16,40,66)` / `#422810`, border `(70,160,250)` / `#faa046`, dot CYAN,
label `"LAND"` in `(235,238,245)`. Disarmed: fill `(26,28,34)` / `#221c1a`, border `(70,78,92)` /
`#5c4e46`, dot and label `HUD_TEXT_DIM` — landing a disarmed vehicle would only be a mode change,
so the control is disabled until the vehicle is armed.

The three controls are laid out right-to-left with `_BUTTON_GAP = 10` and an 8 px window margin,
so they occupy `x = w-398 … w-8` and never reach the centred mode pill (which ends at
`x ≈ 480`):

| Button | Rect (`by = 23`, height 30) |
| --- | --- |
| ARM | `(562, 23, 672, 53)` |
| TAKEOFF 5m | `(682, 23, 832, 53)` |
| LAND | `(842, 23, 952, 53)` |

Clicking a button calls `_handle_arm_button` / `_handle_takeoff_button` / `_handle_land_button`;
the hit tests use `get_arm_button_rect()` / `get_takeoff_button_rect()` / `get_land_button_rect()`.
**Takeoff is refused unless `drone.is_armed()`** — the message is
`"Takeoff refused: vehicle is not armed — press ARM first"`. Arming itself checks GPS fix ≥ 3 and
EKF convergence, and the three actions call `control.arm()`, `control.takeoff(MAX_ALT)` and
`control.land()` (the old combined `control.arm_and_takeoff` is kept for the mock/backend API).

**The vehicle calls run off the render thread.** Arming, the takeoff climb and disarming all block
while the FCU confirms — `sitl.takeoff()` alone polls the climb for up to 30 s. Called inline from
the mouse callback (which runs inside `main_loop`) that froze the camera window and stalled the
detection stream for the whole climb, which looked like the video had paused. Each handler now does
its fast refusal checks inline and hands the blocking part to `_run_flight_command`, which runs it
on a `flight-cmd` worker thread and reports the result through the HUD:

| Phase | Thread | What the operator sees |
| --- | --- | --- |
| refusal checks (armed, GPS, EKF, altitude) | main | immediate red/amber HUD message, nothing sent |
| `control.arm()` / `takeoff()` / `land()` / `disarm()` | `flight-cmd` worker | cyan `Taking off to 5m...` while it runs, then green success or red failure |
| FCU confirmation | MAVLink listener thread | unchanged — it always ran separately |

`_run_flight_command` also holds `_command_in_flight`, so a second click while a command is running
is refused with `Another command is still running` instead of sending a duplicate MAVLink command.

`modules/drone_backend/sitl.py` no longer uses pymavlink's `motors_armed_wait()` /
`motors_disarmed_wait()`. Both are `while True: wait_heartbeat()` loops **with no timeout**, so a
vehicle that never arms (failed pre-arm check, no RC) would hang the caller forever. They are
replaced by `_wait_for_arm_state(expected, timeout=20.0)`, which polls the cached heartbeat and
fails with the vehicle's own status text — e.g. `Vehicle did not report armed within 20s (status:
Pre-arm: EKF not ready) - check the pre-arm checks`.

### Takeoff altitude is an absolute MSL altitude

`MAV_CMD_NAV_TAKEOFF` **param7 is absolute altitude above sea level**, not a height above the
launch point. `sitl.takeoff(max_height)` therefore converts:

```python
current_amsl = _cached_alt          # GLOBAL_POSITION_INT.alt, metres AMSL
target_amsl  = current_amsl + max_height
```

Sending the raw relative height (the old behaviour) asks a vehicle sitting at, say, 412 m MSL to
climb to **5 m MSL** — i.e. to fly into the ground. ArduPilot accepts the command, the vehicle
starts descending, a failsafe fires and the FCU reports `Disarming motors`, which surfaced as the
misleading `Vehicle disarmed during takeoff`. It appeared to work in the original SITL session only because that SITL's home sat near 0 m MSL, so
the two frames coincided. The current SITL instance's home is **584.1 m AMSL**, where the old code
would have commanded a descent into the ground, so the frame bug is reproduced there. At takeoff the
vehicle is on the ground, which makes `current_amsl + max_height` equal to "max_height above the
launch point" **and** "max_height above home", so the existing `max_height` semantics are unchanged.

A descent guard now fails fast instead of waiting for the FCU to kill the motors:

```text
Takeoff commanded +5.0 m to 417.0 m AMSL but the vehicle is descending (-1.0 m,
now 411.0 m AMSL). Check the NAV_TAKEOFF altitude frame and any altitude failsafe.
(status: Arming checks disabled | Disarming motors)
```

Every `STATUSTEXT` is kept in a short ring buffer, so failures quote the last few FCU messages
rather than a single line the autopilot overwrote.

### Takeoff only reports success once the altitude is actually held

`MAV_CMD_NAV_TAKEOFF` is a *climb* command, not a hold. On its own it gives ArduPilot no altitude to
settle on, and the vehicle can sail straight past the requested height. The follow controller cannot
supply one either: `send_movement_command_XYA()` sends
`POSITION_TARGET_TYPEMASK_Z_IGNORE`, and `set_flight_altitude()` only updates local state. So
`takeoff()` now also streams an explicit hold setpoint
(`SET_POSITION_TARGET_LOCAL_NED`, `MAV_FRAME_GLOBAL_RELATIVE_ALT`, x/y/roll/pitch/yaw ignored) and
reports success only after the altitude stays within 1 m of the target for 2 s.

This matters because the old code returned as soon as it saw 95 % of the requested climb, which
produced the worst possible failure: a confident

```text
Airborne — holding at 5m
```

while the vehicle was at 50 m and still climbing. A genuine overshoot — or an autopilot that is not
holding altitude at all — is now an explicit failure, with a hard ceiling of
`target + max(3 m, max_height)` above which the climb is treated as runaway:

```text
Vehicle is climbing away: 1000.0 m above launch (1584.1 m AMSL) and still rising, past the
594.1 m ceiling. The autopilot is not holding the takeoff altitude. Check the altitude
controller tuning and whether anything is sending position targets.
```

If the vehicle stops short of the ceiling but never settles, takeoff times out and says so:

```text
Takeoff did not settle at 5.0 m after 60s (at 8.1 m, 592.2 m AMSL). (status: Arming motors)
```

> **On the provided SITL instance, `takeoff()` currently reports that second failure.** The autopilot
> does arrest the runaway climb once a setpoint is streamed, but it captures the altitude with a
> ~3 m overshoot that does not converge (a 5 m target settles around 9 m, a 1 m target around 3 m).
> That is an altitude-controller tuning problem in the SITL vehicle configuration, not a command
> framing problem — `MAV_CMD_DO_CHANGE_ALTITUDE` and relative-altitude position targets were both
> measured against it. Tune `FS_BATT/INS_*`-era defaults that apply, chiefly the altitude
> controller (`ATC_`/`INS_` gain set and `WPNAV_` defaults for the SITL model) before using this SITL
> instance to validate flight behaviour. The app-side guarantee does not depend on that tuning: it
> either holds the altitude or says it did not.

### Flight-mode changes are confirmed, not assumed

`_set_mode()` used to send the mode, call `recv_match()` to "wait" for the reply, and then cache the
requested mode unconditionally. `recv_match()` on the shared connection races
`_message_listener()`, which already consumes `HEARTBEAT`s — so the wait usually consumed a
heartbeat the listener needed, and the mode was cached whether or not the autopilot accepted it.
`_set_mode()` now serialises on `_command_lock` and polls the listener's cached mode, raising if the
vehicle never arrives:

```text
Vehicle did not enter GUIDED within 5s (still reporting LOITER). Check that the mode is
available and enabled.
```

`arm()` therefore no longer arms a vehicle that never entered GUIDED.

The transport also had to change. `mav.set_mode_send()` was **silently ignored** by the ArduCopter
build under test: the vehicle kept reporting RTL while the app believed it had switched, and a probe
that forced the issue found the vehicle falling into ACRO. `_set_mode()` now uses
`MAV_CMD_DO_SET_MODE` (`MAV_CMD_DO_SET_MODE` is accepted and takes effect; the same change made ARM
and LAND work on that vehicle), waits for its `COMMAND_ACK`, and then still confirms the mode from
the listener's cache. A rejected mode switch raises:

```text
Vehicle rejected the request to enter GUIDED (result 1)
```

### ARMING_CHECKS

`connect_drone()` reads `ARMING_CHECKS` on a daemon thread and prints a `[SAFETY]` warning when it
is `0` (all arming checks disabled) or `1`. The Python-side GPS/EKF gates in the UI are **not** a
substitute for the autopilot's own arming checks — those are the last line of defence on a real
vehicle. `ARMING_CHECKS=0` means nothing verifies throttle-at-zero, GPS, EKF or compass before the
motors spin.

`_handle_land_button()` stops person-following, clears the target selection and then calls
`control.land()` — the app keeps running, so the camera, tracker, SGC link and telemetry stay up
and the vehicle can be disarmed from the same window. `LAND` has a keyboard equivalent, `L`.

**5.5 Disarming** — no button: the header has no room for a fourth control without colliding with
the centred mode pill, and disarming is a ground-only action. It is on the `D` key and the
`disarm` SGC command. `_handle_disarm_action()` refuses unless the vehicle is within
`DISARM_MAX_ALT` (0.5 m) of the recorded home altitude, because a disarmed drone falls:

| Situation | Result |
| --- | --- |
| already disarmed | `"Vehicle is already disarmed"`, nothing sent |
| altitude above 0.5 m | `"Disarm refused: airborne at 12.0m (limit 0.5m) - land first"`, nothing sent |
| altitude unreadable | `"Disarm refused: cannot read altitude - land first"`, nothing sent |
| on the ground, following | follow stopped, then `control.disarm()` |
| on the ground | `control.disarm()` → `"Vehicle disarmed - motors are off"` |
| FCU rejects the command | `"Disarm failed: <reason>"` from the `COMMAND_ACK` |

**5.6 Footer chips** — `y = h - 60 + (60-24)//2 = h - 42`, chip height 24, gap 8, centred
(8 chips ≈ 617 px, so they still fit the 960 px window): `ESC`/deselect, `SPACE`/follow,
`R`/reset, `H`/hud, `L`/land, `D`/disarm, `P`/panic RTL, `Q`/quit. Chip background
`(30,36,50)` / `#32241e`, capsule radius `ch/2`. Key box background `(52,62,88)` / `#583e34`,
capsule radius `(ch-2)/2`, key text `(220,235,255)` / `#ffebdc` font 0.50/1, value text
`HUD_TEXT_DIM` `(150,158,178)` font 0.40/1.

### 6. Prompts and banners

**6.1 Selection prompt** at `y = h - 64`: `"CLICK - SELECT TARGET     {N} OBJECTS"`,
background `(12,36,52)`, text CYAN `(70,205,255)`, font 0.46/1, pad 12/7, radius 14, outline
`(40,120,170)` / `#aa7828`, dot CYAN.

**6.2 Follow prompt** at `y = h - 64`: `"SELECTED - {class}    [SPACE] FOLLOW"`,
background `(48,33,8)` / `#082130`, text `(255,218,130)` / `#82d8ff`, font 0.46/1, pad 12/7,
radius 14, outline AMBER `(255,190,60)`, dot AMBER.

**6.3 Lost banner** — full-width strip, `y ≈ 4…44`. Alpha
`0.42 + 0.18 * |((now * 4) % 2) - 1|` over background `(36,4,10)` / `#0a0424`, radius 6;
diagonal stripes `(90,8,10)` scrolling at alpha 0.35 across the whole band; border RED
`(255,70,70)`; bottom edge `(60,8,10)`. Centred text `"TARGET LOST"` at font 0.58 with
thickness **2**, fill `(255,235,235)` / `#ebebff`, stroke `(255,120,120)` / `#7878ff`.
Optional RTL pill `"RTL {s}s"` / `"RTL ACTIVE"` with background `(70,8,8)`, text
`(255,190,190)` / `#bebeff`, outline RED. A `DISMISS` button sits at the right: radius 12,
background `(52,52,62)`, border `(130,130,150)` / `#968282`, text `(220,220,230)` / `#e6dcdc`.

### 7. Colour reference

| Name | BGR | Hex |
| --- | --- | --- |
| `HUD_TEXT` | `(235,238,245)` | `#f5eeeb` |
| `HUD_TEXT_DIM` | `(150,158,178)` | `#b29e96` |
| `HUD_ACCENT` | `(0,190,235)` | `#ebbe00` |
| `GREEN` | `(45,230,130)` | `#82e62d` |
| `CYAN` | `(70,205,255)` | `#ffcd46` |
| `AMBER` | `(255,190,60)` | `#3cbeff` |
| `RED` | `(255,70,70)` | `#4646ff` |
| selected accent | `(255,178,40)` | `#28b2ff` |
| other-edge | `(140,110,70)` | `#466e8c` |
| centre-reticle | `(70,130,175)` | `#af8246` |
| tracking accent | `(40,225,125)` | `#7de128` |

### 8. Payload → render mapping

| Payload field | Rendered as |
| --- | --- |
| `detections[]` | use `is_selected`. A per-detection `bbox_color` exists, but the section 7 constants win — payload colour is an override only |
| `fps`, `mode`, `tracker_state` | header / mode pill |
| `overlay.counter_text` | object count; on-drone it appears only in the selection prompt, optionally top-left |
| `overlay.prompt_text` | prompt banner |
| `overlay.show_tracking_bar` + `overlay.tracking_bar_text` | follow bar |
| `overlay.show_lost_banner` | lost banner |
| `frame_w` / `frame_h` | canvas size |
| `hud_visible` | global show/hide |

### 9. What this specification does not define

The overlay spec supplies **no JSON schema, no field types, no message envelope, no port
numbers, and no state machine** — only the four-value `tracker_state` enum and a
name-level field mapping. The sections 5.1/6 prompt rules imply the transitions
`idle → selected → tracking → lost`, but that state machine is not written down anywhere.
The authoritative sources are `shared/detection_models.py:363-419` (schema),
`shared/detection_transport.py` (transport) and `autonomous_drone_main.py:1382`
(`args.sgc_cmd_port`). This is the largest documentation gap in the project; a client cannot
be implemented against the overlay spec alone.

---

## Camera calibration

Two paths.

**Automatic** — `--auto-calibrate` runs `AutoCalibrator` (`modules/auto_calibrate.py`) in a
background thread, detecting a chessboard and solving the intrinsics. `--chessboard 9x6`
sets the inner-corner count (default). The target image is
`benchmarks/camera_calibration/checkerboard_9x6.png`, printable via
`tools/print_checkerboard_page.py`. Capture helpers:
`tools/capture_calibration_images.py`, `tools/camera_mirroring_diagnostic.py`.
`tools/calibrate_camera.py` runs calibration interactively.

**Manual** — `--intrinsics-fx/--intrinsics-fy/--intrinsics-cx/--intrinsics-cy` with
`--intrinsics-width/--intrinsics-height` (default 640/480).

The default when nothing is supplied is `fx = fy = 446.7`, `cx = 320.0`, `cy = 240.0` at
640×480, loaded from `benchmarks/distance/manifest.json` — and byte-identical to the hardcoded
fallback `DEFAULT_CONFIGURED_INTRINSICS` in `calibration.py:21-28`, so deleting the manifest
changes nothing. **These are not a chessboard calibration; they are a placeholder derived from
10 desktop images using the estimator's own equation.** Treat them as wrong for the Orin Nano
and recalibrate. See
[the circular-fit warning](#circular-fit-warning-the-0133-m-baseline-is-not-valid-accuracy).

---

## Video streaming

`jetson/streaming/rtsp_server.py` serves the camera feed on `--rtsp-port` (default 8554)
and exposes an MJPEG fallback used on Windows. `--jpeg-quality` (default 30) trades quality
for latency; lower is faster. On Jetson the intended path is hardware H.264 via GStreamer
`nvv4l2h264enc`.

---

## Testing

```bash
python -m pytest -q
```

Current result: **355 passed, 315 subtests passed**.

Tracked test files:

| File | Covers |
| --- | --- |
| `tests/test_servo_mavlink.py` | servo command routing, discovery-cache refresh, consent gates, RC-override watchdog, pulse verification |
| `tests/test_servo_channels.py` | `plan_drive`, `parse_servo_functions`, gimbal-channel picking, drive constants |
| `tests/test_link_selection.py` | `--drone-link` precedence, real-FCU fallback, endpoint recognition |
| `tests/test_sgc_receiver.py` | SGC command parsing, queueing, validation, `panic_rtl` priority |
| `tests/test_app_config.py` | configuration constants |
| `tests/test_distance_horizontal.py`, `tests/test_h_follow_state_machine.py`, `tests/test_horizontal_live_wiring.py`, `tests/test_selected_diagnostic.py` | distance estimation and follow-state wiring |

The suite is the regression gate for the servo/gimbal work: `TestStaleDiscoveryCache` in
`tests/test_servo_mavlink.py` pins the fix for the stale-discovery-cache race where a channel
unknown at connect time was refused even after the FCU answered later.

Smoke checks worth running after any Jetson adaptation:

```bash
# entry point imports and reaches drone connection
python autonomous_drone_main.py --no-prompt --mode sitl
# distance estimator still produces the documented numbers
python tools/run_distance_benchmark.py --no-plots
# camera opens on the Jetson backend
python -c "import cv2; cap=cv2.VideoCapture(0, cv2.CAP_V4L2); print(cap.isOpened())"
```

> **Historical.** The suite was deleted during the deployment trim; the baseline at that point
> (`cd766f91`) was 227 passed / 2 failed, the two failures being the depth-backend import-laziness
> tests in the since-removed `tests/test_distance_estimator_depth.py`. The current suite is
> unrelated to that baseline — it was rebuilt for the servo, link-selection and SGC work and
> does not include the depth-laziness checks.

---

## Jetson deployment

### Target platform: Jetson Orin Nano, verified hardware facts

The earlier documents gave mutually contradictory figures (compute capability 9.0 vs 12.0 vs
`sm_120`, CUDA 12.0 vs 12.2, torchvision `0.26.0+cpu` vs `0.22.0.dev+cu124`). Those were
unverified. The table below is sourced from NVIDIA's published specifications.

| Property | Value |
| --- | --- |
| GPU | NVIDIA Ampere, **CUDA compute capability `sm_87`** |
| CUDA cores | 512 (4 GB) / 1024 (8 GB) |
| AI performance | 20 TOPS INT8 (4 GB) / 40 TOPS INT8 (8 GB) |
| CPU | 6× Arm Cortex-A78AE v8.2, up to 1.5 GHz |
| Memory | 4 GB / 8 GB LPDDR5 |
| OS | Ubuntu 22.04, Linux kernel 5.15 |
| Architecture | aarch64, Python 3.10 |

**`sm_87`, not `sm_120` or `9.0`.** The `sm_120` and compute-capability-9.0 figures in the old
documents were wrong.

### JetPack and CUDA matrix

| JetPack | Jetson Linux (L4T) | CUDA | TensorRT | cuDNN |
| --- | --- | --- | --- | --- |
| 6.0 | 36.2 | 12.2 | 10.3 | 9.0 |
| 6.1 | 36.4 | 12.6 | 10.3 | 9.3 |
| **6.2** | **36.4.3** | **12.6** | **10.3** | **9.3** |
| **6.2.1** | **36.4.4** | **12.6** | **10.3** | **9.3** |

> **The old guidance said `cu121`/`cu122` and was wrong for current JetPack.** `cu122` is only
> correct on JetPack 6.0. JetPack 6.1 and later — including 6.2/6.2.1, the current production
> releases — ship **CUDA 12.6**. Confirm with `cat /etc/nv_tegra_release` and
> `python3 -c "import torch; print(torch.version.cuda)"` before choosing a wheel.

JetPack 6.2 also adds **Super Mode** power profiles for the Orin Nano, which raise clocks and
memory bandwidth. On the 4 GB module the supported modes are 10 W / 25 W / MAXN SUPER; on the
8 GB module, 15 W / 25 W / MAXN SUPER. Select MAXN SUPER after installing for best
inference throughput. Note this requires a device currently on JetPack 6.x and re-flashing
with the matching flashing configuration.

### Install torch on the Jetson: the part that breaks

**PyPI publishes no aarch64 Linux wheels with CUDA support.** This means both of the commands
in the old deployment document fail on a Jetson:

```bash
pip install torch                                             # CPU-only x86 wheel, wrong arch
pip install torch torchvision --index-url https://pytorch.org/whl/cu121   # no aarch64 wheel
```

Use one of these instead.

**Option A — NVIDIA's Jetson aarch64 wheels (recommended).** These are version-pinned to a
specific JetPack, so match `JP_VERSION` to your release:

```bash
# check your JetPack version first
head -1 /etc/nv_tegra_release      # e.g. "# R36 (release), REVISION: 4.4"

JP_VERSION=64     # 36.4.x  -> JetPack 6.1 / 6.2
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v${JP_VERSION}/pytorch/<wheel-filename>
python3 -m pip install --no-cache "$TORCH_INSTALL"
```

Browse the available wheels at
`https://developer.download.nvidia.com/compute/redist/jp/v64/pytorch/`. Pick a `cp310`
`linux_aarch64` wheel built against your CUDA version. For PyTorch 24.06 or newer you must
install `cusparselt` first (see NVIDIA's "Installing PyTorch for Jetson Platform" guide).

**Option B — NGC container.** Avoids wheel matching entirely:

```bash
sudo apt install nvidia-jetpack        # if not already present
sudo docker pull nvcr.io/nvidia/pytorch:<tag>-py3   # pick the tag for your JetPack
sudo docker run --runtime nvidia --network host -it --rm \
  -v $PWD:/workspace nvcr.io/nvidia/pytorch:<tag>-py3
```

**Option C — community wheel indexes.** `https://pypi.jetson-ai-lab.io/jp6/cu126/` and
`https://github.com/sudoRicheek/jetson-wheels/releases` host prebuilt JetPack 6 / CUDA 12.6
`cp310` aarch64 wheels. Convenient, but they are third-party and unpinned by NVIDIA — verify
the wheel hash before installing.

Whichever route you take, **verify before anything else**:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# expect: 2.x.x+nvXX.YY 12.6 True (8, 7)
```

If `get_device_capability()` does not print `(8, 7)` you have the wrong wheel.

If `import torch` raises `ImportError: libcudss.so.0`, that library ships separately and is
not installed by default — install it with `pip install nvidia-cudss-cu12`.

### `requirements-jetson.txt`

`requirements-jetson.txt` now exists in the repository. It deliberately **omits `torch` and
`torchvision`** for the reason above — you install those from the Jetson-specific index, not
PyPI. Everything else in it has working aarch64 wheels.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-jetson.txt
# then install torch/torchvision separately, per the section above
```

#### OpenCV and JetPack

JetPack ships its own CUDA-enabled OpenCV as the system `cv2`. Installing the pip
`opencv-python` into a venv created **without** `--system-site-packages` yields a CPU-only
OpenCV and can trigger a numpy/cv2 ABI mismatch. Pick one:

- `python3 -m venv --system-site-packages .venv` — reuses JetPack's CUDA-enabled `cv2`; or
- `pip install --no-deps opencv-python==4.10.0.84` — keeps pip's build, skips the numpy pin

Since this project uses RTSP/MJPEG streaming and (on Jetson) GStreamer hardware encoding,
you want JetPack's OpenCV if you can use it.

### Required files

| Path | Size | Purpose |
| --- | --- | --- |
| `autonomous_drone_main.py` | 37 KB | entry point |
| `modules/` | ~200 KB | production modules |
| `jetson/` | 30 KB | Jetson communication + streaming |
| `shared/` | 23 KB | schemas + transport |
| `YOLO/yolo11n.pt` | 5.6 MB | model weights |
| `benchmarks/distance/manifest.json` | 6 KB | default intrinsics |
| `requirements-jetson.txt` | 2 KB | Jetson aarch64 pins (no torch — see above) |

Optional: `modules/visualizer_ui/` (~10 KB), `modules/control_system/visualizer.py` (~1 KB),
`modules/drone_visualizer.py` (53 B), `modules/auto_calibrate.py` (5 KB).

> **The old deployment document called the visualizer chain "optional / debug-only" and told
> you to skip it. That was wrong and would have caused an immediate `ImportError` on the
> Jetson.** `modules/control_system/api.py:4` imports
> `modules/control_system/visualizer.py` at module scope, `api.py` is reached through the
> `modules/control.py:1` star-import shim, and `visualizer.py:1` imports
> `modules/drone_visualizer.py`, which imports `modules/visualizer_ui/drone_visualizer.py`.
> All three are required. Treat `modules/visualizer_ui/`, `modules/control_system/visualizer.py`
> and `modules/drone_visualizer.py` as **REQUIRED**, not optional.

Do not deploy: `tools/` (~150 KB, needed on the host for calibration — copy individually if
you want it on-device), `benchmarks/distance/images/` and `benchmarks/distance/results/`
(both already deleted and gitignored), `benchmarks/camera_calibration/`, `.venv/`,
`.pytest_cache/`, and any `__pycache__/`. `tests/` is no longer in the tree.

**Keep the five legacy files.** `modules/navigation.py`, `modules/vision.py` and
`modules/vision_utils/` are unreachable from the flight path, but
`modules/distance_estimator/evaluation.py` and `tools/benchmark_distance_estimator.py` both
import `FollowController` from them. Deleting them removes the "Legacy bbox" baseline column
from the distance benchmark. They total under 7 KB.

### Required adaptations

**1. `requirements-jetson.txt`** — done; it exists in the repository and omits
`torch`/`torchvision` deliberately. See
[`requirements-jetson.txt`](#requirements-jetsontxt) and
[Install torch on the Jetson](#install-torch-on-the-jetson-the-part-that-breaks).
Do **not** use `pip install torch torchvision --index-url https://pytorch.org/whl/cu121` —
there is no aarch64 wheel at that index and the command fails.

GStreamer and TensorRT are system-level, installed with `apt` (not pip):

```bash
sudo apt install -y \
    gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

# TensorRT ships with JetPack
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

**2. Camera source** — `modules/yolo11_detector/source.py` currently uses the Windows MSMF
backend. On Jetson use V4L2 for USB cameras:

```python
cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
```

or GStreamer/Argus for CSI cameras:

```python
cap = cv2.VideoCapture(
    "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink",
    cv2.CAP_GSTREAMER,
)
```

**3. RTSP hardware encoding** — in `rtsp_server.py::_try_gstreamer()`:

```
appsrc ! videoconvert ! nvvidconv ! nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5600
```

**4. Serial port** — `/dev/ttyTHS1` (J17 header on Orin) instead of auto-detected COM ports:

```bash
sudo usermod -a -G dialout $USER
echo 'KERNEL=="ttyTHS1", MODE="0666"' | sudo tee /etc/udev/rules.d/99-ttyTHS1.rules
```

**5. Recalibrate intrinsics** for the Jetson camera. `fx = fy = 446.7` was derived from
desktop-camera data using the estimator's own equation — it is a placeholder, not a
calibration, and will be wrong for a different lens. See
[the circular-fit warning](#circular-fit-warning-the-0133-m-baseline-is-not-valid-accuracy).
Resolution differences are handled automatically (`autonomous_drone_main.py:773` calls
`scaled_to_frame`), but a different **lens** is not.

**6. Replace the mock LiDAR** (`modules/lidar_backend/mock.py`) if a real LiDAR is fitted.

### Checklist

Pre-deployment

- [ ] Flash JetPack 6.2 or 6.2.1 (L4T 36.4.3 / 36.4.4, CUDA 12.6). A factory-fresh unit
      ships JetPack 5.x and needs a firmware update first
- [ ] Select MAXN SUPER power mode (4 GB: 10 W / 25 W / MAXN SUPER; 8 GB: 15 W / 25 W / MAXN SUPER)
- [ ] Install GStreamer via `apt`, and v4l2-utils; TensorRT ships with JetPack
- [ ] `python3 -m venv --system-site-packages .venv` (so JetPack's CUDA-enabled `cv2` is used)
- [ ] `pip install -r requirements-jetson.txt`
- [ ] **Install `torch` and `torchvision` from a Jetson-specific source** — not PyPI
- [ ] Verify: `torch.cuda.get_device_capability()` prints `(8, 7)`
- [ ] Test YOLO11 inference on GPU

File transfer

- [ ] `autonomous_drone_main.py`, `modules/`, `jetson/`, `shared/`
- [ ] `YOLO/yolo11n.pt`
- [ ] `benchmarks/distance/manifest.json`
- [ ] `requirements-jetson.txt`

Post-transfer verification

```bash
# must print: 2.x.x+nvXX.YY 12.6 True (8, 7)
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
python3 -c "from ultralytics import YOLO; YOLO('YOLO/yolo11n.pt').to('cuda')"
python3 -c "import cv2; cap=cv2.VideoCapture(0, cv2.CAP_V4L2); print(cap.isOpened())"
python3 -c "import tensorrt; print(tensorrt.__version__)"   # expect 10.3 on JetPack 6.2
gst-inspect-1.0 nvv4l2h264enc
python3 autonomous_drone_main.py --no-prompt --mode sitl --camera 0
```

### Transfer

```bash
#!/bin/bash
JETSON_USER="ubuntu"; JETSON_HOST="jetson.local"
JETSON_PATH="/home/ubuntu/autonomous-drone"

tar --exclude='.venv' --exclude='.idea' --exclude='.pytest_cache' \
    --exclude='.vtcode' --exclude='tests' --exclude='tools' \
    --exclude='benchmarks/distance/images' --exclude='benchmarks/distance/results' \
    --exclude='__pycache__' --exclude='*.pyc' \
    -czf deploy_package.tar.gz \
    autonomous_drone_main.py modules/ jetson/ shared/ YOLO/ \
    benchmarks/distance/manifest.json requirements-jetson.txt

scp deploy_package.tar.gz ${JETSON_USER}@${JETSON_HOST}:${JETSON_PATH}/
# on the Jetson
# tar -xzf deploy_package.tar.gz && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-jetson.txt
```

> Every source document describes this step as "Copy", which presupposes the weights are
> present in the source tree. That is true here: `YOLO/yolo11n.pt` is tracked. It is not
> true if the file is ever untracked — see
> [Repository hygiene and Git policy](#repository-hygiene-and-git-policy).

---

## Distance benchmark

Real images with **independently measured** ground truth only. The synthetic Task-2 fixture
data in `tools/benchmark_distance_estimator.py` is always labelled synthetic and is never
reused as real-world accuracy.

Layout: `manifest.json`, `images/`, `results/`.

### Ground-truth rule

Ground truth **must** be tape-measured, laser-rangefinder measured, an independently
measured fixed target position, another independently verified range measurement, or a
carefully measured rig. It must **never** be computed from the image — no bbox height, no
YOLO, no monocular geometry, no ZoeDepth, no estimator output. Each sample records a
`measurement_method` for auditable provenance.

### `manifest.json` schema

Top level: `version` (int, currently 1), `camera` (object), `samples` (array).

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `id` | yes | string | unique |
| `image` | yes | string | relative to `benchmarks/distance/`, e.g. `images/…` |
| `ground_truth_distance_m` | yes | number (m) | independently measured |
| `class_name` | yes | string | |
| `measurement_method` | yes | string | provenance, e.g. `manual_measurement` |
| `bbox.x/.y/.width/.height` | **effectively yes** | number (px) | **without it the sample cannot be benchmarked** |
| `detection_confidence` | no | number in (0,1] | default 0.9 |
| `scene`, `lighting`, `pose` | no | string | condition tags |
| `notes` | no | string | free text |

`camera` holds `name`, `frame_w`, `frame_h`, `intrinsics_source`, `intrinsics`, and (in the
shipped file) an extra `intrinsics_notes` string that is not in the documented schema.

`intrinsics_source` **must** be one of `measured | calibrated | configured_default | unknown`.
When a real calibration exists, `intrinsics` holds `fx`, `fy`, `cx`, `cy`; otherwise it is
`null`. The benchmark never silently substitutes a default focal length: with
`intrinsics: null` the geometric-vision and legacy-bbox methods report
`NOT_AVAILABLE (no camera calibration)`. If only the sensor is known, set `frame_w`/`frame_h`
from the image size.

### Adding samples

Never hand-edit the JSON.

```bash
python tools/create_distance_dataset.py --image photos/me_3m.jpg --distance 3.0
python tools/create_distance_dataset.py          # fully interactive
```

The tool copies the image into `images/` and updates the manifest. **If no valid
independently measured distance is supplied the sample is not added — no measurement is
ever invented.**

### Running

```bash
python tools/run_distance_benchmark.py
python tools/run_distance_benchmark.py --enable-depth   # adds ZoeDepth
python tools/run_distance_benchmark.py --no-plots --no-legacy
```

Outputs land in `benchmarks/distance/results/`: `REAL_WORLD_DISTANCE_REPORT.md`,
`results.json`, `results.csv` (per-sample long table), `real_world_plots.png` (requires
matplotlib), and `stability_results.csv`. **The whole `results/` directory is gitignored**,
because the report is a `.md` file and would otherwise break the one-README rule on every
run. With the now-empty manifest the status is `READY FOR DATA COLLECTION` and accuracy is
`NOT AVAILABLE` — this is the expected state until you collect Jetson data.

Recommended collection distances: 1, 2, 3, 4, 5, 6, 8, 10 m with several samples each,
varying size, pose, background, lighting, angle and class. Record only classes actually
present. **Measure the Jetson camera's intrinsics with the chessboard first** — see
[Camera calibration](#camera-calibration) and the circular-fit warning below.

### Circular-fit warning: the 0.133 m baseline is not valid accuracy

**The geometric-vision error figures below are circular and must not be used as expected
field accuracy on the Orin Nano.**

`modules/distance_estimator/vision.py:75` computes:

```python
distance_m = (obj_h * float(self._intrinsics.fy)) / float(bbox_height_px)
```

The manifest states `fy` was derived as "mean d*h/1.3m, person height 1.3 m", i.e.
`d = 1.3 * fy / bbox_height_px`. **That is the estimator's own equation, solved backwards.**

So `fy = 446.7` was obtained by finding the focal length that best reproduced the 10
ground-truth distances — and then the estimator was scored on those same 10 images with that
focal length. The metric rewards the fit rather than measuring generalisation, and the
reported MAE of 0.133 m / MRE of 8.0% is optimistically biased by an unknown amount.

**What this means in practice:** once you run a real chessboard calibration on the Orin Nano
camera, expect MAE **worse** than 0.133 m and MRE **worse** than 8.0%. A drop from 8% to
15–20% would still be consistent with this analysis and would not by itself indicate a
regression. Only re-measure with a genuine calibration and fresh, disjoint samples before
quoting any accuracy figure.

The 10 images and the results that produced these numbers have been deleted from this tree
(see [Repository layout](#what-is-not-in-the-tree)); both are recoverable at `cd766f91`. The
figures are reproduced below for historical reference only.

### Historical results (status `EVALUATED` at commit `cd766f91`, circular — see warning above)

Dataset: 10 samples, 10 usable, 0 rejected. Class `person`. Range **1.150 – 2.500 m**.
Measurement method `manual_measurement`. Camera `live-camera`, **640×480**.
Intrinsics `fx=446.700 fy=446.700 cx=320.000 cy=240.000`, source `configured_default`.

Absolute errors in metres, MRE in percent:

| Method | N | MAE | RMSE | Bias | Median AE | Max AE | MRE |
| --- | --: | --: | --: | --: | --: | --: | --: |
| Legacy bbox | 10 | 0.133 | 0.153 | −0.024 | 0.134 | 0.260 | 8.0% |
| Geometric vision | 10 | 0.133 | 0.153 | −0.024 | 0.134 | 0.260 | 8.0% |
| Metric depth | – | N/A | N/A | N/A | N/A | N/A | N/A — depth backend disabled (needs `--enable-depth`, torch + transformers) |
| Fusion | 10 | 0.133 | 0.153 | −0.024 | 0.134 | 0.260 | 8.0% |
| Filtered fusion | 10 | 0.133 | 0.153 | −0.024 | 0.134 | 0.260 | 8.0% |

The four available methods are numerically identical, which follows from
`distance_estimator/evaluation.py:99` importing the legacy `FollowController` as the
"Legacy bbox" method — the legacy chain *is* the reference implementation for the geometric
path. See [Historical audit record](#historical-audit-record).

Distance bins (only 0–2 m and 2–4 m are populated, 5 samples each; identical for all four
methods):

| Bin | N | MAE | RMSE | Bias | MRE |
| --- | --: | --: | --: | --: | --: |
| 0–2 m | 5 | 0.127 | 0.155 | +0.091 | 9.9% |
| 2–4 m | 5 | 0.140 | 0.150 | −0.140 | 6.1% |

Bias flips sign across the 2 m split, and MRE is worse in the near bin.

Condition analysis collapses to a single row because every sample shares class `person`,
scene `live_camera`, lighting `unknown`, pose `unknown` and method `manual_measurement`.

Confidence vs error, method **Fusion**, 10 samples: **Pearson r = −0.144**. The negative sign
means slightly *higher* confidence is associated with slightly *higher* error — confidence
carries no usable signal at this sample size. All 10 samples sit in the `>= 0.5` band
(mean absolute error 0.133). This is an empirical observation, not a calibrated probability.

Per-sample ground truth / prediction, identical across the four methods:

| Sample | GT (m) | Pred (m) | Abs err (m) |
| --- | --: | --: | --: |
| `live_001` | 1.600 | 1.683 | 0.083 |
| `live_002` | 1.150 | 1.409 | 0.259 (worst) |
| `live_003` | 2.000 | 1.949 | 0.051 (best) |
| `live_004` | 2.300 | 2.143 | 0.157 |
| `live_005` | 2.500 | 2.286 | 0.214 |
| `live_006` | 2.300 | 2.135 | 0.165 |
| `live_007` | 2.100 | 1.989 | 0.111 |
| `live_008` | 1.900 | 1.832 | 0.068 |
| `live_009` | 1.500 | 1.478 | 0.022 |
| `live_010` | 1.200 | 1.403 | 0.203 |

Sample notes carry `source=0; burst=1; 2026-09-24T10:57:48` … `11:01:12`.

**Stability results were never analysed.** A `results/stability_results.csv` file existed
(also deleted), but the report deferred temporal smoothing and lag analysis to a separate
Task 2 temporal benchmark that does not exist in this repository. There is therefore **no
temporal-stability evidence for this estimator at all.**

### Stated limitations

- 10/10 samples is not statistically robust; no broad conclusion should be drawn.
- `manual_measurement` ground truth carries its own unquantified error.
- Intrinsics are `configured_default`, not calibrated.
- Geometric vision relies on per-class object heights from `app_config.py`, which are
  approximations.
- Lighting, scene and pose coverage is limited; all three are `unknown`.
- Depth is `NOT_AVAILABLE`.
- **No LiDAR validation data** — no physical returns, and no fake LiDAR measurements were
  generated.
- Filtered fusion is a single-frame pass through the Kalman filter; the samples are
  independent static frames, so it shows no smoothing benefit here.
- **No flight integration**: `Connected to autonomous flight: NO`,
  `Existing flight behavior changed: NO`. This is offline tooling only — no MAVLink,
  flight-mode, velocity or controller behaviour is touched by the benchmark.

---

## Repository hygiene and Git policy

Remote: `https://github.com/amaregese/Autonomous-Ai-drone.git`

### `YOLO/yolo11n.pt` is tracked — deliberately

`.gitignore` contains `/YOLO/*`, `*.pt` and the negation `!YOLO/yolo11n.pt`. The file is
tracked (blob `45b273b4…`, 5,613,764 bytes) *and* matched by an ignore rule. The ignore
rules are inert for an already-tracked file but active for every future model file — for
example `YOLO/yolo11s.pt` is correctly ignored.

The model is a **hard runtime requirement**:

- `autonomous_drone_main.py:103` — the `--model-path` default is `YOLO/yolo11n.pt`
- `autonomous_drone_main.py:752` — `detector.initialize_detector(args.model_path, …)`
- `modules/yolo11_detector/model.py:6-17` — loads a **local** path behind an `os.path.exists`
  guard
- `modules/yolo11_detector/api.py:57` — API surface
- `tools/audit_person_follow_dataflow.py:25` — tooling

The default path is **relative**, so it resolves against the current working directory. There
is no auto-download: a path containing a separator is treated as local, and no `urllib`,
`requests` or `hf_hub_download` call exists anywhere. There is no deployment script; every
Jetson document says "Copy", which presupposes a source file.

Failure mode if the weights are untracked: `os.path.exists` returns False →
`FileNotFoundError` at `model.py:10` → caught by the broad `except Exception` at line 15 →
`load_model` returns `(None, [])` → `api.py:60-62` returns `False` →
`autonomous_drone_main.py:752` aborts **before flight setup**. Eight capabilities fail at
once: production behaviour, Jetson deployment, SITL testing, mock testing, calibration,
distance estimation, person-follow, and SGC communication — plus both Jetson checklists'
copy steps and their CUDA smoke tests.

Size is not a reason to untrack: 5.35 MB is 0.66% of the 848 MB `.git` directory and 78× the
next-largest tracked blob (`real_world_plots.png`, 71,579 B). Git LFS is not warranted.
Untracking would require three things that do not exist: an out-of-band fetch with a
checksum, a pre-initialisation fetch step, and rewritten deployment checklists.

**Policy: `*.pt` being ignored is not a reason to untrack the model.** The ignore rule
predates the tracked file; it is stale, not authoritative.

### `.idea/` — untracked (Policy B)

Six tracked files were staged for deletion; `workspace.xml` (17,659 B) was already
untracked. All of `.idea/` is now ignored.

The tracked content was defective, not portable: `Autonomous-Ai-drone.iml` is a
**Django / label-studio** module facet pointing at `.labeling-venv` and `venv312`, **neither
of which exists** (only `.venv` is present), with `django` absent from `requirements.txt`, and
an SDK name of `"Python 3.12 (Autonomous-AI-Drone) (2)"`. `misc.xml` binds the same
machine-local SDK to the Black formatter. `modules.xml` is a pure pointer. `vcs.xml` is five
redundant lines. `inspectionProfiles/profiles_settings.xml` sets
`USE_PROJECT_PROFILE=false`, configuring nothing.

The strongest argument is JetBrains' own `.idea/.gitignore`, which ignores `/shelf/`,
`/workspace.xml`, `/httpRequests/`, `/queries/`, `/dataSources/` and
`/dataSources.local.xml` — the rest of `.idea/` is local state. `workspace.xml`, the one
untracked file, was the only one "behaving properly".

Policy A (keep selected files) was rejected because no genuinely portable file exists.
Policy C (keep everything) was rejected because it preserves an active defect. Policy B
(untrack all) was applied.

### `.vtcode/` — untracked

`tool-policy.json` is a 192-line AI tool-permission policy: an `available_tools` allowlist of
19 tools, per-tool `policies` (`allow`/`prompt`, e.g. `apply_patch: prompt`,
`unified_exec: prompt`, `git_status: allow`), an `mcp.allowlist` with `enforce: true` for
`context7`, `sequential-thinking` and `time`, plus empty `constraints`, `providers` and
`approval_cache` placeholders.

It is machine-local and per-installer: the MCP provider set, the local timezone override and
the approval cache reflect one developer's install. Nothing in the flight code, detector,
estimator, SGC or tests reads it, and there are zero references outside the now-deleted
audit scripts. The decisive argument is inconsistency — `.vscode/` and `.idea/` were both
ignored while `.vtcode/` was not, with no stated reason. `.vtcode/` is now ignored.

The counter-argument is on record: keeping it tracked *and documenting the intent* would also
be defensible, "because today it looks accidental rather than deliberate". Untracked is
functionally harmless either way; publishing one developer's MCP allowlist and tool
permissions to every clone is not.

### Hygiene concerns

No credential, token, key or `.env` leak was found in any document. The hygiene concerns are
all IDE/assistant-state disclosure:

- `Autonomous-Ai-drone.iml` + `misc.xml` leaked machine-local SDK names, two non-existent
  venv paths, and a foreign project's Django facet to every clone.
- `workspace.xml` (17,659 B) is effectively a list of that developer's uncommitted changes.
- `tool-policy.json` would publish that developer's MCP allowlist and tool permissions.

A hidden-dependency sweep found `YOLO/yolo11n.pt` referenced from 4 real sites, `.idea`
referenced 0 times, `.vtcode` 0 times, and `*.onnx` 0 times ("completely inert"). No build,
test, package or deploy step reads either editor directory.

### A note on `git check-ignore`

`git check-ignore -v` also reports **negation** matches and exits 0 even when a file is *not*
ignored. Use `git check-ignore --no-index <path>` without `-v` (exit 1 = not ignored), or
`git ls-files -i -c --exclude-standard` (empty output = nothing tracked-and-ignored). The
latter now returns empty.

---

## Known contradictions and defects

Everything the source documents asserted inconsistently or got wrong, preserved rather than
papered over. **Verified against the live tree unless marked *(documented)*.**

### Refuted by direct repository check

1. **The visualizer reachability proof was fabricated.** A verification document claimed
   `autonomous_drone_main.py` imports ten symbols from `modules.control_system.api` and cited
   lines 421, 833, 838 and 889. The entry point has no `control_system` import at all;
   reachability is via the `modules/control.py:1` star-import shim. The real line numbers are
   973, 841 and 913–916, and `control_drone` has **zero** call sites in the entry point. The
   document's entire purpose was to prove reachability, and its evidence was invented.
2. **Wrapper size "42 bytes"** is wrong. `modules/drone_visualizer.py` is **53 bytes**
   (one 52-character line plus a newline). The same wrong figure appears in two documents.
3. **`DISTANCE_ESTIMATOR` production file count "14" is an unfixed arithmetic error.** Its own
   table lists 11 unique files with `filter.py`, `models.py` and `lidar.py` duplicated. The
   document acknowledges the duplication without correcting the count.
4. **The "authoritative" file inventory's `git_tracked` column is wrong for the entire
   distance-estimator surface.** It marks `modules/distance_estimator/__init__.py`,
   `estimator.py`, `vision.py`, `calibration.py`, `config.py`, `filter.py`, `models.py`,
   `lidar.py`, `depth.py`, all three `depth_backend/*` files, and `modules/person_follow.py`
   as **not tracked**. All are in fact tracked. On the inventory's own data, a fresh clone
   would be missing the whole distance pipeline — which contradicts the claim that "a clone is
   functional: detection, distance, follow, SGC, SITL and Jetson deployment all work".
5. **The inventory misclassifies `modules/visualizer_ui/` as legacy.** Both
   `modules/visualizer_ui/__init__.py` and `modules/visualizer_ui/drone_visualizer.py` are
   labelled LEGACY, but they are the **active production implementation**, reached at import
   time via `modules/drone_visualizer.py:1`. The earlier audit's correction was applied to two
   files and never propagated.
6. **The "authoritative" inventory is not exhaustive.** It has zero rows for `docs/`, omits
   `.idea/` and `.vtcode/`, and omits `shared/sgc_config.py`. Its "Total files: 153" claim
   therefore does not cover the tree it claims to inventory.
7. **Tracked file count.** One policy document states 134; the actual figure at the time of
   consolidation was 138.
8. **`.gitignore` is described as "33 lines, unmodified"**; it is 35 lines and modified. Every
   subsequent line citation in that document shifted: `*.onnx` moved 9→10, `.idea/` 21→22,
   `.vscode/` 20→21, and `.vtcode/` is new at 23.

### Contradictions between documents

9. **"Test-only" vs "benchmark baseline".** `modules/navigation.py` is described as
   "test-only" because its importers are `distance_estimator/evaluation.py:99` and
   `tools/benchmark_distance_estimator.py:67`. Those are the **benchmark baseline** — the
   "Legacy bbox" column in the distance report. The chain is the reference implementation for
   the geometric path, not incidental test scaffolding. This is why the four available
   benchmark methods are numerically identical.
10. **`D` means two different things.** One document uses **D = development-only**; another
    insists **D = "implementation detail", not debug-only**. A merged document needs a
    per-document key, and an active production-path file should not be labelled `D` at all.
11. **Manifest schema drift.** The benchmark README documents `camera.name: "webcam-front"`
    and 4-key intrinsics. The shipped `manifest.json` has `camera.name: "live-camera"`, a
    6-key intrinsics object (adds `frame_w`, `frame_h`), and an undocumented
    `intrinsics_notes` string. The README schema is not what ships.
12. **`measurement_method` enum drift.** The README lists `tape_measure`, `laser`, `rig`; the
    shipped data uses `manual_measurement`. The documented example value never appears in the
    dataset.
13. **The "outliers" flag is vacuous.** Samples are flagged when
    `absolute_error > 1 m OR relative_error > 1%`. The `> 1%` arm matches all 10 samples in
    all 4 methods, so the section is really a full per-sample listing mislabelled as outliers.
14. **The outlier threshold makes no sense** given the actual error scale: worst absolute
    error is 0.259 m and worst MRE is 22.5%. A `> 1 m` arm would never fire; a `> 1%` arm
    always does.
15. **Reticle radius default mismatch.** The spec mandates 22 and `display.py:829` calls
    `radius=22`, but the function signature default at `display.py:694` is `radius=26`. Any
    other caller silently gets the wrong radius.
16. **BGR/hex pairs** throughout the overlay spec are the BGR triple read as RGB
    (`(255,178,40)` → `#28b2ff`). This is correct by construction and must be stated once or
    implementers will "fix" it.
17. **`shared/sgc_config.py` was counted in a file total.** It was listed twice and classified
    `H - UNUSED`; the correct class is *non-existent stale artifact*. Its presence
    contributed +4 to a 131-vs-127 count discrepancy.
18. **Four different file totals** appear across documents: 127, 131, 153 and 157 tracked.
    Three different scopes are in play — source files, repository files, and git index.
    Without stating which scope each number covers, they read as contradictions.

### Carried with staleness warnings

19. **Hardware records conflict and are unresolved.** Compute capability 9.0 vs 12.0 /
    `sm_120`; `torch.cuda.is_available()` False vs True; torchvision 0.26.0+cpu vs
    0.22.0.dev+cu124; torch 2.11.0+cpu (CPU-only) vs a claimed cu122/`sm_87` Jetson target.
    These cannot both be true. The measured host was a Windows 11 Lenovo Legion with an
    NVIDIA RTX 5060 and a **CPU-only** torch build; the Jetson numbers were aspirations, not
    measurements. **Now resolved:** the correct Orin Nano figure is `sm_87`, and
    `cu122` was only ever right for JetPack 6.0 — JetPack 6.1+ is CUDA 12.6. See
[Target platform](#target-platform-jetson-orin-nano-verified-hardware-facts) and
[the JetPack/CUDA matrix](#jetpack-and-cuda-matrix).
20. **`FINAL_JETSON_DEPLOYMENT_FILE_MAP.md` is defective.** It contains 7 phantom rows, 27
    duplicate rows, and overstates sizes by roughly 3.6× (claiming ~80/20 MB against a real
    ~62/6.1 MB). Its table is not a deployment manifest.
21. **A project report claimed "No test suite present (planned but missing)".** The suite
    exists and has throughout.
22. **`shared/sgc_config.py` — a non-existent stale artifact.** Six checks: the path is
    absent; a stale `shared/__pycache__/sgc_config.cpython-312.pyc` (1,234 bytes) remained
    after the source was deleted; `git log --all` shows it was never committed; zero Python
    imports of `shared.sgc_config`; the eight non-Python references are all audit documents
    generated in the same session; `git ls-files | grep sgc_config` is empty. The only action
    was to delete the stale bytecode. **No source file ever existed.** The bytecode is gone.
23. **All "captured state" figures in the Git policy document are pre-change** — 157 tracked
    files, the 7 tracked-and-ignored files, the `check-ignore` matrix, the 33-line
    `.gitignore`. `git ls-files -i -c --exclude-standard` is now empty, so the
    tracked-and-ignored set is genuinely gone.
24. **CPU performance baseline** *(documented, not reproduced here)*: YOLO11n 46 ms at
    320×320, ~80 ms at 640×480, full pipeline ~22 FPS, distance < 1 ms, follow < 0.1 ms,
    UDP < 1 ms, MJPEG Q30 ~5 ms. These are single-host figures on CPU-only torch and should
    not be read as Jetson expectations.
25. **The overlay specification is not implementable alone** — no schema, no types, no
    envelope, no ports, no state machine. See
    [What this specification does not define](#9-what-this-specification-does-not-define).

---

## Historical audit record

Preserved because the reasoning matters, and because the underlying files still exist.

### The legacy file question — 3 / 5 / 7 and the resolution

Three documents gave three different lists:

| Count | Source | List |
| --- | --- | --- |
| 3 | `PROJECT_CLEANUP_PLAN.md` | `navigation`, `vision`, `vision_utils.legacy` |
| 5 | `PROJECT_FILE_USAGE_AUDIT.md`, `FINAL_FILE_CLEANUP_RISK_REPORT.md` | the 3 above + `drone_visualizer.py`, `control_system/visualizer.py` |
| 7 | `PROJECT_FILE_USAGE_AUDIT_V2.md`, `FINAL_UNIQUE_FILE_INVENTORY.md` | the 5 above + `vision_utils/geometry.py`, `vision_utils/__init__.py` |

The trap: `7 − 2 = 5` numerically, but the correct 5 is a **different** 5 than the
`PROJECT_FILE_USAGE_AUDIT.md` one. That verification document only ever received the 7-item
input; the 3 and the 5 originate upstream.

**Final set — 5 LEGACY, 2 reclassified ACTIVE:**

| # | File | Verdict | Production dependency |
| --: | --- | --- | --- |
| 1 | `modules/navigation.py` | **LEGACY** | benchmark/test only |
| 2 | `modules/vision.py` | **LEGACY** | none |
| 3 | `modules/vision_utils/geometry.py` | **LEGACY** | none |
| 4 | `modules/vision_utils/legacy.py` | **LEGACY** | none |
| 5 | `modules/vision_utils/__init__.py` | **LEGACY** | none |
| 6 | `modules/drone_visualizer.py` | **ACTIVE** *(correction)* | production |
| 7 | `modules/control_system/visualizer.py` | **ACTIVE** *(correction)* | production |

Every import citation, all re-verified against the live tree:

| Citation | Import |
| --- | --- |
| `modules/distance_estimator/evaluation.py:99` | `from modules.navigation import FollowController` |
| `tools/benchmark_distance_estimator.py:67` | `from modules.navigation import FollowController` |
| `modules/navigation.py:3` | `from modules import app_config, lidar, vision` |
| `modules/vision.py:1` | `from modules.vision_utils import *` |
| `modules/vision_utils/__init__.py:1` | `from modules.vision_utils.geometry import *` |
| `modules/vision_utils/__init__.py:2` | `from modules.vision_utils.legacy import process` |
| `modules/vision_utils/legacy.py:3` | `from modules.vision_utils.geometry import getCenter, getDelta` |
| `modules/control_system/api.py:4` | `from modules.control_system.visualizer import (...)` |
| `modules/control_system/visualizer.py:1` | `from modules.drone_visualizer import DroneVisualizer` |
| `modules/drone_visualizer.py:1` | `from modules.visualizer_ui.drone_visualizer import *` |

The isolated chain: `navigation.py → vision.py → vision_utils/__init__.py →
vision_utils/legacy.py → vision_utils/geometry.py`. It is isolated from **production** code
but **load-bearing for the benchmark baseline**. `tools/benchmark_distance_estimator.py`
documents this in its own docstring at line 6: "Legacy bbox-height distance
(modules.navigation.FollowController)". Deleting the chain removes the "Legacy bbox"
benchmark column.

**Recommendation on record: retain the Group 4 legacy chain.** `DISTANCE_ESTIMATOR.md`
commits to preserving this path.

### YOLO11 implementation

There is **one** YOLO11 implementation, behind a wrapper. The directory
`modules/detector_yolo11/` referenced by some documents **does not exist**; the only
`detector_yolo11.py` is a 42-byte one-line shim. **Keep both** the wrapper
(`modules/detector_yolo11.py`) and the implementation
(`modules/yolo11_detector/`) — the wrapper is the stable import surface and the
implementation is the real code.

### The visualizer chain is production, not debug

`autonomous_drone_main.py → modules/control.py → modules/control_system/api.py →
modules/control_system/visualizer.py → modules/drone_visualizer.py →
modules/visualizer_ui/drone_visualizer.py` (9,880 bytes). None of it is debug-only; all
three layers are required for the production runtime. Note that a class label of "D" here
means *implementation detail*, **not** debug-only — the opposite of the "D" used by the
distance-estimator audit.

### Inventory classification scheme *(historical)*

The `AUTHORITATIVE_FILE_INVENTORY` table used columns `path | classification | git_tracked`
with a stated total of 153 rows and 51 YES / 102 NO in the `git_tracked` column (a third of
the tree untracked). **The letters were never defined** — the following scheme is inferred
from usage and was the basis of the cleanup:

| Letter | Inferred meaning |
| --- | --- |
| A | active production code (45 rows) |
| B | data file (1 row) |
| C | tests (11 rows) |
| D | documentation / generated artifact (63 rows) |
| E | environment / root config (2 rows) |
| F | infrastructure / framework (7 rows) |
| G | legacy (7 rows) |
| H | *(unused — later misused for a file that never existed)* |
| I | inventory tooling (17 rows) |

Both the letter scheme and the `git_tracked` column carry the errors listed in
[Known contradictions and defects](#known-contradictions-and-defects). Treat the inventory
as a point-in-time artefact, not as ground truth.

### Cleanup actions applied

- **16 dead root audit scripts removed** (all unreferenced, all regenerable in principle):
  `build_final_inventory.py`, `check_count.py`, `check_duplicates.py`, `classify_files.py`,
  `classify_final.py`, `classify_fixed.py`, `count_files.py`, `final_inventory.py`,
  `generate_final.py`, `generate_final_inventory.py`, `generate_final_report.py`,
  `generate_inventory.py`, `generate_inventory_final.py`, `generate_inventory_fixed.py`,
  `generate_inventory_v3.py`, `verify_counts.py`.
- **3 unreferenced calibration debug JPGs** deleted from
  `benchmarks/camera_calibration/debug/`, plus ~1.3 MB of regenerable caches.
- **Stale bytecode** `shared/__pycache__/sgc_config.cpython-312.pyc` removed.
- **6 `.idea/` files and `.vtcode/tool-policy.json`** staged for deletion; both directories
  added to `.gitignore`.
- **`/YOLO/*`, `*.pt`, `!YOLO/yolo11n.pt`** added to `.gitignore` so the tracked model stays
  tracked while future weights are ignored.
- **No production code, tests, requirements or model weights were modified.** The test result
  is byte-identical before and after: 227 passed, 2 failed.
- **13 superseded audit reports** were archived under `docs/archive/audit/` (all `R100`
  hash-identical moves) before being folded into this file; the archive and `docs/` are now
  empty as a result of this consolidation.

### Non-Python references to legacy files

`DISTANCE_ESTIMATOR.md` and `PROJECT_FILE_USAGE_AUDIT_V2.md` referenced
`modules/navigation.py` in prose. Both documents are now part of this file.

---

## Consolidation provenance

This README replaces 34 Markdown documents totalling 7,305 lines (417.2 KB):

| Group | Files |
| --: | --: |
| Project reports and dependency maps | 3 |
| Distance estimator docs and verifications | 3 |
| Jetson / GPU / pre-Jetson reports | 5 |
| SGC overlay spec | 1 |
| Legacy, visualizer and SG-config verifications | 3 |
| Git policy and cleanup plans | 2 |
| Authoritative file inventory | 1 |
| Benchmark docs (dataset + results) | 3 |
| Archived audit reports (now folded in) | 13 |
| **Total** | **34** |

What was deliberately **not** carried over: process narration, per-document status banners,
duplicate statements that appeared in three or four files, and line-number citations into
the merged documents themselves (which would be meaningless once those files are removed).
Citations into **source code** were re-verified against the live tree and are accurate.

Nothing here was invented. Where a source document asserted something and no check confirmed
it, the claim is attributed or flagged. Where two documents disagreed and no evidence
resolved it, both readings are stated. Where a document was refuted by the code, the code
wins and the error is recorded.

### Contents at a glance

- **Verified against the tree:** every code line citation in this README; all file sizes; the
  `manifest.json` contents; the CLI argument set; the `app_config.py` values; the import
  graph; the legacy 5-file verdict and all 10 import citations; the SGC schema classes and
  fields; the entry point reaching drone connection after the trim; the benchmark reproducing
  MAE 0.133 / RMSE 0.153 / bias −0.024 / MRE 8.0%.
- **Sourced from NVIDIA's published specs:** the Orin Nano hardware table and the
  JetPack/L4T/CUDA/TensorRT/cuDNN matrix.
- **Carried as documented claims, unverified:** the CPU performance baseline; the size
  figures; the 227/2 test result (measured at `cd766f91`, before the suite was removed).
- **Refuted and corrected:** the fabricated visualizer reachability proof; the 42-byte
  wrapper figure; the "14 production files" count; the inventory's `git_tracked` column and
  its `visualizer_ui` legacy misclassification; the tracked-file count; the `.gitignore`
  line-number citations; the vacuous outlier threshold; the `sgc_config.py` phantom.

`AUTHORITATIVE_FILE_INVENTORY.csv` was removed with the rest of the pre-consolidation
documentation. It remains in history at `cd766f91` and carries the defects noted above, so
the tree in this README is the authoritative description of what is actually present.
