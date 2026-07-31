# Autonomous Drone System Report

1. Executive Summary
The autonomous drone system is a full target-tracking and following pipeline running on the onboard Jetson. It uses a YOLO11 neural network on the live camera feed to detect and track a user-selected object, then flies the drone toward it via ArduPilot/MAVLink commands in GUIDED mode, keeping the target centered in view. The system includes a full OpenCV HUD overlay, RTSP video + UDP detection streaming to the ground station, remote command-and-control back from the SGC, monocular distance estimation, camera auto-calibration, and an automatic Return-to-Launch fallback if the target is lost.

The system runs in three backends — SITL simulation, mock (no hardware), and real flight — enabling full desktop development and validation before any hardware is involved.

2. System Architecture
┌──────────────┐   MJPEG over HTTP (port 8554)   ┌──────────────────────┐
│    JETSON    │ ───────────────────────────────▶ │   SGC (WEB / GROUND) │
│ · YOLO11     │   UDP JSON packets (detections)  │ · Flask + Socket.IO  │
│ · OpenCV     │ ───────────────────────────────▶ │ · Canvas HUD overlay │
│ · MAVLink    │                                  │                      │
│ · RTSP server│ ◀─────────────────────────────── │                      │
└──────────────┘    UDP commands (port 9002)      └──────────────────────┘
                        │
              ┌─────────▼─────────┐
              │  FLIGHT CONTROLLER│
              │  ArduPilot Copter │
              │  (GUIDED / LAND / │
              │   RTL)            │
              └───────────────────┘

Data paths:
• Video: Jetson MJPEG/RTSP stream → shown directly in the SGC feed (HUD rendered client-side)
• Detections: Jetson sends JSON datagrams over UDP → SGC parses → forwarded via Socket.IO to browser → drawn on canvas overlay
• Commands: SGC → UDP port 9002 (select/deselect target, follow start/stop)
• Flight control: Jetson ↔ ArduPilot via MAVLink (arm, takeoff, SET_POSITION_TARGET_LOCAL_NED, LAND/RTL)
• Simulation: ArduCopter SITL launched inside WSL (sim_vehicle.py) for desktop testing

3. Backends & Drone Control
Feature
Status
SITL simulation (ArduPilot in WSL, auto-launch)
✅ Supported (--mode sitl --start-sitl)
Mock vehicle (offline, no hardware)
✅ Supported (default backend)
Real flight over USB serial (/dev/ttyACM0)
✅ Supported (--mode flight)
MAVLink telemetry cache thread (heartbeat, position, battery, EKF, GPS, STATUSTEXT)
✅
GUIDED arm + takeoff with GPS/EKF/altitude readiness gate
✅ Wait for 3D fix, EKF convergence, armed, and ≥95% target altitude
LAND / RTL mode switching
✅
Velocity + yaw-rate commands (SET_POSITION_TARGET_LOCAL_NED, body-frame)
✅
EKF health check (attitude, velocity, horizontal + vertical position)
✅

4. Detection & Tracking Pipeline
• Model: Ultralytics YOLO11n (YOLO/yolo11n.pt, 80 COCO classes, half-precision on CUDA)
• Detection thresholds: conf 0.30, IOU 0.45, min box area ratio 0.0005, inference size 480
• Target selection: mouse click on the OpenCV tracker window, or remotely via SGC UDP command
• Tracking: multi-feature matching (HSV color histogram, mean HSV, edge density, aspect ratio, area) with weighted scoring
• Re-acquisition: feature memory (90 frames), motion prediction, and uniqueness gating after temporary loss
• Tracker states: idle / tracking / lost
• Class filtering: client-side filter per class in SGC, synced back to the Jetson
• Frame skip + threaded camera probe with auto-scan (cameras 0–4)

5. Follow Control & Distance Estimation
• Monocular ranging: distance = (object_height × focal_length) / bbox_height_px, with per-class real-world heights (person 1.3 m, car 1.5 m, dog 0.5 m, …)
• LiDAR + vision distance blending (LiDAR weight 0.65; LiDAR currently mock)
• Forward velocity: proportional gain on distance error with deadband (0.18 m) and quadratic brake zone (0.75 m), capped at 2.0 m/s
• Yaw: proportional gain on normalized horizontal offset, capped at 15°
• Smoothing: moving-average on commands
• Default follow standoff: 1.0 m, flight altitude 2.5 m

6. HUD & Visualization
The tracker window renders a full HUD:
• Status bar: state pill (IDLE / TRACKING / LOST), target class + distance, yaw, movement
• FPS pill, telemetry panel, shortcut bar
• Bounding boxes: unselected = amber border + label; selected = yellow corner brackets + label
• Tracking overlay: leader line, target circle, info pill (following / dist / conf)
• Selection prompt ("Click on any object to track")
• Lost banner (flashing red "TARGET LOST") with live RTL countdown
• Notification toasts; H key toggles header/footer only
• Secondary DroneVisualizer: 2D top-down window (drone, target, movement trail, distance bar, telemetry)

7. Safety Logic
• Target selection requires: motors armed + GPS 3D fix + EKF healthy
• Target lost for 10 s → automatic Return-to-Launch (RTL) with countdown shown on HUD
• RTL triggers only once — the countdown does not restart after triggering
• ESC / SPACE / R / H / Q keyboard shortcuts (deselect, stop follow, reset tracker, toggle HUD, quit & land)

8. Recent Changes / Bug Fixes
Item
Status
Auto-disarm right after takeoff (SITL)
✅ Fixed — takeoff now waits for ≥95% target altitude before the main loop starts; previously velocity=(0,0,0) hold commands overrode the climb and caused an immediate landing/disarm
Lost-countdown restarting after RTL
✅ Fixed — RTL handler no longer clears the RTL-triggered guard, so the countdown stays at 0 (RTL active) instead of restarting
hold_position() coordinate frame
✅ Aligned to MAV_FRAME_BODY_OFFSET_NED for consistency with movement commands

9. Known Issues / Blockers
Item
Status
Automated tests
🔴 No test suite present (PyCharm configured for pytest, coverage installed — planned but missing)
requirements.txt / dependency pinning
🔴 None — reproducibility risk for other machines
Real-flight validation
🟡 SITL-validated only; flight mode not yet tested on hardware
LiDAR
🟡 Mock backend only; real serial backend (/dev/ttyTHS1) not implemented
Documentation
🟡 No README — report is first project doc

10. Next Steps
1. Add automated tests (detector matching, follow-controller math, lost/RTL state machine, mock backend)
2. Pin dependencies via requirements.txt
3. Validate --mode flight on real hardware
4. Implement real LiDAR backend and verify distance blending
5. Write README (install / build / run guide)

11. Key Files
File
Purpose
autonomous_drone_main.py
Entry point — takeoff → main-loop state machine, keyboard controls, landing
modules/app_config.py
Central tuning constants (altitude, speed, gains, object heights, lost threshold)
modules/drone_backend/sitl.py
pymavlink SITL/ArduPilot backend: connect, telemetry thread, arm/takeoff, velocity commands
modules/drone_backend/api.py
Backend dispatch (mock / sitl / flight)
modules/yolo11_detector/api.py
YOLO11 detection + target tracking state machine
modules/yolo11_detector/matching.py
Multi-feature matching & re-acquisition scoring
modules/navigation.py
FollowController — ranging, distance blending, forward/yaw command computation
modules/display.py
HUD / overlay rendering
modules/tracking.py
Mouse-click target selection with safety preconditions
modules/control_system/api.py
Control API (connect, arm, takeoff, land, PID loop)
jetson/communication/detection_sender.py
RTSP + UDP detection/telemetry streaming to SGC
jetson/communication/sgc_receiver.py
UDP command receiver (select/deselect target, follow start/stop)
jetson/streaming/rtsp_server.py
RTSP server (GStreamer → FFmpeg → MJPEG fallback)
shared/detection_models.py
Shared data contracts (FrameDetections, TelemetryData, TrackingState)
tools/calibrate_camera.py
Standalone chessboard camera calibration CLI
