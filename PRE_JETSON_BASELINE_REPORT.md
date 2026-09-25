# PRE_JETSON_BASELINE_REPORT.md

## Project Baseline Inspection

**Date:** 2025-09-25  
**Repository:** Autonomous-AI-Drone  
**Platform:** Windows 11 (Lenovo Legion with NVIDIA RTX 5060)

---

## 1. Project Structure

```
Autonomous-AI-Drone/
├── autonomous_drone_main.py       # Main entry point
├── requirements.txt               # Minimal dependencies
├── YOLO/yolo11n.pt               # YOLO11 nano model weights
├── benchmarks/                    # Distance estimation benchmarks & calibration
│   ├── camera_calibration/
│   └── distance/
├── jetson/                        # Jetson-specific code
│   ├── communication/             # UDP detection/command channels
│   │   ├── detection_sender.py
│   │   └── sgc_receiver.py
│   └── streaming/                 # RTSP/MJPEG streaming
│       └── rtsp_server.py
├── modules/                       # Core modules
│   ├── app_config.py              # Configuration constants
│   ├── auto_calibrate.py          # Chessboard camera calibration
│   ├── control.py                 # PID control interface
│   ├── control_system/            # PID implementation
│   ├── detector_yolo11.py         # YOLO11 detector wrapper
│   ├── detector_yolo11/           # YOLO11 internals
│   │   ├── api.py                 # Main detector API
│   │   ├── config.py              # YOLO config
│   │   ├── model.py               # Model loading
│   │   ├── source.py              # Camera capture
│   │   ├── matching.py            # Object tracking/matching
│   │   └── types.py               # Detection dataclass
│   ├── display.py                 # OpenCV UI/HUD
│   ├── distance_estimator/        # Monocular distance estimation
│   │   ├── estimator.py           # Main DistanceEstimator (fusion + Kalman)
│   │   ├── vision.py              # Geometric vision distance
│   │   ├── calibration.py         # Camera intrinsics
│   │   ├── config.py              # Estimator configs
│   │   ├── filter.py              # Kalman filter
│   │   ├── models.py              # Measurement dataclasses
│   │   └── depth_backend/         # Metric depth backends
│   ├── drone.py                   # Drone abstraction
│   ├── drone_backend/             # MAVLink backends
│   │   ├── api.py                 # Unified backend API
│   │   ├── sitl.py                # ArduPilot SITL
│   │   └── mock_vehicle.py        # Mock for testing
│   ├── lidar.py / lidar_backend/  # LiDAR (mock)
│   ├── navigation.py              # Legacy FollowController
│   ├── person_follow.py           # Task 9 person-follow controller
│   ├── tracking.py                # Mouse-click target selection
│   └── vision_utils/              # Geometry utilities
├── shared/                        # Shared types/transport
│   ├── detection_models.py        # UDP message schemas
│   └── detection_transport.py     # UDP/TCP transport
├── tests/                         # Test suite (229 tests)
├── tools/                         # Diagnostic/benchmark tools
└── .venv/                         # Virtual environment
```

---

## 2. Python Version & Virtual Environment

| Item | Value |
|------|-------|
| Python Version | 3.12.7 |
| Virtual Environment | `.venv/` (active) |
| pip Version | 24.2 |

---

## 3. Dependencies (requirements.txt)

```
opencv-python
numpy
torch
ultralytics
simple-pid
pymavlink
pyserial
```

**Key Installed Versions:**
- torch: 2.11.0+cpu (**CPU-only build**)
- torchvision: 0.26.0+cpu
- ultralytics: 8.4.41
- opencv-python: 4.10.0.84
- opencv-contrib-python: 4.8.1.78
- numpy: 1.26.4
- pymavlink: 2.4.49
- pytest: 9.1.1

---

## 4. YOLO11 Implementation

**Module:** `modules/yolo11_detector/`

| Aspect | Detail |
|--------|--------|
| Model | YOLO11n (nano) - `YOLO/yolo11n.pt` |
| Framework | Ultralytics YOLO |
| Inference | `model.predict()` via ultralytics |
| Input Resolution | Configurable (default 320 via `--imgsz`) |
| Confidence Threshold | Configurable (default 0.3) |
| Tracking | Custom IoU + feature matching (`matching.py`) |
| Camera Capture | OpenCV VideoCapture (MSMF on Windows) |
| Frame Normalization | Horizontal flip (`cv2.flip(frame, 1)`) in `source.py` |

**Classes Detected:** 80 COCO classes

---

## 5. Detector/Tracker Implementation

**Detector (`modules/yolo11_detector/api.py`):**
- Global state for model, camera, tracking
- Frame skipping configurable (`FRAME_SKIP_CAMERA = 2`)
- Tracking: IoU + visual feature matching + reacquisition
- Selected object persistence across frames

**Tracking Features:**
- Unique candidate selection with score margins
- Target memory for reacquisition (up to 90 frames)
- Velocity prediction during occlusion

---

## 6. OpenCV Usage

- Version: 4.10.0.84 (opencv-python) + 4.8.1.78 (opencv-contrib-python)
- Camera: `cv2.VideoCapture` with MSMF backend on Windows
- Display: `cv2.imshow` with custom HUD overlay
- Image processing: resize, flip, color conversion
- MJPEG encoding: `cv2.imencode(".jpg", ...)`

---

## 7. PyTorch Usage

- **CRITICAL ISSUE:** PyTorch installed as **CPU-only** (`torch 2.11.0+cpu`)
- `torch.cuda.is_available()` returns `False`
- YOLO model runs on **CPU** (`next(model.model.parameters()).device == 'cpu'`)
- No CUDA acceleration available for inference

---

## 8. CUDA Usage

- **No CUDA support** in current PyTorch installation
- NVIDIA Driver: 592.01 (CUDA 13.1 compatible)
- GPU: RTX 5060 (8GB VRAM, Compute Capability 9.0)
- PyTorch CUDA runtime: Not installed

---

## 9. MAVLink Implementation

**Backend Architecture (`modules/drone_backend/`):**
- Abstract backend API (`api.py`)
- SITL backend (`sitl.py`) - uses pymavlink + mavutil
- Mock backend (`mock_vehicle.py`) - for testing without hardware

**Commands Supported:**
- `arm_and_takeoff()`, `land()`, `send_rtl()`
- `send_movement_command_XYA(vx, vy, altitude)`
- `send_movement_command_YAW(angle)`
- `send_servo()`, `hold_position()`

**Telemetry:**
- Position, battery, GPS fix, EKF, mode, armed state
- Message listener thread for heartbeat + telemetry

---

## 10. ArduPilot/SITL Integration

- SITL launched via WSL (`ardupilot` in WSL2)
- Connection: `udpin:0.0.0.0:14550`
- Auto-launch option: `--start-sitl`
- Mock backend for Windows development without SITL

---

## 11. Autonomous Control Logic

**Main Loop (`autonomous_drone_main.py`):**
1. Keyboard/SGC command handling
2. YOLO detection + tracking
3. Distance estimation (DistanceEstimator)
4. Target selection (mouse click or SGC)
5. PersonFollowController computes vx/vy/yaw
6. MAVLink commands sent via drone backend
7. Telemetry + UDP detection streaming
8. Display + RTSP/MJPEG streaming

**PersonFollowController (`modules/person_follow.py`):**
- Distance→speed profile (hover ≤2m, cruise >2m)
- Lateral centering via x_delta * GAIN_YAW
- Acceleration limiting (ramp)
- Target loss timeout → RTL or hover

---

## 12. Video Streaming

**RTSP Server (`jetson/streaming/rtsp_server.py`):**
- Multi-backend: GStreamer → FFmpeg → MJPEG HTTP fallback
- MJPEG over HTTP (port configurable, default 8554)
- JPEG quality configurable (default 30)
- FrameHolder with double-buffering

**RTSP Note:** GStreamer/FFmpeg paths unlikely to work on Windows without proper installation.

---

## 13. UDP Detection Channel (Port 9001)

**Transport:** `shared/detection_transport.py`
- UDP with length-prefixed JSON
- Sender: `UDPTransport.send()` 
- Receiver: `UDPTransport.start_receiving(callback)`
- Schema: `FrameDetections` with `Detection[]` + telemetry + overlay config

---

## 14. UDP Command Channel (Port 9002)

**Receiver:** `jetson/communication/sgc_receiver.py`
- JSON commands from SGC (Ground Control)
- Supported: `select_target`, `follow_start`, `follow_stop`, `deselect_target`, `takeoff`, `servo`
- Commands matched to detections via bbox + class

---

## 15. SGC Communication

- SGC = "Simple Ground Control" 
- Detection stream: Drone → SGC (UDP 9001)
- Command stream: SGC → Drone (UDP 9002)
- Overlay config sent with detections for UI rendering

---

## 16. Configuration Files

| File | Purpose |
|------|---------|
| `modules/app_config.py` | All tuning constants (follow, MAVLink, camera) |
| `benchmarks/distance/manifest.json` | Camera intrinsics + dataset manifest |
| `sgc_config.json` | Not found in repo (may be external) |
| CLI args in `autonomous_drone_main.py` | Runtime overrides |

---

## 17. CLI Entry Points

| Script | Purpose |
|--------|---------|
| `autonomous_drone_main.py` | Main autonomous flight |
| `tools/test_project_distance.py` | Distance diagnostic (Tasks 6/7) |
| `tools/live_distance_capture.py` | Data collection |
| `tools/calibrate_camera.py` | Chessboard calibration |
| `tools/benchmark_distance_estimator.py` | Benchmark runner |

---

## 18. Logging

- Stdlib `logging` module used in transport modules
- Print statements for user-facing output
- No structured logging framework

---

## 19. Error Handling

- Try/except around MAVLink operations
- Socket timeouts for UDP
- Camera read failure handling
- Mock fallbacks when backends unavailable

---

## 20. Test Suite

**229 tests collected, 227 passed, 2 failed**

| Test Module | Tests | Focus |
|-------------|-------|-------|
| test_authoritative_follow_distance | 11 | Distance source unification |
| test_camera_mirroring_diagnostic | 9 | Frame normalization |
| test_capture_calibration_images | 16 | Calibration capture |
| test_detection_labels | 10 | UI distance labels |
| test_distance_dataset_benchmark | 18 | Benchmark metrics |
| test_distance_estimator | 36 | Distance estimator core |
| test_distance_estimator_depth | 21 | Metric depth fusion |
| test_live_distance_capture | 24 | Live data capture |
| test_person_follow | 19 | Follow controller |
| test_project_distance | 30 | Distance diagnostic tool |
| test_sgc_receiver | - | SGC commands |
| test_sgc_safety_gate | - | Safety gates |

**Failures (2):** `TestBackendLaziness` - torch already loaded in test environment (expected, not real failures)

---

## 21. Existing Test Reports

| Report | Location |
|--------|----------|
| Real-World Distance Accuracy | `benchmarks/distance/results/REAL_WORLD_DISTANCE_REPORT.md` |
| Benchmark Results | `benchmarks/distance/results/results.json/csv` |
| Stability Results | `benchmarks/distance/results/stability_results.csv` |

---

## 22. Platform-Specific Code

| Module | Windows-Specific | Linux/Jetson-Specific |
|--------|------------------|----------------------|
| `source.py` | MSMF capture, device enumeration | V4L2, GStreamer |
| `drone_backend/sitl.py` | WSL launch (`wsl.exe`) | Native SITL |
| `rtsp_server.py` | MJPEG fallback only | GStreamer/FFmpeg RTSP |
| `lidar_backend/mock.py` | Mock only | Real LiDAR serial |

---

## 23. Jetson-Specific Code

- `jetson/` directory: streaming + communication
- RTSP server designed for Jetson GStreamer hardware encoding
- SGC communication for ground station integration
- Distance estimator designed for Jetson Orin/Nano compute

---

## 24. CPU/GPU Fallback Behavior

| Component | Current Behavior |
|-----------|------------------|
| YOLO Inference | CPU-only (PyTorch CPU build) |
| Distance Estimation | Pure Python/NumPy (no GPU) |
| Kalman Filter | Pure Python |
| RTSP Encoding | Software (MJPEG) |
| MAVLink | Pure Python (pymavlink) |

**No GPU acceleration anywhere in the pipeline currently.**

---

## Summary: Critical Findings

| Issue | Severity | Impact |
|-------|----------|--------|
| PyTorch CPU-only | **CRITICAL** | YOLO inference ~10-20x slower than GPU; Jetson needs CUDA |
| No requirements-pin | High | Version drift risk on Jetson |
| Windows-only camera code | High | Won't work on Jetson V4L2/Argus |
| RTSP GStreamer path untested | High | Primary streaming path on Jetson |
| No Jetson deployment scripts | Medium | Manual setup required |

---

## Next Steps

1. **Install PyTorch with CUDA 12.1+** for Windows verification
2. **Verify YOLO runs on GPU** on Legion
3. **Test SITL connection** end-to-end
4. **Audit Windows→Linux portability** for all modules
5. **Create Jetson-specific requirements** (torch+cu121, tensorrt, etc.)