# FINAL_JETSON_DEPLOYMENT_FILE_MAP.md

## Jetson Deployment File Map

**Generated:** 2025-09-25  
**Purpose:** Identify exact files needed for Jetson deployment based on actual imports and runtime paths

---

## Deployment Categories

| Code | Category | Description |
|------|----------|-------------|
| **R** | RUNTIME REQUIRED | Must be on Jetson for application to run |
| **C** | CALIBRATION REQUIRED | Needed for camera calibration on Jetson |
| **D** | DEVELOPMENT ONLY | Not needed on Jetson |
| **S** | SITL ONLY | Only for simulation testing |
| **M** | MOCK ONLY | Only for mock testing |

---

## Deployment File Map

### R - RUNTIME REQUIRED (Must Deploy)

| File | Size | Reason | Adaptation Needed |
|------|------|--------|-------------------|
| `autonomous_drone_main.py` | 37 KB | Entry point | Camera source, serial port |
| `modules/app_config.py` | 3 KB | Central config | Jetson-specific constants |
| `modules/control.py` | 42 B | Wrapper | None |
| `modules/control_system/__init__.py` | 42 B | Package init | None |
| `modules/control_system/api.py` | 2.5 KB | PID control | None |
| `modules/control_system/config.py` | 97 B | PID config | None |
| `modules/control_system/pid.py` | 680 B | PID algo | None |
| `modules/control_system/state.py` | 166 B | Global state | None |
| `modules/control_system/visualizer.py` | 1 KB | Debug visualizer | None |
| `modules/detector_yolo11.py` | 42 B | Wrapper | None |
| `modules/detector_yolo11/__init__.py` | 42 B | Package init | None |
| `modules/detector_yolo11/api.py` | 11 KB | Core YOLO | Model path |
| `modules/detector_yolo11/config.py` | 1.6 KB | YOLO config | imgsz=320 for Jetson |
| `modules/detector_yolo11/model.py` | 529 B | Model loading | ultralytics |
| `modules/detector_yolo11/source.py` | 4.8 KB | Camera capture | **V4L2/Argus GStreamer** |
| `modules/detector_yolo11/matching.py` | 5.7 KB | Tracking | None |
| `modules/detector_yolo11/types.py` | 1 KB | Detection class | None |
| `modules/display.py` | 34 KB | HUD/UI | **Jetson DRM/KMS** or HDMI |
| `modules/distance_estimator/__init__.py` | 2.7 KB | Package init | None |
| `modules/distance_estimator/estimator.py` | 13 KB | Core estimator | None |
| `modules/distance_estimator/vision.py` | 8 KB | VisionDistanceEstimator | None |
| `modules/distance_estimator/calibration.py` | 9 KB | CameraIntrinsics | Recalibrate for Jetson cam |
| `modules/distance_estimator/config.py` | 3.6 KB | All configs | None |
| `modules/distance_estimator/filter.py` | 5.4 KB | Kalman filter | None |
| `modules/distance_estimator/models.py` | 3.8 KB | Dataclasses | None |
| `modules/distance_estimator/lidar.py` | 2.5 KB | LiDAR protocol | Real LiDAR if available |
| `modules/distance_estimator/depth.py` | 6 KB | Depth extraction | None |
| `modules/distance_estimator/depth_backend/__init__.py` | 400 B | Backend protocol | None |
| `modules/distance_estimator/depth_backend/synthetic.py` | 2.3 KB | Synthetic backend | None |
| `modules/distance_estimator/filter.py` | 5.4 KB | Kalman filter | None |
| `modules/distance_estimator/models.py` | 3.8 KB | Dataclasses | None |
| `modules/distance_estimator/lidar.py` | 2.5 KB | LiDAR protocol | None |
| `modules/distance_estimator/depth.py` | 6 KB | Depth extraction | None |
| `modules/distance_estimator/depth_backend/__init__.py` | 400 B | Backend protocol | None |
| `modules/distance_estimator/depth_backend/metric_depth.py` | 4.2 KB | ZoeDepth | **Optional** - needs torch |
| `modules/distance_estimator/depth_backend/synthetic.py` | 2.3 KB | Synthetic backend | None |
| `modules/drone.py` | 41 B | Wrapper | None |
| `modules/drone_backend/__init__.py` | 41 B | Package init | None |
| `modules/drone_backend/api.py` | 4 KB | Backend abstraction | None |
| `modules/drone_backend/sitl.py` | 15 KB | SITL backend | **Not for flight** |
| `modules/drone_backend/mock_vehicle.py` | 1.5 KB | Mock backend | **Not for flight** |
| `modules/lidar.py` | 42 B | Wrapper | None |
| `modules/lidar_backend/__init__.py` | 42 B | Package init | None |
| `modules/lidar_backend/mock.py` | 831 B | Mock LiDAR | **Real LiDAR driver** if available |
| `modules/person_follow.py` | 15 KB | Follow controller | None |
| `modules/tracking.py` | 3.7 KB | Target selection | None |
| `modules/auto_calibrate.py` | 4.5 KB | Auto calibration | Chessboard |
| `modules/yolo11_detector/__init__.py` | 42 B | Package init | None |
| `modules/yolo11_detector/api.py` | 11 KB | Core YOLO | None |
| `modules/yolo11_detector/config.py` | 1.6 KB | YOLO config | imgsz=320 |
| `modules/yolo11_detector/model.py` | 529 B | Model loading | ultralytics |
| `modules/yolo11_detector/source.py` | 4.8 KB | Camera capture | **V4L2/Argus** |
| `modules/yolo11_detector/matching.py` | 5.7 KB | Tracking | None |
| `modules/yolo11_detector/types.py` | 1 KB | Detection class | None |
| `modules/control.py` | 42 B | Wrapper | None |
| `modules/control_system/__init__.py` | 42 B | Package init | None |
| `modules/control_system/api.py` | 2.5 KB | PID control | None |
| `modules/control_system/config.py` | 97 B | PID config | None |
| `modules/control_system/pid.py` | 680 B | PID algo | None |
| `modules/control_system/state.py` | 166 B | Global state | None |
| `modules/control_system/visualizer.py` | 1 KB | Debug visualizer | **Optional** on Jetson |
| `modules/visualizer_ui/__init__.py` | 54 B | Package init | None |
| `modules/visualizer_ui/drone_visualizer.py` | 9.9 KB | Debug visualizer | **Optional** on Jetson |
| `modules/drone_visualizer.py` | 53 B | Wrapper | None |
| `modules/drone_visualizer.py` | 53 B | Wrapper | None |
| `modules/lidar.py` | 42 B | Wrapper | None |
| `modules/lidar_backend/__init__.py` | 42 B | Package init | None |
| `modules/lidar_backend/mock.py` | 831 B | Mock LiDAR | Real driver if available |
| `modules/auto_calibrate.py` | 4.5 KB | Auto calibration | Chessboard |
| `modules/tracking.py` | 3.7 KB | Target selection | None |
| `modules/auto_calibrate.py` | 4.5 KB | Auto calibration | Chessboard |
| `modules/yolo11_detector/__init__.py` | 42 B | Package init | None |
| `modules/yolo11_detector/api.py` | 11 KB | Core YOLO | None |
| `modules/yolo11_detector/config.py` | 1.6 KB | YOLO config | imgsz=320 |
| `modules/yolo11_detector/model.py` | 529 B | Model loading | ultralytics |
| `modules/yolo11_detector/source.py` | 4.8 KB | Camera capture | **V4L2/Argus** |
| `modules/yolo11_detector/matching.py` | 5.7 KB | Tracking | None |
| `modules/yolo11_detector/types.py` | 1 KB | Detection class | None |
| `jetson/__init__.py` | 0 B | Package init | Required |
| `jetson/communication/__init__.py` | 0 B | Package init | Required |
| `jetson/communication/detection_sender.py` | 15 KB | UDP streaming | Required |
| `jetson/communication/sgc_receiver.py` | 4 KB | SGC commands | Required |
| `jetson/streaming/__init__.py` | 0 B | Package init | Required |
| `jetson/streaming/rtsp_server.py` | 11.5 KB | RTSP/MJPEG | **GStreamer H.264** |
| `shared/__init__.py` | 0 B | Package init | Required |
| `shared/detection_models.py` | 15 KB | UDP schemas | Required |
| `shared/detection_transport.py` | 7.9 KB | UDP/TCP transport | Required |
| `YOLO/yolo11n.pt` | 5.6 MB | Model weights | Required |
| `benchmarks/distance/manifest.json` | 5.8 KB | Camera intrinsics | Recalibrate for Jetson cam |
| `requirements.txt` | 75 B | Dependencies | **Create requirements-jetson.txt** |

**Total Runtime Required: ~80 files (~20 MB + 5.6 MB model)**

---

### C - CALIBRATION REQUIRED (Needed for Camera Setup)

| File | Size | Reason |
|------|------|--------|
| `modules/auto_calibrate.py` | 4.5 KB | Chessboard calibration |
| `tools/calibrate_camera.py` | 4 KB | Camera calibration tool |
| `tools/print_checkerboard_page.py` | 3.6 KB | Checkerboard generator |
| `benchmarks/camera_calibration/checkerboard_9x6.png` | 47 KB | Calibration target |

---

### S - SITL ONLY (Simulation Only)

| File | Size | Reason |
|------|------|--------|
| `modules/drone_backend/sitl.py` | 15 KB | ArduPilot SITL |
| `modules/drone_backend/mock_vehicle.py` | 1.5 KB | Mock vehicle |
| `modules/lidar_backend/mock.py` | 831 B | Mock LiDAR |
| `modules/control_system/visualizer.py` | 1 KB | Debug visualizer (optional) |
| `modules/visualizer_ui/drone_visualizer.py` | 9.9 KB | Debug visualizer (optional) |
| `modules/drone_visualizer.py` | 53 B | Wrapper (optional) |

---

### M - MOCK ONLY (Testing Only)

| File | Size | Reason |
|------|------|--------|
| `modules/drone_backend/mock_vehicle.py` | 1.5 KB | Mock vehicle |
| `modules/lidar_backend/mock.py` | 831 B | Mock LiDAR |
| `modules/distance_estimator/depth_backend/synthetic.py` | 2.3 KB | Synthetic depth |

---

### D - DEVELOPMENT ONLY (Not for Jetson)

| Category | Files | Count |
|----------|-------|-------|
| Test files (`tests/*.py`) | 15 files | ~120 KB |
| Tools (`tools/*.py`) | 16 files | ~150 KB |
| Benchmark images (`benchmarks/distance/images/*.jpg`) | 10 files | ~500 KB |
| Benchmark data (`benchmarks/distance/results/*.csv/json`) | 3 files | ~100 KB |
| Audit reports (generated) | 10 files | ~100 KB |
| Audit scripts (`check_*.py`, `verify_counts.py`) | 3 files | ~5 KB |

---

## Required System Packages on Jetson

```bash
# JetPack 6.x (L4T 36.x) base
sudo apt update && sudo apt install -y \
    python3 python3-venv python3-pip \
    gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    v4l-utils \
    python3-opencv \
    python3-numpy python3-scipy \
    libyaml-dev

# PyTorch (from NVIDIA NGC)
# pip3 install torch torchvision --index-url https://pytorch.org/whl/cu121
# OR from NVIDIA: nvcr.io/nvidia/pytorch:24.01-py3

# TensorRT (included in JetPack)
# python3 -c "import tensorrt; print(tensorrt.__version__)"
```

---

## Python Dependencies for Jetson (`requirements-jetson.txt`)

```txt
# Core
opencv-python==4.10.0.84
numpy==1.26.4
ultralytics==8.4.41
simple-pid==2.0.1
pymavlink==2.4.49
pyserial==3.5

# PyTorch - MUST use JetPack version (cu121/cu124)
# torch==2.5.1+cu124
# torchvision==0.20.1+cu124

# GStreamer Python bindings (system packages above)

# SRT (if needed)
# pysrt

# Optional: TensorRT Python (from JetPack)
# tensorrt==8.6.1
```

---

## Camera Source Adaptation Required

### Current (`modules/yolo11_detector/source.py` - Windows MSMF):
```python
cap = cv2.VideoCapture(index)  # MSMF backend
```

### Jetson Adaptation (V4L2 for USB / Argus for CSI):
```python
# USB Camera
cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# CSI Camera (Argus) - GStreamer pipeline
cap = cv2.VideoCapture(
    "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink",
    cv2.CAP_GSTREAMER
)
```

---

## RTSP Server Adaptation Required

### Current (`jetson/streaming/rtsp_server.py` - MJPEG fallback):
```python
# Falls back to MJPEG HTTP on Windows
```

### Jetson (Hardware H.264 Encoding):
```python
# Enable in rtsp_server.py _try_gstreamer():
pipeline = (
    "appsrc ! videoconvert ! nvvidconv ! "
    "nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! "
    "h264parse ! rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=127.0.0.1 port=5600"
)
```

---

## Serial Port Configuration

### Current (Windows):
```python
# Auto-detects COM ports
```

### Jetson:
```bash
# Default UART for Pixhawk
/dev/ttyTHS1  # J17 header on Jetson Orin

# Udev rule needed:
sudo usermod -a -G dialout $USER
echo 'KERNEL=="ttyTHS1", MODE="0666"' | sudo tee /etc/udev/rules.d/99-ttyTHS1.rules
```

---

## Camera Intrinsics

**Current:** `benchmarks/distance/manifest.json` (fx=446.7, fy=446.7 for 640x480)

**Action:** Recalibrate for Jetson camera using `tools/calibrate_camera.py` on Jetson.

---

## Deployment Checklist

### Pre-Deployment
- [ ] Flash JetPack 6.x (L4T 36.x) on Jetson
- [ ] Install system packages: GStreamer, v4l2, TensorRT
- [ ] Create `requirements-jetson.txt` with matched versions
- [ ] Set up Python 3.10+ venv on Jetson
- [ ] Install dependencies from `requirements-jetson.txt`
- [ ] Verify `torch.cuda.is_available()` and `torch.cuda.get_device_capability()`
- [ ] Test YOLO11 inference on GPU

### File Transfer
- [ ] Copy `autonomous_drone_main.py`
- [ ] Copy `modules/` directory
- [ ] Copy `jetson/` directory
- [ ] Copy `shared/` directory
- [ ] Copy `YOLO/yolo11n.pt`
- [ ] Copy `benchmarks/distance/manifest.json`
- [ ] Copy adapted `requirements-jetson.txt`

### Post-Transfer Verification
- [ ] `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] `python -c "from ultralytics import YOLO; YOLO('YOLO/yolo11n.pt').to('cuda')"`
- [ ] `python -c "import cv2; cap=cv2.VideoCapture(0, cv2.CAP_V4L2); print(cap.isOpened())"`
- [ ] `python -c "import tensorrt; print(tensorrt.__version__)"`
- [ ] `gst-inspect-1.0 nvv4l2h264enc`
- [ ] Run `python autonomous_drone_main.py --no-prompt --mode sitl --camera 0`

### SITL vs Flight Mode
- **SITL:** `--mode sitl --start-sitl` (requires WSL/ArduPilot on separate machine)
- **Flight:** `--mode flight --drone_connection /dev/ttyTHS1`

---

## Summary

| Category | Files | Deploy to Jetson |
|----------|-------|------------------|
| **R** - Runtime Required | ~80 | ✅ YES |
| **C** - Calibration Required | 4 | ✅ YES |
| **S** - SITL Only | 6 | ⚠️ For testing |
| **M** - Mock Only | 3 | ❌ NO |
| **D** - Development Only | ~45 | ❌ NO |

**Total deployment size: ~22 MB (including 5.6 MB model)**

**Key adaptations needed:**
1. `requirements-jetson.txt` with JetPack-matched PyTorch
2. Camera source: MSMF → V4L2/Argus GStreamer
3. RTSP: MJPEG → Hardware H.264 (nvv4l2h264enc)
4. Serial: COM port → `/dev/ttyTHS1`
5. Camera calibration: Recalibrate for Jetson camera