# JETSON_DEPLOYMENT_FILE_SET.md

## Jetson Deployment File Set

**Objective:** Identify the minimum and complete set of files required for Jetson deployment.

---

## Classification for Jetson Deployment

| Code | Classification | Description |
|------|----------------|-------------|
| **REQUIRED** | Must be deployed | Core production functionality |
| **OPTIONAL** | Deploy if space permits | Enhances functionality |
| **TEST-ONLY** | Do NOT deploy | Test files only |
| **DEVELOPMENT-ONLY** | Do NOT deploy | Tools, benchmarks, diagnostics |
| **NOT REQUIRED ON JETSON** | Exclude | Windows-specific, mock-only, legacy |
| **UNKNOWN** | Investigate | Uncertain status |

---

## File Set for Jetson Deployment

### REQUIRED - Core Production (Must Deploy)

| File / Directory | Size | Purpose |
|-----------------|------|---------|
| `autonomous_drone_main.py` | 37 KB | Production entry point |
| `modules/` | ~200 KB | All production modules |
| `├── app_config.py` | 3 KB | Central config |
| `├── auto_calibrate.py` | 5 KB | Chessboard calibration |
| `├── control.py` | 42 B | Abstraction layer |
| `├── control_system/` | 5 KB | PID control |
| `├── detector_yolo11.py` | 42 B | Abstraction layer |
| `├── display.py` | 34 KB | HUD/Overlay rendering |
| `├── distance_estimator/` | 100 KB | Core estimator |
| `├── drone.py` | 41 B | Abstraction layer |
| `├── drone_backend/` | 20 KB | MAVLink backends |
| `├── lidar.py` | 42 B | Abstraction layer |
| `├── lidar_backend/` | 1 KB | Mock LiDAR |
| `├── person_follow.py` | 15 KB | Task 9 follow controller |
| `├── tracking.py` | 4 KB | Target selection |
| `├── auto_calibrate.py` | 5 KB | Auto calibration |
| `├── yolo11_detector/` | 25 KB | YOLO + tracking |
| `└── vision_utils/` | 2 KB | Geometry utils |
| `jetson/` | 30 KB | Jetson-specific |
| `├── communication/` | 20 KB | UDP + SGC |
| `└── streaming/` | 12 KB | RTSP/MJPEG |
| `shared/` | 23 KB | UDP schemas + transport |
| `YOLO/yolo11n.pt` | 5.6 MB | Production model |
| `requirements.txt` | 75 B | Dependencies (will adapt) |
| `benchmarks/distance/manifest.json` | 6 KB | Camera intrinsics |

**Total Required: ~10 MB (excluding .pt model) + 5.6 MB model**

---

### REQUIRED - Configuration Adaptation Needed

| File | Change Required |
|------|-----------------|
| `requirements.txt` | Create `requirements-jetson.txt` with JetPack-matched versions |
| `autonomous_drone_main.py` | Serial port default: `/dev/ttyTHS1` (Jetson UART) |
| `modules/lidar_backend/mock.py` | Replace with real LiDAR backend if available |
| `modules/yolo11_detector/source.py` | V4L2/Argus camera support (replace MSMF) |
| `jetson/streaming/rtsp_server.py` | Enable GStreamer hardware encoding |

---

### OPTIONAL - Deploy If Space Permits

| File / Directory | Size | Reason |
|-----------------|------|--------|
| `modules/visualizer_ui/` | 10 KB | Debug visualizer (control_system.visualizer uses it) |
| `modules/control_system/visualizer.py` | 1 KB | Debug visualizer (used by control_system.api) |
| `modules/drone_visualizer.py` | 53 B | Wrapper for visualizer |
| `modules/auto_calibrate.py` | 5 KB | Useful for field calibration |
| `modules/auto_calibrate.py` dependencies | - | Chessboard calibration in field |
| `benchmarks/distance/manifest.json` | 6 KB | Camera intrinsics reference |

---

### TEST-ONLY - Do NOT Deploy

| File / Directory | Size | Reason |
|-----------------|------|--------|
| `tests/` | ~120 KB | Test suite - not for production |
| `tests/test_*.py` (15 files) | ~120 KB | Test suite |

---

### DEVELOPMENT-ONLY - Do NOT Deploy

| File / Directory | Size | Reason |
|-----------------|------|--------|
| `tools/` | ~150 KB | Calibration, benchmarking, diagnostics |
| `tools/*.py` (16 files) | ~150 KB | Dev tools only |
| `benchmarks/distance/images/` | ~500 KB | Test dataset |
| `benchmarks/camera_calibration/` | ~200 KB | Calibration target + debug |
| `benchmarks/distance/results/` | ~200 KB | Generated results |
| `DISTANCE_ESTIMATOR.md` | 26 KB | Documentation |
| `SGC_TRACKER_OVERLAY_SPEC.md` | 15 KB | Specification |
| `PRE_JETSON_*.md` | ~20 KB | Verification reports |
| `GPU_CUDA_VERIFICATION.md` | 7 KB | Verification report |
| `PROJECT_REPORT.md` | 8 KB | Project overview |

---

### NOT REQUIRED ON JETSON

| File / Directory | Reason |
|-----------------|--------|
| `.idea/` | PyCharm IDE config (Windows) |
| `.pytest_cache/` | Test cache |
| `.vtcode/` | VS Code tool config |
| `.venv/` | Virtual environment (recreate on Jetson) |
| `modules/navigation.py` | LEGACY - not used |
| `modules/vision.py` | LEGACY - not used |
| `modules/vision_utils/legacy.py` | LEGACY - not used |
| `modules/control_system/visualizer.py` | Debug only (optional) |

---

### UNKNOWN - Investigate

| File | Action |
|------|--------|
| `shared/sgc_config.py` | Check if SGC ground station loads it |

---

## Jetson-Specific Adaptations Required

### 1. requirements-jetson.txt (Create New)

```txt
# JetPack 6.x (L4T 36.x) matched versions
# Install from NVIDIA JetPack / NGC, NOT PyPI for torch

# Core
opencv-python==4.10.0.84
numpy==1.26.4
ultralytics==8.4.41
simple-pid==2.0.1
pymavlink==2.4.49
pyserial==3.5

# PyTorch - MUST use JetPack version
# pip install torch torchvision --index-url https://pytorch.org/whl/cu121
# OR from NVIDIA NGC: nvcr.io/nvidia/pytorch:24.01-py3
# torch==2.5.1+cu124 (JetPack 6.0)
# torchvision==0.20.1+cu124

# GStreamer (system packages)
# apt install gstreamer1.0-tools gstreamer1.0-plugins-good \
#   gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
#   gstreamer1.0-libav libgstreamer1.0-dev \
#   libgstreamer-plugins-base1.0-dev

# TensorRT (from JetPack)
# python3 -c "import tensorrt"  # Should work after JetPack install

# SRT
# pip install pysrt  # or use GStreamer srt plugin
```

### 2. Camera Source Adaptation (`modules/yolo11_detector/source.py`)

**Current (Windows MSMF):**
```python
cap = cv2.VideoCapture(index)  # MSMF backend
```

**Jetson (V4L2/Argus):**
```python
# USB camera
cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

# CSI camera (Argus) - use GStreamer
cap = cv2.VideoCapture(
    "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink",
    cv2.CAP_GSTREAMER
)
```

### 3. RTSP Server GStreamer Hardware Encoding

**Current (Software MJPEG fallback):**
```python
# Falls back to MJPEG HTTP on Windows
```

**Jetson (Hardware H.264):**
```python
# Enable in rtsp_server.py _try_gstreamer():
pipeline = (
    "appsrc ! videoconvert ! nvvidconv ! "
    "nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! "
    "h264parse ! rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=127.0.0.1 port=5600"
)
```

### 4. Serial Port Configuration

**Current (Windows):**
```python
# Auto-detects COM ports
```

**Jetson:**
```bash
# Default UART for Pixhawk
/dev/ttyTHS1  # J17 header on Jetson Orin

# Udev rule needed:
# sudo usermod -a -G dialout $USER
# echo 'KERNEL=="ttyTHS1", MODE="0666"' | sudo tee /etc/udev/rules.d/99-ttyTHS1.rules
```

### 5. Camera Intrinsics

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

## File Transfer Script (Reference)

```bash
#!/bin/bash
# deploy_to_jetson.sh - Run from project root on host machine

JETSON_USER="ubuntu"
JETSON_HOST="jetson.local"
JETSON_PATH="/home/ubuntu/autonomous-drone"

# Create deployment package
tar --exclude='.venv' --exclude='.idea' --exclude='.pytest_cache' \
    --exclude='.vtcode' --exclude='tests' --exclude='tools' \
    --exclude='benchmarks/distance/images' --exclude='benchmarks/distance/results' \
    --exclude='__pycache__' --exclude='*.pyc' \
    -czf deploy_package.tar.gz \
    autonomous_drone_main.py \
    modules/ \
    jetson/ \
    shared/ \
    YOLO/ \
    benchmarks/distance/manifest.json \
    requirements-jetson.txt

# Transfer
scp deploy_package.tar.gz ${JETSON_USER}@${JETSON_HOST}:${JETSON_PATH}/

# On Jetson:
# tar -xzf deploy_package.tar.gz
# python3 -m venv .venv
# source .venv/bin/activate
# pip install -r requirements-jetson.txt
```

---

## Summary

| Category | Files | Size | Deploy |
|----------|-------|------|--------|
| **REQUIRED** | 60+ | ~16 MB | ✅ YES |
| **OPTIONAL** | 5 | ~20 KB | ⚠️ If space |
| **TEST-ONLY** | 15 | ~120 KB | ❌ NO |
| **DEVELOPMENT** | 20+ | ~500 KB | ❌ NO |
| **NOT REQUIRED** | 5 | ~5 KB | ❌ NO |
| **UNKNOWN** | 1 | ~1 KB | 🔍 Investigate |

**Total deployment size: ~22 MB (including 5.6 MB model)**

**Key adaptations needed:**
1. `requirements-jetson.txt` with JetPack-matched PyTorch
2. Camera source: MSMF → V4L2/Argus GStreamer
3. RTSP: MJPEG → Hardware H.264 (nvv4l2h264enc)
4. Serial: COM port → `/dev/ttyTHS1`
5. Camera calibration: Recalibrate for Jetson camera