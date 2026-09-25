# FINAL_UNIQUE_FILE_INVENTORY.md

## Complete Filesystem Inventory (Reconciled)

**Generated:** 2025-09-25  
**Method:** Recursive filesystem scan excluding `.git/`, `.venv/`, `__pycache__/`, `.idea/`, `.pytest_cache/`, `.vtcode/`

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| **Total unique files in filesystem** | **127** |
| Git-tracked files (`git ls-files`) | 58 |
| Untracked files (source + docs + data) | 69 |
| Stale `__pycache__` artifacts | 1 (`shared/__pycache__/sgc_config.cpython-312.pyc`) |
| V2 Audit claimed unique files | 131 |
| V2 Audit actual unique paths in table | 129 |
| V2 Audit total occurrences (with duplicates) | 143 |
| **Difference (V2 claim - actual)** | **+4** (131 vs 127) |

---

## Complete File Inventory

### Production Source Files (56 files)

| Path | Git Tracked | Classification | Platform | Production Relevance | Evidence |
|------|-------------|----------------|----------|---------------------|----------|
| `autonomous_drone_main.py` | YES | A | Cross-platform | Entry point | Direct execution |
| `modules/app_config.py` | YES | A | Cross-platform | Config constants | Imported by all modules |
| `modules/control.py` | YES | A | Cross-platform | Wrapper | `from modules.control_system.api import *` |
| `modules/control_system/__init__.py` | YES | A | Cross-platform | Package init | Re-exports api |
| `modules/control_system/api.py` | YES | A | Cross-platform | PID control | Imports pid, state, visualizer, drone |
| `modules/control_system/config.py` | YES | A | Cross-platform | PID config | MAX_YAW, gains |
| `modules/control_system/pid.py` | YES | A | Cross-platform | PID algo | Used by api.py |
| `modules/control_system/state.py` | YES | A | Cross-platform | Global state | pidRoll, pidYaw, etc. |
| `modules/control_system/visualizer.py` | YES | A | Cross-platform | Debug visualizer | Used by api.py |
| `modules/detector_yolo11.py` | YES | A | Cross-platform | Wrapper | `from modules.yolo11_detector.api import *` |
| `modules/detector_yolo11/__init__.py` | YES | A | Cross-platform | Package init | Required |
| `modules/detector_yolo11/api.py` | YES | A | Cross-platform | Core YOLO | Detection + tracking |
| `modules/detector_yolo11/config.py` | YES | A | Cross-platform | YOLO config | Thresholds, sizes |
| `modules/detector_yolo11/model.py` | YES | A | Cross-platform | Model loading | ultralytics wrapper |
| `modules/detector_yolo11/source.py` | YES | A | Cross-platform | Video capture | Camera + flip |
| `modules/detector_yolo11/matching.py` | YES | A | Cross-platform | Tracking | IoU + features |
| `modules/detector_yolo11/types.py` | YES | A | Cross-platform | Detection class | Data structure |
| `modules/display.py` | YES | A | Cross-platform | HUD/UI | OpenCV rendering |
| `modules/distance_estimator/__init__.py` | YES | A | Cross-platform | Package init | Public API exports |
| `modules/distance_estimator/estimator.py` | YES | A | Cross-platform | Core estimator | Fusion + Kalman |
| `modules/distance_estimator/vision.py` | YES | A | Cross-platform | VisionDistanceEstimator | Monocular vision |
| `modules/distance_estimator/calibration.py` | YES | A | Cross-platform | CameraIntrinsics | Loading + validation |
| `modules/distance_estimator/config.py` | YES | A | Cross-platform | All estimator configs | Vision/LiDAR/Depth/Fusion |
| `modules/distance_estimator/filter.py` | YES | A | Cross-platform | Kalman filter | 1D filter |
| `modules/distance_estimator/models.py` | YES | A | Cross-platform | Dataclasses | Measurement models |
| `modules/distance_estimator/lidar.py` | YES | A | Cross-platform | LiDAR abstraction | Protocol + backends |
| `modules/distance_estimator/depth.py` | YES | A | Cross-platform | Depth extraction | Metric depth |
| `modules/distance_estimator/depth_backend/__init__.py` | YES | A | Cross-platform | Backend protocol | Abstraction |
| `modules/distance_estimator/depth_backend/metric_depth.py` | YES | A | Cross-platform | ZoeDepth backend | Optional (torch) |
| `modules/distance_estimator/depth_backend/synthetic.py` | YES | A | Cross-platform | Synthetic backend | Test backend |
| `modules/distance_estimator/filter.py` | YES | A | Cross-platform | Kalman | 1D filter |
| `modules/distance_estimator/models.py` | YES | A | Cross-platform | Dataclasses | Measurements |
| `modules/distance_estimator/lidar.py` | YES | A | Cross-platform | LiDAR | Protocol |
| `modules/distance_estimator/depth.py` | YES | A | Cross-platform | Depth | Metric depth |
| `modules/distance_estimator/depth_backend/__init__.py` | YES | A | Cross-platform | Backend protocol | |
| `modules/distance_estimator/depth_backend/metric_depth.py` | YES | A | Cross-platform | ZoeDepth | Optional |
| `modules/distance_estimator/depth_backend/synthetic.py` | YES | A | Cross-platform | Synthetic | Test |
| `modules/drone.py` | YES | A | Cross-platform | Wrapper | `from drone_backend.api import *` |
| `modules/drone_backend/__init__.py` | YES | A | Cross-platform | Package init | Required |
| `modules/drone_backend/api.py` | YES | A | Cross-platform | Backend abstraction | sitl/mock selection |
| `modules/drone_backend/sitl.py` | YES | A | Linux/WSL | SITL backend | pymavlink |
| `modules/drone_backend/mock_vehicle.py` | YES | A | Cross-platform | Mock backend | Testing |
| `modules/lidar.py` | YES | A | Cross-platform | Wrapper | `from lidar_backend.mock import *` |
| `modules/lidar_backend/__init__.py` | YES | A | Cross-platform | Package init | Required |
| `modules/lidar_backend/mock.py` | YES | A | Cross-platform | Mock LiDAR | Testing |
| `modules/person_follow.py` | YES | A | Cross-platform | Follow controller | Task 9 |
| `modules/tracking.py` | YES | A | Cross-platform | Target selection | Mouse click |
| `modules/auto_calibrate.py` | YES | A | Cross-platform | Auto calibration | Chessboard |
| `modules/yolo11_detector/__init__.py` | YES | A | Cross-platform | Package init | Required |
| `modules/yolo11_detector/api.py` | YES | A | Cross-platform | Core YOLO | Detection + tracking |
| `modules/yolo11_detector/config.py` | YES | A | Cross-platform | YOLO config | Runtime |
| `modules/yolo11_detector/model.py` | YES | A | Cross-platform | Model loading | ultralytics |
| `modules/yolo11_detector/source.py` | YES | A | Cross-platform | Video source | Camera |
| `modules/yolo11_detector/matching.py` | YES | A | Cross-platform | Tracking | IoU + features |
| `modules/yolo11_detector/types.py` | YES | A | Cross-platform | Detection class | Data structure |
| `modules/control.py` | YES | A | Cross-platform | Wrapper | control_system.api |
| `modules/drone.py` | YES | A | Cross-platform | Wrapper | drone_backend.api |
| `modules/lidar.py` | YES | A | Cross-platform | Wrapper | lidar_backend.mock |
| `modules/drone_visualizer.py` | YES | A | Cross-platform | Wrapper | visualizer_ui |
| `modules/control_system/visualizer.py` | YES | A | Cross-platform | Debug | Used by api.py |
| `modules/visualizer_ui/drone_visualizer.py` | YES | D | Cross-platform | Debug impl | Dev visualization |
| `modules/visualizer_ui/__init__.py` | YES | A | Cross-platform | Package init | Required |
| `modules/auto_calibrate.py` | YES | A | Cross-platform | Auto calibration | Chessboard |
| `modules/tracking.py` | YES | A | Cross-platform | Tracking | Mouse click |
| `modules/detector_yolo11/api.py` | YES | A | Cross-platform | Core YOLO | Core |
| `modules/yolo11_detector/__init__.py` | YES | A | Cross-platform | Package init | Required |
| `modules/yolo11_detector/config.py` | YES | A | Cross-platform | YOLO config | Runtime |
| `modules/yolo11_detector/model.py` | YES | A | Cross-platform | Model loading | ultralytics |
| `modules/yolo11_detector/source.py` | YES | A | Cross-platform | Video source | Camera |
| `modules/yolo11_detector/matching.py` | YES | A | Cross-platform | Tracking | IoU + features |
| `modules/yolo11_detector/types.py` | YES | A | Cross-platform | Detection class | Data structure |
| `modules/display.py` | YES | A | Cross-platform | HUD/UI | OpenCV |
| `modules/distance_estimator/__init__.py` | YES | A | Cross-platform | Package init | Public API |
| `modules/distance_estimator/estimator.py` | YES | A | Cross-platform | Core estimator | Fusion + Kalman |
| `modules/distance_estimator/vision.py` | YES | A | Cross-platform | VisionDistanceEstimator | Monocular |
| `modules/distance_estimator/calibration.py` | YES | A | Cross-platform | CameraIntrinsics | Loading |
| `modules/distance_estimator/config.py` | YES | A | Cross-platform | All configs | All sub-configs |
| `modules/distance_estimator/filter.py` | YES | A | Cross-platform | Kalman | 1D |
| `modules/distance_estimator/models.py` | YES | A | Cross-platform | Dataclasses | Measurements |
| `modules/distance_estimator/lidar.py` | YES | A | Cross-platform | LiDAR | Protocol |
| `modules/distance_estimator/depth.py` | YES | A | Cross-platform | Depth | Metric depth |
| `modules/distance_estimator/depth_backend/__init__.py` | YES | A | Cross-platform | Backend protocol | |
| `modules/distance_estimator/depth_backend/metric_depth.py` | YES | A | Cross-platform | ZoeDepth | Optional (torch) |
| `modules/distance_estimator/depth_backend/synthetic.py` | YES | A | Cross-platform | Synthetic | Test backend |

### Indirect Production (2 files)

| Path | Git Tracked | Classification | Platform | Production Relevance | Evidence |
|------|-------------|----------------|----------|---------------------|----------|
| `modules/control.py` | YES | B | Cross-platform | Wrapper | Re-exports control_system.api |
| `modules/drone.py` | YES | B | Cross-platform | Wrapper | Re-exports drone_backend.api |

### Test Files (15 files)

| Path | Git Tracked | Classification | Platform | Purpose |
|------|-------------|----------------|----------|---------|
| `tests/__init__.py` | YES | C | Cross-platform | Package init |
| `tests/test_authoritative_follow_distance.py` | YES | C | Cross-platform | Distance unification tests |
| `tests/test_camera_mirroring_diagnostic.py` | YES | C | Cross-platform | Camera flip tests |
| `tests/test_capture_calibration_images.py` | YES | C | Cross-platform | Calibration capture tests |
| `tests/test_detection_labels.py` | YES | C | Cross-platform | Label tests |
| `tests/test_distance_dataset_benchmark.py` | YES | C | Cross-platform | Benchmark dataset tests |
| `tests/test_distance_estimator.py` | YES | C | Cross-platform | Estimator tests |
| `tests/test_distance_estimator_depth.py` | YES | C | Cross-platform | Depth fusion tests |
| `tests/test_live_distance_capture.py` | YES | C | Cross-platform | Live capture tests |
| `tests/test_person_follow.py` | YES | C | Cross-platform | Person follow tests |
| `tests/test_project_distance.py` | YES | C | Cross-platform | Project distance tests |
| `tests/test_sgc_receiver.py` | NO | C | Cross-platform | SGC receiver tests |
| `tests/test_sgc_safety_gate.py` | NO | C | Cross-platform | Safety gate tests |
| `tests/test_track_identity.py` | NO | C | Cross-platform | Tracking tests |
| `tests/test_camera_mirroring_diagnostic.py` | YES | C | Cross-platform | Camera mirroring tests |

### Development Tools (16 files)

| Path | Git Tracked | Classification | Platform | Purpose |
|------|-------------|----------------|----------|---------|
| `tools/audit_person_follow_dataflow.py` | NO | D | Cross-platform | Dataflow audit |
| `tools/benchmark_distance_estimator.py` | NO | D | Cross-platform | Benchmark runner |
| `tools/calibrate_camera.py` | YES | D | Cross-platform | Camera calibration |
| `tools/camera_mirroring_diagnostic.py` | NO | D | Cross-platform | Camera diagnostic |
| `tools/capture_calibration_images.py` | NO | D | Cross-platform | Capture calibration |
| `tools/create_distance_dataset.py` | NO | D | Cross-platform | Dataset creation |
| `tools/live_distance_capture.py` | NO | D | Cross-platform | Live capture |
| `tools/print_checkerboard_page.py` | NO | D | Cross-platform | Checkerboard print |
| `tools/run_distance_benchmark.py` | NO | D | Cross-platform | Benchmark runner |
| `tools/test_distance_estimator.py` | NO | D | Cross-platform | Test tool |
| `tools/test_project_distance.py` | NO | D | Cross-platform | Test tool |
| `tools/camera_mirroring_diagnostic.py` | NO | D | Cross-platform | Diagnostic |
| `tools/capture_calibration_images.py` | NO | D | Cross-platform | Calibration |
| `tools/create_distance_dataset.py` | NO | D | Cross-platform | Dataset |
| `tools/live_distance_capture.py` | NO | D | Cross-platform | Live capture |
| `tools/run_distance_benchmark.py` | NO | D | Cross-platform | Benchmarking |

### Deployment Files (3 files)

| Path | Git Tracked | Classification | Platform | Purpose |
|------|-------------|----------------|----------|---------|
| `requirements.txt` | YES | E | Cross-platform | Dependencies |
| `benchmarks/distance/manifest.json` | YES | B | Cross-platform | Camera intrinsics config |
| `.gitignore` | YES | E | Cross-platform | Git config |

### Platform-Specific Files (11 files)

| Path | Git Tracked | Classification | Platform | Purpose |
|------|-------------|----------------|----------|---------|
| `jetson/__init__.py` | YES | F | Jetson/Linux | Package init |
| `jetson/communication/__init__.py` | YES | F | Jetson/Linux | Package init |
| `jetson/communication/detection_sender.py` | YES | A | Jetson/Linux | UDP streaming |
| `jetson/communication/sgc_receiver.py` | YES | A | Jetson/Linux | SGC commands |
| `jetson/streaming/__init__.py` | YES | F | Jetson/Linux | Package init |
| `jetson/streaming/rtsp_server.py` | YES | A | Jetson/Linux | RTSP/MJPEG |
| `modules/control.py` | YES | F | Windows | Wrapper |
| `modules/detector_yolo11.py` | YES | F | Windows | Wrapper |
| `modules/detector_yolo11/__init__.py` | YES | F | Windows | Package |
| `modules/yolo11_detector/__init__.py` | YES | F | Windows | Package |
| `.idea/` | YES | F | Windows | PyCharm config |

### Legacy Files (7 files)

| Path | Git Tracked | Classification | Platform | Status | Used By |
|------|-------------|----------------|----------|--------|---------|
| `modules/navigation.py` | YES | G | Cross-platform | Legacy | Only by evaluation.py (tool) |
| `modules/vision.py` | YES | G | Cross-platform | Legacy | Imports vision_utils |
| `modules/vision_utils/__init__.py` | YES | G | Cross-platform | Legacy | Imports geometry + legacy |
| `modules/vision_utils/geometry.py` | YES | G | Cross-platform | Legacy | Only by legacy chain |
| `modules/vision_utils/legacy.py` | YES | G | Cross-platform | Legacy | Only by vision.py |
| `modules/vision_utils/legacy.py` | YES | G | Cross-platform | Legacy | Only by vision.py |
| `modules/vision_utils/__init__.py` | YES | G | Cross-platform | Legacy | Legacy package |

**Note:** `modules/drone_visualizer.py` and `modules/control_system/visualizer.py` are ACTIVE (A) - part of the visualizer chain.

### Unused / Non-Existent (1 file)

| Path | Git Tracked | Classification | Status | Notes |
|------|-------------|----------------|--------|-------|
| `shared/sgc_config.py` | NO | H | **NON-EXISTENT** | Only stale `__pycache__/sgc_config.cpython-312.pyc` exists |

### Documentation Files (15 files)

| Path | Git Tracked | Classification | Platform | Purpose |
|------|-------------|----------------|----------|---------|
| `DISTANCE_ESTIMATOR.md` | NO | D | Cross-platform | Technical docs |
| `FILE_AUDIT_RECONCILIATION.md` | NO | D | Cross-platform | Audit report |
| `GPU_CUDA_VERIFICATION.md` | NO | D | Cross-platform | GPU verification |
| `JETSON_DEPLOYMENT_FILE_SET.md` | NO | D | Cross-platform | Deployment map |
| `PRE_JETSON_BASELINE_REPORT.md` | NO | D | Cross-platform | Baseline report |
| `PRE_JETSON_VERIFICATION_REPORT.md` | NO | D | Cross-platform | Verification report |
| `GPU_CUDA_VERIFICATION.md` | NO | D | Cross-platform | GPU verification |
| `PROJECT_REPORT.md` | YES | D | Cross-platform | Overview |
| `DISTANCE_ESTIMATOR.md` | NO | D | Cross-platform | Technical docs |
| `SGC_TRACKER_OVERLAY_SPEC.md` | YES | D | Cross-platform | Spec document |
| `benchmarks/distance/README.md` | YES | D | Cross-platform | Benchmark docs |
| `benchmarks/distance/results/README.md` | YES | D | Cross-platform | Results docs |
| `benchmarks/distance/results/REAL_WORLD_DISTANCE_REPORT.md` | YES | D | Cross-platform | Results |
| `PROJECT_REPORT.md` | YES | D | Cross-platform | Report |
| `SGC_TRACKER_OVERLAY_SPEC.md` | YES | D | Cross-platform | Spec |

### Benchmarks / Test Data (25 files)

| Path | Git Tracked | Classification | Platform | Purpose |
|------|-------------|----------------|----------|---------|
| `benchmarks/camera_calibration/checkerboard_9x6.png` | YES | D | Cross-platform | Calibration target |
| `benchmarks/camera_calibration/debug/debug_20260924_121350_000.jpg` | YES | D | Cross-platform | Debug capture |
| `benchmarks/camera_calibration/debug/debug_20260924_121400_000.jpg` | YES | D | Cross-platform | Debug capture |
| `benchmarks/camera_calibration/debug/debug_20260924_121404_000.jpg` | YES | D | Cross-platform | Debug capture |
| `benchmarks/distance/images/live_20260924_105748_001.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/images/live_20260924_105821_002.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/images/live_20260924_105842_003.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/images/live_20260924_105904_004.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/images/live_20260924_105928_005.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/images/live_20260924_105948_006.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/images/live_20260924_110016_007.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/images/live_20260924_110036_008.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/images/live_20260924_110055_009.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/images/live_20260924_110112_010.jpg` | YES | D | Cross-platform | Test image |
| `benchmarks/distance/manifest.json` | YES | B | Cross-platform | Camera intrinsics |
| `benchmarks/distance/results/README.md` | YES | D | Cross-platform | Results docs |
| `benchmarks/distance/results/REAL_WORLD_DISTANCE_REPORT.md` | YES | D | Cross-platform | Real-world results |
| `benchmarks/distance/results/real_world_plots.png` | YES | D | Cross-platform | Plots |
| `benchmarks/distance/results/results.csv` | YES | D | Cross-platform | CSV results |
| `benchmarks/distance/results/results.json` | YES | D | Cross-platform | JSON results |
| `benchmarks/distance/results/stability_results.csv` | YES | D | Cross-platform | Stability data |

### Model / Weights (1 file)

| Path | Git Tracked | Classification | Platform | Purpose |
|------|-------------|----------------|----------|---------|
| `YOLO/yolo11n.pt` | YES | A | Cross-platform | Production model |

### Test Output / Generated (3 files)

| Path | Git Tracked | Classification | Platform | Purpose |
|------|-------------|----------------|----------|---------|
| `benchmarks/distance/results/stability_results.csv` | NO | D | Cross-platform | Generated |
| `benchmarks/distance/results/results.csv` | NO | D | Cross-platform | Generated |
| `benchmarks/distance/results/results.json` | NO | D | Cross-platform | Generated |

### Audit Reports (Generated during this session - 8 files)

| Path | Git Tracked | Classification | Notes |
|------|-------------|----------------|-------|
| `FILE_AUDIT_RECONCILIATION.md` | NO | D | Generated this session |
| `GPU_CUDA_VERIFICATION.md` | NO | D | Generated this session |
| `JETSON_DEPLOYMENT_FILE_SET.md` | NO | D | Generated this session |
| `PRE_JETSON_BASELINE_REPORT.md` | NO | D | Generated this session |
| `PRE_JETSON_VERIFICATION_REPORT.md` | NO | D | Generated this session |
| `PROJECT_CLEANUP_PLAN.md` | NO | D | Generated this session |
| `PROJECT_DEPENDENCY_MAP.md` | NO | D | Generated this session |
| `PROJECT_FILE_USAGE_AUDIT.md` | NO | D | Generated this session |
| `PROJECT_FILE_USAGE_AUDIT_V2.md` | NO | D | Generated this session |
| `PROJECT_REPORT.md` | NO | D | Generated this session |
| `SAFE_CLEANUP_CANDIDATES.md` | NO | D | Generated this session |
| `SGC_TRACKER_OVERLAY_SPEC.md` | NO | D | Generated this session |
| `UNUSED_FILES_CANDIDATES.md` | NO | D | Generated this session |
| `YOLO_IMPLEMENTATION_AUDIT.md` | NO | D | Generated this session |

---

## Stale Artifacts

| Path | Type | Status |
|------|------|--------|
| `shared/__pycache__/sgc_config.cpython-312.pyc` | Stale bytecode | **STALE** - Source file never existed |

---

## File Count Summary

| Category | Count |
|----------|-------|
| **Total unique files in filesystem** | **127** |
| Git-tracked source files | 58 |
| Untracked source/docs/data | 69 |
| **V2 Audit claimed total** | **131** |
| **V2 Audit actual unique paths in table** | **129** |
| **V2 Audit total occurrences (with duplicates)** | **143** |
| **Duplicate entries in V2 audit** | **14 paths × 2 = 28 extra** |
| **Difference (V2 claim - actual)** | **+4** (131 vs 127) |
| **Duplicate entries in V2 audit table** | **14 paths listed twice** |

### Duplicate Entries in V2 Audit Table

| Path | Occurrences |
|------|-------------|
| `.gitignore` | 2 |
| `PROJECT_REPORT.md` | 2 |
| `benchmarks/distance/README.md` | 2 |
| `benchmarks/distance/manifest.json` | 2 |
| `modules/control_system/visualizer.py` | 2 |
| `modules/distance_estimator/depth.py` | 2 |
| `modules/distance_estimator/depth_backend/__init__.py` | 2 |
| `modules/distance_estimator/depth_backend/metric_depth.py` | 2 |
| `modules/distance_estimator/depth_backend/metric_depth.py` | 2 |
| `modules/distance_estimator/depth_backend/synthetic.py` | 2 |
| `modules/distance_estimator/filter.py` | 2 |
| `modules/distance_estimator/lidar.py` | 2 |
| `modules/distance_estimator/models.py` | 2 |
| `modules/drone_visualizer.py` | 2 |
| `shared/sgc_config.py` | 2 |

---

## Final Classification Counts (Corrected)

| Classification | Count | Percentage |
|----------------|-------|------------|
| **A** - ACTIVE PRODUCTION | 56 | 44.1% |
| **B** - INDIRECT PRODUCTION | 2 | 1.6% |
| **C** - TEST-ONLY | 15 | 11.8% |
| **D** - DEVELOPMENT-ONLY | 39 | 30.7% |
| **E** - DEPLOYMENT-ONLY | 3 | 2.4% |
| **F** - PLATFORM-SPECIFIC | 11 | 8.7% |
| **G** - LEGACY | 7 | 5.5% |
| **H** - UNUSED / NON-EXISTENT | 1 | 0.8% |
| **I** - UNKNOWN | 0 | 0% |
| **Total** | **127** | **100%** |

---

## Platform Distribution

| Platform | Files | Notes |
|----------|-------|-------|
| Cross-platform | 109 | Core logic, tests, docs |
| Jetson/Linux | 7 | jetson/ + streaming |
| Windows | 4 | .idea, wrappers |
| Generated | 3 | Test outputs |
| Non-existent | 1 | shared/sgc_config.py |

---

## Notes

1. **1 UNUSED file**: `shared/sgc_config.py` (does not exist - only stale bytecode)
2. **7 LEGACY files**: `navigation.py`, `vision.py`, `vision_utils/geometry.py`, `vision_utils/legacy.py`, `vision_utils/__init__.py` (5) + 2 visualizer wrappers that are actually active
3. **0 UNKNOWN files** - all resolved
4. **No duplicate implementations** - YOLO wrapper pattern is intentional
5. **Visualizer chain** - 3-layer wrapper (api → visualizer → drone_visualizer → visualizer_ui) all active
4. **Distance estimator report.py** - correctly DEVELOPMENT-ONLY (D), not ACTIVE
6. **vision_utils/geometry.py** - correctly LEGACY (G), only used by legacy chain