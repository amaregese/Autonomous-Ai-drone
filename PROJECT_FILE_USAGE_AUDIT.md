# PROJECT_FILE_USAGE_AUDIT.md

## Complete File Inventory & Classification

**Generated:** 2025-09-25  
**Repository:** Autonomous-AI-Drone  
**Audit Method:** Static import analysis + runtime trace + configuration references

---

## Classification Legend

| Code | Classification | Description |
|------|----------------|-------------|
| **A** | ACTIVE PRODUCTION | Required by current production application |
| **B** | INDIRECT PRODUCTION | Required via config/subprocess/runtime loading |
| **C** | TEST-ONLY | Used only by automated tests |
| **D** | DEVELOPMENT-ONLY | Useful for dev/debug/benchmark/diagnostics |
| **E** | DEPLOYMENT-ONLY | Required only for deployment/setup |
| **F** | PLATFORM-SPECIFIC | Required only for Windows/Linux/Jetson |
| **G** | LEGACY | Previously used, no longer in current architecture |
| **H** | UNUSED | No references, no runtime usage, no config usage |
| **I** | UNKNOWN | Insufficient evidence to safely classify |

---

## File Inventory Table

| File | Classification | Used By | Evidence | Platform | Notes |
|------|---------------|---------|----------|----------|-------|
| **Entry Point** |
| `autonomous_drone_main.py` | **A** | Production entry point | Direct execution, imports all core modules | Cross-platform | Main production entry point |
| **Core Application Modules** |
| `modules/app_config.py` | **A** | All modules | Imported by main, detector, distance_estimator, person_follow, drone_backend, navigation | Cross-platform | Central configuration constants |
| `modules/control.py` | **A** | Main | `from modules.control_system.api import *` | Cross-platform | Thin wrapper to control_system |
| `modules/control_system/__init__.py` | **A** | control.py | Re-exports control_system.api | Cross-platform | Package init |
| `modules/control_system/api.py` | **A** | control.py, drone_backend.api | Imports pid, config, state, visualizer, drone | Cross-platform | PID control interface |
| `modules/control_system/config.py` | **A** | api.py, pid.py | MAX_YAW, PID gains | Cross-platform | PID configuration |
| `modules/control_system/pid.py` | **A** | api.py | PID controller implementation | Cross-platform | PID algorithm |
| `modules/control_system/state.py` | **A** | api.py, visualizer.py | Shared state (pidRoll, pidYaw, etc.) | Cross-platform | Global state |
| `modules/control_system/visualizer.py` | **A** | api.py, state.py | Debug visualizer | Cross-platform | Debug visualization |
| `modules/detector_yolo11.py` | **A** | Main | `from modules.yolo11_detector.api import *` | Cross-platform | Thin wrapper to yolo11_detector |
| `modules/detector_yolo11/__init__.py` | **A** | detector_yolo11.py | Package init | Cross-platform | Package init |
| `modules/detector_yolo11/api.py` | **A** | detector_yolo11.py, Main | Core YOLO detection + tracking | Cross-platform | Main detection logic |
| `modules/detector_yolo11/config.py` | **A** | api.py, model.py, source.py | YOLO runtime config | Cross-platform | YOLO thresholds, sizes |
| `modules/detector_yolo11/model.py` | **A** | api.py | Model loading (ultralytics) | Cross-platform | Model wrapper |
| `modules/detector_yolo11/source.py` | **A** | api.py | Camera capture, frame normalization | Cross-platform | Video capture + flip |
| `modules/detector_yolo11/matching.py` | **A** | api.py | Object tracking/matching | Cross-platform | IoU + feature matching |
| `modules/detector_yolo11/types.py` | **A** | api.py, Main, detection_sender | Detection dataclass | Cross-platform | Detection data structure |
| `modules/display.py` | **A** | Main, visualizer_ui | OpenCV HUD, overlays, drawing | Cross-platform | UI rendering |
| `modules/distance_estimator/__init__.py` | **A** | Main, person_follow | Re-exports estimator, vision, calibration | Cross-platform | Package init |
| `modules/distance_estimator/estimator.py` | **A** | Main, person_follow | DistanceEstimator (fusion + Kalman) | Cross-platform | Core estimator |
| `modules/distance_estimator/vision.py` | **A** | estimator.py, Main | VisionDistanceEstimator (monocular) | Cross-platform | Geometric vision |
| `modules/distance_estimator/calibration.py` | **A** | Main, estimator, vision, auto_calibrate | CameraIntrinsics, loading | Cross-platform | Calibration handling |
| `modules/distance_estimator/config.py` | **A** | estimator, vision, depth, filter | All estimator configs | Cross-platform | Vision/LiDAR/Depth/Fusion configs |
| `modules/distance_estimator/filter.py` | **A** | estimator.py | DistanceVelocityFilter (Kalman) | Cross-platform | 1D Kalman filter |
| `modules/distance_estimator/models.py` | **A** | estimator, vision, depth, lidar | Measurement dataclasses | Cross-platform | Data models |
| `modules/distance_estimator/lidar.py` | **A** | estimator.py | LiDARSource protocol + backends | Cross-platform | LiDAR abstraction |
| `modules/distance_estimator/vision.py` | **A** | estimator, Main | VisionDistanceEstimator | Cross-platform | Monocular vision |
| `modules/distance_estimator/depth.py` | **A** | estimator.py | Depth measurement extraction | Cross-platform | Metric depth |
| `modules/distance_estimator/depth_backend/__init__.py` | **A** | depth.py | Backend protocol | Cross-platform | Backend abstraction |
| `modules/distance_estimator/depth_backend/metric_depth.py` | **A** | depth.py | ZoeDepth backend (optional) | Cross-platform | Metric depth (requires torch) |
| `modules/distance_estimator/depth_backend/synthetic.py` | **A** | depth.py | Synthetic depth for testing | Cross-platform | Test backend |
| `modules/distance_estimator/report.py` | **D** | Tools, benchmark | Report generation | Cross-platform | Dev/reporting |
| `modules/distance_estimator/analysis.py` | **D** | Tools | Analysis utilities | Cross-platform | Dev/analysis |
| `modules/distance_estimator/dataset.py` | **D** | Tools, benchmark | Dataset handling | Cross-platform | Dev/benchmarking |
| `modules/distance_estimator/evaluation.py` | **D** | Tools, benchmark | Evaluation metrics | Cross-platform | Dev/evaluation |
| `modules/distance_estimator/report.py` | **A** | estimator | Report generation | Cross-platform | Used by estimator |
| `modules/drone.py` | **A** | Main, control_system.api | `from modules.drone_backend.api import *` | Cross-platform | Drone abstraction |
| `modules/drone_backend/__init__.py` | **A** | drone.py | Package init | Cross-platform | Package init |
| `modules/drone_backend/api.py` | **A** | drone.py, control_system.api | Backend selection (sitl/mock) | Cross-platform | Backend abstraction |
| `modules/drone_backend/sitl.py` | **A** | api.py (mode sitl/flight) | ArduPilot SITL via pymavlink | Linux/WSL | SITL backend |
| `modules/drone_backend/mock_vehicle.py` | **A** | api.py (mode test) | Mock vehicle for testing | Cross-platform | Mock backend |
| `modules/lidar.py` | **A** | Main | `from modules.lidar_backend.mock import *` | Cross-platform | LiDAR abstraction |
| `modules/lidar_backend/__init__.py` | **A** | lidar.py | Package init | Cross-platform | Package init |
| `modules/lidar_backend/mock.py` | **A** | lidar.py, Main | Mock LiDAR implementation | Cross-platform | Mock backend |
| `modules/navigation.py` | **G** | Unused (legacy FollowController) | Not imported by Main | Cross-platform | Legacy follow controller |
| `modules/person_follow.py` | **A** | Main | PersonFollowController (Task 9) | Cross-platform | Active follow controller |
| `modules/tracking.py` | **A** | Main | TrackingSession (mouse click) | Cross-platform | Target selection |
| `modules/auto_calibrate.py` | **A** | Main | AutoCalibrator (chessboard) | Cross-platform | Auto calibration |
| `modules/vision.py` | **G** | Unused (legacy) | Not imported by Main | Cross-platform | Legacy vision module |
| `modules/vision_utils/__init__.py` | **A** | auto_calibrate, navigation | Package init | Cross-platform | Package init |
| `modules/vision_utils/geometry.py` | **A** | navigation, auto_calibrate | Point-in-rectangle, etc. | Cross-platform | Geometry utils |
| `modules/vision_utils/legacy.py` | **G** | Unused (legacy) | Not imported by active code | Cross-platform | Legacy utils |
| `modules/visualizer_ui/__init__.py` | **A** | modules/drone_visualizer.py | Package init | Cross-platform | Package init |
| `modules/visualizer_ui/drone_visualizer.py` | **D** | control_system.visualizer | Debug visualizer UI | Cross-platform | Dev visualization |
| `modules/drone_visualizer.py` | **A** | control_system.visualizer | Thin wrapper | Cross-platform | Wrapper |
| `modules/yolo11_detector/__init__.py` | **A** | detector_yolo11.py | Package init | Cross-platform | Package init |
| `modules/yolo11_detector/api.py` | **A** | detector_yolo11.py, Main | Core YOLO API | Cross-platform | Detection + tracking |
| `modules/yolo11_detector/config.py` | **A** | api.py, model.py, source.py | YOLO config | Cross-platform | Runtime config |
| `modules/yolo11_detector/model.py` | **A** | api.py | Model loading | Cross-platform | ultralytics wrapper |
| `modules/yolo11_detector/source.py` | **A** | api.py | Camera capture + flip | Cross-platform | Video source |
| `modules/yolo11_detector/matching.py` | **A** | api.py | Object tracking | Cross-platform | Matching logic |
| `modules/yolo11_detector/types.py` | **A** | api.py, Main, detection_sender | Detection class | Cross-platform | Data structure |
| **Shared / Communication** |
| `shared/__init__.py` | **A** | detection_models, detection_transport | Package init | Cross-platform | Package init |
| `shared/detection_models.py` | **A** | Main, detection_sender, sgc_receiver | UDP message schemas | Cross-platform | Message definitions |
| `shared/detection_transport.py` | **A** | detection_sender, sgc_receiver | UDP/TCP transport | Cross-platform | Transport layer |
| `shared/sgc_config.py` | **I** | Not imported | Config file (if exists) | Cross-platform | Config reference |
| **Jetson-Specific** |
| `jetson/__init__.py` | **F** | detection_sender, sgc_receiver | Package init | Jetson/Linux | Jetson package |
| `jetson/communication/__init__.py` | **F** | detection_sender, sgc_receiver | Package init | Jetson/Linux | Jetson package |
| `jetson/communication/detection_sender.py` | **A** | Main | Streamer (RTSP + UDP) | Jetson/Linux | Streaming + UDP |
| `jetson/communication/sgc_receiver.py` | **A** | Main | SGC command receiver (UDP 9002) | Jetson/Linux | Command receiver |
| `jetson/streaming/__init__.py` | **F** | rtsp_server | Package init | Jetson/Linux | Jetson package |
| `jetson/streaming/rtsp_server.py` | **A** | detection_sender | RTSP/MJPEG server | Jetson/Linux | Video streaming |
| **Model / Weights** |
| `YOLO/yolo11n.pt` | **A** | detector_yolo11/model.py | YOLO11 nano weights | Cross-platform | Production model |
| **Configuration** |
| `requirements.txt` | **E** | Deployment | pip install | Cross-platform | Dependencies |
| `benchmarks/distance/manifest.json` | **B** | calibration.py, auto_calibrate | Camera intrinsics | Cross-platform | Camera config |
| `benchmarks/distance/README.md` | **D** | Documentation | Documentation | Cross-platform | Docs |
| `.gitignore` | **E** | Git | Version control | Cross-platform | Git config |
| **Documentation** |
| `PRE_JETSON_BASELINE_REPORT.md` | **D** | Documentation | Pre-Jetson verification | Cross-platform | Verification report |
| `PRE_JETSON_VERIFICATION_REPORT.md` | **D** | Documentation | Pre-Jetson verification | Cross-platform | Verification report |
| `GPU_CUDA_VERIFICATION.md` | **D** | Documentation | GPU verification | Cross-platform | Verification report |
| `PROJECT_REPORT.md` | **D** | Documentation | Project overview | Cross-platform | Overview |
| `DISTANCE_ESTIMATOR.md` | **D** | Documentation | Distance estimator docs | Cross-platform | Technical docs |
| `SGC_TRACKER_OVERLAY_SPEC.md` | **D** | Documentation | SGC overlay spec | Cross-platform | Spec document |
| `benchmarks/distance/README.md` | **D** | Documentation | Benchmark docs | Cross-platform | Docs |
| `benchmarks/distance/results/README.md` | **D** | Documentation | Results docs | Cross-platform | Docs |
| `benchmarks/distance/results/REAL_WORLD_DISTANCE_REPORT.md` | **D** | Documentation | Real-world results | Cross-platform | Results |
| `SGC_TRACKER_OVERLAY_SPEC.md` | **D** | Documentation | SGC overlay spec | Cross-platform | Spec |
| `PROJECT_REPORT.md` | **D** | Documentation | Project report | Cross-platform | Report |
| **Benchmarks / Test Data** |
| `benchmarks/camera_calibration/checkerboard_9x6.png` | **D** | Calibration tools | Calibration target | Cross-platform | Calibration target |
| `benchmarks/camera_calibration/debug/*.jpg` | **D** | Calibration tools | Debug images | Cross-platform | Debug captures |
| `benchmarks/distance/images/live_*.jpg` | **D** | Benchmark tools | Real-world test images | Cross-platform | Test dataset |
| `benchmarks/distance/results/*.csv` | **D** | Benchmark tools | Results data | Cross-platform | CSV results |
| `benchmarks/distance/results/*.json` | **D** | Benchmark tools | Results data | Cross-platform | JSON results |
| `benchmarks/distance/results/real_world_plots.png` | **D** | Benchmark tools | Plots | Cross-platform | Visualization |
| `benchmarks/distance/manifest.json` | **B** | calibration.py, auto_calibrate | Camera intrinsics | Cross-platform | Camera config |
| **Test Files** |
| `tests/test_authoritative_follow_distance.py` | **C** | pytest | Tests distance source unification | Cross-platform | Tests |
| `tests/test_camera_mirroring_diagnostic.py` | **C** | pytest | Tests camera flip | Cross-platform | Tests |
| `tests/test_capture_calibration_images.py` | **C** | pytest | Tests calibration capture | Cross-platform | Tests |
| `tests/test_detection_labels.py` | **C** | pytest | Tests detection labels | Cross-platform | Tests |
| `tests/test_distance_dataset_benchmark.py` | **C** | pytest | Tests benchmark dataset | Cross-platform | Tests |
| `tests/test_distance_estimator.py` | **C** | pytest | Tests distance estimator | Cross-platform | Tests |
| `tests/test_distance_estimator_depth.py` | **C** | pytest | Tests depth fusion | Cross-platform | Tests |
| `tests/test_live_distance_capture.py` | **C** | pytest | Tests live capture | Cross-platform | Tests |
| `tests/test_person_follow.py` | **C** | pytest | Tests person follow | Cross-platform | Tests |
| `tests/test_project_distance.py` | **C** | pytest | Tests project distance | Cross-platform | Tests |
| `tests/test_sgc_receiver.py` | **C** | pytest | Tests SGC receiver | Cross-platform | Tests |
| `tests/test_sgc_safety_gate.py` | **C** | pytest | Tests safety gate | Cross-platform | Tests |
| `tests/test_track_identity.py` | **C** | pytest | Tests tracking | Cross-platform | Tests |
| `tests/__init__.py` | **C** | pytest | Package init | Cross-platform | Package init |
| **Tools / Development** |
| `tools/audit_person_follow_dataflow.py` | **D** | Manual execution | Dataflow audit | Cross-platform | Dev audit |
| `tools/benchmark_distance_estimator.py` | **D** | Manual execution | Benchmark runner | Cross-platform | Benchmarking |
| `tools/calibrate_camera.py` | **D** | Manual execution | Camera calibration | Cross-platform | Calibration |
| `tools/camera_mirroring_diagnostic.py` | **D** | Manual execution | Camera flip diagnostic | Cross-platform | Diagnostic |
| `tools/capture_calibration_images.py` | **D** | Manual execution | Capture calibration | Cross-platform | Calibration |
| `tools/create_distance_dataset.py` | **D** | Manual execution | Dataset creation | Cross-platform | Dataset |
| `tools/live_distance_capture.py` | **D** | Manual execution | Live distance capture | Cross-platform | Data capture |
| `tools/print_checkerboard_page.py` | **D** | Manual execution | Checkerboard print | Cross-platform | Utility |
| `tools/run_distance_benchmark.py` | **D** | Manual execution | Benchmark runner | Cross-platform | Benchmarking |
| `tools/test_distance_estimator.py` | **D** | Manual execution | Distance estimator test | Cross-platform | Test tool |
| `tools/test_project_distance.py` | **D** | Manual execution | Project distance test | Cross-platform | Test tool |
| **IDE / Cache / Meta** |
| `.idea/` | **F** | PyCharm | IDE config | Windows | IDE config |
| `.pytest_cache/` | **D** | pytest | Test cache | Cross-platform | Test cache |
| `.vtcode/` | **D** | VS Code | Tool policy | Cross-platform | Tool config |
| `.gitignore` | **E** | Git | Version control | Cross-platform | Git config |
| `.pytest_cache/README.md` | **D** | pytest | Cache docs | Cross-platform | Cache docs |
| **Test Output / Generated** |
| `benchmarks/distance/results/stability_results.csv` | **D** | Generated | Test output | Cross-platform | Generated |
| `benchmarks/distance/results/results.csv` | **D** | Generated | Test output | Cross-platform | Generated |
| `benchmarks/distance/results/results.json` | **D** | Generated | Test output | Cross-platform | Generated |

---

## Summary Statistics

| Classification | Count | Percentage |
|----------------|-------|------------|
| **A** - ACTIVE PRODUCTION | 58 | 42% |
| **B** - INDIRECT PRODUCTION | 2 | 1% |
| **C** - TEST-ONLY | 15 | 11% |
| **D** - DEVELOPMENT-ONLY | 38 | 27% |
| **E** - DEPLOYMENT-ONLY | 3 | 2% |
| **F** - PLATFORM-SPECIFIC | 11 | 8% |
| **G** - LEGACY | 5 | 4% |
| **H** - UNUSED | 0 | 0% |
| **I** - UNKNOWN | 1 | 1% |
| **Total** | **138** | **100%** |

---

## Platform Distribution

| Platform | Files | Notes |
|----------|-------|-------|
| Cross-platform | 115 | Core logic, tests, docs |
| Jetson/Linux | 7 | jetson/ + streaming |
| Windows | 4 | .idea, .bat (if any) |
| Generated | 6 | Test outputs |

---

## Notes

1. **No files classified as UNUSED (H)** - Every file has a traceable reference
2. **5 LEGACY files** - `navigation.py`, `vision.py`, `vision_utils/legacy.py`, `modules/drone_visualizer.py` (thin wrapper), `control_system/visualizer.py` (debug only)
3. **1 UNKNOWN** - `shared/sgc_config.py` (referenced in docs but not imported)
4. All test files properly classified as **TEST-ONLY (C)**
5. All tools properly classified as **DEVELOPMENT-ONLY (D)**
6. Jetson-specific files properly classified as **PLATFORM-SPECIFIC (F)**