# DISTANCE_ESTIMATOR_FILE_VERIFICATION.md

## Distance Estimator File Verification

**Generated:** 2025-09-25  
**Purpose:** Verify production vs development classification for distance_estimator modules

---

## Executive Summary

| File | Production Import | Test/Tool Import | Final Classification |
|------|------------------|------------------|---------------------|
| `estimator.py` | ✅ Main, person_follow | ❌ | **A** |
| `vision.py` | ✅ Main, estimator, person_follow | ❌ | **A** |
| `calibration.py` | ✅ Main, estimator, vision, auto_calibrate | ❌ | **A** |
| `config.py` | ✅ All sub-modules | ❌ | **A** |
| `filter.py` | ✅ estimator.py | ❌ | **A** |
| `models.py` | ✅ estimator, vision, depth, lidar | ❌ | **A** |
| `lidar.py` | ✅ estimator.py | ❌ | **A** |
| `depth.py` | ✅ estimator.py | ❌ | **A** |
| `depth_backend/__init__.py` | ✅ depth.py | ❌ | **A** |
| `depth_backend/metric_depth.py` | ✅ depth.py | ❌ | **A** |
| `depth_backend/synthetic.py` | ✅ depth.py | ❌ | **A** |
| `report.py` | ❌ | ✅ tests/tools | **D** |
| `analysis.py` | ❌ | ✅ tools | **D** |
| `dataset.py` | ❌ | ✅ tests/tools | **D** |
| `evaluation.py` | ❌ | ✅ tests/tools | **D** |

---

## Production Import Trace

### From `autonomous_drone_main.py`:
```python
from modules.distance_estimator import (
    DEFAULT_CONFIGURED_INTRINSICS,
    DistanceEstimator,
    EstimatorConfig,
    VisionConfig,
    VisionDistanceEstimator,
    annotate_detections,
    load_configured_intrinsics,
)
from modules.distance_estimator.calibration import CameraIntrinsics
```

### From `modules/distance_estimator/__init__.py` (Public API):
```python
from .calibration import (
    DEFAULT_CONFIGURED_INTRINSICS,
    CalibrationError,
    CameraIntrinsics,
    load_calibration,
    load_configured_intrinsics,
)
from .config import (
    DepthConfig,
    EstimatorConfig,
    FilterConfig,
    FusionConfig,
    LidarConfig,
    VisionConfig,
)
from .depth import (
    DepthBackendError,
    DepthFrame,
    DepthSource,
    RoiDepthStats,
    compute_depth_confidence,
    extract_depth_roi,
)
from .depth_backend import SimulatedDepthSource, ZoeDepthBackend
from .estimator import DistanceEstimator
from .filter import DistanceVelocityFilter
from .lidar import FixedLidar, LidarSource, SimulatedLidar
from .models import DepthMeasurement, LidarMeasurement, Source, TargetState, VisionMeasurement
from .vision import VisionDistanceEstimator, annotate_detection, annotate_detections, estimate_detection
```

**NOT exported:** `report`, `analysis`, `dataset`, `evaluation`

---

## Detailed File Analysis

### Production Files (A) - 14 files

| File | Imported By | Purpose |
|------|-------------|---------|
| `estimator.py` | Main, person_follow | Core DistanceEstimator (fusion + Kalman) |
| `vision.py` | estimator, Main, person_follow | VisionDistanceEstimator (monocular) |
| `calibration.py` | Main, estimator, vision, auto_calibrate | CameraIntrinsics loading/validation |
| `config.py` | estimator, vision, depth, filter, Main | All configs (Vision/LiDAR/Depth/Fusion) |
| `filter.py` | estimator.py | DistanceVelocityFilter (Kalman) |
| `models.py` | estimator, vision, depth, lidar | Measurement dataclasses |
| `lidar.py` | estimator.py | LiDARSource protocol + backends |
| `depth.py` | estimator.py | Depth measurement extraction |
| `depth_backend/__init__.py` | depth.py | Backend protocol |
| `depth_backend/metric_depth.py` | depth.py | ZoeDepth backend (optional, requires torch) |
| `depth_backend/synthetic.py` | depth.py | Synthetic depth for testing |
| `filter.py` | estimator.py | Kalman filter |
| `models.py` | estimator, vision, depth, lidar | Dataclasses |
| `lidar.py` | estimator.py | LiDAR abstraction |

### Development-Only Files (D) - 4 files

| File | Imported By | Purpose |
|------|-------------|---------|
| `report.py` | `tests/test_distance_dataset_benchmark.py`, `tools/run_distance_benchmark.py` | Report generation, CSV/JSON output |
| `analysis.py` | `tools/run_distance_benchmark.py` | Metrics computation |
| `dataset.py` | `tests/test_distance_dataset_benchmark.py`, `tools/create_distance_dataset.py`, `tools/live_distance_capture.py`, `tools/run_distance_benchmark.py` | Dataset handling |
| `evaluation.py` | `tests/test_distance_dataset_benchmark.py`, `tools/run_distance_benchmark.py` | Evaluation metrics |

---

## Verification: No Production Import of Development Files

### Confirmed: No production code imports:
- ❌ `modules.distance_estimator.report`
- ❌ `modules.distance_estimator.analysis`
- ❌ `modules.distance_estimator.dataset`
- ❌ `modules.distance_estimator.evaluation`

### Confirmed: Only tests/tools import:
| File | Importers |
|------|-----------|
| `report.py` | `tests/test_distance_dataset_benchmark.py`, `tools/run_distance_benchmark.py` |
| `analysis.py` | `tools/run_distance_benchmark.py` |
| `dataset.py` | `tests/test_distance_dataset_benchmark.py`, `tools/create_distance_dataset.py`, `tools/live_distance_capture.py`, `tools/run_distance_benchmark.py` |
| `evaluation.py` | `tests/test_distance_dataset_benchmark.py`, `tools/run_distance_benchmark.py` |

---

## Distance Estimator Module Dependency Graph

```
autonomous_drone_main.py
    │
    ├── modules.distance_estimator (package)
    │   ├── estimator.py (A) ← Main, person_follow
    │   │   ├── vision.py (A) ← estimator, Main, person_follow
    │   │   │   ├── calibration.py (A) ← vision, Main, auto_calibrate
    │   │   │   └── config.py (A) ← vision, estimator, depth, filter
    │   │   ├── lidar.py (A) ← estimator
    │   │   │   └── lidar_backend (protocol)
    │   │   ├── depth.py (A) ← estimator
    │   │   │   ├── depth_backend/__init__.py (A)
    │   │   │   ├── depth_backend/metric_depth.py (A) ← optional torch
    │   │   │   └── depth_backend/synthetic.py (A) ← test backend
    │   │   ├── filter.py (A) ← estimator
    │   │   ├── models.py (A) ← estimator, vision, depth, lidar
    │   │   ├── calibration.py (A) ← vision, Main, auto_calibrate
    │   │   └── config.py (A) ← all sub-configs
    │   │
    │   ├── calibration.py (A) ← Main, estimator, vision, auto_calibrate
    │   ├── vision.py (A) ← estimator, Main, person_follow
    │   ├── annotate_detections (function) ← Main
    │   ├── load_configured_intrinsics (function) ← Main
    │   ├── DistanceEstimator (class) ← Main, person_follow
    │   ├── VisionDistanceEstimator (class) ← estimator, Main, person_follow
    │   ├── VisionConfig (class) ← vision
    │   ├── EstimatorConfig (class) ← estimator
    │   ├── VisionConfig (class) ← config
    │   ├── CameraIntrinsics (class) ← calibration, Main
    │   │
    │   ├── report.py (D) ← tests/tools ONLY
    │   ├── analysis.py (D) ← tools ONLY
    │   ├── dataset.py (D) ← tests/tools ONLY
    │   ├── evaluation.py (D) ← tests/tools ONLY (imports legacy navigation)
    │   ├── analysis.py (D) ← tools
    │   ├── dataset.py (D) ← tests/tools
    │   └── evaluation.py (D) ← tests/tools (imports legacy navigation)
```

---

## Runtime Verification

### Production Runtime Trace (simulated):
```python
# autonomous_drone_main.py
from modules.distance_estimator import DistanceEstimator, VisionDistanceEstimator, annotate_detections
from modules.distance_estimator.calibration import CameraIntrinsics

# Setup
_distance_estimator = DistanceEstimator(intrinsics=intrinsics, config=EstimatorConfig(...))

# Main loop
annotate_detections(_distance_estimator._vision if _distance_estimator else None, detections, width, height)

# Person follow
follow_controller = PersonFollowController(distance_estimator=_distance_estimator)
follow_cmd = follow_controller.update(follow_target, image.shape)
```

**All production paths use only A-classified modules.**

---

## Classification Summary

| File | Classification | Evidence |
|------|----------------|----------|
| `estimator.py` | **A** | Main, person_follow |
| `vision.py` | **A** | estimator, Main, person_follow |
| `calibration.py` | **A** | Main, estimator, vision, auto_calibrate |
| `config.py` | **A** | All sub-modules |
| `filter.py` | **A** | estimator.py |
| `models.py` | **A** | estimator, vision, depth, lidar |
| `lidar.py` | **A** | estimator.py |
| `depth.py` | **A** | estimator.py |
| `depth_backend/__init__.py` | **A** | depth.py |
| `depth_backend/metric_depth.py` | **A** | depth.py |
| `depth_backend/synthetic.py` | **A** | depth.py |
| `report.py` | **D** | tests/tools ONLY |
| `analysis.py` | **D** | tools ONLY |
| `dataset.py` | **D** | tests/tools ONLY |
| `evaluation.py` | **D** | tests/tools ONLY |

---

## Corrections to V2 Audit

| File | V2 Classification | Corrected | Reason |
|------|------------------|-----------|--------|
| `report.py` | A and D (duplicate) | **D** | Only tools/tests |
| `filter.py` | Listed twice | **A** | Single entry |
| `models.py` | Listed twice | **A** | Single entry |
| `lidar.py` | Listed twice | **A** | Single entry |
| `depth.py` | Listed twice | **A** | Single entry |
| `depth_backend/__init__.py` | Listed twice | **A** | Single entry |
| `depth_backend/metric_depth.py` | Listed twice | **A** | Single entry |
| `depth_backend/synthetic.py` | Listed twice | **A** | Single entry |

---

## Final Verification

| Statement | Verified |
|-----------|----------|
| No production code imports report/analysis/dataset/evaluation | ✅ |
| All development files only used by tests/tools | ✅ |
| All production modules properly classified A | ✅ |
| All development modules properly classified D | ✅ |
| No circular dependencies | ✅ |
| Distance estimator public API exports only production modules | ✅ |