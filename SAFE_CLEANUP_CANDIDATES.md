# SAFE_CLEANUP_CANDIDATES.md

## Safe Cleanup Candidates - Evidence-Based

**Generated:** 2025-09-25 (Reconciled)  
**Based on:** FILE_AUDIT_RECONCILIATION.md + PROJECT_FILE_USAGE_AUDIT_V2.md

---

## Classification Method

Every file classified using evidence:
- **Static imports** (from entry point `autonomous_drone_main.py`)
- **Runtime import trace** (actual imports during execution)
- **Configuration references** (argparse, config files, manifest.json)
- **Test imports** (pytest test files)
- **Tool imports** (manual execution scripts)
- **Non-Python references** (md, json, yaml, sh, bat, ps1, txt)
- **Git history** (last modification, replacement evidence)

---

## Category 1: STRONG EVIDENCE OF UNUSED (Safe to Remove)

| # | File | Evidence | Risk if Removed |
|---|------|----------|-----------------|
| 1 | `shared/sgc_config.py` | **Does not exist as source file.** Only stale `__pycache__/sgc_config.cpython-312.pyc` exists. No git history. No code references. No documentation references (only in audit reports). | **ZERO** - File doesn't exist. Only stale bytecode to clean. |

### Action:
```bash
rm -f shared/__pycache__/sgc_config.cpython-312.pyc
```
**Safe. No source file to remove.**

---

## Category 2: LEGACY BUT POTENTIALLY USEFUL (Archive, Don't Delete)

| # | File | Current Classification | Why Legacy | Potential Value | Archive Action |
|---|------|----------------------|------------|-----------------|----------------|
| 1 | `modules/navigation.py` | **G** (LEGACY) | Replaced by `person_follow.py` (Task 9). Only used by `distance_estimator/evaluation.py` (benchmark tool). | Historical reference. Shows old FollowController implementation. | Move to `archive/legacy/navigation.py` |
| 2 | `modules/vision.py` | **G** (LEGACY) | Replaced by `distance_estimator/vision.py`. Thin wrapper importing `vision_utils`. | Historical reference. Shows old vision module structure. | Move to `archive/legacy/vision.py` |
| 3 | `modules/vision_utils/geometry.py` | **G** (LEGACY) | Only used by `vision_utils/legacy.py` and `vision.py` (both legacy). `auto_calibrate.py` does NOT use it. | Geometry utilities (point-in-rectangle). May be useful for future calibration. | Move to `archive/legacy/vision_utils_geometry.py` |
| 4 | `modules/vision_utils/legacy.py` | **G** (LEGACY) | Contains `process()` function. Only used by `vision.py` (legacy). | Legacy utilities. Reference for image processing patterns. | Move to `archive/legacy/vision_utils_legacy.py` |
| 5 | `modules/vision_utils/__init__.py` | **G** (LEGACY) | Only imports geometry and legacy. Used only by `vision.py` (legacy). | Package marker for legacy vision_utils. | Move to `archive/legacy/vision_utils/__init__.py` |

### Archive Procedure:
```bash
mkdir -p archive/legacy/vision_utils
git mv modules/navigation.py archive/legacy/
git mv modules/vision.py archive/legacy/
git mv modules/vision_utils/geometry.py archive/legacy/vision_utils/
git mv modules/vision_utils/legacy.py archive/legacy/vision_utils/
git mv modules/vision_utils/__init__.py archive/legacy/vision_utils/

# Add deprecation header to each moved file:
# """
# DEPRECATED - Archived YYYY-MM-DD
# This module is legacy and no longer used in production.
# See git history for original location.
# """
```

**Risk: LOW** - These files are not imported by any production code. Only `distance_estimator/evaluation.py` (a benchmark tool) imports `navigation.py`.

---

## Category 3: DUPLICATE IMPLEMENTATION (None Found)

| Investigation | Result |
|---------------|--------|
| Two YOLO implementations | **NOT duplicates** - wrapper + implementation pattern |
| Two vision.py files | **Different modules** - `modules/vision.py` (legacy) vs `distance_estimator/vision.py` (active) |
| Visualizer chain | **Wrapper chain** - api → visualizer → drone_visualizer → visualizer_ui (all active in call chain) |
| Two report.py entries | **Audit error** - single file listed twice |

**No actual duplicate implementations found.** The wrapper pattern is intentional architecture.

---

## Category 4: KEEP - Required for Production/Testing/Deployment

### Production Core (56 files - MUST KEEP)
```
autonomous_drone_main.py
modules/app_config.py
modules/control.py + control_system/*
modules/detector_yolo11.py + yolo11_detector/*
modules/display.py
modules/distance_estimator/* (except report.py, analysis.py, dataset.py, evaluation.py)
modules/drone.py + drone_backend/*
modules/lidar.py + lidar_backend/*
modules/person_follow.py
modules/tracking.py
modules/auto_calibrate.py
modules/yolo11_detector/*
jetson/communication/* + jetson/streaming/*
shared/detection_models.py + detection_transport.py
YOLO/yolo11n.pt
```

### Platform-Specific (11 files - MUST KEEP)
```
jetson/__init__.py
jetson/communication/__init__.py + detection_sender.py + sgc_receiver.py
jetson/streaming/__init__.py + rtsp_server.py
.idea/ (team IDE config)
```

### Test Suite (15 files - MUST KEEP)
```
tests/test_*.py (15 files)
tests/__init__.py
```

### Development Tools (16 files - MUST KEEP for Development)
```
tools/*.py (16 files)
benchmarks/distance/images/ (10 images)
benchmarks/camera_calibration/ (target + debug)
```

### Documentation (15 files - MUST KEEP)
```
All .md files in root and benchmarks/
```

### Configuration (3 files - MUST KEEP)
```
requirements.txt
benchmarks/distance/manifest.json
.gitignore
```

### Package Markers (10 files - MUST KEEP)
```
All __init__.py files (required for Python imports)
```

### Model/Data (11 files - MUST KEEP)
```
YOLO/yolo11n.pt
benchmarks/distance/images/live_*.jpg (10)
benchmarks/camera_calibration/checkerboard_9x6.png
```

---

## Category 5: HUMAN REVIEW REQUIRED

| # | File / Issue | Why Uncertain | Decision Needed |
|---|--------------|---------------|-----------------|
| 1 | `modules/drone_visualizer.py` (wrapper) | Thin wrapper (42 bytes). Could be inlined. | Keep as-is (abstraction layer) or inline? |
| 2 | `modules/drone.py` (wrapper) | Thin wrapper (41 bytes). Could be inlined. | Keep as-is (abstraction layer) or inline? |
| 3 | `modules/control.py` (wrapper) | Thin wrapper (42 bytes). Could be inlined. | Keep as-is (abstraction layer) or inline? |
| 4 | `modules/detector_yolo11.py` (wrapper) | Thin wrapper (42 bytes). Could be inlined. | Keep as-is (abstraction layer) or inline? |
| 5 | `modules/drone_backend/__init__.py` | Empty file (41 bytes). Package marker. | Required for Python imports. |
| 6 | `modules/control_system/__init__.py` | Single export (42 bytes). Package marker. | Required for Python imports. |
| 7 | `modules/yolo11_detector/__init__.py` | Empty file (42 bytes). Package marker. | Required for Python imports. |
| 8 | `jetson/__init__.py` | Empty file. Package marker. | Required for Python imports. |
| 9 | `jetson/communication/__init__.py` | Empty file. Package marker. | Required for Python imports. |
| 10 | `jetson/streaming/__init__.py` | Empty file. Package marker. | Required for Python imports. |
| 11 | `shared/__init__.py` | Empty file. Package marker. | Required for Python imports. |
| 12 | `modules/visualizer_ui/__init__.py` | Empty file. Package marker. | Required for Python imports. |
| 13 | `modules/lidar_backend/__init__.py` | Single import (42 bytes). Package marker. | Required for Python imports. |
| 14 | `modules/detector_yolo11/__init__.py` | Empty file. Package marker. | Required for Python imports. |
| 15 | `modules/yolo11_detector/__init__.py` | Empty file. Package marker. | Required for Python imports. |

**Decision:** All wrapper files and `__init__.py` files are **intentional architecture** (abstraction layers, package markers). Keep as-is. Inlining would reduce modularity.

---

## Summary Table

| Category | Count | Action |
|----------|-------|--------|
| **Category 1: Safe to Remove** | 1 (sgc_config.py bytecode only) | Remove stale bytecode |
| **Category 2: Archive Legacy** | 5 files | Move to `archive/legacy/` |
| **Category 3: Duplicate** | 0 | None found |
| **Category 4: Keep** | ~120 files | No action |
| **Category 5: Human Review** | 15 items | Document as intentional architecture |

---

## Files DEFINITELY Required for Jetson Deployment

| File | Reason |
|------|--------|
| `autonomous_drone_main.py` | Entry point |
| `modules/app_config.py` | Configuration |
| `modules/yolo11_detector/*` | YOLO detection |
| `modules/distance_estimator/*` (A/D split) | Distance estimation |
| `modules/person_follow.py` | Follow controller |
| `modules/display.py` | HUD rendering |
| `modules/drone.py` + `drone_backend/*` | MAVLink |
| `jetson/communication/*` | UDP + SGC |
| `jetson/streaming/rtsp_server.py` | RTSP/MJPEG |
| `shared/detection_models.py` + `detection_transport.py` | UDP transport |
| `YOLO/yolo11n.pt` | Model weights |
| `benchmarks/distance/manifest.json` | Camera intrinsics |

**NOT Required for Jetson:**
- `tests/`, `tools/`, `benchmarks/`
- `modules/navigation.py`, `modules/vision.py`, `modules/vision_utils/*`
- `.idea/`, `.pytest_cache/`, `.vtcode/`
- Audit reports (`.md` files generated during verification)

---

## Final Recommendation

### Immediate Actions (Safe):
```bash
# 1. Clean stale bytecode
rm -f shared/__pycache__/sgc_config.cpython-312.pyc

# 2. Archive legacy (after review)
mkdir -p archive/legacy/vision_utils
git mv modules/navigation.py archive/legacy/
git mv modules/vision.py archive/legacy/
git mv modules/vision_utils/geometry.py archive/legacy/vision_utils/
git mv modules/vision_utils/legacy.py archive/legacy/vision_utils/
git mv modules/vision_utils/__init__.py archive/legacy/vision_utils/
```

### No Deletions of Source Files:
- No production code removed
- No test files removed
- No tools removed
- No configuration removed
- No documentation removed
- Wrapper pattern preserved (intentional architecture)

**Total source files removed: 0** (only stale bytecode and archived legacy)