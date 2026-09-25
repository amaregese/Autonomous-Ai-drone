# SAFE_CLEANUP_CANDIDATES_V2.md

## Safe Cleanup Candidates - Evidence-Based (V2)

**Generated:** 2025-09-25 (Reconciled)  
**Based on:** All verification evidence from Tasks 1-10

---

## Classification Method

Every file classified using evidence from:
- Static imports (from `autonomous_drone_main.py` entry point)
- Runtime import trace (actual imports during execution)
- Configuration references (argparse, config files, manifest.json)
- Test imports (pytest test files)
- Tool imports (manual execution scripts)
- Non-Python references (md, json, yaml, sh, bat, ps1, txt)
- Git history (last modification, replacement evidence)

---

## CATEGORY A — SAFE TO REMOVE LATER

**Criteria:** No production imports, no runtime loading, no deployment requirement, no test requirement, no active documentation/tool dependency, no meaningful generated-data role, no current operational purpose.

| # | File | Evidence | Risk if Removed | Action |
|---|------|----------|-----------------|--------|
| 1 | `shared/__pycache__/sgc_config.cpython-312.pyc` | Stale bytecode; source file never existed; no git history | **ZERO** - Not a source file | `rm -f shared/__pycache__/sgc_config.cpython-312.pyc` |

### Audit Scripts (Generated During This Session)
| # | File | Evidence | Risk | Action |
|---|------|----------|------|--------|
| 2 | `check_count.py` | Created during this audit session | ZERO | `rm -f check_count.py` |
| 3 | `check_duplicates.py` | Created during this audit session | ZERO | `rm -f check_duplicates.py` |
| 4 | `verify_counts.py` | Created during this audit session | ZERO | `rm -f verify_counts.py` |

---

## CATEGORY B — ARCHIVE LEGACY (Move to archive/, Don't Delete)

**Criteria:** No production use, but historical/debug/reference value.

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

## CATEGORY C — KEEP (Required for Production/Testing/Deployment)

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

### Development/Tools (16 files - MUST KEEP for Development)
```
tools/*.py (16 files)
benchmarks/distance/images/ (10 images)
benchmarks/camera_calibration/ (target + debug)
benchmarks/distance/results/ (generated)
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

### Model/Data (11 files - MUST KEEP)
```
YOLO/yolo11n.pt
benchmarks/distance/images/live_*.jpg (10)
benchmarks/camera_calibration/checkerboard_9x6.png
```

### Package Markers (10 files - MUST KEEP)
```
All __init__.py files (required for Python imports)
```

---

## CATEGORY D — DO NOT TOUCH

Files where removal could affect architecture or deployment and more evidence is needed.

| # | File | Reason | Decision |
|---|------|--------|----------|
| 1 | `modules/drone_visualizer.py` (wrapper) | Thin wrapper (42 bytes). Could be inlined. | Keep as-is (abstraction layer) |
| 2 | `modules/drone.py` (wrapper) | Thin wrapper (41 bytes). Could be inlined. | Keep as-is (abstraction layer) |
| 3 | `modules/control.py` (wrapper) | Thin wrapper (42 bytes). Could be inlined. | Keep as-is (abstraction layer) |
| 4 | `modules/detector_yolo11.py` (wrapper) | Thin wrapper (42 bytes). Could be inlined. | Keep as-is (abstraction layer) |
| 5 | `modules/drone_backend/__init__.py` | Empty file (41 bytes). Package marker. | Required for Python imports |
| 6 | `modules/control_system/__init__.py` | Single export (42 bytes). Package marker. | Required for Python imports |
| 7 | `modules/yolo11_detector/__init__.py` | Empty file (42 bytes). Package marker. | Required for Python imports |
| 8 | `jetson/__init__.py` | Empty file. Package marker. | Required for Python imports |
| 9 | `jetson/communication/__init__.py` | Empty file. Package marker. | Required for Python imports |
| 10 | `jetson/streaming/__init__.py` | Empty file. Package marker. | Required for Python imports |
| 11 | `shared/__init__.py` | Empty file. Package marker. | Required for Python imports |
| 12 | `modules/visualizer_ui/__init__.py` | Empty file. Package marker. | Required for Python imports |
| 11 | `modules/lidar_backend/__init__.py` | Single import (42 bytes). Package marker. | Required for Python imports |
| 12 | `modules/detector_yolo11/__init__.py` | Empty file. Package marker. | Required for Python imports |
| 13 | `modules/yolo11_detector/__init__.py` | Empty file. Package marker. | Required for Python imports |

**Decision:** All wrapper files and `__init__.py` files are **intentional architecture** (abstraction layers, package markers). Keep as-is. Inlining would reduce modularity.

---

## Summary Table

| Category | Count | Action |
|----------|-------|--------|
| **Category A: Safe to Remove** | 4 | Remove stale bytecode + 3 audit scripts |
| **Category B: Archive Legacy** | 5 | Move to `archive/legacy/` after review |
| **Category C: Keep** | ~120 | No action |
| **Category D: Human Review** | 15 | Document as intentional architecture |

---

## Final Recommendation

**Do not delete any source files at this time.**

The only immediate cleanup:
```bash
# 1. Clean stale bytecode
rm -f shared/__pycache__/sgc_config.cpython-312.pyc

# 2. Remove audit scripts created during this session
rm -f check_count.py check_duplicates.py verify_counts.py

# 3. (After review) Archive legacy files
mkdir -p archive/legacy/vision_utils
git mv modules/navigation.py archive/legacy/
git mv modules/vision.py archive/legacy/
git mv modules/vision_utils/geometry.py archive/legacy/vision_utils/
git mv modules/vision_utils/legacy.py archive/legacy/vision_utils/
git mv modules/vision_utils/__init__.py archive/legacy/vision_utils/
```

**Total source files removed: 0** (only stale bytecode and temporary audit scripts).

The cost of accidentally removing a needed file far exceeds the storage savings.