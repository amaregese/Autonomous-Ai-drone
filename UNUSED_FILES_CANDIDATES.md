# UNUSED_FILES_CANDIDATES.md

## Unused File Candidates Analysis

**Status:** After comprehensive static analysis + runtime trace + configuration review, **NO files are classified as definitively UNUSED (H)**.

Every file in the repository has at least one traceable reference path (import, configuration, test, documentation, or subprocess).

---

## Files Requiring Human Review (Classification I - UNKNOWN)

| File | Current Classification | Why Uncertain | Risk if Removed |
|------|----------------------|---------------|-----------------|
| `shared/sgc_config.py` | **I** - UNKNOWN | Referenced in documentation (`SGC_TRACKER_OVERLAY_SPEC.md`) but not imported by any Python module. May be loaded by SGC ground station (external process) or intended for future use. | **LOW** - If SGC loads it externally, removal breaks SGC integration. If purely documentation artifact, safe to remove. |

---

## Files Often Mistaken as Unused (But Are Actually Used)

| File | Actual Classification | Evidence of Use |
|------|----------------------|-----------------|
| `modules/drone_visualizer.py` | **A** (Wrapper) | Imported by `modules/control_system/visualizer.py` as thin wrapper |
| `modules/control_system/visualizer.py` | **D** (Dev) | Imported by `modules/control_system/api.py` for debug visualization |
| `modules/navigation.py` | **G** (Legacy) | Contains `FollowController` - NOT imported by Main (replaced by `person_follow.py`) |
| `modules/vision.py` | **G** (Legacy) | Old vision module - NOT imported by Main |
| `modules/vision_utils/legacy.py` | **G** (Legacy) | Old geometry functions - NOT imported by active code |
| `shared/sgc_config.py` | **I** (Unknown) | Referenced in docs but not imported |

---

## Files That Are NOT Unused (Common False Positives)

| File | Why It Might Seem Unused | Actual Usage |
|------|-------------------------|--------------|
| `modules/drone.py` | Just 1 line: `from modules.drone_backend.api import *` | **Active abstraction layer** - Main imports `drone`, not `drone_backend` directly |
| `modules/control.py` | Just 1 line: `from modules.control_system.api import *` | **Active abstraction layer** - Main imports `control` |
| `modules/detector_yolo11.py` | Just 1 line: `from modules.yolo11_detector.api import *` | **Active abstraction layer** - Main imports `detector_yolo11 as detector` |
| `modules/drone_backend/__init__.py` | Empty file | **Package marker** - Required for Python imports |
| `jetson/__init__.py` | Empty file | **Package marker** - Required for Python imports |
| `jetson/communication/__init__.py` | Empty file | **Package marker** - Required for Python imports |
| `jetson/streaming/__init__.py` | Empty file | **Package marker** - Required for Python imports |
| `shared/__init__.py` | Empty file | **Package marker** - Required for Python imports |
| `modules/yolo11_detector/__init__.py` | Empty file | **Package marker** - Required for Python imports |
| `modules/control_system/__init__.py` | 1 line export | **Package marker** |
| `modules/lidar_backend/__init__.py` | 1 line import | **Package marker** |
| `modules/lidar.py` | 1 line import | **Abstraction layer** - Main imports `lidar` |
| `modules/visualizer_ui/__init__.py` | Empty | **Package marker** |
| `modules/visualizer_ui/drone_visualizer.py` | Not directly in Main | **Used by** `modules/control_system/visualizer.py` |
| `modules/drone_visualizer.py` | 1 line import | **Wrapper** - Used by control_system.visualizer |
| `modules/vision_utils/__init__.py` | Empty | **Package marker** |
| `shared/__init__.py` | Empty | **Package marker** |

---

## Benchmark/Test Data Files (Not Code - But Referenced)

| File | Referenced By | Purpose |
|------|---------------|---------|
| `benchmarks/distance/manifest.json` | `calibration.py`, `auto_calibrate.py` | Camera intrinsics loading |
| `benchmarks/distance/images/live_*.jpg` | `tools/live_distance_capture.py`, benchmark tools | Test dataset |
| `benchmarks/camera_calibration/checkerboard_9x6.png` | `tools/calibrate_camera.py`, `tools/print_checkerboard_page.py` | Calibration target |
| `benchmarks/camera_calibration/debug/*.jpg` | `tools/calibrate_camera.py` | Debug captures |

---

## Documentation Files (Referenced in README/Guides)

| File | Likely Referenced By |
|------|---------------------|
| `DISTANCE_ESTIMATOR.md` | Developer documentation |
| `SGC_TRACKER_OVERLAY_SPEC.md` | SGC integration spec (references `shared/sgc_config.py`) |
| `PRE_JETSON_BASELINE_REPORT.md` | Verification artifacts |
| `PRE_JETSON_VERIFICATION_REPORT.md` | Verification artifacts |
| `GPU_CUDA_VERIFICATION.md` | Verification artifacts |
| `PROJECT_REPORT.md` | Project overview |

---

## Analysis Methodology

### Static Analysis Performed:
1. **Import graph traversal** from `autonomous_drone_main.py` (entry point)
2. **Recursive import resolution** for all reachable modules
3. **Configuration file parsing** for module references
4. **Test file analysis** for production module usage
5. **Tools analysis** for development module usage

### Runtime Trace:
- Application startup trace confirms all core imports succeed
- Only CUDA runtime error (RTX 5060 unsupported) - not import-related

### Configuration Review:
- `autonomous_drone_main.py` argparse references
- `requirements.txt` dependencies
- `benchmarks/distance/manifest.json` references in code

---

## Conclusion

**No files can be safely deleted without human review.**

The only file with truly unknown status is `shared/sgc_config.py` - it may be loaded by the external SGC ground station software (not part of this repository). All other files have clear traceable references.

### Recommended Action:
1. **Investigate `shared/sgc_config.py`** - Check if SGC ground station loads it
2. **Document LEGACY files** - Add deprecation notices to `navigation.py`, `vision.py`, `vision_utils/legacy.py`
3. **Keep all package `__init__.py` files** - Required for Python import system
4. **Keep abstraction layers** (`drone.py`, `control.py`, `detector_yolo11.py`, `lidar.py`) - They provide clean API boundaries

---

## Risk Assessment

| Action | Risk |
|--------|------|
| Delete `shared/sgc_config.py` | **MEDIUM** - May break SGC integration if loaded externally |
| Delete LEGACY files (`navigation.py`, `vision.py`, `vision_utils/legacy.py`) | **LOW** - Not imported, but may be referenced in docs or future work |
| Delete package `__init__.py` files | **HIGH** - Breaks Python imports |
| Delete abstraction layers (`drone.py`, `control.py`, etc.) | **MEDIUM** - Breaks Main imports, requires refactoring |
| Delete test files | **LOW** - But loses test coverage |
| Delete tool files | **LOW** - But loses dev/debug capabilities |
| Delete benchmark data | **LOW** - But loses validation datasets |