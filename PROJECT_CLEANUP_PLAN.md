# PROJECT_CLEANUP_PLAN.md

## Project Cleanup Recommendations

**Based on:** PROJECT_FILE_USAGE_AUDIT.md + PROJECT_DEPENDENCY_MAP.md + UNUSED_FILES_CANDIDATES.md

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| **SAFE TO REMOVE** | 0 | None - no files meet criteria |
| **REVIEW BEFORE REMOVAL** | 5 | LEGACY files + 1 UNKNOWN |
| **KEEP** | 133 | All other files |

---

## SAFE TO REMOVE

**None.**

No files meet the criteria for safe removal (no references, no config usage, no test usage, no runtime trace).

---

## REVIEW BEFORE REMOVAL

These files appear legacy or unknown but require human confirmation before removal.

| # | File | Current Classification | Recommendation | Review Steps |
|---|------|----------------------|----------------|--------------|
| 1 | `modules/navigation.py` | **G** - LEGACY | **ARCHIVE** | 1. Confirm no SITL test imports it<br>2. Check if any tool imports it<br>3. Move to `archive/legacy/` if confirmed unused |
| 2 | `modules/vision.py` | **G** - LEGACY | **ARCHIVE** | 1. Confirm no imports anywhere<br>2. Check docs references<br>3. Move to `archive/legacy/` |
| 3 | `modules/vision_utils/legacy.py` | **G** - LEGACY | **ARCHIVE** | 1. Confirm no imports<br>2. Move to `archive/legacy/` |
| 4 | `modules/drone_visualizer.py` | **A** (Wrapper) | **KEEP** (but document) | Thin wrapper - keep for API stability |
| 5 | `shared/sgc_config.py` | **I** - UNKNOWN | **INVESTIGATE** | Check if SGC ground station loads it externally |

### Archive Procedure for LEGACY Files:
```bash
mkdir -p archive/legacy
git mv modules/navigation.py archive/legacy/
git mv modules/vision.py archive/legacy/
git mv modules/vision_utils/legacy.py archive/legacy/
# Add deprecation notice to each file header
```

---

## KEEP - All Other Files

### Core Production (58 files) - **MUST KEEP**
- `autonomous_drone_main.py` + all modules in `modules/` (except LEGACY)
- `jetson/` communication + streaming
- `shared/` detection models + transport

### Platform-Specific (11 files) - **MUST KEEP**
- `jetson/` - Required for Jetson deployment
- `.idea/` - PyCharm config (team IDE)

### Test Files (15 files) - **MUST KEEP**
- All `tests/test_*.py` - Regression prevention
- `tests/__init__.py` - Package marker

### Development/Tools (38 files) - **MUST KEEP**
- `tools/*.py` - Calibration, benchmarking, diagnostics, data capture
- Benchmark data in `benchmarks/distance/`

### Documentation (15 files) - **MUST KEEP**
- All `.md` files - Technical documentation
- `README.md` files in benchmarks

### Configuration (3 files) - **MUST KEEP**
- `requirements.txt`
- `benchmarks/distance/manifest.json`
- `.gitignore`

### Model/Data (11 files) - **MUST KEEP**
- `YOLO/yolo11n.pt` - Production model
- Benchmark images in `benchmarks/distance/images/`
- Calibration target `benchmarks/camera_calibration/checkerboard_9x6.png`

### Package Markers (10 files) - **MUST KEEP**
- All `__init__.py` files - Required for Python imports

---

## Cleanup Actions by Priority

### Priority 1: Code Hygiene (No File Removal)

| Action | Files | Effort |
|--------|-------|--------|
| Add deprecation headers to LEGACY files | `navigation.py`, `vision.py`, `vision_utils/legacy.py` | 5 min |
| Document abstraction layers (`drone.py`, `control.py`, etc.) | `drone.py`, `control.py`, `detector_yolo11.py`, `lidar.py` | 10 min |
| Investigate `shared/sgc_config.py` usage | `shared/sgc_config.py` | 15 min |

### Priority 2: Archive LEGACY (After Review)

```bash
# Create archive directory
mkdir -p archive/legacy

# Move legacy files
git mv modules/navigation.py archive/legacy/
git mv modules/vision.py archive/legacy/
git mv modules/vision_utils/legacy.py archive/legacy/

# Update any documentation references
# Commit with message: "Archive legacy modules (navigation, vision, vision_utils.legacy)"
```

### Priority 3: Jetson Deployment Prep (No Removal)

| Action | Details |
|--------|---------|
| Create `requirements-jetson.txt` | JetPack-matched PyTorch, TensorRT, GStreamer |
| Document Jetson-specific paths | Camera (`/dev/video*`), Serial (`/dev/ttyTHS1`) |
| Verify GStreamer plugins | `nvv4l2decoder`, `nvv4l2h264enc`, `nvvidconv` |
| Test SRT pipeline | `gst-launch-1.0 srtsink` |

---

## Files That Should NEVER Be Removed

| File | Reason |
|------|--------|
| `autonomous_drone_main.py` | Production entry point |
| `modules/app_config.py` | Central configuration |
| `modules/detector_yolo11/api.py` | Core detection logic |
| `modules/yolo11_detector/api.py` | YOLO + tracking |
| `modules/distance_estimator/estimator.py` | Core estimator |
| `modules/distance_estimator/vision.py` | Monocular vision |
| `modules/person_follow.py` | Task 9 follow controller |
| `modules/drone_backend/sitl.py` | SITL backend |
| `jetson/communication/detection_sender.py` | Streaming + UDP |
| `jetson/communication/sgc_receiver.py` | SGC commands |
| `jetson/streaming/rtsp_server.py` | RTSP/MJPEG |
| `shared/detection_models.py` | UDP schemas |
| `shared/detection_transport.py` | Transport layer |
| `YOLO/yolo11n.pt` | Production model |
| `requirements.txt` | Dependencies |
| All `__init__.py` | Python package structure |

---

## Jetson Deployment File Set

See `JETSON_DEPLOYMENT_FILE_SET.md` for complete list.

---

## Final Recommendation

**Do not delete any files at this time.**

The project has clean separation between:
- **Production code** (58 files)
- **Test code** (15 files)
- **Development tools** (38 files)
- **Documentation** (15 files)
- **Legacy code** (3 files - clearly identified)
- **1 unknown** (needs investigation)

The "cleanest" action is to:
1. **Archive the 3 LEGACY files** to `archive/legacy/`
2. **Investigate `shared/sgc_config.py`**
3. **Proceed to Jetson deployment** with the full source tree

The repository size (~10MB excluding `.venv`) is not a deployment concern. The cost of accidentally removing a needed file far exceeds the storage savings.