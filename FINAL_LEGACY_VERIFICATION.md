# FINAL_LEGACY_VERIFICATION.md

## Legacy File Verification

**Generated:** 2025-09-25  
**Purpose:** Verify each of the 7 purported legacy files

---

## Summary

| File | Verdict | Classification | Production Dependency | Notes |
|------|---------|----------------|----------------------|-------|
| `modules/navigation.py` | **LEGACY** | G | Test-only (evaluation.py) | Replaced by person_follow.py |
| `modules/vision.py` | **LEGACY** | G | None | Replaced by distance_estimator/vision.py |
| `modules/vision_utils/geometry.py` | **LEGACY** | G | None | Only used by legacy chain |
| `modules/vision_utils/legacy.py` | **LEGACY** | G | None | Only used by vision.py |
| `modules/vision_utils/__init__.py` | **LEGACY** | G | None | Legacy package init |
| `modules/drone_visualizer.py` | **ACTIVE (A)** | A | Production | Visualizer chain wrapper |
| `modules/control_system/visualizer.py` | **ACTIVE (A)** | A | Production | Debug visualizer |

**CORRECTION:** The previous audit misclassified `drone_visualizer.py` and `control_system/visualizer.py` as legacy. They are ACTIVE in the production visualizer chain.

---

## Detailed Verification

---

### 1. `modules/navigation.py`

#### Classification: **G - LEGACY**

#### Import Analysis
```bash
# Production imports: NONE
# Test/Tool imports:
modules/distance_estimator/evaluation.py:99: from modules.navigation import FollowController  # noqa: PLC0415
tools/benchmark_distance_estimator.py:67: from modules.navigation import FollowController
```

#### Non-Python References
- Referenced in `DISTANCE_ESTIMATOR.md` (documentation)
- Referenced in `PROJECT_FILE_USAGE_AUDIT_V2.md` (audit report)

#### Git History
- Present in early commits
- Not modified recently

#### Dynamic Imports / Subprocess / Config
- None found

#### Verdict: **LEGACY (G)**
- Only used by `distance_estimator/evaluation.py` (benchmark tool) and `tools/benchmark_distance_estimator.py` (tool)
- Replaced by `modules/person_follow.py` (Task 9 PersonFollowController)
- No production runtime dependency

---

### 2. `modules/vision.py`

#### Classification: **G - LEGACY**

#### Import Analysis
```bash
# Production imports: NONE
# Legacy chain imports:
modules/navigation.py:3: from modules import app_config, lidar, vision
modules/vision_utils/__init__.py:1: from modules.vision_utils.geometry import *
modules/vision_utils/__init__.py:2: from modules.vision_utils.legacy import process

# vision.py itself:
modules/vision.py:1: from modules.vision_utils import *
```

#### Non-Python References
- Referenced in documentation

#### Git History
- Present in early commits
- Not modified recently

#### Dynamic Imports / Subprocess / Config
- None found

#### Verdict: **LEGACY (G)**
- Thin wrapper importing `vision_utils` (legacy chain)
- Not imported by any production module
- Replaced by `modules/distance_estimator/vision.py` (active VisionDistanceEstimator)

---

### 3. `modules/vision_utils/geometry.py`

#### Classification: **G - LEGACY**

#### Import Analysis
```bash
# Used by:
modules/vision_utils/legacy.py:3: from modules.vision_utils.geometry import getCenter, getDelta
modules/vision_utils/__init__.py:1: from modules.vision_utils.geometry import *
modules/vision.py:1: from modules.vision_utils import *  # imports geometry via __init__

# NOT used by:
modules/auto_calibrate.py - DOES NOT import vision_utils
modules/distance_estimator/* - DOES NOT import vision_utils
```

#### Non-Python References
- None found

#### Git History
- Present in early commits

#### Dynamic Imports / Subprocess / Config
- None found

#### Verdict: **LEGACY (G)**
- Only used by legacy chain: `vision_utils/__init__.py` → `vision_utils/legacy.py` → `vision.py` → `navigation.py`
- `auto_calibrate.py` (active) does NOT use it
- `distance_estimator` (active) does NOT use it

---

### 4. `modules/vision_utils/legacy.py`

#### Classification: **G - LEGACY**

#### Import Analysis
```bash
# Used by:
modules/vision_utils/__init__.py:2: from modules.vision_utils.legacy import process
modules/vision.py:1: from modules.vision_utils import *  # imports legacy via __init__

# Legacy.py imports:
modules/vision_utils/legacy.py:3: from modules.vision_utils.geometry import getCenter, getDelta
```

#### Non-Python References
- None found

#### Git History
- Present in early commits

#### Dynamic Imports / Subprocess / Config
- None found

#### Verdict: **LEGACY (G)**
- Only used by legacy `vision.py` and `vision_utils/__init__.py`
- No production dependency

---

### 5. `modules/vision_utils/__init__.py`

#### Classification: **G - LEGACY**

#### Import Analysis
```bash
# Imports:
1: from modules.vision_utils.geometry import *
2: from modules.vision_utils.legacy import process

# Used by:
modules/vision.py:1: from modules.vision_utils import *
modules/navigation.py:3: from modules import app_config, lidar, vision  # imports vision which imports this
```

#### Non-Python References
- None found

#### Git History
- Present in early commits

#### Verdict: **LEGACY (G)**
- Legacy package init for vision_utils
- Only used by legacy `vision.py` → `navigation.py` chain

---

### 6. `modules/drone_visualizer.py`

#### Classification: **A - ACTIVE PRODUCTION (Wrapper)**

#### Import Analysis
```bash
# Imported by:
modules/control_system/visualizer.py:1: from modules.drone_visualizer import DroneVisualizer

# Content:
from modules.visualizer_ui.drone_visualizer import *
```

#### Runtime Chain
```
autonomous_drone_main.py
    → modules.control_system.api (imports visualizer)
        → modules.control_system.visualizer (imports DroneVisualizer from drone_visualizer)
            → modules.drone_visualizer (wrapper)
                → modules.visualizer_ui.drone_visualizer (actual implementation)
```

#### Runtime Trace
```
IMPORT: modules.drone_visualizer
IMPORT: modules.visualizer_ui.drone_visualizer
```

#### Non-Python References
- None

#### Git History
- Present in early commits
- Maintained as wrapper

#### Verdict: **ACTIVE PRODUCTION (A)**
- **NOT LEGACY** - Part of active visualizer chain
- Thin wrapper (42 bytes) providing abstraction layer
- Required by production runtime chain

---

### 7. `modules/control_system/visualizer.py`

#### Classification: **A - ACTIVE PRODUCTION (Debug Visualizer)**

#### Import Analysis
```bash
# Imported by:
modules/control_system/api.py:4: from modules.control_system.visualizer import (
    close_visualizer, draw_visualizer, initialize_debug_logs,
    update_telemetry_from_track, update_visualizer_target
)

# Imports:
from modules.drone_visualizer import DroneVisualizer
```

#### Runtime Usage in `control_system/api.py`
```python
# Line 59-65
def get_visualizer():
    return state.visualizer

def set_visualizer_status(message, color=(0, 255, 0), duration=0):
    if state.visualizer:
        state.visualizer.set_status(message, color, duration)

# Line 75-76
if state.visualizer:
    state.visualizer.update(state.movementYawAngle * 0.1, 0)

# Line 84-85
if state.visualizer:
    state.visualizer.update(0, state.movementRollAngle * 0.1)

# Line 87-88
if state.visualizer:
    state.visualizer.draw()
```

#### Runtime Trace
```
IMPORT: modules.control_system.visualizer
IMPORT: modules.drone_visualizer
IMPORT: modules.visualizer_ui.drone_visualizer
```

#### Non-Python References
- None

#### Git History
- Present in early commits
- Actively maintained

#### Verdict: **ACTIVE PRODUCTION (A)**
- **NOT LEGACY** - Debug visualizer used by `control_system/api.py`
- Called from `autonomous_drone_main.py` via `control_system.api`
- Provides debug visualization during runtime

---

## Legacy Dependency Chain (Complete)

```
modules/navigation.py (LEGACY)
    ↑ imports
modules/vision.py (LEGACY)
    ↑ imports
modules/vision_utils/__init__.py (LEGACY)
    ↑ imports
modules/vision_utils/legacy.py (LEGACY)
    ↑ imports
modules/vision_utils/geometry.py (LEGACY)
```

**This entire chain is isolated from production code.**

### Active Production Chain (Separate)

```
autonomous_drone_main.py (PRODUCTION)
    ↓
modules.distance_estimator.vision (ACTIVE - VisionDistanceEstimator)
    ↓
modules.distance_estimator.estimator (ACTIVE - DistanceEstimator)
    ↓
modules.person_follow (ACTIVE - PersonFollowController)
```

---

## Summary Table

| File | Classification | Production Dependency | Used By |
|------|----------------|----------------------|---------|
| `modules/navigation.py` | **G - LEGACY** | Test-only | `distance_estimator/evaluation.py` (tool), `tools/benchmark_distance_estimator.py` (tool) |
| `modules/vision.py` | **G - LEGACY** | None | `modules/navigation.py` (legacy) |
| `modules/vision_utils/geometry.py` | **G - LEGACY** | None | `modules/vision_utils/legacy.py`, `modules/vision_utils/__init__.py` |
| `modules/vision_utils/legacy.py` | **G - LEGACY** | None | `modules/vision_utils/__init__.py`, `modules/vision.py` |
| `modules/vision_utils/__init__.py` | **G - LEGACY** | None | `modules/vision.py` |
| `modules/drone_visualizer.py` | **A - ACTIVE** | Production | `modules/control_system/visualizer.py` |
| `modules/control_system/visualizer.py` | **A - ACTIVE** | Production | `modules/control_system/api.py` |

---

## Corrections to Previous Audit

| File | Previous Classification | Corrected | Reason |
|------|------------------------|-----------|--------|
| `modules/drone_visualizer.py` | G (Legacy) | **A (Active)** | Part of active visualizer chain |
| `modules/control_system/visualizer.py` | G (Legacy) | **A (Active)** | Used by control_system.api in production |

**The previous audit misclassified 2 files as legacy that are actually active production wrappers.**