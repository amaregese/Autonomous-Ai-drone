# Distance Estimator (Tasks 1 + 2 + 3)

> **The geometric vision distance is the authoritative per-detection range in the active YOLO11 loop.**
> It is measured once per frame, attached to each detection, and consumed by follow control, local UI, console output, and shared tracking/UDP payloads.
>
> **Metric-depth integration remains optional and is not connected to autonomous flight control.**
>
> **Real-world accuracy validation remains an offline workflow.**

The subsystem lives in `modules/distance_estimator/` and is used by the active
YOLO11 loop plus offline tools under `tools/` and `benchmarks/distance/`.
The legacy distance pipeline in `modules/navigation.py` and its LiDAR path are
preserved for compatibility; they are not the active person-follow range path.

---

## 1. Why the old distance method is inaccurate

The legacy pipeline (`modules/navigation.py` + `modules/lidar_backend/mock.py`)
has several problems:

1. **Hard-coded fallback.** `estimate_distance_from_size()` returns a flat
   `3.0 m` whenever the focal length or bbox is invalid — a silent, fabricated
   distance fed directly into the controller.
2. **No validation.** Zero/negative bbox heights, bad focal lengths and
   nonsensical ranges are never rejected.
3. **Fixed 65/35 blend.** `blend_distance_estimate()` always mixes
   `0.65 * lidar + 0.35 * vision`, regardless of source quality.
4. **No confidence concept.** Everything is treated equally.
5. **Random mock LiDAR.** `read_lidar_distance()` returns
   `random.uniform(1.0, 5.0)` — non-deterministic noise.
6. **Blind 7-sample moving average.** A moving average adds lag but models no
   physics; it cannot estimate velocity and has no stability/quality gating.

Task 1 built a correct, isolated foundation. Task 2 added an optional metric
monocular-depth source and an offline accuracy benchmark. Task 3 added a
real-world validation workflow: a reproducible dataset format with
independently measured ground truth, a capture helper, and a real-image
benchmark that never fabricates results. The active application uses only the
validated geometric vision measurement for per-detection follow range; optional
metric depth and the offline benchmark remain separate.

---

## 2. Architecture

```
modules/distance_estimator/
    __init__.py     public API / exports (lightweight, no NN deps)
    models.py       VisionMeasurement, LidarMeasurement, DepthMeasurement,
                    TargetState, Source
    calibration.py  CameraIntrinsics + strict JSON loader
    vision.py       VisionDistanceEstimator (validated geometry + confidence)
    lidar.py        LidarSource protocol + SimulatedLidar / FixedLidar
    depth.py        DepthSource protocol, DepthFrame, ROI extraction,
                    depth confidence
    depth_backend/
        __init__.py       lazy backend package
        metric_depth.py   ZoeDepthBackend (optional, CPU, lazy-loaded)
        synthetic.py      SimulatedDepthSource (deterministic, no NN)
    filter.py       DistanceVelocityFilter (1-D Kalman: range + velocity)
    estimator.py    DistanceEstimator (orchestration / fusion / state)
    config.py       EstimatorConfig & sub-configs (isolated defaults)
    dataset.py      real-world manifest schema + validation (Task 3)
    analysis.py     metrics / distance bins / confidence-vs-error / outliers
    evaluation.py   RealImageEvaluator: runs methods on real images (Task 3)
    report.py       results.json / results.csv / .md report generation (Task 3)

tools/
    test_distance_estimator.py        offline demo (no flight)
    benchmark_distance_estimator.py   offline accuracy benchmark (Task 2)
    create_distance_dataset.py        interactive dataset capture helper (Task 3)
    run_distance_benchmark.py         real-world benchmark runner (Task 3)
    test_project_distance.py          observe-only YOLO11 -> distance diagnostic (Task 6)
                                      + stability test over 30-60 s (Task 7)

benchmarks/distance/
    README.md       real-image dataset format + collection guidance
    manifest.json   ground-truth manifest (empty until data is captured)
    images/         real frames (added by the capture tool)
    results/        generated: report, JSON, CSV, plots (never hand-edited)
```

Flow per call to `DistanceEstimator.update(...)`:

1. Validate / sanitize measurements (bounds, finite, typed).
2. Compute/verify source confidence from the measurements.
3. Detect sensor disagreement across all available sources.
4. Fuse or select a measurement (no fixed weighting ratio).
5. Feed the fused measurement into the Kalman filter.
6. Return a typed `TargetState`.

```python
estimator.update(
    vision_measurement=...,   # VisionMeasurement | None
    lidar_measurement=...,    # LidarMeasurement   | None
    depth_measurement=...,    # DepthMeasurement   | None
    dt=...,
)
```

The three sources are considered by validity and confidence. No source has a
hard-wired percentage.

---

## 3. Measurement models

All models are `slots=True` dataclasses in `models.py`; no unstructured
dictionaries flow through the estimator.

**`VisionMeasurement`** — `distance_m`, `confidence`, `bbox_height_px`,
`bbox_width_px`, `class_name`, `truncated`, `valid`, `timestamp`.

**`LidarMeasurement`** — `distance_m`, `confidence`, `valid`, `timestamp`.

**`DepthMeasurement`** — `depth_z_m`, `range_m`, `confidence`, `valid`,
`source`, `timestamp`, `n_valid_pixels`, `roi_size_px`, `depth_stdev_m`.

**`TargetState`** — `range_m`, `velocity_mps`, `confidence`, `source`,
`valid`, `fused_range_m`, `disagreement_m`, `note`, `timestamp`.

`Source` is an enum (`NONE / VISION / LIDAR / DEPTH / FUSED / PREDICTED`), so
the state origin is byte-exact and testable.

---

## 4. Metric-depth backend (Task 2)

`depth_backend/metric_depth.py` provides `ZoeDepthBackend`, a `DepthSource`
implementation:

| Property | Value |
|---|---|
| Model | ZoeDepth, `Intel/zoedepth-nyu-kitti` hub checkpoint |
| Backend | `transformers.ZoeDepthForDepthEstimation` + `AutoProcessor` |
| Expected input | RGB image, `HxWx3` array |
| Expected output | full-frame metric depth image (meters) + valid mask |
| Units | **meters**, camera-axis depth `Z` (not Euclidean range) |
| CPU | fully supported, `device="cpu"`; no CUDA / Jetson requirement |
| Model download | one-time hub download of the checkpoint (a few hundred MB) |
| Optional deps | `torch`, `transformers`, `numpy` — **lazy-loaded** |

**Optional / lazy loading.** `import distance_estimator` does **not** import
`torch`, `transformers`, `opencv` or `ultralytics`. Those are imported only
when `ZoeDepthBackend.load()` / `estimate()` is called by the caller. The
core estimator stays lightweight (verified by
`TestBackendLaziness`).

`depth_backend/synthetic.py` provides `SimulatedDepthSource`, a deterministic
backing source that returns a programmed constant depth (meters) with no
neural network — used by tests, the demo and the synthetic benchmark.

**Metric vs relative depth.** ZoeDepth outputs *metric* depth directly; a
relative-depth model output must never be relabeled as meters. If a chosen
model required scale/calibration alignment, that alignment must be explicit in
the backend — the current ZoeDepth checkpoint needs none beyond the one-time
weight download.

**Z-depth vs Euclidean range.** The model gives camera-axis depth `Z`. When
camera intrinsics are available, ROI extraction converts each valid pixel to
Euclidean range `Z / sqrt(1 + x_n^2 + y_n^2)` and reports `range_m`. When
intrinsics are absent, `range_m` is left `None` rather than invented.

---

## 5. ROI extraction

Depth is **not** read from the center pixel. `extract_depth_roi()` trims a
configurable margin (`roi_margin_fraction`, default 0.2) from every bbox edge,
keeping only the interior ROI where background/sky/ground/neighbor pixels
usually do not live:

```
┌─────────────────────┐
│     detection       │
│   ┌─────────────┐   │
│   │ depth ROI   │   │  <- interior only (margin-trimmed)
│   │  (median)   │   │
│   └─────────────┘   │
└─────────────────────┘
```

- Invalid / non-finite / non-positive depth pixels are rejected via the valid
  mask before statistics.
- The statistic is the **median** of valid depths (robust to outliers), plus
  median absolute deviation (MAD) and standard deviation for dispersion.
- A minimum valid fraction (`min_valid_fraction`, default 0.5) gates the ROI:
  if too little of the ROI contains usable depth, the statistic is rejected.
- The ROI strategy (`margin_fraction`, `min_valid_fraction`) is configurable
  via `DepthConfig`.

---

## 6. Depth confidence

`compute_depth_confidence()` produces a deterministic score in `[0.0, 1.0]`
from measurable ROI properties only — unit-testable without running any neural
network. It is a **measurement-quality** score, not a claimed accuracy.

| Component | Weight (default) | Basis |
|---|---|---|
| coverage | 0.35 | valid-pixel fraction inside the ROI |
| stability | 0.35 | inverse of relative MAD (lower dispersion = higher) |
| detection confidence | 0.20 | detector score, clamped to [0, 1] |
| distance band | 0.10 | 1.0 inside `[min, max]` m, else 0.5 |

An invalid ROI yields confidence `0.0`. Weights live in `DepthConfig`.

---

## 7. Vision calculation

`VisionDistanceEstimator.estimate(...)` implements the monocular pinhole
model:

```
distance_m = real_object_height_m * fy_px / object_height_px
```

- Rejects zero/negative/non-finite bbox heights/widths and invalid focal
  lengths.
- Rejects distances outside `[minimum_distance_m, maximum_distance_m]`
  (no silent `3.0 m` fallback — invalid inputs yield `valid=False`,
  `confidence=0.0`).
- Object height comes from the per-class table in `config.py`; if neither the
  class nor the default height is available, the measurement is invalid.
- Without camera calibration, `vision_geometry_available == False` and
  measurements are invalid rather than pretending `fy` is known.

Vision confidence is a deterministic score in `[0,1]` combining bbox size,
detection confidence, class availability and edge/truncation proximity, with a
consistency penalty for jumps away from the previous estimate.

---

## 8. Sensor disagreement

`DistanceEstimator._fuse(...)` computes the max **pairwise** disagreement
across all usable sources each cycle:

```
LiDAR  = 3.0 m, Vision = 3.1 m, Depth = 3.2 m  -> agree (0.2 m)  -> fused
LiDAR  = 3.0 m, Vision = 4.8 m, Depth = 4.6 m  -> disagree (1.8 m)
```

When disagreement exceeds `fusion.maximum_sensor_disagreement_m` (default
2.0 m):

- the measurements are **not averaged**;
- the highest-confidence source is selected;
- confidence is scaled by `fusion.disagreement_confidence_penalty`
  (default 0.5);
- `disagreement_m` and a human-readable `note` are exposed on `TargetState`.

Disagreement is never silently hidden; the threshold is configurable.

---

## 9. Fusion behavior

No fixed percentages (no 65/35 or otherwise):

- **All usable sources agree:** confidence-weighted average. A source's weight
  is proportional to its confidence, so high-confidence sources dominate while
  good measurements from others still contribute.
- **One usable source:** that source is used (`Source.VISION` / `LIDAR` /
  `DEPTH`).
- **Usable sources disagree:** see §8 — select, do not average, penalize,
  report.
- **None usable:** the Kalman filter is only predicted; the state reports
  `Source.PREDICTED` with decaying confidence.

Every threshold lives in `config.py` and is configurable.

---

## 10. Kalman filter

`filter.py` `DistanceVelocityFilter` is a self-contained 1-D **constant-velocity
Kalman filter** (pure Python, no external dependency):

```
state  x = [distance, velocity]
model  F = [[1, dt], [0, 1]],  H = [1, 0]
```

- Variable `dt` handled explicitly; non-finite measurements rejected.
- Measurement variance scales inversely with confidence (trustworthy sources
  trusted more), clamped to `[minimum, maximum]`.
- Filtered range clamped to `minimum_range_m`; `reset()` returns to pristine
  state.

---

## 11. Configuration

`config.py` contains `EstimatorConfig` with isolated sub-configs. These are
deliberately **not** the flight controller constants (`MAX_FOLLOW_DIST`,
`FORWARD_DEADBAND`, `FORWARD_BRAKE_ZONE`, `GAIN_FORWARD`), which remain
untouched in `modules/app_config.py`.

| Field | Default | Meaning |
|---|---|---|
| `vision.minimum_distance_m` | 0.25 | reject closer |
| `vision.maximum_distance_m` | 20.0 | reject farther |
| `vision.vision_confidence_threshold` | 0.30 | min usable vision confidence |
| `lidar.lidar_confidence_threshold` | 0.30 | min usable LiDAR confidence |
| `depth.roi_margin_fraction` | 0.20 | interior-ROI border trim |
| `depth.min_valid_fraction` | 0.50 | min valid-depth coverage in ROI |
| `depth.depth_confidence_threshold` | 0.30 | min usable depth confidence |
| `fusion.maximum_sensor_disagreement_m` | 2.0 | disagreement gate |
| `fusion.disagreement_confidence_penalty` | 0.5 | confidence scale on disagreement |
| `filter.process_variance_m2_s3` | 0.05 | process noise density |
| `filter.measurement_variance_m2` | 0.25 | nominal measurement variance |

**These are sensible starting numbers only — they require real-world
calibration and tuning before use on hardware.**

---

## 12. Benchmark methodology (Task 2)

`tools/benchmark_distance_estimator.py` compares five methods against known
ground-truth distances:

```
Method                 MAE       RMSE      Mean Error  Max Abs   MRE %
----------------------------------------------------------------------
Legacy bbox
Geometric vision
Metric depth
Fusion                 <- confidence-weighted, pre-filter
Filtered fusion        <- Kalman-filtered, post-filter
```

Metrics: per-target `absolute_error` and `relative_error_percent`, aggregated
into **MAE**, **RMSE**, **maximum absolute error** and **mean relative
error**.

Three properties are enforced:

1. **Synthetic vs real are kept apart.** Synthetic data (scripted ground truths
   at 1/2/3/4/5 m + deterministic noise) is always labelled *SYNTHETIC* in the
   output; it is test-fixture data, not real camera measurements, and no
   accuracy claim is derived from it alone.
2. **No fabricated results.** If a method cannot be evaluated because no
   suitable data exists, the table prints
   `N/A - insufficient benchmark data`. No method is ranked or declared a
   winner.
3. **Temporal benchmark.** A separate sequence (wobble around 3 m, a step
   3.0 -> 5.0 m, then invalid-measurement gaps) measures raw measurement noise,
   filtered noise, step-response delay, estimated velocity and behavior during
   invalid frames — to show whether the Kalman filter introduces excessive lag
   (responsiveness matters for control).

Real-world results are evaluated by the **Task 3** runner
(`tools/run_distance_benchmark.py`) against a real-image dataset in
`benchmarks/distance/`. Until such a dataset exists, the Task 2 synthetic
benchmark ends with a notice that no ground-truth camera dataset exists. The
Task 3 runner, on an empty manifest, reports:

```text
Real-world benchmark status: READY FOR DATA COLLECTION
Real-world accuracy results: NOT AVAILABLE (no evaluated samples yet)
```

---

## 13. Real-world dataset format & capture (Task 3)

Ground-truth distances MUST be **independently measured** (tape measure, laser
rangefinder, measured rig position, independently verified range). They are
never computed from bbox height, YOLO, ZoeDepth, monocular geometry or the
estimator itself. Each sample records its `measurement_method` so ground-truth
provenance is auditable.

`benchmarks/distance/` defines the real-image format (see its `README.md`):

```json
{
  "version": 1,
  "camera": {
    "name": "webcam-front",
    "frame_w": 640,
    "frame_h": 480,
    "intrinsics_source": "unknown",
    "intrinsics": {"fx": 1112.0, "fy": 1112.0, "cx": 320.0, "cy": 240.0}
  },
  "samples": [
    {
      "id": "sample_001",
      "image": "images/sample_001.jpg",
      "ground_truth_distance_m": 3.0,
      "class_name": "person",
      "measurement_method": "tape_measure",
      "bbox": {"x": 100, "y": 80, "width": 120, "height": 300},
      "scene": "outdoor",
      "lighting": "daylight",
      "pose": "standing",
      "notes": ""
    }
  ]
}
```

- `ground_truth_distance_m` must be >= 0.25 m, finite and **measured**, never
  guessed.
- `bbox` (optional but recommended) is `x, y, width, height` in native
  pixels; a sample without a bbox is rejected by the benchmark with reason
  `no bounding box`.
- `camera.intrinsics_source` must be one of
  `measured | calibrated | configured_default | unknown`; the benchmark never
  substitutes a default focal length. With `intrinsics: null`, the
  geometric-vision and legacy-bbox methods report `NOT_AVAILABLE`.
- Images are **not committed**; the shipped manifest is intentionally empty.

Samples are added without manual JSON editing:

```bash
python tools/create_distance_dataset.py --image photos/me_3m.jpg --distance 3.0
python tools/create_distance_dataset.py     # fully interactive
```

The capture tool copies the image into `images/` and updates `manifest.json`.
If no valid independently measured distance is provided, the sample is **not**
added — no measurement is ever invented.

Recommended starting distances: 1, 2, 3, 4, 5, 6, 8, 10 m with several samples
per distance and variation in pose/background/lighting/class.

---

## 14. Running the offline tests / demo / benchmark

Everything runs on CPU with **no** drone, camera, network or GPU (except the
optional ZoeDepth run, which downloads weights once):

```bash
# Unit tests (94 total: 41 Task 1 + 27 Task 2 + 26 Task 3)
python -m unittest discover -s tests -t . -p "test_*.py"

# Offline interactive demo
python tools/test_distance_estimator.py --frames 30 --movement approach

# Task 2 offline benchmark (synthetic + temporal + real-if-present)
python tools/benchmark_distance_estimator.py
python tools/benchmark_distance_estimator.py --seed 7 --frames 25
python tools/benchmark_distance_estimator.py --temporal-only

# Task 3 real-world benchmark (evaluates the manifest against real images)
python tools/run_distance_benchmark.py
python tools/run_distance_benchmark.py --enable-depth     # run ZoeDepth
python tools/run_distance_benchmark.py --no-plots

# Task 3 dataset capture helper
python tools/create_distance_dataset.py --image <file> --distance <measured-m>
```

### Task 6 observe-only diagnostic

`tools/test_project_distance.py` reuses the existing YOLO11 pipeline and the
isolated distance estimator (current configured intrinsics, `fy ~= 446.7 px`)
to print per-detection distances:

```bash
python tools/test_project_distance.py --source 0
python tools/test_project_distance.py --source 0 --frames 20 --no-window
python tools/test_project_distance.py --self-test       # headless mock run
```

Example output:

```text
person conf=0.86 bbox_height=182px distance=3.19m
```

It is strictly observe-only: no MAVLink command, no velocity/altitude/mode
change, no call into `modules.navigation` / `modules.lidar_backend`, no
takeoff/landing/RTL, no SGC change, and it never recalibrates or refits the
estimator. Covered by `tests/test_project_distance.py` with mock detections
(no physical camera required).

### Task 7 distance-stability test

`--stability` extends the same observe-only tool to verify that an
approximately stationary person produces a stable distance estimate: it
continuously detects the person, records one sample (frame number, detection
confidence, bbox height, estimated distance) per frame for a fixed duration
(default 30 s; the requirement is 30-60 s), then prints summary statistics and
saves the measurements + summary:

```bash
python tools/test_project_distance.py --source 0 --stability --duration 45
python tools/test_project_distance.py --source 0 --stability --duration 60 --no-window
python tools/test_project_distance.py --source 0 --stability --duration 60 --no-window --out-format csv
```

Per-frame log line (all four Task 7 fields):

```text
  frame 8 | 29.3 fps | person conf=0.92 bbox_height=317px distance=1.83m
```

At the end the tool reports:

```text
=== Distance stability summary ===
Valid measurements: 412
Minimum distance:   1.78 m
Maximum distance:   1.87 m
Mean distance:      1.83 m
Std deviation:      0.016 m
Relative std (of mean): 0.9%
Stability: STABLE (approximately stationary, low relative std-dev)
```

- Verdict is **STABLE** only when at least `--stability-min-samples` (default
  3) valid person frames were recorded and the sample std-dev is at most
  `--stability-max-relative-std` percent of the mean (default 15%). This is a
  diagnostic report, not a hard pass/fail gate.
- Results default to
  `benchmarks/distance/results/stability_results.json` (per-frame `samples`
  plus the `metrics` summary); `--out <path>` and `--out-format csv` change
  the target. Both contain only measured values — nothing is fabricated.
- The same estimator, `fy = 446.7` and YOLO11 pipeline are reused unchanged;
  no estimator parameter, camera calibration or flight code is touched.
- Statistics calculation is covered by automated tests
  (`TestStabilityMetrics`, `TestStabilityWrites`, `TestStabilityLoop` in
  `tests/test_project_distance.py`).

### Real-world benchmark output

`tools/run_distance_benchmark.py` evaluates every available method
(legacy bbox-height, geometric vision, metric depth, fusion, filtered fusion)
per usable sample and writes to `benchmarks/distance/results/`:

- `REAL_WORLD_DISTANCE_REPORT.md` — human-readable report (dataset summary,
  method table, distance-bin table, condition analysis, confidence-vs-error,
  outliers, limitations, flight status);
- `results.json` — full machine-readable payload;
- `results.csv` — long-format per-sample table
  (`sample_id, method, ground_truth_m, prediction_m, absolute_error_m,
  signed_error_m, relative_error, confidence, ...`);
- `real_world_plots.png` — predicted-vs-GT / error-vs-distance / error
  histograms (only when real data exists and matplotlib is available; the
  report notes if plots were skipped).

Metric rules enforced by the runner:

- Metrics are MAE, RMSE, bias (mean signed error), median absolute error,
  maximum absolute error and mean relative error %, computed only from actual
  evaluated pairs; zero/absent predictions are excluded, zero ground truth is
  rejected upstream.
- Distance-bin analysis reports only populated bins.
- Condition analysis (class / scene / lighting / pose / method) reports only
  categories actually present, with small-sample sizes stated.
- Confidence-vs-error is reported as an **empirical** observation (with Pearson
  r), never claimed as a calibrated probability.
- Outliers (default `absolute_error > 1.0 m` or relative error > 1%) are
  flagged with sample metadata and reported, never deleted.
- Missing image / missing bbox samples are counted as rejected, not skipped.
- Unavailable methods (no calibration, no depth backend) report
  `NOT_AVAILABLE` with a reason; depth is never replaced by synthetic values;
  LiDAR is never simulated (fusion uses only usable vision/depth sources).

---

## 15. CPU requirements

- Core estimator + synthetic depth + benchmark: pure Python + `numpy` only,
  CPU-only, no GPU/CUDA/Jetson requirement.
- Optional real metric-depth backend: `torch` + `transformers` on **CPU**
  (`device="cpu"`), plus a one-time checkpoint download. Explicitly no
  CUDA-specific requirement and no Jetson-specific code.

---

## 16. Known limitations

- Vision relies on a prior real-world object height per class; errors in that
  table directly scale the distance error.
- Single-camera pinhole geometry assumes an upright target and roughly level
  sensor axis; pitch/tilt and occlusion distort the result.
- ZoeDepth metric depth is affine-per-pixel correct but assumes its training
  data distribution; its useful range is roughly 0.5–25 m depending on scene,
  and production accuracy requires a depth-validated sensor.
- Depth gives camera-axis `Z`; `range_m` is only derived when intrinsics are
  installed.
- LiDAR is a simulation; a real range sensor has beam spread, multi-path and
  dropouts that this task does not model.
- Confidence weights, disagreement thresholds and Kalman noise terms are
  untuned defaults.
- **No real ground-truth camera dataset exists yet**, so real-world accuracy is
  unmeasured. The Task 3 dataset format, capture tool and benchmark runner are
  in place; once images with measured distances are collected the runner
  produces real results, and the synthetic Task 2 results are never mixed in.

---

## 17. Flight integration status

```text
Authoritative geometric vision range: CONNECTED
Metric-depth range: NOT CONNECTED
Legacy modules/navigation.py path: PRESERVED
```

The active `autonomous_drone_main.py` loop maintains one
`VisionDistanceEstimator` for the active frame size, annotates every YOLO11
detection once per frame, and passes those detection objects through selection,
`PersonFollowController`, `modules/display.py`, console status, and the shared
UDP/tracking payload. The selected detection identity is carried in the
movement and serialized detection records so a range cannot be copied to a
different object. Invalid measurements are serialized as unavailable rather
than replaced with a fabricated value.

The existing `modules/navigation.py` and `modules/lidar_backend/` code is kept
for compatibility and is not used to override the active geometric result.
MAVLink, takeoff/landing, yaw, emergency/RTL, SITL, RTSP, and SGC behavior remain
owned by the existing application control paths.

---

## 18. Next task (not implemented here)

- **Capture the first real distance dataset** with `tools/create_distance_dataset.py`
  (1–10 m, multiple samples per distance, calibrated camera) and run
  `tools/run_distance_benchmark.py` to obtain honest real-world accuracy
  numbers.
- Decide whether optional metric depth should be evaluated as a separate
  measurement source; it must not silently replace the authoritative geometric
  detection result.
- Field benchmarks to calibrate object heights, agreement thresholds and filter
  noise parameters.
