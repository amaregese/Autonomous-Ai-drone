# Real-world distance dataset (Task 3)

This directory holds a **real-image + independently measured ground-truth**
dataset for validating the isolated distance estimator
(`modules/distance_estimator/`) against reality.

Synthetic (scripted) fixture data from the Task 2 benchmark is generated inside
`tools/benchmark_distance_estimator.py` and is always labelled *synthetic*. It
does **not** live here and it is **never** reused as real-world accuracy.

## Layout

```
benchmarks/distance/
    README.md          this file
    manifest.json      machine-readable ground-truth manifest
    images/            JPG/PNG frames (added by the capture tool)
    results/           generated benchmark outputs (report, JSON, CSV, plots)
```

## Ground-truth rule

`ground_truth_distance_m` MUST be an independently measured distance:

- tape-measured distance,
- laser rangefinder,
- independently measured fixed target position,
- another independently verified range measurement,
- carefully measured experimental setup.

It is NEVER allowed to be computed from the image (no bbox-height, no YOLO, no
monocular geometry, no ZoeDepth, no estimator output). The purpose is to
compare the estimator against reality, not against another estimate derived
from the same image. Each sample records its `measurement_method` so the
ground-truth provenance is auditable.

## Manifest schema

```json
{
  "version": 1,
  "camera": {
    "name": "webcam-front",
    "frame_w": 640,
    "frame_h": 480,
    "intrinsics_source": "unknown",
    "intrinsics": null
  },
  "samples": [
    {
      "id": "sample_001",
      "image": "images/sample_001.jpg",
      "ground_truth_distance_m": 3.0,
      "class_name": "person",
      "measurement_method": "tape_measure",
      "bbox": {"x": 100, "y": 80, "width": 120, "height": 300},
      "detection_confidence": 0.95,
      "scene": "outdoor",
      "lighting": "daylight",
      "pose": "standing",
      "notes": ""
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | unique sample identifier |
| `image` | yes | path relative to this directory (e.g. `images/...`) |
| `ground_truth_distance_m` | yes | independently measured range to the target (m) |
| `class_name` | yes | target class |
| `measurement_method` | yes | how GT was measured (tape_measure, laser, rig, ...) |
| `bbox.x/y/width/height` | optional | target box in pixels; **without it the sample cannot be benchmarked** |
| `detection_confidence` | optional | detector score in (0,1], used for confidence (default 0.9) |
| `scene` / `lighting` / `pose` | optional | condition tags for per-condition analysis |
| `notes` | optional | free text |

### Camera intrinsics

The benchmark must know whether the camera is calibrated. `intrinsics_source`
must be one of:

```text
measured | calibrated | configured_default | unknown
```

`intrinsics` holds `{"fx","fy","cx","cy"}` when a real calibration exists;
otherwise `null`. The benchmark **never** silently substitutes a default focal
length. With `intrinsics: null`, the geometric-vision / legacy-bbox methods are
reported `NOT_AVAILABLE (no camera calibration)`.

If you only know the sensor, set `frame_w`/`frame_h` from the image size.

## Adding a sample (no manual JSON editing)

Use the capture helper:

```bash
python tools/create_distance_dataset.py --image photos/me_3m.jpg --distance 3.0
python tools/create_distance_dataset.py     # fully interactive, prompts for everything
```

It copies the image into `images/` and updates `manifest.json`. If no valid
independently measured distance is provided, the sample is **not** added (no
measurement is ever invented).

## Collection guidance

Recommended initial distances (collect several samples per distance):

```text
1 m   2 m   3 m   4 m   5 m   6 m   8 m   10 m
```

Vary target size, pose, background, lighting, camera angle and class. Only
record classes actually present in the images (person / vehicle / large object,
etc.). Use the same camera that will later be used for autonomy, and measure
its intrinsics if possible.

## Running the benchmark

```bash
# basic run (no metric-depth backend)
python tools/run_distance_benchmark.py

# with the optional ZoeDepth metric-depth backend (downloads the model once)
python tools/run_distance_benchmark.py --enable-depth

# skip plots / skip legacy method
python tools/run_distance_benchmark.py --no-plots --no-legacy
```

Outputs land in `benchmarks/distance/results/`:

- `REAL_WORLD_DISTANCE_REPORT.md`
- `results.json` (full machine-readable payload)
- `results.csv` (per-sample long table)
- `real_world_plots.png` (when matplotlib is available)

With an empty manifest the benchmark reports:

```text
Real-world benchmark status: READY FOR DATA COLLECTION
Real-world accuracy results: NOT AVAILABLE
```