# Real-World Distance Accuracy Report

Real-image evaluation of the isolated distance estimator (`modules/distance_estimator/`). Synthetic Task 2 results are intentionally absent here.

## Real-world benchmark status

```text
Real-world benchmark status: EVALUATED
Real-world accuracy results: see table below
```

## Dataset

| Property | Value |
|---|---|
| Total samples | 10 |
| Usable samples | 10 |
| Rejected samples | 0 |
| Target classes | person |
| Distance range | 1.150 - 2.500 m |
| Measurement methods | manual_measurement |
| Camera | live-camera |
| Image resolution | 640 x 480 (480 rows) |
| Camera intrinsics | present (fx=446.700, fy=446.700, cx=320.000, cy=240.000); source: configured_default |


**Intrinsics provenance:** fy fitted from 10 live samples (pinhole: mean d*h/1.3m, person height 1.3 m); fx=cx=cy assumed. NOT chessboard-calibrated.

No rejected samples.

## Baseline benchmark vs independent validation

### Baseline benchmark

The current 10 real-world samples were evaluated with the configured intrinsics exactly as shipped.
The intrinsics used include a focal length that was **previously fitted from these same 10 samples**
(`fy ≈ 446.7 px`, pinhole model `d = real_height * fy / bbox_height`, fitted mean across the 1.15–2.5 m
person-height dataset). Regenerating the results with the same fitted intrinsics is expected to reproduce
this baseline, but the effect of the fit on the metrics is not independent.
These 10 samples therefore count as an **initial benchmark / baseline**, not as independent validation.

### Independent validation

**Not yet available** — additional samples collected without parameter fitting are required.
(i.e. a second set of real images whose distances were measured independently and that were
never used to derive the focal length or any estimator parameter.)

## Benchmark

| Method | Samples | MAE | RMSE | Bias | Median AE | Max AE | MRE |
| ------ | ------: | --: | ---: | ---: | --------: | -----: | --: |
| Legacy bbox | 10 | 0.133 | 0.153 | -0.024 | 0.134 | 0.260 | 8.0% |
| Geometric vision | 10 | 0.133 | 0.153 | -0.024 | 0.134 | 0.260 | 8.0% |
| Metric depth | - | N/A | N/A | N/A | N/A | N/A | N/A (NOT_AVAILABLE: depth backend disabled (pass --enable-depth; requires torch + transformers)) |
| Fusion | 10 | 0.133 | 0.153 | -0.024 | 0.134 | 0.260 | 8.0% |
| Filtered fusion | 10 | 0.133 | 0.153 | -0.024 | 0.134 | 0.260 | 8.0% |

> Metrics are absolute errors in meters; MRE is mean relative error in percent.

## Distance-bin results (populated bins only)

### Legacy bbox

| Bin | Samples | MAE | RMSE | Bias | MRE |
| --- | ------: | --: | ---: | ---: | --: |
| 0-2 m | 5 | 0.127 | 0.155 | 0.091 | 9.9% |
| 2-4 m | 5 | 0.140 | 0.150 | -0.140 | 6.1% |

### Geometric vision

| Bin | Samples | MAE | RMSE | Bias | MRE |
| --- | ------: | --: | ---: | ---: | --: |
| 0-2 m | 5 | 0.127 | 0.155 | 0.091 | 9.9% |
| 2-4 m | 5 | 0.140 | 0.150 | -0.140 | 6.1% |

### Fusion

| Bin | Samples | MAE | RMSE | Bias | MRE |
| --- | ------: | --: | ---: | ---: | --: |
| 0-2 m | 5 | 0.127 | 0.155 | 0.091 | 9.9% |
| 2-4 m | 5 | 0.140 | 0.150 | -0.140 | 6.1% |

### Filtered fusion

| Bin | Samples | MAE | RMSE | Bias | MRE |
| --- | ------: | --: | ---: | ---: | --: |
| 0-2 m | 5 | 0.127 | 0.155 | 0.091 | 9.9% |
| 2-4 m | 5 | 0.140 | 0.150 | -0.140 | 6.1% |

## Condition analysis

### By class_name

| Value | Samples | MAE | RMSE | Bias | MRE |
| --- | ------: | --: | ---: | ---: | --: |
| person | 10 | 0.133 | 0.153 | -0.024 | 8.0% |

### By scene

| Value | Samples | MAE | RMSE | Bias | MRE |
| --- | ------: | --: | ---: | ---: | --: |
| live_camera | 10 | 0.133 | 0.153 | -0.024 | 8.0% |

### By lighting

| Value | Samples | MAE | RMSE | Bias | MRE |
| --- | ------: | --: | ---: | ---: | --: |
| unknown | 10 | 0.133 | 0.153 | -0.024 | 8.0% |

### By pose

| Value | Samples | MAE | RMSE | Bias | MRE |
| --- | ------: | --: | ---: | ---: | --: |
| unknown | 10 | 0.133 | 0.153 | -0.024 | 8.0% |

### By measurement_method

| Value | Samples | MAE | RMSE | Bias | MRE |
| --- | ------: | --: | ---: | ---: | --: |
| manual_measurement | 10 | 0.133 | 0.153 | -0.024 | 8.0% |

Small sample sizes are noted; no broad conclusion should be drawn from 1-2 images.

## Confidence vs error (empirical)

Analyzed method: **Fusion** (10 samples).

Pearson r (confidence vs absolute error): **-0.144**

| Confidence band | Samples | Mean abs error (m) |
| --- | ------: | --: |
| confidence < 0.5 | 0 | N/A |
| confidence >= 0.5 | 10 | 0.133 |

This is an empirical observation, NOT a calibrated probability that confidence equals accuracy.

## Outliers

### Legacy bbox — flagged by: absolute_error > 1 m OR relative_error > 1%

| Sample | Method | GT (m) | Pred (m) | Abs err (m) | Class | Confidence | Notes |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| live_001 | Legacy bbox | 1.600 | 1.683 | 0.083 | person | N/A | source=0; burst=1; 2026-09-24T10:57:48 |
| live_002 | Legacy bbox | 1.150 | 1.409 | 0.259 | person | N/A | source=0; burst=1; 2026-09-24T10:58:21 |
| live_003 | Legacy bbox | 2.000 | 1.949 | 0.051 | person | N/A | source=0; burst=1; 2026-09-24T10:58:42 |
| live_004 | Legacy bbox | 2.300 | 2.143 | 0.157 | person | N/A | source=0; burst=1; 2026-09-24T10:59:04 |
| live_005 | Legacy bbox | 2.500 | 2.286 | 0.214 | person | N/A | source=0; burst=1; 2026-09-24T10:59:28 |
| live_006 | Legacy bbox | 2.300 | 2.135 | 0.165 | person | N/A | source=0; burst=1; 2026-09-24T10:59:48 |
| live_007 | Legacy bbox | 2.100 | 1.989 | 0.111 | person | N/A | source=0; burst=1; 2026-09-24T11:00:16 |
| live_008 | Legacy bbox | 1.900 | 1.832 | 0.068 | person | N/A | source=0; burst=1; 2026-09-24T11:00:36 |
| live_009 | Legacy bbox | 1.500 | 1.478 | 0.022 | person | N/A | source=0; burst=1; 2026-09-24T11:00:55 |
| live_010 | Legacy bbox | 1.200 | 1.403 | 0.203 | person | N/A | source=0; burst=1; 2026-09-24T11:01:12 |

### Geometric vision — flagged by: absolute_error > 1 m OR relative_error > 1%

| Sample | Method | GT (m) | Pred (m) | Abs err (m) | Class | Confidence | Notes |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| live_001 | Geometric vision | 1.600 | 1.683 | 0.083 | person | 0.778 | source=0; burst=1; 2026-09-24T10:57:48 |
| live_002 | Geometric vision | 1.150 | 1.409 | 0.259 | person | 0.853 | source=0; burst=1; 2026-09-24T10:58:21 |
| live_003 | Geometric vision | 2.000 | 1.949 | 0.051 | person | 0.873 | source=0; burst=1; 2026-09-24T10:58:42 |
| live_004 | Geometric vision | 2.300 | 2.143 | 0.157 | person | 0.870 | source=0; burst=1; 2026-09-24T10:59:04 |
| live_005 | Geometric vision | 2.500 | 2.286 | 0.214 | person | 0.800 | source=0; burst=1; 2026-09-24T10:59:28 |
| live_006 | Geometric vision | 2.300 | 2.135 | 0.165 | person | 0.857 | source=0; burst=1; 2026-09-24T10:59:48 |
| live_007 | Geometric vision | 2.100 | 1.989 | 0.111 | person | 0.868 | source=0; burst=1; 2026-09-24T11:00:16 |
| live_008 | Geometric vision | 1.900 | 1.832 | 0.068 | person | 0.881 | source=0; burst=1; 2026-09-24T11:00:36 |
| live_009 | Geometric vision | 1.500 | 1.478 | 0.022 | person | 0.855 | source=0; burst=1; 2026-09-24T11:00:55 |
| live_010 | Geometric vision | 1.200 | 1.403 | 0.203 | person | 0.860 | source=0; burst=1; 2026-09-24T11:01:12 |

### Fusion — flagged by: absolute_error > 1 m OR relative_error > 1%

| Sample | Method | GT (m) | Pred (m) | Abs err (m) | Class | Confidence | Notes |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| live_001 | Fusion | 1.600 | 1.683 | 0.083 | person | 0.778 | source=0; burst=1; 2026-09-24T10:57:48 |
| live_002 | Fusion | 1.150 | 1.409 | 0.259 | person | 0.853 | source=0; burst=1; 2026-09-24T10:58:21 |
| live_003 | Fusion | 2.000 | 1.949 | 0.051 | person | 0.873 | source=0; burst=1; 2026-09-24T10:58:42 |
| live_004 | Fusion | 2.300 | 2.143 | 0.157 | person | 0.870 | source=0; burst=1; 2026-09-24T10:59:04 |
| live_005 | Fusion | 2.500 | 2.286 | 0.214 | person | 0.800 | source=0; burst=1; 2026-09-24T10:59:28 |
| live_006 | Fusion | 2.300 | 2.135 | 0.165 | person | 0.857 | source=0; burst=1; 2026-09-24T10:59:48 |
| live_007 | Fusion | 2.100 | 1.989 | 0.111 | person | 0.868 | source=0; burst=1; 2026-09-24T11:00:16 |
| live_008 | Fusion | 1.900 | 1.832 | 0.068 | person | 0.881 | source=0; burst=1; 2026-09-24T11:00:36 |
| live_009 | Fusion | 1.500 | 1.478 | 0.022 | person | 0.855 | source=0; burst=1; 2026-09-24T11:00:55 |
| live_010 | Fusion | 1.200 | 1.403 | 0.203 | person | 0.860 | source=0; burst=1; 2026-09-24T11:01:12 |

### Filtered fusion — flagged by: absolute_error > 1 m OR relative_error > 1%

| Sample | Method | GT (m) | Pred (m) | Abs err (m) | Class | Confidence | Notes |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| live_001 | Filtered fusion | 1.600 | 1.683 | 0.083 | person | 0.778 | source=0; burst=1; 2026-09-24T10:57:48 |
| live_002 | Filtered fusion | 1.150 | 1.409 | 0.259 | person | 0.853 | source=0; burst=1; 2026-09-24T10:58:21 |
| live_003 | Filtered fusion | 2.000 | 1.949 | 0.051 | person | 0.873 | source=0; burst=1; 2026-09-24T10:58:42 |
| live_004 | Filtered fusion | 2.300 | 2.143 | 0.157 | person | 0.870 | source=0; burst=1; 2026-09-24T10:59:04 |
| live_005 | Filtered fusion | 2.500 | 2.286 | 0.214 | person | 0.800 | source=0; burst=1; 2026-09-24T10:59:28 |
| live_006 | Filtered fusion | 2.300 | 2.135 | 0.165 | person | 0.857 | source=0; burst=1; 2026-09-24T10:59:48 |
| live_007 | Filtered fusion | 2.100 | 1.989 | 0.111 | person | 0.868 | source=0; burst=1; 2026-09-24T11:00:16 |
| live_008 | Filtered fusion | 1.900 | 1.832 | 0.068 | person | 0.881 | source=0; burst=1; 2026-09-24T11:00:36 |
| live_009 | Filtered fusion | 1.500 | 1.478 | 0.022 | person | 0.855 | source=0; burst=1; 2026-09-24T11:00:55 |
| live_010 | Filtered fusion | 1.200 | 1.403 | 0.203 | person | 0.860 | source=0; burst=1; 2026-09-24T11:01:12 |

Outliers are reported, not deleted. Possible causes listed in the report are hypotheses only.

## Limitations

- Dataset size: only 10 total / 10 usable samples. Error statistics are not statistically robust at this size.
- Ground-truth uncertainty: each sample's distance was measured with manual_measurement; the measurement itself carries its own error not captured here.
- Camera calibration: intrinsics are present (source=configured_default). No default focal length was silently assumed.
- Target classes: person. Geometric vision relies on per-class object heights from config, which are approximations.
- Lighting/scene/pose coverage is limited by whatever samples exist; see condition analysis.
- Depth method: NOT_AVAILABLE (depth backend disabled (pass --enable-depth; requires torch + transformers))
- LiDAR validation data: **unavailable** — no physical LiDAR returns exist in this dataset; no fake LiDAR measurements were generated.
- Filtered fusion here is a single-frame pass through the Kalman filter (samples are independent static frames); temporal smoothing/lag behavior is separately validated by the Task 2 temporal benchmark.

## Plots

- `real_world_plots.png`

## Flight integration

```text
Connected to autonomous flight: NO
Existing flight behavior changed: NO
```

This task only created/ran offline dataset and benchmark tooling. No MAVLink command, flight mode,
velocity command or controller behavior was modified.
