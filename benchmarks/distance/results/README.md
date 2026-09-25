# Distance benchmark results

Generated output directory — never edit these files by hand.

`tools/run_distance_benchmark.py` writes:

- `REAL_WORLD_DISTANCE_REPORT.md` human-readable report
- `results.json` full machine-readable payload
- `results.csv` per-sample long-format table
- `real_world_plots.png` plots (real data + matplotlib)

## Current state: initial 10-sample baseline benchmark (EVALUATED)

The current run evaluates the **existing 10 real-world person samples** in
`benchmarks/distance/images/` (1.15–2.5 m, 640×480) using the intrinsics exactly
as configured and reported in `manifest.json`:

- `fx = 446.7`, `fy = 446.7`, `cx = 320.0`, `cy = 240.0`
- `intrinsics_source = configured_default`
- provenance: `fy ≈ 446.7 px` was **previously fitted from these same 10
  samples** (pinhole `d = 1.3 m * fy / bbox_height`, person height 1.3 m), not
  from a chessboard calibration.

Because `fy` was derived from this same 10-sample set, these results are the
**initial benchmark / baseline** — they establish a reproducible starting point
but are **not fully independent validation**.

### Independent validation

`Not yet available — additional samples collected without parameter fitting are required.`

A second, disjoint set of real images with independently measured distances,
never used to derive the focal length or any estimator parameter, is needed
before the estimator can be called independently validated.

### Adding more samples

Additional samples can be collected with
`tools/create_distance_dataset.py` and added to `manifest.json` **without
changing the estimator configuration or re-fitting `fy`**, then re-run with:

```bash
python tools/run_distance_benchmark.py
```

samples without a bbox become independent validation once they are disjoint
from the points used for the `fy` fit.

## Generated output

These files only exist once a benchmark run has produced them. This directory
currently ships with the real baseline run (10 samples).