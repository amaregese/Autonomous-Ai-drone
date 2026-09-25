"""
Real-world distance accuracy benchmark for the isolated estimator.

Processes a validated dataset manifest (real images + independently measured
ground-truth distances) and evaluates every available estimator method:

    A. Legacy bbox-height, B. Geometric vision, C. Metric monocular depth,
    D. Fusion, E. Filtered fusion.

Run:

    python tools/run_distance_benchmark.py
    python tools/run_distance_benchmark.py --manifest benchmarks/distance/manifest.json
    python tools/run_distance_benchmark.py --enable-depth     # runs ZoeDepth (downloads model)
    python tools/run_distance_benchmark.py --no-plots         # skip matplotlib plots

Outputs are written to ``benchmarks/distance/results/``:

    REAL_WORLD_DISTANCE_REPORT.md   human-readable report
    results.json                    full machine-readable payload
    results.csv                     long-format per-sample table
    real_world_plots.png            plots (when matplotlib is available)

Rules:

  * Synthetic data from the Task 2 benchmark is never mixed in.
  * A method with no usable data reports NOT_AVAILABLE + reason.
  * Without a real camera dataset the tool reports
    "Real-world results: NOT AVAILABLE" instead of fabricating numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.distance_estimator.analysis import compute_metrics  # noqa: E402
from modules.distance_estimator.dataset import DatasetError, load_manifest  # noqa: E402
from modules.distance_estimator.evaluation import EvaluationOptions, RealImageEvaluator  # noqa: E402
from modules.distance_estimator.report import METHODS, write_all_results  # noqa: E402

DEFAULT_MANIFEST = os.path.join("benchmarks", "distance", "manifest.json")
DEFAULT_RESULTS_DIR = os.path.join("benchmarks", "distance", "results")


def _console_table(payload: Dict) -> str:
    methods = payload["methods"]
    width = 22
    header = f"{'Method':<{width}} {'Avail':<7} {'n':>4} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'MRE%':>7}"
    lines = [header, "-" * len(header)]
    for key in METHODS:
        info = methods[key]
        if not info.get("available"):
            lines.append(f"{METHODS[key]:<{width}} {'no':<7} {'-':>4}")
            continue
        summary = info.get("metrics", {})
        n = summary.get("n", 0)
        if n == 0:
            lines.append(f"{METHODS[key]:<{width}} {'yes':<7} {0:>4}")
            continue
        lines.append(
            f"{METHODS[key]:<{width}} {'yes':<7} {n:>4} "
            f"{summary['mae_m'] if summary['mae_m'] is not None else float('nan'):>8.3f} "
            f"{summary['rmse_m'] if summary['rmse_m'] is not None else float('nan'):>8.3f} "
            f"{summary['bias_m'] if summary['bias_m'] is not None else float('nan'):>8.3f} "
            f"{summary['mre_pct'] if summary['mre_pct'] is not None else float('nan'):>6.1f}%"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Real-world distance accuracy benchmark")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--enable-depth", action="store_true",
                        help="run the metric monocular-depth backend (ZoeDepth; may download weights)")
    parser.add_argument("--no-legacy", action="store_true", help="skip the legacy bbox-height method")
    parser.add_argument("--no-plots", action="store_true", help="skip matplotlib plots")
    parser.add_argument("--outlier-threshold", type=float, default=1.0,
                        help="absolute-error outlier flag threshold in meters (default 1.0)")
    parser.add_argument("--no-relative-outliers", action="store_true",
                        help="do not apply the relative-error outlier rule")
    parser.add_argument("--outlier-relative-threshold", type=float, default=1.0,
                        help="relative-error outlier flag threshold in percent (default 1.0)")
    args = parser.parse_args(argv)

    manifest_path = args.manifest if os.path.isabs(args.manifest) else os.path.join(PROJECT_ROOT, args.manifest)
    results_dir = os.path.join(PROJECT_ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir

    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        print("Use tools/create_distance_dataset.py to build a dataset, or leave the empty manifest in place.", file=sys.stderr)
        return 1

    try:
        dataset = load_manifest(manifest_path)
    except DatasetError as exc:
        print(f"Manifest invalid: {exc}", file=sys.stderr)
        return 1

    options = EvaluationOptions(
        enable_legacy=not args.no_legacy,
        enable_depth=args.enable_depth,
        make_plots=not args.no_plots,
        outlier_threshold_m=args.outlier_threshold,
        outlier_relative_threshold_pct=None if args.no_relative_outliers else args.outlier_relative_threshold,
    )

    evaluator = RealImageEvaluator(dataset, options)
    payload = evaluator.run()

    paths = write_all_results(payload, results_dir)

    ds = payload["dataset"]
    print("=" * 72)
    print("REAL-WORLD DISTANCE BENCHMARK")
    print("=" * 72)
    print(f"Manifest:          {dataset.manifest_path}")
    print(f"Total samples:     {ds['total_samples']}")
    print(f"Usable samples:    {ds['usable_samples']}")
    print(f"Rejected samples:  {ds['rejected_samples']} {ds['rejections']}")
    print(f"Classes:           {', '.join(ds['classes']) if ds['classes'] else 'none'}")
    print(f"Distance range:    {ds['distance_min_m']} - {ds['distance_max_m']} m")
    print(f"Camera intrinsics: {'present (source=%s)' % payload['camera']['intrinsics_source'] if payload['camera']['intrinsics_present'] else 'ABSENT (no default assumed)'}")
    print(f"Depth method:      {('available' if payload['methods']['depth']['available'] else 'NOT_AVAILABLE: ' + (payload['methods']['depth']['reason'] or ''))}")
    print()
    print(_console_table(payload))
    print()

    if ds["usable_samples"] == 0:
        print("Real-world benchmark status: READY FOR DATA COLLECTION")
        print("Real-world accuracy results: NOT AVAILABLE (no evaluated samples yet)")
    else:
        print("Real-world benchmark status: EVALUATED")
    print()
    print(f"Report:    {paths['report']}")
    print(f"Results:   {paths['json']}")
    print(f"CSV:       {paths['csv']}")
    if paths.get("plots"):
        print(f"Plots:     {paths['plots']}")
    else:
        print("Plots:     not generated (no real data or matplotlib unavailable)")
    print()
    print("Connected to autonomous flight: NO")
    print("Existing flight behavior changed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())