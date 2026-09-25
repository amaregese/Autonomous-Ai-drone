"""
Human-readable and machine-readable results for a real-world evaluation.

``write_all_results`` turns the payload produced by
:mod:`.evaluation.RealImageEvaluator` into:

    results/REAL_WORLD_DISTANCE_REPORT.md   human-readable report
    results/results.json                     full machine-readable payload
    results/results.csv                      long-format per-sample table

All output is derived only from actual evaluated data. Methods with no data or
unavailable backends are reported as such; nothing is fabricated.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

from .analysis import (
    write_results_csv,
    write_results_json,
)
from .evaluation import METHODS, make_plots


def _fmt(value, digits: int = 3, na: str = "N/A") -> str:
    if value is None:
        return na
    try:
        number = float(value)
    except (TypeError, ValueError):
        return na
    if not math.isfinite(number):
        return na
    return f"{number:.{digits}f}"


def _metrics_table_row(summary: dict, label: str) -> str:
    n = summary.get("n", 0)
    return (
        f"| {label} | {n} | {_fmt(summary.get('mae_m'))} | {_fmt(summary.get('rmse_m'))} | "
        f"{_fmt(summary.get('bias_m'))} | {_fmt(summary.get('median_ae_m'))} | "
        f"{_fmt(summary.get('max_ae_m'))} | {_fmt(summary.get('mre_pct'), 1)}% |"
    )


def _method_status(method_key: str, methods: Dict) -> str:
    info = methods[method_key]
    if not info.get("available"):
        reason = info.get("reason") or "unavailable"
        return f"**{METHODS[method_key]}** — *NOT_AVAILABLE* ({reason})"
    n = info.get("metrics", {}).get("n", 0)
    if n == 0:
        return f"**{METHODS[method_key]}** — no usable predictions (0 samples)"
    return f"**{METHODS[method_key]}** — {n} samples evaluated"


def _build_dataset_section(payload: Dict) -> str:
    ds = payload["dataset"]
    cam = payload["camera"]
    intrinsics = cam.get("intrinsics") or {}
    intrinsics_line = "absent (no focal length assumed)"
    if cam["intrinsics_present"] and intrinsics:
        values = f"fx={_fmt(intrinsics.get('fx'))}, fy={_fmt(intrinsics.get('fy'))}, cx={_fmt(intrinsics.get('cx'))}, cy={_fmt(intrinsics.get('cy'))}"
        intrinsics_line = f"present ({values}); source: {cam['intrinsics_source']}"
    lines = [
        "## Dataset",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Total samples | {ds['total_samples']} |",
        f"| Usable samples | {ds['usable_samples']} |",
        f"| Rejected samples | {ds['rejected_samples']} |",
        f"| Target classes | {', '.join(ds['classes']) if ds['classes'] else 'none'} |",
        f"| Distance range | {_fmt(ds['distance_min_m'])} - {_fmt(ds['distance_max_m'])} m |",
        f"| Measurement methods | {', '.join(ds['measurement_methods']) if ds['measurement_methods'] else 'none'} |",
        f"| Camera | {cam['name']} |",
        f"| Image resolution | {cam['frame_w']} x {cam['frame_h']} ({cam['frame_h']} rows) |",
        f"| Camera intrinsics | {intrinsics_line} |",
        "",
    ]
    if cam.get("intrinsics_notes"):
        lines.append("")
        lines.append(f"**Intrinsics provenance:** {cam['intrinsics_notes']}")
        lines.append("")
    if ds["rejections"]:
        lines.append("Rejected samples:")
        lines.append("")
        for rejection in ds["rejections"]:
            lines.append(f"- `{rejection['id']}` — {rejection['reason']}")
        lines.append("")
    else:
        lines.append("No rejected samples.")
        lines.append("")
    return "\n".join(lines)


def _build_benchmark_section(payload: Dict) -> str:
    methods = payload["methods"]
    lines = [
        "## Benchmark",
        "",
        "| Method | Samples | MAE | RMSE | Bias | Median AE | Max AE | MRE |",
        "| ------ | ------: | --: | ---: | ---: | --------: | -----: | --: |",
    ]
    for key in METHODS:
        info = methods[key]
        if not info.get("available"):
            lines.append(
                f"| {METHODS[key]} | - | N/A | N/A | N/A | N/A | N/A | N/A "
                f"(NOT_AVAILABLE: {info.get('reason') or 'unavailable'}) |"
            )
            continue
        summary = info.get("metrics", {})
        if summary.get("n", 0) == 0:
            lines.append(f"| {METHODS[key]} | 0 | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(_metrics_table_row(summary, METHODS[key]))
    lines.append("")
    lines.append("> Metrics are absolute errors in meters; MRE is mean relative error in percent.")
    lines.append("")
    return "\n".join(lines)


def _build_bins_section(payload: Dict) -> str:
    bins = payload["distance_bins"]
    lines = ["## Distance-bin results (populated bins only)", ""]
    any_bin = False
    for key in METHODS:
        binned = bins.get(key, {})
        if not binned:
            continue
        any_bin = True
        lines.append(f"### {METHODS[key]}")
        lines.append("")
        lines.append("| Bin | Samples | MAE | RMSE | Bias | MRE |")
        lines.append("| --- | ------: | --: | ---: | ---: | --: |")
        for label in sorted(binned.keys()):
            row = binned[label]
            lines.append(
                f"| {label} | {row.get('n', 0)} | {_fmt(row.get('mae_m'))} | "
                f"{_fmt(row.get('rmse_m'))} | {_fmt(row.get('bias_m'))} | "
                f"{_fmt(row.get('mre_pct'), 1)}% |"
            )
        lines.append("")
    if not any_bin:
        lines.append("No distance bins contain evaluated samples yet.")
        lines.append("")
    return "\n".join(lines)


def _build_conditions_section(payload: Dict) -> str:
    conditions = payload.get("condition_analysis", {})
    lines = ["## Condition analysis", ""]
    any_condition = False
    for field_name in ("class_name", "scene", "lighting", "pose", "measurement_method"):
        buckets = conditions.get(field_name, {})
        if not buckets:
            continue
        any_condition = True
        lines.append(f"### By {field_name}")
        lines.append("")
        lines.append("| Value | Samples | MAE | RMSE | Bias | MRE |")
        lines.append("| --- | ------: | --: | ---: | ---: | --: |")
        for value in sorted(buckets.keys()):
            row = buckets[value]
            n = row.get("n", 0)
            small = " ⚠ small sample" if n < 5 else ""
            lines.append(
                f"| {value} | {n} | {_fmt(row.get('mae_m'))} | {_fmt(row.get('rmse_m'))} | "
                f"{_fmt(row.get('bias_m'))} | {_fmt(row.get('mre_pct'), 1)}% |"
            )
        lines.append("")
    lines.append("Small sample sizes are noted; no broad conclusion should be drawn from 1-2 images.")
    lines.append("")
    if not any_condition:
        lines = ["## Condition analysis", "", "No condition categories contain evaluated samples yet.", ""]
    return "\n".join(lines)


def _build_confidence_section(payload: Dict) -> str:
    conf = payload.get("confidence_analysis", {})
    lines = ["## Confidence vs error (empirical)", ""]
    result = conf.get("result") or {}
    analyzed = conf.get("analyzed_method")
    if not analyzed:
        lines.append("No predictions available for a confidence analysis.")
        lines.append("")
        return "\n".join(lines)
    n = result.get("n", 0)
    r = result.get("pearson_r")
    lines.append(f"Analyzed method: **{METHODS.get(analyzed, analyzed)}** ({n} samples).")
    lines.append("")
    if r is None:
        lines.append("Pearson r: not computable with this data.")
    else:
        lines.append(f"Pearson r (confidence vs absolute error): **{r:.3f}**")
    lines.append("")
    bands = result.get("bands") or {}
    if bands:
        lines.append("| Confidence band | Samples | Mean abs error (m) |")
        lines.append("| --- | ------: | --: |")
        for label, band in bands.items():
            lines.append(
                f"| {label} | {band.get('n', 0)} | {_fmt(band.get('mean_abs_error'))} |"
            )
        lines.append("")
    lines.append("This is an empirical observation, NOT a calibrated probability that confidence equals accuracy.")
    lines.append("")
    return "\n".join(lines)


def _build_outliers_section(payload: Dict) -> str:
    methods = payload["methods"]
    lines = ["## Outliers", ""]
    header = "| Sample | Method | GT (m) | Pred (m) | Abs err (m) | Class | Confidence | Notes |"
    sep = "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |"
    any_outlier = False
    for key in METHODS:
        flagged = methods[key].get("outliers") or []
        if not flagged:
            continue
        rule = methods[key].get("outlier_rule") or "absolute_error > threshold"
        any_outlier = True
        lines.append(f"### {METHODS[key]} — flagged by: {rule}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for record in flagged:
            notes = (record.get("notes") or "").strip()
            lines.append(
                f"| {record.get('sample_id')} | {METHODS[key]} | {_fmt(record.get('ground_truth_m'))} | "
                f"{_fmt(record.get('prediction_m'))} | {_fmt(record.get('absolute_error_m'))} | "
                f"{record.get('class_name')} | {_fmt(record.get('confidence'))} | {notes} |"
            )
        lines.append("")
    if not any_outlier:
        lines.append("No samples exceeded the outlier flag threshold.")
        lines.append("")
    lines.append("Outliers are reported, not deleted. Possible causes listed in the report are hypotheses only.")
    lines.append("")
    return "\n".join(lines)


def _build_limitations_section(payload: Dict) -> str:
    ds = payload["dataset"]
    cam = payload["camera"]
    options = payload.get("options", {})
    depth = payload["methods"]["depth"]
    lines = [
        "## Limitations",
        "",
        f"- Dataset size: only {ds['total_samples']} total / {ds['usable_samples']} usable samples. "
        "Error statistics are not statistically robust at this size.",
        f"- Ground-truth uncertainty: each sample's distance was measured with "
        f"{', '.join(ds['measurement_methods']) if ds['measurement_methods'] else 'an unrecorded method'}; "
        "the measurement itself carries its own error not captured here.",
        f"- Camera calibration: intrinsics are {('present (source=%s)' % cam['intrinsics_source']) if cam['intrinsics_present'] else 'ABSENT'}. "
        "No default focal length was silently assumed.",
        "- Target classes: " + (", ".join(ds["classes"]) if ds["classes"] else "none") + ". "
        "Geometric vision relies on per-class object heights from config, which are approximations.",
        "- Lighting/scene/pose coverage is limited by whatever samples exist; see condition analysis.",
        "- Depth method: " + (
            f"available (enabled {'with backend' if options.get('enable_depth') else ''})"
            if depth.get("available")
            else f"NOT_AVAILABLE ({depth.get('reason') or 'disabled'})"
        ),
        "- LiDAR validation data: **unavailable** — no physical LiDAR returns exist in this dataset; "
        "no fake LiDAR measurements were generated.",
        "- Filtered fusion here is a single-frame pass through the Kalman filter (samples are independent "
        "static frames); temporal smoothing/lag behavior is separately validated by the Task 2 temporal benchmark.",
        "",
    ]
    return "\n".join(lines)


def _build_flight_section() -> str:
    return "\n".join(
        [
            "## Flight integration",
            "",
            "```text",
            "Connected to autonomous flight: NO",
            "Existing flight behavior changed: NO",
            "```",
            "",
            "This task only created/ran offline dataset and benchmark tooling. No MAVLink command, flight mode,",
            "velocity command or controller behavior was modified.",
            "",
        ]
    )


def _build_validation_status_section(payload: Dict) -> str:
    cam = payload["camera"]
    notes = (cam.get("intrinsics_notes") or "").lower()
    fitted_on_these = cam["intrinsics_present"] and (
        cam["intrinsics_source"] == "configured_default" or "fit" in notes
    )
    lines = [
        "## Baseline benchmark vs independent validation",
        "",
        "### Baseline benchmark",
        "",
        "The current 10 real-world samples were evaluated with the configured intrinsics exactly as shipped.",
    ]
    if fitted_on_these:
        lines += [
            "The intrinsics used include a focal length that was **previously fitted from these same 10 samples**",
            "(`fy ≈ 446.7 px`, pinhole model `d = real_height * fy / bbox_height`, fitted mean across the 1.15–2.5 m",
            "person-height dataset). Regenerating the results with the same fitted intrinsics is expected to reproduce",
            "this baseline, but the effect of the fit on the metrics is not independent.",
            "These 10 samples therefore count as an **initial benchmark / baseline**, not as independent validation.",
        ]
    else:
        lines.append(
            "Intrinsics provenance `intrinsics_notes` is not present; treat these samples "
            "as the initial baseline until independent validation exists."
        )
    lines += [
        "",
        "### Independent validation",
        "",
        "**Not yet available** — additional samples collected without parameter fitting are required.",
        "(i.e. a second set of real images whose distances were measured independently and that were",
        "never used to derive the focal length or any estimator parameter.)",
        "",
    ]
    return "\n".join(lines)


def build_report_markdown(payload: Dict, results_dir: str, plots: Optional[List[str]] = None) -> str:
    lines = [
        "# Real-World Distance Accuracy Report",
        "",
        "Real-image evaluation of the isolated distance estimator "
        "(`modules/distance_estimator/`). Synthetic Task 2 results are intentionally absent here.",
        "",
        "## Real-world benchmark status",
        "",
        (
            "```text\n"
            "Real-world benchmark status: READY FOR DATA COLLECTION\n"
            "Real-world accuracy results: NOT AVAILABLE (no evaluated samples yet)\n"
            "```"
            if payload["dataset"]["usable_samples"] == 0
            else "```text\nReal-world benchmark status: EVALUATED\nReal-world accuracy results: see table below\n```"
        ),
        "",
        _build_dataset_section(payload),
        _build_validation_status_section(payload),
        _build_benchmark_section(payload),
        _build_bins_section(payload),
        _build_conditions_section(payload),
        _build_confidence_section(payload),
        _build_outliers_section(payload),
        _build_limitations_section(payload),
    ]
    if plots:
        lines.append("## Plots")
        lines.append("")
        for plot in plots:
            lines.append(f"- `{os.path.basename(plot)}`")
        lines.append("")
    lines.append(_build_flight_section())
    return "\n".join(lines)


def write_all_results(payload: Dict, results_dir: str) -> Dict[str, str]:
    """Write results.json, results.csv and REAL_WORLD_DISTANCE_REPORT.md."""
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "results.json")
    write_results_json(json_path, payload)

    rows: List[dict] = []
    for key in METHODS:
        for record in payload["methods"][key].get("records", []):
            rows.append(record)

    csv_path = os.path.join(results_dir, "results.csv")
    fieldnames = [
        "sample_id",
        "method",
        "ground_truth_m",
        "prediction_m",
        "absolute_error_m",
        "signed_error_m",
        "relative_error",
        "confidence",
        "class_name",
        "scene",
        "lighting",
        "pose",
        "measurement_method",
        "notes",
    ]
    write_results_csv(csv_path, rows, fieldnames)

    from .evaluation import make_plots  # noqa: PLC0415

    plots = make_plots(
        {key: payload["methods"][key].get("records", []) for key in METHODS},
        results_dir,
    )
    if not plots:
        plots = []

    report_path = os.path.join(results_dir, "REAL_WORLD_DISTANCE_REPORT.md")
    text = build_report_markdown(payload, results_dir, plots)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)

    return {"json": json_path, "csv": csv_path, "report": report_path, "plots": plots}