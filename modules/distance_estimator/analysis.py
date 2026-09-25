"""
Real-world accuracy analysis: metrics, distance bins, condition analysis,
confidence-vs-error, outlier flags, and result serialization (JSON / CSV / MD).

Every function here is deterministic and pure; no estimator or neural network
is required. Metrics are computed only from ``(ground_truth_m, prediction_m)``
pairs and are never fabricated — a method with no usable predictions simply
reports ``n == 0`` / ``N/A``.

Ground truth is required to be a positive finite distance (validated upstream by
:mod:`.dataset`); ``compute_metrics`` defends against zero/None again so the
math can never divide by zero.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from statistics import mean as _mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class AnalysisError(Exception):
    """Raised when metric/report calculations receive malformed input."""


@dataclass
class MetricSummary:
    """Aggregate error metrics for a set of (ground truth, prediction) pairs."""

    n: int = 0
    mae: float = float("nan")
    rmse: float = float("nan")
    bias: float = float("nan")  # mean signed error (pred - gt)
    median_ae: float = float("nan")
    max_ae: float = float("nan")
    mre_pct: float = float("nan")  # mean relative error (%)

    def available(self) -> bool:
        return self.n > 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "mae_m": _fmt(self.mae),
            "rmse_m": _fmt(self.rmse),
            "bias_m": _fmt(self.bias),
            "median_ae_m": _fmt(self.median_ae),
            "max_ae_m": _fmt(self.max_ae),
            "mre_pct": _fmt(self.mre_pct),
        }


def _fmt(value: float) -> object:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 4)


def _pairs(records_or_pairs) -> List[Tuple[float, float]]:
    """Accept either (gt, pred) pairs or sample records with gt/pred keys."""
    pairs: List[Tuple[float, float]] = []
    for item in records_or_pairs:
        if isinstance(item, (tuple, list)):
            gt, pred = item
        else:
            gt = item.get("ground_truth_m")
            pred = item.get("prediction_m")
        if gt is None or pred is None:
            continue
        gt = float(gt)
        pred = float(pred)
        if not math.isfinite(gt) or gt <= 0.0:
            continue
        if not math.isfinite(pred):
            continue
        pairs.append((gt, pred))
    return pairs


def compute_metrics(records_or_pairs) -> MetricSummary:
    """Compute MAE / RMSE / bias / median AE / max AE / MRE from samples."""
    pairs = _pairs(records_or_pairs)
    if not pairs:
        return MetricSummary(n=0)

    errors = [abs(gt - pred) for gt, pred in pairs]
    signed = [pred - gt for gt, pred in pairs]
    relative = [abs(gt - pred) / gt * 100.0 for gt, pred in pairs]
    sorted_errors = sorted(errors)
    mid = len(sorted_errors) // 2
    median_ae = (
        sorted_errors[mid]
        if len(sorted_errors) % 2
        else (sorted_errors[mid - 1] + sorted_errors[mid]) / 2.0
    )

    return MetricSummary(
        n=len(pairs),
        mae=_mean(errors),
        rmse=math.sqrt(_mean(e * e for e in errors)),
        bias=_mean(signed),
        median_ae=median_ae,
        max_ae=max(errors),
        mre_pct=_mean(relative),
    )


DISTANCE_BINS: Tuple[Tuple[str, Optional[float], Optional[float]], ...] = (
    ("0-2 m", 0.0, 2.0),
    ("2-4 m", 2.0, 4.0),
    ("4-6 m", 4.0, 6.0),
    ("6-8 m", 6.0, 8.0),
    ("8-10 m", 8.0, 10.0),
    ("10+ m", 10.0, None),
)


def distance_bin_membership(distance_m: float) -> Optional[str]:
    """Return the label of the bin containing ``distance_m``."""
    for label, low, high in DISTANCE_BINS:
        if high is None:
            if distance_m >= low:
                return label
        elif low <= distance_m < high:
            return label
    return None


def group_by_distance_bin(records_or_pairs) -> Dict[str, List[dict]]:
    """Group predictions into the populated distance bins only."""
    grouped: Dict[str, List[dict]] = {}
    for item in records_or_pairs:
        if isinstance(item, (tuple, list)):
            gt, pred = item
            record = {"ground_truth_m": gt, "prediction_m": pred}
        else:
            record = dict(item)
            gt = record.get("ground_truth_m")
        if gt is None:
            continue
        label = distance_bin_membership(float(gt))
        if label is None:
            continue
        grouped.setdefault(label, []).append(record)
    return grouped


def group_by_condition(
    records: Sequence[dict],
    key: str,
    min_samples: int = 3,
) -> Dict[str, MetricSummary]:
    """Group predictions by a condition field; only groups with real samples.

    ``min_samples`` is the smallest group that is reported (it is still
    reported, with a small-sample warning, not silently dropped — the caller
    decides whether to include it).
    """
    buckets: Dict[str, List[dict]] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        buckets.setdefault(str(value), []).append(record)
    result: Dict[str, MetricSummary] = {}
    for value, group in sorted(buckets.items(), key=lambda kv: kv[0]):
        result[value] = compute_metrics(group)
    return result


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson r. Returns None when the data does not support a correlation."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    n = len(xs)
    mx = _mean(xs)
    my = _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sxx * syy)
    if denom <= 1e-12:
        return None
    return sxy / denom


def flag_outliers(
    records: Sequence[dict],
    threshold_m: float = 1.0,
    relative_threshold_pct: Optional[float] = None,
) -> Tuple[List[dict], str]:
    """Flag samples exceeding an absolute (and optionally relative) error.

    Default flag rule: ``absolute_error > threshold_m``. If
    ``relative_threshold_pct`` is given, a sample is also flagged when
    ``relative_error > relative_threshold_pct``.
    """
    flagged: List[dict] = []
    for record in records:
        gt = record.get("ground_truth_m")
        pred = record.get("prediction_m")
        if gt is None or pred is None:
            continue
        try:
            inverse_relative = 1.0 / float(gt)
        except ZeroDivisionError:
            continue
        abs_err = abs(float(pred) - float(gt))
        rel_err = abs_err * inverse_relative * 100.0
        over_abs = abs_err > threshold_m
        over_rel = relative_threshold_pct is not None and rel_err > relative_threshold_pct
        if over_abs or over_rel:
            flagged.append(dict(record))
    rule = (
        f"absolute_error > {threshold_m:g} m"
        + (f" OR relative_error > {relative_threshold_pct:g}%" if relative_threshold_pct is not None else "")
    )
    return flagged, rule


def confidence_vs_error(records: Sequence[dict]) -> Dict[str, object]:
    """Empirical confidence-vs-error analysis (not a calibrated probability)."""
    paired = [
        (float(r["confidence"]), abs(float(r["prediction_m"]) - float(r["ground_truth_m"])))
        for r in records
        if r.get("confidence") is not None
        and r.get("prediction_m") is not None
        and r.get("ground_truth_m") is not None
    ]
    if len(paired) < 3:
        return {"n": len(paired), "pearson_r": None, "note": "insufficient data for a confidence-vs-error analysis"}

    confidences = [c for c, _ in paired]
    errors = [e for _, e in paired]
    r = pearson_correlation(confidences, errors)

    low = [e for c, e in paired if c < 0.5]
    high = [e for c, e in paired if c >= 0.5]
    bands = {
        "confidence < 0.5": {"n": len(low), "mean_abs_error": _mean(low) if low else None},
        "confidence >= 0.5": {"n": len(high), "mean_abs_error": _mean(high) if high else None},
    }

    return {
        "n": len(paired),
        "pearson_r": r,
        "bands": bands,
        "note": (
            "Empirical only. Negative r suggests higher confidence coincides with "
            "lower error on this dataset; it is NOT a calibrated probability."
        ),
    }


# ------------------------------------------------------------------ serialization


def write_results_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_results_csv(
    path: str,
    rows: Iterable[dict],
    fieldnames: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: (row.get(name) if name in row else None) for name in fieldnames})