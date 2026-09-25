"""
Real-world distance dataset: manifest schema, validation and serialization.

Ground truth MUST be measured independently (tape measure, laser rangefinder,
measured rig position, ...). It is **never** computed from the image itself
(no bbox-height, YOLO, ZoeDepth, monocular geometry or estimator output).

A manifest is a JSON file of the form::

    {
      "version": 1,
      "camera": {
        "name": "webcam-front",
        "frame_w": 640,
        "frame_h": 480,
        "intrinsics_source": "unknown",   // measured | calibrated |
                                          // configured_default | unknown
        "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...} | null
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

Structural problems raise :class:`DatasetError`; the manifest never silently
accepts a missing/zero/negative/non-finite ground truth.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_NUMERIC = (int, float)

# Smallest sensible independently measured target distance; matches the
# estimator's minimum supported vision/depth range (0.25 m). Effectively-zero
# values are treated as invalid ground truth.
MIN_GT_DISTANCE_M = 0.25

INTRINSICS_SOURCE_OPTIONS = ("measured", "calibrated", "configured_default", "unknown")


class DatasetError(Exception):
    """Raised when a dataset manifest is missing, malformed or invalid."""


def _is_number(value: object) -> bool:
    return isinstance(value, _NUMERIC) and not isinstance(value, bool)


def _is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def validate_ground_truth(value: object) -> List[str]:
    """Validate an independently measured ground-truth distance."""
    if not _is_number(value):
        return ["ground_truth_distance_m must be a number"]
    if not math.isfinite(float(value)):
        return ["ground_truth_distance_m must be finite"]
    if float(value) < MIN_GT_DISTANCE_M:
        return [f"ground_truth_distance_m must be >= {MIN_GT_DISTANCE_M:g} m (never 0/negative/effectively-zero)"]
    return []


@dataclass(slots=True)
class BoundingBox:
    """Target bounding box in pixel coordinates (top-left + size)."""

    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_dict(cls, data: object) -> "BoundingBox":
        if not isinstance(data, dict):
            raise DatasetError("bbox must be a JSON object")
        missing = [k for k in ("x", "y", "width", "height") if k not in data]
        if missing:
            raise DatasetError(f"bbox missing required key(s): {', '.join(missing)}")
        x, y, w, h = data["x"], data["y"], data["width"], data["height"]
        if not all(_is_number(v) for v in (x, y, w, h)):
            raise DatasetError("bbox x/y/width/height must be numbers")
        problems = []
        if float(x) < 0.0 or float(y) < 0.0:
            problems.append("bbox x/y must be non-negative")
        if float(w) <= 0.0 or float(h) <= 0.0:
            problems.append("bbox width/height must be positive")
        if problems:
            raise DatasetError("invalid bbox: " + "; ".join(problems))
        return cls(x=float(x), y=float(y), width=float(w), height=float(h))

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @property
    def ltrb(self) -> tuple:
        """(left, top, right, bottom) pixel coordinates for the estimator API."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(slots=True)
class GroundTruthSample:
    """One independently measured real-world ground-truth sample."""

    id: str
    image: str
    ground_truth_distance_m: float
    class_name: str
    measurement_method: str
    bbox: Optional[BoundingBox] = None
    detection_confidence: Optional[float] = None
    scene: Optional[str] = None
    lighting: Optional[str] = None
    pose: Optional[str] = None
    notes: str = ""
    # internal (set after load); not part of the serialized manifest
    _manifest_dir: str = field(default="", init=False, repr=False, compare=False)

    @property
    def image_path(self) -> str:
        """Absolute path resolved against the manifest directory."""
        return os.path.normpath(os.path.join(self._manifest_dir, self.image)) if self._manifest_dir else self.image

    def _set_manifest_dir(self, directory: str) -> None:
        self._manifest_dir = os.path.dirname(directory) if os.path.isfile(directory) else directory

    @classmethod
    def from_dict(
        cls,
        data: object,
        index: Optional[int] = None,
    ) -> "GroundTruthSample":
        if not isinstance(data, dict):
            raise DatasetError(f"sample[{index}] must be a JSON object")
        where = f"sample[{index}]" if index is not None else "sample"

        errors: List[str] = []
        required = ("id", "image", "ground_truth_distance_m", "class_name", "measurement_method")
        for key in required:
            if key not in data:
                errors.append(f"{where} missing required field {key!r}")

        if errors:
            raise DatasetError("; ".join(errors))

        sample_id = data["id"]
        if not _is_non_empty_str(sample_id):
            raise DatasetError(f"{where} id must be a non-empty string")
        if not _is_non_empty_str(data["image"]):
            raise DatasetError(f"{where} image must be a non-empty string")
        if not _is_non_empty_str(data["class_name"]):
            raise DatasetError(f"{where} class_name must be a non-empty string")
        if not _is_non_empty_str(data["measurement_method"]):
            raise DatasetError(f"{where} measurement_method must be a non-empty string")

        gt_errors = validate_ground_truth(data["ground_truth_distance_m"])
        if gt_errors:
            raise DatasetError(f"{where} {gt_errors[0]}")

        bbox = None
        if "bbox" in data and data["bbox"] is not None:
            bbox = BoundingBox.from_dict(data["bbox"])

        detection_confidence = None
        if "detection_confidence" in data and data["detection_confidence"] is not None:
            value = data["detection_confidence"]
            if not _is_number(value) or not math.isfinite(float(value)):
                raise DatasetError(f"{where} detection_confidence must be a finite number")
            if not (0.0 < float(value) <= 1.0):
                raise DatasetError(f"{where} detection_confidence must be in (0, 1]")
            detection_confidence = float(value)

        optional_strings = ("scene", "lighting", "pose")
        normalized: Dict[str, str] = {}
        for key in optional_strings:
            raw = data.get(key)
            normalized[key] = raw if _is_non_empty_str(raw) else ("" if raw is None else str(raw))

        notes = data.get("notes") or ""
        notes = str(notes)

        return cls(
            id=sample_id,
            image=data["image"],
            ground_truth_distance_m=float(data["ground_truth_distance_m"]),
            class_name=data["class_name"],
            measurement_method=data["measurement_method"],
            bbox=bbox,
            detection_confidence=detection_confidence,
            scene=normalized.get("scene") or None,
            lighting=normalized.get("lighting") or None,
            pose=normalized.get("pose") or None,
            notes=notes,
        )

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "id": self.id,
            "image": self.image,
            "ground_truth_distance_m": self.ground_truth_distance_m,
            "class_name": self.class_name,
            "measurement_method": self.measurement_method,
            "notes": self.notes,
        }
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        if self.detection_confidence is not None:
            data["detection_confidence"] = self.detection_confidence
        for key in ("scene", "lighting", "pose"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass(slots=True)
class CameraRecord:
    """Camera metadata for a dataset (intrinsics source is explicit)."""

    name: str = "unknown"
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None
    intrinsics_source: str = "unknown"
    intrinsics: Optional[Dict[str, float]] = None
    intrinsics_notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: object) -> "CameraRecord":
        if not isinstance(data, dict):
            raise DatasetError("camera must be a JSON object")
        name = data.get("name") or "unknown"
        if not _is_non_empty_str(name):
            raise DatasetError("camera.name must be a non-empty string")

        frame_w = data.get("frame_w")
        frame_h = data.get("frame_h")
        if frame_w is not None:
            if not isinstance(frame_w, int) or frame_w <= 0:
                raise DatasetError("camera.frame_w must be a positive integer")
        if frame_h is not None:
            if not isinstance(frame_h, int) or frame_h <= 0:
                raise DatasetError("camera.frame_h must be a positive integer")

        source = data.get("intrinsics_source") or "unknown"
        if not _is_non_empty_str(source):
            raise DatasetError("camera.intrinsics_source must be a non-empty string")

        intrinsics = None
        raw = data.get("intrinsics")
        if raw is not None:
            if not isinstance(raw, dict):
                raise DatasetError("camera.intrinsics must be a JSON object or null")
            missing = [k for k in ("fx", "fy", "cx", "cy") if k not in raw]
            if missing:
                raise DatasetError(f"camera.intrinsics missing key(s): {', '.join(missing)}")
            values = {k: raw[k] for k in ("fx", "fy", "cx", "cy")}
            if not all(_is_number(v) for v in values.values()):
                raise DatasetError("camera.intrinsics fx/fy/cx/cy must be numbers")
            if float(values["fx"]) <= 0.0 or float(values["fy"]) <= 0.0:
                raise DatasetError("camera.intrinsics fx/fy must be positive")
            if float(values["cx"]) < 0.0 or float(values["cy"]) < 0.0:
                raise DatasetError("camera.intrinsics cx/cy must be non-negative")
            intrinsics = {k: float(v) for k, v in values.items()}

        intrinsics_notes = data.get("intrinsics_notes")
        intrinsics_notes = str(intrinsics_notes).strip() if intrinsics_notes else None

        return cls(
            name=name,
            frame_w=frame_w,
            frame_h=frame_h,
            intrinsics_source=source,
            intrinsics=intrinsics,
            intrinsics_notes=intrinsics_notes,
        )

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "name": self.name,
            "intrinsics_source": self.intrinsics_source,
        }
        if self.frame_w is not None:
            data["frame_w"] = self.frame_w
        if self.frame_h is not None:
            data["frame_h"] = self.frame_h
        data["intrinsics"] = self.intrinsics
        if self.intrinsics_notes:
            data["intrinsics_notes"] = self.intrinsics_notes
        return data


@dataclass(slots=True)
class DistanceDataset:
    """A validated real-world distance dataset loaded from a manifest."""

    version: int
    camera: CameraRecord
    samples: List[GroundTruthSample]
    manifest_path: str
    camera_available: bool = False

    @property
    def manifest_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.manifest_path))

    @property
    def sample_ids(self) -> List[str]:
        return [s.id for s in self.samples]

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "camera": self.camera.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
        }

    def save_manifest(self, path: Optional[str] = None) -> None:
        target = path or self.manifest_path
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


def _attach_manifest_dirs(dataset: DistanceDataset) -> None:
    manifest_dir = dataset.manifest_dir
    for sample in dataset.samples:
        sample._set_manifest_dir(manifest_dir)


def load_manifest(path: str) -> DistanceDataset:
    """Load and validate a dataset manifest. Raises :class:`DatasetError`."""
    if not os.path.exists(path):
        raise DatasetError(f"manifest not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"failed to read manifest {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise DatasetError("manifest must be a JSON object")

    version = raw.get("version")
    if not isinstance(version, int) or version <= 0:
        raise DatasetError("manifest version must be a positive integer")

    raw_samples = raw.get("samples")
    if raw_samples is None:
        raise DatasetError("manifest missing required field 'samples'")
    if not isinstance(raw_samples, list):
        raise DatasetError("manifest 'samples' must be a list")

    samples: List[GroundTruthSample] = []
    seen_ids = set()
    problems: List[str] = []
    for index, item in enumerate(raw_samples):
        try:
            sample = GroundTruthSample.from_dict(item, index=index)
        except DatasetError as exc:
            problems.append(str(exc))
            continue
        if sample.id in seen_ids:
            problems.append(f"duplicate sample id {sample.id!r}")
        seen_ids.add(sample.id)
        samples.append(sample)

    if problems:
        raise DatasetError("manifest validation failed: " + "; ".join(problems))

    camera_raw = raw.get("camera")
    if camera_raw is None:
        camera = CameraRecord()
    else:
        try:
            camera = CameraRecord.from_dict(camera_raw)
        except DatasetError as exc:
            raise DatasetError(f"camera metadata: {exc}") from exc

    dataset = DistanceDataset(
        version=version,
        camera=camera,
        samples=samples,
        manifest_path=os.path.abspath(path),
        camera_available=camera.intrinsics is not None,
    )
    _attach_manifest_dirs(dataset)
    return dataset


def create_empty_dataset(
    path: str,
    camera_name: str = "unknown",
    intrinsics_source: str = "unknown",
    intrinsics: Optional[Dict[str, float]] = None,
    version: int = 1,
) -> DistanceDataset:
    """Create (and write) a new empty dataset manifest."""
    camera = CameraRecord(
        name=camera_name,
        intrinsics_source=intrinsics_source,
        intrinsics=intrinsics,
    )
    dataset = DistanceDataset(
        version=version,
        camera=camera,
        samples=[],
        manifest_path=os.path.abspath(path),
        camera_available=intrinsics is not None,
    )
    dataset.save_manifest()
    return dataset