"""
Interactive helper to add a real sample to the distance dataset.

Adds ONE sample per invocation:

    python tools/create_distance_dataset.py --image photos/me_3m.jpg --distance 3.0

    python tools/create_distance_dataset.py
        (prompts for everything)

All arguments can be given on the command line; anything missing is prompted
for. The tool NEVER estimates or invents a ground-truth distance:

  * the distance you enter must be an independently measured value
    (tape measure / laser rangefinder / measured rig position);
  * if no valid distance is provided, the sample is NOT added.

The image is copied into ``benchmarks/distance/images/`` and the manifest is
rewritten. Requires OpenCV (or Pillow) to read image dimensions; the copy is
plain file I/O.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.distance_estimator.dataset import (  # noqa: E402
    DatasetError,
    GroundTruthSample,
    create_empty_dataset,
    load_manifest,
)

DEFAULT_MANIFEST = os.path.join("benchmarks", "distance", "manifest.json")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


def _parse_distance(text, where="distance") -> float:
    """Parse a strictly valid ground truth or raise."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be a number, got {text!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{where} must be a positive finite number, got {text!r}")
    return value


def _parse_bbox(text) -> tuple:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 4 numbers: x,y,width,height")
    values = [_parse_distance(p, "bbox component") for p in parts]
    x, y, w, h = values
    if float(w) <= 0.0 or float(h) <= 0.0:
        raise ValueError("bbox width and height must be positive")
    return (x, y, w, h)


def _next_id(existing_ids) -> str:
    index = 1
    while f"sample_{index:03d}" in existing_ids:
        index += 1
    return f"sample_{index:03d}"


def _ask(prompt, default=None, interactive=True):
    if not interactive:
        return None
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if value == "" and default is not None:
        return default
    return value or None


def _intrinsics_from_json(text):
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--intrinsics must be valid JSON: {exc}") from exc
    missing = [k for k in ("fx", "fy", "cx", "cy") if k not in data]
    if missing:
        raise ValueError(f"--intrinsics missing {', '.join(missing)}")
    return {k: float(data[k]) for k in ("fx", "fy", "cx", "cy")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Add one real sample to the distance dataset")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--image", help="path to the image to import")
    parser.add_argument("--distance", help="independently measured distance in meters (required)")
    parser.add_argument("--class", dest="class_name", help="target class (e.g. person)")
    parser.add_argument("--method", help="measurement method (e.g. tape_measure, laser)")
    parser.add_argument("--scene", help="optional scene description")
    parser.add_argument("--lighting", help="optional lighting condition")
    parser.add_argument("--pose", help="optional target pose")
    parser.add_argument("--notes", help="optional notes")
    parser.add_argument("--bbox", help="bounding box as x,y,width,height")
    parser.add_argument("--no-bbox", action="store_true", help="skip bounding box handling")
    parser.add_argument("--camera-name", default="unknown")
    parser.add_argument("--intrinsics-source", default="unknown",
                        choices=["measured", "calibrated", "configured_default", "unknown"])
    parser.add_argument("--intrinsics", help='camera intrinsics JSON e.g. \'{"fx":1112,"fy":1112,"cx":320,"cy":240}\'')
    parser.add_argument("--non-interactive", action="store_true", help="never prompt (fail when info is missing)")
    args = parser.parse_args(argv)

    interactive = not args.non_interactive

    manifest_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.manifest) if not os.path.isabs(args.manifest)
                                     else args.manifest)
    manifest_dir = os.path.dirname(manifest_path)
    images_dir = os.path.join(manifest_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    dataset = load_manifest(manifest_path) if os.path.exists(manifest_path) else create_empty_dataset(
        manifest_path,
        camera_name=args.camera_name,
        intrinsics_source=args.intrinsics_source,
        intrinsics=_intrinsics_from_json(args.intrinsics),
    )
    if args.intrinsics:
        dataset.camera.intrinsics = _intrinsics_from_json(args.intrinsics)
        dataset.camera.intrinsics_source = args.intrinsics_source
        dataset.camera_available = True

    # --- image -----------------------------------------------------------
    image_arg = args.image or _ask("Path to image", interactive=interactive)
    if not image_arg:
        print("No image given; sample NOT added.", file=sys.stderr)
        return 1
    image_path = os.path.abspath(image_arg)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}; sample NOT added.", file=sys.stderr)
        return 1

    # --- ground truth (never invented) ------------------------------------
    distance_arg = args.distance or _ask("Independently measured distance (m)", interactive=interactive)
    try:
        distance = _parse_distance(distance_arg)
    except ValueError as exc:
        print(f"{exc}; sample NOT added (ground truth must be independently measured).", file=sys.stderr)
        return 1

    # --- class name -------------------------------------------------------
    class_name = args.class_name or _ask("Target class", interactive=interactive)
    if not class_name:
        print("No target class; sample NOT added.", file=sys.stderr)
        return 1

    # --- measurement method -------------------------------------------------
    method = args.method or _ask("Measurement method", default="tape_measure", interactive=interactive)
    if not method:
        print("No measurement method; sample NOT added.", file=sys.stderr)
        return 1

    # --- optional fields -----------------------------------------------------
    optional = {}
    for key, prompt in (("scene", "Scene"), ("lighting", "Lighting"), ("pose", "Pose")):
        value = getattr(args, key)
        if not value and interactive:
            value = _ask(prompt, interactive=True)
        if value:
            optional[key] = value
    notes = args.notes or _ask("Notes (optional)", interactive=interactive) or ""

    # --- bounding box -------------------------------------------------------
    bbox = None
    if args.no_bbox:
        bbox = None
    else:
        bbox_arg = args.bbox or _ask("bbox x,y,width,height [skip]", interactive=interactive)
        if bbox_arg:
            try:
                x, y, w, h = _parse_bbox(bbox_arg)
                from modules.distance_estimator.dataset import BoundingBox  # noqa: E402

                bbox = BoundingBox(x=x, y=y, width=w, height=h)
            except ValueError as exc:
                print(f"Invalid bbox ({exc}); continuing WITHOUT bbox. "
                      "Bbox-less samples cannot be benchmarked.", file=sys.stderr)

    # --- copy image + register sample ----------------------------------------
    sample_id = _next_id(set(dataset.sample_ids))
    ext = os.path.splitext(image_path)[1] or ".jpg"
    safe_ext = ext if re.fullmatch(r"\.[A-Za-z0-9]+", ext) else ".jpg"
    target_image = os.path.join("images", sample_id + safe_ext)
    target_abs = os.path.join(images_dir, sample_id + safe_ext)
    shutil.copy2(image_path, target_abs)

    sample = GroundTruthSample(
        id=sample_id,
        image=target_image,
        ground_truth_distance_m=distance,
        class_name=class_name,
        measurement_method=method,
        bbox=bbox,
        detection_confidence=None,
        scene=optional.get("scene"),
        lighting=optional.get("lighting"),
        pose=optional.get("pose"),
        notes=str(notes),
    )
    dataset.samples.append(sample)
    dataset.save_manifest()
    print(f"Added sample {sample_id} (distance={distance:.3f} m, class={class_name}).")
    print(f"Image copied to {target_abs}")
    print("Ground truth must be independently measured; it was NOT estimated here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())