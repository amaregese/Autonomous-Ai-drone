import math

import cv2
import numpy as np


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


def compute_color_histogram(frame, obj):
    if frame is None or obj is None or obj.area <= 0:
        return None

    roi = frame[obj.Top:obj.Bottom, obj.Left:obj.Right]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compute_mean_hsv(frame, obj):
    if frame is None or obj is None or obj.area <= 0:
        return None

    roi = frame[obj.Top:obj.Bottom, obj.Left:obj.Right]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return tuple(float(value) for value in np.mean(hsv.reshape(-1, 3), axis=0))


def compute_edge_density(frame, obj):
    if frame is None or obj is None or obj.area <= 0:
        return 0.0

    roi = frame[obj.Top:obj.Bottom, obj.Left:obj.Right]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges)) / max(edges.size, 1)


def extract_features(frame, obj):
    return {
        "class_name": obj.class_name,
        "area": obj.area,
        "aspect_ratio": obj.aspect_ratio,
        "center": obj.Center,
        "color_histogram": compute_color_histogram(frame, obj),
        "mean_hsv": compute_mean_hsv(frame, obj),
        "edge_density": compute_edge_density(frame, obj),
        "width": obj.width,
        "height": obj.height,
    }


def compare_histograms(hist1, hist2):
    if hist1 is None or hist2 is None:
        return 0.0
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


def compute_iou(obj_a, obj_b):
    inter_left = max(obj_a.Left, obj_b.Left)
    inter_top = max(obj_a.Top, obj_b.Top)
    inter_right = min(obj_a.Right, obj_b.Right)
    inter_bottom = min(obj_a.Bottom, obj_b.Bottom)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    union_area = obj_a.area + obj_b.area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def score_detection_match(reference_obj, reference_features, candidate, frame):
    if reference_obj is None or reference_features is None:
        return 0.0

    if candidate.class_name != reference_features["class_name"]:
        return 0.0

    score = 40.0
    score += compute_iou(reference_obj, candidate) * 30.0

    area_ratio = min(reference_features["area"], candidate.area) / max(reference_features["area"], candidate.area, 1)
    score += area_ratio * 10.0

    aspect_ratio = reference_features["aspect_ratio"]
    aspect_delta = abs(aspect_ratio - candidate.aspect_ratio) / max(aspect_ratio, candidate.aspect_ratio, 1e-6)
    score += max(0.0, 1.0 - aspect_delta) * 5.0

    ref_center = reference_features["center"]
    dx = candidate.Center[0] - ref_center[0]
    dy = candidate.Center[1] - ref_center[1]
    distance = math.hypot(dx, dy)
    frame_diagonal = math.hypot(frame.shape[1], frame.shape[0])
    score += max(0.0, 1.0 - (distance / max(frame_diagonal * 0.35, 1.0))) * 10.0

    hist_score = compare_histograms(reference_features["color_histogram"], compute_color_histogram(frame, candidate))
    score += max(0.0, hist_score) * 5.0
    score += candidate.confidence * 5.0
    return score


def score_reacquisition_match(reference_features, candidate, frame, predicted_center=None):
    if not reference_features or candidate.class_name != reference_features["class_name"]:
        return 0.0

    area_ratio = min(reference_features["area"], candidate.area) / max(reference_features["area"], candidate.area, 1)
    if area_ratio < 0.45:
        return 0.0

    aspect_ratio = reference_features["aspect_ratio"]
    aspect_delta = abs(aspect_ratio - candidate.aspect_ratio) / max(aspect_ratio, candidate.aspect_ratio, 1e-6)
    aspect_similarity = max(0.0, 1.0 - aspect_delta)
    if aspect_similarity < 0.55:
        return 0.0

    reference_center = predicted_center or reference_features["center"]
    dx = candidate.Center[0] - reference_center[0]
    dy = candidate.Center[1] - reference_center[1]
    distance = math.hypot(dx, dy)
    frame_diagonal = math.hypot(frame.shape[1], frame.shape[0])
    if distance > frame_diagonal * 0.45:
        return 0.0

    score = 25.0
    score += area_ratio * 18.0

    score += aspect_similarity * 14.0
    score += max(0.0, 1.0 - (distance / max(frame_diagonal * 0.55, 1.0))) * 16.0

    hist_score = compare_histograms(reference_features["color_histogram"], compute_color_histogram(frame, candidate))
    score += max(0.0, hist_score) * 12.0

    ref_hsv = reference_features.get("mean_hsv")
    cand_hsv = compute_mean_hsv(frame, candidate)
    if ref_hsv is not None and cand_hsv is not None:
        hue_delta = min(abs(ref_hsv[0] - cand_hsv[0]), 180 - abs(ref_hsv[0] - cand_hsv[0])) / 90.0
        sat_delta = abs(ref_hsv[1] - cand_hsv[1]) / 255.0
        val_delta = abs(ref_hsv[2] - cand_hsv[2]) / 255.0
        hsv_similarity = max(0.0, 1.0 - ((hue_delta * 0.5) + (sat_delta * 0.3) + (val_delta * 0.2)))
        score += hsv_similarity * 8.0

    edge_delta = abs(reference_features.get("edge_density", 0.0) - compute_edge_density(frame, candidate))
    score += max(0.0, 1.0 - min(edge_delta / 0.25, 1.0)) * 5.0

    score += candidate.confidence * 8.0
    return score
