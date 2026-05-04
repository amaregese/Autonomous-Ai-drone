import os
import time
import cv2
import numpy as np
from pathlib import Path
from contextlib import contextmanager

from src.perception.detectors.yolo11_detector import config
from src.perception.detectors.yolo11_detector.matching import (
    clamp_box,
    extract_features,
    score_detection_match,
    score_reacquisition_match,
)
from src.perception.detectors.yolo11_detector.model import load_model
from src.perception.detectors.yolo11_detector.source import initialize_capture, read_frame
from src.perception.detectors.yolo11_detector.types import Detection


@contextmanager
def _suppress_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

# Global variables
cap = None
source_type = None
model = None
laser_model = None
classes = []
laser_classes = {}
output_width = config.DEFAULT_WIDTH
output_height = config.DEFAULT_HEIGHT
selected_class = None
selected_object = None
selected_features = {}
target_memory = {}
last_known_object = None
last_known_center = None
last_known_velocity = (0.0, 0.0)
lost_frame_count = 0
tracking_lost = False
tracking_confidence = 0
last_frame = None
frame_counter = 0
last_detections = []
last_fps_timestamp = time.perf_counter()
current_fps = 0.0

# Model selection (keep internal, no display)
USE_LASER_MODEL = False
LASER_MODEL_PATH = "models/yolo11_laser_points.pt"
MODEL_MODE = "combined"  # Internal only, not shown on display


def _pick_unique_candidate(scored_candidates, minimum_score, uniqueness_margin):
    if not scored_candidates:
        return None, 0.0

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_match = scored_candidates[0]
    if best_score < minimum_score:
        return None, best_score

    if len(scored_candidates) > 1:
        second_best_score = scored_candidates[1][0]
        if best_score - second_best_score < uniqueness_margin:
            return None, best_score

    return best_match, best_score


def initialize_detector(model_path="models/yolo11n.pt", laser_model_path=None, source='camera', video_path=None):
    """Initialize both general and laser-specific models"""
    global cap, source_type, output_width, output_height, model, classes
    global laser_model, laser_classes, USE_LASER_MODEL, LASER_MODEL_PATH

    # Load general YOLO model
    with _suppress_stderr():
        model, classes = load_model(model_path)
    if model is None:
        print(f"❌ Failed to load general model from {model_path}")
        return False

    print(f"✅ Loaded general model: {model_path}")
    print(f"   Classes: {len(classes)}")

    # Load laser-specific model if provided
    if laser_model_path and Path(laser_model_path).exists():
        with _suppress_stderr():
            laser_model, laser_classes = load_model(laser_model_path)
        if laser_model:
            print(f"✅ Loaded laser model: {laser_model_path}")
            if isinstance(laser_classes, dict):
                laser_class_names = list(laser_classes.values())
            else:
                laser_class_names = laser_classes if isinstance(laser_classes, list) else [str(laser_classes)]
            print(f"   Laser classes: {laser_class_names}")
            USE_LASER_MODEL = True
            LASER_MODEL_PATH = laser_model_path
        else:
            print(f"⚠️ Failed to load laser model from {laser_model_path}")
    else:
        print(f"ℹ️ No laser model specified or file not found")

    # Initialize video capture
    cap, source_type, output_width, output_height = initialize_capture(source, video_path)
    return cap is not None


def set_model_mode(mode):
    """Set which model to use internally"""
    global MODEL_MODE
    valid_modes = ['general', 'laser', 'combined']
    if mode in valid_modes:
        MODEL_MODE = mode
        mode_names = {'general': 'General', 'laser': 'Laser Only', 'combined': 'Combined'}
        print(f"Model mode: {mode_names.get(mode, mode)}")
        return True
    return False


def get_available_models():
    """Get information about available models"""
    general_class_list = classes if isinstance(classes, list) else list(classes.values()) if isinstance(classes,
                                                                                                        dict) else []
    laser_class_list = []
    if laser_classes:
        if isinstance(laser_classes, dict):
            laser_class_list = list(laser_classes.values())
        elif isinstance(laser_classes, list):
            laser_class_list = laser_classes
        else:
            laser_class_list = [str(laser_classes)]

    models_info = {
        'general': {
            'loaded': model is not None,
            'classes': len(general_class_list),
        },
        'laser': {
            'loaded': laser_model is not None,
            'classes': len(laser_class_list),
        },
        'mode': MODEL_MODE
    }
    return models_info


def configure_detector(
        confidence_threshold=None,
        iou_threshold=None,
        min_box_area_ratio=None,
        inference_img_size=None,
        laser_confidence_threshold=None,
):
    config.apply_runtime_overrides(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        min_box_area_ratio=min_box_area_ratio,
        inference_img_size=inference_img_size,
    )

    if laser_confidence_threshold is not None:
        config.LASER_CONFIDENCE_THRESHOLD = laser_confidence_threshold


def _should_reuse_last_detections():
    frame_skip = config.FRAME_SKIP_VIDEO if source_type == "video" else config.FRAME_SKIP_CAMERA
    return frame_counter % frame_skip != 0 and len(last_detections) > 0


def _predict_with_model(model_to_use, frame, is_laser_model=False):
    """Predict using specified model"""
    conf_threshold = config.LASER_CONFIDENCE_THRESHOLD if is_laser_model else config.CONFIDENCE_THRESHOLD

    with _suppress_stderr():
        results = model_to_use.predict(
            source=frame,
            conf=conf_threshold,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            verbose=False,
        )

    detections = []
    frame_height, frame_width = frame.shape[:2]
    min_box_area = frame_height * frame_width * config.MIN_BOX_AREA_RATIO

    if results[0].boxes is None:
        return detections

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    if is_laser_model:
        if isinstance(laser_classes, dict):
            class_list = laser_classes
        else:
            class_list = {i: name for i, name in enumerate(laser_classes)} if laser_classes else {}
    else:
        class_list = {i: classes[i] for i in range(len(classes))} if classes else {}

    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = clamp_box(*box[:4], frame_width, frame_height)
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = class_list.get(cls_id, f"class_{cls_id}")
        detection = Detection(x1, y1, x2, y2, cls_id, class_name, float(conf))

        if detection.area < min_box_area:
            continue

        detections.append(detection)

    return detections


def _predict_boxes(frame):
    """Predict boxes using selected model(s) internally"""
    global MODEL_MODE

    all_detections = []

    if MODEL_MODE == 'general':
        all_detections = _predict_with_model(model, frame, is_laser_model=False)
    elif MODEL_MODE == 'laser' and USE_LASER_MODEL:
        all_detections = _predict_with_model(laser_model, frame, is_laser_model=True)
    elif MODEL_MODE == 'combined' and USE_LASER_MODEL:
        general_detections = _predict_with_model(model, frame, is_laser_model=False)
        laser_detections = _predict_with_model(laser_model, frame, is_laser_model=True)
        all_detections = general_detections + laser_detections
    else:
        all_detections = _predict_with_model(model, frame, is_laser_model=False)

    return all_detections


def _update_selected_object(detections, frame):
    global selected_object, selected_features, target_memory
    global last_known_object, last_known_center, last_known_velocity
    global lost_frame_count, tracking_lost, tracking_confidence

    tracking_confidence = 0
    if not selected_class:
        tracking_lost = False
        return

    same_class = [detection for detection in detections if detection.class_name == selected_class]
    if selected_object is not None and same_class:
        reference_features = selected_features or extract_features(last_frame if last_frame is not None else frame,
                                                                   selected_object)
        scored_candidates = []

        for detection in same_class:
            score = score_detection_match(selected_object, reference_features, detection, frame)
            scored_candidates.append((score, detection))

        best_match, best_score = _pick_unique_candidate(
            scored_candidates,
            config.TRACK_MATCH_THRESHOLD,
            config.TRACK_UNIQUENESS_MARGIN,
        )
        tracking_confidence = min(100.0, best_score)
        if best_match is not None:
            previous_center = last_known_center or selected_object.Center
            best_match.is_selected = True
            selected_object = best_match
            selected_features = extract_features(frame, best_match)
            target_memory = dict(selected_features)
            last_known_object = best_match
            last_known_velocity = (
                best_match.Center[0] - previous_center[0],
                best_match.Center[1] - previous_center[1],
            )
            last_known_center = best_match.Center
            lost_frame_count = 0
            tracking_lost = False
            return

    if selected_object is not None:
        selected_object.is_selected = False
        last_known_object = selected_object
        last_known_center = selected_object.Center
    selected_object = None
    selected_features = {}
    tracking_lost = True
    lost_frame_count += 1

    if not same_class or lost_frame_count > config.TARGET_MEMORY_FRAMES:
        return

    predicted_center = None
    if last_known_center is not None:
        predicted_center = (
            int(last_known_center[0] + (last_known_velocity[0] * min(lost_frame_count, 10))),
            int(last_known_center[1] + (last_known_velocity[1] * min(lost_frame_count, 10))),
        )

    memory_features = target_memory or (extract_features(last_frame if last_frame is not None else frame,
                                                         last_known_object) if last_known_object is not None else {})
    if not memory_features:
        return

    scored_candidates = []
    for detection in same_class:
        area_ratio = min(memory_features["area"], detection.area) / max(memory_features["area"], detection.area, 1)
        if area_ratio < config.MIN_REACQUIRE_AREA_RATIO:
            continue

        if predicted_center is not None:
            dx = detection.Center[0] - predicted_center[0]
            dy = detection.Center[1] - predicted_center[1]
            distance = (dx * dx + dy * dy) ** 0.5
            frame_diagonal = (frame.shape[1] ** 2 + frame.shape[0] ** 2) ** 0.5
            if distance > frame_diagonal * config.MAX_REACQUIRE_CENTER_SHIFT_RATIO:
                continue

        score = score_reacquisition_match(memory_features, detection, frame, predicted_center)
        scored_candidates.append((score, detection))

    best_match, best_score = _pick_unique_candidate(
        scored_candidates,
        config.REACQUIRE_MATCH_THRESHOLD,
        config.REACQUIRE_UNIQUENESS_MARGIN,
    )
    tracking_confidence = min(100.0, best_score)
    if best_match is None:
        return

    best_match.is_selected = True
    selected_object = best_match
    selected_features = extract_features(frame, best_match)
    target_memory = dict(selected_features)
    if last_known_center is not None:
        last_known_velocity = (
            best_match.Center[0] - last_known_center[0],
            best_match.Center[1] - last_known_center[1],
        )
    last_known_center = best_match.Center
    last_known_object = best_match
    tracking_lost = False
    lost_frame_count = 0


def detect_objects(frame):
    global frame_counter, last_detections, last_frame

    frame_counter += 1
    if _should_reuse_last_detections():
        return last_detections

    detections = _predict_boxes(frame)
    _update_selected_object(detections, frame)

    last_detections = detections
    last_frame = frame.copy()
    return detections


def get_detections():
    global output_width, output_height, last_fps_timestamp, current_fps

    ret, frame = read_frame(cap, source_type)
    if not ret:
        return [], 0.0, np.zeros((config.DEFAULT_HEIGHT, config.DEFAULT_WIDTH, 3), dtype=np.uint8)

    if frame.shape[1] > config.DEFAULT_WIDTH:
        scale = config.DEFAULT_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (config.DEFAULT_WIDTH, int(frame.shape[0] * scale)))

    output_width, output_height = frame.shape[1], frame.shape[0]
    detections = detect_objects(frame)

    now = time.perf_counter()
    elapsed = now - last_fps_timestamp
    if elapsed > 0:
        current_fps = 1.0 / elapsed
    last_fps_timestamp = now

    return detections, current_fps, frame


def get_image_size():
    return output_width, output_height


def get_camera():
    return cap, source_type


def get_selected_object():
    return selected_object


def get_selected_class():
    return selected_class


def get_selected_object_size():
    if selected_object:
        return selected_object.area
    return 0


def select_object(obj, frame=None):
    global selected_class, selected_object, selected_features, target_memory
    global last_known_object, last_known_center, last_known_velocity
    global lost_frame_count, tracking_lost, tracking_confidence

    if obj is None:
        return False

    if selected_object:
        selected_object.is_selected = False

    selected_class = obj.class_name
    selected_object = obj
    obj.is_selected = True
    reference_frame = frame if frame is not None else last_frame
    selected_features = extract_features(reference_frame, obj)
    target_memory = dict(selected_features)
    last_known_object = obj
    last_known_center = obj.Center
    last_known_velocity = (0.0, 0.0)
    lost_frame_count = 0
    tracking_lost = False
    tracking_confidence = obj.confidence * 100
    print(f"\n✓ Selected: {selected_class} ({obj.confidence * 100:.1f}%)")
    print(f"  Object size: {obj.area} pixels")
    return True


def clear_selection():
    global selected_class, selected_object, selected_features, target_memory
    global last_known_object, last_known_center, last_known_velocity
    global lost_frame_count, tracking_lost, tracking_confidence

    if selected_object:
        selected_object.is_selected = False
    selected_class = None
    selected_object = None
    selected_features = {}
    target_memory = {}
    last_known_object = None
    last_known_center = None
    last_known_velocity = (0.0, 0.0)
    lost_frame_count = 0
    tracking_lost = False
    tracking_confidence = 0


def get_tracking_status():
    return tracking_lost


def reset_tracking_lost():
    global tracking_lost
    tracking_lost = False


def get_tracking_confidence():
    return tracking_confidence


def set_selection_mode(mode):
    pass


def cleanup():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()