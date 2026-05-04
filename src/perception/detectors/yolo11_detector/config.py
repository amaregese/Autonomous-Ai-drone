# Default thresholds for general object detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MIN_BOX_AREA_RATIO = 0.0005
DEFAULT_INFERENCE_IMG_SIZE = 480

# Laser-specific detection thresholds (lower for better sensitivity)
DEFAULT_LASER_CONFIDENCE_THRESHOLD = 0.12
DEFAULT_LASER_IOU_THRESHOLD = 0.3
DEFAULT_LASER_MIN_BOX_AREA_RATIO = 0.00001  # Very small boxes allowed
DEFAULT_LASER_INFERENCE_IMG_SIZE = 960  # Higher resolution for small laser points

# Current configuration values
CONFIDENCE_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD
IOU_THRESHOLD = DEFAULT_IOU_THRESHOLD
MIN_BOX_AREA_RATIO = DEFAULT_MIN_BOX_AREA_RATIO
INFERENCE_IMG_SIZE = DEFAULT_INFERENCE_IMG_SIZE

# Laser model specific configuration
LASER_CONFIDENCE_THRESHOLD = DEFAULT_LASER_CONFIDENCE_THRESHOLD
LASER_IOU_THRESHOLD = DEFAULT_LASER_IOU_THRESHOLD
LASER_MIN_BOX_AREA_RATIO = DEFAULT_LASER_MIN_BOX_AREA_RATIO
LASER_INFERENCE_IMG_SIZE = DEFAULT_LASER_INFERENCE_IMG_SIZE

# Model selection modes
MODEL_MODES = {
    'general': 'General object detection only',
    'laser': 'Laser point detection only',
    'auto': 'Auto-select based on context',
    'combined': 'Use both models simultaneously'
}
CURRENT_MODEL_MODE = 'combined'  # Options: 'general', 'laser', 'auto', 'combined'

# Frame skipping for performance
FRAME_SKIP_CAMERA = 1
FRAME_SKIP_VIDEO = 1

# Video settings
VIDEO_FOLDER = "data/videos"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

# Tracking configuration (general objects)
TRACK_MATCH_THRESHOLD = 45.0
REACQUIRE_MATCH_THRESHOLD = 38.0
TARGET_MEMORY_FRAMES = 90
TRACK_UNIQUENESS_MARGIN = 7.0
REACQUIRE_UNIQUENESS_MARGIN = 12.0
MIN_REACQUIRE_AREA_RATIO = 0.45
MAX_REACQUIRE_CENTER_SHIFT_RATIO = 0.45

# Laser tracking configuration (more sensitive)
LASER_TRACK_MATCH_THRESHOLD = 35.0  # Lower threshold for laser points
LASER_REACQUIRE_MATCH_THRESHOLD = 28.0
LASER_TARGET_MEMORY_FRAMES = 45  # Shorter memory for fast-moving lasers
LASER_TRACK_UNIQUENESS_MARGIN = 5.0
LASER_MIN_REACQUIRE_AREA_RATIO = 0.25  # Allow more size variation
LASER_MAX_REACQUIRE_CENTER_SHIFT_RATIO = 0.6  # Allow more movement

# Model paths
GENERAL_MODEL_PATH = "models/yolo11n.pt"
LASER_MODEL_PATH = "models/yolo11_laser_points.pt"

# Object size classification (for auto mode switching)
SMALL_OBJECT_AREA_RATIO = 0.01  # Objects smaller than 1% of frame are considered "small"
LASER_CLASS_KEYWORDS = ['laser', 'point', 'dot', 'spot']  # Keywords to identify laser classes

# Display settings
SHOW_LASER_ONLY_MODE = False  # When True, only show laser detections
DRAW_LASER_BOXES_COLOR = (0, 255, 255)  # Yellow for laser points
DRAW_GENERAL_BOXES_COLOR = (0, 255, 0)  # Green for general objects
DRAW_SELECTED_BOXES_COLOR = (255, 0, 0)  # Red for selected object

# Confidence color thresholds
HIGH_CONFIDENCE = 0.7
MEDIUM_CONFIDENCE = 0.3
CONFIDENCE_COLORS = {
    'high': (0, 255, 0),  # Green
    'medium': (0, 255, 255),  # Yellow
    'low': (0, 0, 255)  # Red
}


def reset_runtime_config():
    """Reset general detection config to defaults"""
    global CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MIN_BOX_AREA_RATIO, INFERENCE_IMG_SIZE

    CONFIDENCE_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD
    IOU_THRESHOLD = DEFAULT_IOU_THRESHOLD
    MIN_BOX_AREA_RATIO = DEFAULT_MIN_BOX_AREA_RATIO
    INFERENCE_IMG_SIZE = DEFAULT_INFERENCE_IMG_SIZE


def reset_laser_config():
    """Reset laser detection config to defaults"""
    global LASER_CONFIDENCE_THRESHOLD, LASER_IOU_THRESHOLD, LASER_MIN_BOX_AREA_RATIO, LASER_INFERENCE_IMG_SIZE

    LASER_CONFIDENCE_THRESHOLD = DEFAULT_LASER_CONFIDENCE_THRESHOLD
    LASER_IOU_THRESHOLD = DEFAULT_LASER_IOU_THRESHOLD
    LASER_MIN_BOX_AREA_RATIO = DEFAULT_LASER_MIN_BOX_AREA_RATIO
    LASER_INFERENCE_IMG_SIZE = DEFAULT_LASER_INFERENCE_IMG_SIZE


def apply_runtime_overrides(
        confidence_threshold=None,
        iou_threshold=None,
        min_box_area_ratio=None,
        inference_img_size=None,
        laser_confidence_threshold=None,
        laser_iou_threshold=None,
        laser_min_box_area_ratio=None,
        laser_inference_img_size=None,
):
    """Apply overrides to both general and laser configurations"""

    # Reset both configs to defaults first
    reset_runtime_config()
    reset_laser_config()

    # Apply general detection overrides
    if confidence_threshold is not None:
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = float(confidence_threshold)

    if iou_threshold is not None:
        global IOU_THRESHOLD
        IOU_THRESHOLD = float(iou_threshold)

    if min_box_area_ratio is not None:
        global MIN_BOX_AREA_RATIO
        MIN_BOX_AREA_RATIO = float(min_box_area_ratio)

    if inference_img_size is not None:
        global INFERENCE_IMG_SIZE
        INFERENCE_IMG_SIZE = int(inference_img_size)

    # Apply laser detection overrides
    if laser_confidence_threshold is not None:
        global LASER_CONFIDENCE_THRESHOLD
        LASER_CONFIDENCE_THRESHOLD = float(laser_confidence_threshold)

    if laser_iou_threshold is not None:
        global LASER_IOU_THRESHOLD
        LASER_IOU_THRESHOLD = float(laser_iou_threshold)

    if laser_min_box_area_ratio is not None:
        global LASER_MIN_BOX_AREA_RATIO
        LASER_MIN_BOX_AREA_RATIO = float(laser_min_box_area_ratio)

    if laser_inference_img_size is not None:
        global LASER_INFERENCE_IMG_SIZE
        LASER_INFERENCE_IMG_SIZE = int(laser_inference_img_size)


def set_model_mode(mode):
    """Set the detection model mode"""
    global CURRENT_MODEL_MODE

    if mode in MODEL_MODES:
        CURRENT_MODEL_MODE = mode
        return True
    return False


def get_tracking_params(is_laser=False):
    """Get tracking parameters based on object type"""
    if is_laser:
        return {
            'track_match_threshold': LASER_TRACK_MATCH_THRESHOLD,
            'reacquire_match_threshold': LASER_REACQUIRE_MATCH_THRESHOLD,
            'target_memory_frames': LASER_TARGET_MEMORY_FRAMES,
            'track_uniqueness_margin': LASER_TRACK_UNIQUENESS_MARGIN,
            'reacquire_uniqueness_margin': LASER_REACQUIRE_UNIQUENESS_MARGIN,
            'min_reacquire_area_ratio': LASER_MIN_REACQUIRE_AREA_RATIO,
            'max_reacquire_center_shift_ratio': LASER_MAX_REACQUIRE_CENTER_SHIFT_RATIO,
        }
    else:
        return {
            'track_match_threshold': TRACK_MATCH_THRESHOLD,
            'reacquire_match_threshold': REACQUIRE_MATCH_THRESHOLD,
            'target_memory_frames': TARGET_MEMORY_FRAMES,
            'track_uniqueness_margin': TRACK_UNIQUENESS_MARGIN,
            'reacquire_uniqueness_margin': REACQUIRE_UNIQUENESS_MARGIN,
            'min_reacquire_area_ratio': MIN_REACQUIRE_AREA_RATIO,
            'max_reacquire_center_shift_ratio': MAX_REACQUIRE_CENTER_SHIFT_RATIO,
        }


def is_laser_object(class_name):
    """Check if a class name likely represents a laser point"""
    class_lower = class_name.lower()
    return any(keyword in class_lower for keyword in LASER_CLASS_KEYWORDS)


def get_confidence_color(confidence):
    """Return color based on confidence level"""
    if confidence >= HIGH_CONFIDENCE:
        return CONFIDENCE_COLORS['high']
    elif confidence >= MEDIUM_CONFIDENCE:
        return CONFIDENCE_COLORS['medium']
    else:
        return CONFIDENCE_COLORS['low']


def print_config_status():
    """Print current configuration status for debugging"""
    print("\n" + "=" * 50)
    print("CONFIGURATION STATUS")
    print("=" * 50)
    print(f"Model Mode: {CURRENT_MODEL_MODE}")
    print(f"\nGeneral Detection:")
    print(f"  Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"  IOU: {IOU_THRESHOLD}")
    print(f"  Min Area Ratio: {MIN_BOX_AREA_RATIO}")
    print(f"  Image Size: {INFERENCE_IMG_SIZE}")
    print(f"\nLaser Detection:")
    print(f"  Confidence: {LASER_CONFIDENCE_THRESHOLD}")
    print(f"  IOU: {LASER_IOU_THRESHOLD}")
    print(f"  Min Area Ratio: {LASER_MIN_BOX_AREA_RATIO}")
    print(f"  Image Size: {LASER_INFERENCE_IMG_SIZE}")
    print(f"\nTracking:")
    print(f"  General Match Threshold: {TRACK_MATCH_THRESHOLD}")
    print(f"  Laser Match Threshold: {LASER_TRACK_MATCH_THRESHOLD}")
    print("=" * 50 + "\n")


# Example usage in your main script:
if __name__ == "__main__":
    # Configure with different settings for general and laser
    apply_runtime_overrides(
        confidence_threshold=0.5,  # Higher for general objects
        iou_threshold=0.45,
        laser_confidence_threshold=0.12,  # Lower for laser points
        laser_inference_img_size=960  # Higher resolution for lasers
    )

    # Switch to combined mode
    set_model_mode('combined')

    # Print current config
    print_config_status()