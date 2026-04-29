# YOLO-based detector for all objects with smart tracking
import cv2
import numpy as np
import os
import math


class Detection:
    def __init__(self, left, top, right, bottom, class_id, class_name, confidence):
        self.Left = left
        self.Top = top
        self.Right = right
        self.Bottom = bottom
        self.Center = ((left + right) // 2, (top + bottom) // 2)
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.is_selected = False

        # Smart tracking features
        self.width = right - left
        self.height = bottom - top
        self.area = self.width * self.height
        self.aspect_ratio = self.width / self.height if self.height > 0 else 1

        # Color histogram (will be computed when selected)
        self.color_histogram = None
        self.hsv_image = None


# Global variables
cap = None
source_type = None
video_folder = "test_videos"
net = None
classes = []
output_width = 640
output_height = 480
selected_class = None
selected_object = None
selected_features = {}  # Store features of selected object
tracking_lost = False
tracking_confidence = 0  # Confidence score for current tracking
last_frame = None

# YOLO settings
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
FRAME_SKIP = 6
INPUT_SIZE = (160, 160)

frame_counter = 0
last_detections = []


def load_yolo():
    global net, classes
    try:
        with open("coco.namesj", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return True
    except Exception as e:
        print(f"YOLO Error: {e}")
        return False


def compute_color_histogram(frame, obj):
    """Compute color histogram for an object"""
    if frame is None or obj is None:
        return None

    # Extract object region
    roi = frame[obj.Top:obj.Bottom, obj.Left:obj.Right]
    if roi.size == 0:
        return None

    # Convert to HSV for better color representation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate histogram for Hue channel
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    return hist


def compare_color_histogram(hist1, hist2):
    """Compare two color histograms"""
    if hist1 is None or hist2 is None:
        return 0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def extract_features(frame, obj):
    """Extract all features for an object"""
    features = {
        'class_name': obj.class_name,
        'width': obj.width,
        'height': obj.height,
        'area': obj.area,
        'aspect_ratio': obj.aspect_ratio,
        'center': obj.Center,
        'color_histogram': compute_color_histogram(frame, obj)
    }
    return features


def match_features(features1, features2):
    """Match two objects by features with weighted scoring"""
    score = 0

    # Same class (highest weight)
    if features1['class_name'] == features2['class_name']:
        score += 50

    # Size similarity (area)
    area_ratio = min(features1['area'], features2['area']) / max(features1['area'], features2['area'])
    score += area_ratio * 20

    # Aspect ratio similarity
    ar_ratio = 1 - abs(features1['aspect_ratio'] - features2['aspect_ratio']) / max(features1['aspect_ratio'],
                                                                                    features2['aspect_ratio'])
    score += ar_ratio * 15

    # Color histogram similarity
    if features1['color_histogram'] is not None and features2['color_histogram'] is not None:
        color_score = compare_color_histogram(features1['color_histogram'], features2['color_histogram'])
        score += color_score * 15

    # Position proximity (lower weight - helps with tracking)
    dx = abs(features1['center'][0] - features2['center'][0])
    dy = abs(features1['center'][1] - features2['center'][1])
    distance = math.sqrt(dx ** 2 + dy ** 2)
    if distance < 200:
        score += (1 - distance / 200) * 10

    return score


def list_videos():
    if not os.path.exists(video_folder):
        os.makedirs(video_folder, exist_ok=True)
        return []
    return [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]


def select_video():
    videos = list_videos()
    if not videos:
        print("No videos found in 'test_videos' folder")
        return None

    print("\nAvailable videos:")
    for i, vid in enumerate(videos):
        print(f"  {i + 1}. {vid}")

    try:
        choice = int(input("Select video number: ")) - 1
        if 0 <= choice < len(videos):
            return os.path.join(video_folder, videos[choice])
    except:
        pass
    return None


def initialize_detector():
    global cap, source_type, output_width, output_height

    if not load_yolo():
        return False

    print("\nSelect input source:")
    print("1 - Live Camera")
    print("2 - Recorded Video")
    choice = input("Enter choice: ").strip()

    if choice == "1":
        cap = cv2.VideoCapture(0)
        source_type = "camera"
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, output_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, output_height)
        return True
    elif choice == "2":
        video_path = select_video()
        if video_path:
            cap = cv2.VideoCapture(video_path)
            source_type = "video"
            if cap.isOpened():
                output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.set(cv2.CAP_PROP_FPS, 30)
                return True
    return False


def detect_objects(frame):
    global frame_counter, last_detections, selected_object, selected_features, tracking_confidence, last_frame

    frame_counter += 1

    if frame_counter % FRAME_SKIP != 0:
        return last_detections if last_detections else []

    if net is None:
        return []

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]] if class_ids[i] < len(classes) else "unknown"
            detections.append(Detection(x, y, x + w, y + h, class_ids[i], class_name, confidences[i]))

    # Smart tracking: Match selected object using features
    if selected_class and len(detections) > 0 and selected_object is not None:
        best_match = None
        best_score = 0

        # Extract features for current frame
        for d in detections:
            current_features = extract_features(frame, d)

            # Compute matching score
            score = match_features(selected_features, current_features)

            if score > best_score and score > 30:  # Threshold for good match
                best_score = score
                best_match = d

        # Update tracking confidence
        tracking_confidence = best_score

        if best_match is not None:
            # Found a good match
            if selected_object:
                selected_object.is_selected = False
            best_match.is_selected = True
            selected_object = best_match

            # Update features (adaptive tracking)
            selected_features = extract_features(frame, best_match)
        else:
            # No match found - object lost
            if selected_object:
                selected_object.is_selected = False
                selected_object = None

    last_detections = detections
    last_frame = frame.copy()
    return detections


def get_detections():
    global cap, source_type, output_width, output_height, last_frame

    if cap is None or not cap.isOpened():
        return [], 30, np.zeros((480, 640, 3), dtype=np.uint8)

    ret, frame = cap.read()
    if not ret:
        if source_type == "video":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            return [], 30, np.zeros((480, 640, 3), dtype=np.uint8)

    # Fast resize
    if frame.shape[1] > 640:
        scale = 640 / frame.shape[1]
        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)), interpolation=cv2.INTER_FAST)

    output_width, output_height = frame.shape[1], frame.shape[0]
    detections = detect_objects(frame)

    return detections, 30, frame


def get_image_size():
    return (output_width, output_height)


def get_selected_object():
    global selected_object
    return selected_object


def get_selected_class():
    global selected_class
    return selected_class


def get_tracking_confidence():
    global tracking_confidence
    return tracking_confidence


def select_object(obj, frame):
    """Select an object and extract its features for smart tracking"""
    global selected_class, selected_object, selected_features, tracking_lost

    if obj is None or frame is None:
        return False

    if selected_object:
        selected_object.is_selected = False

    selected_class = obj.class_name
    selected_object = obj
    obj.is_selected = True
    tracking_lost = False

    # Extract and store features for smart tracking
    selected_features = extract_features(frame, obj)

    print(f"\n✓ Selected: {selected_class} ({obj.confidence:.1f}%)")
    print(f"  Features: Size={obj.width}x{obj.height}, Area={obj.area}, Aspect={obj.aspect_ratio:.2f}")
    print(f"  Tracking confidence: High")
    return True


def clear_selection():
    """Clear the currently selected object"""
    global selected_class, selected_object, selected_features, tracking_lost, tracking_confidence
    if selected_object:
        selected_object.is_selected = False
    selected_class = None
    selected_object = None
    selected_features = {}
    tracking_lost = False
    tracking_confidence = 0
    print("\n✓ Selection cleared")


def get_tracking_status():
    global tracking_lost
    return tracking_lost


def set_tracking_lost(lost):
    global tracking_lost
    tracking_lost = lost


def cleanup():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()