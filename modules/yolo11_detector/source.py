import threading

import cv2

from modules.yolo11_detector.config import DEFAULT_HEIGHT, DEFAULT_WIDTH


def _try_open_and_read(index, result):
    try:
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                result[index] = (cap, frame.shape[1], frame.shape[0])
                return
            cap.release()
        else:
            cap.release()
    except Exception:
        pass
    result[index] = None


def initialize_capture():
    # Fast path: try camera 0 directly (most common case)
    probe_result = {}
    t = threading.Thread(target=_try_open_and_read, args=(0, probe_result))
    t.daemon = True
    t.start()
    t.join(timeout=5)

    if probe_result.get(0) is not None:
        cap, w, h = probe_result[0]
        print(f"Camera 0 opened ({w}x{h})")
        return cap, "camera", w, h

    # Camera 0 failed — scan for available cameras
    print("Camera 0 not available, scanning...")
    available = []
    for i in range(1, 5):
        result_i = {}
        t = threading.Thread(target=_try_open_and_read, args=(i, result_i))
        t.daemon = True
        t.start()
        t.join(timeout=3)
        if result_i.get(i) is not None:
            available.append(i)
            probe_result[i] = result_i[i]

    if not available:
        print("No cameras found")
        return None, None, DEFAULT_WIDTH, DEFAULT_HEIGHT

    if len(available) == 1:
        index = available[0]
        cap, w, h = probe_result[index]
        print(f"Single camera found (index {index})")
        return cap, "camera", w, h

    print(f"\n{len(available)} cameras found:")
    for i, idx in enumerate(available, start=1):
        print(f"  {i}. Camera {idx}")
    try:
        choice = int(input("Select camera number: ")) - 1
        index = available[choice] if 0 <= choice < len(available) else available[0]
    except (ValueError, IndexError):
        index = available[0]
    print(f"Using camera {index}")

    cap, w, h = probe_result[index]
    return cap, "camera", w, h


def read_frame(cap, source_type):
    if cap is None or not cap.isOpened():
        return False, None
    return cap.read()
