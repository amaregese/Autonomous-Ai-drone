import os
import time

import cv2

from src.perception.detectors.yolo11_detector.config import DEFAULT_HEIGHT, DEFAULT_WIDTH, VIDEO_FOLDER

_video_fps = 30.0
_last_frame_time = 0.0


def list_videos():
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER, exist_ok=True)
        return []
    return [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]


def select_video():
    videos = list_videos()
    if not videos:
        print(f"No videos found in '{VIDEO_FOLDER}' folder")
        return None

    print("\nAvailable videos:")
    for i, video_name in enumerate(videos, start=1):
        print(f"  {i}. {video_name}")

    try:
        choice = int(input("Select video number: ")) - 1
        if 0 <= choice < len(videos):
            return os.path.join(VIDEO_FOLDER, videos[choice])
    except Exception:
        pass
    return None


def initialize_capture(source='camera', video_path=None):
    global _video_fps

    if source == 'camera':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not found")
            return None, None, DEFAULT_WIDTH, DEFAULT_HEIGHT
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
        print("✓ Camera opened")
        return cap, "camera", DEFAULT_WIDTH, DEFAULT_HEIGHT

    if source == 'video':
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                _video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✓ Video opened: {output_width}x{output_height} @ {_video_fps:.1f} FPS")
                return cap, "video", output_width, output_height
            else:
                print(f"Failed to open video: {video_path}")
                return None, None, DEFAULT_WIDTH, DEFAULT_HEIGHT
        else:
            video_path = select_video()
            if video_path:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    _video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✓ Video opened: {output_width}x{output_height} @ {_video_fps:.1f} FPS")
                    return cap, "video", output_width, output_height

    return None, None, DEFAULT_WIDTH, DEFAULT_HEIGHT


def read_frame(cap, source_type):
    global _last_frame_time

    if cap is None or not cap.isOpened():
        return False, None

    if source_type == "camera":
        for _ in range(5):
            cap.grab()

    if source_type == "video":
        frame_interval = 1.0 / _video_fps
        elapsed = time.perf_counter() - _last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        _last_frame_time = time.perf_counter()

    ret, frame = cap.read()
    if not ret and source_type == "video":
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _last_frame_time = time.perf_counter()
        ret, frame = cap.read()

    return ret, frame
