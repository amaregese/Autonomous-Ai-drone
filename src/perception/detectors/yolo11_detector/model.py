import os

from ultralytics import YOLO


def load_model(model_path="YOLO/yolo11n.pt"):
    print(f"Loading YOLO model from: {model_path}")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        model = YOLO(model_path)
        classes = model.names
        print(f"✓ YOLO loaded! {len(classes)} classes available")
        return model, classes
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return None, []
