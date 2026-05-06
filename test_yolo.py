from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/laser_points_train/weights/best.pt")

# Run prediction on image
results = model.predict(
    source="datasets/laser_red_only_v2/images/test/IMG_20250415_114516_jpg.rf.6b1e48c871af292adfa4fcf86a6b1155.jpg",   # change to your image path
    conf=0.12,
    imgsz=960,
    show=True,           # opens window with detection
    save=True            # saves output in runs/detect/predict
)

print(results)