import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained laser detection model")
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "runs/detect/laser_points_train/weights/best.pt"),
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets/laser_red_only_v2/images/val"),
        help="Directory with test images"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.12,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "test_results/laser_detection"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=2000,
        help="Delay in milliseconds between images (0 = wait for key press)"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save annotated images to output directory"
    )
    return parser.parse_args()


def test_single_image(model, image_path, conf_threshold, imgsz, output_dir=None, save_result=True):
    """Test model on a single image and optionally save result"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return None, None

    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Run inference
    results = model(image, conf=conf_threshold, imgsz=imgsz, verbose=False)

    # Extract detections
    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf
            })

    # Annotate image
    annotated = results[0].plot()

    # Add custom text overlay
    h, w = annotated.shape[:2]
    # Add title bar
    cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(annotated, f"Laser Detection - {image_path.name}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add info panel
    info_y = h - 60
    cv2.rectangle(annotated, (0, info_y - 5), (w, h), (0, 0, 0), -1)
    cv2.putText(annotated, f"Detections: {len(detections)} | Confidence: {args.conf_threshold}",
                (10, info_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save results if requested
    if save_result and output_dir:
        output_path = output_dir / f"result_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), annotated)

    return detections, annotated


def display_detection_window(image_path, annotated_image, detections, delay=2000):
    """Display annotated image in OpenCV window"""
    if annotated_image is None:
        return

    # Create window name
    window_name = f"Laser Detection - {image_path.name}"

    # Display the image
    cv2.imshow(window_name, annotated_image)

    # Add detection info to console
    print(f"\n📸 Image: {image_path.name}")
    print(f"   Found {len(detections)} laser point(s)")
    for i, det in enumerate(detections, 1):
        print(f"   Point {i}: confidence={det['confidence']:.3f}")

    # Wait for key press or delay
    if delay > 0:
        key = cv2.waitKey(delay) & 0xFF
    else:
        print("   Press any key to continue...")
        key = cv2.waitKey(0) & 0xFF

    # Close window on key press
    cv2.destroyWindow(window_name)

    # Return key for special handling
    return key


def display_all_images_grid(images_data, cols=2):
    """Display multiple images in a grid layout"""
    if not images_data:
        return

    num_images = len(images_data)
    rows = (num_images + cols - 1) // cols

    # Calculate figure size
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    # Display each image
    for idx, (img_path, annotated, detections) in enumerate(images_data):
        if annotated is not None:
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(annotated_rgb)
            axes[idx].set_title(f"{img_path.name}\n{len(detections)} laser points", fontsize=10)
            axes[idx].axis('off')

    plt.suptitle(f"Laser Detection Results (Confidence ≥ {args.conf_threshold})", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    global args
    args = parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        print("Checking common locations...")

        alt_paths = [
            PROJECT_ROOT / "runs/detect/laser_points_train/weights/best.pt",
            PROJECT_ROOT / "runs/detect/laser_points_train/weights/last.pt",
        ]

        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                print(f"Found model at: {model_path}")
                break
        else:
            raise FileNotFoundError(f"Model not found. Please specify --model parameter")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Setup directories
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    if args.save_results:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check if test directory exists
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        possible_dirs = [
            PROJECT_ROOT / "datasets/laser_red_only_v2/images/val",
            PROJECT_ROOT / "datasets/laser_red_only_v2/images/train",
        ]

        for possible_dir in possible_dirs:
            if possible_dir.exists():
                test_dir = possible_dir
                print(f"Using images from: {test_dir}")
                break
        else:
            print("No test images found. Please specify --test-dir")
            return

    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_dir.glob(ext))

    if not test_images:
        print(f"No test images found in {test_dir}")
        return

    print(f"\n📊 Configuration:")
    print(f"   Test images: {len(test_images)}")
    print(f"   Confidence threshold: {args.conf_threshold}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Display delay: {args.delay}ms (0 = manual)")
    print(f"   Saving results: {args.save_results}")
    print(f"   Output directory: {output_dir}\n")

    # Test each image and display
    all_results = []
    total_detections = 0
    images_with_detections = 0

    for i, img_path in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] Processing: {img_path.name}")

        # Run detection
        detections, annotated = test_single_image(
            model, img_path, args.conf_threshold, args.imgsz,
            output_dir if args.save_results else None,
            args.save_results
        )

        if detections is not None:
            num_detections = len(detections)
            total_detections += num_detections
            if num_detections > 0:
                images_with_detections += 1

            all_results.append((img_path, annotated, detections))

            # Display the image in a window
            display_detection_window(img_path, annotated, detections, args.delay)

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Images processed: {len(test_images)}")
    print(f"✅ Images with detections: {images_with_detections}")
    print(f"✅ Total laser points detected: {total_detections}")
    print(f"📈 Average per image: {total_detections / len(test_images):.2f}")

    if images_with_detections > 0:
        # Ask if user wants to see all results in a grid
        print("\n🎯 Press 'g' to see all results in a grid, or any other key to continue...")
        key = cv2.waitKey(1000) & 0xFF
        if key == ord('g'):
            display_all_images_grid(all_results, cols=3)

    print(f"\n💾 Results saved to: {output_dir}")
    cv2.destroyAllWindows()


def test_webcam():
    """Test model on webcam feed with live display"""
    model_path = PROJECT_ROOT / "runs/detect/laser_points_train/weights/best.pt"
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(0)

    print("\n📹 WEBCAM TEST MODE")
    print("   Controls:")
    print("   'q' - Quit")
    print("   's' - Save current frame")
    print("   '+' - Increase confidence threshold")
    print("   '-' - Decrease confidence threshold")

    conf_threshold = 0.12

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=conf_threshold, imgsz=960, verbose=False)
        annotated = results[0].plot()

        # Display info
        h, w = annotated.shape[:2]
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0

        # Add overlay text
        cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(annotated, f"Laser Detection (Webcam)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Detections: {num_detections} | Conf: {conf_threshold:.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show confidence for each detection
        if results[0].boxes is not None:
            y_offset = 80
            for i, box in enumerate(results[0].boxes):
                conf = float(box.conf[0])
                cv2.putText(annotated, f"Point {i + 1}: {conf:.3f}", (10, y_offset + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('Laser Detection - Webcam', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = cv2.getTickCount()
            filename = f"webcam_laser_{timestamp}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"💾 Saved: {filename}")
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(1.0, conf_threshold + 0.05)
            print(f"Confidence threshold: {conf_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            conf_threshold = max(0.01, conf_threshold - 0.05)
            print(f"Confidence threshold: {conf_threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    # Ask user for mode
    print("Laser Detection Test Tool")
    print("=" * 40)
    print("1. Test on images (default)")
    print("2. Test on webcam")

    # mode = input("Select mode (1/2): ").strip()
    mode = "1"  # Default to image testing

    if mode == "2":
        test_webcam()
    else:
        main()