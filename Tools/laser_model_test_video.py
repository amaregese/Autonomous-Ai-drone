import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import time
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Test laser detection on video")
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "runs/detect/laser_points_train/weights/best.pt"),
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "datasets/laser_red_only_v2/videos/red_laser.mp4"),
        help="Path to video file"
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
        "--output-video",
        type=str,
        default=str(PROJECT_ROOT / "test_results/laser_detection_output.mp4"),
        help="Path to save output video"
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip N frames between detections (0 = process every frame)"
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        default=True,
        help="Show FPS counter"
    )
    return parser.parse_args()


def process_video(args):
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Open video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        print(f"Looking for videos in: {PROJECT_ROOT / 'videos'}")
        # List available videos
        video_dir = PROJECT_ROOT / "videos"
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
            if videos:
                print(f"\nAvailable videos:")
                for v in videos:
                    print(f"  - {v.name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\n🎥 Video Info:")
    print(f"   File: {video_path.name}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames / fps:.2f} seconds")
    print(f"\n🎯 Detection Settings:")
    print(f"   Confidence threshold: {args.conf_threshold}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Skip frames: {args.skip_frames}")
    print(f"\n💾 Output video: {output_path}")
    print(f"\n🎮 Controls:")
    print("   'q' - Quit")
    print("   'p' - Pause/Resume")
    print("   's' - Save current frame as image")
    print("   '+' - Increase confidence threshold")
    print("   '-' - Decrease confidence threshold")
    print("   'r' - Reset to initial confidence\n")

    # Tracking variables
    frame_count = 0
    processed_frames = 0
    detections_history = deque(maxlen=60)  # Store last 60 seconds of detections
    paused = False
    current_conf = args.conf_threshold
    start_time = time.time()

    # Create window
    window_name = "Laser Detection - Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames if configured
            if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
                continue

            processed_frames += 1

            # Run inference
            results = model(frame, conf=current_conf, imgsz=args.imgsz, verbose=False)

            # Get detections
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf
                    })

            detections_history.append(len(detections))

            # Annotate frame
            annotated = results[0].plot()

            # Add overlay information
            h, w = annotated.shape[:2]

            # Semi-transparent overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            # Title
            cv2.putText(annotated, "LASER DETECTION SYSTEM", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Detection info
            cv2.putText(annotated, f"Laser Points: {len(detections)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Confidence: {current_conf:.2f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Frame progress
            progress = (frame_count / total_frames) * 100
            cv2.putText(annotated, f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # FPS counter
            if args.show_fps and processed_frames > 10:
                elapsed_time = time.time() - start_time
                fps_display = processed_frames / elapsed_time
                cv2.putText(annotated, f"FPS: {fps_display:.1f}", (w - 100, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw detection boxes with confidence scores
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                # Color based on confidence
                if conf > 0.7:
                    color = (0, 255, 0)  # Green - High confidence
                elif conf > 0.3:
                    color = (0, 255, 255)  # Yellow - Medium confidence
                else:
                    color = (0, 0, 255)  # Red - Low confidence

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated, (x1, y1 - 20), (x1 + label_size[0] + 5, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Add detection timeline
            timeline_height = 60
            timeline_y = h - timeline_height
            cv2.rectangle(annotated, (0, timeline_y), (w, h), (0, 0, 0), -1)

            # Draw detection history graph
            if len(detections_history) > 1:
                max_detections = max(detections_history) if max(detections_history) > 0 else 1
                bar_width = w / len(detections_history)
                for i, count in enumerate(detections_history):
                    bar_height = int((count / max_detections) * (timeline_height - 10))
                    x_pos = int(i * bar_width)
                    y_pos = timeline_y + (timeline_height - 10) - bar_height
                    color = (0, 255, 0) if count > 0 else (50, 50, 50)
                    cv2.rectangle(annotated, (x_pos, y_pos),
                                  (int(x_pos + bar_width - 1), timeline_y + timeline_height - 10),
                                  color, -1)

            cv2.putText(annotated, "Detection History (last 60 frames)", (10, timeline_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Write to output video
            out.write(annotated)

            # Display
            cv2.imshow(window_name, annotated)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n⚠️ User interrupted playback")
            break
        elif key == ord('p'):
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"\n⏸️ {status}")
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            frame_filename = f"laser_frame_{frame_count}_{timestamp}.jpg"
            cv2.imwrite(frame_filename, annotated)
            print(f"💾 Saved frame: {frame_filename}")
        elif key == ord('+') or key == ord('='):
            current_conf = min(1.0, current_conf + 0.05)
            print(f"🔽 Confidence threshold: {current_conf:.2f}")
        elif key == ord('-') or key == ord('_'):
            current_conf = max(0.01, current_conf - 0.05)
            print(f"🔼 Confidence threshold: {current_conf:.2f}")
        elif key == ord('r'):
            current_conf = args.conf_threshold
            print(f"🔄 Reset confidence to: {current_conf:.2f}")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Final statistics
    print("\n" + "=" * 50)
    print("📊 VIDEO PROCESSING SUMMARY")
    print("=" * 50)
    print(f"✅ Total frames processed: {processed_frames}")
    print(f"✅ Total frames in video: {frame_count}")
    print(f"✅ Processing ratio: {processed_frames / frame_count * 100:.1f}%")

    if detections_history:
        total_detections = sum(detections_history)
        frames_with_detections = sum(1 for d in detections_history if d > 0)
        print(f"✅ Total laser detections: {total_detections}")
        print(f"✅ Frames with detections: {frames_with_detections}")
        print(f"✅ Detection rate: {frames_with_detections / len(detections_history) * 100:.1f}%")
        print(f"✅ Average detections per frame: {total_detections / len(detections_history):.2f}")

    print(f"\n💾 Output saved to: {output_path}")
    print(f"📁 Output video size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")


def main():
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()