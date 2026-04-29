import cv2


WINDOW_NAME = "Object Detection"


def draw_detection_window(image, detections, detector):
    selected_obj = detector.get_selected_object()

    for obj in detections:
        is_selected = obj is selected_obj
        frame_color = (0, 255, 255) if is_selected else (255, 0, 0)
        circle_color = (0, 255, 0) if is_selected else (0, 165, 255)
        thickness = 2 if is_selected else 1

        cv2.rectangle(image, (obj.Left, obj.Top), (obj.Right, obj.Bottom), frame_color, thickness)

        if is_selected:
            cv2.putText(
                image,
                f"{obj.class_name} ({obj.confidence:.0f}%)",
                (obj.Left, obj.Top - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                frame_color,
                2,
            )
            center = (image.shape[1] // 2, image.shape[0] // 2)
            cv2.line(image, center, obj.Center, circle_color, 1)
            cv2.circle(image, obj.Center, 7, circle_color, -1)
        else:
            cv2.putText(
                image,
                f"{obj.class_name}",
                (obj.Left, obj.Top - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                frame_color,
                1,
            )

    cv2.circle(image, (image.shape[1] // 2, image.shape[0] // 2), 6, (255, 255, 255), -1)
    return image


def annotate_tracking_overlay(image, fps, selected_obj, movement):
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(
        image,
        f"Following: {selected_obj.class_name}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        image,
        f"Dist: {movement['lidar_dist']:.1f}m | Speed: {movement['vel_z']:.1f}m/s",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        image,
        f"Size: {movement['normalized_size']:.1%}",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


def annotate_selection_overlay(image, detections):
    height, width = image.shape[:2]
    cv2.putText(
        image,
        "Click on any object to track",
        (width // 2 - 120, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        image,
        f"Objects detected: {len(detections)}",
        (width // 2 - 100, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )


def update_drone_visualizer_status(detector, control, lost_threshold):
    selected_obj = detector.get_selected_object()
    selected_class = detector.get_selected_class()
    tracking_lost = detector.get_tracking_status()
    tracking_conf = detector.get_tracking_confidence()

    if tracking_lost and selected_class:
        status = f"OBJECT LOST: {selected_class}"
        status_color = (0, 0, 255)
    elif selected_obj is None:
        status = "OBJECT NOT SELECTED"
        status_color = (100, 100, 100)
    elif tracking_conf < lost_threshold:
        status = f"OBJECT LOST: {selected_obj.class_name}"
        status_color = (0, 0, 255)
    else:
        status = f"TRACKING: {selected_obj.class_name} ({tracking_conf:.0f}%)"
        status_color = (0, 255, 0)

    control.set_visualizer_status(status, status_color, duration=0)
