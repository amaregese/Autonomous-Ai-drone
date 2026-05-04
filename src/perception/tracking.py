import cv2


class TrackingSession:
    def __init__(self):
        self.mouse_click_x = -1
        self.mouse_click_y = -1
        self.mouse_click_type = "single"
        self.last_click_time = 0
        self.double_click_threshold = 0.3
        self.lost_shown = False
        self.last_target_name = None

    def handle_mouse_event(self, event, x, y, flags, param):
        import time
        if event == cv2.EVENT_LBUTTONDOWN:
            current_time = time.time()
            if current_time - self.last_click_time < self.double_click_threshold:
                self.mouse_click_type = "double"
            else:
                self.mouse_click_type = "single"
            self.mouse_click_x = x
            self.mouse_click_y = y
            self.last_click_time = current_time

    def has_pending_click(self):
        return self.mouse_click_x != -1 and self.mouse_click_y != -1

    def clear_click(self):
        self.mouse_click_x = -1
        self.mouse_click_y = -1
        self.mouse_click_type = "single"

    def find_object_at_click(self, detections):
        for obj in detections:
            if obj.Left <= self.mouse_click_x <= obj.Right and obj.Top <= self.mouse_click_y <= obj.Bottom:
                return obj
        return None

    def find_innermost_object(self, detections):
        candidates = []
        for obj in detections:
            if obj.Left <= self.mouse_click_x <= obj.Right and obj.Top <= self.mouse_click_y <= obj.Bottom:
                candidates.append(obj)
        if not candidates:
            return None
        candidates.sort(key=lambda o: o.area)
        return candidates[0]

    def process_click(self, detections, detector, control):
        if not self.has_pending_click():
            return

        if self.mouse_click_type == "double":
            clicked_obj = self.find_innermost_object(detections)
        else:
            clicked_obj = self.find_object_at_click(detections)

        previous_obj = detector.get_selected_object()

        if clicked_obj:
            if previous_obj is not None:
                control.set_visualizer_status(
                    f"Switched: {previous_obj.class_name} -> {clicked_obj.class_name}",
                    (0, 255, 255),
                    1.5,
                )

            if detector.select_object(clicked_obj):
                control.set_visualizer_status(f"Tracking: {clicked_obj.class_name}", (0, 255, 0), 1.5)
                self.lost_shown = False
                self.last_target_name = clicked_obj.class_name

        self.clear_click()

    def reset_loss_state(self):
        self.lost_shown = False
        self.last_target_name = None

    def show_lost_status_once(self, target_name, tracking_confidence, threshold, control):
        if not target_name or tracking_confidence >= threshold or self.lost_shown:
            return

        self.lost_shown = True
        self.last_target_name = target_name
        print(f"\n⚠️ {target_name} LOST! (Score: {tracking_confidence:.0f}%)")
        control.set_visualizer_status(f"EMERGENCY: LOST {target_name.upper()}", (0, 0, 255), 2.4)
