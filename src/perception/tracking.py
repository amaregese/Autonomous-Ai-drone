import cv2


class TrackingSession:
    def __init__(self):
        self.mouse_click_x = -1
        self.mouse_click_y = -1
        self.lost_shown = False
        self.last_target_name = None

    def handle_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_click_x = x
            self.mouse_click_y = y

    def has_pending_click(self):
        return self.mouse_click_x != -1 and self.mouse_click_y != -1

    def clear_click(self):
        self.mouse_click_x = -1
        self.mouse_click_y = -1

    def find_object_at_click(self, detections):
        for obj in detections:
            if obj.Left <= self.mouse_click_x <= obj.Right and obj.Top <= self.mouse_click_y <= obj.Bottom:
                return obj
        return None

    def process_click(self, detections, detector, control):
        if not self.has_pending_click():
            return

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
