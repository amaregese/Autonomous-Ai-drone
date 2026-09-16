import cv2
import modules.app_config
from modules import drone
from modules.display import set_hud_status


class TrackingSession:
    def __init__(self):
        self.mouse_click_x = -1
        self.mouse_click_y = -1
        self.right_click = False
        self.lost_shown = False
        self.last_target_name = None
        self._frame_w = 640
        self._frame_h = 480
        self._display_w = 960
        self._display_h = 720

    def set_frame_size(self, frame_w, frame_h, display_w, display_h):
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._display_w = display_w
        self._display_h = display_h

    def _to_frame_coords(self, display_x, display_y):
        fx = display_x * self._frame_w / self._display_w
        fy = display_y * self._frame_h / self._display_h
        return int(fx), int(fy)

    def handle_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_click_x, self.mouse_click_y = self._to_frame_coords(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_click = True

    def has_pending_click(self):
        return self.mouse_click_x != -1 and self.mouse_click_y != -1

    def clear_click(self):
        self.mouse_click_x = -1
        self.mouse_click_y = -1

    def has_right_click(self):
        return self.right_click

    def clear_right_click(self):
        self.right_click = False

    def find_object_at_click(self, detections):
        for obj in detections:
            if obj.Left <= self.mouse_click_x <= obj.Right and obj.Top <= self.mouse_click_y <= obj.Bottom:
                return obj
        return None

    def process_click(self, detections, detector, control):
        if self.has_right_click():
            self.clear_right_click()
            detector.clear_selection()
            control.set_visualizer_status("Selection cleared", (200, 200, 200), 1.0)
            self.clear_click()
            return True

        if not self.has_pending_click():
            return False

        clicked_obj = self.find_object_at_click(detections)
        previous_obj = detector.get_selected_object()

        if clicked_obj:
            missing = []
            if not drone.is_armed():
                missing.append("not armed")
            if drone.get_gps_fix_type() < 3:
                missing.append("no 3D GPS fix")
            if not drone.is_ekf_ok():
                missing.append("EKF unhealthy")
            try:
                rel_alt = drone.get_position()[2] - modules.app_config.HOME_ALT
                if rel_alt < modules.app_config.MIN_FOLLOW_ALT:
                    missing.append(
                        f"altitude {rel_alt:.1f}m below {modules.app_config.MIN_FOLLOW_ALT}m"
                    )
            except Exception:
                missing.append("cannot read altitude")
            if missing:
                reason = ", ".join(missing)
                control.set_visualizer_status(f"Cannot track: {reason}", (0, 0, 200), 2.0)
                set_hud_status(f"Cannot track: {reason}", (0, 0, 200), 2.5)
                print(f"[TRACK] Rejected selection — {reason}")
                self.clear_click()
                return False
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
                return True

        self.clear_click()
        return False

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
