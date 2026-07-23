import math
import time

import cv2
import numpy as np


class DroneVisualizer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.window_name = "Drone Position Tracker"
        self.drone_x = width // 2
        self.drone_z = height - 100
        self.target_x = width // 2
        self.target_z = height - 50
        self.target_name = "No Target"
        self.target_confidence = 0
        self.target_distance = 0
        self.has_target = False
        self.movement_history = []
        self.total_distance = 0
        self.speed = 5
        self.fps = 0
        self.yaw = 0
        self.forward = 0
        self.lidar_on_target = False
        self.x_delta = 0
        self.y_delta = 0
        self.status_message = "OBJECT NOT SELECTED"
        self.status_color = (100, 100, 100)
        self.status_duration = 0
        self.temp_message = ""
        self.temp_color = (0, 255, 0)
        self.temp_start_time = 0
        self.temp_duration = 0
        self.COLOR_DRONE = (255, 100, 0)
        self.COLOR_TARGET = (0, 255, 0)
        self.COLOR_DISTANCE = (255, 255, 0)
        self.COLOR_GRID = (50, 50, 50)
        cv2.namedWindow(self.window_name)

    def set_status(self, message, color=(0, 255, 0), duration=0):
        if duration > 0:
            self.temp_message = message
            self.temp_color = color
            self.temp_start_time = time.time()
            self.temp_duration = duration
        else:
            self.status_message = message
            self.status_color = color

    def update(self, movement_x, movement_z):
        self.drone_x += movement_x * self.speed
        self.drone_z += movement_z * self.speed
        self.drone_x = np.clip(self.drone_x, 50, self.width - 50)
        self.drone_z = np.clip(self.drone_z, 50, self.height - 50)
        self.movement_history.append((self.drone_x, self.drone_z))
        if len(self.movement_history) > 50:
            self.movement_history.pop(0)

    def set_target(self, x, z, name="Target", confidence=0, distance=0):
        self.target_x = np.clip(x, 50, self.width - 50)
        self.target_z = np.clip(z, 50, self.height - 50)
        self.target_name = name
        self.target_confidence = confidence
        self.target_distance = distance
        self.has_target = bool(name and name != "No Target" and confidence > 0)

    def update_telemetry(self, fps=0, yaw=0, forward=0, lidar_on_target=False, x_delta=0, y_delta=0):
        self.fps = fps
        self.yaw = yaw
        self.forward = forward
        self.lidar_on_target = lidar_on_target
        self.x_delta = x_delta
        self.y_delta = y_delta

    def get_distance_to_target(self):
        if not self.has_target:
            return 0
        dx = self.drone_x - self.target_x
        dz = self.drone_z - self.target_z
        return math.sqrt(dx ** 2 + dz ** 2)

    def draw_temp_message(self, img):
        if "LOST" in self.status_message.upper():
            elapsed = time.time()
            blink_on = int(elapsed / 0.18) % 2 == 0
            if not blink_on:
                return

            danger_color = (0, 0, 255)
            bg_color = (0, 0, 60)
            text = self.status_message
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (self.width - text_size[0]) // 2
            text_y = 50
            cv2.rectangle(img, (text_x - 18, text_y - 36), (text_x + text_size[0] + 18, text_y + 16), bg_color, -1)
            cv2.rectangle(img, (text_x - 18, text_y - 36), (text_x + text_size[0] + 18, text_y + 16), danger_color, 3)
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, danger_color, 2)
            return

        elapsed = time.time() - self.temp_start_time
        if self.temp_message and elapsed < self.temp_duration:
            blink_on = int(elapsed / 0.18) % 2 == 0
            if not blink_on:
                return

            fade_ratio = max(0.25, 1 - (elapsed / self.temp_duration))
            text_color = tuple(int(channel * fade_ratio) for channel in self.temp_color)
            bg_color = tuple(int(channel * 0.25) for channel in self.temp_color)
            text_size = cv2.getTextSize(self.temp_message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (self.width - text_size[0]) // 2
            text_y = 50
            cv2.rectangle(img, (text_x - 14, text_y - 34), (text_x + text_size[0] + 14, text_y + 14), bg_color, -1)
            cv2.rectangle(img, (text_x - 14, text_y - 34), (text_x + text_size[0] + 14, text_y + 14), text_color, 2)
            cv2.putText(img, self.temp_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    def draw(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(0, self.width, 50):
            cv2.line(img, (i, 0), (i, self.height), self.COLOR_GRID, 1)
        for i in range(0, self.height, 50):
            cv2.line(img, (0, i), (self.width, i), self.COLOR_GRID, 1)
        for i in range(1, len(self.movement_history)):
            cv2.line(
                img,
                (int(self.movement_history[i - 1][0]), int(self.movement_history[i - 1][1])),
                (int(self.movement_history[i][0]), int(self.movement_history[i][1])),
                (100, 100, 255),
                2,
            )

        distance = self.get_distance_to_target()
        if self.has_target:
            cv2.line(img, (int(self.drone_x), int(self.drone_z)), (int(self.target_x), int(self.target_z)), self.COLOR_DISTANCE, 2)

        if self.has_target:
            confidence_color = (0, int(255 * self.target_confidence / 100), 255)
            cv2.circle(img, (int(self.target_x), int(self.target_z)), 25, confidence_color, 3)
            cv2.circle(img, (int(self.target_x), int(self.target_z)), 15, confidence_color, -1)
            cv2.putText(img, self.target_name, (int(self.target_x) - 30, int(self.target_z) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        drone_pts = np.array(
            [
                [int(self.drone_x - 15), int(self.drone_z - 10)],
                [int(self.drone_x + 15), int(self.drone_z - 10)],
                [int(self.drone_x + 20), int(self.drone_z)],
                [int(self.drone_x + 15), int(self.drone_z + 10)],
                [int(self.drone_x - 15), int(self.drone_z + 10)],
                [int(self.drone_x - 20), int(self.drone_z)],
            ],
            np.int32,
        )
        cv2.fillPoly(img, [drone_pts], self.COLOR_DRONE)

        rotor_angle = int(time.time() * 20) % 360
        rotor_radius = 12
        for angle in range(0, 360, 90):
            rad = math.radians(angle + rotor_angle)
            x = self.drone_x + math.cos(rad) * rotor_radius
            y = self.drone_z + math.sin(rad) * rotor_radius
            cv2.circle(img, (int(x), int(y)), 4, (200, 200, 200), -1)

        y = 40
        cv2.putText(img, f"fps: {self.fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"yaw: {self.yaw:.2f}", (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"forward: {self.forward:.2f}", (10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        status_color = (0, 255, 0) if self.lidar_on_target else (0, 0, 255)
        cv2.putText(img, f"lidar_on_target: {self.lidar_on_target}", (10, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        cv2.putText(img, f"x_delta: {self.x_delta:.3f}", (10, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"y_delta: {self.y_delta:.3f}", (10, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        rx = self.width - 200
        if self.has_target:
            cv2.putText(img, f"{self.target_name}: {self.target_confidence:.1f}%", (rx, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TARGET, 2)
        else:
            cv2.putText(img, "No Target Selected", (rx, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        dist_percent = min(100, (distance / 400) * 100) if self.has_target else 0
        distance_color = self.COLOR_DISTANCE if self.has_target else (150, 150, 150)
        cv2.putText(img, f"distance: {dist_percent:.1f}%", (rx, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 1)
        bar_width = 150
        fill = int((dist_percent / 100) * bar_width)
        cv2.rectangle(img, (rx, 115), (rx + bar_width, 130), (100, 100, 100), -1)
        if fill > 0:
            cv2.rectangle(img, (rx, 115), (rx + fill, 130), self.COLOR_DISTANCE, -1)

        bottom_y = self.height - 70
        cv2.putText(img, self.status_message, (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.status_color, 2)
        cv2.putText(img, "Click on any object to track | Click another to switch | Press 'q' to quit", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Drone: ({self.drone_x:.0f}, {self.drone_z:.0f})", (self.width - 200, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        target_label = f"Target: ({self.target_x:.0f}, {self.target_z:.0f})" if self.has_target else "Target: --"
        pixel_dist_label = f"Dist: {distance:.0f} px" if self.has_target else "Dist: 0 px"
        cv2.putText(img, target_label, (self.width - 200, bottom_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, pixel_dist_label, (self.width - 200, bottom_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, distance_color, 1)
        self.draw_temp_message(img)
        cv2.imshow(self.window_name, img)

    def should_quit(self):
        return cv2.waitKey(1) & 0xFF == ord("q")

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
