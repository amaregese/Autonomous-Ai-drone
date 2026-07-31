import collections

from modules import app_config, lidar, vision
from modules.app_config import (
    FORWARD_BRAKE_ZONE,
    FORWARD_DEADBAND,
    GAIN_FORWARD,
    GAIN_YAW,
    LIDAR_BLEND_WEIGHT,
    MAX_FOLLOW_DIST,
    MAX_SPEED,
    MAX_YAW,
)


class FollowController:
    def __init__(self):
        self.ma_x = collections.deque(maxlen=5)
        self.ma_z = collections.deque(maxlen=7)
        self._fy = None

    def set_focal_length(self, fy: float):
        self._fy = fy

    def reset(self):
        self.ma_x.clear()
        self.ma_z.clear()

    def estimate_distance_from_size(self, bbox_height_px: float, class_name: str = "") -> float:
        if self._fy is None or self._fy <= 0 or bbox_height_px <= 0:
            return 3.0
        obj_h = app_config.OBJECT_HEIGHTS.get(class_name, app_config.OBJECT_HEIGHT)
        return (obj_h * self._fy) / bbox_height_px

    @staticmethod
    def blend_distance_estimate(vision_dist, lidar_dist, lidar_on_target):
        if not lidar_on_target or lidar_dist <= 0:
            return vision_dist
        blended = (LIDAR_BLEND_WEIGHT * lidar_dist) + ((1.0 - LIDAR_BLEND_WEIGHT) * vision_dist)
        return min(vision_dist, blended)

    @staticmethod
    def calculate_ma(values):
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def compute_forward_velocity(distance_error):
        if abs(distance_error) <= FORWARD_DEADBAND:
            return 0.0

        scaled_error = distance_error
        if abs(distance_error) < FORWARD_BRAKE_ZONE:
            scaled_error = (distance_error / FORWARD_BRAKE_ZONE) * abs(distance_error)

        vel_z = scaled_error * GAIN_FORWARD
        return max(-MAX_SPEED, min(MAX_SPEED, vel_z))

    def compute_follow_command(self, selected_obj, frame_shape):
        height, width = frame_shape[:2]
        center = selected_obj.Center
        x_delta = (center[0] - width / 2) / width
        y_delta = (center[1] - height / 2) / height

        lidar_on_target = vision.point_in_rectangle(
            (width / 2, height / 2),
            selected_obj.Left,
            selected_obj.Right,
            selected_obj.Top,
            selected_obj.Bottom,
        )

        object_width = selected_obj.Right - selected_obj.Left
        object_height = selected_obj.Bottom - selected_obj.Top
        vision_dist = self.estimate_distance_from_size(object_height, selected_obj.class_name)
        mock_lidar = lidar.read_lidar_distance()[0]
        fused_dist = self.blend_distance_estimate(vision_dist, mock_lidar, lidar_on_target)

        self.ma_z.append(fused_dist)
        self.ma_x.append(x_delta)

        z_ma = self.calculate_ma(self.ma_z)
        distance_error = z_ma - MAX_FOLLOW_DIST
        vel_z = self.compute_forward_velocity(distance_error)

        x_ma = self.calculate_ma(self.ma_x)
        yaw_cmd = max(-MAX_YAW, min(MAX_YAW, x_ma * MAX_YAW * GAIN_YAW))

        target_x = center[0] * 800 / width
        target_z = center[1] * 600 / height

        return {
            "x_delta": x_delta,
            "y_delta": y_delta,
            "lidar_on_target": lidar_on_target,
            "object_height": object_height,
            "object_width": object_width,
            "lidar_dist": z_ma,
            "vision_dist": vision_dist,
            "fused_dist": fused_dist,
            "mock_lidar": mock_lidar,
            "distance_error": distance_error,
            "vel_z": vel_z,
            "yaw_cmd": yaw_cmd,
            "z_ma": z_ma,
            "x_ma": x_ma,
        }
