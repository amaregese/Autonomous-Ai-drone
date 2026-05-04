import random

last_object_area = None


def connect_lidar(port):
    print(f"Mock: Connecting to lidar on {port} (Ubuntu mode)")


def read_lidar_distance():
    distance = random.uniform(1.0, 5.0)
    print(f"Mock: Lidar distance = {distance:.2f}m")
    return distance, 0


def update_distance_from_object_size(object_area, frame_area):
    global last_object_area

    if object_area is None or object_area == 0:
        return 3.0

    normalized_size = object_area / frame_area
    min_distance = 1.0
    max_distance = 5.0
    min_size = 0.02
    max_size = 0.3

    size = max(min_size, min(max_size, normalized_size))
    distance = max_distance - (size - min_size) / (max_size - min_size) * (max_distance - min_distance)
    distance = max(min_distance, min(max_distance, distance))

    last_object_area = object_area
    return distance

