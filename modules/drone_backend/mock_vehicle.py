class MockVehicle:
    def __init__(self):
        self.location = type("obj", (object,), {"global_relative_frame": type("obj", (object,), {"alt": 0})})
        self.armed = False
        self.mode = type("obj", (object,), {"name": "GUIDED"})
        self.current_x = 0
        self.current_y = 0
        self.current_z = 2.5
        self.battery = 100
        self.home_lat = 14.5995
        self.home_lon = 120.9842

    def simple_takeoff(self, alt):
        print(f"Mock: Taking off to {alt}m")
        self.current_z = alt

    def is_armable(self):
        return True

    def arm(self, wait=True):
        print("Mock: Arming drone")
        self.armed = True

    def simple_goto(self, location):
        print("Mock: Going to location")

    def land(self):
        self.armed = False
        self.current_z = 0
        print("Mock: Landing")

    def close(self):
        print("Mock: Closing connection")

    def get_EKF_status(self):
        return "Mock: EKF status OK"

    def get_position(self):
        return self.current_x, self.current_y, self.current_z

    def get_battery_level(self):
        return self.battery

    def get_gps_position(self):
        return self.home_lat, self.home_lon, self.current_z
