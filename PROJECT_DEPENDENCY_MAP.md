# PROJECT_DEPENDENCY_MAP.md

## Production Dependency Structure

This map shows the actual production dependency chain starting from `autonomous_drone_main.py`.

---

## Entry Point

```
autonomous_drone_main.py
│
├── Standard Library
│   ├── sys, time, argparse, glob, math, os, shutil
│
├── Third-Party
│   ├── cv2 (OpenCV)
│   ├── numpy
│   ├── ultralytics (YOLO)
│   ├── torch (via ultralytics)
│
├── Configuration
│   └── modules.app_config
│
├── Core Modules (modules/)
│   ├── lidar → modules.lidar_backend.mock
│   ├── control → modules.control_system.api
│   ├── detector_yolo11 → modules.yolo11_detector.api
│   ├── drone → modules.drone_backend.api
│   ├── display → modules.display
│   ├── distance_estimator
│   │   ├── DEFAULT_CONFIGURED_INTRINSICS
│   │   ├── DistanceEstimator
│   │   ├── EstimatorConfig
│   │   ├── VisionConfig
│   │   ├── VisionDistanceEstimator
│   │   ├── annotate_detections
│   │   └── load_configured_intrinsics
│   ├── distance_estimator.calibration → CameraIntrinsics
│   ├── person_follow → PersonFollowController
│   ├── tracking → TrackingSession
│   └── auto_calibrate → AutoCalibrator
│
├── Shared
│   └── shared.detection_models → TelemetryData
│
├── Jetson Communication
│   ├── jetson.communication.sgc_receiver → SGCCommandReceiver, _find_best_match
│   └── jetson.communication.detection_sender → Streamer (via detection_sender)
│
└── Jetson Streaming
    └── jetson.streaming.rtsp_server → RTSPServer
```

---

## Detailed Module Dependencies

### autonomous_drone_main.py (Root)

```
autonomous_drone_main.py
│
├── modules.app_config
│   └── (constants only - no imports)
│
├── modules.lidar → modules.lidar_backend.mock
│   └── (mock implementation)
│
├── modules.control → modules.control_system.api
│   ├── modules.control_system.pid
│   │   └── modules.control_system.config
│   ├── modules.control_system.state
│   ├── modules.control_system.visualizer
│   │   └── modules.control_system.state
│   ├── modules.drone → modules.drone_backend.api
│   │   ├── modules.drone_backend.sitl (conditional)
│   │   │   └── pymavlink.mavutil
│   │   └── modules.drone_backend.mock_vehicle
│   └── modules.control_system.config
│
├── modules.detector_yolo11 → modules.yolo11_detector.api
│   ├── modules.yolo11_detector.config
│   ├── modules.yolo11_detector.model
│   │   └── ultralytics.YOLO
│   ├── modules.yolo11_detector.source
│   │   ├── cv2
│   │   └── modules.yolo11_detector.config
│   ├── modules.yolo11_detector.matching
│   │   ├── cv2
│   │   └── modules.yolo11_detector.config
│   ├── modules.yolo11_detector.types (Detection class)
│   └── ultralytics.YOLO
│
├── modules.drone → modules.drone_backend.api
│   ├── modules.drone_backend.sitl (mode: sitl/flight)
│   │   ├── pymavlink.mavutil
│   │   ├── threading, socket, subprocess
│   │   └── time, math
│   └── modules.drone_backend.mock_vehicle
│
├── modules.display
│   ├── cv2
│   ├── numpy
│   ├── math, time
│   └── (self-contained drawing functions)
│
├── modules.distance_estimator
│   ├── modules.distance_estimator.estimator
│   │   ├── modules.distance_estimator.calibration
│   │   ├── modules.distance_estimator.config
│   │   ├── modules.distance_estimator.depth
│   │   │   └── modules.distance_estimator.depth_backend
│   │   │       ├── modules.distance_estimator.depth_backend.metric_depth
│   │   │       └── modules.distance_estimator.depth_backend.synthetic
│   │   ├── modules.distance_estimator.filter
│   │   ├── modules.distance_estimator.models
│   │   └── modules.distance_estimator.lidar
│   │       └── modules.distance_estimator.lidar_backend (protocol)
│   ├── modules.distance_estimator.vision
│   │   ├── modules.distance_estimator.calibration
│   │   └── modules.distance_estimator.config
│   ├── modules.distance_estimator.calibration
│   │   └── (dataclasses, json, os)
│   ├── modules.distance_estimator.config (all sub-configs)
│   │   ├── VisionConfig
│   │   ├── LidarConfig
│   │   ├── DepthConfig
│   │   ├── FusionConfig
│   │   └── FilterConfig
│   └── modules.distance_estimator.calibration
│       └── CameraIntrinsics
│
├── modules.person_follow → PersonFollowController
│   ├── modules.app_config
│   ├── modules.distance_estimator.vision → estimate_detection
│   └── modules.yolo11_detector.types (Detection)
│
├── modules.tracking → TrackingSession
│   ├── cv2
│   └── (mouse event handling)
│
├── modules.auto_calibrate → AutoCalibrator
│   ├── cv2
│   ├── numpy
│   ├── modules.vision_utils.geometry
│   └── json, os
│
├── shared.detection_models → TelemetryData
│   └── (dataclasses only)
│
├── jetson.communication.sgc_receiver → SGCCommandReceiver
│   ├── socket, threading, json, struct
│   └── shared.detection_transport
│
├── jetson.communication.detection_sender → Streamer
│   ├── jetson.streaming.rtsp_server
│   │   ├── socket, threading, http.server, socketserver
│   │   ├── subprocess (gst-launch, ffmpeg)
│   │   ├── cv2
│   │   └── numpy
│   ├── shared.detection_transport → UDPTransport
│   │   ├── socket, threading, json, struct
│   │   └── shared.detection_models
│   ├── modules.detector_yolo11 → detector (for get_selected_object)
│   ├── modules.display → hud (for notifications)
│   └── shared.detection_models → FrameDetections, etc.
│
└── jetson.streaming.rtsp_server → RTSPServer
    ├── socket, threading, http.server, socketserver
    ├── subprocess (gst-launch, ffmpeg)
    ├── cv2
    └── numpy
```

---

### modules.yolo11_detector.api (Core Detection)

```
modules.yolo11_detector.api
│
├── modules.yolo11_detector.config
│   └── (runtime config constants)
│
├── modules.yolo11_detector.matching
│   ├── cv2
│   ├── numpy
│   ├── math, time
│   └── modules.yolo11_detector.config
│
├── modules.yolo11_detector.model
│   └── ultralytics.YOLO
│
├── modules.yolo11_detector.source
│   ├── cv2
│   ├── numpy
│   └── modules.yolo11_detector.config
│
├── modules.yolo11_detector.types (Detection)
│   └── (dataclass with time, math)
│
└── ultralytics.YOLO
    ├── torch
    ├── numpy
    └── (ultralytics internals)
```

---

### modules.distance_estimator.estimator (Core Estimator)

```
modules.distance_estimator.estimator
│
├── modules.distance_estimator.calibration → CameraIntrinsics
├── modules.distance_estimator.config → EstimatorConfig
│   ├── VisionConfig
│   ├── LidarConfig
│   ├── DepthConfig
│   ├── FusionConfig
│   └── FilterConfig
├── modules.distance_estimator.depth → Depth
│   ├── modules.distance_estimator.depth_backend
│   │   ├── modules.distance_estimator.depth_backend.metric_depth
│   │   │   ├── torch, torchvision.transforms
│   │   │   ├── transformers (optional - ZoeDepth)
│   │   │   └── modules.distance_estimator.calibration
│   │   └── modules.distance_estimator.depth_backend.synthetic
│   └── modules.distance_estimator.calibration
├── modules.distance_estimator.filter → DistanceVelocityFilter
│   └── (Kalman filter implementation)
├── modules.distance_estimator.models
│   └── (Measurement dataclasses: Vision, LiDAR, Depth, TargetState)
├── modules.distance_estimator.lidar → LiDARSource protocol
│   ├── modules.distance_estimator.lidar_backend
│   │   ├── FixedLidar
│   │   └── SimulatedLidar
│   └── (protocol only)
├── modules.distance_estimator.vision → VisionDistanceEstimator
│   ├── modules.distance_estimator.calibration
│   └── modules.distance_estimator.config → VisionConfig
└── time, math
```

---

### modules.person_follow (Task 9 Controller)

```
modules.person_follow
│
├── modules.app_config (all FOLLOW_* constants)
├── modules.distance_estimator.vision → estimate_detection
│   ├── modules.distance_estimator.calibration
│   └── modules.distance_estimator.config → VisionConfig
├── modules.yolo11_detector.types → Detection
├── math, time
└── dataclasses
```

---

### modules.drone_backend.sitl (SITL Backend)

```
modules.drone_backend.sitl
│
├── pymavlink.mavutil
├── threading, socket, subprocess
├── time, math
├── modules.drone_backend.mock_vehicle (fallback)
└── (MAVLink message handling, telemetry, commands)
```

---

### jetson.communication.detection_sender (Streamer)

```
jetson.communication.detection_sender
│
├── jetson.streaming.rtsp_server → RTSPServer
│   ├── cv2
│   ├── numpy
│   ├── socket, threading
│   ├── http.server, socketserver
│   ├── subprocess (gst-launch, ffmpeg)
│   └── _get_lan_ip (socket)
│
├── shared.detection_transport → UDPTransport
│   ├── socket, threading
│   ├── json, struct
│   └── shared.detection_models
│
├── modules.detector_yolo11 → detector
│   └── get_selected_object, get_tracking_confidence
│
├── modules.display → hud
│   └── (notification state)
│
├── shared.detection_models
│   ├── FrameDetections
│   ├── Detection
│   ├── BBox, Point
│   ├── TelemetryData
│   ├── TrackingData
│   ├── CameraIntrinsics
│   └── OverlayConfig
│
├── modules.yolo11_detector.types → Detection (for conversion)
├── time, math, logging
└── sys
```

---

### jetson.streaming.rtsp_server (RTSP Server)

```
jetson.streaming.rtsp_server
│
├── cv2
├── numpy
├── socket
├── threading
├── http.server, socketserver
├── subprocess (gst-launch-1.0, ffmpeg)
├── logging
└── time
```

---

### modules.display (HUD & Overlay)

```
modules.display
│
├── cv2
├── numpy
├── math, time
├── dataclasses
├── shutil (terminal size)
└── (all drawing functions self-contained)
```

---

## Dependency Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    autonomous_drone_main.py                     │
└─────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Detection   │      │  Distance Est.  │      │    Tracking     │
│  (YOLO11)     │      │  (Estimator)    │      │  (TrackingSession)
└───────┬───────┘      └────────┬────────┘      └────────┬────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   PersonFollow        │
                    │   (PersonFollowController)
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌─────────────────┐
│  MAVLink      │      │  UDP 9001     │      │    Display      │
│  (Drone Cmd)  │      │  (SGC Detections)    │    (HUD/Overlay)  │
└───────────────┘      └───────────────┘      └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌─────────────────┐
│  SITL/Mock    │      │  Streamer     │      │   OpenCV Window │
│  Backend      │      │  (RTSP/MJPEG) │      │                 │
└───────────────┘      └───────────────┘      └─────────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  RTSP Server  │
                        │  (GStreamer/  │
                        │   FFmpeg/MJPEG)    │
                        └───────────────┘
```

---

## Configuration Flow

```
modules/app_config.py
    │
    ├── FOLLOW_* constants → modules.person_follow.PersonFollowController
    ├── MAX_ALT, TRACKING_LOST_THRESHOLD → autonomous_drone_main
    ├── MIN_FOLLOW_* → autonomous_drone_main (preflight)
    ├── OBJECT_HEIGHT, OBJECT_HEIGHTS → distance_estimator.vision
    ├── LIDAR_BLEND_WEIGHT → modules.navigation (LEGACY)
    ├── GAIN_YAW, MAX_YAW → modules.person_follow (yaw control)
    ├── GAIN_FORWARD, FORWARD_DEADBAND, etc. → modules.navigation (LEGACY)
    └── DEFAULT_WIDTH, HEIGHT → yolo11_detector.config
```

---

## Model Loading Chain

```
autonomous_drone_main.py
    │
    ├── --model-path argument (default: YOLO/yolo11n.pt)
    │
    ▼
modules.detector_yolo11.api.initialize_detector()
    │
    ├── modules.yolo11_detector.model.load_model()
    │   │
    │   └── ultralytics.YOLO(model_path)
    │
    ├── modules.yolo11_detector.source.initialize_capture()
    │   │
    │   ├── cv2.VideoCapture (MSMF on Windows)
    │   └── frame normalization (cv2.flip horizontal)
    │
    ▼
modules.yolo11_detector.api.get_detections()
    │
    ├── read_frame()
    ├── detect_objects() → _predict_boxes()
    │   │
    │   ├── model.predict() (ultralytics)
    │   ├── _update_selected_object() (tracking)
    │   └── modules.yolo11_detector.matching (IoU + features)
    │
    ▼
Detection[] with distance annotations (via distance_estimator)
```

---

## UDP Communication Flow

### Detection Channel (9001)

```
autonomous_drone_main.main_loop()
    │
    ├── detector.get_detections() → Detection[]
    │
    ├── annotate_detections() → adds distance_m, distance_valid, etc.
    │
    ├── streamer.push(frame, detections, fps, movement, telemetry)
    │
    ▼
jetson.communication.detection_sender.Streamer.push()
    │
    ├── _convert_detection() → SharedDetection
    ├── apply_authoritative_target_distance()
    ├── _build_overlay() → OverlayConfig
    ├── TrackingData (if following)
    ├── FrameDetections assembly
    │
    ▼
shared.detection_transport.UDPTransport.send()
    │
    ├── JSON serialization (FrameDetections.to_dict())
    ├── Length-prefixed binary (struct.pack)
    │
    ▼
UDP socket → 127.0.0.1:9001 (or --sgc-host:--sgc-port)
```

### Command Channel (9002)

```
SGC (Ground Control)
    │
    ├── UDP packet (JSON command)
    │
    ▼
jetson.communication.sgc_receiver.SGCCommandReceiver
    │
    ├── _receive_loop() (thread)
    ├── json.loads()
    ├── SGCCommand parsing
    │
    ▼
autonomous_drone_main._handle_sgc_command()
    │
    ├── select_target → detector.select_object()
    ├── follow_start → _preflight_follow() + follow_controller.reset()
    ├── follow_stop → _following = False
    ├── deselect_target → detector.clear_selection()
    ├── takeoff → _handle_takeoff_button()
    └── servo → drone.send_servo()
```

---

## SITL Connection Flow

```
autonomous_drone_main._pick_connection()
    │
    ├── --drone_connection arg OR interactive menu
    │
    ▼
modules.control.connect_drone()
    │
    ├── modules.drone.set_backend(args.mode)  # 'sitl' or 'flight'
    │
    ▼
modules.drone_backend.api.connect_drone()
    │
    ├── modules.drone_backend.sitl.connect_drone()
    │   │
    │   ├── mavutil.mavlink_connection()
    │   ├── wait_heartbeat()
    │   ├── _message_listener() thread start
    │   ├── _set_mode("GUIDED")
    │   └── command acknowledgment handling
    │
    ▼
modules.drone_backend.api._vehicle = SitlVehicle
```

---

## Key Dependency Characteristics

| Characteristic | Detail |
|----------------|--------|
| **Circular Dependencies** | None detected |
| **Deepest Chain** | main → distance_estimator.estimator → depth → metric_depth → transformers (optional) |
| **Optional Dependencies** | transformers (ZoeDepth), gst-launch/ffmpeg (RTSP), pymavlink (SITL) |
| **Platform-Specific** | jetson/* (Linux), MSMF capture (Windows), GStreamer (Jetson) |
| **Mock Fallbacks** | mock_vehicle, mock LiDAR, synthetic depth |
| **Configuration-Driven** | app_config, yolo11_detector.config, distance_estimator.config |
| **Runtime Plugin Loading** | drone_backend (sitl/mock), depth_backend (metric/synthetic) |

---

## Test Dependency Isolation

```
tests/
├── test_*.py
│   ├── modules.* (production modules)
│   ├── unittest.mock (extensive mocking)
│   ├── numpy, cv2 (test fixtures)
│   └── NO jetson.* imports
│
└── tools/
    ├── modules.* (production modules)
    ├── jetson.* (some tools)
    └── NO autonomous_drone_main import
```

**Key Finding**: Tests and tools are properly isolated from the main application entry point. They import production modules directly but don't create circular dependencies.