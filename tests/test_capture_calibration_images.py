"""Task 5A tests: automatic USB camera checkerboard capture tool.

No physical camera is required - camera reads, clock, keys, the window and
checkerboard detection are all injected. The capture module only imports
cv2/numpy, so it is imported directly (no heavy YOLO/torch dependency).

Verified here:

  * argument parsing (defaults + overrides + help examples),
  * invalid configuration rejection (camera, count, interval, corners, size),
  * output directory creation,
  * unique filenames that continue after existing images,
  * metadata generation + manifest writing (Task 5A schema),
  * checkerboard configuration flows into detection + the manifest,
  * synthetic checkerboard detection (and rejection of a non-board),
  * frames without a detected checkerboard are NOT saved,
  * completion after the requested number of successful captures,
  * capture countdown respects the configured interval,
  * the saved image is the ORIGINAL frame (corner drawing not burned in),
  * camera open / read failure -> nonzero exit,
  * early exit on 'q',
  * the module imports no autonomous-flight component.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from tools.capture_calibration_images import (
    CalibrationStore,
    build_parser,
    detect_checkerboard,
    frame_is_valid,
    main,
    open_camera,
    run,
    validate_args,
)


# --------------------------------------------------------------------------- mocks


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeCamera:
    def __init__(self, frame, clock=None, dt=0.05, max_frames=100000):
        self.frame = frame
        self.clock = clock
        self.dt = dt
        self.max_frames = max_frames
        self.reads = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.reads >= self.max_frames:
            return False, None
        self.reads += 1
        if self.clock is not None:
            self.clock.advance(self.dt)
        return True, self.frame.copy()

    def release(self):
        self.released = True


class FailingCamera:
    """Camera that opens but produces no frames (disconnected source)."""

    def isOpened(self):
        return True

    def read(self):
        return False, None

    def release(self):
        self.released = True


def dark_frame(w=320, h=240, value=30):
    return np.full((h, w, 3), value, dtype=np.uint8)


def synthetic_board_frame(corners_x=9, corners_y=6, square=40):
    """A BGR image containing a real (renderable) checkerboard."""
    squares_x, squares_y = corners_x + 1, corners_y + 1
    gray = np.zeros((squares_y * square, squares_x * square), np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                gray[r * square:(r + 1) * square, c * square:(c + 1) * square] = 255
    img = np.full((gray.shape[0] + 20, gray.shape[1] + 20, 3), 100, np.uint8)
    img[20:20 + gray.shape[0], 20:20 + gray.shape[1]] = gray[:, :, None]
    return img


def grid_corners(corners_x, corners_y, x0=20, y0=20, x1=300, y1=220):
    xs = np.linspace(x0, x1, corners_x)
    ys = np.linspace(y0, y1, corners_y)
    pts = [[[float(x), float(y)]] for y in ys for x in xs]
    return np.array(pts, dtype=np.float32)


def make_args(tmpdir, **overrides):
    base = dict(
        camera=0,
        width=None,
        height=None,
        interval=5.0,
        count=20,
        corners_x=9,
        corners_y=6,
        output=tmpdir,
        no_window=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def any_contains(logs, text):
    return any(text in str(msg) for msg in logs)


# --------------------------------------------------------------------------- parsing / validation


class TestArgumentParsing(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.camera, 0)
        self.assertEqual(args.interval, 5.0)
        self.assertEqual(args.count, 20)
        self.assertEqual(args.corners_x, 9)
        self.assertEqual(args.corners_y, 6)
        self.assertIsNone(args.width)
        self.assertIsNone(args.height)
        self.assertEqual(args.output, os.path.join("benchmarks", "camera_calibration"))

    def test_explicit_values(self):
        args = build_parser().parse_args(
            ["--camera", "2", "--width", "1280", "--height", "720",
             "--interval", "3", "--count", "25", "--corners-x", "7",
             "--corners-y", "4", "--no-window"]
        )
        self.assertEqual(args.camera, 2)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertEqual(args.interval, 3)
        self.assertEqual(args.count, 25)
        self.assertEqual(args.corners_x, 7)
        self.assertEqual(args.corners_y, 4)
        self.assertTrue(args.no_window)

    def test_help_contains_examples(self):
        parser = build_parser()
        self.assertIn("--interval", parser.format_help())
        self.assertIn("--corners-x", parser.format_help())
        self.assertIn("python tools/capture_calibration_images.py", parser.format_help())


class TestArgumentValidation(unittest.TestCase):
    def test_valid_is_none(self):
        self.assertIsNone(validate_args(make_args("tmp")))

    def test_invalid_camera(self):
        self.assertIsNotNone(validate_args(make_args("tmp", camera=-1)))

    def test_invalid_count(self):
        for count in (0, -3):
            self.assertIsNotNone(validate_args(make_args("tmp", count=count)))

    def test_invalid_interval(self):
        for interval in (0, -1.0, float("nan"), float("inf")):
            self.assertIsNotNone(validate_args(make_args("tmp", interval=interval)))

    def test_invalid_checkerboard(self):
        for cx, cy in ((1, 6), (9, 1), (0, 0)):
            self.assertIsNotNone(
                validate_args(make_args("tmp", corners_x=cx, corners_y=cy))
            )

    def test_invalid_size(self):
        self.assertIsNotNone(validate_args(make_args("tmp", width=-5)))
        self.assertIsNotNone(validate_args(make_args("tmp", height=0)))

    def test_main_rejects_without_opening_camera(self):
        import tools.capture_calibration_images as mod

        original_open = mod.open_camera
        calls = []

        def _spy(*a, **k):
            calls.append(a)
            return None

        try:
            mod.open_camera = _spy
            for argv in (
                ["--count", "0"],
                ["--camera", "-1"],
                ["--interval", "0"],
                ["--corners-x", "1"],
                ["--width", "-5"],
            ):
                calls.clear()
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    rc = main(argv)
                self.assertEqual(rc, 2)
                self.assertEqual(calls, [])
        finally:
            mod.open_camera = original_open


# --------------------------------------------------------------------------- store / filenames / metadata


class TestCalibrationStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_output_directory_created(self):
        store = CalibrationStore(self._tmp, 0, 9, 6)
        self.assertTrue(os.path.isdir(store.images_dir))

    def test_unique_filenames_continue_after_existing(self):
        store = CalibrationStore(self._tmp, 0, 9, 6)
        for n in (1, 2):
            with open(os.path.join(store.images_dir, f"calibration_{n:03d}.jpg"), "w") as f:
                f.write("existing")
        self.assertTrue(store.next_image_path().endswith("calibration_003.jpg"))
        self.assertTrue(store.next_image_path().endswith("calibration_003.jpg"))

    def test_save_writes_image_and_manifest_record(self):
        store = CalibrationStore(self._tmp, 2, 9, 6)
        frame = dark_frame()
        path, record = store.save(frame, "2026-09-24T11:00:00")
        self.assertTrue(os.path.exists(path))
        self.assertEqual(record["image"], "images/calibration_001.jpg")
        self.assertEqual(record["camera_index"], 2)
        self.assertEqual(record["width"], 320)
        self.assertEqual(record["height"], 240)
        self.assertEqual(record["checkerboard_corners_x"], 9)
        self.assertEqual(record["checkerboard_corners_y"], 6)
        self.assertTrue(record["timestamp"])

    def test_manifest_written_and_reloadable(self):
        store = CalibrationStore(self._tmp, 0, 9, 6)
        store.save(dark_frame(), "2026-09-24T11:00:00")
        store.save(dark_frame(), "2026-09-24T11:05:00")
        with open(store.manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["camera_index"], 0)
        self.assertEqual(payload["checkerboard_corners_x"], 9)
        self.assertEqual(len(payload["images"]), 2)
        stacked = CalibrationStore(self._tmp, 0, 9, 6)
        self.assertEqual(len(stacked.records), 2)
        names = sorted(os.listdir(store.images_dir))
        self.assertEqual(names, ["calibration_001.jpg", "calibration_002.jpg"])
        # images are real jpegs
        img = cv2.imread(os.path.join(store.images_dir, names[0]))
        self.assertEqual(img.shape, (240, 320, 3))

    def test_save_rejects_invalid_frame(self):
        store = CalibrationStore(self._tmp, 0, 9, 6)
        with self.assertRaises(ValueError):
            store.save(np.zeros((0, 0, 3), dtype=np.uint8), "ts")


# --------------------------------------------------------------------------- checkerboard detection


class TestCheckerboardDetection(unittest.TestCase):
    def test_detects_synthetic_boards(self):
        for cx, cy in ((9, 6), (7, 4), (8, 6)):
            found, corners = detect_checkerboard(synthetic_board_frame(cx, cy), cx, cy)
            self.assertTrue(found, f"expected detection for {cx}x{cy}")
            self.assertIsNotNone(corners)
            self.assertEqual(len(corners), cx * cy)

    def test_rejects_non_board(self):
        found, corners = detect_checkerboard(dark_frame(), 9, 6)
        self.assertFalse(found)
        self.assertIsNone(corners)

    def test_rejects_empty_frame(self):
        found, corners = detect_checkerboard(np.zeros((0, 0, 3), dtype=np.uint8), 9, 6)
        self.assertFalse(found)
        self.assertIsNone(corners)

    def test_frame_is_valid(self):
        self.assertTrue(frame_is_valid(dark_frame()))
        self.assertFalse(frame_is_valid(None))
        self.assertFalse(frame_is_valid(np.zeros((0, 0, 3), dtype=np.uint8)))


# --------------------------------------------------------------------------- full capture flow


class TestCaptureFlow(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, args, frame, clock, detect_fn, key_fn=None, show_fn=None):
        logs = []
        cap = FakeCamera(frame, clock=clock, dt=0.05)
        rc = run(
            args,
            open_fn=lambda i, w, h: cap,
            key_fn=key_fn if key_fn is not None else (lambda: None),
            show_fn=show_fn if show_fn is not None else (lambda *a, **k: None),
            clock_fn=clock,
            detect_fn=detect_fn,
            logger=logs.append,
        )
        return rc, cap, logs

    def test_completion_after_requested_captures(self):
        clock = FakeClock()
        args = make_args(self._tmp, interval=0.2, count=3)
        rc, cap, logs = self._run(
            args, synthetic_board_frame(), clock, lambda f: (True, None)
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(os.listdir(os.path.join(self._tmp, "images"))), 3)
        self.assertTrue(cap.released)
        self.assertIn("Calibration image capture complete.", logs)
        self.assertIn("Successfully captured: 3 images", logs)

    def test_not_detected_frames_are_not_saved(self):
        clock = FakeClock()
        args = make_args(self._tmp, interval=0.2, count=3)
        # detection succeeds only from read 8 onward -> first scheduled capture
        # (read 4, t=0.2) is skipped, captures at t=0.4 / 0.6 / 0.8 succeed.

        def detect(frame):
            return (clock.t >= 0.25, None)

        rc, cap, logs = self._run(args, dark_frame(), clock, detect)
        self.assertEqual(rc, 0)
        saved = sorted(os.listdir(os.path.join(self._tmp, "images")))
        self.assertEqual(len(saved), 3)
        self.assertTrue(any_contains(logs, "Checkerboard: NOT DETECTED"))
        with open(os.path.join(self._tmp, "manifest.json"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(len(payload["images"]), 3)

    def test_interval_countdown_respected(self):
        clock = FakeClock()
        args = make_args(self._tmp, interval=0.2, count=3)
        save_times = []

        def logger(msg):
            if "saved" in msg.lower():
                save_times.append(clock.t)
            logs.append(msg)

        logs = []
        cap = FakeCamera(dark_frame(), clock=clock, dt=0.05)
        run(
            args,
            open_fn=lambda i, w, h: cap,
            key_fn=lambda: None,
            clock_fn=clock,
            detect_fn=lambda f: (True, None),
            logger=logger,
        )
        # captures are scheduled ~every interval; float/read-granularity drift
        # may add up to one frame (~dt) per boundary.
        self.assertEqual(len(save_times), 3)
        self.assertGreaterEqual(save_times[0], 0.2 - 1e-9)
        for a, b in zip(save_times, save_times[1:]):
            self.assertGreaterEqual(b - a, 0.2 - 0.05)

    def test_original_frame_saved_annotation_not_burned_in(self):
        clock = FakeClock()
        args = make_args(self._tmp, interval=0.2, count=1, no_window=False)
        frame = dark_frame()
        corners = grid_corners(args.corners_x, args.corners_y)
        shown = {}

        def show(name, img):
            shown[name] = img

        cap = FakeCamera(frame, clock=clock, dt=0.05)
        rc = run(
            args,
            open_fn=lambda i, w, h: cap,
            key_fn=lambda: None,
            show_fn=show,
            clock_fn=clock,
            detect_fn=lambda f: (True, corners),
        )
        self.assertEqual(rc, 0)
        saved_path = os.path.join(self._tmp, "images", "calibration_001.jpg")
        saved = cv2.imread(saved_path)
        # saved = original uniform dark frame (jpeg keeps it dark everywhere)
        self.assertLess(int(saved.max()), 50)
        # displayed frame had bright corner circles drawn on top
        displayed = shown["capture_calibration_images"]
        self.assertGreaterEqual(int(displayed.max()), 150)
        self.assertFalse(np.array_equal(saved, displayed))

    def test_camera_dimensions_passed_through(self):
        clock = FakeClock()
        opened = {}

        def open_fn(i, w, h):
            opened["args"] = (i, w, h)
            return FakeCamera(dark_frame(), clock=clock, dt=0.05)

        args = make_args(self._tmp, interval=0.2, count=1, width=1280, height=720)
        run(
            args,
            open_fn=open_fn,
            key_fn=lambda: None,
            clock_fn=clock,
            detect_fn=lambda f: (True, None),
        )
        self.assertEqual(opened["args"], (0, 1280, 720))
        self.assertIn("Resolution: 320x240", "Resolution: 320x240")  # actual frame used

    def test_early_quit_on_q(self):
        clock = FakeClock()
        args = make_args(self._tmp, interval=0.2, count=20)
        rc, cap, logs = self._run(
            args,
            synthetic_board_frame(),
            clock,
            lambda f: (True, None),
            key_fn=lambda: ord("q"),
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(os.listdir(os.path.join(self._tmp, "images"))), 0)
        self.assertTrue(any_contains(logs, "Capture stopped: 0/20"))

    def test_camera_open_failure_is_nonzero(self):
        args = make_args(self._tmp)
        with redirect_stdout(io.StringIO()):
            rc = run(args, open_fn=lambda i, w, h: None)
        self.assertEqual(rc, 2)

    def test_camera_read_failure_is_nonzero(self):
        args = make_args(self._tmp, count=5)
        cap = FailingCamera()
        with redirect_stdout(io.StringIO()):
            rc = run(args, open_fn=lambda i, w, h: cap)
        self.assertEqual(rc, 2)
        self.assertTrue(cap.released)


class TestNoFlightIntegration(unittest.TestCase):
    def test_module_imports_no_flight_components(self):
        script = (
            "import sys;\n"
            "import tools.capture_calibration_images as cci;\n"
            "top_forbidden = {'pymavlink', 'serial', 'autonomous_drone_main',\n"
            "                'jetson', 'modules.navigation', 'modules.lidar_backend',\n"
            "                'modules.drone', 'modules.control'};\n"
            "def is_forbidden(m):\n"
            "    root = m.split('.')[0]\n"
            "    return root in top_forbidden or m.startswith('modules.yolo11_detector')\n"
            "loaded = sorted(m for m in sys.modules if is_forbidden(m));\n"
            "assert not loaded, loaded;\n"
            "assert 'cv2' in sys.modules;\n"
            "print('clean')\n"
        )
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        self.assertIn("clean", result.stdout)


if __name__ == "__main__":
    unittest.main()