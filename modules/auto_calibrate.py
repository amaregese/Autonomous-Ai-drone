"""
Auto-calibration: passively detects a chessboard in the live camera feed
and computes intrinsics when enough samples are collected.

Usage:
  Pass frames via `feed(frame)` each iteration. The thread detects chessboard
  corners in the background. When 15+ samples are collected, it computes
  calibration and saves to disk.

  On next startup, if a saved calibration exists, it loads automatically.
"""
import json
import os
import threading
import time

import cv2
import numpy as np

_CALIBRATION_FILE = "calibration.json"
_MIN_SAMPLES = 15
_MAX_SAMPLES = 40
_DETECT_EVERY_N_FRAMES = 3


class AutoCalibrator:
    def __init__(self, board_size=(9, 6), calibration_file=_CALIBRATION_FILE):
        self._board_w, self._board_h = board_size
        self._objp = np.zeros((self._board_w * self._board_h, 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:self._board_w, 0:self._board_h].T.reshape(-1, 2)

        self._obj_points = []
        self._img_points = []
        self._frame_shape = None
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._done = False
        self._intrinsics = None
        self._calibration_file = calibration_file
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self._load_saved()

    def _load_saved(self):
        if not os.path.exists(self._calibration_file):
            return
        try:
            with open(self._calibration_file, "r") as f:
                data = json.load(f)
            self._intrinsics = {
                "fx": data["fx"],
                "fy": data["fy"],
                "cx": data["cx"],
                "cy": data["cy"],
                "calib_w": data.get("frame_w", 0),
                "calib_h": data.get("frame_h", 0),
            }
            self._done = True
            print(f"[CALIB] Loaded saved calibration: fx={data['fx']:.1f} fy={data['fy']:.1f} "
                  f"cx={data['cx']:.1f} cy={data['cy']:.1f} @ {data.get('frame_w', '?')}x{data.get('frame_h', '?')} "
                  f"(RMS {data.get('rms_error', '?')})")
        except Exception:
            pass

    def feed(self, frame):
        if self._done:
            return

        self._frame_counter += 1
        if self._frame_counter % _DETECT_EVERY_N_FRAMES != 0:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (self._board_w, self._board_h), None)

        if not found:
            return

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)

        with self._lock:
            self._obj_points.append(self._objp.copy())
            self._img_points.append(corners2)
            self._frame_shape = gray.shape[::-1]
            count = len(self._obj_points)

        if count % 5 == 0:
            print(f"[CALIB] Captured {count}/{_MIN_SAMPLES} frames")

        if count >= _MIN_SAMPLES:
            self._calibrate()

    def _calibrate(self):
        with self._lock:
            obj_pts = list(self._obj_points)
            img_pts = list(self._img_points)
            shape = self._frame_shape

        print(f"[CALIB] Computing calibration from {len(obj_pts)} frames...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts, img_pts, shape, None, None
        )

        data = {
            "fx": float(mtx[0][0]),
            "fy": float(mtx[1][1]),
            "cx": float(mtx[0][2]),
            "cy": float(mtx[1][2]),
            "rms_error": float(ret),
            "frames": len(obj_pts),
            "frame_w": shape[0],
            "frame_h": shape[1],
            "timestamp": time.time(),
        }

        with open(self._calibration_file, "w") as f:
            json.dump(data, f, indent=2)

        self._intrinsics = {
            "fx": data["fx"],
            "fy": data["fy"],
            "cx": data["cx"],
            "cy": data["cy"],
            "calib_w": data["frame_w"],
            "calib_h": data["frame_h"],
        }
        self._done = True

        print(f"[CALIB] Done! RMS error: {ret:.4f}")
        print(f"[CALIB] fx={data['fx']:.1f} fy={data['fy']:.1f} cx={data['cx']:.1f} cy={data['cy']:.1f}")
        print(f"[CALIB] Saved to {self._calibration_file}")

    @property
    def done(self):
        return self._done

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def sample_count(self):
        with self._lock:
            return len(self._obj_points)
