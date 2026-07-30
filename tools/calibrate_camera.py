"""
Camera intrinsic calibration using a chessboard pattern.

Usage:
  python tools/calibrate_camera.py [--camera 0] [--board 9x6]

Run once per camera. Press 'c' to capture a frame when the chessboard is visible.
Press 'q' when you have enough captures (10+ recommended).

The printed fx/fy/cx/cy values are the intrinsics to pass to autonomous_drone_main.py:
  python autonomous_drone_main.py --intrinsics-fx <fx> --intrinsics-fy <fy> --intrinsics-cx <cx> --intrinsics-cy <cy>
"""
import argparse
import json
import sys

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--board", type=str, default="9x6", help="Inner corners WxH (default: 9x6)")
    parser.add_argument("--save", type=str, default=None, help="Save intrinsics to JSON file")
    args = parser.parse_args()

    w, h = map(int, args.board.lower().split("x"))
    CHECKERBOARD = (w, h)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []
    frame_shape = None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        sys.exit(1)

    print(f"Camera {args.camera} opened")
    print(f"Chessboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners")
    print("Press 'c' to capture a frame with the chessboard visible")
    print("Press 'q' to finish and compute calibration")
    print(f"Captures: 0/{10}", end="\r")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_shape = gray.shape[::-1]

        display = frame.copy()
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

        info = f"Captures: {len(obj_points)} | Press 'c' to capture, 'q' to finish"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Calibrate", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners2)
                print(f"Frame {len(obj_points)} captured")
            else:
                print("Chessboard not detected in this frame")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 5:
        print(f"Error: Need at least 5 captures, got {len(obj_points)}")
        sys.exit(1)

    print(f"\nComputing calibration from {len(obj_points)} frames...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, frame_shape, None, None
    )

    fx = mtx[0][0]
    fy = mtx[1][1]
    cx = mtx[0][2]
    cy = mtx[1][2]

    print(f"\nCalibration result (RMS error: {ret:.4f}):")
    print(f"  fx = {fx:.1f}")
    print(f"  fy = {fy:.1f}")
    print(f"  cx = {cx:.1f}")
    print(f"  cy = {cy:.1f}")
    print(f"\nRun with:")
    print(f"  python autonomous_drone_main.py --intrinsics-fx {fx:.1f} --intrinsics-fy {fy:.1f} --intrinsics-cx {cx:.1f} --intrinsics-cy {cy:.1f}")

    if args.save:
        data = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "rms_error": ret, "frames": len(obj_points)}
        with open(args.save, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
