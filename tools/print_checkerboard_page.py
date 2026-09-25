"""Print a checkerboard calibration board (no calibration performed).

Renders a PNG sized for 300 dpi A4/Letter printing with the given internal
corner count and square size. Usage:

    python tools/print_checkerboard_page.py                # 9x6, 24 mm squares
    python tools/print_checkerboard_page.py --corners-x 9 --corners-y 6 --square-mm 24
    python tools/print_checkerboard_page.py --output checkerboard_9x6.png
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

DEFAULT_OUTPUT = os.path.join("benchmarks", "camera_calibration", "checkerboard_9x6.png")
DPI = 300
PIXELS_PER_MM = DPI / 25.4
BACKGROUND = 255


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a printable OpenCV checkerboard calibration board (PNG)."
    )
    parser.add_argument("--corners-x", type=int, default=9,
                        help="inner corners per row (default: 9)")
    parser.add_argument("--corners-y", type=int, default=6,
                        help="inner corners per column (default: 6)")
    parser.add_argument("--square-mm", type=float, default=24.0,
                        help="black/white square side length in mm (default: 24)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"output PNG path (default: {DEFAULT_OUTPUT})")
    return parser


def render_board(corners_x: int, corners_y: int, square_px: int) -> np.ndarray:
    margin = int(round(square_px * 1.5))
    cols = corners_x + 1
    rows = corners_y + 1
    width = margin * 2 + cols * square_px
    height = margin * 2 + rows * square_px
    board = np.full((height, width), BACKGROUND, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x0 = margin + c * square_px
                y0 = margin + r * square_px
                board[y0:y0 + square_px, x0:x0 + square_px] = 0
    return board


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.corners_x < 2 or args.corners_y < 2:
        print("--corners-x / --corners-y must be >= 2", file=__import__("sys").stderr)
        return 2
    if args.square_mm <= 0:
        print("--square-mm must be > 0", file=__import__("sys").stderr)
        return 2

    square_px = int(round(args.square_mm * PIXELS_PER_MM))
    cols = args.corners_x + 1
    rows = args.corners_y + 1
    board = render_board(args.corners_x, args.corners_y, square_px)
    margin = int(round(square_px * 1.5))
    out = args.output
    if not os.path.isabs(out):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = os.path.normpath(os.path.join(root, out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not cv2.imwrite(out, board):
        print(f"failed to write {out}", file=__import__("sys").stderr)
        return 2

    mm_per_px = 25.4 / DPI
    print(f"Wrote: {out}")
    print(f"Board: {args.corners_x}x{args.corners_y} inner corners "
          f"({cols} x {rows} squares), {args.square_mm:g} mm squares "
          f"({square_px} px at 300 dpi).")
    print(f"Print size: {cols * args.square_mm:g} x {rows * args.square_mm:g} mm "
          f"(+ {margin * mm_per_px:.0f} mm margin).")
    print(f"Page: {board.shape[1] * mm_per_px / 25.4:.1f} x "
          f"{board.shape[0] * mm_per_px / 25.4:.1f} in")
    print("Print at 100% / 'actual size' and verify one square is "
          f"{args.square_mm:g} mm with a ruler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())