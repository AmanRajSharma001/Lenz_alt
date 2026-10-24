#!/usr/bin/env python3
"""
make_test_video.py — generates a realistic synthetic test video using real person
crops from ultralytics assets so YOLOv8 detects and tracks people reliably:
  - Person 1 starts in Dining Area (0-15s), traverses Server Room Door (20-25s, restricted zone intrusion),
    and exits.
  - Person 2 sits in Queue Area (0-90s) with low motion (promoting to IDLE_OCCUPIED and loitering).
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np

# Ensure YOLO config dir doesn't write to restricted mac location
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")


def _get_sprites():
    """Extract person sprites from ultralytics assets or create textured fallbacks."""
    assets_dir = None
    try:
        import ultralytics
        assets_dir = Path(ultralytics.__file__).parent / "assets"
    except Exception:
        pass

    bus_path = assets_dir / "bus.jpg" if assets_dir else None
    if bus_path and bus_path.exists():
        bus = cv2.imread(str(bus_path))
        # Crop 2 real persons from bus image
        p1 = bus[398:902, 48:245]
        p2 = bus[392:877, 669:810]
    else:
        # Fallback textured shapes
        p1 = np.full((250, 100, 3), (180, 120, 70), dtype=np.uint8)
        p2 = np.full((250, 90, 3), (70, 120, 180), dtype=np.uint8)

    def scale_to_h(img, target_h):
        s = target_h / max(1, img.shape[0])
        return cv2.resize(img, (max(10, int(img.shape[1] * s)), target_h))

    s1 = scale_to_h(p1, 200)
    s2 = scale_to_h(p2, 200)
    return s1, s2


def _paste_sprite(frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    sh, sw = sprite.shape[:2]
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + sw), min(fh, y + sh)
    if x2 <= x1 or y2 <= y1:
        return
    crop_x1 = x1 - x
    crop_y1 = y1 - y
    crop_x2 = crop_x1 + (x2 - x1)
    crop_y2 = crop_y1 + (y2 - y1)
    frame[y1:y2, x1:x2] = sprite[crop_y1:crop_y2, crop_x1:crop_x2]


def make_test_video(out_path: str = "sample.mp4", fps: int = 25, duration_s: int = 90) -> None:
    W, H = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    s1, s2 = _get_sprites()
    total_frames = fps * duration_s

    for i in range(total_frames):
        t = i / fps
        # Office floor background
        frame = np.full((H, W, 3), 110, dtype=np.uint8)
        # Add floor grid texture
        frame[::40, :, :] = 95
        frame[:, ::40, :] = 95

        # ── Zone outlines for visual reference ──────────────────────────
        # Zone 1: Dining Area (left: x 50..450, y 200..550)
        cv2.rectangle(frame, (50, 200), (450, 550), (80, 80, 80), 2)
        cv2.putText(frame, "Zone: Dining Area", (55, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Server Room Door: Restricted Zone (center: x 588..691, y 288..504)
        cv2.rectangle(frame, (588, 288), (691, 504), (60, 60, 180), 2)
        cv2.putText(frame, "RESTRICTED: Server Door", (560, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 220), 1)

        # Zone 2: Queue Area (right: x 700..1200, y 100..650)
        cv2.rectangle(frame, (700, 100), (1200, 650), (80, 80, 80), 2)
        cv2.putText(frame, "Zone: Queue Area", (705, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── Person 1:
        # t in [0, 15): standing in Dining Area (x ~ 180, y ~ 260)
        # t in [15, 20): walking toward Server Room Door
        # t in [20, 26): inside Server Room Door (Restricted Zone Intrusion)
        # t in [26, 35): walking off frame
        if t < 15:
            p1_x = int(180 + 10 * math.sin(t * 0.5))
            p1_y = 260
            _paste_sprite(frame, s1, p1_x, p1_y)
        elif t < 20:
            frac = (t - 15) / 5.0
            p1_x = int(180 + frac * (600 - 180))
            p1_y = 260
            _paste_sprite(frame, s1, p1_x, p1_y)
        elif t < 26:
            p1_x = 610
            p1_y = 270
            _paste_sprite(frame, s1, p1_x, p1_y)
        elif t < 35:
            frac = (t - 26) / 9.0
            p1_x = int(610 + frac * 800)
            p1_y = 260
            _paste_sprite(frame, s1, p1_x, p1_y)

        # ── Person 2: sits in Queue Area, barely moving (entire video) ──────
        jitter_x = int(2 * math.sin(t * 0.2))
        jitter_y = int(2 * math.cos(t * 0.3))
        q_x = 880 + jitter_x
        q_y = 260 + jitter_y
        _paste_sprite(frame, s2, q_x, q_y)

        # ── OSD ────────────────────────────────────────────────────────────
        cv2.putText(frame, f"t={t:.1f}s  frame={i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        writer.write(frame)

    writer.release()
    print(f"Synthetic test video written: {out_path}  ({total_frames} frames @ {fps}fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sample.mp4")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--duration", type=int, default=90)
    args = parser.parse_args()
    make_test_video(args.out, args.fps, args.duration)
