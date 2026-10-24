"""
app/incidents.py — Rolling frame buffer + evidence clip saving.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# (video_time, bgr_frame)
FrameEntry = Tuple[float, np.ndarray]

# Haar cascade for optional face blurring
_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cascade_path):
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def _blur_faces(frame: np.ndarray) -> np.ndarray:
    if not settings.BLUR_FACES:
        return frame
    cascade = _get_face_cascade()
    if cascade is None:
        return frame
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)
    result = frame.copy()
    for x, y, w, h in faces:
        roi = result[y:y+h, x:x+w]
        result[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (51, 51), 0)
    return result


class FrameBuffer:
    """Rolling per-camera buffer of the last N seconds of frames."""

    def __init__(self, buffer_s: float = 15.0) -> None:
        self._buffer_s = buffer_s
        self._frames: Deque[FrameEntry] = deque()
        self._lock = Lock()

    def push(self, video_time: float, frame: np.ndarray) -> None:
        with self._lock:
            self._frames.append((video_time, frame))
            # Trim old frames
            cutoff = video_time - self._buffer_s
            while self._frames and self._frames[0][0] < cutoff:
                self._frames.popleft()

    def get_frames_around(self, video_time: float) -> List[FrameEntry]:
        """Return frames from (video_time - buffer_s) to (video_time + buffer_s)."""
        with self._lock:
            return list(self._frames)

    def get_recent(self, n: int = 5) -> List[np.ndarray]:
        with self._lock:
            entries = list(self._frames)[-n:]
            return [f for _, f in entries]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class EvidenceSaver:
    """Saves evidence clips and keyframes for confirmed incidents."""

    def __init__(self, buffer: FrameBuffer) -> None:
        self._buffer = buffer
        settings.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        incident_id: str,
        event_type: str,
        keyframes: List[np.ndarray],
        video_time: float,
        fps: float,
    ) -> Dict[str, str]:
        """
        Save evidence for an incident.
        Returns dict with 'clip_path', 'clip_hash', and 'snapshot_path'.
        """
        out_dir = settings.EVIDENCE_DIR / incident_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Keyframes ────────────────────────────────────────────────────────
        snapshot_path: Optional[str] = None
        frames_to_save = keyframes[:5] if keyframes else self._buffer.get_recent(5)
        for i, kf in enumerate(frames_to_save):
            blurred = _blur_faces(kf)
            kf_path = out_dir / f"keyframe_{i:02d}.jpg"
            cv2.imwrite(str(kf_path), blurred, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if i == 0:
                snapshot_path = str(kf_path)

        # ── Evidence clip ────────────────────────────────────────────────────
        clip_path: Optional[str] = None
        clip_hash: Optional[str] = None
        all_frames = self._buffer.get_frames_around(video_time)
        if all_frames:
            h, w = all_frames[0][1].shape[:2]
            clip_file = out_dir / "clip.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = min(settings.EVIDENCE_FPS, fps) or 10
            writer = cv2.VideoWriter(str(clip_file), fourcc, out_fps, (w, h))
            for _, fr in all_frames:
                writer.write(_blur_faces(fr))
            writer.release()
            clip_path = str(clip_file)
            clip_hash = _sha256_file(clip_file)
            logger.info("Evidence saved: %s  (hash=%s…)", clip_file.name, clip_hash[:8])

        return {
            "clip_path": clip_path or "",
            "clip_hash": clip_hash or "",
            "snapshot_path": snapshot_path or "",
        }
