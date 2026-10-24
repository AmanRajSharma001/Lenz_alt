"""
app/motion.py — Idle / motion scoring for a zone.

The score is a weighted combination of:
1. Centroid displacement (normalised by bbox height) — movement between frames.
2. Frame-difference pixel motion inside person bounding boxes.
3. Pose keypoint variance (wrists + shoulders) from the pose model.

Score range: [0.0, 1.0].  Low score (< IDLE_MOTION_THRESHOLD) = idle.
"""
from __future__ import annotations

import collections
import logging
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.detector import Detection, PoseResult

logger = logging.getLogger(__name__)

# Sliding window length in VIDEO seconds
_WINDOW_S = 45.0

# Keypoint indices (COCO17 format)
_KP_WRIST_L = 9
_KP_WRIST_R = 10
_KP_SHOULDER_L = 5
_KP_SHOULDER_R = 6
_POSE_KPS = [_KP_WRIST_L, _KP_WRIST_R, _KP_SHOULDER_L, _KP_SHOULDER_R]


class MotionScorer:
    """
    Maintains a sliding window of motion observations per zone.
    Call update() once per processed frame for a given zone.
    """

    def __init__(self, zone_id: str) -> None:
        self.zone_id = zone_id
        # Each entry: (video_time, displacement_score, pixel_score, pose_score)
        self._window: Deque[Tuple[float, float, float, float]] = collections.deque()
        # Previous frame's centroids {track_id: (cx, cy, h)}
        self._prev_centroids: Dict[int, Tuple[float, float, float]] = {}
        # Previous grey frame crop for pixel diff
        self._prev_grey: Optional[np.ndarray] = None
        self._last_score: float = 0.0

    def update(
        self,
        video_time: float,
        persons_in_zone: List[Detection],
        poses_in_zone: List[PoseResult],
        zone_mask: Optional[np.ndarray],   # binary mask, same size as frame
        frame_grey: Optional[np.ndarray],  # full greyscale frame
    ) -> float:
        """Return the current motion score for the zone."""
        if not persons_in_zone:
            self._prev_centroids = {}
            self._prev_grey = None
            self._last_score = 0.0
            self._trim(video_time)
            return 0.0

        disp_score = self._centroid_displacement(persons_in_zone)
        pixel_score = self._pixel_motion(persons_in_zone, frame_grey)
        pose_score = self._pose_variance(poses_in_zone)

        # Update prev state
        self._prev_centroids = {
            p.track_id: (
                (p.bbox[0] + p.bbox[2]) / 2,
                (p.bbox[1] + p.bbox[3]) / 2,
                p.height,
            )
            for p in persons_in_zone
            if p.track_id >= 0
        }
        self._prev_grey = frame_grey

        self._window.append((video_time, disp_score, pixel_score, pose_score))
        self._trim(video_time)

        if not self._window:
            return 0.0

        # Weighted average across window entries
        ds = [e[1] for e in self._window]
        ps = [e[2] for e in self._window]
        qs = [e[3] for e in self._window]

        w_d, w_p, w_q = 0.4, 0.4, 0.2
        score = (
            w_d * float(np.mean(ds))
            + w_p * float(np.mean(ps))
            + w_q * float(np.mean(qs))
        )
        self._last_score = float(np.clip(score, 0.0, 1.0))
        return self._last_score

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _trim(self, video_time: float) -> None:
        cutoff = video_time - _WINDOW_S
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _centroid_displacement(self, persons: List[Detection]) -> float:
        if not self._prev_centroids:
            return 0.0
        disps: List[float] = []
        for p in persons:
            if p.track_id >= 0 and p.track_id in self._prev_centroids:
                px, py, ph = self._prev_centroids[p.track_id]
                cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                norm = dist / (ph + 1e-6)
                disps.append(float(np.clip(norm, 0.0, 1.0)))
        return float(np.mean(disps)) if disps else 0.0

    def _pixel_motion(
        self, persons: List[Detection], frame_grey: Optional[np.ndarray]
    ) -> float:
        if frame_grey is None or self._prev_grey is None:
            return 0.0
        if frame_grey.shape != self._prev_grey.shape:
            return 0.0
        scores: List[float] = []
        for p in persons:
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame_grey.shape[1], x2)
            y2 = min(frame_grey.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cur_crop = frame_grey[y1:y2, x1:x2].astype(np.float32)
            prv_crop = self._prev_grey[y1:y2, x1:x2].astype(np.float32)
            diff = np.abs(cur_crop - prv_crop)
            # Fraction of pixels that changed by more than 10 grey levels
            frac = float(np.mean(diff > 10))
            scores.append(frac)
        return float(np.mean(scores)) if scores else 0.0

    def _pose_variance(self, poses: List[PoseResult]) -> float:
        if not poses:
            return 0.0
        variances: List[float] = []
        for pr in poses:
            kps = pr.keypoints  # (17, 3)
            selected = kps[_POSE_KPS]  # (4, 3)
            conf = selected[:, 2]
            visible = selected[conf > 0.5]
            if len(visible) < 2:
                continue
            xy = visible[:, :2]
            var = float(np.var(xy[:, 0]) + np.var(xy[:, 1]))
            # Normalise by bbox area
            area = (pr.bbox[2] - pr.bbox[0]) * (pr.bbox[3] - pr.bbox[1]) + 1e-6
            variances.append(float(np.clip(var / area, 0.0, 1.0)))
        return float(np.mean(variances)) if variances else 0.0
