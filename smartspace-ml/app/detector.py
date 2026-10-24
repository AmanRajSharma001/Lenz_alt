"""
app/detector.py — YOLO detection and ByteTrack tracking wrapper.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# COCO class names we care about
PERSON_CLASS = "person"

# Objects that indicate someone is sitting purposefully (anti-idle evidence)
PURPOSEFUL_OBJECTS = {"cup", "bottle", "bowl", "laptop", "book", "cell phone", "keyboard"}

# Objects that indicate "dirty / unattended" zone
UNATTENDED_OBJECTS = {"cup", "bottle", "bowl", "plate", "fork", "knife", "spoon", "wine glass"}

# Safety-related objects
SAFETY_OBJECTS = {"backpack", "suitcase", "handbag"}


@dataclass
class Detection:
    track_id: int            # -1 if not tracked
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1 y1 x2 y2 (pixel)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return self.width / (h + 1e-6)


@dataclass
class PoseResult:
    track_id: int
    keypoints: np.ndarray    # (17, 3) x,y,conf
    bbox: Tuple[float, float, float, float]


@dataclass
class FrameResult:
    frame_index: int
    video_time: float
    persons: List[Detection] = field(default_factory=list)
    objects: List[Detection] = field(default_factory=list)
    poses: List[PoseResult] = field(default_factory=list)
    raw_frame: Optional[np.ndarray] = None   # resized BGR frame


class Detector:
    """Wraps YOLOv8 detection + ByteTrack and optional pose estimation."""

    def __init__(self) -> None:
        self._det_model = None
        self._pose_model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics") from exc

        logger.info("Loading detection model: %s", settings.YOLO_WEIGHTS)
        self._det_model = YOLO(settings.YOLO_WEIGHTS)

        if settings.POSE_ENABLED:
            logger.info("Loading pose model: %s", settings.POSE_WEIGHTS)
            try:
                self._pose_model = YOLO(settings.POSE_WEIGHTS)
            except Exception as e:
                logger.warning("Could not load pose model: %s — pose disabled", e)
                self._pose_model = None

        self._loaded = True
        logger.info("Detector ready.")

    def process_frame(self, frame: np.ndarray, frame_index: int, fps: float) -> FrameResult:
        """Run detection + tracking on a single frame."""
        if not self._loaded:
            self.load()

        video_time = frame_index / max(fps, 1e-6)
        result = FrameResult(
            frame_index=frame_index,
            video_time=video_time,
            raw_frame=frame,
        )

        # ── Detection + ByteTrack ────────────────────────────────────────────
        det_results = self._det_model.track(
            frame,
            persist=True,
            conf=settings.YOLO_CONF,
            iou=settings.YOLO_IOU,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        if det_results and det_results[0].boxes is not None:
            boxes = det_results[0].boxes
            names = det_results[0].names

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                tid = int(box.id[0]) if box.id is not None else -1

                det = Detection(
                    track_id=tid,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
                if cls_name == PERSON_CLASS:
                    result.persons.append(det)
                else:
                    result.objects.append(det)

        # ── Pose estimation ──────────────────────────────────────────────────
        if self._pose_model is not None:
            try:
                pose_results = self._pose_model.track(
                    frame,
                    persist=True,
                    conf=settings.YOLO_CONF,
                    verbose=False,
                )
                if pose_results and pose_results[0].keypoints is not None:
                    kp_data = pose_results[0].keypoints.data  # (N, 17, 3)
                    pb = pose_results[0].boxes
                    for j in range(len(kp_data)):
                        tid = int(pb.id[j]) if (pb.id is not None) else -1
                        x1, y1, x2, y2 = pb.xyxy[j].tolist()
                        result.poses.append(
                            PoseResult(
                                track_id=tid,
                                keypoints=kp_data[j].cpu().numpy(),
                                bbox=(x1, y1, x2, y2),
                            )
                        )
            except Exception as e:
                logger.debug("Pose estimation error on frame %d: %s", frame_index, e)

        return result
