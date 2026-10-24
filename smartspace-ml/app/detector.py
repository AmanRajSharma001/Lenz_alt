"""
app/detector.py — Open-Vocabulary Detection & Segmentation Pipeline.
===================================================================
Replaces YOLO and COCO integer classes with Grounding DINO, SAM 2.1, and CLIP.
Strictly zero imports or invocations of Ultralytics/YOLO.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.config import settings
from app.open_vocab_detector import (
    BELONGINGS_CLASSES,
    FURNITURE_CLASSES,
    PEOPLE_CLASSES,
    PURPOSEFUL_OBJECTS,
    RUBBISH_OBJECTS,
    TABLEWARE_OBJECTS,
    VEHICLE_CLASSES,
    Detection,
    OpenVocabDetector,
)
from app.sam_segmenter import SAMSegmenter
from app.table_classifier import CLIPTableClassifier

logger = logging.getLogger(__name__)

# Re-export semantic groupings
PERSON_CLASS = "person"
SAFETY_OBJECTS = BELONGINGS_CLASSES
UNATTENDED_OBJECTS = TABLEWARE_OBJECTS.union(BELONGINGS_CLASSES).union(RUBBISH_OBJECTS)


@dataclass
class PoseResult:
    track_id: int
    keypoints: np.ndarray    # (17, 3) x, y, conf
    bbox: Tuple[float, float, float, float]


@dataclass
class FrameResult:
    frame_index: int
    video_time: float
    persons: List[Detection] = field(default_factory=list)
    vehicles: List[Detection] = field(default_factory=list)
    tableware: List[Detection] = field(default_factory=list)
    objects: List[Detection] = field(default_factory=list)
    poses: List[PoseResult] = field(default_factory=list)
    raw_frame: Optional[np.ndarray] = None   # BGR frame
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class Detector:
    """
    Open-Vocabulary Vision Pipeline combining:
      - Primary Detector: IDEA-Research/grounding-dino-tiny
      - Pixel-Level Segmenter: facebook/sam2.1-hiera-tiny
      - Tabletop Classifier: openai/clip-vit-base-patch32
    """

    def __init__(self) -> None:
        self.dino_detector = OpenVocabDetector(
            model_id=settings.GROUNDING_DINO_MODEL,
            prompts_path=settings.PROMPTS_CONFIG_PATH,
            device_preference=settings.DEVICE_PREFERENCE,
            box_threshold=settings.DETECTION_CONFIDENCE,
        )
        self.sam_segmenter = SAMSegmenter(
            model_id=settings.SAM_MODEL,
            device_preference=settings.DEVICE_PREFERENCE,
        )
        self.table_classifier = CLIPTableClassifier(
            model_id=settings.CLIP_MODEL,
            device_preference=settings.DEVICE_PREFERENCE,
        )
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        logger.info(
            "Initializing Open-Vocabulary pipeline with Grounding DINO (%s), SAM (%s), CLIP (%s)...",
            settings.GROUNDING_DINO_MODEL,
            settings.SAM_MODEL,
            settings.CLIP_MODEL,
        )
        self.dino_detector.load()
        if settings.SEGMENTATION_ENABLED:
            self.sam_segmenter.load()
        if settings.TABLE_CLASSIFICATION_ENABLED:
            self.table_classifier.load()
        self._loaded = True
        logger.info("Open-Vocabulary Detector ready on %s.", self.dino_detector.device)

    def process_frame(self, frame: np.ndarray, frame_index: int, fps: float) -> FrameResult:
        """Run open-vocabulary detection, SAM refinement, and spatial organization."""
        if not self._loaded:
            self.load()

        video_time = frame_index / max(fps, 1e-6)
        result = FrameResult(
            frame_index=frame_index,
            video_time=video_time,
            raw_frame=frame,
        )

        # 1. Open-Vocabulary Grounding DINO detection + tracking
        detections, diag = self.dino_detector.process_frame(
            frame=frame,
            frame_index=frame_index,
            fps=fps,
        )

        # 2. SAM Segmentation refinement for key spatial objects
        if settings.SEGMENTATION_ENABLED and detections:
            detections = self.sam_segmenter.segment_objects(
                frame=frame,
                detections=detections,
                frame_index=frame_index,
                classes_to_segment=VEHICLE_CLASSES.union({"dining table", "person"}),
            )

        # 3. Categorize detections without silently discarding any candidates
        for det in detections:
            cls_name = det.class_name.lower().strip()

            if cls_name in PEOPLE_CLASSES:
                result.persons.append(det)
            else:
                result.objects.append(det)
                if cls_name in VEHICLE_CLASSES or any(v in cls_name for v in ("car", "vehicle", "truck", "bus")):
                    result.vehicles.append(det)
                if cls_name in TABLEWARE_OBJECTS or any(t in cls_name for t in ("dishes", "bottle", "cup", "plate")):
                    result.tableware.append(det)

        diag["segmentation_model_id"] = settings.SAM_MODEL
        diag["clip_model_id"] = settings.CLIP_MODEL
        result.diagnostics = diag

        return result
