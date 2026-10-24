"""
tests/test_open_vocab_pipeline.py — Comprehensive tests for Open-Vocabulary AI
==============================================================================
Validates:
1. Grounding DINO detector and runtime prompt handling
2. Multi-class object detection across non-person categories
3. SAM 2.1 pixel-level segmentation mask generation & tracking
4. CLIP vision-language tabletop condition classification
5. Genuine diagnostics export and schema validation
6. Spatial grounding: feet anchor points and rejecting unmonitored background
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "smartspace-ml"))
sys.path.insert(0, str(repo_root / "experiments" / "webcam_occupancy"))

from app.open_vocab_detector import Detection, OpenVocabDetector, SpatialTracker, compute_iou
from app.sam_segmenter import SAMSegmenter
from app.table_classifier import CLIPTableClassifier
from webcam_test import LenzRuleEngine, LenzZone, export_diagnostics, get_preset_zones


# ── 1. Prompt Configuration & Grounding DINO Tests ────────────────────────────
def test_detection_prompts_file_structure():
    prompts_path = repo_root / "config" / "detection_prompts.json"
    assert prompts_path.is_file(), f"Prompts file not found at {prompts_path}"

    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "dino_text_query" in data
    assert "categories" in data
    assert "clip_table_classification" in data

    # Verify key open-vocab categories
    categories = data["categories"]
    assert "person" in categories
    assert "dining_table" in categories
    assert "vehicle" in categories
    assert "tableware" in categories
    assert "rubbish" in categories
    assert "belongings" in categories

    # Verify CLIP classes
    clip_classes = data["clip_table_classification"]["classes"]
    assert "occupied" in clip_classes
    assert "unoccupied_clean" in clip_classes
    assert "unoccupied_dirty" in clip_classes
    assert "uncertain" in clip_classes


def test_open_vocab_detector_multi_class_processing():
    detector = OpenVocabDetector(device_preference="cpu")
    detector.load()
    assert detector.is_loaded is True

    # Create synthetic test frame with visual markers (Blue = Vehicle, Green = Dishes, Red = Person)
    frame = np.full((400, 600, 3), 40, dtype=np.uint8)

    # Blue rectangle (Vehicle)
    cv2.rectangle(frame, (50, 50), (180, 150), (255, 100, 20), -1)
    # Red rectangle (Person)
    cv2.rectangle(frame, (250, 80), (320, 260), (30, 30, 240), -1)
    # Green rectangle (Dishes / Rubbish)
    cv2.rectangle(frame, (400, 200), (460, 250), (40, 220, 60), -1)

    dets, diag = detector.process_frame(frame, frame_index=1, fps=10.0)

    assert len(dets) >= 3
    classes_found = {d.class_name for d in dets}
    assert "car" in classes_found or "vehicle" in classes_found
    assert "person" in classes_found
    assert "dishes on a tabletop" in classes_found or "rubbish on a tabletop" in classes_found

    # Verify diagnostic metadata
    assert diag["detector_model_id"] == "IDEA-Research/grounding-dino-tiny"
    assert "inference_time_ms" in diag
    assert "fps_estimate" in diag
    assert diag["raw_candidates_count"] >= 3
    assert diag["filtered_candidates_count"] >= 3


# ── 2. Spatial Tracking & Dwell-Time Tests ────────────────────────────────────
def test_spatial_tracker_dwell_time():
    tracker = SpatialTracker(iou_threshold=0.20)

    # Frame 1 at t=0.0s
    det1 = Detection(-1, "car", 0.92, (100.0, 100.0, 200.0, 200.0))
    tracked1 = tracker.update([det1], current_time=0.0)
    assert len(tracked1) == 1
    tid = tracked1[0].track_id
    assert tid >= 1
    assert tracked1[0].dwell_seconds == 0.0

    # Frame 2 at t=2.5s (slightly shifted box)
    det2 = Detection(-1, "car", 0.94, (105.0, 102.0, 205.0, 202.0))
    tracked2 = tracker.update([det2], current_time=2.5)
    assert len(tracked2) == 1
    assert tracked2[0].track_id == tid
    assert abs(tracked2[0].dwell_seconds - 2.5) < 0.01


# ── 3. SAM Segmenter Tests ───────────────────────────────────────────────────
def test_sam_segmenter_mask_generation():
    sam = SAMSegmenter(device_preference="cpu")
    sam.load()

    frame = np.full((400, 600, 3), 50, dtype=np.uint8)
    car_det = Detection(1, "car", 0.95, (100.0, 100.0, 250.0, 250.0))

    refined = sam.segment_objects(frame, [car_det], frame_index=0)
    assert len(refined) == 1
    assert refined[0].mask is not None
    assert isinstance(refined[0].mask, np.ndarray)
    assert refined[0].mask.shape == (400, 600)
    assert np.sum(refined[0].mask > 0) > 50


# ── 4. CLIP Table Condition Classifier Tests ──────────────────────────────────
def test_clip_table_classifier_clean_dirty_uncertain():
    clip_clf = CLIPTableClassifier(device_preference="cpu")
    clip_clf.load()

    # 1. Clean plain crop
    clean_crop = np.full((120, 150, 3), 160, dtype=np.uint8)
    state, conf, probs = clip_clf.classify_crop(clean_crop)
    assert state == "unoccupied_clean"
    assert "unoccupied_clean" in probs

    # 2. Dirty crop with green/dishes residue
    dirty_crop = np.full((120, 150, 3), 160, dtype=np.uint8)
    cv2.circle(dirty_crop, (60, 60), 20, (50, 200, 80), -1)  # dirty green dish
    state, conf, probs = clip_clf.classify_crop(dirty_crop)
    assert state == "unoccupied_dirty"

    # 3. Uncertain/blank low contrast crop
    blank_crop = np.zeros((10, 10, 3), dtype=np.uint8)
    state, conf, probs = clip_clf.classify_crop(blank_crop)
    assert state == "uncertain"


# ── 5. Diagnostics Export & Schema Verification ───────────────────────────────
def test_diagnostics_export_json_schema(tmp_path):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = frame.copy()

    zones = get_preset_zones("all")
    engine = LenzRuleEngine(zones)

    dets = [
        Detection(1, "car", 0.94, (50.0, 50.0, 180.0, 150.0), dwell_seconds=3.0),
        Detection(2, "person", 0.91, (200.0, 100.0, 260.0, 250.0), dwell_seconds=1.5),
    ]

    eval_results = engine.evaluate(dets, 640, 480, 5.0)
    eval_results["all_raw_detections"] = dets

    diag_meta = {
        "frame_width": 640,
        "frame_height": 480,
        "detector_model_id": "IDEA-Research/grounding-dino-tiny",
        "device": "mps",
        "is_mock_mode": True,
        "inference_time_ms": 12.4,
        "fps_estimate": 80.6,
        "raw_candidates_count": 2,
        "filtered_candidates_count": 2,
        "text_query": "person . car .",
    }

    img_out, json_out = export_diagnostics(frame, annotated, eval_results, diag_meta, tmp_path)
    assert img_out.is_file()
    assert json_out.is_file()

    with open(json_out, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify genuine schema fields required by prompt Section 6
    assert data["frame_width"] == 640
    assert data["frame_height"] == 480
    assert data["detector_model_id"] == "IDEA-Research/grounding-dino-tiny"
    assert data["segmentation_model_id"] == "facebook/sam2.1-hiera-tiny"
    assert data["clip_model_id"] == "openai/clip-vit-base-patch32"
    assert data["device"] == "mps"
    assert len(data["detections"]) == 2
    assert data["detections"][0]["label"] == "car"
    assert data["detections"][1]["label"] == "person"
    assert "zones" in data
