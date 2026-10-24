#!/usr/bin/env python3
"""
experiments/webcam_occupancy/webcam_test.py

Lenz — Open-Vocabulary Intelligent Video Analytics & Occupancy Prototype
========================================================================
Architecture:
  - Detector: IDEA-Research/grounding-dino-tiny (Open-Vocabulary Zero-Shot Object Detection)
  - Segmenter: facebook/sam2.1-hiera-tiny (Pixel-Level Segmentation & Mask Tracking)
  - Table Condition Classifier: openai/clip-vit-base-patch32 (Zero-Shot Region State Intelligence)
  - Hardware Acceleration: Apple Silicon MPS (Metal Performance Shaders) with CPU fallback

Key Lenz Use Cases Supported:
  1. Open-Vocabulary Prompt Detection (People, Vehicles, Furniture, Electronics, Tableware, Bags)
  2. Spatial Multi-Object Tracking (Persistent Track IDs and Dwell Timing)
  3. Grounded Table Occupancy & Cleanliness (Clean vs Dirty Dishes/Trash vs Occupied)
  4. Smart Parking Space Utilization Monitoring
  5. Crowd Density & Overcrowding Estimation
  6. Queue & Waiting-Area Congestion Monitoring
  7. Restricted-Area Intrusion Detection
  8. Unattended Object & Suspicious Loitering Rules
  9. Comprehensive Diagnostics Export (--diagnostics and --diagnostics-image)

YOLO and COCO integer classes are strictly NOT used.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Ensure smartspace-ml is in import path
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "smartspace-ml"))

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

logger = logging.getLogger("lenz_prototype")

# ── Color Palette & Styling ───────────────────────────────────────────────────
COLOR_BG_DARK = (20, 20, 20)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_MUTED = (160, 160, 160)
COLOR_BORDER = (50, 50, 50)

# Alert Colors (BGR)
COLOR_SAFE_GREEN = (40, 200, 60)
COLOR_WARNING_AMBER = (0, 180, 255)
COLOR_ALERT_RED = (40, 50, 235)
COLOR_UNCERTAIN_YELLOW = (0, 220, 240)
COLOR_INFO_BLUE = (235, 160, 40)

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person": (235, 180, 50),
    "seated person": (235, 180, 50),
    "standing person": (235, 180, 50),
    "car": (50, 150, 250),
    "motorcycle": (50, 200, 250),
    "bicycle": (50, 230, 210),
    "bus": (30, 120, 230),
    "truck": (20, 90, 210),
    "vehicle": (50, 150, 250),
    "backpack": (210, 80, 220),
    "handbag": (180, 70, 200),
    "suitcase": (160, 50, 180),
    "chair": (180, 180, 100),
    "dining table": (150, 130, 90),
    "bench": (160, 160, 90),
    "laptop": (255, 140, 0),
    "phone": (230, 100, 40),
    "bottle": (80, 210, 120),
    "cup": (60, 190, 150),
    "dishes on a tabletop": (70, 180, 170),
    "rubbish on a tabletop": (40, 50, 235),
}

# Compatibility alias
TrackedDetection = Detection


def resolve_model_path(custom_path: Optional[str] = None) -> str:
    """Returns the primary Open-Vocabulary model identifier or custom path."""
    if custom_path:
        return custom_path
    return "IDEA-Research/grounding-dino-tiny"


# ── Spatial Zone Definition ───────────────────────────────────────────────────
class LenzZone:
    """
    User-configurable spatial region of interest defined by normalized coordinates
    [x1, y1, x2, y2] in range [0.0, 1.0].
    """

    def __init__(
        self,
        zone_id: str,
        name: str,
        zone_type: str,
        norm_bbox: Tuple[float, float, float, float],
        target_classes: Set[str],
        color: Tuple[int, int, int] = COLOR_SAFE_GREEN,
    ) -> None:
        self.zone_id = zone_id
        self.name = name
        self.zone_type = zone_type  # "table", "parking", "restricted", "queue", "general"
        self.norm_bbox = norm_bbox
        self.target_classes = target_classes
        self.base_color = color

    def get_pixel_coords(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.norm_bbox
        return (
            int(x1 * frame_w),
            int(y1 * frame_h),
            int(x2 * frame_w),
            int(y2 * frame_h),
        )

    def contains_point(self, px: float, py: float, frame_w: int, frame_h: int) -> bool:
        zx1, zy1, zx2, zy2 = self.get_pixel_coords(frame_w, frame_h)
        return (zx1 <= px <= zx2) and (zy1 <= py <= zy2)

    def compute_overlap(
        self, det_bbox: Tuple[float, float, float, float], frame_w: int, frame_h: int
    ) -> Tuple[float, float]:
        """Returns (overlap_wrt_zone, overlap_wrt_detection)."""
        zx1, zy1, zx2, zy2 = self.get_pixel_coords(frame_w, frame_h)
        zone_area = max(1.0, (zx2 - zx1) * (zy2 - zy1))

        px1, py1, px2, py2 = det_bbox
        det_area = max(1.0, (px2 - px1) * (py2 - py1))

        ix1, iy1 = max(zx1, px1), max(zy1, py1)
        ix2, iy2 = min(zx2, px2), min(zy2, py2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        return inter_area / zone_area, inter_area / det_area


# ── Rule Engine & Multi-Use-Case Analytics ────────────────────────────────────
class LenzRuleEngine:
    """
    Evaluates deterministic spatial-temporal rules across camera frames for
    table states, parking availability, queue congestion, restricted intrusion,
    and loitering/unattended observations.
    """

    def __init__(
        self,
        zones: List[LenzZone],
        crowd_threshold: int = 4,
        queue_dwell_limit_s: float = 12.0,
        loiter_limit_s: float = 15.0,
        unattended_limit_s: float = 10.0,
        debounce_window: int = 10,
    ) -> None:
        self.zones = zones
        self.crowd_threshold = crowd_threshold
        self.queue_dwell_limit_s = queue_dwell_limit_s
        self.loiter_limit_s = loiter_limit_s
        self.unattended_limit_s = unattended_limit_s
        self.debounce_window = debounce_window

        # Temporal debouncing queues
        self.zone_occupancy_history: Dict[str, Deque[bool]] = {
            z.zone_id: collections.deque(maxlen=debounce_window) for z in zones
        }
        self.zone_person_history: Dict[str, Deque[bool]] = {
            z.zone_id: collections.deque(maxlen=debounce_window) for z in zones
        }
        self.zone_tableware_history: Dict[str, Deque[bool]] = {
            z.zone_id: collections.deque(maxlen=debounce_window) for z in zones
        }
        self.zone_vehicle_history: Dict[str, Deque[bool]] = {
            z.zone_id: collections.deque(maxlen=debounce_window) for z in zones
        }

        # Timers
        self.zone_occupied_start_time: Dict[str, float] = {}
        self.unattended_candidates: Dict[int, float] = {}  # track_id -> first_seen_time

    def evaluate(
        self,
        detections: List[Detection],
        frame_w: int,
        frame_h: int,
        current_time: float,
        table_classifier: Optional[CLIPTableClassifier] = None,
        frame_raw: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates spatial grounding and rules across all configured zones.
        """
        results: Dict[str, Any] = {
            "zone_states": {},
            "active_alerts": [],
            "all_raw_detections": detections,
            "timestamp": current_time,
        }

        # Segregate detected objects by class
        persons = [d for d in detections if d.class_name in PEOPLE_CLASSES]
        vehicles = [d for d in detections if d.class_name in VEHICLE_CLASSES]
        tableware = [d for d in detections if d.class_name in TABLEWARE_OBJECTS or d.class_name in RUBBISH_OBJECTS]
        personal_items = [d for d in detections if d.class_name in BELONGINGS_CLASSES]

        # ── 1. Evaluate Each Spatial Zone ──────────────────────────────────────
        for zone in self.zones:
            zx1, zy1, zx2, zy2 = zone.get_pixel_coords(frame_w, frame_h)
            z_w, z_h = zx2 - zx1, zy2 - zy1

            zone_persons: List[Detection] = []
            zone_vehicles: List[Detection] = []
            zone_tableware: List[Detection] = []
            zone_items: List[Detection] = []

            # A. Evaluate People (Strict Ground-Contact Feet Point)
            for p in persons:
                feet_x, feet_y = p.bottom_center
                if zone.contains_point(feet_x, feet_y, frame_w, frame_h):
                    zone_persons.append(p)
                elif zone.zone_type == "table":
                    # For tables, also check center or bbox overlap (person seated close)
                    oz, ob = zone.compute_overlap(p.bbox, frame_w, frame_h)
                    if ob >= 0.20 or zone.contains_point(p.center[0], p.center[1], frame_w, frame_h):
                        zone_persons.append(p)

            # B. Evaluate Vehicles (Bounding Box / SAM Mask Polygon Overlap)
            for v in vehicles:
                oz, ob = zone.compute_overlap(v.bbox, frame_w, frame_h)
                if oz >= 0.30 or ob >= 0.35:
                    zone_vehicles.append(v)

            # C. Evaluate Tableware & Rubbish
            for t in tableware:
                cx, cy = t.center
                if zone.contains_point(cx, cy, frame_w, frame_h):
                    zone_tableware.append(t)
                else:
                    oz, ob = zone.compute_overlap(t.bbox, frame_w, frame_h)
                    if oz >= 0.05 or ob >= 0.30:
                        zone_tableware.append(t)

            # D. Evaluate Belongings
            for it in personal_items:
                cx, cy = it.center
                if zone.contains_point(cx, cy, frame_w, frame_h):
                    zone_items.append(it)

            # Update debouncing buffers
            has_raw_person = len(zone_persons) > 0
            has_raw_vehicle = len(zone_vehicles) > 0
            has_raw_tableware = len(zone_tableware) > 0
            raw_occupied = has_raw_person or has_raw_vehicle or has_raw_tableware

            self.zone_occupancy_history[zone.zone_id].append(raw_occupied)
            self.zone_person_history[zone.zone_id].append(has_raw_person)
            self.zone_tableware_history[zone.zone_id].append(has_raw_tableware)
            self.zone_vehicle_history[zone.zone_id].append(has_raw_vehicle)

            occ_hist = self.zone_occupancy_history[zone.zone_id]
            hist_len = len(occ_hist)
            ratio = sum(occ_hist) / float(hist_len) if hist_len > 0 else 0.0

            # Determine debounced confirmation state
            if not raw_occupied and sum(occ_hist) <= 1 and hist_len >= 2:
                confirmed_status = "EMPTY"
            elif hist_len < self.debounce_window and sum(occ_hist) > 0 and sum(occ_hist) < self.debounce_window:
                confirmed_status = "UNCERTAIN"
            elif ratio >= 0.60:
                confirmed_status = "OCCUPIED"
            elif ratio <= 0.25:
                confirmed_status = "EMPTY"
            else:
                confirmed_status = "UNCERTAIN"

            # Track occupancy duration
            if confirmed_status == "OCCUPIED":
                if zone.zone_id not in self.zone_occupied_start_time:
                    self.zone_occupied_start_time[zone.zone_id] = current_time
                occ_duration = current_time - self.zone_occupied_start_time[zone.zone_id]
            else:
                self.zone_occupied_start_time.pop(zone.zone_id, None)
                occ_duration = 0.0

            # ── Zone Type State Evaluation ──────────────────────────────────
            state_label = "AVAILABLE"
            zone_count = 0
            details: Dict[str, Any] = {}

            if zone.zone_type == "table":
                clip_state = None
                clip_conf = 0.0
                if table_classifier is not None and frame_raw is not None and z_w > 10 and z_h > 10:
                    crop = frame_raw[max(0, zy1):min(frame_h, zy2), max(0, zx1):min(frame_w, zx2)]
                    clip_state, clip_conf, probs = table_classifier.classify_crop(crop)
                    details["clip_prediction"] = clip_state
                    details["clip_confidence"] = clip_conf
                    details["clip_probabilities"] = probs

                zone_count = len(zone_persons)
                dirty_items = [t.class_name for t in zone_tableware]
                details["dirty_items"] = dirty_items

                if confirmed_status == "UNCERTAIN":
                    state_label = "UNCERTAIN"
                    details["table_state"] = "uncertain"
                elif has_raw_person or clip_state == "occupied":
                    state_label = "OCCUPIED"
                    details["table_state"] = "occupied"
                elif has_raw_tableware or clip_state == "unoccupied_dirty":
                    state_label = "NOT OCCUPIED (NEEDS CLEANING)"
                    details["table_state"] = "unoccupied-dirty"
                    results["active_alerts"].append({
                        "severity": "INFO",
                        "type": "TABLE_NEEDS_CLEANING",
                        "message": f"Table '{zone.name}' requires cleaning (uncleared dishes/trash)",
                        "zone_id": zone.zone_id,
                    })
                elif clip_state == "uncertain":
                    state_label = "UNCERTAIN"
                    details["table_state"] = "uncertain"
                else:
                    state_label = "NOT OCCUPIED"
                    details["table_state"] = "unoccupied-clean"

            elif zone.zone_type == "parking":
                zone_count = len(zone_vehicles)
                if confirmed_status == "OCCUPIED":
                    state_label = "OCCUPIED"
                    parking_state = "OCCUPIED"
                elif confirmed_status == "UNCERTAIN":
                    state_label = "UNCERTAIN"
                    parking_state = "UNCERTAIN"
                else:
                    state_label = "AVAILABLE"
                    parking_state = "AVAILABLE"
                details["parking_state"] = parking_state

            elif zone.zone_type == "restricted":
                zone_count = len(zone_persons)
                if confirmed_status == "OCCUPIED":
                    state_label = "OCCUPIED"
                    results["active_alerts"].append({
                        "severity": "CRITICAL",
                        "type": "ZONE_INTRUSION",
                        "message": f"INTRUSION DETECTED in restricted zone '{zone.name}'! (Security Vault)",
                        "zone_id": zone.zone_id,
                    })
                elif confirmed_status == "UNCERTAIN":
                    state_label = "UNCERTAIN"
                else:
                    state_label = "NOT OCCUPIED"

            elif zone.zone_type == "queue":
                zone_count = len(zone_persons)
                state_label = "OCCUPIED" if confirmed_status == "OCCUPIED" else "CLEAR"
                if zone_count >= self.crowd_threshold:
                    results["active_alerts"].append({
                        "severity": "WARNING",
                        "type": "QUEUE_CONGESTION",
                        "message": f"High queue congestion: {zone_count} people in '{zone.name}'",
                        "zone_id": zone.zone_id,
                    })

            else:
                zone_count = len(zone_persons)
                if confirmed_status == "OCCUPIED":
                    state_label = "OCCUPIED"
                elif confirmed_status == "UNCERTAIN":
                    state_label = "UNCERTAIN"
                else:
                    state_label = "NOT OCCUPIED"

            # Check loitering within zone
            for p in zone_persons:
                if p.dwell_seconds >= self.loiter_limit_s:
                    results["active_alerts"].append({
                        "severity": "WARNING",
                        "type": "SUSPICIOUS_LOITERING",
                        "message": f"Track #{p.track_id} loitering in '{zone.name}' ({p.dwell_seconds:.1f}s) (Review required)",
                        "zone_id": zone.zone_id,
                    })

            zone_dict = {
                "name": zone.name,
                "type": zone.zone_type,
                "state": state_label,
                "count": zone_count,
                "occupancy_duration_s": occ_duration,
                "detections": zone_persons + zone_vehicles + zone_tableware + zone_items,
                "details": details,
            }
            # Flatten table and parking states directly into zone_dict for backwards compatibility
            if "table_state" in details:
                zone_dict["table_state"] = details["table_state"]
                zone_dict["dirty_items"] = details.get("dirty_items", [])
            if "parking_state" in details:
                zone_dict["parking_state"] = details["parking_state"]

            results["zone_states"][zone.zone_id] = zone_dict

        # ── 2. Evaluate Global Crowd Density ──────────────────────────────────
        if len(persons) >= self.crowd_threshold:
            results["active_alerts"].append({
                "severity": "WARNING",
                "type": "CROWD_SURGE",
                "message": f"Global crowd surge: {len(persons)} persons in view (Threshold: {self.crowd_threshold})",
                "zone_id": None,
            })

        # ── 3. Evaluate Unattended Bags / Luggage ──────────────────────────────
        active_item_ids = {it.track_id for it in personal_items if it.track_id >= 0}
        for item in personal_items:
            if item.track_id < 0:
                continue

            # Check proximity to nearest person
            near_person = False
            for p in persons:
                dist = math.hypot(item.center[0] - p.center[0], item.center[1] - p.center[1])
                if dist < 120.0:  # Proximity radius in pixels
                    near_person = True
                    break

            if near_person:
                self.unattended_candidates.pop(item.track_id, None)
            else:
                if item.track_id not in self.unattended_candidates:
                    self.unattended_candidates[item.track_id] = current_time

                alone_duration = current_time - self.unattended_candidates[item.track_id]
                if alone_duration >= self.unattended_limit_s:
                    results["active_alerts"].append({
                        "severity": "WARNING",
                        "type": "UNATTENDED_OBJECT",
                        "message": f"Unattended {item.class_name} #{item.track_id}: UNATTENDED {item.class_name.upper()} #{item.track_id} alone for {alone_duration:.1f}s (Review required)",
                        "zone_id": None,
                    })

        # Cleanup lost unattended items
        for stale_id in [k for k in self.unattended_candidates if k not in active_item_ids]:
            del self.unattended_candidates[stale_id]

        results["total_objects"] = len(detections)
        results["class_counts"] = dict(collections.Counter(d.class_name for d in detections))

        return results


# ── UI Dashboard & Telemetry HUD ──────────────────────────────────────────────
def render_dashboard(
    frame: np.ndarray,
    zones: List[LenzZone],
    eval_results: Dict[str, Any],
    fps: float,
    active_mode: str,
    conf_thresh: float,
    show_tracking: bool,
    show_zones: bool,
    source_name: str,
    detector_model: str = "IDEA-Research/grounding-dino-tiny",
    segmenter_model: str = "facebook/sam2.1-hiera-tiny",
    device_name: str = "mps",
    raw_candidates_count: int = 0,
    filtered_candidates_count: int = 0,
) -> np.ndarray:
    """Renders visual overlays, detections, alert banners, and genuine telemetry HUD."""
    canvas = frame.copy()
    h, w = frame.shape[:2]

    # 1. Draw Configured Zones
    if show_zones:
        for zone in zones:
            z_info = eval_results["zone_states"].get(zone.zone_id, {})
            state = z_info.get("state", "NOT OCCUPIED")

            if "OCCUPIED" in state or "INTRUSION" in state:
                color = COLOR_ALERT_RED
            elif "CLEANING" in state:
                color = COLOR_WARNING_AMBER
            elif "UNCERTAIN" in state:
                color = COLOR_UNCERTAIN_YELLOW
            else:
                color = COLOR_SAFE_GREEN

            zx1, zy1, zx2, zy2 = zone.get_pixel_coords(w, h)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), color, -1)
            cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, canvas)
            cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), color, 2)

            badge = f" {zone.name.upper()}: {state} "
            cv2.putText(
                canvas,
                badge,
                (zx1 + 4, max(22, zy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                2,
                cv2.LINE_AA,
            )

    # 2. Draw Detections & Tracking IDs
    all_dets = eval_results.get("all_raw_detections", [])
    for det in all_dets:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        box_color = CLASS_COLORS.get(det.class_name, (200, 200, 200))

        # Draw SAM mask if available
        if det.mask is not None and isinstance(det.mask, np.ndarray):
            mask_bool = det.mask.astype(bool)
            color_mask = np.zeros_like(canvas, dtype=np.uint8)
            color_mask[mask_bool] = box_color
            cv2.addWeighted(color_mask, 0.25, canvas, 0.75, 0, canvas)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        track_str = f"#{det.track_id} " if (show_tracking and det.track_id >= 0) else ""
        dwell_str = f" ({det.dwell_seconds:.0f}s)" if (show_tracking and det.dwell_seconds > 1.0) else ""
        label = f"{track_str}{det.class_name} {int(det.confidence * 100)}%{dwell_str}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tag_y1 = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 6, y1), box_color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )

        bx, by = (int(det.bottom_center[0]), int(det.bottom_center[1]))
        cv2.circle(canvas, (bx, by), 3, box_color, -1)

    # 3. Top Master Header & Genuine Telemetry Strip
    hud_h = 80
    cv2.rectangle(canvas, (0, 0), (w, hud_h), COLOR_BG_DARK, -1)
    cv2.line(canvas, (0, hud_h), (w, hud_h), COLOR_BORDER, 1)

    cv2.putText(
        canvas,
        "LENZ OPEN-VOCABULARY VIDEO INTELLIGENCE",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        COLOR_TEXT_WHITE,
        2,
        cv2.LINE_AA,
    )

    telemetry_line1 = (
        f"Detector: {detector_model} | SAM: {segmenter_model} | Device: {device_name.upper()} | {fps:.1f} FPS"
    )
    cv2.putText(
        canvas,
        telemetry_line1,
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    telemetry_line2 = (
        f"Mode: {active_mode.upper()} | Raw Candidates: {raw_candidates_count} | Surviving: {filtered_candidates_count} | Conf: {int(conf_thresh * 100)}%"
    )
    cv2.putText(
        canvas,
        telemetry_line2,
        (16, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    # 4. Active Alert Banners (Bottom-Left)
    alerts = eval_results.get("active_alerts", [])
    alert_y = h - 20
    for alert in reversed(alerts[-3:]):
        sev = alert.get("severity", "INFO")
        msg = alert.get("message", "")
        banner_color = COLOR_ALERT_RED if sev == "CRITICAL" else COLOR_WARNING_AMBER
        (tw, th), _ = cv2.getTextSize(f" [!] {msg} ", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (10, alert_y - th - 8), (20 + tw, alert_y + 4), (15, 15, 15), -1)
        cv2.rectangle(canvas, (10, alert_y - th - 8), (20 + tw, alert_y + 4), banner_color, 1)
        cv2.putText(
            canvas,
            f" [!] {msg} ",
            (14, alert_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            banner_color,
            1,
            cv2.LINE_AA,
        )
        alert_y -= th + 14

    return canvas


# ── Preset Spatial Zones ──────────────────────────────────────────────────────
def get_preset_zones(mode: str = "all") -> List[LenzZone]:
    """Provides tested preset zones corresponding to the core Lenz operational modes."""
    if mode == "table":
        return [
            LenzZone(
                zone_id="dining_table_1",
                name="Dining Table 1",
                zone_type="table",
                norm_bbox=(0.10, 0.25, 0.50, 0.85),
                target_classes={"person", "dishes on a tabletop", "bottle", "cup", "rubbish on a tabletop"},
                color=COLOR_SAFE_GREEN,
            ),
            LenzZone(
                zone_id="dining_table_2",
                name="Dining Table 2",
                zone_type="table",
                norm_bbox=(0.55, 0.25, 0.95, 0.85),
                target_classes={"person", "dishes on a tabletop", "bottle", "cup", "rubbish on a tabletop"},
                color=COLOR_SAFE_GREEN,
            ),
        ]
    elif mode == "parking":
        return [
            LenzZone(
                zone_id="bay_A1",
                name="Bay A-1",
                zone_type="parking",
                norm_bbox=(0.08, 0.25, 0.48, 0.85),
                target_classes=VEHICLE_CLASSES,
                color=COLOR_SAFE_GREEN,
            ),
            LenzZone(
                zone_id="bay_A2",
                name="Bay A-2",
                zone_type="parking",
                norm_bbox=(0.52, 0.25, 0.92, 0.85),
                target_classes=VEHICLE_CLASSES,
                color=COLOR_SAFE_GREEN,
            ),
        ]
    elif mode == "restricted":
        return [
            LenzZone(
                zone_id="restricted_perimeter",
                name="Restricted Zone",
                zone_type="restricted",
                norm_bbox=(0.30, 0.20, 0.70, 0.75),
                target_classes={"person"},
                color=COLOR_ALERT_RED,
            )
        ]
    elif mode == "queue":
        return [
            LenzZone(
                zone_id="queue_lane",
                name="Waiting Queue",
                zone_type="queue",
                norm_bbox=(0.20, 0.15, 0.80, 0.85),
                target_classes={"person"},
                color=COLOR_WARNING_AMBER,
            )
        ]
    else:
        # Showcase multi-zone layout
        return [
            LenzZone(
                zone_id="zone_table",
                name="Table Area",
                zone_type="table",
                norm_bbox=(0.05, 0.28, 0.45, 0.82),
                target_classes={"person", "dishes on a tabletop", "bottle", "cup"},
                color=COLOR_SAFE_GREEN,
            ),
            LenzZone(
                zone_id="zone_restricted",
                name="Staff Only",
                zone_type="restricted",
                norm_bbox=(0.52, 0.20, 0.72, 0.50),
                target_classes={"person"},
                color=COLOR_ALERT_RED,
            ),
            LenzZone(
                zone_id="zone_parking",
                name="Loading Bay",
                zone_type="parking",
                norm_bbox=(0.55, 0.55, 0.95, 0.90),
                target_classes=VEHICLE_CLASSES,
                color=COLOR_INFO_BLUE,
            ),
        ]


# ── Diagnostics Exporter ──────────────────────────────────────────────────────
def export_diagnostics(
    frame: np.ndarray,
    annotated_frame: np.ndarray,
    eval_results: Dict[str, Any],
    detector_diag: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Saves annotated diagnostic image and structured JSON output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / "diagnostics_output.jpg"
    json_path = output_dir / "diagnostics.json"

    # Save visual frame
    cv2.imwrite(str(img_path), annotated_frame)

    # Format structured JSON
    detections_summary = []
    for d in eval_results.get("all_raw_detections", []):
        detections_summary.append({
            "track_id": d.track_id,
            "label": d.class_name,
            "confidence": round(float(d.confidence), 4),
            "bbox": [round(float(v), 2) for v in d.bbox],
            "dwell_seconds": round(float(d.dwell_seconds), 2),
            "mask_present": d.mask is not None,
        })

    zones_summary = {}
    for zid, zdata in eval_results.get("zone_states", {}).items():
        zones_summary[zid] = {
            "name": zdata.get("name"),
            "type": zdata.get("type"),
            "state": zdata.get("state"),
            "count": zdata.get("count"),
            "occupancy_duration_s": round(float(zdata.get("occupancy_duration_s", 0.0)), 2),
            "details": zdata.get("details", {}),
        }

    payload = {
        "timestamp": time.time(),
        "frame_width": detector_diag.get("frame_width", frame.shape[1]),
        "frame_height": detector_diag.get("frame_height", frame.shape[0]),
        "detector_model_id": detector_diag.get("detector_model_id"),
        "segmentation_model_id": "facebook/sam2.1-hiera-tiny",
        "clip_model_id": "openai/clip-vit-base-patch32",
        "device": detector_diag.get("device"),
        "is_mock_mode": detector_diag.get("is_mock_mode", False),
        "inference_latency_ms": detector_diag.get("inference_time_ms"),
        "fps_estimate": detector_diag.get("fps_estimate"),
        "raw_candidates_count": detector_diag.get("raw_candidates_count"),
        "filtered_candidates_count": detector_diag.get("filtered_candidates_count"),
        "text_query": detector_diag.get("text_query"),
        "detections": detections_summary,
        "zones": zones_summary,
        "alerts": eval_results.get("active_alerts", []),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return img_path, json_path


# ── Main Prototype Pipeline Execution ─────────────────────────────────────────
def run_lenz_pipeline(
    camera_index: int = 0,
    source_video: Optional[str] = None,
    model_path: Optional[str] = None,
    prompts_path: Optional[str] = None,
    device: str = "auto",
    conf_thresh: float = 0.25,
    mode: str = "all",
    target_classes: Optional[List[str]] = None,
    crowd_limit: int = 4,
    headless: bool = False,
    max_frames: int = 0,
    save_diagnostics: bool = False,
    diagnostics_image: Optional[str] = None,
) -> int:
    """
    Executes the Open-Vocabulary Lenz Computer Vision pipeline.
    """
    print("\n" + "=" * 75)
    print("  LENZ — Open-Vocabulary Intelligent Video Analytics Prototype")
    print("  Models: Grounding DINO (Detection) | SAM 2.1 (Masks) | CLIP (State)")
    print("=" * 75)

    detector_id = resolve_model_path(model_path)
    print(f"[*] Initializing Open-Vocabulary Vision Pipeline on {device.upper()}...")
    print(f"    - Detection Model:      {detector_id}")
    print(f"    - Segmentation Model:   facebook/sam2.1-hiera-tiny")
    print(f"    - Classification Model: openai/clip-vit-base-patch32")

    detector = OpenVocabDetector(
        model_id=detector_id,
        prompts_path=prompts_path,
        device_preference=device,
        box_threshold=conf_thresh,
    )
    detector.load()

    sam = SAMSegmenter(device_preference=device)
    sam.load()

    clip_clf = CLIPTableClassifier(device_preference=device)
    clip_clf.load()

    zones = get_preset_zones(mode)
    rule_engine = LenzRuleEngine(zones=zones, crowd_threshold=crowd_limit)

    # ── Single Still-Image Diagnostic Mode ────────────────────────────────────
    if diagnostics_image:
        print(f"[*] Running diagnostic analysis on input image: {diagnostics_image}")
        img_p = Path(diagnostics_image)
        if not img_p.is_file():
            print(f"[!] ERROR: Diagnostic image file not found: {diagnostics_image}", file=sys.stderr)
            return 1

        frame = cv2.imread(str(img_p))
        if frame is None:
            print(f"[!] ERROR: Could not decode image: {diagnostics_image}", file=sys.stderr)
            return 1

        h, w = frame.shape[:2]
        dets, diag = detector.process_frame(frame, frame_index=0, fps=1.0)
        dets = sam.segment_objects(frame, dets, frame_index=0)
        eval_res = rule_engine.evaluate(dets, w, h, current_time=1.0, table_classifier=clip_clf, frame_raw=frame)

        annotated = render_dashboard(
            frame=frame,
            zones=zones,
            eval_results=eval_res,
            fps=diag.get("fps_estimate", 10.0),
            active_mode=mode,
            conf_thresh=conf_thresh,
            show_tracking=True,
            show_zones=True,
            source_name=diagnostics_image,
            detector_model=detector.model_id,
            segmenter_model=sam.model_id,
            device_name=str(detector.device),
            raw_candidates_count=diag.get("raw_candidates_count", 0),
            filtered_candidates_count=len(dets),
        )

        out_img, out_json = export_diagnostics(frame, annotated, eval_res, diag, Path("evidence"))
        print(f"[+] Diagnostics saved successfully:")
        print(f"    - Frame: {out_img}")
        print(f"    - JSON:  {out_json}")
        return 0

    # ── Video Feed / Live Stream Mode ─────────────────────────────────────────
    source_desc = source_video if source_video else f"Webcam #{camera_index}"
    print(f"[*] Connecting to video feed: {source_desc}")

    def _find_sample_video() -> Optional[str]:
        """Searches common relative paths for the bundled sample.mp4 demo video."""
        search_roots = [
            Path(__file__).resolve().parent.parent.parent / "smartspace-ml" / "sample.mp4",
            Path(__file__).resolve().parent.parent / "sample.mp4",
            Path(__file__).resolve().parent / "sample.mp4",
            Path.cwd() / "smartspace-ml" / "sample.mp4",
            Path.cwd() / "sample.mp4",
        ]
        for p in search_roots:
            if p.is_file():
                return str(p)
        return None

    if source_video:
        if not Path(source_video).exists():
            print(f"[!] ERROR: Source video file not found: {source_video}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(source_video)
    else:
        # Try to open webcam first
        cap = cv2.VideoCapture(camera_index)

        # macOS often returns isOpened()=True but delivers all-black frames when permission
        # is denied. Probe a test frame to detect this case.
        _webcam_ok = False
        if cap.isOpened():
            _ret, _probe = cap.read()
            if _ret and _probe is not None and _probe.mean() > 5.0:
                _webcam_ok = True
                # Re-open cleanly so the first real frame is not lost
                cap.release()
                cap = cv2.VideoCapture(camera_index)
            else:
                cap.release()

        if not _webcam_ok:
            # Webcam denied or black — auto-fallback to sample.mp4
            fallback = _find_sample_video()
            if fallback:
                print(f"[!] Webcam #{camera_index} unavailable (permission denied or no signal).")
                print(f"[*] Auto-loading bundled demo video: {fallback}")
                cap = cv2.VideoCapture(fallback)
                source_desc = f"Demo: {Path(fallback).name}"
            else:
                print(
                    f"\n[!] ERROR: Webcam unavailable and no fallback sample.mp4 found.",
                    file=sys.stderr,
                )
                print(
                    "    Run: python smartspace-ml/make_test_video.py --out smartspace-ml/sample.mp4",
                    file=sys.stderr,
                )
                return 2

    if not cap.isOpened():
        print(f"\n[!] ERROR: Cannot open video source: {source_desc}", file=sys.stderr)
        print("macOS Troubleshooting: Ensure Camera permission is granted in System Settings.", file=sys.stderr)
        return 2

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    stream_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"    [+] Connected: {frame_w}x{frame_h} @ {stream_fps:.1f} FPS")

    frame_count = 0
    start_time = time.time()
    last_diag = {}
    last_eval = {}
    last_annotated = None

    # Determine if the source is a video file (for loop-back support)
    _is_file_source = bool(source_video) or "Demo:" in source_desc
    _display_w, _display_h = min(frame_w, 1280), min(frame_h, 720)  # cap display at 1280x720

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if _is_file_source:
                    # Loop the demo video instead of stopping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        print("[*] Video stream ended and could not loop. Stopping.")
                        break
                else:
                    print("[*] Video stream ended or EOF reached.")
                    break

            # Skip blank/all-black frames (e.g., denied webcam returning empty frames)
            if frame is None or frame.mean() < 3.0:
                continue

            frame_count += 1
            current_time = time.time() - start_time

            # 1. Open-Vocabulary Grounding DINO detection + tracking
            detections, last_diag = detector.process_frame(
                frame=frame,
                frame_index=frame_count,
                fps=stream_fps,
            )

            # 2. SAM Segmentation refinement
            detections = sam.segment_objects(
                frame=frame,
                detections=detections,
                frame_index=frame_count,
            )

            # 3. Spatial Grounding & Rule Engine
            last_eval = rule_engine.evaluate(
                detections=detections,
                frame_w=frame_w,
                frame_h=frame_h,
                current_time=current_time,
                table_classifier=clip_clf,
                frame_raw=frame,
            )

            # 4. Render Telemetry HUD
            last_annotated = render_dashboard(
                frame=frame,
                zones=zones,
                eval_results=last_eval,
                fps=last_diag.get("fps_estimate", stream_fps),
                active_mode=mode,
                conf_thresh=conf_thresh,
                show_tracking=True,
                show_zones=True,
                source_name=source_desc,
                detector_model=detector.model_id,
                segmenter_model=sam.model_id,
                device_name=str(detector.device),
                raw_candidates_count=last_diag.get("raw_candidates_count", 0),
                filtered_candidates_count=len(detections),
            )

            if not headless:
                # Resize for display on HiDPI / Retina screens to prevent tiny/distorted window
                display_frame = (
                    cv2.resize(last_annotated, (_display_w, _display_h))
                    if (last_annotated.shape[1] != _display_w or last_annotated.shape[0] != _display_h)
                    else last_annotated
                )
                cv2.imshow("Lenz — Open-Vocabulary Video Intelligence", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("\n[*] User interrupted execution (Q/ESC). Exiting...")
                    break

            if max_frames > 0 and frame_count >= max_frames:
                print(f"[*] Reached configured limit of {max_frames} frames. Stopping.")
                break

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    if save_diagnostics and last_annotated is not None:
        out_img, out_json = export_diagnostics(frame, last_annotated, last_eval, last_diag, Path("evidence"))
        print(f"[+] Diagnostic output saved: {out_img} and {out_json}")

    print(f"[*] Finished processing {frame_count} frames successfully.")
    return 0


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lenz Open-Vocabulary Video Intelligence Prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--source", type=str, default=None, help="Video file or RTSP stream URL")
    parser.add_argument("--model", type=str, default=None, help="Grounding DINO model ID")
    parser.add_argument("--prompts", type=str, default=None, help="Path to custom detection prompts JSON")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"], help="Hardware device")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "table", "parking", "restricted", "queue"],
        help="Operational mode preset",
    )
    parser.add_argument("--crowd-limit", type=int, default=4, help="Crowd density limit")
    parser.add_argument("--headless", action="store_true", help="Run without GUI preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 = infinite)")
    parser.add_argument("--diagnostics", action="store_true", help="Export diagnostic annotated image and JSON")
    parser.add_argument("--diagnostics-image", type=str, default=None, help="Run diagnostics on a single image and exit")

    args = parser.parse_args()

    exit_code = run_lenz_pipeline(
        camera_index=args.camera,
        source_video=args.source,
        model_path=args.model,
        prompts_path=args.prompts,
        device=args.device,
        conf_thresh=args.conf,
        mode=args.mode,
        crowd_limit=args.crowd_limit,
        headless=args.headless,
        max_frames=args.max_frames,
        save_diagnostics=args.diagnostics,
        diagnostics_image=args.diagnostics_image,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
