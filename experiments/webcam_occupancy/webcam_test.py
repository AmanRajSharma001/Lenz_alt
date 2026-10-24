#!/usr/bin/env python3
"""
experiments/webcam_occupancy/webcam_test.py

Lenz — Intelligent Video Analytics & Occupancy Prototype
=========================================================
A local, multi-class object detection and spatial analytics prototype implementing
the core Lenz use cases described in the platform technical specification:

  1. Multi-Class Object Detection (Person, Vehicles, Bags, Electronics, Furniture)
  2. Multi-Object Tracking (ByteTrack persistent IDs and dwell time)
  3. Zone-Based Table / Space Occupancy Monitoring
  4. Smart Parking Space Utilization Monitoring
  5. Crowd Density & Overcrowding Estimation
  6. Queue & Waiting-Area Congestion Monitoring
  7. Restricted-Area Intrusion Detection
  8. Rule-Based Loitering & Unattended Object Observations

Usage:
    python webcam_test.py                        # Default built-in webcam
    python webcam_test.py --camera 1             # External USB / Continuity camera
    python webcam_test.py --source video.mp4     # Prerecorded video file
    python webcam_test.py --mode parking         # Focus on parking space monitoring
    python webcam_test.py --mode queue           # Focus on queue & crowd monitoring
    python webcam_test.py --mode table           # Focus on table occupancy
    python webcam_test.py --mode restricted      # Focus on restricted intrusion
    python webcam_test.py --headless             # Headless / CI execution without GUI
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

# Redirect Ultralytics settings to /tmp to prevent macOS sandbox permissions errors
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

import cv2
import numpy as np


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

# Class Category Color Mapping
CLASS_COLORS = {
    # Persons
    "person": (235, 180, 50),       # Cyan / Light Blue
    # Vehicles
    "car": (50, 150, 250),          # Orange
    "motorcycle": (50, 200, 250),   # Light Orange
    "bicycle": (50, 230, 210),      # Yellow-Orange
    "bus": (30, 120, 230),          # Deep Orange
    "truck": (20, 90, 210),         # Rust Orange
    # Bags / Luggage
    "backpack": (210, 80, 220),     # Magenta
    "handbag": (180, 70, 200),      # Purple-Pink
    "suitcase": (160, 50, 180),     # Deep Purple
    # Furniture
    "chair": (180, 180, 100),       # Olive / Teal
    "bench": (160, 160, 90),
    "couch": (190, 150, 110),
    "dining table": (150, 130, 90),
    # Office & Electronics
    "laptop": (255, 140, 0),        # Vivid Blue
    "cell phone": (230, 100, 40),   # Royal Blue
    "mouse": (200, 120, 80),
    "keyboard": (210, 130, 90),
    "tv": (180, 110, 70),
    # Tableware / Food
    "bottle": (80, 210, 120),       # Sea Green
    "cup": (60, 190, 150),
    "wine glass": (70, 180, 170),
    "bowl": (90, 200, 140),
    "book": (120, 160, 220),
}


# ── Semantic Class Groupings ──────────────────────────────────────────────────
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PERSONAL_ITEM_CLASSES = {"backpack", "handbag", "suitcase", "laptop"}
PURPOSEFUL_TABLE_CLASSES = {"laptop", "bottle", "cup", "cell phone", "book", "bowl"}
TABLEWARE_CLASSES = {"cup", "bottle", "wine glass", "bowl", "plate", "fork", "knife", "spoon"}


# ── Model Resolution ──────────────────────────────────────────────────────────
def resolve_model_path(custom_path: Optional[str] = None) -> str:
    """Finds pretrained model weights locally or returns default name."""
    if custom_path and Path(custom_path).exists():
        return str(Path(custom_path).resolve())

    # Preference order: yolo11n.pt -> yolov8n.pt -> fallback name
    candidates = [
        Path(__file__).parent / "yolo11n.pt",
        Path(__file__).parent / "yolov8n.pt",
        Path(__file__).parent.parent.parent / "smartspace-ml" / "yolov8n.pt",
        Path("yolo11n.pt"),
        Path("yolov8n.pt"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    return "yolo11n.pt"


# ── Tracked Detection Entity ──────────────────────────────────────────────────
class TrackedDetection:
    """Represents an individual detected and tracked object."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        dwell_seconds: float = 0.0,
    ) -> None:
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # x1, y1, x2, y2 in pixels
        self.dwell_seconds = dwell_seconds

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return max(1.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


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
        self.zone_type = zone_type  # "table", "parking", "restricted", "queue"
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
    occupancy, smart parking, queue congestion, restricted intrusion, and loitering.
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

        # Temporal history buffers per zone
        self.zone_occupancy_history: Dict[str, Deque[bool]] = {
            z.zone_id: collections.deque(maxlen=debounce_window) for z in zones
        }
        self.zone_person_history: Dict[str, Deque[bool]] = {
            z.zone_id: collections.deque(maxlen=debounce_window) for z in zones
        }
        self.zone_tableware_history: Dict[str, Deque[bool]] = {
            z.zone_id: collections.deque(maxlen=debounce_window) for z in zones
        }

        # Unattended item state: {track_id: first_seen_alone_timestamp}
        self.unattended_candidates: Dict[int, float] = {}

    def evaluate(
        self,
        detections: List[TrackedDetection],
        frame_w: int,
        frame_h: int,
        current_time: float,
    ) -> Dict[str, dict]:
        """
        Executes all active rules. Returns status dictionary and active alerts.
        """
        results: Dict[str, dict] = {
            "zone_states": {},
            "active_alerts": [],
            "class_counts": collections.Counter(d.class_name for d in detections),
            "total_objects": len(detections),
            "disclaimers": [],
        }

        persons = [d for d in detections if d.class_name == "person"]
        vehicles = [d for d in detections if d.class_name in VEHICLE_CLASSES]
        portable_items = [d for d in detections if d.class_name in PERSONAL_ITEM_CLASSES]

        # ── 1. Evaluate Each Configured Zone ──────────────────────────────────
        for zone in self.zones:
            in_zone_dets: List[TrackedDetection] = []

            if zone.zone_type == "parking":
                # Vehicles occupy parking slot if overlap exceeds 30% or det overlap exceeds 35%
                for det in detections:
                    if det.class_name not in zone.target_classes and "all" not in zone.target_classes:
                        continue
                    overlap_zone, overlap_det = zone.compute_overlap(det.bbox, frame_w, frame_h)
                    if overlap_zone >= 0.30 or overlap_det >= 0.35:
                        in_zone_dets.append(det)

                # Temporal smoothing for parking slot
                hist = self.zone_occupancy_history[zone.zone_id]
                hist.append(len(in_zone_dets) > 0)
                true_ratio = sum(1 for h in hist if h) / max(1, len(hist))

                if len(hist) < 2:
                    smoothed_state = "UNCERTAIN"
                    parking_state = "UNCERTAIN"
                elif true_ratio >= 0.60:
                    smoothed_state = "OCCUPIED"
                    parking_state = "OCCUPIED"
                elif true_ratio <= 0.40:
                    smoothed_state = "AVAILABLE"
                    parking_state = "AVAILABLE"
                else:
                    smoothed_state = "UNCERTAIN"
                    parking_state = "UNCERTAIN"

                results["zone_states"][zone.zone_id] = {
                    "name": zone.name,
                    "type": zone.zone_type,
                    "state": smoothed_state,
                    "parking_state": parking_state,
                    "count": len(in_zone_dets),
                    "detections": in_zone_dets,
                }

            elif zone.zone_type == "table":
                # Grounded table condition classification: occupied, unoccupied-clean, unoccupied-dirty
                table_people: List[TrackedDetection] = []
                table_dirty: List[TrackedDetection] = []

                for det in detections:
                    overlap_zone, overlap_det = zone.compute_overlap(det.bbox, frame_w, frame_h)
                    bx, by = det.bottom_center
                    cx, cy = det.center

                    # Person or work item at table
                    if det.class_name == "person" or det.class_name in {"laptop", "book"}:
                        if (
                            zone.contains_point(bx, by, frame_w, frame_h)
                            or (zone.contains_point(cx, cy, frame_w, frame_h) and overlap_det > 0.15)
                            or (overlap_zone > 0.20)
                        ):
                            table_people.append(det)
                            in_zone_dets.append(det)
                    elif det.class_name in TABLEWARE_CLASSES or det.class_name in zone.target_classes:
                        if (
                            zone.contains_point(cx, cy, frame_w, frame_h)
                            or zone.contains_point(bx, by, frame_w, frame_h)
                            or overlap_zone > 0.05
                        ):
                            table_dirty.append(det)
                            in_zone_dets.append(det)

                # Temporal history for table states
                hist_p = self.zone_person_history[zone.zone_id]
                hist_p.append(len(table_people) > 0)
                p_ratio = sum(1 for h in hist_p if h) / max(1, len(hist_p))

                hist_t = self.zone_tableware_history[zone.zone_id]
                hist_t.append(len(table_dirty) > 0)
                t_ratio = sum(1 for h in hist_t if h) / max(1, len(hist_t))

                hist = self.zone_occupancy_history[zone.zone_id]
                hist.append(len(in_zone_dets) > 0)

                if len(hist_p) < 2:
                    smoothed_state = "UNCERTAIN"
                    table_state = "uncertain"
                elif p_ratio >= 0.50:
                    smoothed_state = "OCCUPIED"
                    table_state = "occupied"
                elif t_ratio >= 0.50:
                    smoothed_state = "OCCUPIED"
                    table_state = "unoccupied-dirty"
                else:
                    smoothed_state = "NOT OCCUPIED"
                    table_state = "unoccupied-clean"

                results["zone_states"][zone.zone_id] = {
                    "name": zone.name,
                    "type": zone.zone_type,
                    "state": smoothed_state,
                    "table_state": table_state,
                    "dirty_items": [d.class_name for d in table_dirty],
                    "count": len(table_people),
                    "detections": in_zone_dets,
                }

                if table_state == "unoccupied-dirty":
                    dirty_names = ", ".join(set(d.class_name for d in table_dirty)) or "tableware"
                    results["active_alerts"].append({
                        "severity": "WARNING",
                        "type": "TABLE_NEEDS_CLEANING",
                        "message": f"DIRTY TABLE in '{zone.name}': {dirty_names} left behind",
                        "zone_id": zone.zone_id,
                    })

            else:
                # Floor / Restricted / Queue / General: STRICT ground-contact feet test
                for det in detections:
                    if det.class_name not in zone.target_classes and "all" not in zone.target_classes:
                        continue
                    bx, by = det.bottom_center
                    if zone.contains_point(bx, by, frame_w, frame_h):
                        in_zone_dets.append(det)

                hist = self.zone_occupancy_history[zone.zone_id]
                hist.append(len(in_zone_dets) > 0)
                true_ratio = sum(1 for h in hist if h) / max(1, len(hist))

                min_samples = min(2, self.debounce_window)
                if len(hist) < min_samples:
                    smoothed_state = "UNCERTAIN"
                elif true_ratio >= 0.60:
                    smoothed_state = "OCCUPIED"
                elif true_ratio <= 0.40:
                    smoothed_state = "NOT OCCUPIED"
                else:
                    smoothed_state = "UNCERTAIN"

                results["zone_states"][zone.zone_id] = {
                    "name": zone.name,
                    "type": zone.zone_type,
                    "state": smoothed_state,
                    "count": len(in_zone_dets),
                    "detections": in_zone_dets,
                }

            # ── 2. Use-Case Rules per Zone ────────────────────────────────────

            # Use Case G: Restricted Area Intrusion
            if zone.zone_type == "restricted" and smoothed_state == "OCCUPIED":
                intruder_ids = [str(d.track_id) for d in in_zone_dets if d.track_id >= 0]
                id_str = f" [Tracks: {', '.join(intruder_ids)}]" if intruder_ids else ""
                results["active_alerts"].append({
                    "severity": "CRITICAL",
                    "type": "ZONE_INTRUSION",
                    "message": f"RESTRICTED INTRUSION in '{zone.name}'!{id_str}",
                    "zone_id": zone.zone_id,
                })

            # Use Case F: Queue & Waiting Area Congestion
            if zone.zone_type == "queue":
                long_dwellers = [
                    d for d in in_zone_dets if d.dwell_seconds >= self.queue_dwell_limit_s
                ]
                if long_dwellers or len(in_zone_dets) >= 3:
                    max_d = max([d.dwell_seconds for d in in_zone_dets], default=0.0)
                    results["active_alerts"].append({
                        "severity": "WARNING",
                        "type": "QUEUE_CONGESTION",
                        "message": (
                            f"QUEUE CONGESTION in '{zone.name}': {len(in_zone_dets)} waiting "
                            f"(Max dwell: {max_d:.1f}s)"
                        ),
                        "zone_id": zone.zone_id,
                    })

            # Use Case H: Suspicious Loitering in Sensitive Zone
            if zone.zone_type in ("restricted", "table", "queue"):
                loitering_people = [
                    d for d in in_zone_dets
                    if d.class_name == "person" and d.dwell_seconds >= self.loiter_limit_s
                ]
                for p in loitering_people:
                    results["active_alerts"].append({
                        "severity": "WARNING",
                        "type": "SUSPICIOUS_LOITERING",
                        "message": (
                            f"LOITERING in '{zone.name}': Track #{p.track_id} "
                            f"dwelling for {p.dwell_seconds:.1f}s (Review required)"
                        ),
                        "zone_id": zone.zone_id,
                    })

        # ── 3. Global Use Cases (Crowd Density & Unattended Objects) ───────────

        # Use Case E: Crowd Density & Overcrowding
        if len(persons) >= self.crowd_threshold:
            results["active_alerts"].append({
                "severity": "WARNING",
                "type": "CROWD_SURGE",
                "message": (
                    f"CROWD THRESHOLD REACHED: {len(persons)} persons in view "
                    f"(Threshold: {self.crowd_threshold})"
                ),
                "zone_id": None,
            })

        # Use Case H: Unattended Stationary Personal Objects
        active_item_ids = set()
        for item in portable_items:
            if item.track_id < 0:
                continue
            active_item_ids.add(item.track_id)

            # Check if any person is within proximity radius (2.0x item width)
            proximity_radius = max(80.0, 2.0 * item.width)
            ix, iy = item.center

            has_person_near = any(
                math.hypot(p.center[0] - ix, p.center[1] - iy) < proximity_radius
                for p in persons
            )

            if has_person_near:
                # Reset candidate timer if person is nearby
                self.unattended_candidates.pop(item.track_id, None)
            else:
                # Start tracking alone duration
                if item.track_id not in self.unattended_candidates:
                    self.unattended_candidates[item.track_id] = current_time

                alone_duration = current_time - self.unattended_candidates[item.track_id]
                if alone_duration >= self.unattended_limit_s:
                    results["active_alerts"].append({
                        "severity": "WARNING",
                        "type": "UNATTENDED_OBJECT",
                        "message": (
                            f"UNATTENDED {item.class_name.upper()} #{item.track_id} "
                            f"alone for {alone_duration:.1f}s (Review required)"
                        ),
                        "zone_id": None,
                    })

        # Clean up departed unattended items
        stale_ids = [k for k in self.unattended_candidates if k not in active_item_ids]
        for sid in stale_ids:
            del self.unattended_candidates[sid]

        return results


# ── UI Dashboard & Visualization Renderer ─────────────────────────────────────
def render_dashboard(
    frame: np.ndarray,
    zones: List[LenzZone],
    eval_results: dict,
    fps: float,
    active_mode: str,
    conf_thresh: float,
    show_tracking: bool,
    show_zones: bool,
    source_name: str,
) -> np.ndarray:
    """Renders visual overlays, detections, alert banners, and telemetry HUD."""
    canvas = frame.copy()
    h, w = frame.shape[:2]

    # ── 1. Draw Configured Zones ──────────────────────────────────────────────
    if show_zones:
        for zone in zones:
            # Highlight zone according to state
            z_info = eval_results["zone_states"].get(zone.zone_id, {})
            state = z_info.get("state", "NOT OCCUPIED")

            if state in ("OCCUPIED",):
                color = COLOR_ALERT_RED
            elif state in ("AVAILABLE", "NOT OCCUPIED"):
                color = COLOR_SAFE_GREEN
            else:
                color = COLOR_UNCERTAIN_YELLOW

            zx1, zy1, zx2, zy2 = zone.get_pixel_coords(w, h)

            # Translucent fill
            overlay = canvas.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), color, -1)
            cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, canvas)
            cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), color, 2)

            # Header tag
            badge = f" {zone.name.upper()}: {state} "
            cv2.putText(
                canvas,
                badge,
                (zx1 + 4, max(22, zy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

    # ── 2. Draw Detections & Tracklet IDs ──────────────────────────────────────
    # Flatten all detections present in frame
    all_detections: List[TrackedDetection] = []
    seen_track_ids = set()

    for z_info in eval_results["zone_states"].values():
        for d in z_info.get("detections", []):
            if d.track_id not in seen_track_ids:
                all_detections.append(d)
                if d.track_id >= 0:
                    seen_track_ids.add(d.track_id)

    # Also include any detections that were not inside any zone
    # (provided in eval_results if passed, or from caller)
    raw_dets = eval_results.get("all_raw_detections", all_detections)
    for det in raw_dets:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        box_color = CLASS_COLORS.get(det.class_name, (200, 200, 200))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        # Label badge
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

        # Bottom anchor dot
        bx, by = (int(det.bottom_center[0]), int(det.bottom_center[1]))
        cv2.circle(canvas, (bx, by), 3, box_color, -1)

    # ── 3. Top Master Header & Control Strip ──────────────────────────────────
    hud_h = 76
    cv2.rectangle(canvas, (0, 0), (w, hud_h), COLOR_BG_DARK, -1)
    cv2.line(canvas, (0, hud_h), (w, hud_h), COLOR_BORDER, 1)

    # Lenz Branding & Source
    cv2.putText(
        canvas,
        "LENZ VIDEO INTELLIGENCE",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_TEXT_WHITE,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Source: {source_name} | Mode: {active_mode.upper()} | Conf: {int(conf_thresh * 100)}% | {fps:.1f} FPS",
        (16, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    # Interactive Keyboard Shortcuts Guide
    shortcuts = "Keys: [m]ode | [t]rack | [z]ones | [+/-] conf | [q]uit"
    (sw, _), _ = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(
        canvas,
        shortcuts,
        (max(16, w - sw - 16), 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    # ── 4. Active Alert Banners (Right Side) ───────────────────────────────────
    alerts = eval_results.get("active_alerts", [])
    alert_y = hud_h + 16

    for alert in alerts[:4]:  # Show top 4 active alerts
        severity = alert.get("severity", "INFO")
        banner_color = COLOR_ALERT_RED if severity == "CRITICAL" else COLOR_WARNING_AMBER
        text = f" ! {alert['message']} "

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        bx1 = w - tw - 24
        bx2 = w - 8

        # Draw alert pill
        cv2.rectangle(canvas, (bx1, alert_y), (bx2, alert_y + th + 10), banner_color, -1)
        cv2.putText(
            canvas,
            text,
            (bx1 + 4, alert_y + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
        alert_y += th + 16

    # ── 5. Bottom Telemetry & Multi-Class Summary Bar ──────────────────────────
    footer_h = 42
    cv2.rectangle(canvas, (0, h - footer_h), (w, h), (15, 15, 15), -1)
    cv2.line(canvas, (0, h - footer_h), (w, h - footer_h), COLOR_BORDER, 1)

    # Class Breakdown Summary
    counts = eval_results.get("class_counts", {})
    if counts:
        summary_items = [f"{cls}: {cnt}" for cls, cnt in counts.most_common(7)]
        summary_str = f"Live Detections ({eval_results.get('total_objects', 0)} total): " + "  |  ".join(summary_items)
    else:
        summary_str = "Live Detections: No objects currently detected"

    cv2.putText(
        canvas,
        summary_str,
        (16, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_TEXT_WHITE,
        1,
        cv2.LINE_AA,
    )

    # Transparent Disclaimer
    disclaimer = (
        "* Lenz Prototype: Rule-based behavioral observations. "
        "Does not claim intent inference or 3D posture without depth calibration."
    )
    cv2.putText(
        canvas,
        disclaimer,
        (16, h - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    return canvas


# ── Preset Zone Configurations for Lenz Use Cases ─────────────────────────────
def get_preset_zones(mode: str) -> List[LenzZone]:
    """
    Returns preset spatial zones corresponding to the selected Lenz use case.
    All zones use normalized coordinates (0.0 - 1.0).
    """
    if mode == "table":
        # Focus: Table occupancy & unattended dining monitoring
        return [
            LenzZone(
                zone_id="table_01",
                name="Monitored Table",
                zone_type="table",
                norm_bbox=(0.22, 0.32, 0.78, 0.88),
                target_classes={"person", "laptop", "bottle", "cup", "cell phone", "wine glass", "bowl", "plate", "fork", "knife", "spoon"},
                color=COLOR_SAFE_GREEN,
            )
        ]

    elif mode == "parking":
        # Focus: Smart Parking Bay monitoring
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
        # Focus: Restricted area perimeter intrusion & loitering
        return [
            LenzZone(
                zone_id="restricted_vault",
                name="Restricted Zone",
                zone_type="restricted",
                norm_bbox=(0.30, 0.20, 0.70, 0.75),
                target_classes={"person"},
                color=COLOR_ALERT_RED,
            )
        ]

    elif mode == "queue":
        # Focus: Queue depth & waiting area congestion
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
        # Default "all" mode: Showcase multi-zone layout
        return [
            LenzZone(
                zone_id="zone_table",
                name="Table Area",
                zone_type="table",
                norm_bbox=(0.05, 0.28, 0.45, 0.82),
                target_classes={"person", "laptop", "bottle", "cup"},
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


# ── Main Prototype Pipeline ───────────────────────────────────────────────────
def run_lenz_pipeline(
    camera_index: int = 0,
    source_video: Optional[str] = None,
    model_path: Optional[str] = None,
    conf_thresh: float = 0.35,
    mode: str = "all",
    target_classes: Optional[List[str]] = None,
    crowd_limit: int = 4,
    headless: bool = False,
    max_frames: int = 0,
) -> int:
    """
    Executes the real-time Lenz computer vision pipeline.
    """
    print("\n" + "=" * 70)
    print("  LENZ — Intelligent Video Analytics & Occupancy Prototype")
    print("  'Every Camera. Smarter Insights. Safer Spaces.'")
    print("=" * 70)

    # 1. Load Pretrained YOLO Multi-Class Model
    resolved_model = resolve_model_path(model_path)
    print(f"[*] Loading model weights: {resolved_model}")

    try:
        from ultralytics import YOLO
        model = YOLO(resolved_model)
        supported_names = model.names
        print(f"    [+] Model successfully loaded with {len(supported_names)} pretrained COCO classes.")
    except Exception as exc:
        print(f"\n[!] ERROR: Failed to load detection model: {exc}", file=sys.stderr)
        print("    Ensure ultralytics and torch are installed in your environment:", file=sys.stderr)
        print("        pip install -r requirements.txt\n", file=sys.stderr)
        return 1

    # 2. Setup Video / Webcam Stream
    source_desc = source_video if source_video else f"Webcam #{camera_index}"
    print(f"[*] Initializing video feed: {source_desc}")

    if source_video:
        if not Path(source_video).exists():
            print(f"[!] ERROR: Source video file not found: {source_video}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(source_video)
    else:
        # Open default or selected camera
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[!] ERROR: Cannot open video source: {source_desc}", file=sys.stderr)
        print("-" * 70, file=sys.stderr)
        print("macOS Camera Troubleshooting Steps:", file=sys.stderr)
        print("  1. Permissions: Open 'System Settings' -> 'Privacy & Security' -> 'Camera'", file=sys.stderr)
        print("     and verify that Terminal / iTerm / VS Code / Cursor is checked.", file=sys.stderr)
        print("  2. In-Use: Close any other app using the camera (FaceTime, Zoom, Meet, Photo Booth).", file=sys.stderr)
        print("  3. External Camera: If using iPhone Continuity Camera or external USB webcam,", file=sys.stderr)
        print("     try specifying index 1 or 2 with:  python webcam_test.py --camera 1", file=sys.stderr)
        print("-" * 70 + "\n", file=sys.stderr)
        return 2

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    stream_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"    [+] Stream connected: {frame_w}x{frame_h} @ {stream_fps:.1f} FPS")

    # 3. Initialize Presets, Tracker, and Rule Engine
    zones = get_preset_zones(mode)
    rule_engine = LenzRuleEngine(
        zones=zones,
        crowd_threshold=crowd_limit,
        queue_dwell_limit_s=12.0,
        loiter_limit_s=15.0,
        unattended_limit_s=10.0,
        debounce_window=10,
    )

    # Class filter handling
    active_class_filter: Optional[Set[str]] = None
    if target_classes:
        active_class_filter = set(target_classes)
        print(f"[*] Filtering to {len(active_class_filter)} requested classes: {', '.join(target_classes)}")
    else:
        print(f"[*] Multi-class mode enabled: detecting all {len(supported_names)} supported COCO classes.")

    # Persistent dwell tracker: {track_id: (first_seen_timestamp, last_seen_timestamp)}
    track_dwell_registry: Dict[int, Tuple[float, float]] = {}

    print(f"[*] Active Mode: {mode.upper()} ({len(zones)} zone[s] configured)")
    print(f"[*] ByteTrack multi-object tracking enabled.")
    if not headless:
        print("[*] Displaying interactive GUI. Press 'q' or 'ESC' to exit cleanly.")
    else:
        print("[*] Running in headless mode (no window display).")
    print("-" * 70)

    window_name = "Lenz — Intelligent Video Analytics Prototype"
    frame_index = 0
    t_start = time.time()
    t_prev = time.time()
    fps_smooth = 0.0

    show_tracking = True
    show_zones = True
    current_mode = mode

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if source_video:
                    print("\n[*] Reached end of recorded video source.")
                else:
                    print("\n[!] Frame acquisition failed from camera feed.")
                break

            frame_index += 1
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            fps_smooth = (
                0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-4))
                if fps_smooth > 0
                else 1.0 / max(dt, 1e-4)
            )

            # 4. Multi-Object Tracking & Inference via ByteTrack
            try:
                track_results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=conf_thresh,
                    verbose=False,
                )
            except Exception as trk_err:
                # Fallback to standard prediction if tracking tracker YAML fails
                track_results = model.predict(source=frame, conf=conf_thresh, verbose=False)

            detections: List[TrackedDetection] = []

            if track_results and track_results[0].boxes is not None:
                boxes = track_results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]

                    if active_class_filter and cls_name not in active_class_filter:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    tid = int(box.id[0]) if (box.id is not None) else -1

                    # Update tracklet dwell time
                    dwell_s = 0.0
                    if tid >= 0:
                        if tid not in track_dwell_registry:
                            track_dwell_registry[tid] = (t_now, t_now)
                        else:
                            first_t, _ = track_dwell_registry[tid]
                            track_dwell_registry[tid] = (first_t, t_now)
                            dwell_s = t_now - first_t

                    detections.append(
                        TrackedDetection(
                            track_id=tid,
                            class_name=cls_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            dwell_seconds=dwell_s,
                        )
                    )

            # Purge tracks not seen for > 15 seconds
            stale_tracks = [
                tid for tid, (_, last_t) in track_dwell_registry.items()
                if (t_now - last_t) > 15.0
            ]
            for st in stale_tracks:
                del track_dwell_registry[st]

            # 5. Evaluate Lenz Spatial-Temporal Rules
            eval_results = rule_engine.evaluate(
                detections=detections,
                frame_w=frame.shape[1],
                frame_h=frame.shape[0],
                current_time=t_now,
            )
            eval_results["all_raw_detections"] = detections

            # 6. Render Output Frame
            annotated_frame = render_dashboard(
                frame=frame,
                zones=rule_engine.zones,
                eval_results=eval_results,
                fps=fps_smooth,
                active_mode=current_mode,
                conf_thresh=conf_thresh,
                show_tracking=show_tracking,
                show_zones=show_zones,
                source_name=Path(source_desc).name,
            )

            # Periodic Terminal Telemetry
            if frame_index % 30 == 0 or frame_index == 1:
                counts_str = ", ".join(
                    f"{k}:{v}" for k, v in eval_results["class_counts"].most_common(4)
                ) or "none"
                alerts_str = f" | Alerts: {len(eval_results['active_alerts'])}" if eval_results['active_alerts'] else ""
                print(
                    f"[Frame {frame_index:04d}] Objects: {len(detections):<2} ({counts_str}) "
                    f"| {fps_smooth:.1f} FPS{alerts_str}"
                )

            # 7. GUI Display & Interactive Keyboard Controls
            if not headless:
                try:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                        print("\n[*] Exit requested by user.")
                        break
                    elif key in (ord("m"), ord("M")):
                        # Cycle through modes
                        modes = ["all", "table", "parking", "restricted", "queue"]
                        curr_idx = modes.index(current_mode) if current_mode in modes else 0
                        current_mode = modes[(curr_idx + 1) % len(modes)]
                        rule_engine.zones = get_preset_zones(current_mode)
                        print(f"[*] Switched mode to: {current_mode.upper()}")
                    elif key in (ord("t"), ord("T")):
                        show_tracking = not show_tracking
                        print(f"[*] Tracking display: {'ENABLED' if show_tracking else 'DISABLED'}")
                    elif key in (ord("z"), ord("Z")):
                        show_zones = not show_zones
                        print(f"[*] Zone overlay: {'ENABLED' if show_zones else 'DISABLED'}")
                    elif key in (ord("+"), ord("=")):
                        conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
                        print(f"[*] Confidence threshold increased: {conf_thresh:.2f}")
                    elif key in (ord("-"), ord("_")):
                        conf_thresh = max(0.10, round(conf_thresh - 0.05, 2))
                        print(f"[*] Confidence threshold decreased: {conf_thresh:.2f}")

                except cv2.error as cv_err:
                    print(f"\n[!] Warning: cv2.imshow failed ({cv_err}). Switching to headless mode.")
                    headless = True

            if max_frames and frame_index >= max_frames:
                print(f"\n[*] Reached frame limit ({max_frames}).")
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user (Ctrl+C).")

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        total_elapsed = max(0.001, time.time() - t_start)
        avg_fps = frame_index / total_elapsed
        print(f"[*] Webcam and resources released cleanly.")
        print(f"    Processed {frame_index} frames in {total_elapsed:.1f}s ({avg_fps:.1f} avg FPS).")
        print("=" * 70 + "\n")

    return 0


# ── CLI Interface ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lenz Intelligent Video Analytics & Occupancy Prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (0 = default MacBook FaceTime camera, 1 = external / continuity)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to pre-recorded video file instead of live camera feed",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model weights (default: auto-detects yolo11n.pt / yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence detection threshold (0.05 to 0.95)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "table", "parking", "restricted", "queue"],
        help="Operational mode preset corresponding to Lenz use cases",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated subset of classes to detect (default: all 80 COCO classes)",
    )
    parser.add_argument(
        "--crowd-limit",
        type=int,
        default=4,
        help="Headcount threshold to trigger crowd surge warning",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening GUI preview window (for server/testing environments)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop processing after N frames (0 = run indefinitely)",
    )

    args = parser.parse_args()

    target_classes = None
    if args.classes:
        target_classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]

    exit_code = run_lenz_pipeline(
        camera_index=args.camera,
        source_video=args.source,
        model_path=args.model,
        conf_thresh=args.conf,
        mode=args.mode,
        target_classes=target_classes,
        crowd_limit=args.crowd_limit,
        headless=args.headless,
        max_frames=args.max_frames,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
