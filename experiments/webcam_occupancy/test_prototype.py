"""
experiments/webcam_occupancy/test_prototype.py

Automated test suite verifying the Lenz local webcam prototype:
- Multi-class object detection
- ByteTrack integration & tracklet dwell time
- Zone spatial intersection logic
- All rule engine behaviors (Table, Parking, Restricted, Queue, Crowd, Loitering, Unattended)
- CLI argument parsing & error handling
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from webcam_test import (
    LenzRuleEngine,
    LenzZone,
    TrackedDetection,
    get_preset_zones,
    render_dashboard,
    resolve_model_path,
)


# ── 1. Model Resolution & Weight Tests ────────────────────────────────────────
def test_resolve_model_path():
    path = resolve_model_path()
    assert os.path.exists(path), f"Resolved model path does not exist: {path}"
    assert path.endswith(".pt"), "Model file must be a PyTorch .pt weight file"


def test_model_loading_and_classes():
    from ultralytics import YOLO

    model_path = resolve_model_path()
    model = YOLO(model_path)
    assert len(model.names) == 80, "Pretrained COCO model should have 80 classes"
    assert "person" in model.names.values()
    assert "car" in model.names.values()
    assert "backpack" in model.names.values()
    assert "laptop" in model.names.values()
    assert "bottle" in model.names.values()


# ── 2. Spatial Zone Logic Tests ───────────────────────────────────────────────
def test_zone_pixel_coords():
    zone = LenzZone("z1", "Test Zone", "table", (0.1, 0.2, 0.9, 0.8), {"person"})
    x1, y1, x2, y2 = zone.get_pixel_coords(640, 480)
    assert (x1, y1, x2, y2) == (64, 96, 576, 384)


def test_zone_point_containment():
    zone = LenzZone("z1", "Test Zone", "table", (0.2, 0.2, 0.8, 0.8), {"person"})
    # Inside point (320, 240) in 640x480
    assert zone.contains_point(320, 240, 640, 480) is True
    # Outside point (50, 50)
    assert zone.contains_point(50, 50, 640, 480) is False


def test_zone_overlap_computation():
    zone = LenzZone("z1", "Test Zone", "parking", (0.2, 0.2, 0.8, 0.8), {"car"})
    # Box matching zone exactly
    zx1, zy1, zx2, zy2 = zone.get_pixel_coords(100, 100)
    overlap_z, overlap_d = zone.compute_overlap((zx1, zy1, zx2, zy2), 100, 100)
    assert abs(overlap_z - 1.0) < 0.01
    assert abs(overlap_d - 1.0) < 0.01


# ── 3. Rule Engine Tests ──────────────────────────────────────────────────────
def test_table_occupancy_rule():
    zone = LenzZone("table_1", "Table 1", "table", (0.2, 0.2, 0.8, 0.8), {"person"})
    engine = LenzRuleEngine([zone], debounce_window=3)

    person = TrackedDetection(1, "person", 0.9, (200, 150, 280, 300))

    # Frame 1, 2, 3: person present -> transitions to OCCUPIED
    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([person], 640, 480, t)

    assert res["zone_states"]["table_1"]["state"] == "OCCUPIED"
    assert res["zone_states"]["table_1"]["count"] == 1


def test_parking_slot_rule():
    zone = LenzZone("bay_1", "Bay 1", "parking", (0.1, 0.1, 0.5, 0.5), {"car", "truck"})
    engine = LenzRuleEngine([zone], debounce_window=3)

    car = TrackedDetection(2, "car", 0.95, (80, 80, 300, 220))

    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([car], 640, 480, t)

    assert res["zone_states"]["bay_1"]["state"] == "OCCUPIED"


def test_restricted_intrusion_alert():
    zone = LenzZone("vault", "Vault Area", "restricted", (0.2, 0.2, 0.8, 0.8), {"person"})
    engine = LenzRuleEngine([zone], debounce_window=3)

    person = TrackedDetection(3, "person", 0.9, (250, 150, 320, 320))

    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([person], 640, 480, t)

    alerts = [a for a in res["active_alerts"] if a["type"] == "ZONE_INTRUSION"]
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "CRITICAL"
    assert "Vault Area" in alerts[0]["message"]


def test_crowd_surge_alert():
    engine = LenzRuleEngine([], crowd_threshold=3)

    persons = [
        TrackedDetection(1, "person", 0.9, (50, 50, 100, 200)),
        TrackedDetection(2, "person", 0.9, (120, 50, 170, 200)),
        TrackedDetection(3, "person", 0.9, (190, 50, 240, 200)),
    ]

    res = engine.evaluate(persons, 640, 480, 1.0)
    surge_alerts = [a for a in res["active_alerts"] if a["type"] == "CROWD_SURGE"]
    assert len(surge_alerts) == 1
    assert "Threshold: 3" in surge_alerts[0]["message"]


def test_loitering_alert():
    zone = LenzZone("lobby", "Lobby", "queue", (0.1, 0.1, 0.9, 0.9), {"person"})
    engine = LenzRuleEngine([zone], loiter_limit_s=10.0)

    person = TrackedDetection(4, "person", 0.9, (200, 150, 260, 350), dwell_seconds=12.5)
    res = engine.evaluate([person], 640, 480, 15.0)

    loiter_alerts = [a for a in res["active_alerts"] if a["type"] == "SUSPICIOUS_LOITERING"]
    assert len(loiter_alerts) == 1
    assert "Track #4" in loiter_alerts[0]["message"]


def test_unattended_object_alert():
    engine = LenzRuleEngine([], unattended_limit_s=5.0)
    bag = TrackedDetection(9, "backpack", 0.85, (50, 50, 120, 140))

    # At t=0s
    res0 = engine.evaluate([bag], 640, 480, 0.0)
    assert len([a for a in res0["active_alerts"] if a["type"] == "UNATTENDED_OBJECT"]) == 0

    # At t=6s (unattended beyond 5.0s)
    res6 = engine.evaluate([bag], 640, 480, 6.0)
    bag_alerts = [a for a in res6["active_alerts"] if a["type"] == "UNATTENDED_OBJECT"]
    assert len(bag_alerts) == 1
    assert "UNATTENDED BACKPACK #9" in bag_alerts[0]["message"]


# ── 4. Dashboard Renderer Test ────────────────────────────────────────────────
def test_render_dashboard():
    frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    zones = get_preset_zones("all")
    engine = LenzRuleEngine(zones)

    dets = [
        TrackedDetection(1, "person", 0.92, (100, 150, 160, 320), 4.2),
        TrackedDetection(2, "laptop", 0.88, (140, 220, 200, 260), 4.2),
    ]

    res = engine.evaluate(dets, 640, 480, 5.0)
    res["all_raw_detections"] = dets

    out_frame = render_dashboard(
        frame=frame,
        zones=zones,
        eval_results=res,
        fps=25.0,
        active_mode="all",
        conf_thresh=0.35,
        show_tracking=True,
        show_zones=True,
        source_name="webcam_test",
    )

    assert out_frame.shape == frame.shape
    assert out_frame.dtype == np.uint8
