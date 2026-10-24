"""
tests/test_lenz_regression.py — Comprehensive regression test suite for Lenz
spatial grounding, occupancy intelligence, and rule engine logic.

Covers all 15 required evaluation cases:
1. Person in valid monitored region
2. Person bbox extending into sky with ground-contact feet outside region
3. Car inside configured parking slot
4. Car outside every monitored parking slot
5. Genuinely empty parking slot
6. Visible table with occupied state
7. Empty and visibly clean table
8. Empty table with visible dirt/dishes
9. Partially occluded or borderline table
10. Person entering and leaving restricted region
11. Person dwelling beyond configured loiter limit
12. Multiple people in crowd polygon exceeding threshold
13. Object present for only 1 or 2 frames (debouncing)
14. Low confidence / ambiguous observation (UNCERTAIN)
15. Empty frame with zero detections
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

# Ensure experiments/webcam_occupancy is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "experiments" / "webcam_occupancy"))
sys.path.insert(0, str(repo_root / "smartspace-ml"))

from webcam_test import LenzRuleEngine, LenzZone, TrackedDetection


# ── Case 1: Person in valid monitored region ──────────────────────────────────
def test_case_01_person_in_valid_monitored_region():
    # Zone: 640x480 pixel space, normalized (0.2, 0.2, 0.8, 0.8) -> x: 128..512, y: 96..384
    zone = LenzZone("zone_general", "Main Lobby", "general", (0.2, 0.2, 0.8, 0.8), {"person"})
    engine = LenzRuleEngine([zone], debounce_window=3)

    # Person with feet at (300, 320), fully inside zone
    person = TrackedDetection(1, "person", 0.92, (260, 160, 340, 320))

    # Evaluate across 3 frames
    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([person], 640, 480, t)

    assert res["zone_states"]["zone_general"]["state"] == "OCCUPIED"
    assert res["zone_states"]["zone_general"]["count"] == 1


# ── Case 2: Person bbox extending into sky but feet outside region ───────────
def test_case_02_person_bbox_in_sky_feet_outside_region():
    # Floor zone: y: 200..350 (pixel), x: 200..400
    zone = LenzZone("floor_zone", "Restricted Area", "restricted", (0.3125, 0.4167, 0.625, 0.7292), {"person"})
    engine = LenzRuleEngine([zone], debounce_window=3)

    # Person standing close to camera:
    # Bounding box covers y: 50..450 (crosses into y=200..350 vertically)
    # BUT ground-contact feet are at bottom_center: (300, 450), which is OUTSIDE floor zone (y max = 350)!
    person_in_front = TrackedDetection(2, "person", 0.90, (250, 50, 350, 450))

    # Feet (300, 450) are outside the zone
    assert zone.contains_point(300, 450, 640, 480) is False

    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([person_in_front], 640, 480, t)

    # Floor zone must NOT be occupied because feet are outside
    assert res["zone_states"]["floor_zone"]["count"] == 0
    assert res["zone_states"]["floor_zone"]["state"] in ("NOT OCCUPIED", "AVAILABLE")


# ── Case 3: Car inside configured parking slot ────────────────────────────────
def test_case_03_car_inside_configured_parking_slot():
    slot = LenzZone("bay_01", "Parking Bay 1", "parking", (0.1, 0.1, 0.5, 0.6), {"car", "truck"})
    engine = LenzRuleEngine([slot], debounce_window=3)

    # Vehicle well-centered inside parking bay (100, 80, 280, 250)
    car = TrackedDetection(10, "car", 0.95, (100, 80, 280, 250))

    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([car], 640, 480, t)

    assert res["zone_states"]["bay_01"]["parking_state"] == "OCCUPIED"
    assert res["zone_states"]["bay_01"]["state"] == "OCCUPIED"


# ── Case 4: Car outside every monitored parking slot ─────────────────────────
def test_case_04_car_outside_every_monitored_parking_slot():
    slot = LenzZone("bay_01", "Parking Bay 1", "parking", (0.1, 0.1, 0.4, 0.4), {"car"})
    engine = LenzRuleEngine([slot], debounce_window=3)

    # Car on driving road in bottom-right corner (450, 350, 600, 460)
    car = TrackedDetection(11, "car", 0.91, (450, 350, 600, 460))

    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([car], 640, 480, t)

    assert res["zone_states"]["bay_01"]["parking_state"] == "AVAILABLE"
    assert res["zone_states"]["bay_01"]["count"] == 0


# ── Case 5: Genuinely empty parking slot ──────────────────────────────────────
def test_case_05_genuinely_empty_parking_slot():
    slot = LenzZone("bay_empty", "Empty Bay", "parking", (0.2, 0.2, 0.6, 0.7), {"car", "truck"})
    engine = LenzRuleEngine([slot], debounce_window=2)

    # Completely empty frame
    for t in [0.0, 0.5]:
        res = engine.evaluate([], 640, 480, t)

    assert res["zone_states"]["bay_empty"]["parking_state"] == "AVAILABLE"
    assert res["zone_states"]["bay_empty"]["count"] == 0


# ── Case 6: Visible table with occupied state ─────────────────────────────────
def test_case_06_visible_table_occupied_state():
    table = LenzZone("table_01", "Dining Table 1", "table", (0.2, 0.2, 0.8, 0.8), {"person", "cup", "laptop"})
    engine = LenzRuleEngine([table], debounce_window=3)

    # Seated guest at table
    person = TrackedDetection(5, "person", 0.94, (200, 180, 300, 340))

    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([person], 640, 480, t)

    assert res["zone_states"]["table_01"]["table_state"] == "occupied"
    assert res["zone_states"]["table_01"]["state"] == "OCCUPIED"


# ── Case 7: Empty and visibly clean table ─────────────────────────────────────
def test_case_07_empty_visibly_clean_table():
    table = LenzZone("table_clean", "Dining Table 2", "table", (0.2, 0.2, 0.8, 0.8), {"person", "cup", "bottle"})
    engine = LenzRuleEngine([table], debounce_window=3)

    # Empty frame (table clear)
    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([], 640, 480, t)

    assert res["zone_states"]["table_clean"]["table_state"] == "unoccupied-clean"
    assert res["zone_states"]["table_clean"]["state"] == "NOT OCCUPIED"
    assert len(res["zone_states"]["table_clean"]["dirty_items"]) == 0


# ── Case 8: Empty table with visible dirt or rubbish ──────────────────────────
def test_case_08_empty_table_with_visible_dirt_or_rubbish():
    table = LenzZone("table_dirty", "Dining Table 3", "table", (0.2, 0.2, 0.8, 0.8), {"person", "cup", "bottle", "plate"})
    engine = LenzRuleEngine([table], debounce_window=3)

    # No people present, but used cup and bowl left on table (300, 250)
    cup = TrackedDetection(20, "cup", 0.88, (280, 220, 320, 270))
    bowl = TrackedDetection(21, "bowl", 0.85, (330, 230, 380, 280))

    for t in [0.0, 0.5, 1.0]:
        res = engine.evaluate([cup, bowl], 640, 480, t)

    assert res["zone_states"]["table_dirty"]["table_state"] == "unoccupied-dirty"
    assert "cup" in res["zone_states"]["table_dirty"]["dirty_items"]
    # Verify cleaning alert triggered
    cleaning_alerts = [a for a in res["active_alerts"] if a["type"] == "TABLE_NEEDS_CLEANING"]
    assert len(cleaning_alerts) == 1
    assert "Dining Table 3" in cleaning_alerts[0]["message"]


# ── Case 9: Partially occluded or borderline table ────────────────────────────
def test_case_09_partially_occluded_or_borderline_table():
    table = LenzZone("table_edge", "Edge Table", "table", (0.2, 0.2, 0.8, 0.8), {"person", "cup"})
    engine = LenzRuleEngine([table], debounce_window=5)

    # Single frame observation (below minimum confirmation threshold)
    person = TrackedDetection(30, "person", 0.60, (200, 180, 280, 320))
    res = engine.evaluate([person], 640, 480, 0.0)

    # With only 1 frame observed in window 5, state must be UNCERTAIN
    assert res["zone_states"]["table_edge"]["table_state"] == "uncertain"


# ── Case 10: Person entering and leaving restricted region ────────────────────
def test_case_10_person_entering_and_leaving_restricted_region():
    restricted = LenzZone("vault", "Security Vault", "restricted", (0.3, 0.3, 0.7, 0.7), {"person"})
    engine = LenzRuleEngine([restricted], debounce_window=2)

    person = TrackedDetection(40, "person", 0.95, (250, 160, 350, 260))  # feet at (300, 260) -> strictly inside 144..336

    # Step 1: Person enters restricted zone (confirmed over 3 frames)
    for t in [1.0, 1.5, 2.0]:
        res_in = engine.evaluate([person], 640, 480, t)

    alerts_in = [a for a in res_in["active_alerts"] if a["type"] == "ZONE_INTRUSION"]
    assert len(alerts_in) == 1
    assert alerts_in[0]["severity"] == "CRITICAL"
    assert "Security Vault" in alerts_in[0]["message"]

    # Step 2: Person leaves (cleared over debounce window)
    for t in [2.5, 3.0, 3.5, 4.0]:
        res_out = engine.evaluate([], 640, 480, t)

    assert res_out["zone_states"]["vault"]["state"] == "NOT OCCUPIED"
    alerts_out = [a for a in res_out["active_alerts"] if a["type"] == "ZONE_INTRUSION"]
    assert len(alerts_out) == 0


# ── Case 11: Person dwelling beyond configured loiter limit ───────────────────
def test_case_11_person_remaining_inside_sensitive_zone_beyond_dwell():
    zone = LenzZone("atrium", "Atrium Corridor", "queue", (0.1, 0.1, 0.9, 0.9), {"person"})
    engine = LenzRuleEngine([zone], loiter_limit_s=15.0)

    # Person has dwelled for 22.0 seconds (> 15.0s limit)
    loitering_person = TrackedDetection(50, "person", 0.92, (300, 200, 380, 360), dwell_seconds=22.0)
    res = engine.evaluate([loitering_person], 640, 480, 25.0)

    loiter_alerts = [a for a in res["active_alerts"] if a["type"] == "SUSPICIOUS_LOITERING"]
    assert len(loiter_alerts) == 1
    assert "Track #50" in loiter_alerts[0]["message"]
    assert "Review required" in loiter_alerts[0]["message"]


# ── Case 12: Multiple people within crowd polygon ─────────────────────────────
def test_case_12_multiple_people_within_crowd_polygon():
    engine = LenzRuleEngine([], crowd_threshold=4)

    crowd = [
        TrackedDetection(101, "person", 0.90, (50, 100, 100, 250)),
        TrackedDetection(102, "person", 0.88, (120, 100, 170, 250)),
        TrackedDetection(103, "person", 0.91, (190, 100, 240, 250)),
        TrackedDetection(104, "person", 0.87, (260, 100, 310, 250)),
    ]

    res = engine.evaluate(crowd, 640, 480, 5.0)
    surge_alerts = [a for a in res["active_alerts"] if a["type"] == "CROWD_SURGE"]
    assert len(surge_alerts) == 1
    assert "4 persons in view" in surge_alerts[0]["message"]


# ── Case 13: Object present for only one or two frames ────────────────────────
def test_case_13_object_present_for_one_or_two_frames():
    zone = LenzZone("test_debounce", "Debounce Zone", "general", (0.2, 0.2, 0.8, 0.8), {"person"})
    engine = LenzRuleEngine([zone], debounce_window=6)

    person = TrackedDetection(60, "person", 0.90, (250, 200, 330, 340))

    # Frame 1: Person appears briefly
    res1 = engine.evaluate([person], 640, 480, 0.0)
    assert res1["zone_states"]["test_debounce"]["state"] == "UNCERTAIN"

    # Frame 2 & 3: Gone immediately (glitch/noise)
    res2 = engine.evaluate([], 640, 480, 0.5)
    res3 = engine.evaluate([], 640, 480, 1.0)

    # Must NOT have transitioned to OCCUPIED
    assert res3["zone_states"]["test_debounce"]["state"] == "NOT OCCUPIED"


# ── Case 14: Detection that cannot be classified confidently ──────────────────
def test_case_14_detection_low_confidence_uncertain():
    slot = LenzZone("slot_border", "Border Slot", "parking", (0.1, 0.1, 0.5, 0.5), {"car"})
    engine = LenzRuleEngine([slot], debounce_window=4)

    # 50/50 alternating history represents ambiguous state
    for t in [0.0, 0.5]:
        engine.evaluate([TrackedDetection(70, "car", 0.5, (90, 90, 250, 200))], 640, 480, t)
    for t in [1.0, 1.5]:
        res = engine.evaluate([], 640, 480, t)

    assert res["zone_states"]["slot_border"]["state"] == "UNCERTAIN"


# ── Case 15: No detections in a frame ─────────────────────────────────────────
def test_case_15_no_detections_in_frame():
    zones = [
        LenzZone("t1", "Table", "table", (0.1, 0.1, 0.4, 0.4), {"person", "cup"}),
        LenzZone("p1", "Parking", "parking", (0.5, 0.1, 0.9, 0.5), {"car"}),
    ]
    engine = LenzRuleEngine(zones, debounce_window=2)

    res = engine.evaluate([], 640, 480, 0.0)

    assert res["total_objects"] == 0
    assert len(res["active_alerts"]) == 0
    assert res["class_counts"] == {}
    assert "t1" in res["zone_states"]
    assert "p1" in res["zone_states"]
