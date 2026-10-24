"""
tests/test_zones.py — Unit tests for ZoneState debounce logic and point_in_polygon.
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub out env so config loads without .env file
os.environ.setdefault("GEMMA_MODE", "mock")
os.environ.setdefault("OCCUPY_ENTER_S", "2")
os.environ.setdefault("OCCUPY_EXIT_S", "5")
os.environ.setdefault("IDLE_LIMIT_S", "20")
os.environ.setdefault("IDLE_MOTION_THRESHOLD", "0.02")

from app.config import settings

settings.OCCUPY_ENTER_S = 2.0
settings.OCCUPY_EXIT_S = 5.0
settings.IDLE_LIMIT_S = 20.0
settings.IDLE_MOTION_THRESHOLD = 0.02

try:
    import pytest
except ImportError:
    pytest = None

from app.schemas import ZoneStateEnum
from app.zones import ZoneConfig, ZoneState, point_in_polygon


# ── point_in_polygon ──────────────────────────────────────────────────────────

class TestPointInPolygon:
    def _square(self):
        # Square from (0,0) to (100,100)
        return [(0, 0), (100, 0), (100, 100), (0, 100)]

    def test_centre_inside(self):
        assert point_in_polygon(50, 50, self._square()) is True

    def test_corner_inside(self):
        # Near corner, still inside
        assert point_in_polygon(1, 1, self._square()) is True

    def test_outside_right(self):
        assert point_in_polygon(150, 50, self._square()) is False

    def test_outside_top(self):
        assert point_in_polygon(50, -1, self._square()) is False

    def test_triangle_inside(self):
        tri = [(0, 0), (100, 0), (50, 100)]
        assert point_in_polygon(50, 30, tri) is True

    def test_triangle_outside(self):
        tri = [(0, 0), (100, 0), (50, 100)]
        assert point_in_polygon(90, 90, tri) is False

    def test_degenerate_line(self):
        # Less than 3 points → always False
        assert point_in_polygon(50, 50, [(0, 0), (100, 0)]) is False

    def test_normalised_polygon(self):
        # Normalised 0-1 coordinates
        poly = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        assert point_in_polygon(0.5, 0.5, poly) is True
        assert point_in_polygon(0.05, 0.5, poly) is False


# ── ZoneState debounce ────────────────────────────────────────────────────────

def _make_cfg(**overrides) -> ZoneConfig:
    defaults = dict(
        id="z1",
        camera_id="cam_01",
        name="Test Zone",
        type="general",
        polygon_norm=[[0, 0], [1, 0], [1, 1], [0, 1]],
        idle_limit_s=20.0,
        max_people=5,
        loiter_s=30.0,
    )
    defaults.update(overrides)
    return ZoneConfig(**defaults)


def _make_state() -> ZoneState:
    return ZoneState(zone_id="z1", zone_name="Test Zone", camera_id="cam_01")


class TestZoneStateDebounce:
    def test_initial_state_empty(self):
        s = _make_state()
        assert s.state == ZoneStateEnum.EMPTY

    def test_no_transition_before_enter_threshold(self):
        s = _make_state()
        cfg = _make_cfg()
        # Occupy for 1s (OCCUPY_ENTER_S=2 → should NOT transition)
        result = s.update(1.0, [1], 1, 0.1, [], cfg)
        assert result is None
        assert s.state == ZoneStateEnum.EMPTY

    def test_transition_to_occupied_after_enter_threshold(self):
        s = _make_state()
        cfg = _make_cfg()
        s.update(0.5, [1], 1, 0.1, [], cfg)
        result = s.update(2.5, [1], 1, 0.1, [], cfg)   # 2.5 - 0.5 = 2.0 >= OCCUPY_ENTER_S
        assert result == ZoneStateEnum.OCCUPIED.value
        assert s.state == ZoneStateEnum.OCCUPIED

    def test_no_transition_on_brief_absence(self):
        s = _make_state()
        cfg = _make_cfg()
        # Get into OCCUPIED first
        s.update(0.5, [1], 1, 0.1, [], cfg)
        s.update(2.5, [1], 1, 0.1, [], cfg)
        assert s.state == ZoneStateEnum.OCCUPIED
        # Brief absence (2s < OCCUPY_EXIT_S=5)
        result = s.update(4.5, [], 0, 0.0, [], cfg)
        assert result is None
        assert s.state == ZoneStateEnum.OCCUPIED

    def test_transition_to_empty_after_exit_threshold(self):
        s = _make_state()
        cfg = _make_cfg()
        s.update(0.5, [1], 1, 0.1, [], cfg)
        s.update(2.5, [1], 1, 0.1, [], cfg)
        # Absent for > OCCUPY_EXIT_S=5
        result = s.update(8.0, [], 0, 0.0, [], cfg)
        assert result == ZoneStateEnum.EMPTY.value
        assert s.state == ZoneStateEnum.EMPTY

    def test_idle_occupied_promotion(self):
        s = _make_state()
        cfg = _make_cfg(idle_limit_s=5.0)
        # Get to OCCUPIED
        s.update(0.0, [1], 1, 0.1, [], cfg)
        s.update(2.5, [1], 1, 0.1, [], cfg)
        assert s.state == ZoneStateEnum.OCCUPIED
        # Remain with low motion for >idle_limit_s
        result = s.update(8.0, [1], 1, 0.005, [], cfg)  # 8 - 2.5 = 5.5 > 5.0, motion=0.005 < 0.02
        assert result == ZoneStateEnum.IDLE_OCCUPIED.value

    def test_dwell_seconds(self):
        s = _make_state()
        cfg = _make_cfg()
        s.update(0.0, [1], 1, 0.1, [], cfg)
        s.update(2.5, [1], 1, 0.1, [], cfg)
        # After transition, state_since = 2.5
        dwell = s.dwell_seconds(5.0)
        assert abs(dwell - 2.5) < 0.01

    def test_re_entry_after_exit(self):
        s = _make_state()
        cfg = _make_cfg()
        # Occupy then empty
        s.update(0.0, [1], 1, 0.1, [], cfg)
        s.update(2.5, [1], 1, 0.1, [], cfg)
        s.update(8.0, [], 0, 0.0, [], cfg)
        assert s.state == ZoneStateEnum.EMPTY
        # Re-enter
        s.update(9.0, [2], 1, 0.1, [], cfg)
        result = s.update(12.0, [2], 1, 0.1, [], cfg)
        assert result == ZoneStateEnum.OCCUPIED.value

    def test_max_track_dwell(self):
        s = _make_state()
        cfg = _make_cfg()
        s.update(0.0, [1], 1, 0.1, [], cfg)
        s.update(2.5, [1], 1, 0.1, [], cfg)
        # Track 1 first seen at 0.0
        dwell = s.max_track_dwell(10.0)
        assert dwell >= 10.0


if __name__ == "__main__":
    if pytest is not None:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        print("Running tests without pytest runner...")
        total, passed = 0, 0
        for cls in [TestPointInPolygon, TestZoneStateDebounce]:
            instance = cls()
            for attr in dir(instance):
                if attr.startswith("test_"):
                    total += 1
                    try:
                        getattr(instance, attr)()
                        print(f"  PASS: {cls.__name__}.{attr}")
                        passed += 1
                    except Exception as e:
                        print(f"  FAIL: {cls.__name__}.{attr}: {e}")
        print(f"\n{passed}/{total} tests passed.")
        sys.exit(0 if passed == total else 1)
