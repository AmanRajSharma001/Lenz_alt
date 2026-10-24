"""
tests/test_rules.py — Unit tests for rule engine logic.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.detector import Detection, PoseResult
from app.rules import (
    rule_abandoned_object,
    rule_crowd_surge,
    rule_fall,
    rule_fight,
    rule_in_vain,
    rule_intrusion,
    rule_loitering,
    rule_needs_cleaning,
)
from app.schemas import EventType, ZoneStateEnum
from app.zones import ZoneConfig, ZoneState


def _make_cfg(**overrides) -> ZoneConfig:
    defaults = dict(
        id="z1",
        camera_id="cam_01",
        name="Test Zone",
        type="general",
        polygon_norm=[[0, 0], [1, 0], [1, 1], [0, 1]],
        idle_limit_s=20.0,
        max_people=3,
        loiter_s=30.0,
    )
    defaults.update(overrides)
    return ZoneConfig(**defaults)


def _make_state() -> ZoneState:
    return ZoneState(zone_id="z1", zone_name="Test Zone", camera_id="cam_01")


def _make_person(track_id: int, bbox=(10, 10, 50, 100)) -> Detection:
    return Detection(
        track_id=track_id,
        class_name="person",
        confidence=0.9,
        bbox=bbox,
    )


def test_rule_intrusion_restricted():
    cfg = _make_cfg(type="restricted")
    state = _make_state()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    persons = [_make_person(1)]

    candidates = rule_intrusion(cfg, state, persons, 10.0, frame)
    assert len(candidates) == 1
    assert candidates[0].event_type == EventType.INTRUSION
    assert candidates[0].confidence == 0.85

    # Non-restricted should return empty
    cfg_general = _make_cfg(type="general")
    assert rule_intrusion(cfg_general, state, persons, 10.0, frame) == []


def test_rule_loitering():
    cfg = _make_cfg(loiter_s=20.0)
    state = _make_state()
    state.motion_score = 0.01  # low motion
    state.track_first_seen[1] = 5.0
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    persons = [_make_person(1)]

    # Dwell = 30.0 - 5.0 = 25.0 >= loiter_s (20.0)
    candidates = rule_loitering(cfg, state, persons, 30.0, frame)
    assert len(candidates) == 1
    assert candidates[0].event_type == EventType.LOITERING
    assert candidates[0].metadata["dwell_s"] == 25.0

    # Before loiter_s
    candidates_early = rule_loitering(cfg, state, persons, 15.0, frame)
    assert len(candidates_early) == 0


def test_rule_crowd_surge():
    cfg = _make_cfg(max_people=2)
    state = _make_state()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    persons = [_make_person(1), _make_person(2), _make_person(3)]

    candidates = rule_crowd_surge(cfg, state, persons, 10.0, frame, prev_count=1)
    assert len(candidates) == 1
    assert candidates[0].event_type == EventType.CROWD_SURGE
    assert candidates[0].metadata["count"] == 3


def test_rule_in_vain():
    cfg = _make_cfg(idle_limit_s=10.0)
    state = _make_state()
    state.state = ZoneStateEnum.OCCUPIED
    state.state_since = 0.0
    state.motion_score = 0.005  # below threshold
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Occupied for 15s without purposeful objects
    candidates = rule_in_vain(cfg, state, [_make_person(1)], [], 15.0, frame)
    assert len(candidates) == 1
    assert candidates[0].event_type == EventType.IN_VAIN

    # With purposeful object (laptop) -> no event
    laptop = Detection(track_id=-1, class_name="laptop", confidence=0.8, bbox=(20, 20, 40, 40))
    candidates_with_obj = rule_in_vain(cfg, state, [_make_person(1)], [laptop], 15.0, frame)
    assert len(candidates_with_obj) == 0


def test_rule_needs_cleaning():
    cfg = _make_cfg()
    state = _make_state()
    state.state = ZoneStateEnum.EMPTY
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cup = Detection(track_id=-1, class_name="cup", confidence=0.8, bbox=(20, 20, 40, 40))
    seen_since = {}

    # First seen at t=0
    candidates1 = rule_needs_cleaning(cfg, state, [cup], 0.0, frame, seen_since)
    assert len(candidates1) == 0

    # At t=130s (> UNATTENDED_OBJECT_S=120)
    candidates2 = rule_needs_cleaning(cfg, state, [cup], 130.0, frame, seen_since)
    assert len(candidates2) == 1
    assert candidates2[0].event_type == EventType.NEEDS_CLEANING


def test_rule_abandoned_object():
    cfg = _make_cfg()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    backpack = Detection(track_id=10, class_name="backpack", confidence=0.8, bbox=(50, 50, 80, 80))
    # No persons nearby
    abandoned_since = {}

    rule_abandoned_object([backpack], [], 0.0, frame, abandoned_since, cfg)
    assert 10 in abandoned_since

    # After UNATTENDED_OBJECT_S (120s)
    candidates = rule_abandoned_object([backpack], [], 130.0, frame, abandoned_since, cfg)
    assert len(candidates) == 1
    assert candidates[0].event_type == EventType.ABANDONED_OBJECT


def test_rule_fall():
    cfg = _make_cfg()
    state = _make_state()
    state.motion_score = 0.02
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    # Prev aspect ratio < 0.7 (standing tall: width 40, height 100)
    prev_aspect = {1: 0.4}
    # Current aspect ratio > 1.3 (prone: width 150, height 50)
    person_prone = _make_person(1, (10, 10, 160, 60))

    candidates = rule_fall(cfg, state, [person_prone], [], 5.0, frame, prev_aspect)
    assert len(candidates) == 1
    assert candidates[0].event_type == EventType.FALL


def test_rule_fight():
    cfg = _make_cfg()
    state = _make_state()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    pa = _make_person(1, (50, 50, 90, 150))
    pb = _make_person(2, (70, 50, 110, 150))

    kps1 = np.zeros((17, 3), dtype=np.float32)
    kps2 = np.ones((17, 3), dtype=np.float32) * 50.0  # high movement

    pose1 = PoseResult(track_id=1, keypoints=kps2, bbox=pa.bbox)
    prev_poses = {1: kps1}

    candidates = rule_fight(cfg, state, [pa, pb], [pose1], 5.0, frame, prev_poses)
    assert len(candidates) == 1
    assert candidates[0].event_type == EventType.FIGHT
