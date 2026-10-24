"""
tests/test_motion.py — Unit tests for MotionScorer.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.detector import Detection, PoseResult
from app.motion import MotionScorer


def _make_person(track_id: int, bbox=(100, 100, 150, 200)) -> Detection:
    return Detection(
        track_id=track_id,
        class_name="person",
        confidence=0.9,
        bbox=bbox,
    )


def test_motion_zero_when_no_persons():
    scorer = MotionScorer("zone_1")
    score = scorer.update(1.0, [], [], None, None)
    assert score == 0.0


def test_stationary_person_low_motion():
    scorer = MotionScorer("zone_1")
    # Same position across multiple frames
    p = _make_person(1, (100, 100, 150, 200))
    grey = np.zeros((300, 300), dtype=np.uint8)

    score = 0.0
    for t in [0.0, 1.0, 2.0, 3.0]:
        score = scorer.update(t, [p], [], None, grey)

    # Displacements are 0 and grey differences are 0
    assert score == 0.0


def test_moving_person_increases_motion():
    scorer = MotionScorer("zone_1")
    grey1 = np.full((300, 300), 50, dtype=np.uint8)
    grey2 = np.full((300, 300), 200, dtype=np.uint8)

    p1 = _make_person(1, (50, 50, 100, 150))
    scorer.update(0.0, [p1], [], None, grey1)

    p2 = _make_person(1, (150, 150, 200, 250))
    score = scorer.update(1.0, [p2], [], None, grey2)

    assert score > 0.1


def test_pose_variance_computation():
    scorer = MotionScorer("zone_1")
    # 17 keypoints with non-zero coordinates
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[5] = [100.0, 100.0, 0.9]  # left shoulder
    kps[6] = [120.0, 100.0, 0.9]  # right shoulder
    kps[9] = [90.0, 150.0, 0.9]   # left wrist
    kps[10] = [130.0, 150.0, 0.9] # right wrist

    pose = PoseResult(track_id=1, keypoints=kps, bbox=(80, 80, 140, 200))
    p = _make_person(1, (80, 80, 140, 200))

    score = scorer.update(0.0, [p], [pose], None, None)
    # Pose variance should contribute
    assert score > 0.0
