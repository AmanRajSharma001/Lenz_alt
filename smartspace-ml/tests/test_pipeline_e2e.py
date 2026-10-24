"""
tests/test_pipeline_e2e.py — End-to-end test for Detector and Pipeline.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
import app.db as db
from app.detector import Detector
from app.pipeline import Pipeline


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch, tmp_path):
    temp_db = tmp_path / "test_pipeline.db"
    monkeypatch.setattr(db, "_DB_PATH", temp_db)
    monkeypatch.setattr(settings, "DB_PATH", temp_db)
    monkeypatch.setattr(settings, "GEMMA_MODE", "mock")
    db.init_db()
    yield


def test_detector_process_frame():
    detector = Detector()
    detector.load()

    # Frame with dark background
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    res = detector.process_frame(frame, frame_index=0, fps=5.0)

    assert res.frame_index == 0
    assert res.video_time == 0.0
    assert isinstance(res.persons, list)
    assert isinstance(res.objects, list)


def test_pipeline_on_sample_video():
    # Verify sample.mp4 exists
    video_path = os.path.join(os.path.dirname(__file__), "..", "sample.mp4")
    if not os.path.exists(video_path):
        pytest.skip("sample.mp4 not found")

    p = Pipeline()
    # Ensure zones are loaded in DB
    db.insert_zone({
        "camera_id": "cam_e2e",
        "name": "E2E Test Zone",
        "type": "general",
        "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "idle_limit_s": 5.0,
        "max_people": 5,
        "loiter_s": 10.0,
        "active": True,
    })

    # Run for 2 seconds
    p.start(source=video_path, camera_id="cam_e2e")
    assert p.running is True

    import time
    time.sleep(1.5)
    p.stop()
    assert p.running is False
    assert p.frames_processed > 0
