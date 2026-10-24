"""
tests/test_incidents.py — Unit tests for rolling frame buffer and evidence packaging.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.incidents import EvidenceSaver, FrameBuffer, _blur_faces, _sha256_file


def test_frame_buffer_push_and_trim():
    buffer = FrameBuffer(buffer_s=5.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    buffer.push(0.0, frame)
    buffer.push(2.0, frame)
    buffer.push(4.0, frame)

    assert len(buffer.get_recent(5)) == 3

    # Push frame at t=10.0 -> older than 10.0 - 5.0 = 5.0 are pruned
    buffer.push(10.0, frame)
    recent = buffer.get_frames_around(10.0)
    assert len(recent) == 1
    assert recent[0][0] == 10.0


def test_evidence_saver(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_DIR", tmp_path / "evidence")
    buffer = FrameBuffer(buffer_s=10.0)

    f1 = np.full((120, 160, 3), 100, dtype=np.uint8)
    f2 = np.full((120, 160, 3), 150, dtype=np.uint8)
    buffer.push(1.0, f1)
    buffer.push(2.0, f2)

    saver = EvidenceSaver(buffer)
    result = saver.save(
        incident_id="inc_test_1",
        event_type="INTRUSION",
        keyframes=[f1, f2],
        video_time=2.0,
        fps=10,
    )

    assert "clip_path" in result
    assert "clip_hash" in result
    assert "snapshot_path" in result

    assert os.path.exists(result["clip_path"])
    assert os.path.exists(result["snapshot_path"])
    assert len(result["clip_hash"]) == 64  # SHA-256 length


def test_blur_faces_setting():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # When BLUR_FACES is False, frame is returned as-is
    res = _blur_faces(frame)
    assert np.array_equal(res, frame)
