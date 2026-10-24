"""
tests/test_db.py — Unit tests for SQLite database operations.
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
import app.db as db


@pytest.fixture(autouse=True)
def setup_temp_db(monkeypatch, tmp_path):
    temp_db = tmp_path / "test_smartspace.db"
    monkeypatch.setattr(db, "_DB_PATH", temp_db)
    monkeypatch.setattr(settings, "DB_PATH", temp_db)
    db.init_db()
    yield temp_db


def test_zone_crud():
    zone_data = {
        "camera_id": "cam_test",
        "name": "Break Room",
        "type": "table",
        "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
        "idle_limit_s": 45.0,
        "max_people": 4,
        "loiter_s": 90.0,
        "active": True,
    }
    zid = db.insert_zone(zone_data)
    assert zid is not None

    # Get zone
    fetched = db.get_zone(zid)
    assert fetched is not None
    assert fetched["name"] == "Break Room"
    assert fetched["camera_id"] == "cam_test"
    assert fetched["polygon"] == zone_data["polygon"]
    assert fetched["active"] is True

    # List zones
    all_zones = db.list_zones("cam_test")
    assert len(all_zones) == 1
    assert all_zones[0]["id"] == zid

    # Filter with non-existent camera
    assert len(db.list_zones("cam_other")) == 0

    # Update zone
    updated = db.update_zone(zid, {"name": "Cafeteria", "max_people": 8})
    assert updated is True
    fetched_after = db.get_zone(zid)
    assert fetched_after["name"] == "Cafeteria"
    assert fetched_after["max_people"] == 8

    # Delete zone (soft delete)
    deleted = db.delete_zone(zid)
    assert deleted is True
    assert len(db.list_zones("cam_test")) == 0


def test_events_crud():
    ev_id = str(uuid.uuid4())
    event_data = {
        "event_id": ev_id,
        "camera_id": "cam_01",
        "zone_id": "zone_100",
        "zone_name": "Lobby",
        "type": "STATE_CHANGE",
        "state": "OCCUPIED",
        "since": 12.0,
        "duration_s": 5.0,
        "people_count": 2,
        "motion_score": 0.15,
        "confidence": 0.85,
        "reason": "Person entered",
        "snapshot_path": "/tmp/snap.jpg",
        "timestamp": 1000.0,
    }
    db.insert_event(event_data)

    events = db.list_events(zone_id="zone_100")
    assert len(events) == 1
    assert events[0]["event_id"] == ev_id
    assert events[0]["zone_name"] == "Lobby"
    assert events[0]["confidence"] == 0.85

    # Filter by non-matching event_type
    assert len(db.list_events(event_type="INTRUSION")) == 0
    # Filter by matching event_type
    assert len(db.list_events(event_type="STATE_CHANGE")) == 1


def test_incidents_crud():
    inc_id = str(uuid.uuid4())
    inc_data = {
        "id": inc_id,
        "camera_id": "cam_01",
        "zone_id": "zone_200",
        "type": "INTRUSION",
        "timestamp": 12345.0,
        "confidence": 0.95,
        "reason": "Restricted zone accessed",
        "clip_path": "/tmp/clip.mp4",
        "clip_hash": "abcdef123456",
        "status": "new",
    }
    db.insert_incident(inc_data)

    incidents = db.list_incidents()
    assert len(incidents) == 1
    assert incidents[0]["id"] == inc_id
    assert incidents[0]["status"] == "new"

    # Update incident status
    updated = db.update_incident_status(inc_id, "reviewed")
    assert updated is True
    incidents_after = db.list_incidents()
    assert incidents_after[0]["status"] == "reviewed"


def test_run_select():
    rows = db.run_select("SELECT 1 AS num")
    assert len(rows) == 1
    assert rows[0]["num"] == 1
