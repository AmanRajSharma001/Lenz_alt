"""
tests/test_api.py — Integration tests for FastAPI endpoints.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
import app.db as db
from app.main import app


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, tmp_path):
    temp_db = tmp_path / "test_api.db"
    monkeypatch.setattr(db, "_DB_PATH", temp_db)
    monkeypatch.setattr(settings, "DB_PATH", temp_db)
    monkeypatch.setattr(settings, "GEMMA_MODE", "mock")
    db.init_db()
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_pipeline_status(client):
    res = client.get("/pipeline/status")
    assert res.status_code == 200
    data = res.json()
    assert "running" in data


def test_zone_crud_endpoints(client):
    # Create zone
    payload = {
        "camera_id": "cam_01",
        "name": "Meeting Room",
        "type": "general",
        "polygon": [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.8]],
        "idle_limit_s": 120.0,
        "max_people": 6,
        "loiter_s": 180.0,
        "active": True,
    }
    create_res = client.post("/zones", json=payload)
    assert create_res.status_code == 201
    zone = create_res.json()
    zid = zone["id"]
    assert zone["name"] == "Meeting Room"

    # List zones
    list_res = client.get("/zones")
    assert list_res.status_code == 200
    zones = list_res.json()
    assert any(z["id"] == zid for z in zones)

    # Get single zone
    get_res = client.get(f"/zones/{zid}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Meeting Room"

    # Update zone
    update_res = client.put(f"/zones/{zid}", json={"name": "Conference Hall"})
    assert update_res.status_code == 200
    assert update_res.json()["name"] == "Conference Hall"

    # Delete zone
    del_res = client.delete(f"/zones/{zid}")
    assert del_res.status_code == 200

    # Ensure zone no longer in active list
    list_after = client.get("/zones").json()
    assert not any(z["id"] == zid for z in list_after)


def test_events_and_incidents_endpoints(client):
    # Seed an event
    ev = {
        "event_id": "ev_test_1",
        "camera_id": "cam_01",
        "zone_id": "z1",
        "zone_name": "Test Zone",
        "type": "STATE_CHANGE",
        "state": "OCCUPIED",
        "since": 10.0,
        "duration_s": 2.0,
        "people_count": 1,
        "motion_score": 0.05,
        "confidence": 0.8,
        "reason": "Test event",
        "timestamp": 1234.0,
    }
    db.insert_event(ev)

    res = client.get("/events")
    assert res.status_code == 200
    events = res.json()
    assert len(events) >= 1
    assert events[0]["event_id"] == "ev_test_1"

    # Seed an incident
    inc = {
        "id": "inc_test_1",
        "camera_id": "cam_01",
        "zone_id": "z1",
        "type": "INTRUSION",
        "timestamp": 1234.0,
        "confidence": 0.9,
        "reason": "Intrusion detected",
        "status": "new",
    }
    db.insert_incident(inc)

    res_inc = client.get("/incidents")
    assert res_inc.status_code == 200
    incs = res_inc.json()
    assert any(i["id"] == "inc_test_1" for i in incs)

    # Patch incident status
    patch_res = client.patch("/incidents/inc_test_1", json={"status": "escalated"})
    assert patch_res.status_code == 200
    assert patch_res.json()["status"] == "escalated"


def test_query_endpoint(client):
    res = client.post("/query", json={"question": "How many incidents occurred today?"})
    assert res.status_code == 200
    data = res.json()
    assert data["question"] == "How many incidents occurred today?"
    assert "summary" in data


def test_suggest_zones_endpoint(client):
    # Create a small dummy image in memory
    img = Image.new("RGB", (320, 240), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("frame.jpg", buf, "image/jpeg")}
    res = client.post("/zones/suggest", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "suggested_zones" in data
    assert len(data["suggested_zones"]) >= 1
