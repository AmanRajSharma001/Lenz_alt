"""
app/db.py — SQLite schema creation and helper functions.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from app.config import settings

_DB_PATH: Path = settings.DB_PATH


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    id                  TEXT PRIMARY KEY,
    camera_id           TEXT NOT NULL,
    name                TEXT NOT NULL,
    type                TEXT NOT NULL DEFAULT 'general',
    polygon             TEXT NOT NULL,   -- JSON [[x,y],...]
    idle_limit_s        REAL NOT NULL DEFAULT 300,
    max_people          INTEGER NOT NULL DEFAULT 10,
    loiter_s            REAL NOT NULL DEFAULT 180,
    active              INTEGER NOT NULL DEFAULT 1,
    baseline_image_path TEXT,
    created_at          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    camera_id       TEXT NOT NULL,
    zone_id         TEXT NOT NULL,
    zone_name       TEXT NOT NULL,
    type            TEXT NOT NULL,
    state           TEXT NOT NULL,
    since           REAL NOT NULL,
    duration_s      REAL NOT NULL,
    people_count    INTEGER NOT NULL,
    motion_score    REAL NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0,
    reason          TEXT NOT NULL DEFAULT '',
    snapshot_path   TEXT,
    timestamp       REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_zone  ON events(zone_id);
CREATE INDEX IF NOT EXISTS idx_events_type  ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_ts    ON events(timestamp DESC);

CREATE TABLE IF NOT EXISTS incidents (
    id          TEXT PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    zone_id     TEXT NOT NULL,
    type        TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    confidence  REAL NOT NULL,
    reason      TEXT NOT NULL,
    clip_path   TEXT,
    clip_hash   TEXT,
    status      TEXT NOT NULL DEFAULT 'new'
);
"""


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(_SCHEMA)


# ── Zone helpers ──────────────────────────────────────────────────────────────

def insert_zone(zone: Dict[str, Any]) -> str:
    zid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO zones
               (id, camera_id, name, type, polygon, idle_limit_s, max_people,
                loiter_s, active, baseline_image_path, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                zid,
                zone["camera_id"],
                zone["name"],
                zone.get("type", "general"),
                json.dumps(zone["polygon"]),
                zone.get("idle_limit_s", 300),
                zone.get("max_people", 10),
                zone.get("loiter_s", 180),
                1 if zone.get("active", True) else 0,
                zone.get("baseline_image_path"),
                time.time(),
            ),
        )
    return zid


def list_zones(camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        if camera_id:
            rows = conn.execute(
                "SELECT * FROM zones WHERE camera_id=? AND active=1", (camera_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM zones WHERE active=1").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["polygon"] = json.loads(d["polygon"])
        d["active"] = bool(d["active"])
        result.append(d)
    return result


def get_zone(zone_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM zones WHERE id=?", (zone_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["polygon"] = json.loads(d["polygon"])
    d["active"] = bool(d["active"])
    return d


def update_zone(zone_id: str, updates: Dict[str, Any]) -> bool:
    allowed = {
        "name", "type", "polygon", "idle_limit_s", "max_people",
        "loiter_s", "active", "baseline_image_path",
    }
    fields = {k: v for k, v in updates.items() if k in allowed}
    if not fields:
        return False
    if "polygon" in fields:
        fields["polygon"] = json.dumps(fields["polygon"])
    if "active" in fields:
        fields["active"] = 1 if fields["active"] else 0
    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [zone_id]
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE zones SET {set_clause} WHERE id=?", values)
    return cur.rowcount > 0


def delete_zone(zone_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("UPDATE zones SET active=0 WHERE id=?", (zone_id,))
    return cur.rowcount > 0


# ── Event helpers ─────────────────────────────────────────────────────────────

def insert_event(ev: Dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO events
               (event_id,camera_id,zone_id,zone_name,type,state,since,
                duration_s,people_count,motion_score,confidence,reason,
                snapshot_path,timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ev["event_id"],
                ev["camera_id"],
                ev["zone_id"],
                ev["zone_name"],
                ev["type"],
                ev["state"],
                ev["since"],
                ev["duration_s"],
                ev["people_count"],
                ev["motion_score"],
                ev.get("confidence", 0.0),
                ev.get("reason", ""),
                ev.get("snapshot_path"),
                ev["timestamp"],
            ),
        )


def list_events(
    zone_id: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    if zone_id:
        clauses.append("zone_id=?")
        params.append(zone_id)
    if event_type:
        clauses.append("type=?")
        params.append(event_type)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?", params
        ).fetchall()
    return [dict(r) for r in rows]


def run_select(sql: str) -> List[Dict[str, Any]]:
    """Run a read-only SELECT (caller must validate)."""
    with get_conn() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


# ── Incident helpers ──────────────────────────────────────────────────────────

def insert_incident(inc: Dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO incidents
               (id,camera_id,zone_id,type,timestamp,confidence,
                reason,clip_path,clip_hash,status)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                inc["id"],
                inc["camera_id"],
                inc["zone_id"],
                inc["type"],
                inc["timestamp"],
                inc["confidence"],
                inc["reason"],
                inc.get("clip_path"),
                inc.get("clip_hash"),
                inc.get("status", "new"),
            ),
        )


def list_incidents() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_incident_status(incident_id: str, status: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE incidents SET status=? WHERE id=?", (status, incident_id)
        )
    return cur.rowcount > 0
