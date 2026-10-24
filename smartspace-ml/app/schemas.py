"""
app/schemas.py — Pydantic models shared across the application.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Zone ──────────────────────────────────────────────────────────────────────

class ZoneType(str, Enum):
    table = "table"
    queue = "queue"
    restricted = "restricted"
    general = "general"


class ZoneCreate(BaseModel):
    camera_id: str
    name: str
    type: ZoneType = ZoneType.general
    # polygon: list of [x, y] pairs, normalised 0-1
    polygon: List[List[float]]
    idle_limit_s: float = 300.0
    max_people: int = 10
    loiter_s: float = 180.0
    active: bool = True
    baseline_image_path: Optional[str] = None


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[ZoneType] = None
    polygon: Optional[List[List[float]]] = None
    idle_limit_s: Optional[float] = None
    max_people: Optional[int] = None
    loiter_s: Optional[float] = None
    active: Optional[bool] = None
    baseline_image_path: Optional[str] = None


class Zone(ZoneCreate):
    id: str

    model_config = {"from_attributes": True}


# ── Zone state ────────────────────────────────────────────────────────────────

class ZoneStateEnum(str, Enum):
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"
    IDLE_OCCUPIED = "IDLE_OCCUPIED"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"


class LiveZoneState(BaseModel):
    zone_id: str
    zone_name: str
    camera_id: str
    state: ZoneStateEnum
    since: float          # video timestamp (s) of last state change
    dwell_seconds: float
    people_count: int
    motion_score: float
    objects_in_zone: List[str] = Field(default_factory=list)
    snapshot_path: Optional[str] = None


# ── Events ────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    STATE_CHANGE = "STATE_CHANGE"
    IN_VAIN = "IN_VAIN"
    NEEDS_CLEANING = "NEEDS_CLEANING"
    INTRUSION = "INTRUSION"
    LOITERING = "LOITERING"
    ABANDONED_OBJECT = "ABANDONED_OBJECT"
    CROWD_SURGE = "CROWD_SURGE"
    FALL = "FALL"
    FIGHT = "FIGHT"


class Event(BaseModel):
    event_id: str
    camera_id: str
    zone_id: str
    zone_name: str
    type: EventType
    state: ZoneStateEnum
    since: float
    duration_s: float
    people_count: int
    motion_score: float
    confidence: float = 0.0
    reason: str = ""
    snapshot_path: Optional[str] = None
    timestamp: float        # wall-clock UTC unix timestamp


# ── Incidents ─────────────────────────────────────────────────────────────────

class IncidentStatus(str, Enum):
    new = "new"
    reviewed = "reviewed"
    dismissed = "dismissed"
    escalated = "escalated"


class Incident(BaseModel):
    id: str
    camera_id: str
    zone_id: str
    type: str
    timestamp: float
    confidence: float
    reason: str
    clip_path: Optional[str] = None
    clip_hash: Optional[str] = None
    status: IncidentStatus = IncidentStatus.new


class IncidentStatusUpdate(BaseModel):
    status: IncidentStatus


# ── Pipeline control ──────────────────────────────────────────────────────────

class PipelineStartRequest(BaseModel):
    source: str
    camera_id: str = "cam_01"


class PipelineStatus(BaseModel):
    running: bool
    camera_id: Optional[str] = None
    source: Optional[str] = None
    frames_processed: int = 0
    video_time_s: float = 0.0
    fps_actual: float = 0.0


# ── Gemma ─────────────────────────────────────────────────────────────────────

class GemmaVerdictInVain(BaseModel):
    purposeful: bool = False
    reason: str = ""
    confidence: float = 0.0


class GemmaVerdictDirty(BaseModel):
    needs_cleaning: bool = False
    reason: str = ""
    confidence: float = 0.0


class GemmaVerdictIncident(BaseModel):
    is_incident: bool = False
    type: str = ""
    reason: str = ""
    confidence: float = 0.0


class ZoneSuggestResponse(BaseModel):
    suggested_zones: List[Dict[str, Any]]


class NLQueryRequest(BaseModel):
    question: str


class NLQueryResponse(BaseModel):
    question: str
    sql: str
    result: Any
    summary: str
