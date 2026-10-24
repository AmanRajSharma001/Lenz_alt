"""
app/zones.py — Zone loading, point-in-polygon test, and ZoneState machine.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import settings
from app.schemas import ZoneStateEnum, ZoneType

logger = logging.getLogger(__name__)


# ── Point-in-polygon (ray-casting) ───────────────────────────────────────────

def point_in_polygon(px: float, py: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """
    Ray-casting algorithm.  polygon is a sequence of (x, y) tuples
    in the same coordinate space as (px, py).
    Returns True if the point is strictly inside or on the boundary.
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    x, y = px, py
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ── Zone descriptor ───────────────────────────────────────────────────────────

@dataclass
class ZoneConfig:
    id: str
    camera_id: str
    name: str
    type: str                             # ZoneType string
    polygon_norm: List[List[float]]       # normalised [[x,y], …]
    idle_limit_s: float = 300.0
    max_people: int = 10
    loiter_s: float = 180.0
    active: bool = True
    baseline_image_path: Optional[str] = None

    def polygon_px(self, frame_w: int, frame_h: int) -> List[Tuple[float, float]]:
        return [(x * frame_w, y * frame_h) for x, y in self.polygon_norm]


# ── ZoneState machine ─────────────────────────────────────────────────────────

@dataclass
class ZoneState:
    """
    Per-zone state machine.
    All timers use VIDEO time (frame_index / process_fps), not wall-clock.
    """
    zone_id: str
    zone_name: str
    camera_id: str

    state: ZoneStateEnum = ZoneStateEnum.EMPTY
    state_since: float = 0.0           # video time of last transition

    # debounce helpers
    _enter_candidate_since: Optional[float] = field(default=None, repr=False)
    _exit_candidate_since: Optional[float] = field(default=None, repr=False)
    _last_occupied_at: Optional[float] = field(default=None, repr=False)

    # live metrics
    people_count: int = 0
    motion_score: float = 0.0
    objects_in_zone: List[str] = field(default_factory=list)

    # per-track dwell bookkeeping  {track_id: first_seen_video_time}
    track_first_seen: Dict[int, float] = field(default_factory=dict)

    def dwell_seconds(self, video_time: float) -> float:
        return video_time - self.state_since

    def update(
        self,
        video_time: float,
        present_track_ids: List[int],
        people_count: int,
        motion_score: float,
        objects_in_zone: List[str],
        cfg: ZoneConfig,
    ) -> Optional[str]:
        """
        Feed the current frame's observations.
        Returns the new state name if a transition occurred, else None.
        """
        self.people_count = people_count
        self.motion_score = motion_score
        self.objects_in_zone = objects_in_zone

        # update per-track first-seen bookkeeping
        for tid in present_track_ids:
            if tid not in self.track_first_seen:
                self.track_first_seen[tid] = video_time
        # purge gone tracks
        gone = [tid for tid in self.track_first_seen if tid not in present_track_ids]
        for tid in gone:
            del self.track_first_seen[tid]

        occupied_now = people_count > 0
        transition: Optional[str] = None

        if self.state == ZoneStateEnum.EMPTY:
            if occupied_now:
                if self._enter_candidate_since is None:
                    self._enter_candidate_since = video_time
                if video_time - self._enter_candidate_since >= settings.OCCUPY_ENTER_S:
                    transition = self._transition(ZoneStateEnum.OCCUPIED, video_time)
                    self._enter_candidate_since = None
                    self._last_occupied_at = video_time
            else:
                self._enter_candidate_since = None

        elif self.state in (ZoneStateEnum.OCCUPIED, ZoneStateEnum.IDLE_OCCUPIED, ZoneStateEnum.NEEDS_ATTENTION):
            if occupied_now:
                self._last_occupied_at = video_time
                self._exit_candidate_since = None
            else:
                if self._exit_candidate_since is None:
                    self._exit_candidate_since = (
                        self._last_occupied_at if self._last_occupied_at is not None else video_time
                    )
                if video_time - self._exit_candidate_since >= settings.OCCUPY_EXIT_S:
                    transition = self._transition(ZoneStateEnum.EMPTY, video_time)
                    self._exit_candidate_since = None
                    self._last_occupied_at = None

            # Promote to IDLE_OCCUPIED?
            if self.state == ZoneStateEnum.OCCUPIED and occupied_now:
                dwell = video_time - self.state_since
                if (
                    dwell >= cfg.idle_limit_s
                    and motion_score < settings.IDLE_MOTION_THRESHOLD
                ):
                    transition = self._transition(ZoneStateEnum.IDLE_OCCUPIED, video_time)

        return transition

    def _transition(self, new_state: ZoneStateEnum, video_time: float) -> str:
        old = self.state.value
        self.state = new_state
        self.state_since = video_time
        logger.info(
            "Zone %s: %s → %s  (video_time=%.1fs)",
            self.zone_id, old, new_state.value, video_time,
        )
        return new_state.value

    def mark_needs_attention(self, video_time: float) -> None:
        if self.state != ZoneStateEnum.NEEDS_ATTENTION:
            self._transition(ZoneStateEnum.NEEDS_ATTENTION, video_time)

    def max_track_dwell(self, video_time: float) -> float:
        if not self.track_first_seen:
            return 0.0
        return max(video_time - t for t in self.track_first_seen.values())


# ── Zone registry (loaded from DB, reloaded periodically) ────────────────────

class ZoneRegistry:
    """Thread-safe dict of ZoneConfig objects, refreshed from the DB."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._configs: Dict[str, ZoneConfig] = {}
        self._states: Dict[str, ZoneState] = {}
        self._last_reload: float = 0.0

    def reload_if_needed(self, now: float) -> None:
        if now - self._last_reload < settings.ZONE_RELOAD_INTERVAL_S:
            return
        self._last_reload = now
        self._reload()

    def _reload(self) -> None:
        from app.db import list_zones  # avoid circular at module level

        rows = list_zones(self.camera_id)
        new_ids = {r["id"] for r in rows}

        for r in rows:
            zid = r["id"]
            cfg = ZoneConfig(
                id=zid,
                camera_id=r["camera_id"],
                name=r["name"],
                type=r.get("type", "general"),
                polygon_norm=r["polygon"],
                idle_limit_s=r.get("idle_limit_s", settings.IDLE_LIMIT_S),
                max_people=r.get("max_people", settings.MAX_PEOPLE_PER_ZONE),
                loiter_s=r.get("loiter_s", settings.LOITER_S),
                active=bool(r.get("active", True)),
                baseline_image_path=r.get("baseline_image_path"),
            )
            self._configs[zid] = cfg
            if zid not in self._states:
                self._states[zid] = ZoneState(
                    zone_id=zid,
                    zone_name=r["name"],
                    camera_id=self.camera_id,
                )

        # remove deleted
        for old_id in list(self._configs.keys()):
            if old_id not in new_ids:
                del self._configs[old_id]
                self._states.pop(old_id, None)

    @property
    def configs(self) -> Dict[str, ZoneConfig]:
        return self._configs

    @property
    def states(self) -> Dict[str, ZoneState]:
        return self._states

    def add_config(self, cfg: ZoneConfig) -> None:
        self._configs[cfg.id] = cfg
        if cfg.id not in self._states:
            self._states[cfg.id] = ZoneState(
                zone_id=cfg.id,
                zone_name=cfg.name,
                camera_id=cfg.camera_id,
            )
