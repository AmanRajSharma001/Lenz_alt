"""
app/rules.py — Rule engine: generates candidate events for Gemma verification.

Each rule function returns a list of RuleCandidate objects.
The pipeline calls Gemma for verification and persists confirmed events.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.detector import (
    PURPOSEFUL_OBJECTS,
    SAFETY_OBJECTS,
    UNATTENDED_OBJECTS,
    Detection,
    PoseResult,
)
from app.schemas import EventType, ZoneStateEnum
from app.zones import ZoneConfig, ZoneState

logger = logging.getLogger(__name__)


@dataclass
class RuleCandidate:
    event_type: EventType
    zone_id: str
    zone_name: str
    camera_id: str
    reason: str
    confidence: float
    keyframes: List[np.ndarray] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bbox_distance(a: Detection, b: Detection) -> float:
    """Euclidean distance between centres."""
    ax, ay = a.center
    bx, by = b.center
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# ── Individual rules ──────────────────────────────────────────────────────────

def rule_intrusion(
    cfg: ZoneConfig,
    state: ZoneState,
    persons_in_zone: List[Detection],
    video_time: float,
    frame: np.ndarray,
) -> List[RuleCandidate]:
    if cfg.type != "restricted":
        return []
    if not persons_in_zone:
        return []
    return [
        RuleCandidate(
            event_type=EventType.INTRUSION,
            zone_id=cfg.id,
            zone_name=cfg.name,
            camera_id=cfg.camera_id,
            reason=(
                f"{len(persons_in_zone)} person(s) detected in restricted zone "
                f"'{cfg.name}' at video time {video_time:.1f}s"
            ),
            confidence=0.85,
            keyframes=[frame.copy()],
        )
    ]


def rule_loitering(
    cfg: ZoneConfig,
    state: ZoneState,
    persons_in_zone: List[Detection],
    video_time: float,
    frame: np.ndarray,
) -> List[RuleCandidate]:
    candidates = []
    for p in persons_in_zone:
        if p.track_id < 0:
            continue
        first_seen = state.track_first_seen.get(p.track_id, video_time)
        dwell = video_time - first_seen
        if dwell >= cfg.loiter_s and state.motion_score < settings.IDLE_MOTION_THRESHOLD * 2:
            candidates.append(
                RuleCandidate(
                    event_type=EventType.LOITERING,
                    zone_id=cfg.id,
                    zone_name=cfg.name,
                    camera_id=cfg.camera_id,
                    reason=(
                        f"Track {p.track_id} has been in zone '{cfg.name}' "
                        f"for {dwell:.0f}s (loiter_s={cfg.loiter_s})"
                    ),
                    confidence=0.75,
                    keyframes=[frame.copy()],
                    metadata={"track_id": p.track_id, "dwell_s": dwell},
                )
            )
    return candidates


def rule_crowd_surge(
    cfg: ZoneConfig,
    state: ZoneState,
    persons_in_zone: List[Detection],
    video_time: float,
    frame: np.ndarray,
    prev_count: int,
) -> List[RuleCandidate]:
    n = len(persons_in_zone)
    rate = n - prev_count
    if n > cfg.max_people:
        return [
            RuleCandidate(
                event_type=EventType.CROWD_SURGE,
                zone_id=cfg.id,
                zone_name=cfg.name,
                camera_id=cfg.camera_id,
                reason=(
                    f"Zone '{cfg.name}' has {n} people (max={cfg.max_people}), "
                    f"growth rate={rate}"
                ),
                confidence=0.80,
                keyframes=[frame.copy()],
                metadata={"count": n, "max_people": cfg.max_people, "rate": rate},
            )
        ]
    return []


def rule_in_vain(
    cfg: ZoneConfig,
    state: ZoneState,
    persons_in_zone: List[Detection],
    objects_in_zone: List[Detection],
    video_time: float,
    frame: np.ndarray,
) -> List[RuleCandidate]:
    if state.state not in (ZoneStateEnum.OCCUPIED, ZoneStateEnum.IDLE_OCCUPIED):
        return []
    dwell = video_time - state.state_since
    if dwell < cfg.idle_limit_s:
        return []
    if state.motion_score >= settings.IDLE_MOTION_THRESHOLD:
        return []
    # Check if relevant objects present (evidence of purposeful activity)
    obj_names = {o.class_name for o in objects_in_zone}
    has_purposeful = bool(obj_names & PURPOSEFUL_OBJECTS)
    if has_purposeful:
        return []
    return [
        RuleCandidate(
            event_type=EventType.IN_VAIN,
            zone_id=cfg.id,
            zone_name=cfg.name,
            camera_id=cfg.camera_id,
            reason=(
                f"Zone '{cfg.name}' occupied for {dwell:.0f}s with motion_score="
                f"{state.motion_score:.3f} (< {settings.IDLE_MOTION_THRESHOLD}) "
                "and no purposeful objects nearby."
            ),
            confidence=0.65,
            keyframes=[frame.copy()],
            metadata={"dwell_s": dwell, "motion_score": state.motion_score},
        )
    ]


def rule_needs_cleaning(
    cfg: ZoneConfig,
    state: ZoneState,
    objects_in_zone: List[Detection],
    video_time: float,
    frame: np.ndarray,
    object_seen_since: Dict[str, float],  # class_name → first video_time seen while EMPTY
) -> List[RuleCandidate]:
    # Table-specific grounded cleanliness check
    if cfg.type == "table":
        if state.table_state == "unoccupied-dirty":
            dirty_str = ", ".join(state.dirty_items) if state.dirty_items else "tableware/rubbish"
            return [
                RuleCandidate(
                    event_type=EventType.NEEDS_CLEANING,
                    zone_id=cfg.id,
                    zone_name=cfg.name,
                    camera_id=cfg.camera_id,
                    reason=(
                        f"Dining table '{cfg.name}' is unoccupied-dirty: {dirty_str} left behind."
                    ),
                    confidence=0.80,
                    keyframes=[frame.copy()],
                    metadata={"table_state": "unoccupied-dirty", "items": state.dirty_items},
                )
            ]
        return []

    if state.state != ZoneStateEnum.EMPTY:
        return []
    candidates = []
    for obj in objects_in_zone:
        if obj.class_name not in UNATTENDED_OBJECTS:
            continue
        first = object_seen_since.get(obj.class_name, video_time)
        object_seen_since[obj.class_name] = first
        duration = video_time - first
        if duration >= settings.UNATTENDED_OBJECT_S:
            candidates.append(
                RuleCandidate(
                    event_type=EventType.NEEDS_CLEANING,
                    zone_id=cfg.id,
                    zone_name=cfg.name,
                    camera_id=cfg.camera_id,
                    reason=(
                        f"Object '{obj.class_name}' in empty zone '{cfg.name}' "
                        f"for {duration:.0f}s"
                    ),
                    confidence=0.70,
                    keyframes=[frame.copy()],
                    metadata={"object": obj.class_name, "duration_s": duration},
                )
            )
    # Purge objects no longer in zone
    gone = [k for k in object_seen_since if k not in {o.class_name for o in objects_in_zone}]
    for k in gone:
        del object_seen_since[k]
    return candidates


def rule_parking(
    cfg: ZoneConfig,
    state: ZoneState,
    vehicles_in_zone: List[Detection],
    video_time: float,
    frame: np.ndarray,
) -> List[RuleCandidate]:
    if cfg.type != "parking":
        return []
    if state.parking_state == "OCCUPIED" and vehicles_in_zone:
        v_types = [v.class_name for v in vehicles_in_zone]
        return [
            RuleCandidate(
                event_type=EventType.PARKING_OCCUPIED,
                zone_id=cfg.id,
                zone_name=cfg.name,
                camera_id=cfg.camera_id,
                reason=(
                    f"Parking bay '{cfg.name}' occupied by {', '.join(v_types)}."
                ),
                confidence=0.85,
                keyframes=[frame.copy()],
                metadata={"vehicles": v_types, "count": len(vehicles_in_zone)},
            )
        ]
    return []


def rule_abandoned_object(
    all_objects: List[Detection],
    all_persons: List[Detection],
    video_time: float,
    frame: np.ndarray,
    abandoned_since: Dict[int, float],  # track_id → first video_time seen alone
    cfg: ZoneConfig,
) -> List[RuleCandidate]:
    from app.zones import point_in_polygon
    poly_px = cfg.polygon_px(frame.shape[1], frame.shape[0])

    candidates = []
    current_obj_ids = set()

    for obj in all_objects:
        if obj.class_name not in SAFETY_OBJECTS:
            continue
        if obj.track_id < 0:
            continue
        if not point_in_polygon(obj.bottom_center[0], obj.bottom_center[1], poly_px):
            continue
        current_obj_ids.add(obj.track_id)
        # Check if any person is within 1.5× the object width
        radius = 1.5 * obj.width
        near = any(_bbox_distance(obj, p) < radius for p in all_persons)
        if near:
            abandoned_since.pop(obj.track_id, None)
            continue
        if obj.track_id not in abandoned_since:
            abandoned_since[obj.track_id] = video_time
            continue
        alone_for = video_time - abandoned_since[obj.track_id]
        if alone_for >= settings.UNATTENDED_OBJECT_S:
            candidates.append(
                RuleCandidate(
                    event_type=EventType.ABANDONED_OBJECT,
                    zone_id=cfg.id,
                    zone_name=cfg.name,
                    camera_id=cfg.camera_id,
                    reason=(
                        f"'{obj.class_name}' (track {obj.track_id}) alone "
                        f"for {alone_for:.0f}s"
                    ),
                    confidence=0.80,
                    keyframes=[frame.copy()],
                    metadata={"object": obj.class_name, "track_id": obj.track_id},
                )
            )
    # Clean up stale entries
    gone = [k for k in abandoned_since if k not in current_obj_ids]
    for k in gone:
        del abandoned_since[k]
    return candidates


def rule_fall(
    cfg: ZoneConfig,
    state: ZoneState,
    persons_in_zone: List[Detection],
    poses_in_zone: List[PoseResult],
    video_time: float,
    frame: np.ndarray,
    prev_aspect: Dict[int, float],  # track_id → previous aspect ratio
) -> List[RuleCandidate]:
    candidates = []
    for p in persons_in_zone:
        if p.track_id < 0:
            continue
        ratio = p.aspect_ratio   # > 1 = wide (prone)
        prev = prev_aspect.get(p.track_id, ratio)
        # If transitioned from tall (< 0.7) to wide (> 1.3) and motion score is low
        if prev < 0.7 and ratio > 1.3 and state.motion_score < 0.05:
            candidates.append(
                RuleCandidate(
                    event_type=EventType.FALL,
                    zone_id=cfg.id,
                    zone_name=cfg.name,
                    camera_id=cfg.camera_id,
                    reason=(
                        f"Track {p.track_id} aspect ratio changed "
                        f"{prev:.2f} → {ratio:.2f} (fall candidate)"
                    ),
                    confidence=0.60,
                    keyframes=[frame.copy()],
                    metadata={"track_id": p.track_id, "aspect_ratio": ratio},
                )
            )
        prev_aspect[p.track_id] = ratio
    return candidates


def rule_fight(
    cfg: ZoneConfig,
    state: ZoneState,
    persons_in_zone: List[Detection],
    poses_in_zone: List[PoseResult],
    video_time: float,
    frame: np.ndarray,
    prev_poses: Dict[int, np.ndarray],  # track_id → previous keypoints
) -> List[RuleCandidate]:
    """Detect two people close together with high keypoint velocity."""
    candidates = []
    if len(persons_in_zone) < 2:
        return []

    # Compute pairs that are close together
    for i, pa in enumerate(persons_in_zone):
        for pb in persons_in_zone[i + 1:]:
            dist = _bbox_distance(pa, pb)
            avg_h = (pa.height + pb.height) / 2
            if dist > avg_h:
                continue  # not close enough
            # Check keypoint velocity for at least one of them
            high_vel = False
            for track in [pa, pb]:
                pose = next((pr for pr in poses_in_zone if pr.track_id == track.track_id), None)
                if pose is None or track.track_id < 0:
                    continue
                prev_kp = prev_poses.get(track.track_id)
                if prev_kp is not None:
                    delta = np.linalg.norm(pose.keypoints[:, :2] - prev_kp[:, :2], axis=1)
                    vel = float(np.mean(delta)) / (track.height + 1e-6)
                    if vel > 0.15:
                        high_vel = True
            if high_vel:
                candidates.append(
                    RuleCandidate(
                        event_type=EventType.FIGHT,
                        zone_id=cfg.id,
                        zone_name=cfg.name,
                        camera_id=cfg.camera_id,
                        reason=(
                            f"Tracks {pa.track_id} and {pb.track_id} are {dist:.0f}px apart "
                            "with high keypoint velocity — fight candidate"
                        ),
                        confidence=0.55,
                        keyframes=[frame.copy()],
                    )
                )

    # Update previous poses
    for pr in poses_in_zone:
        if pr.track_id >= 0:
            prev_poses[pr.track_id] = pr.keypoints.copy()

    return candidates
