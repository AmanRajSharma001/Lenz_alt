"""
app/pipeline.py — Main processing loop.

Runs in a background thread.  Reads from a video source, drives the
detector, zone state machine, rule engine, and Gemma verification.
"""
from __future__ import annotations

import logging
import time
import uuid
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from app.config import settings
from app.db import insert_event, insert_incident
from app.detector import Detector, FrameResult
from app.gemma import get_gemma_worker
from app.incidents import EvidenceSaver, FrameBuffer
from app.motion import MotionScorer
from app.rules import (
    RuleCandidate,
    rule_abandoned_object,
    rule_crowd_surge,
    rule_fall,
    rule_fight,
    rule_in_vain,
    rule_intrusion,
    rule_loitering,
    rule_needs_cleaning,
    rule_parking,
)
from app.schemas import EventType, ZoneStateEnum
from app.zones import ZoneConfig, ZoneRegistry

logger = logging.getLogger(__name__)


# ── Telegram alert ────────────────────────────────────────────────────────────

def _send_telegram(text: str, image_path: Optional[str] = None) -> None:
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.info("[ALERT] %s", text)
        return
    try:
        import httpx

        base = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}"
        if image_path:
            with open(image_path, "rb") as f:
                httpx.post(
                    f"{base}/sendPhoto",
                    data={"chat_id": settings.TELEGRAM_CHAT_ID, "caption": text[:1024]},
                    files={"photo": f},
                    timeout=10.0,
                )
        else:
            httpx.post(
                f"{base}/sendMessage",
                json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": text},
                timeout=10.0,
            )
    except Exception as exc:
        logger.warning("Telegram alert failed: %s", exc)


# ── Pipeline class ────────────────────────────────────────────────────

class Pipeline:
    def __init__(self) -> None:
        self.camera_id: str = settings.CAMERA_ID
        self.source: str = settings.VIDEO_SOURCE
        self.frames_processed: int = 0
        self.video_time_s: float = 0.0
        self.running: bool = False
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

        # Live annotated frame (for MJPEG stream)
        self.latest_frame: Optional[np.ndarray] = None
        # Live zone states broadcast callback (set by API)
        self.on_event: Optional[Callable[[Dict[str, Any]], None]] = None

        # Per-pipeline state
        self._detector = Detector()
        self._registry: Optional[ZoneRegistry] = None
        self._motions: Dict[str, MotionScorer] = {}
        self._buffer: Optional[FrameBuffer] = None
        self._saver: Optional[EvidenceSaver] = None
        self._gemma = get_gemma_worker()

        # Per-zone/object rolling state
        self._object_seen_since: Dict[str, Dict[str, float]] = {}   # zone_id → {class_name → vt}
        self._abandoned_since: Dict[int, float] = {}                 # track_id → vt
        self._prev_aspect: Dict[int, float] = {}
        self._prev_poses: Dict[int, np.ndarray] = {}
        self._prev_counts: Dict[str, int] = {}
        self._alert_cooldown: Dict[str, float] = {}                  # key → last alert wall time

    # ── Public control API ────────────────────────────────────────────────────

    def start(self, source: str, camera_id: str) -> None:
        if self.running:
            return
        self.source = source
        self.camera_id = camera_id
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name=f"pipeline-{camera_id}")
        self._thread.start()
        self.running = True
        logger.info("Pipeline started  source=%s  camera=%s", source, camera_id)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.running = False
        logger.info("Pipeline stopped.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        self._detector.load()
        self._registry = ZoneRegistry(self.camera_id)
        self._registry._reload()

        self._buffer = FrameBuffer(settings.EVIDENCE_BUFFER_S)
        self._saver = EvidenceSaver(self._buffer)

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("Cannot open video source: %s", self.source)
            self.running = False
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # We sample 1 out of every N frames
        target_fps = settings.PROCESS_FPS
        skip = max(1, round(src_fps / target_fps))

        frame_index = 0    # index among all source frames
        proc_index = 0     # index of processed frames
        t_loop_start = time.time()

        while not self._stop_event.is_set():
            ret, raw = cap.read()
            if not ret:
                logger.info("End of video / stream lost.")
                break

            frame_index += 1
            if frame_index % skip != 0:
                continue

            # Resize
            h, w = raw.shape[:2]
            if w != settings.FRAME_WIDTH:
                scale = settings.FRAME_WIDTH / w
                raw = cv2.resize(raw, (settings.FRAME_WIDTH, int(h * scale)))

            # Reload zones periodically
            self._registry.reload_if_needed(proc_index / target_fps)

            # Detect & track
            result: FrameResult = self._detector.process_frame(raw, proc_index, target_fps)
            video_time = result.video_time

            self.video_time_s = video_time
            self.frames_processed = proc_index

            # Push frame into rolling buffer
            self._buffer.push(video_time, raw)

            # Convert frame to greyscale for motion scoring
            grey = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

            # ── Per-zone processing ────────────────────────────────────────
            annotated = raw.copy()
            all_events: List[Dict[str, Any]] = []

            for zone_id, cfg in list(self._registry.configs.items()):
                zone_state = self._registry.states[zone_id]

                poly_px = cfg.polygon_px(annotated.shape[1], annotated.shape[0])
                from app.zones import compute_polygon_bbox_overlap, point_in_polygon

                vehicles_in: List[Detection] = []
                tableware_in: List[Detection] = []
                is_uncertain = False

                if cfg.type == "parking":
                    # Grounded parking evaluation using vehicle bounding box overlap
                    for v in result.vehicles:
                        oz, ob = compute_polygon_bbox_overlap(poly_px, v.bbox, annotated.shape[1], annotated.shape[0])
                        if oz >= 0.30 or ob >= 0.35:
                            vehicles_in.append(v)
                        elif oz >= 0.10 or ob >= 0.10:
                            is_uncertain = True
                    persons_in = []
                    objects_in = []
                elif cfg.type == "table":
                    # Grounded dining table evaluation
                    persons_in = [
                        p for p in result.persons
                        if point_in_polygon(p.bottom_center[0], p.bottom_center[1], poly_px)
                        or point_in_polygon(p.center[0], p.center[1], poly_px)
                        or compute_polygon_bbox_overlap(poly_px, p.bbox, annotated.shape[1], annotated.shape[0])[1] >= 0.15
                    ]
                    tableware_in = [
                        o for o in result.tableware
                        if point_in_polygon(o.bottom_center[0], o.bottom_center[1], poly_px)
                        or point_in_polygon(o.center[0], o.center[1], poly_px)
                        or compute_polygon_bbox_overlap(poly_px, o.bbox, annotated.shape[1], annotated.shape[0])[0] >= 0.05
                    ]
                    objects_in = tableware_in
                else:
                    # Grounded floor / restricted / queue / general: strict ground-contact feet test
                    persons_in = [
                        p for p in result.persons
                        if point_in_polygon(p.bottom_center[0], p.bottom_center[1], poly_px)
                    ]
                    objects_in = [
                        o for o in result.objects
                        if point_in_polygon(o.bottom_center[0], o.bottom_center[1], poly_px)
                    ]

                poses_in = [
                    pr for pr in result.poses
                    if any(pr.track_id == p.track_id for p in persons_in)
                ]

                obj_names = [o.class_name for o in objects_in]

                # Motion scoring
                if zone_id not in self._motions:
                    self._motions[zone_id] = MotionScorer(zone_id)
                motion_score = self._motions[zone_id].update(
                    video_time, persons_in, poses_in, None, grey
                )

                # State machine update
                prev_count = self._prev_counts.get(zone_id, 0)
                transition = zone_state.update(
                    video_time=video_time,
                    present_track_ids=[p.track_id for p in persons_in if p.track_id >= 0],
                    people_count=len(persons_in),
                    motion_score=motion_score,
                    objects_in_zone=obj_names,
                    cfg=cfg,
                    vehicle_count=len(vehicles_in),
                    tableware_in_zone=[t.class_name for t in tableware_in],
                    is_uncertain=is_uncertain,
                )
                self._prev_counts[zone_id] = len(persons_in)

                if transition:
                    ev = self._make_event(
                        EventType.STATE_CHANGE, zone_id, cfg, zone_state, video_time
                    )
                    insert_event(ev)
                    all_events.append(ev)

                # ── Rule engine ────────────────────────────────────────────
                candidates: List[RuleCandidate] = []

                candidates += rule_intrusion(cfg, zone_state, persons_in, video_time, raw)
                candidates += rule_loitering(cfg, zone_state, persons_in, video_time, raw)
                candidates += rule_crowd_surge(cfg, zone_state, persons_in, video_time, raw, prev_count)
                candidates += rule_in_vain(cfg, zone_state, persons_in, objects_in, video_time, raw)
                candidates += rule_needs_cleaning(
                    cfg, zone_state, objects_in, video_time, raw,
                    self._object_seen_since.setdefault(zone_id, {})
                )
                candidates += rule_parking(cfg, zone_state, vehicles_in, video_time, raw)
                candidates += rule_fall(cfg, zone_state, persons_in, poses_in, video_time, raw, self._prev_aspect)
                candidates += rule_fight(cfg, zone_state, persons_in, poses_in, video_time, raw, self._prev_poses)

                for c in candidates:
                    self._dispatch_candidate(c, video_time, target_fps)

                # ── Abandoned objects (global, not per-zone) ───────────────
                abandoned = rule_abandoned_object(
                    result.objects, result.persons, video_time, raw, self._abandoned_since, cfg
                )
                for c in abandoned:
                    self._dispatch_candidate(c, video_time, target_fps)

                # ── Annotate zone on frame ─────────────────────────────────
                self._draw_zone(annotated, cfg, zone_state, video_time)

            # Draw detections
            for p in result.persons:
                x1, y1, x2, y2 = (int(v) for v in p.bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 200, 50), 1)
                if p.track_id >= 0:
                    cv2.putText(annotated, f"#{p.track_id}", (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 200, 50), 1)

            # OSD
            vt_str = f"T={video_time:.1f}s  F={proc_index}"
            cv2.putText(annotated, vt_str, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            self.latest_frame = annotated

            # Broadcast events via WS callback
            if self.on_event and all_events:
                for ev in all_events:
                    try:
                        self.on_event(ev)
                    except Exception:
                        pass

            proc_index += 1

        cap.release()
        self.running = False
        elapsed = time.time() - t_loop_start
        logger.info("Pipeline finished. Processed %d frames in %.1fs", proc_index, elapsed)

    # ── Gemma dispatch & confirmation ─────────────────────────────────────────

    def _dispatch_candidate(self, c: RuleCandidate, video_time: float, fps: float) -> None:
        cooldown_key = f"{c.zone_id}:{c.event_type.value}"
        task = "incident"
        if c.event_type == EventType.IN_VAIN:
            task = "in_vain"
        elif c.event_type == EventType.NEEDS_CLEANING:
            task = "dirty_zone"

        metadata = dict(c.metadata)
        metadata.update({
            "event_type": c.event_type.value,
            "zone_id": c.zone_id,
            "zone_name": c.zone_name,
            "video_time": video_time,
            "reason": c.reason,
        })

        def _on_verdict(verdict: Dict[str, Any]) -> None:
            confidence = float(verdict.get("confidence", 0.0))
            if confidence < settings.ALERT_CONFIDENCE:
                return
            # Save as incident
            inc_id = str(uuid.uuid4())
            ev = self._make_event_from_candidate(c, video_time, confidence, verdict)
            insert_event(ev)

            evidence = self._saver.save(inc_id, c.event_type.value, c.keyframes, video_time, fps)
            insert_incident({
                "id": inc_id,
                "camera_id": c.camera_id,
                "zone_id": c.zone_id,
                "type": c.event_type.value,
                "timestamp": time.time(),
                "confidence": confidence,
                "reason": verdict.get("reason", c.reason),
                "clip_path": evidence.get("clip_path"),
                "clip_hash": evidence.get("clip_hash"),
                "status": "new",
            })
            self._maybe_alert(c, confidence, evidence.get("snapshot_path"), verdict)
            if self.on_event:
                try:
                    self.on_event(ev)
                except Exception:
                    pass

        self._gemma.submit(task, c.keyframes, metadata, _on_verdict, cooldown_key=cooldown_key)

    def _maybe_alert(
        self,
        c: RuleCandidate,
        confidence: float,
        snapshot_path: Optional[str],
        verdict: Dict[str, Any],
    ) -> None:
        alert_key = f"{c.zone_id}:{c.event_type.value}:alert"
        now = time.time()
        last = self._alert_cooldown.get(alert_key, 0.0)
        if now - last < settings.ALERT_COOLDOWN_MINUTES * 60:
            return
        self._alert_cooldown[alert_key] = now
        text = (
            f"🚨 SmartSpace Alert\n"
            f"Camera: {c.camera_id}\n"
            f"Zone: {c.zone_name}\n"
            f"Event: {c.event_type.value}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Reason: {verdict.get('reason', c.reason)}"
        )
        _send_telegram(text, snapshot_path)

    # ── Event builders ────────────────────────────────────────────────────────

    def _make_event(
        self,
        etype: EventType,
        zone_id: str,
        cfg: ZoneConfig,
        state,
        video_time: float,
        confidence: float = 0.0,
        reason: str = "",
        snapshot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "event_id": str(uuid.uuid4()),
            "camera_id": self.camera_id,
            "zone_id": zone_id,
            "zone_name": cfg.name,
            "type": etype.value,
            "state": state.state.value,
            "since": state.state_since,
            "duration_s": state.dwell_seconds(video_time),
            "people_count": state.people_count,
            "motion_score": state.motion_score,
            "confidence": confidence,
            "reason": reason,
            "snapshot_path": snapshot_path,
            "timestamp": time.time(),
        }

    def _make_event_from_candidate(
        self,
        c: RuleCandidate,
        video_time: float,
        confidence: float,
        verdict: Dict[str, Any],
    ) -> Dict[str, Any]:
        state = self._registry.states.get(c.zone_id) if self._registry else None
        return {
            "event_id": str(uuid.uuid4()),
            "camera_id": c.camera_id,
            "zone_id": c.zone_id,
            "zone_name": c.zone_name,
            "type": c.event_type.value,
            "state": state.state.value if state else ZoneStateEnum.EMPTY.value,
            "since": state.state_since if state else 0.0,
            "duration_s": (video_time - state.state_since) if state else 0.0,
            "people_count": state.people_count if state else 0,
            "motion_score": state.motion_score if state else 0.0,
            "confidence": confidence,
            "reason": verdict.get("reason", c.reason),
            "snapshot_path": None,
            "timestamp": time.time(),
        }

    # ── Drawing ───────────────────────────────────────────────────────────────

    _STATE_COLOURS = {
        ZoneStateEnum.EMPTY: (0, 200, 0),         # green
        ZoneStateEnum.AVAILABLE: (0, 200, 0),     # green
        ZoneStateEnum.OCCUPIED: (0, 165, 255),     # amber
        ZoneStateEnum.IDLE_OCCUPIED: (0, 0, 220),  # red
        ZoneStateEnum.NEEDS_ATTENTION: (128, 0, 128),  # purple
        ZoneStateEnum.UNCERTAIN: (0, 215, 255),    # yellow
    }

    def _draw_zone(self, frame: np.ndarray, cfg: ZoneConfig, state, video_time: float) -> None:
        poly_px = cfg.polygon_px(frame.shape[1], frame.shape[0])
        pts = np.array([[int(x), int(y)] for x, y in poly_px], dtype=np.int32)
        colour = self._STATE_COLOURS.get(state.state, (200, 200, 200))

        # Check for dirty table warning tint
        if cfg.type == "table" and state.table_state == "unoccupied-dirty":
            colour = (0, 140, 255)  # orange/warning for dirty table

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        dwell = state.dwell_seconds(video_time)
        if cfg.type == "table" and state.table_state:
            badge = state.table_state.upper()
            label = f"{cfg.name} | {badge} | {dwell:.0f}s"
        elif cfg.type == "parking" and state.parking_state:
            badge = state.parking_state
            label = f"{cfg.name} | {badge} | {dwell:.0f}s"
        else:
            label = f"{cfg.name} | {state.state.value} | {dwell:.0f}s | {state.people_count}p"
        cv2.putText(frame, label, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


# ── Global pipeline singleton ─────────────────────────────────────────────────
_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline
