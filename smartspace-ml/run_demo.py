#!/usr/bin/env python3
"""
run_demo.py — CLI demo: process a local video file, write an annotated output,
and print a summary table of zone states and events.

Usage:
    python run_demo.py --video sample.mp4 --out out.mp4 [--zones zones.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

# ── Environment defaults for demo ──────────────────────────────────────────────
os.environ.setdefault("GEMMA_MODE", "mock")
os.environ.setdefault("PROCESS_FPS", "5")
os.environ.setdefault("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
from app.config import settings
from app.db import init_db, insert_event, insert_zone, list_zones
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
)
from app.schemas import EventType, ZoneStateEnum
from app.zones import ZoneConfig, ZoneRegistry, point_in_polygon

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demo")


# ── Helpers ───────────────────────────────────────────────────────────────────

_STATE_COLOURS = {
    ZoneStateEnum.EMPTY: (0, 200, 0),
    ZoneStateEnum.OCCUPIED: (0, 165, 255),
    ZoneStateEnum.IDLE_OCCUPIED: (0, 0, 220),
    ZoneStateEnum.NEEDS_ATTENTION: (128, 0, 128),
}


def _draw_zone(frame: np.ndarray, cfg: ZoneConfig, state, video_time: float) -> None:
    poly_px = cfg.polygon_px(frame.shape[1], frame.shape[0])
    pts = np.array([[int(x), int(y)] for x, y in poly_px], dtype=np.int32)
    colour = _STATE_COLOURS.get(state.state, (200, 200, 200))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    dwell = state.dwell_seconds(video_time)
    label = f"{cfg.name} | {state.state.value} | {dwell:.0f}s | {state.people_count}p"
    cv2.putText(frame, label, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


def _load_zones_from_json(path: str, camera_id: str) -> None:
    """Insert zones from a JSON file into the DB (idempotent by name)."""
    existing = {z["name"] for z in list_zones(camera_id)}
    with open(path) as f:
        zones = json.load(f)
    for z in zones:
        z.setdefault("camera_id", camera_id)
        if z["name"] not in existing:
            insert_zone(z)
            logger.info("Loaded zone: %s", z["name"])


def _print_summary(registry: ZoneRegistry, events: list, video_time: float) -> None:
    print("\n" + "=" * 70)
    print(f"{'ZONE STATE SUMMARY':^70}")
    print("=" * 70)
    print(f"{'Zone':<25} {'State':<20} {'Dwell(s)':<12} {'People':<8} {'Motion'}")
    print("-" * 70)
    for zid, state in registry.states.items():
        print(
            f"{state.zone_name:<25} {state.state.value:<20} "
            f"{state.dwell_seconds(video_time):<12.1f} {state.people_count:<8} "
            f"{state.motion_score:.3f}"
        )

    print("\n" + "=" * 70)
    print(f"{'EVENTS ({} total)'.format(len(events)):^70}")
    print("=" * 70)
    if not events:
        print("  (none)")
    else:
        print(f"{'Type':<22} {'Zone':<20} {'VT(s)':<8} {'Conf'}")
        print("-" * 70)
        for ev in events[-20:]:
            print(
                f"{ev['type']:<22} {ev['zone_name']:<20} "
                f"{ev.get('since', 0):<8.1f} {ev.get('confidence', 0):.2f}"
            )
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SmartSpace ML Demo")
    parser.add_argument("--video", default="sample.mp4", help="Input video file")
    parser.add_argument("--out", default="out.mp4", help="Annotated output video")
    parser.add_argument("--zones", default="zones.json", help="Zone definitions JSON")
    parser.add_argument("--camera-id", default="cam_01")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    # Validate input
    if not Path(args.video).exists():
        sys.exit(f"ERROR: Video file not found: {args.video}\n"
                 "  Generate a synthetic test video with:  python make_test_video.py")

    # Bootstrap DB
    init_db()

    # Load zones
    if Path(args.zones).exists():
        _load_zones_from_json(args.zones, args.camera_id)
    else:
        logger.warning("zones.json not found — running with no zones.")

    # Build components
    detector = Detector()
    detector.load()
    registry = ZoneRegistry(args.camera_id)
    registry._reload()
    motions: dict = {}
    buffer = FrameBuffer(settings.EVIDENCE_BUFFER_S)
    saver = EvidenceSaver(buffer)
    gemma_worker = get_gemma_worker()

    # Per-zone rule state
    object_seen_since: dict = {}
    abandoned_since: dict = {}
    prev_aspect: dict = {}
    prev_poses: dict = {}
    prev_counts: dict = {}
    all_events: list = []

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = settings.PROCESS_FPS
    skip = max(1, round(src_fps / target_fps))

    # Output writer
    out_w = settings.FRAME_WIDTH
    out_h = int(src_h * (out_w / src_w))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, target_fps, (out_w, out_h))

    logger.info(
        "Processing %s  src_fps=%.1f  target_fps=%d  skip=%d",
        args.video, src_fps, target_fps, skip,
    )

    frame_index = 0
    proc_index = 0
    t0 = time.time()

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % skip != 0:
            continue
        if args.max_frames and proc_index >= args.max_frames:
            break

        # Resize
        raw = cv2.resize(raw, (out_w, out_h))

        # Reload zones periodically
        registry.reload_if_needed(proc_index / target_fps)

        result: FrameResult = detector.process_frame(raw, proc_index, target_fps)
        video_time = result.video_time
        grey = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        buffer.push(video_time, raw)

        annotated = raw.copy()

        for zone_id, cfg in list(registry.configs.items()):
            state = registry.states[zone_id]
            poly_px = cfg.polygon_px(out_w, out_h)

            persons_in = [
                p for p in result.persons
                if point_in_polygon(p.bottom_center[0], p.bottom_center[1], poly_px)
            ]
            objects_in = [
                o for o in result.objects
                if point_in_polygon(o.bottom_center[0], o.bottom_center[1], poly_px)
            ]
            poses_in = [pr for pr in result.poses if any(pr.track_id == p.track_id for p in persons_in)]

            if zone_id not in motions:
                motions[zone_id] = MotionScorer(zone_id)
            motion_score = motions[zone_id].update(video_time, persons_in, poses_in, None, grey)

            prev_count = prev_counts.get(zone_id, 0)
            transition = state.update(
                video_time=video_time,
                present_track_ids=[p.track_id for p in persons_in if p.track_id >= 0],
                people_count=len(persons_in),
                motion_score=motion_score,
                objects_in_zone=[o.class_name for o in objects_in],
                cfg=cfg,
            )
            prev_counts[zone_id] = len(persons_in)

            if transition:
                ev = {
                    "event_id": str(uuid.uuid4()),
                    "camera_id": args.camera_id,
                    "zone_id": zone_id,
                    "zone_name": cfg.name,
                    "type": EventType.STATE_CHANGE.value,
                    "state": state.state.value,
                    "since": state.state_since,
                    "duration_s": state.dwell_seconds(video_time),
                    "people_count": state.people_count,
                    "motion_score": state.motion_score,
                    "confidence": 0.0,
                    "reason": f"Transition to {transition}",
                    "snapshot_path": None,
                    "timestamp": time.time(),
                }
                insert_event(ev)
                all_events.append(ev)

            # Rules
            candidates: list = []
            candidates += rule_intrusion(cfg, state, persons_in, video_time, raw)
            candidates += rule_loitering(cfg, state, persons_in, video_time, raw)
            candidates += rule_crowd_surge(cfg, state, persons_in, video_time, raw, prev_count)
            candidates += rule_in_vain(cfg, state, persons_in, objects_in, video_time, raw)
            candidates += rule_needs_cleaning(
                cfg, state, objects_in, video_time, raw,
                object_seen_since.setdefault(zone_id, {})
            )
            candidates += rule_fall(cfg, state, persons_in, poses_in, video_time, raw, prev_aspect)
            candidates += rule_fight(cfg, state, persons_in, poses_in, video_time, raw, prev_poses)
            candidates += rule_abandoned_object(
                result.objects, result.persons, video_time, raw, abandoned_since, cfg
            )

            for c in candidates:
                cooldown_key = f"{c.zone_id}:{c.event_type.value}"

                def _cb(verdict: dict, _c: RuleCandidate = c) -> None:
                    conf = float(verdict.get("confidence", 0.0))
                    ev2 = {
                        "event_id": str(uuid.uuid4()),
                        "camera_id": args.camera_id,
                        "zone_id": _c.zone_id,
                        "zone_name": _c.zone_name,
                        "type": _c.event_type.value,
                        "state": state.state.value,
                        "since": state.state_since,
                        "duration_s": state.dwell_seconds(video_time),
                        "people_count": state.people_count,
                        "motion_score": state.motion_score,
                        "confidence": conf,
                        "reason": verdict.get("reason", _c.reason),
                        "snapshot_path": None,
                        "timestamp": time.time(),
                    }
                    insert_event(ev2)
                    all_events.append(ev2)
                    logger.info("✔  %s  conf=%.2f  %s", _c.event_type.value, conf, ev2["reason"])

                gemma_worker.submit(
                    "in_vain" if c.event_type == EventType.IN_VAIN else
                    "dirty_zone" if c.event_type == EventType.NEEDS_CLEANING else "incident",
                    c.keyframes,
                    {"event_type": c.event_type.value, "reason": c.reason},
                    _cb,
                    cooldown_key=cooldown_key,
                )

            _draw_zone(annotated, cfg, state, video_time)

        # Draw persons
        for p in result.persons:
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 200, 50), 1)
            if p.track_id >= 0:
                cv2.putText(annotated, f"#{p.track_id}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 200, 50), 1)

        cv2.putText(annotated, f"T={video_time:.1f}s  F={proc_index}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        writer.write(annotated)
        proc_index += 1

        if proc_index % 50 == 0:
            elapsed = time.time() - t0
            logger.info("  Processed %d frames  %.1fs elapsed  VT=%.1fs", proc_index, elapsed, video_time)

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    logger.info("Done. %d processed frames in %.1fs  →  %s", proc_index, elapsed, args.out)

    _print_summary(registry, all_events, video_time if proc_index > 0 else 0.0)


if __name__ == "__main__":
    main()
