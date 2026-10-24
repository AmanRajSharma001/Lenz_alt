"""
app/main.py — FastAPI application.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.config import settings
from app.db import (
    delete_zone,
    get_zone,
    init_db,
    insert_zone,
    list_events,
    list_incidents,
    list_zones,
    update_incident_status,
    update_zone,
)
from app.gemma import get_gemma_client
from app.pipeline import Pipeline, get_pipeline
from app.schemas import (
    EventType,
    IncidentStatusUpdate,
    LiveZoneState,
    NLQueryRequest,
    NLQueryResponse,
    PipelineStartRequest,
    PipelineStatus,
    Zone,
    ZoneCreate,
    ZoneSuggestResponse,
    ZoneStateEnum,
    ZoneUpdate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# WS connections
_ws_clients: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("SmartSpace ML API ready.")

    # Wire pipeline events to WS broadcast
    pipeline = get_pipeline()
    loop = asyncio.get_running_loop()

    def _push_event(ev: Dict[str, Any]) -> None:
        payload = json.dumps(ev, default=str)
        dead: List[WebSocket] = []
        for ws in list(_ws_clients):
            try:
                loop.call_soon_threadsafe(
                    asyncio.ensure_future, ws.send_text(payload)
                )
            except Exception:
                dead.append(ws)
        for d in dead:
            if d in _ws_clients:
                _ws_clients.remove(d)

    pipeline.on_event = _push_event
    yield


app = FastAPI(title="SmartSpace ML", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "timestamp": str(time.time())}


# ── Pipeline control ───────────────────────────────────────────────────────────


@app.post("/pipeline/start")
async def pipeline_start(req: PipelineStartRequest) -> PipelineStatus:
    p = get_pipeline()
    if p.running:
        raise HTTPException(400, "Pipeline already running")
    p.start(req.source, req.camera_id)
    return PipelineStatus(
        running=True, camera_id=req.camera_id, source=req.source
    )


@app.post("/pipeline/stop")
async def pipeline_stop() -> PipelineStatus:
    p = get_pipeline()
    p.stop()
    return PipelineStatus(running=False)


@app.get("/pipeline/status")
async def pipeline_status() -> PipelineStatus:
    p = get_pipeline()
    return PipelineStatus(
        running=p.running,
        camera_id=p.camera_id,
        source=p.source,
        frames_processed=p.frames_processed,
        video_time_s=p.video_time_s,
    )


# ── Zone CRUD ──────────────────────────────────────────────────────────────────


@app.get("/zones", response_model=List[Zone])
async def get_zones(camera_id: Optional[str] = None):
    rows = list_zones(camera_id)
    return [Zone(**r) for r in rows]


@app.post("/zones", response_model=Zone, status_code=201)
async def create_zone(z: ZoneCreate):
    zid = insert_zone(z.model_dump())
    row = get_zone(zid)
    return Zone(**row)


@app.get("/zones/{zone_id}", response_model=Zone)
async def read_zone(zone_id: str):
    row = get_zone(zone_id)
    if not row:
        raise HTTPException(404, "Zone not found")
    return Zone(**row)


@app.put("/zones/{zone_id}", response_model=Zone)
async def update_zone_endpoint(zone_id: str, updates: ZoneUpdate):
    ok = update_zone(zone_id, {k: v for k, v in updates.model_dump().items() if v is not None})
    if not ok:
        raise HTTPException(404, "Zone not found")
    return Zone(**get_zone(zone_id))


@app.delete("/zones/{zone_id}")
async def delete_zone_endpoint(zone_id: str):
    ok = delete_zone(zone_id)
    if not ok:
        raise HTTPException(404, "Zone not found")
    return {"deleted": zone_id}


# ── Live zone state ────────────────────────────────────────────────────────────


@app.get("/zones/state", response_model=List[LiveZoneState])
async def zones_state():
    p = get_pipeline()
    if p._registry is None:
        return []
    result = []
    for zid, state in p._registry.states.items():
        cfg = p._registry.configs.get(zid)
        result.append(
            LiveZoneState(
                zone_id=zid,
                zone_name=state.zone_name,
                camera_id=state.camera_id,
                state=state.state,
                since=state.state_since,
                dwell_seconds=state.dwell_seconds(p.video_time_s),
                people_count=state.people_count,
                motion_score=state.motion_score,
                objects_in_zone=state.objects_in_zone,
            )
        )
    return result


# ── Events ─────────────────────────────────────────────────────────────────────


@app.get("/events")
async def get_events(
    zone_id: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 100,
):
    return list_events(zone_id=zone_id, event_type=type, limit=limit)


# ── Incidents ──────────────────────────────────────────────────────────────────


@app.get("/incidents")
async def get_incidents():
    return list_incidents()


@app.patch("/incidents/{incident_id}")
async def patch_incident(incident_id: str, body: IncidentStatusUpdate):
    ok = update_incident_status(incident_id, body.status.value)
    if not ok:
        raise HTTPException(404, "Incident not found")
    return {"updated": incident_id, "status": body.status}


# ── MJPEG stream ───────────────────────────────────────────────────────────────


def _mjpeg_generator(camera_id: str):
    p = get_pipeline()
    while True:
        frame = p.latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(1 / 15)   # ~15 fps stream


@app.get("/stream/{camera_id}")
async def mjpeg_stream(camera_id: str):
    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket ──────────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            # Keep connection alive; server pushes events via on_event callback
            await asyncio.sleep(5)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


# ── Zone suggest ───────────────────────────────────────────────────────────────


@app.post("/zones/suggest", response_model=ZoneSuggestResponse)
async def suggest_zones(file: UploadFile):
    import numpy as np
    from PIL import Image as PILImage

    data = await file.read()
    img = PILImage.open(io.BytesIO(data)).convert("RGB")
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    client = get_gemma_client()
    zones = client.suggest_zones(frame)
    return ZoneSuggestResponse(suggested_zones=zones)


# ── NL query ───────────────────────────────────────────────────────────────────


@app.post("/query", response_model=NLQueryResponse)
async def nl_query(req: NLQueryRequest):
    client = get_gemma_client()
    summary = client.sql_answer(req.question)
    return NLQueryResponse(
        question=req.question,
        sql="",
        result=None,
        summary=summary,
    )
