"""
app/gemma.py — Local Gemma 4 client with three backends.

Backends:
  mock         — instant plausible fake JSON (no model required)
  transformers — HuggingFace transformers (loads model into process)
  llamacpp     — calls a local llama-server through its /v1/chat/completions endpoint

The GemmaWorker runs in a background thread.
Main pipeline enqueues tasks; never blocks on model generation.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import queue
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ── Image utility ─────────────────────────────────────────────────────────────

def _pil_from_bgr(bgr: np.ndarray):
    from PIL import Image as PILImage
    rgb = bgr[:, :, ::-1]
    return PILImage.fromarray(rgb)


def _resize_pil(img, max_px: int = 512):
    from PIL import Image as PILImage
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)


def _frames_to_pil(frames: List[np.ndarray], max_px: int = 512):
    """Convert at most 3 BGR numpy frames to downscaled PIL images."""
    pil_imgs = []
    for f in frames[:3]:
        pil_imgs.append(_resize_pil(_pil_from_bgr(f), max_px))
    return pil_imgs


def _pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


# ── JSON parsing helpers ──────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


_SAFE_VERDICT: Dict[str, Any] = {"confidence": 0.0}

# ── Abstract base ─────────────────────────────────────────────────────────────


class GemmaClient(ABC):
    """Abstract Gemma client interface."""

    @abstractmethod
    def verdict(
        self, task: str, images: List[np.ndarray], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        task: 'in_vain' | 'dirty_zone' | 'incident'
        Returns a parsed JSON dict.
        """

    @abstractmethod
    def suggest_zones(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Return suggested zone polygons (normalised) for operator confirmation."""

    @abstractmethod
    def sql_answer(self, question: str) -> str:
        """Generate a SELECT, run it, and return a plain-language summary."""


# ── Mock backend ──────────────────────────────────────────────────────────────


class MockGemma(GemmaClient):
    """Instant fake verdicts — no model required."""

    def verdict(self, task: str, images, metadata) -> Dict[str, Any]:
        time.sleep(0.05)   # realistic tiny latency
        if task == "in_vain":
            return {"purposeful": False, "reason": "[mock] occupant appears idle", "confidence": 0.72}
        if task == "dirty_zone":
            return {"needs_cleaning": True, "reason": "[mock] leftover items detected", "confidence": 0.80}
        # incident
        return {"is_incident": True, "type": metadata.get("event_type", "UNKNOWN"),
                "reason": "[mock] rule-triggered incident confirmed", "confidence": 0.75}

    def suggest_zones(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        return [
            {
                "name": "Mock Zone 1",
                "type": "general",
                "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.6], [0.1, 0.6]],
            }
        ]

    def sql_answer(self, question: str) -> str:
        return f"[mock] Cannot answer '{question}' without a real model."


# ── Transformers backend ──────────────────────────────────────────────────────


class TransformersGemma(GemmaClient):
    _instance: Optional["TransformersGemma"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._model = None
                obj._processor = None
                obj._loaded = False
                cls._instance = obj
        return cls._instance

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        t0 = time.time()
        logger.info("Loading Gemma model: %s (device=%s)", settings.GEMMA_MODEL_ID, settings.GEMMA_DEVICE)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            settings.GEMMA_MODEL_ID,
            torch_dtype=dtype,
            device_map=settings.GEMMA_DEVICE,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(settings.GEMMA_MODEL_ID)
        self._loaded = True
        logger.info("Gemma loaded in %.1fs", time.time() - t0)

    def _generate(self, messages: List[Dict[str, Any]]) -> str:
        import torch

        self._load()
        t0 = time.time()
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=settings.GEMMA_MAX_NEW_TOKENS,
                do_sample=False,
            )
        new_tokens = out[0][input_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        logger.debug("Gemma latency %.2fs", time.time() - t0)
        return text

    def _call(self, prompt: str, pil_images) -> Dict[str, Any]:
        # Build multi-modal chat message
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        raw = self._generate(messages)
        parsed = _parse_json(raw)
        if parsed is None:
            logger.warning("Gemma parse error (attempt 1). Retrying…")
            raw2 = self._generate(messages)
            parsed = _parse_json(raw2)
        return parsed or _SAFE_VERDICT

    def verdict(self, task: str, images, metadata) -> Dict[str, Any]:
        pil_imgs = _frames_to_pil(images, settings.GEMMA_IMAGE_MAX_PX)
        prompt = _verdict_prompt(task, metadata)
        return self._call(prompt, pil_imgs)

    def suggest_zones(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        pil = _resize_pil(_pil_from_bgr(frame), settings.GEMMA_IMAGE_MAX_PX)
        prompt = (
            "Analyse this CCTV frame. Identify tables, chairs, waiting areas, and "
            "restricted zones. Return a JSON array of objects: "
            '[{"name":"…","type":"table|queue|restricted|general","bbox":[x1,y1,x2,y2]}] '
            "where coordinates are normalised 0-1. Return ONLY the JSON array."
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
        raw = self._generate(messages)
        parsed = _parse_json(raw)
        if isinstance(parsed, list):
            return _bboxes_to_zones(parsed)
        return []

    def sql_answer(self, question: str) -> str:
        from app.db import run_select
        prompt = (
            f"You are a SQL assistant. The database has a table called 'events' with columns: "
            "event_id,camera_id,zone_id,zone_name,type,state,since,duration_s,people_count,"
            "motion_score,confidence,reason,snapshot_path,timestamp. "
            f"Write a read-only SELECT query to answer: '{question}'. "
            "Return ONLY the SQL, no markdown."
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        sql = self._generate(messages).strip()
        sql = _strip_fences(sql)
        if not _is_safe_sql(sql):
            return f"Generated SQL is not safe: {sql}"
        try:
            rows = run_select(sql)
            summary_prompt = (
                f"SQL: {sql}\nResult: {json.dumps(rows[:20])}\n"
                f"Question: {question}\nAnswer in plain English in 1-2 sentences."
            )
            messages2 = [{"role": "user", "content": [{"type": "text", "text": summary_prompt}]}]
            return self._generate(messages2).strip()
        except Exception as e:
            return f"SQL error: {e}"


# ── LlamaCpp backend ──────────────────────────────────────────────────────────


class LlamaCppGemma(GemmaClient):
    def __init__(self) -> None:
        self._base_url = settings.GEMMA_LLAMACPP_URL.rstrip("/")

    def _chat(self, messages: List[Dict[str, Any]]) -> str:
        import httpx

        t0 = time.time()
        resp = httpx.post(
            f"{self._base_url}/v1/chat/completions",
            json={"model": "gemma", "messages": messages, "max_tokens": settings.GEMMA_MAX_NEW_TOKENS},
            timeout=120.0,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        logger.debug("LlamaCpp latency %.2fs", time.time() - t0)
        return text

    def verdict(self, task: str, images, metadata) -> Dict[str, Any]:
        pil_imgs = _frames_to_pil(images, settings.GEMMA_IMAGE_MAX_PX)
        prompt = _verdict_prompt(task, metadata)
        content: List[Any] = []
        for img in pil_imgs:
            b64 = _pil_to_b64(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        raw = self._chat(messages)
        parsed = _parse_json(raw)
        if parsed is None:
            raw2 = self._chat(messages)
            parsed = _parse_json(raw2)
        return parsed or _SAFE_VERDICT

    def suggest_zones(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        pil = _resize_pil(_pil_from_bgr(frame), settings.GEMMA_IMAGE_MAX_PX)
        b64 = _pil_to_b64(pil)
        prompt = (
            "Identify tables, chairs, waiting areas, and restricted zones. "
            'Return JSON array: [{"name":"…","type":"…","bbox":[x1,y1,x2,y2]}] (0-1 normalised). JSON only.'
        )
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}]
        raw = self._chat(messages)
        parsed = _parse_json(raw)
        if isinstance(parsed, list):
            return _bboxes_to_zones(parsed)
        return []

    def sql_answer(self, question: str) -> str:
        from app.db import run_select
        prompt = (
            f"events table columns: event_id,camera_id,zone_id,zone_name,type,state,since,"
            f"duration_s,people_count,motion_score,confidence,reason,snapshot_path,timestamp. "
            f"Write a SELECT to answer: '{question}'. SQL only."
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        sql = _strip_fences(self._chat(messages)).strip()
        if not _is_safe_sql(sql):
            return f"Generated SQL is not safe: {sql}"
        try:
            rows = run_select(sql)
            summary_prompt = f"SQL: {sql}\nResult: {json.dumps(rows[:20])}\nAnswer '{question}' in 1-2 sentences."
            messages2 = [{"role": "user", "content": [{"type": "text", "text": summary_prompt}]}]
            return self._chat(messages2).strip()
        except Exception as e:
            return f"SQL error: {e}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _verdict_prompt(task: str, metadata: Dict[str, Any]) -> str:
    meta_str = json.dumps(metadata, default=str)
    if task == "in_vain":
        return (
            "You are a smart space analyst. Look at the image(s) of a zone. "
            "Determine if the occupant is doing something purposeful (working, eating, reading) "
            "or occupying the space in vain (sleeping, idle, phone-zoning excessively). "
            f"Context: {meta_str}. "
            'Return ONLY JSON: {"purposeful": bool, "reason": "...", "confidence": 0.0-1.0}'
        )
    if task == "dirty_zone":
        return (
            "You are a facility cleanliness inspector. Look at the image(s) of an empty zone. "
            "Determine if it needs cleaning (leftover food, dirty plates, rubbish). "
            f"Context: {meta_str}. "
            'Return ONLY JSON: {"needs_cleaning": bool, "reason": "...", "confidence": 0.0-1.0}'
        )
    return (
        "You are a safety analyst reviewing CCTV footage. "
        "Determine if a safety incident is occurring in the image(s). "
        f"Context: {meta_str}. "
        'Return ONLY JSON: {"is_incident": bool, "type": "...", "reason": "...", "confidence": 0.0-1.0}'
    )


def _is_safe_sql(sql: str) -> bool:
    up = sql.upper().strip()
    if not up.startswith("SELECT"):
        return False
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "REPLACE"]
    return not any(kw in up for kw in forbidden)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _bboxes_to_zones(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    zones = []
    for item in raw:
        bbox = item.get("bbox", [0, 0, 1, 1])
        x1, y1, x2, y2 = bbox
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        zones.append({
            "name": item.get("name", "Suggested Zone"),
            "type": item.get("type", "general"),
            "polygon": polygon,
        })
    return zones


# ── Background worker ─────────────────────────────────────────────────────────


class GemmaTask:
    def __init__(
        self,
        task_id: str,
        task: str,
        images: List[np.ndarray],
        metadata: Dict[str, Any],
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        self.task_id = task_id
        self.task = task
        self.images = images
        self.metadata = metadata
        self.callback = callback
        self.created_at = time.time()


class GemmaWorker:
    """
    Single background thread that processes Gemma tasks.
    Per-zone cooldowns prevent spamming.
    """

    def __init__(self) -> None:
        self._client: Optional[GemmaClient] = None
        self._q: queue.Queue[GemmaTask] = queue.Queue(maxsize=settings.GEMMA_QUEUE_SIZE)
        self._gen_lock = threading.Lock()
        self._cooldown: Dict[str, float] = {}   # key → last call wall-clock time
        self._thread = threading.Thread(target=self._run, daemon=True, name="gemma-worker")
        self._thread.start()
        logger.info("GemmaWorker started (mode=%s)", settings.GEMMA_MODE)

    def _get_client(self) -> GemmaClient:
        if self._client is None:
            mode = settings.GEMMA_MODE
            if mode == "transformers":
                self._client = TransformersGemma()
            elif mode == "llamacpp":
                self._client = LlamaCppGemma()
            else:
                self._client = MockGemma()
        return self._client

    def submit(
        self,
        task: str,
        images: List[np.ndarray],
        metadata: Dict[str, Any],
        callback: Callable[[Dict[str, Any]], None],
        cooldown_key: Optional[str] = None,
    ) -> bool:
        """
        Enqueue a task.  Returns True if enqueued, False if on cooldown or queue full.
        """
        if cooldown_key:
            now = time.time()
            last = self._cooldown.get(cooldown_key, 0.0)
            if now - last < settings.GEMMA_COOLDOWN_S:
                return False
            self._cooldown[cooldown_key] = now

        gt = GemmaTask(
            task_id=str(uuid.uuid4()),
            task=task,
            images=[f.copy() for f in images[:3]],
            metadata=metadata,
            callback=callback,
        )
        try:
            self._q.put_nowait(gt)
            return True
        except queue.Full:
            # Drop the oldest, put the new one
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(gt)
                return True
            except queue.Full:
                logger.warning("Gemma queue full — task dropped")
                return False

    def _run(self) -> None:
        while True:
            try:
                gt = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                with self._gen_lock:
                    client = self._get_client()
                    result = client.verdict(gt.task, gt.images, gt.metadata)
                gt.callback(result)
            except Exception as e:
                logger.exception("GemmaWorker error on task %s: %s", gt.task_id, e)
                try:
                    gt.callback(_SAFE_VERDICT)
                except Exception:
                    pass


# ── Singleton worker ──────────────────────────────────────────────────────────
_worker: Optional[GemmaWorker] = None
_worker_lock = threading.Lock()


def get_gemma_worker() -> GemmaWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = GemmaWorker()
    return _worker


def get_gemma_client() -> GemmaClient:
    """Return a direct (synchronous) client — use only for suggest_zones and sql_answer."""
    worker = get_gemma_worker()
    return worker._get_client()
