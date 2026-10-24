"""
app/config.py — centralised settings loaded from .env
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


class Settings:
    # ── Video ─────────────────────────────────────────────────────
    VIDEO_SOURCE: str = os.getenv("VIDEO_SOURCE", "sample.mp4")
    CAMERA_ID: str = os.getenv("CAMERA_ID", "cam_01")
    PROCESS_FPS: int = _int("PROCESS_FPS", 5)
    FRAME_WIDTH: int = _int("FRAME_WIDTH", 640)

    # ── Open-Vocabulary Vision Stack ──────────────────────────────
    GROUNDING_DINO_MODEL: str = os.getenv("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")
    SAM_MODEL: str = os.getenv("SAM_MODEL", "facebook/sam2.1-hiera-tiny")
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    PROMPTS_CONFIG_PATH: str = os.getenv("PROMPTS_CONFIG_PATH", "config/detection_prompts.json")
    DEVICE_PREFERENCE: str = os.getenv("DEVICE_PREFERENCE", "auto")
    DETECTION_CONFIDENCE: float = _float("DETECTION_CONFIDENCE", 0.25)
    SEGMENTATION_ENABLED: bool = _bool("SEGMENTATION_ENABLED", True)
    TABLE_CLASSIFICATION_ENABLED: bool = _bool("TABLE_CLASSIFICATION_ENABLED", True)

    # ── Zone state-machine ────────────────────────────────────────
    OCCUPY_ENTER_S: float = _float("OCCUPY_ENTER_S", 2.0)
    OCCUPY_EXIT_S: float = _float("OCCUPY_EXIT_S", 15.0)
    IDLE_LIMIT_S: float = _float("IDLE_LIMIT_S", 300.0)
    IDLE_MOTION_THRESHOLD: float = _float("IDLE_MOTION_THRESHOLD", 0.02)
    UNATTENDED_OBJECT_S: float = _float("UNATTENDED_OBJECT_S", 120.0)
    LOITER_S: float = _float("LOITER_S", 180.0)
    MAX_PEOPLE_PER_ZONE: int = _int("MAX_PEOPLE_PER_ZONE", 10)
    ZONE_RELOAD_INTERVAL_S: float = _float("ZONE_RELOAD_INTERVAL_S", 5.0)

    # ── Gemma ─────────────────────────────────────────────────────
    GEMMA_MODE: Literal["mock", "transformers", "llamacpp"] = os.getenv(  # type: ignore[assignment]
        "GEMMA_MODE", "mock"
    )
    GEMMA_MODEL_ID: str = os.getenv("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
    GEMMA_DEVICE: str = os.getenv("GEMMA_DEVICE", "auto")
    GEMMA_MAX_NEW_TOKENS: int = _int("GEMMA_MAX_NEW_TOKENS", 200)
    GEMMA_LLAMACPP_URL: str = os.getenv("GEMMA_LLAMACPP_URL", "http://localhost:8080")
    GEMMA_COOLDOWN_S: float = _float("GEMMA_COOLDOWN_S", 30.0)
    GEMMA_QUEUE_SIZE: int = _int("GEMMA_QUEUE_SIZE", 10)
    GEMMA_IMAGE_MAX_PX: int = _int("GEMMA_IMAGE_MAX_PX", 512)

    # ── Alerts ────────────────────────────────────────────────────
    ALERT_CONFIDENCE: float = _float("ALERT_CONFIDENCE", 0.7)
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    ALERT_COOLDOWN_MINUTES: int = _int("ALERT_COOLDOWN_MINUTES", 5)

    # ── Evidence ──────────────────────────────────────────────────
    EVIDENCE_DIR: Path = Path(os.getenv("EVIDENCE_DIR", "./evidence"))
    EVIDENCE_BUFFER_S: float = _float("EVIDENCE_BUFFER_S", 15.0)
    BLUR_FACES: bool = _bool("BLUR_FACES", False)
    EVIDENCE_FPS: int = _int("EVIDENCE_FPS", 10)

    # ── API ───────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = _int("API_PORT", 8000)

    # ── DB ────────────────────────────────────────────────────────
    DB_PATH: Path = Path(os.getenv("DB_PATH", "./smartspace.db"))


settings = Settings()
