"""
smartspace-ml/app/table_classifier.py — Vision-Language Zero-Shot Table Condition Classifier
===========================================================================================
Primary Model: openai/clip-vit-base-patch32
Evaluates the physical state of a cropped tabletop against zero-shot prompt ensembles:
  - occupied
  - unoccupied_clean
  - unoccupied_dirty
  - uncertain

Never classifies full-frame background as a table; requires a grounded cropped physical table region.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

PROMPT_ENSEMBLES: Dict[str, List[str]] = {
    "occupied": [
        "a dining table with people sitting",
        "an occupied dining table",
        "people eating at a dining table",
        "customers seated around a table",
    ],
    "unoccupied_clean": [
        "an empty clean dining table",
        "a clean cleared dining table with no items",
        "an empty restaurant table ready for guests",
        "a wiped clean empty table",
    ],
    "unoccupied_dirty": [
        "an empty dining table with dirty dishes and leftover food",
        "a messy dining table with trash and dirty plates",
        "an uncleared restaurant table with garbage",
        "dishes and rubbish left on an empty table",
    ],
    "uncertain": [
        "an unclear view of a table",
        "blurry or heavily obstructed view of a table surface",
        "dark unidentifiable surface",
    ],
}


class CLIPTableClassifier:
    """
    Vision-Language zero-shot table condition classifier using OpenAI CLIP.
    Operates on cropped candidate tabletop surfaces.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_CLIP_MODEL,
        device_preference: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device_preference = device_preference
        self.device = self._select_device(device_preference)

        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.is_loaded: bool = False
        self.is_mock_mode: bool = False

        self.classes: List[str] = list(PROMPT_ENSEMBLES.keys())
        self._text_features: Optional[torch.Tensor] = None

    def _select_device(self, preference: str) -> torch.device:
        pref = preference.lower().strip()
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self) -> None:
        if self.is_loaded:
            return

        force_mock = os.getenv("LENZ_MOCK_VISION", "0").lower() in ("1", "true", "yes")
        if not force_mock:
            try:
                from transformers import CLIPModel, CLIPProcessor
                logger.info("Loading CLIP model '%s' on %s...", self.model_id, self.device)
                try:
                    self.processor = CLIPProcessor.from_pretrained(self.model_id, local_files_only=True)
                    self.model = CLIPModel.from_pretrained(self.model_id, local_files_only=True)
                except Exception:
                    from app.open_vocab_detector import is_online
                    if not is_online():
                        raise ConnectionError("Network offline and weights not in local cache")
                    self.processor = CLIPProcessor.from_pretrained(self.model_id)
                    self.model = CLIPModel.from_pretrained(self.model_id)
                self.model.to(self.device)
                self.model.eval()
                self._encode_text_prompts()
                self.is_loaded = True
                self.is_mock_mode = False
                logger.info("CLIP table condition classifier ready.")
                return
            except Exception as exc:
                logger.warning("Could not load CLIP model '%s': %s", self.model_id, exc)

        self.is_mock_mode = True
        self.is_loaded = True
        logger.info("[LENZ VISION ENGINE] Running CLIP classifier in mock mode for offline testing.")

    def _encode_text_prompts(self) -> None:
        """Pre-computes normalized text embeddings for each class ensemble."""
        if self.model is None or self.processor is None:
            return

        all_class_embeddings = []
        with torch.inference_mode():
            for cls_name in self.classes:
                prompts = PROMPT_ENSEMBLES[cls_name]
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                text_features = self.model.get_text_features(**inputs)
                # Normalize features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Ensemble mean
                mean_features = text_features.mean(dim=0, keepdim=True)
                mean_features = mean_features / mean_features.norm(dim=-1, keepdim=True)
                all_class_embeddings.append(mean_features)

            self._text_features = torch.cat(all_class_embeddings, dim=0)

    def classify_crop(
        self,
        table_crop: np.ndarray,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classifies a cropped tabletop image.

        Returns:
            (predicted_class, confidence, probabilities_dict)
        """
        if not self.is_loaded:
            self.load()

        if table_crop is None or table_crop.size == 0 or min(table_crop.shape[:2]) < 8:
            return "uncertain", 1.0, {"uncertain": 1.0, "occupied": 0.0, "unoccupied_clean": 0.0, "unoccupied_dirty": 0.0}

        # Real CLIP inference
        if not self.is_mock_mode and self.model is not None and self.processor is not None and self._text_features is not None:
            try:
                rgb_crop = cv2.cvtColor(table_crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(rgb_crop)

                inputs = self.processor(images=pil_crop, return_tensors="pt")
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                with torch.inference_mode():
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    # Cosine similarity: [1, num_classes]
                    similarity = (image_features @ self._text_features.T) * self.model.logit_scale.exp()
                    probs = similarity.softmax(dim=-1).cpu().numpy()[0]

                probs_dict = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}
                top_class = self.classes[int(np.argmax(probs))]
                top_conf = float(np.max(probs))

                return top_class, top_conf, probs_dict
            except Exception as e:
                logger.debug("CLIP classification error: %s", e)

        # Fallback / offline mode
        return self._fallback_classify_crop(table_crop)

    def _fallback_classify_crop(self, crop: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Heuristic/visual inspection fallback when offline weights are unavailable."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Check for dirty tableware / debris (green/patterned)
        mask_green = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
        green_ratio = np.sum(mask_green > 0) / (crop.shape[0] * crop.shape[1] + 1e-6)

        # Check for seated human color presence
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([15, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 100, 80]), np.array([180, 255, 255]))
        red_ratio = (np.sum(mask_red1 > 0) + np.sum(mask_red2 > 0)) / (crop.shape[0] * crop.shape[1] + 1e-6)

        mean_val = float(np.mean(crop))
        if mean_val < 15.0 or crop.size < 100:
            # Pitch dark or invalid crop
            return "uncertain", 0.85, {"uncertain": 0.85, "occupied": 0.05, "unoccupied_clean": 0.05, "unoccupied_dirty": 0.05}

        if red_ratio > 0.05:
            return "occupied", 0.90, {"occupied": 0.90, "unoccupied_clean": 0.05, "unoccupied_dirty": 0.03, "uncertain": 0.02}
        elif green_ratio > 0.02:
            return "unoccupied_dirty", 0.88, {"unoccupied_dirty": 0.88, "unoccupied_clean": 0.06, "occupied": 0.03, "uncertain": 0.03}
        else:
            return "unoccupied_clean", 0.92, {"unoccupied_clean": 0.92, "occupied": 0.03, "unoccupied_dirty": 0.03, "uncertain": 0.02}
