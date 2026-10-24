"""
training/dataset_validator.py — Validates YOLO dataset integrity, annotations,
split leakage, and class distributions before model training.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import yaml


class DatasetValidator:
    def __init__(self, data_yaml_path: str, output_viz_dir: Optional[str] = None) -> None:
        self.data_yaml_path = Path(data_yaml_path).resolve()
        if not self.data_yaml_path.is_file():
            raise FileNotFoundError(f"data.yaml not found at: {self.data_yaml_path}")

        self.root_dir = self.data_yaml_path.parent
        self.output_viz_dir = Path(output_viz_dir) if output_viz_dir else self.root_dir / "runs" / "dataset_verification"

        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.names: Dict[int, str] = self._parse_names(self.cfg.get("names", []))
        self.nc: int = self.cfg.get("nc", len(self.names))
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.class_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
        self.split_hashes: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    def _parse_names(self, names_entry) -> Dict[int, str]:
        if isinstance(names_entry, dict):
            return {int(k): str(v) for k, v in names_entry.items()}
        elif isinstance(names_entry, list):
            return {i: str(v) for i, v in enumerate(names_entry)}
        return {}

    def _resolve_split_path(self, split_key: str) -> Optional[Path]:
        raw = self.cfg.get(split_key)
        if not raw:
            return None
        p = Path(raw)
        if p.is_absolute() and p.exists():
            return p
        # relative to yaml parent
        cand1 = (self.root_dir / p).resolve()
        if cand1.exists():
            return cand1
        # relative to 'path' key if defined
        if "path" in self.cfg:
            base = Path(self.cfg["path"])
            cand2 = (base / p).resolve()
            if cand2.exists():
                return cand2
        return cand1

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def validate_split(self, split: str) -> Dict[str, int]:
        img_dir = self._resolve_split_path(split)
        stats = {"images": 0, "labels": 0, "corrupt": 0, "empty_labels": 0, "invalid_boxes": 0}

        if not img_dir or not img_dir.exists():
            if split in ("train", "val"):
                self.errors.append(f"Required split '{split}' path does not exist: {img_dir}")
            else:
                self.warnings.append(f"Optional split '{split}' path does not exist: {img_dir}")
            return stats

        # Find matching label directory
        # Standard: images/<split> -> labels/<split>
        label_dir = None
        if "images" in img_dir.parts:
            parts = list(img_dir.parts)
            idx = parts.index("images")
            parts[idx] = "labels"
            cand = Path(*parts)
            if cand.is_dir():
                label_dir = cand
        if not label_dir:
            cand = img_dir.parent.parent / "labels" / img_dir.name
            if cand.is_dir():
                label_dir = cand

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        stats["images"] = len(image_files)

        if stats["images"] == 0:
            self.errors.append(f"Split '{split}' contains 0 readable images in {img_dir}")
            return stats

        for img_path in sorted(image_files):
            # 1. Image integrity check
            im = cv2.imread(str(img_path))
            if im is None:
                self.errors.append(f"Corrupt or unreadable image: {img_path}")
                stats["corrupt"] += 1
                continue

            h, w = im.shape[:2]
            if h < 10 or w < 10:
                self.warnings.append(f"Image too small ({w}x{h}): {img_path}")

            # Check split hash leakage
            img_hash = self._file_hash(img_path)
            self.split_hashes[split].add(img_hash)

            # 2. Matching label check
            lbl_file = (label_dir / f"{img_path.stem}.txt") if label_dir else None
            if not lbl_file or not lbl_file.is_file():
                # Some empty images are valid backgrounds, but track them
                stats["empty_labels"] += 1
                continue

            stats["labels"] += 1
            try:
                with open(lbl_file, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()
            except Exception as e:
                self.errors.append(f"Failed to read label {lbl_file}: {e}")
                continue

            for line_no, line in enumerate(lines, 1):
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) != 5:
                    self.errors.append(
                        f"Invalid bbox format in {lbl_file.name}:{line_no} (expected 5 values, got {len(tokens)})"
                    )
                    stats["invalid_boxes"] += 1
                    continue

                try:
                    cls_id = int(tokens[0])
                    xc, yc, bw, bh = float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
                except ValueError as ve:
                    self.errors.append(f"Non-numeric bbox values in {lbl_file.name}:{line_no}: {ve}")
                    stats["invalid_boxes"] += 1
                    continue

                # Bounds check
                if cls_id < 0 or cls_id >= self.nc:
                    self.errors.append(
                        f"Class ID {cls_id} out of bounds [0, {self.nc-1}] in {lbl_file.name}:{line_no}"
                    )
                    stats["invalid_boxes"] += 1

                for val, name in [(xc, "x_center"), (yc, "y_center"), (bw, "width"), (bh, "height")]:
                    if val < 0.0 or val > 1.0:
                        self.errors.append(
                            f"Coordinate {name}={val} outside [0.0, 1.0] in {lbl_file.name}:{line_no}"
                        )
                        stats["invalid_boxes"] += 1

                self.class_counts[split][cls_id] += 1

        return stats

    def check_split_leakage(self) -> None:
        """Verifies that no identical images exist across train, val, and test splits."""
        train_val = self.split_hashes["train"] & self.split_hashes["val"]
        if train_val:
            self.errors.append(f"DATA LEAKAGE DETECTED: {len(train_val)} identical image(s) shared between train and val splits!")

        train_test = self.split_hashes["train"] & self.split_hashes["test"]
        if train_test:
            self.errors.append(f"DATA LEAKAGE DETECTED: {len(train_test)} identical image(s) shared between train and test splits!")

        val_test = self.split_hashes["val"] & self.split_hashes["test"]
        if val_test:
            self.warnings.append(f"Notice: {len(val_test)} identical image(s) shared between val and test splits.")

    def generate_visual_inspection_samples(self, max_samples: int = 5) -> None:
        """Renders bounding boxes on sample images to verify label alignment visually."""
        self.output_viz_dir.mkdir(parents=True, exist_ok=True)
        img_dir = self._resolve_split_path("train")
        if not img_dir or not img_dir.exists():
            return

        label_dir = img_dir.parent.parent / "labels" / img_dir.name
        samples = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        count = 0
        for img_path in sorted(samples):
            lbl_file = label_dir / f"{img_path.stem}.txt"
            if not lbl_file.exists():
                continue

            im = cv2.imread(str(img_path))
            if im is None:
                continue

            h, w = im.shape[:2]
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = (float(v) for v in parts[1:])
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)

                        label_name = self.names.get(cls_id, f"cls_{cls_id}")
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            im,
                            label_name,
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

            out_path = self.output_viz_dir / f"sample_{img_path.name}"
            cv2.imwrite(str(out_path), im)
            count += 1
            if count >= max_samples:
                break

    def run(self) -> bool:
        print("=" * 70)
        print("Lenz Dataset Verification & Audit Tool")
        print(f"Data configuration: {self.data_yaml_path}")
        print(f"Declared classes ({self.nc}): {self.names}")
        print("=" * 70)

        for split in ["train", "val", "test"]:
            stats = self.validate_split(split)
            print(f"Split [{split.upper()}]:")
            print(f"  Images: {stats['images']} | Valid Labels: {stats['labels']} | Corrupt: {stats['corrupt']}")
            counts = self.class_counts[split]
            if counts:
                breakdown = [f"{self.names.get(k, k)}: {v}" for k, v in sorted(counts.items())]
                print(f"  Class distribution: {', '.join(breakdown)}")
            else:
                print(f"  Class distribution: [No annotations]")

        self.check_split_leakage()
        self.generate_visual_inspection_samples()

        print("-" * 70)
        if self.warnings:
            print(f"[!] {len(self.warnings)} WARNING(S):")
            for w in self.warnings:
                print(f"    - {w}")

        if self.errors:
            print(f"[X] {len(self.errors)} ERROR(S) FOUND:")
            for e in self.errors:
                print(f"    - {e}")
            print("\nDataset validation FAILED. Resolve errors before training.")
            return False
        else:
            print("[+] DATASET VALIDATION PASSED! Clean dataset ready for training.")
            return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Lenz Dataset Validator")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--viz", type=str, default=None, help="Output directory for visual inspection samples")
    args = parser.parse_args()

    validator = DatasetValidator(args.data, output_viz_dir=args.viz)
    success = validator.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
