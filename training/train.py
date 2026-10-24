"""
training/train.py — Reproducible training pipeline for Lenz custom models.
Supports Apple Silicon MPS hardware acceleration and CPU fallback.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure YOLO does not attempt to write to non-writable system dirs
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

try:
    import torch
    from ultralytics import YOLO
except ImportError as err:
    print(f"[!] ERROR: Missing deep learning dependencies: {err}", file=sys.stderr)
    print("    Install dependencies via: pip install ultralytics torch", file=sys.stderr)
    sys.exit(1)

from dataset_validator import DatasetValidator


def detect_optimal_device(requested_device: str = "auto") -> str:
    """Detects optimal hardware device (Apple Silicon MPS -> CUDA -> CPU)."""
    if requested_device != "auto":
        return requested_device

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "0"
    return "cpu"


def resolve_base_weights(base_name: str) -> str:
    """Resolves local base checkpoint or falls back to name for Ultralytics download."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path(base_name),
        repo_root / "experiments" / "webcam_occupancy" / base_name,
        repo_root / "smartspace-ml" / base_name,
        repo_root / "experiments" / "webcam_occupancy" / "yolo11n.pt",
        repo_root / "experiments" / "webcam_occupancy" / "yolov8n.pt",
        repo_root / "smartspace-ml" / "yolov8n.pt",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return base_name


def run_training(
    model_name: str,
    data_yaml: str,
    epochs: int = 20,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "auto",
    project: str = "runs/train",
    name: str = "lenz_custom",
    seed: int = 42,
    patience: int = 10,
) -> int:
    resolved_yaml = Path(data_yaml).resolve()
    if not resolved_yaml.is_file():
        print(f"[!] ERROR: data.yaml not found: {resolved_yaml}", file=sys.stderr)
        return 1

    # Step 1: Pre-training dataset audit & validation
    print("[*] Running pre-training dataset integrity check...")
    validator = DatasetValidator(str(resolved_yaml))
    if not validator.run():
        print("[!] Dataset validation failed. Aborting training to avoid corrupt checkpoints.", file=sys.stderr)
        return 2

    # Step 2: Device selection
    active_device = detect_optimal_device(device)
    print(f"[*] Target device: {active_device.upper()} (MPS Apple Silicon / CPU)")

    # Step 3: Base model loading
    resolved_base = resolve_base_weights(model_name)
    print(f"[*] Loading base model: {resolved_base}")
    model = YOLO(resolved_base)

    # Step 4: Execute training
    print(f"[*] Starting training for {epochs} epochs (batch={batch_size}, imgsz={imgsz}, seed={seed})...")
    project_dir = Path(project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = model.train(
            data=str(resolved_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=active_device,
            project=str(project_dir),
            name=name,
            seed=seed,
            deterministic=True,
            patience=patience,
            save=True,
            verbose=True,
        )
    except Exception as exc:
        print(f"\n[!] Training runtime error: {exc}", file=sys.stderr)
        return 3

    save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir") else project_dir / name
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    print("\n" + "=" * 70)
    print("[+] Training completed successfully!")
    print(f"    Run directory:  {save_dir}")
    if best_weights.is_file():
        print(f"    Best checkpoint: {best_weights} ({best_weights.stat().st_size / (1024*1024):.2f} MB)")
    if last_weights.is_file():
        print(f"    Last checkpoint: {last_weights} ({last_weights.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lenz Custom Model Training Pipeline")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model checkpoint (default: yolo11n.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml dataset config")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'mps', or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train", help="Project output folder")
    parser.add_argument("--name", type=str, default="lenz_custom", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    args = parser.parse_args()

    return run_training(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
    )


if __name__ == "__main__":
    sys.exit(main())
