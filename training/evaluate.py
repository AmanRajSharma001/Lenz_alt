"""
training/evaluate.py — Evaluates trained checkpoints on an independent test split.
Outputs precision, recall, mAP50, and mAP50-95 metrics.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

try:
    import torch
    from ultralytics import YOLO
except ImportError as err:
    print(f"[!] ERROR: Missing dependencies: {err}", file=sys.stderr)
    sys.exit(1)


def evaluate_model(weights_path: str, data_yaml: str, split: str = "test", device: str = "auto") -> int:
    resolved_weights = Path(weights_path).resolve()
    resolved_yaml = Path(data_yaml).resolve()

    if not resolved_weights.is_file():
        print(f"[!] ERROR: Weights file not found: {resolved_weights}", file=sys.stderr)
        return 1
    if not resolved_yaml.is_file():
        print(f"[!] ERROR: data.yaml not found: {resolved_yaml}", file=sys.stderr)
        return 1

    if device == "auto":
        device = "mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else "cpu"

    print("=" * 70)
    print("Lenz Offline Model Evaluation")
    print(f"Checkpoint: {resolved_weights}")
    print(f"Dataset:    {resolved_yaml}")
    print(f"Split:      {split.upper()}")
    print(f"Device:     {device.upper()}")
    print("=" * 70)

    model = YOLO(str(resolved_weights))

    try:
        metrics = model.val(
            data=str(resolved_yaml),
            split=split,
            device=device,
            verbose=True,
        )
    except Exception as exc:
        print(f"[!] Evaluation failed: {exc}", file=sys.stderr)
        return 2

    # Extract metrics
    mp = metrics.box.mp       # Mean precision
    mr = metrics.box.mr       # Mean recall
    map50 = metrics.box.map50 # mAP @ 0.50 IoU
    map = metrics.box.map     # mAP @ 0.50:0.95 IoU

    print("\n" + "=" * 70)
    print("Evaluation Results Summary")
    print("-" * 70)
    print(f"Mean Precision (P):   {mp:.4f} ({mp*100:.1f}%)")
    print(f"Mean Recall (R):      {mr:.4f} ({mr*100:.1f}%)")
    print(f"mAP@50:               {map50:.4f} ({map50*100:.1f}%)")
    print(f"mAP@50-95:            {map:.4f} ({map*100:.1f}%)")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lenz Model Evaluation")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights file")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate: 'test' or 'val'")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'mps', or 'cpu'")
    args = parser.parse_args()

    return evaluate_model(args.weights, args.data, split=args.split, device=args.device)


if __name__ == "__main__":
    sys.exit(main())
