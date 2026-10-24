# Lenz — Custom Model Training & Dataset Pipeline

This directory provides the end-to-end, reproducible training and validation suite for custom fine-tuned YOLO models within the **Lenz Video Intelligence Platform**.

---

## 1. Scope: What Requires Custom Training?

Lenz uses a **pretrained COCO model (`yolo11n.pt` / `yolov8n.pt`)** for standard multi-class object detection and tracking across 80 classes (people, vehicles, tableware, personal belongings, electronics, and furniture).

However, **COCO does not distinguish fine-grained state conditions**, such as:
1. **Table Cleanliness & Condition:**
   - `occupied`: Person seated/standing at table or actively using purposeful work items.
   - `unoccupied-clean`: Completely clear, sanitized dining table ready for new guests.
   - `unoccupied-dirty`: Table vacated by guests, but leftover tableware, used cups, plates, bottles, or food debris remain.
2. **Specialized Vehicle Parking Alignments:**
   - Differentiating legal slot parking from aisle blocking or double parking.
3. **Domain-Specific Uniforms or Equipment:**
   - Security badges, high-vis vests, or hard hats in industrial zones.

Rather than trying to train one monolithic model on arbitrary labels, Lenz uses a **grounded two-stage pipeline**:
- Stage 1: Spatial physical region grounding (monitoring designated tables, parking bays, and queue corridors).
- Stage 2: State classification and condition estimation via dedicated classifiers or fine-tuned YOLO detectors.

---

## 2. Dataset Layout Standard

Custom datasets must follow the official YOLO dataset format:

```
datasets/<dataset_name>/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### `data.yaml` Specification:
```yaml
path: datasets/lenz_table_condition
train: images/train
val: images/val
test: images/test
nc: 3
names:
  0: occupied
  1: unoccupied-clean
  2: unoccupied-dirty
```

Each label file (`.txt`) must have normalized coordinates `[0.0, 1.0]`:
```
<class_id> <x_center> <y_center> <width> <height>
```

---

## 3. Dataset Audit & Pre-Training Validation

Always validate dataset integrity before launching training. This prevents wasted compute from corrupt images, misaligned bounding boxes, or data leakage:

```bash
python training/dataset_validator.py --data datasets/lenz_table_condition/data.yaml
```

The validator checks:
- Bounding box coordinates $\in [0.0, 1.0]$.
- Image readability and dimension constraints.
- Image hash collisions (detecting duplicate frames shared between `train`, `val`, and `test`).
- Class distribution and imbalance warnings.
- Visual inspection overlays saved to `runs/dataset_verification/`.

---

## 4. Reproducible Model Training

Train the model on your local Apple Silicon MacBook (automatically detects **MPS** acceleration) or CPU:

```bash
python training/train.py \
  --model yolo11n.pt \
  --data datasets/lenz_table_condition/data.yaml \
  --epochs 30 \
  --batch 8 \
  --imgsz 640 \
  --device auto \
  --project runs/train \
  --name lenz_table_run1 \
  --seed 42
```

Outputs:
- Best weights: `runs/train/lenz_table_run1/weights/best.pt`
- Last checkpoint: `runs/train/lenz_table_run1/weights/last.pt`
- Validation metrics: `runs/train/lenz_table_run1/results.csv`

---

## 5. Offline Evaluation on the Untouched Test Split

Evaluate model performance on the independent `test` split:

```bash
python training/evaluate.py \
  --weights runs/train/lenz_table_run1/weights/best.pt \
  --data datasets/lenz_table_condition/data.yaml \
  --split test
```

Reports:
- Mean Precision ($P$)
- Mean Recall ($R$)
- $\text{mAP}@50$
- $\text{mAP}@50\text{--}95$

---

## 6. How to Deploy Trained Weights into Lenz

To use your fine-tuned weights:
1. Copy the best weights:
   ```bash
   cp runs/train/lenz_table_run1/weights/best.pt smartspace-ml/lenz_custom.pt
   ```
2. In `smartspace-ml/.env`, update:
   ```ini
   YOLO_WEIGHTS=lenz_custom.pt
   ```
3. Or in the webcam prototype:
   ```bash
   python experiments/webcam_occupancy/webcam_test.py --model runs/train/lenz_table_run1/weights/best.pt
   ```
