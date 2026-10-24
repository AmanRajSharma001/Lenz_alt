# Lenz — Open-Vocabulary Video Intelligence & Spatial Analytics Prototype

This folder contains the active, local proof-of-concept for **Lenz — AI-Powered Intelligent Video Surveillance**.

**Important Architectural Change:** The previous YOLO/COCO object detection pipeline has been completely decommissioned from active inference. Lenz now uses an **Open-Vocabulary Computer Vision Architecture** powered by Hugging Face Transformers, Apple Silicon MPS hardware acceleration, Segment Anything (SAM 2.1), and zero-shot Vision-Language classification (OpenAI CLIP).

---

## 1. Root Cause: Why Previous Pipeline Behaved as Human-Centric

Audits of the previous system identified why detections appeared human-only and empty spaces were incorrectly represented:

1. **COCO Class ID Hardcoding & Bias:**
   Fixed integer class IDs (such as COCO class 0 = `person`) and narrow post-processing filters silently favored person bounding boxes over subtle or diverse workplace objects, vehicles, and tabletop items.

2. **Absence of Open-Vocabulary Text Query Grounding:**
   Closed-vocabulary detectors cannot accept descriptive English phrases (e.g., *"rubbish on a tabletop"*, *"dishes on a tabletop"*, *"seated person"*). Anything outside fixed categories was dropped.

3. **Faulty "Empty-Space" Heuristics:**
   Empty space was previously treated as if it were a visual class. In real-world physical surveillance, empty space is **not an object class**—it is a **state of an explicitly configured physical facility** (a parking slot, dining table, or floor region). Background pixels, empty sky, ceilings, and unmonitored floors must never be labeled as available spaces.

---

## 2. New Open-Vocabulary Model Architecture

| Component | Active Model ID | Purpose & Capabilities |
| :--- | :--- | :--- |
| **Open-Vocabulary Detector** | `IDEA-Research/grounding-dino-tiny` | Zero-shot object detection driven by runtime natural language prompts. Eliminates fixed integer class lists. |
| **Pixel Segmenter & Tracker** | `facebook/sam2.1-hiera-tiny` | Refines candidate bounding boxes into pixel-level binary masks for vehicle parking slots, tabletops, and human footprints. |
| **Table State Classifier** | `openai/clip-vit-base-patch32` | Zero-shot Vision-Language ensemble classification on cropped tabletops: `occupied`, `unoccupied_clean`, `unoccupied_dirty`, `uncertain`. |
| **Spatial Multi-Object Tracker** | `SpatialTracker` (IoU + Centroid) | Independent, deterministic association engine assigning persistent track IDs (`#ID`) and live dwell times. Zero closed-source/YOLO tracker dependencies. |
| **Hardware Accelerator** | Apple Silicon **MPS** (`torch.backends.mps`) | Hardware-accelerated Metal Performance Shaders on macOS, with seamless fallback to CPU. |

---

## 3. Configurable Prompt Management (`config/detection_prompts.json`)

Prompt definitions are decoupled from code in `config/detection_prompts.json`:

```json
{
  "dino_text_query": "person . seated person . standing person . dining table . chair . occupied chair . car . motorcycle . bicycle . bus . truck . backpack . handbag . suitcase . bottle . cup . laptop . phone . rubbish on a tabletop . dishes on a tabletop .",
  "categories": {
    "person": { "threshold": 0.28, "canonical_name": "person" },
    "vehicle": { "threshold": 0.28, "canonical_name": "vehicle" },
    "dining_table": { "threshold": 0.25, "canonical_name": "dining table" },
    "tableware": { "threshold": 0.22, "canonical_name": "tableware" },
    "rubbish": { "threshold": 0.20, "canonical_name": "rubbish" },
    "belongings": { "threshold": 0.25, "canonical_name": "bag" }
  }
}
```

New candidate phrases can be tested at runtime via `--prompts path/to/prompts.json` without modifying any Python code.

---

## 4. Physical Spatial Grounding Rules (The Empty-Space Mandate)

Lenz enforces strict physical grounding rules across all zones:

* **Floor / Restricted / Queue Zones:**
  Evaluated using the object's **ground-contact feet anchor** (`bottom_center` of the bounding box / mask). If a tall person's head/torso enters the camera's line-of-sight against the sky or wall while their feet remain outside the polygon, the zone remains **NOT OCCUPIED**.
* **Dining Tables:**
  Evaluated using table surface overlap and CLIP zero-shot classification:
  * **Occupied:** Guest seated at table or CLIP ensemble detects diners.
  * **Not Occupied (Clean):** Empty table with clean surface, ready for customers.
  * **Not Occupied (Needs Cleaning):** Empty table with uncleared dishes, glasses, or food rubbish.
* **Smart Parking Bays:**
  Evaluated using SAM vehicle mask / bounding box polygon overlap:
  * $\ge 30\%$ overlap $\rightarrow$ **OCCUPIED**.
  * $10\%$ to $30\%$ partial overlap $\rightarrow$ **UNCERTAIN** (requires inspection).
  * $0\%$ overlap $\rightarrow$ **AVAILABLE**.

---

## 5. Setup & Launch Instructions on macOS

### Step 1: Activate Virtual Environment
```bash
cd /Users/amanrajsharma/Desktop/Projects_Final/Lenz
source smartspace-ml/venv/bin/activate
```

### Step 2: Install / Verify Dependencies
```bash
pip install -r smartspace-ml/requirements.txt
```

### Step 3: Model Weight Loading Behavior
The application automatically detects cached Hugging Face model weights in `~/.cache/huggingface/hub/`.
To download weights locally for offline execution:
```bash
huggingface-cli download IDEA-Research/grounding-dino-tiny
huggingface-cli download facebook/sam2.1-hiera-tiny
huggingface-cli download openai/clip-vit-base-patch32
```
*Note: In offline test environments where weights are not yet cached, Lenz automatically activates an offline synthetic engine to allow unit tests, dry-runs, and diagnostic evaluation without network errors.*

---

## 6. Execution Commands

### A. Live Webcam Feed
```bash
python experiments/webcam_occupancy/webcam_test.py
```

### B. Prerecorded Video File
```bash
python experiments/webcam_occupancy/webcam_test.py --source smartspace-ml/sample.mp4
```

### C. Specific Operational Modes
```bash
# Focus on Smart Parking Bay monitoring
python experiments/webcam_occupancy/webcam_test.py --mode parking

# Focus on Dining Table occupancy & cleanliness
python experiments/webcam_occupancy/webcam_test.py --mode table

# Focus on Restricted Area intrusion
python experiments/webcam_occupancy/webcam_test.py --mode restricted

# Focus on Queue congestion
python experiments/webcam_occupancy/webcam_test.py --mode queue
```

### D. Single-Image Diagnostic Analysis
Performs open-vocabulary detection, SAM mask generation, and CLIP state classification on an individual image, saving annotated output and structured telemetry:
```bash
python experiments/webcam_occupancy/webcam_test.py --diagnostics-image smartspace-ml/evidence/sample_frame.jpg
```
Output files saved:
* `evidence/diagnostics_output.jpg`
* `evidence/diagnostics.json`

### E. Diagnostic Headless Run (CI / Automated Testing)
```bash
python experiments/webcam_occupancy/webcam_test.py --source smartspace-ml/sample.mp4 --max-frames 20 --headless --diagnostics
```

---

## 7. Running Automated Tests

Run the complete test suite verifying regression scenarios, open-vocabulary pipelines, spatial grounding, and rule engine evaluations:

```bash
python -m pytest tests/ smartspace-ml/tests/ experiments/webcam_occupancy/ -v
```

---

## 8. macOS Camera Permissions & Troubleshooting

If `Cannot open video source: Webcam #0` appears on macOS:

1. **System Permissions:** Open **System Settings $\rightarrow$ Privacy & Security $\rightarrow$ Camera**, and verify that your terminal application (Terminal, iTerm, VS Code, or Cursor) has camera access enabled.
2. **Camera In Use:** Close any applications actively using the camera (FaceTime, Zoom, Google Meet, Photo Booth).
3. **External / Continuity Camera:** If using an iPhone Continuity Camera or external USB webcam, select index 1 or 2:
   ```bash
   python experiments/webcam_occupancy/webcam_test.py --camera 1
   ```
