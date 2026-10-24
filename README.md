# Lenz — Make Every Camera Intelligent with AI

> **Proposed Architecture** — The system architecture, data models, API specifications, and pipeline designs described in this document represent the intended production design. The current repository contains a validated ML pipeline prototype (`smartspace-ml/`) and a standalone webcam experiment (`experiments/webcam_occupancy/`) that prove the core computer-vision functionality.

---

## What Is Lenz?

Lenz is an AI-powered video intelligence platform that transforms standard CCTV cameras and webcams into real-time operational sensors. Instead of requiring a human to watch camera feeds, Lenz continuously monitors video streams, detects events of interest, and surfaces actionable alerts — without cloud LLM dependencies for the core detection and rule engine.

### The Problem

| Passive CCTV Today | Lenz Active Intelligence |
|---|---|
| Humans must watch feeds continuously | Automated detection and alerting |
| Incidents are discovered after the fact | Real-time rule evaluation per frame |
| No spatial awareness of zones | Configurable polygon zones per camera |
| Alert fatigue from false positives | VLM-verified alerts with evidence clips |
| Siloed recordings with no indexing | Structured event database with full audit trail |

---

## Core Capabilities

### 1. Multi-Class Object Detection & Tracking
- Real-time detection across **80 COCO object classes** using YOLO
- ByteTrack multi-object tracking with persistent IDs and dwell time measurement
- Person, vehicle, luggage, electronics, and furniture detection

### 2. Zone-Based Spatial Intelligence
- Configurable polygon monitoring zones per camera
- Point-in-polygon spatial containment checks
- Zone state machine: `EMPTY` → `OCCUPIED` → `IDLE_OCCUPIED` → `NEEDS_ATTENTION`
- Temporal debounce timers to prevent state flickering

### 3. Automated Rule Engine
| Rule | Description |
|---|---|
| **Zone Occupancy** | Tracks who is in each zone, for how long, and whether occupancy is expected |
| **In-Vain / Idle Detection** | Flags occupants who remain without purposeful activity for extended periods |
| **Restricted Intrusion** | Immediate alert when a person enters a staff-only or sensitive zone |
| **Crowd Surge** | Headcount threshold monitoring with surge alerting |
| **Queue Congestion** | Dwell-time tracking for waiting areas with congestion flags |
| **Smart Parking** | Vehicle presence detection over designated parking bays |
| **Unattended Objects** | Flags bags, luggage, or electronics left without a person nearby |
| **Loitering** | Persistent person presence in sensitive zones beyond a time limit |

### 4. Local Vision-Language Model Verification (Gemma)
- Rule-triggered candidate keyframes dispatched asynchronously to a local Gemma client
- Supports in-process Hugging Face Transformers, local `llama-server` HTTP, or mock mode
- Per-zone cooldowns, queue management, and non-blocking background workers
- Reduces alert noise by verifying borderline rule triggers with image understanding

### 5. Incident Evidence Packaging
- Rolling in-memory buffer: 15 s pre-incident + 15 s post-incident MP4 clips
- SHA-256 integrity hashes on all evidence clips
- Keyframe extraction and optional face blurring
- Structured metadata stored in SQLite

### 6. FastAPI REST + MJPEG + WebSocket Service
- Live zone state API and dynamic zone CRUD
- Historical event and incident retrieval
- Live annotated video streaming (MJPEG)
- AI zone suggestion and natural-language analytics queries
- WebSocket push channel for real-time operator dashboard

---

## Repository Structure

```
Lenz/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── .gitignore                       # Repository exclusion rules
│
├── docs/                            # Architecture diagrams and visual assets
│   ├── README_ASSETS.md             # Asset index and description
│   ├── diagrams/                    # Mermaid source files (.mmd)
│   └── images/                      # Rendered PNG/SVG exports & analytics charts
│
├── experiments/
│   └── webcam_occupancy/            # Standalone webcam prototype (RUNNABLE)
│       ├── webcam_test.py           # Main script — multi-class detection, tracking & zone rules
│       ├── test_prototype.py        # 12 automated unit tests (all passing)
│       ├── requirements.txt         # Minimal dependencies
│       └── README.md                # Detailed usage guide & keyboard shortcuts
│
├── smartspace-ml/                   # Full ML pipeline prototype (RUNNABLE)
│   ├── app/                         # Detector, zones, rules, pipeline, motion, FastAPI
│   ├── tests/                       # 44 automated tests across 7 test files
│   ├── evidence/                    # Saved incident clips & keyframes (runtime)
│   ├── .env.example                 # Environment configuration template
│   ├── make_test_video.py           # Synthetic test video generator
│   ├── run_demo.py                  # Standalone CLI demo runner
│   ├── zones.json                   # Calibrated zones (table, parking, queue, restricted)
│   ├── requirements.txt
│   └── README.md                    # Full smartspace-ml usage guide
│
├── training/                        # Custom YOLO fine-tuning & evaluation suite
│   ├── train.py                     # Training entry point (supports Apple Silicon MPS & CPU)
│   ├── dataset_validator.py         # Dataset integrity, annotation bounds & leakage audit
│   ├── evaluate.py                  # Offline evaluation on untouched test split
│   └── README.md                    # Detailed training & annotation guide
│
├── datasets/                        # Standardized dataset directory
│   └── lenz_table_condition/        # Table cleanliness & occupancy dataset (data.yaml, splits)
│
└── tests/                           # Top-level regression test suite
    └── test_lenz_regression.py      # 15 end-to-end regression test cases (all passing)
```

---

## Quick Start

### Webcam Experiment (Fastest)

```bash
cd experiments/webcam_occupancy
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python webcam_test.py
```

Press **Q** to quit. The script downloads `yolo11n.pt` (~5 MB) on first run via `ultralytics`.

See [`experiments/webcam_occupancy/README.md`](experiments/webcam_occupancy/README.md) for full configuration options.

### SmartSpace ML Pipeline

```bash
cd smartspace-ml
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # Edit .env for your configuration
python make_test_video.py          # Generate sample.mp4 test video
python run_demo.py                 # Run CLI demo
```

See [`smartspace-ml/README.md`](smartspace-ml/README.md) for API server startup, zone configuration, and advanced options.

---

## Model Weights

YOLO model weights (`*.pt`) are **not committed to this repository** — they are large binaries that are automatically downloaded by `ultralytics` on first run.

| Model | Size | Download |
|---|---|---|
| `yolo11n.pt` | ~5.4 MB | Auto-downloaded by ultralytics |
| `yolov8n.pt` | ~6.5 MB | Auto-downloaded by ultralytics |

---

## Environment Configuration

Copy `.env.example` to `.env` in `smartspace-ml/` and configure:

```bash
cp smartspace-ml/.env.example smartspace-ml/.env
```

Key settings:

```env
VIDEO_SOURCE=sample.mp4        # Path to video file or RTSP URL
CAMERA_ID=cam_01
YOLO_WEIGHTS=yolov8n.pt
GEMMA_MODE=mock                # mock | transformers | llamacpp
TELEGRAM_BOT_TOKEN=            # Optional: for alert notifications
TELEGRAM_CHAT_ID=              # Optional: for alert notifications
```

> **Never commit your `.env` file.** It is excluded by `.gitignore`.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Video Source                       │
│         (Webcam / RTSP / Recorded File)             │
└────────────────────┬────────────────────────────────┘
                     │ Frames
                     ▼
┌─────────────────────────────────────────────────────┐
│            YOLO Inference + ByteTrack               │
│   Detection (80 COCO classes) + Tracking (IDs)      │
└────────────────────┬────────────────────────────────┘
                     │ Detections + Track IDs
                     ▼
┌─────────────────────────────────────────────────────┐
│              Zone State Machine                      │
│  Point-in-polygon → Temporal debounce → State emit  │
└────────────────────┬────────────────────────────────┘
                     │ Zone events
                     ▼
┌─────────────────────────────────────────────────────┐
│               Rule Engine                            │
│  Intrusion / Crowd / Loitering / Idle / Unattended  │
└────────────────────┬────────────────────────────────┘
                     │ Rule triggers
             ┌───────┴───────┐
             ▼               ▼
┌────────────────┐  ┌──────────────────────┐
│ Gemma VLM      │  │ Evidence Packager     │
│ Verification   │  │ Pre/post clip + hash  │
└────────────────┘  └──────────────────────┘
             │               │
             └───────┬───────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│         FastAPI + WebSocket + MJPEG                  │
│   REST API / Live stream / Operator dashboard push  │
└─────────────────────────────────────────────────────┘
```

---

## Lenz Use Cases

| Use Case | Prototype Status | Pipeline Status |
|---|---|---|
| Person & Vehicle Detection | ✅ Working | ✅ Working |
| Multi-Object Tracking (ByteTrack) | ✅ Working | ✅ Working |
| Zone-Based Occupancy | ✅ Working | ✅ Working |
| Table Cleanliness & Occupancy | ✅ Working (3 states) | ✅ Working (3 states) |
| Smart Parking Bay Monitoring | ✅ Working | ✅ Working |
| Crowd Density & Surge | ✅ Working | ✅ Working |
| Queue / Waiting Area Congestion | ✅ Working | ✅ Working |
| Restricted Area Intrusion | ✅ Working (Ground-Contact) | ✅ Working (Ground-Contact) |
| Loitering & Unattended Items | ✅ Working | ✅ Working |
| In-Vain / Idle Detection | — | ✅ Working |
| Local VLM Verification (Gemma) | — | ✅ Working (mock) |
| Incident Evidence Packaging | — | ✅ Working |
| Custom Model Fine-Tuning | ✅ Working (`training/`) | ✅ Working (`training/`) |
| FastAPI REST + WebSocket + MJPEG | — | ✅ Working |

---

## Table Occupancy & Cleanliness States

Lenz implements the grounded 3-state table condition model described in the specification:

1. **`occupied`**: Person is seated or standing at the table, or active work items (`laptop`, `phone`) are in use at the table.
2. **`unoccupied-clean`**: The table is unoccupied and visibly sanitized with zero leftover dishes, tableware, or rubbish. Ready for the next guest.
3. **`unoccupied-dirty`**: The table is unoccupied by people, but leftover tableware (`cup`, `bottle`, `wine glass`, `bowl`, `plate`, `fork`, `knife`, `spoon`) or food debris remain. Emits a `TABLE_NEEDS_CLEANING` alert.
4. **`uncertain`**: Borderline or occluded observations during transition frames, preventing state flickering.

Empty background, wall, or floor pixels are never identified as tables. Only configured physical table polygons or detected table anchors are assessed.

---

## Custom Model Training & Evaluation

When custom fine-tuning is required (e.g. for specialized uniforms, domain-specific tableware, or custom seat configurations):

### 1. Validate Dataset Integrity
```bash
python training/dataset_validator.py --data datasets/lenz_table_condition/data.yaml
```
Audits bounding box ranges $[0.0, 1.0]$, checks image readability, verifies class distribution, and checks for zero data leakage between splits.

### 2. Run Reproducible Training (Apple Silicon MPS / CPU)
```bash
python training/train.py \
  --model yolo11n.pt \
  --data datasets/lenz_table_condition/data.yaml \
  --epochs 30 \
  --batch 8 \
  --imgsz 640 \
  --device auto \
  --project runs/train \
  --name lenz_table_condition
```
Automatically detects Apple Silicon Metal Performance Shaders (MPS) for hardware acceleration. Saves `best.pt` and metrics to `runs/train/lenz_table_condition/`.

### 3. Evaluate on Untouched Test Split
```bash
python training/evaluate.py \
  --weights runs/train/lenz_table_condition/weights/best.pt \
  --data datasets/lenz_table_condition/data.yaml \
  --split test
```

---

## Running the Automated Test Suites (71 Tests Total)

### 1. Run the Entire Project Test Suite (All 71 Tests)
```bash
python -m pytest tests/ smartspace-ml/tests/ experiments/webcam_occupancy/test_prototype.py -v
```

### 2. Comprehensive Spatial & Regression Test Suite (15 Cases)
```bash
python -m pytest tests/test_lenz_regression.py -v
```
Covers:
- Person inside valid monitored region
- Person bbox extending into sky but feet ground-contact point outside (strict rejection)
- Car inside parking slot vs car on road
- Genuinely empty parking slot (`AVAILABLE`)
- Visible table: occupied vs unoccupied-clean vs unoccupied-dirty
- Partially occluded / borderline table (`UNCERTAIN`)
- Restricted area entry and departure
- Loitering dwell time alert
- Crowd surge threshold warning
- Debouncing transient 1-frame glitches
- Ambiguous detection handling
- Clean zero-detection frames

### 3. SmartSpace ML Pipeline Tests (44 Tests)
```bash
cd smartspace-ml && python -m pytest tests/ -v
```

### 4. Webcam Prototype Tests (12 Tests)
```bash
cd experiments/webcam_occupancy && python -m pytest test_prototype.py -v
```

---

## Detected Object Classes

The YOLO model supports **80 COCO object classes**, including:

- **People:** `person`
- **Vehicles:** `car`, `motorcycle`, `bicycle`, `bus`, `truck`, `airplane`, `boat`, `train`
- **Personal Belongings:** `backpack`, `handbag`, `suitcase`, `umbrella`
- **Workplace & Electronics:** `laptop`, `cell phone`, `mouse`, `keyboard`, `tv`, `remote`, `book`
- **Furniture:** `chair`, `bench`, `couch`, `dining table`, `bed`
- **Tableware:** `bottle`, `cup`, `wine glass`, `bowl`, `plate`, `fork`, `knife`, `spoon`

Detection results depend on visibility, lighting, distance, and occlusion in the camera feed.

---

## Spatial Grounding & Honest Scope

- **Ground-Contact Evaluation:** For floor, queue, and restricted zones, Lenz evaluates the tracked person's **bottom-center ground contact point** `(x_center, y_max)`, rather than the top of the bounding box. A person standing outside a restricted zone whose head or bounding box projects over the zone in perspective projection will **not** trigger an intrusion.
- **Physical Region Grounding:** Background pixels, ceiling, sky, or unmonitored floors are **never** classified as available parking slots or clean tables. A region must be a configured physical polygon or detected anchor object.
- **COCO Class Mapping:** A "bag" maps natively to `backpack`, `handbag`, and `suitcase` — all three are tracked and counted separately.
- **Hardware Acceleration:** Native Apple Silicon Metal (MPS) support with automatic CPU fallback. Full compatibility on macOS and Linux.

---

## Documentation & Diagrams

Full architecture diagrams are in [`docs/`](docs/):

- [System Architecture](docs/images/system-architecture.png)
- [AI Pipeline](docs/images/ai-pipeline.png)
- [System Workflow](docs/images/system-workflow.png)
- [Data Flow](docs/images/data-flow.png)
- [Database ER Diagram](docs/images/database-er.png)
- [Use Case Diagram](docs/images/use-case.png)
- [Dashboard Mockup](docs/images/dashboard-mockup.png)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Aman Raj Sharma**  
GitHub: [@AmanRajSharma001](https://github.com/AmanRajSharma001)  
Repository: [github.com/AmanRajSharma001/Lenz_alt](https://github.com/AmanRajSharma001/Lenz_alt)
