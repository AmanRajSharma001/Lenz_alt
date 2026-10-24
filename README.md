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
│
├── docs/                            # Architecture diagrams and visual assets
│   ├── README_ASSETS.md             # Asset index and description
│   ├── diagrams/                    # Mermaid source files (.mmd)
│   │   ├── system-architecture.mmd
│   │   ├── ai-pipeline.mmd
│   │   ├── data-flow.mmd
│   │   ├── system-workflow.mmd
│   │   ├── database-er.mmd
│   │   └── use-case.mmd
│   └── images/                      # Rendered PNG/SVG exports
│       ├── lenz-hero.png
│       ├── system-architecture.png
│       ├── ai-pipeline.png
│       ├── system-workflow.png
│       ├── database-er.png
│       ├── use-case.png
│       ├── data-flow.png
│       ├── dashboard-mockup.png
│       ├── problem-vs-solution.png
│       └── analytics/               # Illustrative operational analytics charts
│
├── experiments/
│   └── webcam_occupancy/            # Standalone webcam prototype (RUNNABLE)
│       ├── webcam_test.py           # Main script — multi-class detection + zone rules
│       ├── test_prototype.py        # 12 automated unit tests (all passing)
│       ├── requirements.txt         # Minimal dependencies
│       └── README.md                # Detailed usage guide
│
└── smartspace-ml/                   # Full ML pipeline prototype (RUNNABLE)
    ├── app/
    │   ├── config.py                # Settings loaded from .env
    │   ├── db.py                    # SQLite schema, migrations & CRUD
    │   ├── detector.py              # YOLO detection + ByteTrack tracking
    │   ├── gemma.py                 # Local Gemma client (Mock/Transformers/LlamaCpp)
    │   ├── incidents.py             # Evidence clip packager & SHA-256 hasher
    │   ├── main.py                  # FastAPI REST, WebSocket & MJPEG app
    │   ├── motion.py                # Multi-factor idle & motion scoring
    │   ├── pipeline.py              # Main video processing loop & rule dispatcher
    │   ├── rules.py                 # In-vain, cleaning, intrusion, loitering, crowd
    │   ├── schemas.py               # Pydantic models for zones, events, incidents
    │   └── zones.py                 # Point-in-polygon & ZoneState state machine
    ├── tests/                       # 44 automated tests across 7 test files
    ├── evidence/                    # Saved incident clips & keyframes (runtime)
    ├── .env.example                 # Environment configuration template
    ├── make_test_video.py           # Synthetic test video generator
    ├── run_demo.py                  # Standalone CLI demo runner
    ├── requirements.txt
    └── README.md                    # Full smartspace-ml usage guide
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
| Smart Parking Bay Monitoring | ✅ Working | ✅ Working |
| Crowd Density & Surge | ✅ Working | ✅ Working |
| Queue / Waiting Area Congestion | ✅ Working | ✅ Working |
| Restricted Area Intrusion | ✅ Working | ✅ Working |
| Loitering & Unattended Items | ✅ Working | ✅ Working |
| In-Vain / Idle Detection | — | ✅ Working |
| Local VLM Verification (Gemma) | — | ✅ Working (mock) |
| Incident Evidence Packaging | — | ✅ Working |
| FastAPI REST + WebSocket + MJPEG | — | ✅ Working |

---

## Running Tests

### Webcam Prototype Tests

```bash
cd experiments/webcam_occupancy
source .venv/bin/activate
python -m pytest test_prototype.py -v
```

All 12 tests cover zone logic, occupancy smoothing, detection filtering, and tracker integration.

### SmartSpace ML Pipeline Tests

```bash
cd smartspace-ml
source venv/bin/activate
python -m pytest tests/ -v
```

44 tests across 7 test files covering zones, rules, motion, incidents, database, pipeline, and API endpoints.

---

## Detected Object Classes

The YOLO model supports **80 COCO object classes**, including:

- **People:** `person`
- **Vehicles:** `car`, `motorcycle`, `bicycle`, `bus`, `truck`, `airplane`, `boat`, `train`
- **Personal Belongings:** `backpack`, `handbag`, `suitcase`, `umbrella`
- **Workplace & Electronics:** `laptop`, `cell phone`, `mouse`, `keyboard`, `tv`, `remote`, `book`
- **Furniture:** `chair`, `bench`, `couch`, `dining table`, `bed`
- **Tableware:** `bottle`, `cup`, `wine glass`, `bowl`, `fork`, `knife`, `spoon`

Detection results depend on visibility, lighting, distance, and occlusion in the camera feed.

---

## Limitations & Honest Scope

- **Zone-based occupancy** uses spatial containment (bounding-box centroid inside polygon), **not** posture recognition. A standing person in a zone and a seated person in a zone both register as `OCCUPIED`.
- **COCO classes** are used as-is. A "bag" in the README maps to `backpack`, `handbag`, and `suitcase` — all three are detected and counted separately.
- **Gemma VLM** runs in `mock` mode by default. Production use requires a local Gemma model download (several GB).
- **Performance** is CPU-optimised for MacBook-class hardware using the nano-scale YOLO model. GPU acceleration is supported if available via PyTorch.

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
