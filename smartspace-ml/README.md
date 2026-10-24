# SmartSpace ML — AI-Powered Space & Crowd Monitoring

A CPU-friendly Python ML video analytics pipeline for continuous space, queue, and crowd intelligence on CCTV video. Transforms standard recorded video or RTSP camera streams into real-time operational insights without cloud LLM dependencies.

---

## Key Capabilities

- **Zone State Machine (`EMPTY`, `OCCUPIED`, `IDLE_OCCUPIED`, `NEEDS_ATTENTION`):**
  Continuous occupancy tracking per zone with debounce timers (`OCCUPY_ENTER_S`, `OCCUPY_EXIT_S`) in video time.
- **In-Vain (Idle) Detection:**
  Combines normalized track displacement, bounding-box frame-difference pixel motion, and pose keypoint variance (wrists & shoulders) across a 45-second sliding window. Flags occupants who loiter without purposeful objects (laptop, cup, book).
- **Unattended / Dirty Zone Detection:**
  Detects items (cups, bottles, plates) left in `EMPTY` zones exceeding `UNATTENDED_OBJECT_S`.
- **Safety Incident Rules:**
  Deterministic geometric rules for restricted zone intrusion, loitering, crowd surges, falls, fights, and abandoned luggage.
- **Local Vision-Language Model Verification (Gemma 4):**
  Rule-triggered candidate keyframes are dispatched asynchronously to a local Gemma client (running in-process via Hugging Face Transformers, via a local `llama-server` HTTP endpoint, or mock mode) with queue management, per-zone cooldowns, and non-blocking background workers.
- **Incident Evidence Packaging:**
  Rolling in-memory buffer saves 15s pre-incident + 15s post-incident MP4 clips, SHA-256 integrity hashes, keyframes, and optional face blurring.
- **FastAPI REST + MJPEG + WebSocket Service:**
  Complete API for live zone states, dynamic zone CRUD, historical events, incident triage, live annotated video streaming, AI zone suggestion, and natural language analytics queries.

---

## Project Structure

```
smartspace-ml/
├── app/
│   ├── __init__.py
│   ├── config.py          # Settings loaded from .env
│   ├── db.py              # SQLite schema, migrations & CRUD helpers
│   ├── detector.py        # Open-Vocabulary detection (Grounding DINO), SAM 2.1, CLIP & spatial tracking
│   ├── open_vocab_detector.py # Hugging Face Grounding DINO detector & SpatialTracker
│   ├── sam_segmenter.py   # SAM / SAM 2 pixel-level segmentation mask engine
│   ├── table_classifier.py # Zero-shot CLIP table condition classifier
│   ├── gemma.py           # Local Gemma client (Mock, Transformers, LlamaCpp)
│   ├── incidents.py       # Rolling buffer + evidence clip packager & hasher
│   ├── main.py            # FastAPI REST, WebSocket & MJPEG application
│   ├── motion.py          # Multi-factor idle & motion scoring engine
│   ├── pipeline.py        # Main video processing loop & rule dispatcher
│   ├── rules.py           # In-vain, cleaning, intrusion, loitering, crowd rules
│   ├── schemas.py         # Pydantic models for zones, events, incidents
│   └── zones.py           # Point-in-polygon & ZoneState finite state machine
├── evidence/              # Saved incident clips & keyframes
├── tests/
│   └── test_zones.py      # Unit tests for debounce & point-in-polygon logic
├── .env.example           # Environment template
├── .env                   # Local active configuration
├── make_test_video.py     # Synthetic test video generator
├── run_demo.py            # Standalone CLI demo runner
├── requirements.txt       # Python package dependencies
├── zones.json             # Sample calibrated zones
└── README.md
```

---

## Quickstart Installation

### 1. Prerequisites
- Python 3.10+
- `ffmpeg` installed on your system (optional, for video encoding utilities)

### 2. Setup Virtual Environment

```bash
cd smartspace-ml
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the CLI Demo

### 1. Generate Synthetic Test Video

If you do not have a recorded CCTV video ready, generate a 90-second synthetic test video:

```bash
python make_test_video.py --out sample.mp4 --fps 25 --duration 90
```

This creates `sample.mp4` featuring:
- A walking person traversing the dining table zone.
- A stationary person sitting in the queue zone.
- An abandoned item appearing on the dining table.

### 2. Execute Demo

Run the end-to-end pipeline in `mock` mode (runs completely on CPU without downloading heavy LLM weights):

```bash
python run_demo.py --video sample.mp4 --out out.mp4 --zones zones.json
```

Output:
- Generates `out.mp4` with colored zone overlays (green: `EMPTY`, amber: `OCCUPIED`, red: `IDLE_OCCUPIED`), bounding boxes, track IDs, and OSD timers.
- Prints a terminal summary table of all zones and detected events.

---

## Running the FastAPI Service

### 1. Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive OpenAPI docs are accessible at:
[http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Core API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status |
| `POST` | `/pipeline/start` | Start background processing loop (`{"source": "sample.mp4", "camera_id": "cam_01"}`) |
| `POST` | `/pipeline/stop` | Stop background video pipeline |
| `GET` | `/pipeline/status` | Current pipeline execution status and video time |
| `GET` | `/zones` | List all active monitoring zones |
| `POST` | `/zones` | Create new zone polygon with custom thresholds |
| `GET` | `/zones/state` | Current live state of every zone (`EMPTY`, `OCCUPIED`, dwell time, motion) |
| `GET` | `/events` | Filter historical events by `zone_id`, `type`, and `limit` |
| `GET` | `/incidents` | List confirmed incidents with evidence clip paths |
| `PATCH` | `/incidents/{id}` | Update incident review status (`reviewed`, `dismissed`, `escalated`) |
| `GET` | `/stream/{camera_id}` | Live MJPEG video stream with visual annotations |
| `WS` | `/ws` | Real-time WebSocket event and state broadcast channel |
| `POST` | `/zones/suggest` | Upload a CCTV frame to receive AI-suggested zone polygons |
| `POST` | `/query` | Natural language analytics query (e.g., *"How many people visited today?"*) |

---

## Gemma 4 Model Modes & Configuration

The local vision-language model verifies rule-triggered candidate incidents. Set `GEMMA_MODE` in `.env` to switch between backends:

### Mode 1: `mock` (Default)
Returns instant, realistic synthetic JSON verdicts. Zero GPU, zero weights download. Ideal for unit tests, local development, and CI/CD pipelines.

```ini
GEMMA_MODE=mock
```

### Mode 2: `transformers` (In-Process PyTorch)
Loads the Hugging Face vision-language model directly into the Python process using `AutoProcessor` and `device_map="auto"`:

```ini
GEMMA_MODE=transformers
GEMMA_MODEL_ID=google/gemma-3-4b-it
GEMMA_DEVICE=auto
GEMMA_MAX_NEW_TOKENS=200
```

1. Log in to Hugging Face:
   ```bash
   huggingface-cli login
   ```
2. Accept the Gemma license terms on the Hugging Face model card.
3. The model will download and cache automatically on first start.

### Mode 3: `llamacpp` (High-Performance Local Server)
Offloads vision-language inference to an external, hardware-accelerated `llama.cpp` server via an OpenAI-compatible HTTP interface:

```ini
GEMMA_MODE=llamacpp
GEMMA_LLAMACPP_URL=http://localhost:8080
GEMMA_MAX_NEW_TOKENS=200
```

#### Starting `llama-server`:
1. Build `llama.cpp` with your preferred hardware acceleration (Metal on Apple Silicon, CUDA on Linux):
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   cmake -B build -DGGML_METAL=ON   # or -DGGML_CUDA=ON
   cmake --build build --config Release -j
   ```
2. Download the GGUF model and mmproj vision projector:
   ```bash
   huggingface-cli download google/gemma-3-4b-it-GGUF \
     gemma-3-4b-it-Q4_K_M.gguf \
     mmproj-model-f16.gguf \
     --local-dir ./models
   ```
3. Launch `llama-server`:
   ```bash
   ./build/bin/llama-server \
     -m ./models/gemma-3-4b-it-Q4_K_M.gguf \
     --mmproj ./models/mmproj-model-f16.gguf \
     --port 8080 \
     --ctx-size 4096 \
     --n-gpu-layers 99
   ```

---

## Running Unit Tests

Execute the unit test suite covering point-in-polygon ray-casting and ZoneState debounce transitions:

```bash
python tests/test_zones.py
```
*(or via pytest: `pytest tests/test_zones.py -v`)*

---

## Incident Evidence Storage & Privacy

When a safety rule is confirmed by Gemma with confidence $\ge$ `ALERT_CONFIDENCE`:
1. Pre- and post-incident frames are extracted from the in-memory ring buffer.
2. A compressed `.mp4` clip is saved to `evidence/<incident_id>/clip.mp4`.
3. A SHA-256 checksum is computed and saved to the database for forensic chain-of-custody.
4. Keyframes (`keyframe_00.jpg`) are saved alongside the clip.
5. If `BLUR_FACES=true` in `.env`, OpenCV Haar cascade blurring is automatically applied across saved evidence.
