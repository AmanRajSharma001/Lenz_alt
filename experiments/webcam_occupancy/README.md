# Lenz — Multi-Class Video Analytics & Occupancy Prototype

This folder contains the standalone, local proof-of-concept (PoC) for **Lenz — Make Every Camera Intelligent with AI**.

It tests real-time multi-class object detection, ByteTrack multi-object tracking, spatial zone analytics, and automated operational rules using your local MacBook webcam or pre-recorded video files.

---

## 1. Root Cause: Why Previous Program Detected Only Humans

The previous prototype contained hardcoded logic that actively discarded all non-human objects:

1. **Explicit Class Filter in Inference Call:**
   The inference call in `webcam_test.py` was executed with `classes=[0]`:
   ```python
   # Previous faulty code:
   results = model.predict(source=frame, classes=[0], ...)
   ```
   In the COCO dataset, class index `0` represents exclusively `person`. This argument caused YOLO to discard cars, bicycles, backpacks, laptops, chairs, bottles, and all other 79 COCO classes at inference time.

2. **Single-Class Data Structures and Rendering:**
   The downstream data structures and HUD display logic were hardcoded for person detections only (`person_detections`), labeling every detected box with hardcoded string `"person"`.

**Fix Applied:**
- Removed `classes=[0]` to run inference across all 80 pretrained COCO classes.
- Switched to the state-of-the-art `yolo11n.pt` model (with `yolov8n.pt` local fallback).
- Integrated ByteTrack (`bytetrack.yaml`) for multi-object tracking, assigning persistent IDs (`#ID`) and tracking dwell times.
- Added multi-class color coding, class-by-class live headcount counters, and spatial-temporal rule evaluations across vehicles, people, luggage, electronics, and furniture.

---

## 2. Supported Object Classes

The model uses real pretrained COCO weights supporting **80 object classes**, including:

* **People:** `person`
* **Vehicles:** `car`, `motorcycle`, `bicycle`, `bus`, `truck`, `airplane`, `boat`, `train`
* **Personal Belongings & Luggage:** `backpack`, `handbag`, `suitcase`, `umbrella`
* **Workplace & Electronics:** `laptop`, `cell phone`, `mouse`, `keyboard`, `tv`, `remote`, `book`
* **Furniture:** `chair`, `bench`, `couch`, `dining table`, `bed`
* **Tableware & Containers:** `bottle`, `cup`, `wine glass`, `bowl`, `fork`, `knife`, `spoon`

*(A complete list of all 80 classes is accessible via `model.names`.)*

---

## 3. Implemented Lenz Use Cases

The prototype implements the core rules specified in the Lenz architecture:

| Lenz Use Case | Status | How It Is Implemented in Prototype |
| :--- | :---: | :--- |
| **A. Person & Vehicle Detection** | **Working** | Simultaneous multi-class bounding boxes, class labels, confidence %, and live breakdown counts. |
| **B. Multi-Object Tracking** | **Working** | ByteTrack assigns persistent IDs (`#ID`) across frames, calculating live dwell duration (`dwell: X.Xs`). |
| **C. Zone-Based Occupancy** | **Working** | Rectangular monitoring zone with temporal debounce smoothing to avoid flicker (`OCCUPIED` / `NOT OCCUPIED` / `UNCERTAIN`). |
| **D. Smart Parking Bay Monitoring** | **Working** | Monitors vehicle classes (`car`, `truck`, `bus`, `motorcycle`) over designated parking bay polygons (`AVAILABLE` vs `OCCUPIED`). |
| **E. Crowd Density & Surge** | **Working** | Headcount threshold monitoring. Triggers `CROWD THRESHOLD REACHED` alert banner when count $\ge$ threshold. |
| **F. Queue / Waiting Area** | **Working** | Tracks persons within a queue polygon and flags congestion if dwell exceeds configured limit. |
| **G. Restricted Area Intrusion** | **Working** | Immediate `CRITICAL: RESTRICTED INTRUSION` banner when a person is detected inside staff-only / sensitive zones. |
| **H. Loitering & Unattended Items** | **Working** | **Loitering:** Flags tracked persons remaining in sensitive zones beyond duration limit.<br/>**Unattended Object:** Flags bags/laptops left without a person nearby for $>10\text{s}$. |
| **I. Video File Input** | **Working** | Accepts pre-recorded MP4/AVI videos via `--source` for repeatable offline validation without a webcam. |

---

## 4. Exact Run Instructions for macOS (MacBook)

Run these exact commands in your macOS Terminal:

### Step 1: Navigate to the experiment folder
```bash
cd /Users/amanrajsharma/Desktop/Projects_Final/Lenz/experiments/webcam_occupancy
```

### Step 2: Create a virtual environment
```bash
python3 -m venv .venv
```

### Step 3: Activate the virtual environment
```bash
source .venv/bin/activate
```

### Step 4: Upgrade pip and install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: If you have already configured `smartspace-ml/venv`, you can also run directly with `../../smartspace-ml/venv/bin/python webcam_test.py`.)*

---

## 5. Launching the Application

### A. Run Live Webcam (Default Mode — All Features Active)
```bash
python webcam_test.py
```
This opens a live window showing:
- Real-time video with color-coded bounding boxes.
- Persistent track IDs (`#ID`) and dwell times.
- Active spatial zones (Table, Restricted Area, Loading Bay).
- Top HUD with FPS and active alert banners (Intrusion, Crowd Surge, Unattended Object).
- Bottom telemetry bar with per-class detection counts.

### B. Keyboard Controls During Live Preview
While the preview window is focused:
* **`q`** or **`ESC`**: Exit application and release webcam cleanly.
* **`m`**: Cycle operational modes (`all` $\rightarrow$ `table` $\rightarrow$ `parking` $\rightarrow$ `restricted` $\rightarrow$ `queue`).
* **`t`**: Toggle track ID and dwell time labels on/off.
* **`z`**: Toggle spatial zone overlays on/off.
* **`+`** / **`-`**: Increase or decrease detection confidence threshold in real time.

### C. Specific Operational Presets
```bash
# Table Occupancy Mode
python webcam_test.py --mode table

# Smart Parking Mode (detects cars, trucks, motorcycles in bays)
python webcam_test.py --mode parking

# Restricted Zone Intrusion Mode
python webcam_test.py --mode restricted

# Queue & Waiting Area Congestion Mode
python webcam_test.py --mode queue
```

### D. Run Against a Pre-recorded Video File
```bash
python webcam_test.py --source ../../smartspace-ml/sample.mp4
```

### E. Run on Alternate Camera (External USB or iPhone Continuity Camera)
```bash
python webcam_test.py --camera 1
```

### F. Filter Specific Classes
```bash
# Only track people, cars, and laptops
python webcam_test.py --classes person,car,laptop
```

### G. Run Headless Automated Test (Server / No Display)
```bash
python webcam_test.py --source ../../smartspace-ml/sample.mp4 --headless --max-frames 50
```

---

## 6. Troubleshooting on macOS

### 1. Camera does not open (`Cannot open Camera #0`)
* **macOS Permissions:** Open **System Settings** $\rightarrow$ **Privacy & Security** $\rightarrow$ **Camera**, and ensure your terminal app (**Terminal**, **iTerm2**, **VS Code**, or **Cursor**) is toggled **ON**.
* **Camera Busy:** Close any application using the camera (Zoom, FaceTime, Google Meet, Photo Booth).
* **Multiple Cameras:** If you have an external monitor, USB camera, or iPhone Continuity Camera connected, try `--camera 1` or `--camera 2`.

### 2. CPU Performance / Framerate
* The prototype defaults to `yolo11n.pt` (~5.4 MB) which runs at **25–35+ FPS on Apple Silicon CPU**.
* If you experience lag on older Intel MacBooks, increase confidence to `--conf 0.45` or filter target classes with `--classes person,car,laptop`.

### 3. Model Weights Resolution
* `yolo11n.pt` and `yolov8n.pt` are bundled directly inside this folder.
* If running offline in a restricted environment, the script will automatically load the local file without attempting network access.

---

## 7. Known Limitations & Scope Boundaries

* **Posture Estimation:** 2D bounding-box detection indicates presence in a zone, but cannot confirm if a person is seated vs. standing without a dedicated pose/keypoint model.
* **Camera Viewing Angle:** Laptop webcams view horizontally or upwards, whereas CCTV cameras mount overhead. Real-world parking and crowd density monitoring require calibrated overhead angles.
* **No Intent Inference:** Alerts (such as loitering or unattended items) are rule-based duration checks requiring human operator review; Lenz never claims to infer theft or criminal intent.
* **Architectural Boundaries:** RTSP/ONVIF camera registration, PostgreSQL storage, Redis message brokering, and cloud dashboards belong to the broader Lenz architecture and are not part of this isolated prototype.
