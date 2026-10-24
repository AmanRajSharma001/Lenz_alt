# Lenz — Documentation Assets & Visual Index

This index catalogs the technical documentation assets, system architecture diagrams, and illustrative scenario mockups for the **Lenz AI-Powered Video Intelligence Platform**.

In strict accordance with technical documentation accuracy and evidence-based reporting:
- All visual assets follow a **professional monochrome technical aesthetic** (white background, black text, thin grey borders, and grayscale fills).
- The repository is currently at the initial repository setup stage (MIT License initialized). All architectural designs, data models, pipelines, and workflows are designated as **Proposed Architecture**.
- No synthetic values or simulated scenario curves are presented as empirical production benchmarks. All chart mockups are explicitly designated as **Illustrative Synthetic Data — Not Actual Lenz Results**.

---

## 1. High-Level System Visual Assets (`docs/images/`)

| Asset Filename | Visual Title | Format & Style | Nature & Status | Technical Description & Purpose |
| :--- | :--- | :--- | :--- | :--- |
| [`lenz-hero.png`](./images/lenz-hero.png) | **Lenz Platform Specification Banner** | Monochrome PNG / SVG | Proposed Architecture | System specification schematic displaying video ingestion, deep learning inference, spatial rule engine, and operator verification stages. |
| [`problem-vs-solution.png`](./images/problem-vs-solution.png) | **Comparative System Analysis** | Monochrome PNG / SVG | Technical Comparison | High-contrast comparison contrasting passive CCTV recording (fatigue, data silos, delayed investigation) with the proposed active video intelligence architecture. |
| [`system-workflow.png`](./images/system-workflow.png) | **End-to-End Processing Workflow** | Monochrome PNG / SVG | Proposed Architecture | Detailed processing pipeline from camera stream ingestion through decimation, YOLO detection, ByteTrack tracking, rule evaluation, persistence, and human review. |
| [`system-architecture.png`](./images/system-architecture.png) | **Layered System Architecture** | Monochrome PNG / SVG | Proposed Architecture | Decoupled 5-tier architecture: Presentation & Operator Tier, Application & Gateway Tier, Computer Vision & Inference Tier, Ingestion Tier, and Persistence Tier. |
| [`database-er.png`](./images/database-er.png) | **Proposed Database Design** | Monochrome PNG / SVG | Proposed (Not Yet Implemented) | Relational entity-relationship model covering `USERS`, `LOCATIONS`, `CAMERAS`, `MONITORING_ZONES`, `DETECTION_MODELS`, `DETECTION_EVENTS`, `ALERTS`, `PROCESSING_JOBS`, and `AUDIT_LOGS`. |
| [`ai-pipeline.png`](./images/ai-pipeline.png) | **AI Computer Vision Pipeline** | Monochrome PNG / SVG | Proposed Architecture | Stage-by-stage data transformation: frame sampling, preprocessing (640x640 letterbox), YOLO detection, ByteTrack tracking, spatial-temporal rules, and event dispatch. |
| [`use-case.png`](./images/use-case.png) | **System Actor Use Case Diagram** | Monochrome PNG / SVG | Proposed Architecture | UML-style functional mapping between Administrator, Security Operator, Facility Manager, Camera Sources, Notification Services, and the Lenz Core Platform. |
| [`data-flow.png`](./images/data-flow.png) | **Level-1 Data Flow Diagram (DFD)** | Monochrome PNG / SVG | Proposed Architecture | Data movement and state transitions between ingestion buffers, inference tensors, spatial zone checks, PostgreSQL/S3 stores, and WebSocket push channels. |
| [`dashboard-mockup.png`](./images/dashboard-mockup.png) | **Dashboard Wireframe Specification** | Monochrome PNG / SVG | Proposed UI Wireframe | Technical wireframe of operator portal: multi-stream live grid, bounding box overlays, parking status slots, real-time incident verification feed, and fleet health telemetry. |

---

## 2. Illustrative Operational Analytics (`docs/images/analytics/`)

> [!IMPORTANT]
> **Data Accuracy Notice:** The graphs below demonstrate proposed chart specifications, API schemas, and data visualization models. They contain **Illustrative Synthetic Data — Not Actual Lenz Results**. Lenz has not completed physical hardware deployment or empirical benchmark testing.

| Asset Filename | Metric Title | Format & Style | Nature & Status | Technical Purpose |
| :--- | :--- | :--- | :--- | :--- |
| [`crowd-density-trend.png`](./images/analytics/crowd-density-trend.png) | **Crowd Density Over 24 Hours** | Monochrome Line Chart | Illustrative Synthetic Data | Demonstrates temporal crowd surge profile against a configurable 80% critical safety threshold. |
| [`parking-occupancy-trend.png`](./images/analytics/parking-occupancy-trend.png) | **Parking Occupancy by Zone** | Monochrome Line Chart | Illustrative Synthetic Data | Demonstrates multi-zone occupancy rate tracking across surface parking, covered garages, and EV bays over an operational day. |
| [`events-by-category.png`](./images/analytics/events-by-category.png) | **Detections by Operational Category** | Monochrome Bar Chart | Illustrative Synthetic Data | Demonstrates query aggregation schema across crowd surges, queue delays, parking occupancy, unattended areas, and zone intrusions. |
| [`alert-severity-distribution.png`](./images/analytics/alert-severity-distribution.png) | **Alert Severity Triage Specification** | Monochrome Metric Cards | Design Specification | Demonstrates triage priority levels (Critical, Warning, Info, Resolved) and target operator response times. |
| [`hourly-event-frequency.png`](./images/analytics/hourly-event-frequency.png) | **Hourly Event Trigger Distribution** | Monochrome Bar Chart | Illustrative Synthetic Data | Demonstrates time-of-day query aggregation across a 24-hour operational cycle. |
| [`camera-processing-latency.png`](./images/analytics/camera-processing-latency.png) | **Projected Latency Scaling vs Streams** | Monochrome Line Chart | Theoretical Design Analysis | Analytical projection based on published YOLOv8 benchmarks comparing NVIDIA GPU (TensorRT FP16) and host CPU (ONNX Runtime). |

---

## 3. Editable Mermaid Diagram Sources (`docs/diagrams/`)

Editable Mermaid (`.mmd`) sources are maintained in `docs/diagrams/`:

- [`system-workflow.mmd`](./diagrams/system-workflow.mmd): Pipeline and error branches in monochrome styling.
- [`system-architecture.mmd`](./diagrams/system-architecture.mmd): Layered system tiers and inter-service communications.
- [`database-er.mmd`](./diagrams/database-er.mmd): Relational schema entities, primary keys, and foreign keys.
- [`ai-pipeline.mmd`](./diagrams/ai-pipeline.mmd): Computer vision and tracking stages.
- [`use-case.mmd`](./diagrams/use-case.mmd): System actor interactions and use case boundaries.
- [`data-flow.mmd`](./diagrams/data-flow.mmd): Level-1 DFD processes, data stores, and data movements.

---

## 4. README Flowcharts & Visual Architecture (`docs/images/readme/`)

The following visual flowcharts and schematics are integrated directly into the root [`README.md`](../README.md):

| Asset Filename | Visual Title | Section Placement | Purpose |
| :--- | :--- | :--- | :--- |
| [`problem-statement-flowchart.png`](./images/readme/problem-statement-flowchart.png) | **Problem Statement Flowchart** | Section 1 (Problem Statement) | Illustrates the breakdown of manual CCTV surveillance, cognitive fatigue, and lack of real-time telemetry. |
| [`targeted-users-diagram.png`](./images/readme/targeted-users-diagram.png) | **Target Users Diagram** | Section 2 (Target Users) | Maps operational user segments (Transit, Retail, Security, Parking, Healthcare). |
| [`system-workflow-flowchart.png`](./images/readme/system-workflow-flowchart.png) | **System Workflow Flowchart** | Section 4 (System Workflow) | Step-by-step frame ingestion, inference, tracking, rule validation, and alert dispatch pipeline. |
| [`proposed-solution-flowchart.png`](./images/readme/proposed-solution-flowchart.png) | **Proposed Solution Flowchart** | Section 6 (Proposed Solution) | Schematic representation of the Lenz AI intelligence platform capabilities. |
| [`system-architecture-diagram.png`](./images/readme/system-architecture-diagram.png) | **System Architecture Diagram** | Section 7 (System Architecture) | Multi-tier architectural topology covering client, API gateway, computer vision workers, and data stores. |

---

## 5. Asset Generation & Reproduction

All assets are generated from vector SVGs and converted to PNG via `sips`:

```bash
# Execute asset regeneration script
python3 scripts/generate_assets.py
```
