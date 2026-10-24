#!/usr/bin/env python3
"""
Lenz Platform - Professional Monochrome Asset Generator
Generates clean, publication-grade technical diagrams, wireframes, and charts
strictly adhering to monochrome rules (black, white, light/medium/dark grey)
with white backgrounds and explicit [Proposed] / [Illustrative] labelling.
"""

import os
import subprocess

DOCS_IMG = "/Users/amanrajsharma/Desktop/Projects_Final/Lenz/docs/images"
DOCS_ANALYTICS = os.path.join(DOCS_IMG, "analytics")
os.makedirs(DOCS_IMG, exist_ok=True)
os.makedirs(DOCS_ANALYTICS, exist_ok=True)

def render(svg_str, png_name, is_analytics=False):
    target_dir = DOCS_ANALYTICS if is_analytics else DOCS_IMG
    png_path = os.path.join(target_dir, png_name)
    svg_path = png_path.replace(".png", ".svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str.strip())
    subprocess.run(["sips", "-s", "format", "png", svg_path, "--out", png_path], check=True, stdout=subprocess.DEVNULL)
    print(f"Generated: {png_name}")

# ========================================================
# 1. HERO BANNER (1200x440) - Monochrome Engineering Spec
# ========================================================
def gen_hero():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="440" viewBox="0 0 1200 440">
  <!-- Background & Outer Border -->
  <rect width="1200" height="440" fill="#ffffff"/>
  <rect x="1" y="1" width="1198" height="438" fill="none" stroke="#ced4da" stroke-width="1"/>

  <!-- Top System Header Bar -->
  <rect x="30" y="25" width="1140" height="30" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
  <text x="45" y="45" fill="#495057" font-family="system-ui, sans-serif" font-size="11" font-weight="bold" letter-spacing="1">
    LENZ PLATFORM SPECIFICATION • REPOSITORY IMPLEMENTATION STATUS: ARCHITECTURAL BLUEPRINT [PROPOSED]
  </text>
  <text x="1155" y="45" text-anchor="end" fill="#6c757d" font-family="monospace" font-size="11">REF: IEEE/ISO-CV-ARCH</text>

  <!-- Title & Tagline -->
  <text x="600" y="115" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="38" font-weight="900" letter-spacing="3">LENZ</text>
  <text x="600" y="145" text-anchor="middle" fill="#333333" font-family="system-ui, sans-serif" font-size="16" font-weight="600">AI-Powered Video Intelligence Platform</text>
  <text x="600" y="168" text-anchor="middle" fill="#666666" font-family="system-ui, sans-serif" font-size="13">"Every Camera. Smarter Insights. Safer Spaces."</text>

  <!-- High-Level Pipeline Schematic -->
  <g transform="translate(60, 205)" font-family="system-ui, sans-serif" font-size="11">
    <!-- Box 1: Sources -->
    <rect width="190" height="90" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <rect width="190" height="24" fill="#343a40"/>
    <text x="95" y="16" text-anchor="middle" fill="#ffffff" font-weight="bold">1. VIDEO INPUT [PROPOSED]</text>
    <text x="95" y="45" text-anchor="middle" fill="#212529">RTSP / ONVIF IP Streams</text>
    <text x="95" y="65" text-anchor="middle" fill="#495057">MP4 / AVI File Uploads</text>
    <text x="95" y="80" text-anchor="middle" fill="#6c757d">5 - 15 FPS Decimation</text>

    <!-- Arrow 1 to 2 -->
    <path d="M 190 45 L 230 45" stroke="#212529" stroke-width="1.5" fill="none"/>
    <polygon points="230,45 224,41 224,49" fill="#212529"/>

    <!-- Box 2: Ingestion & Buffer -->
    <rect x="235" width="190" height="90" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <rect x="235" width="190" height="24" fill="#343a40"/>
    <text x="330" y="16" text-anchor="middle" fill="#ffffff" font-weight="bold">2. INGESTION [PROPOSED]</text>
    <text x="330" y="45" text-anchor="middle" fill="#212529">OpenCV / FFmpeg Pipeline</text>
    <text x="330" y="65" text-anchor="middle" fill="#495057">Shared Ring Buffers</text>
    <text x="330" y="80" text-anchor="middle" fill="#6c757d">Automatic Reconnect</text>

    <!-- Arrow 2 to 3 -->
    <path d="M 425 45 L 465 45" stroke="#212529" stroke-width="1.5" fill="none"/>
    <polygon points="465,45 459,41 459,49" fill="#212529"/>

    <!-- Box 3: AI Inference & Tracking -->
    <rect x="470" width="200" height="90" fill="#ffffff" stroke="#111111" stroke-width="2"/>
    <rect x="470" width="200" height="24" fill="#111111"/>
    <text x="570" y="16" text-anchor="middle" fill="#ffffff" font-weight="bold">3. CV &amp; TRACKING [PROPOSED]</text>
    <text x="570" y="45" text-anchor="middle" fill="#111111" font-weight="bold">YOLOv8/v11 (ONNX / TensorRT)</text>
    <text x="570" y="65" text-anchor="middle" fill="#333333">ByteTrack Trajectory IDs</text>
    <text x="570" y="80" text-anchor="middle" fill="#6c757d">Person &amp; Vehicle Classes</text>

    <!-- Arrow 3 to 4 -->
    <path d="M 670 45 L 710 45" stroke="#212529" stroke-width="1.5" fill="none"/>
    <polygon points="710,45 704,41 704,49" fill="#212529"/>

    <!-- Box 4: Event Engine -->
    <rect x="715" width="190" height="90" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <rect x="715" width="190" height="24" fill="#343a40"/>
    <text x="810" y="16" text-anchor="middle" fill="#ffffff" font-weight="bold">4. RULE ENGINE [PROPOSED]</text>
    <text x="810" y="45" text-anchor="middle" fill="#212529">Polygon ROI Ray-Casting</text>
    <text x="810" y="65" text-anchor="middle" fill="#495057">Temporal Dwell Counters</text>
    <text x="810" y="80" text-anchor="middle" fill="#6c757d">Debounce &amp; Deduplication</text>

    <!-- Arrow 4 to 5 -->
    <path d="M 905 45 L 945 45" stroke="#212529" stroke-width="1.5" fill="none"/>
    <polygon points="945,45 939,41 939,49" fill="#212529"/>

    <!-- Box 5: Dispatch & UI -->
    <rect x="950" width="185" height="90" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <rect x="950" width="185" height="24" fill="#343a40"/>
    <text x="1042" y="16" text-anchor="middle" fill="#ffffff" font-weight="bold">5. DISPATCH [PROPOSED]</text>
    <text x="1042" y="45" text-anchor="middle" fill="#212529">PostgreSQL &amp; S3 Evidence</text>
    <text x="1042" y="65" text-anchor="middle" fill="#495057">WebSocket Push Gateway</text>
    <text x="1042" y="80" text-anchor="middle" fill="#6c757d">Human-in-the-Loop Review</text>
  </g>

  <!-- Target Environment Capabilities Row -->
  <g transform="translate(60, 335)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="1075" height="65" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
    <text x="20" y="24" fill="#212529" font-weight="bold">EVALUATED TARGET DOMAINS &amp; PLANNED CAPABILITIES:</text>
    <text x="20" y="46" fill="#495057">
      • Transit Hubs (Crowd Surge)  |  • Restaurants &amp; Cafes (Queue &amp; Table Turn)  |  • Parking Lots (Bay Occupancy)  |  • Healthcare &amp; Retail (Waiting Areas)  |  • Perimeters (Intrusion)
    </text>
  </g>

  <!-- Footer Disclaimer -->
  <text x="600" y="425" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Technical Architecture Blueprint • Source code repository initialized under MIT License • Real-time processing specifications are proposed designs
  </text>
</svg>'''
    render(svg, "lenz-hero.png")

# ========================================================
# 2. PROBLEM VS SOLUTION (1200x560) - Technical Comparison
# ========================================================
def gen_problem_vs_solution():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560">
  <rect width="1200" height="560" fill="#ffffff"/>
  <rect x="1" y="1" width="1198" height="558" fill="none" stroke="#ced4da" stroke-width="1"/>

  <!-- Title -->
  <text x="600" y="45" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="22" font-weight="bold">
    Comparative System Analysis: Traditional CCTV Surveillance vs. Lenz Video Intelligence
  </text>
  <text x="600" y="70" text-anchor="middle" fill="#6c757d" font-family="system-ui, sans-serif" font-size="13">
    Technical differentiation between passive video archiving and active automated event extraction
  </text>

  <!-- Left: Traditional CCTV -->
  <g transform="translate(60, 100)">
    <rect width="520" height="400" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
    <rect width="520" height="35" fill="#495057"/>
    <text x="260" y="23" text-anchor="middle" fill="#ffffff" font-family="system-ui, sans-serif" font-size="13" font-weight="bold">
      TRADITIONAL PASSIVE SURVEILLANCE (CURRENT STATE)
    </text>

    <g transform="translate(25, 60)" font-family="system-ui, sans-serif">
      <text x="0" y="15" fill="#111111" font-size="13" font-weight="bold">• Continuous Unanalyzed Video Recording</text>
      <text x="16" y="35" fill="#495057" font-size="12">Terabytes of footage are archived; over 99% is never reviewed unless an incident is</text>
      <text x="16" y="52" fill="#495057" font-size="12">reported after the loss or damage has already occurred.</text>

      <text x="0" y="90" fill="#111111" font-size="13" font-weight="bold">• Operator Attention Fatigue</text>
      <text x="16" y="110" fill="#495057" font-size="12">Human monitoring performance declines drastically after 20 minutes across multi-screen</text>
      <text x="16" y="127" fill="#495057" font-size="12">grids, resulting in missed crowd surges, intrusions, and bottlenecks.</text>

      <text x="0" y="165" fill="#111111" font-size="13" font-weight="bold">• Absence of Operational Telemetry</text>
      <text x="16" y="185" fill="#495057" font-size="12">Managers lack automated data on queue lengths, table dwell times, and parking capacity,</text>
      <text x="16" y="202" fill="#495057" font-size="12">relying on manual counting, delayed walkthroughs, and guesswork.</text>

      <text x="0" y="240" fill="#111111" font-size="13" font-weight="bold">• Forensic-Only Investigation Workflow</text>
      <text x="16" y="260" fill="#495057" font-size="12">Hours are spent manually scrubbing timelines to locate relevant footage after an event,</text>
      <text x="16" y="277" fill="#495057" font-size="12">creating operational delays and high investigative costs.</text>
    </g>

    <rect x="20" y="350" width="480" height="30" fill="#e9ecef" stroke="#ced4da" stroke-width="1"/>
    <text x="260" y="370" text-anchor="middle" fill="#343a40" font-family="system-ui, sans-serif" font-size="11" font-weight="bold">
      RESULT: Blind spots, delayed intervention, and unutilized camera infrastructure.
    </text>
  </g>

  <!-- Right: Lenz Platform -->
  <g transform="translate(620, 100)">
    <rect width="520" height="400" fill="#ffffff" stroke="#111111" stroke-width="2"/>
    <rect width="520" height="35" fill="#111111"/>
    <text x="260" y="23" text-anchor="middle" fill="#ffffff" font-family="system-ui, sans-serif" font-size="13" font-weight="bold">
      LENZ VIDEO INTELLIGENCE PLATFORM (PROPOSED ARCHITECTURE)
    </text>

    <g transform="translate(25, 60)" font-family="system-ui, sans-serif">
      <text x="0" y="15" fill="#111111" font-size="13" font-weight="bold">• Automated Computer Vision Ingestion</text>
      <text x="16" y="35" fill="#333333" font-size="12">Standard RTSP/ONVIF streams are continuously processed using lightweight YOLO models</text>
      <text x="16" y="52" fill="#333333" font-size="12">and ByteTrack tracking to detect persons, vehicles, and relevant objects.</text>

      <text x="0" y="90" fill="#111111" font-size="13" font-weight="bold">• Human-in-the-Loop Incident Prioritization</text>
      <text x="16" y="110" fill="#333333" font-size="12">Filters background activity and dispatches verified, deduplicated alerts with cropped</text>
      <text x="16" y="127" fill="#333333" font-size="12">evidence snapshots for prompt operator review and acknowledgment.</text>

      <text x="0" y="165" fill="#111111" font-size="13" font-weight="bold">• Continuous Operational Telemetry</text>
      <text x="16" y="185" fill="#333333" font-size="12">Calculates real-time crowd density ratios, queue lengths, table turnaround dwell times,</text>
      <text x="16" y="202" fill="#333333" font-size="12">and parking bay occupancy without requiring specialized proprietary sensors.</text>

      <text x="0" y="240" fill="#111111" font-size="13" font-weight="bold">• Indexed Event Storage &amp; Search</text>
      <text x="16" y="260" fill="#333333" font-size="12">Events are stored as structured JSON records in PostgreSQL, enabling millisecond queries</text>
      <text x="16" y="277" fill="#333333" font-size="12">by camera, zone, severity, and timestamp with linked keyframe snapshots.</text>
    </g>

    <rect x="20" y="350" width="480" height="30" fill="#f8f9fa" stroke="#343a40" stroke-width="1"/>
    <text x="260" y="370" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="11" font-weight="bold">
      RESULT: Proactive awareness, reduced operator fatigue, and maximized camera value.
    </text>
  </g>

  <!-- Note -->
  <text x="600" y="535" text-anchor="middle" fill="#6c757d" font-family="system-ui, sans-serif" font-size="11">
    Design Note: Lenz augments existing surveillance workflows. It does not replace human operators or claim autonomous intent detection.
  </text>
</svg>'''
    render(svg, "problem-vs-solution.png")

# ========================================================
# 3. SYSTEM WORKFLOW (1200x660) - Monochrome Pipeline
# ========================================================
def gen_system_workflow():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="660" viewBox="0 0 1200 660">
  <rect width="1200" height="660" fill="#ffffff"/>
  <rect x="1" y="1" width="1198" height="658" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="600" y="40" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="20" font-weight="bold">
    Lenz End-to-End Processing Workflow [Proposed Architecture]
  </text>
  <text x="600" y="62" text-anchor="middle" fill="#6c757d" font-family="system-ui, sans-serif" font-size="12">
    Data flow from camera stream ingestion through deep learning inference, spatial rule evaluation, persistence, and operator review
  </text>

  <!-- 5 Workflow Columns -->
  <!-- Col 1: Ingestion -->
  <g transform="translate(50, 95)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="190" height="460" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
    <rect width="190" height="32" fill="#495057"/>
    <text x="95" y="21" text-anchor="middle" fill="#ffffff" font-weight="bold">1. VIDEO INGESTION [PROPOSED]</text>

    <g transform="translate(15, 50)">
      <rect width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="22" text-anchor="middle" fill="#111111" font-weight="bold">RTSP / ONVIF Input</text>
      <text x="80" y="38" text-anchor="middle" fill="#6c757d">IP Camera Feeds</text>

      <rect y="70" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="92" text-anchor="middle" fill="#111111" font-weight="bold">Video File Upload</text>
      <text x="80" y="108" text-anchor="middle" fill="#6c757d">MP4, MKV, AVI Files</text>

      <rect y="140" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="162" text-anchor="middle" fill="#111111" font-weight="bold">Demux &amp; Decode</text>
      <text x="80" y="178" text-anchor="middle" fill="#6c757d">OpenCV / FFmpeg</text>

      <rect y="210" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="232" text-anchor="middle" fill="#111111" font-weight="bold">Frame Decimation</text>
      <text x="80" y="248" text-anchor="middle" fill="#6c757d">Sampling (5 - 15 FPS)</text>

      <!-- Error Branch -->
      <rect y="280" width="160" height="60" fill="#f1f3f5" stroke="#868e96" stroke-width="1" stroke-dasharray="4,3"/>
      <text x="80" y="302" text-anchor="middle" fill="#212529" font-weight="bold">[Error Path] Stream Drop</text>
      <text x="80" y="320" text-anchor="middle" fill="#6c757d">Exponential Backoff</text>
      <text x="80" y="334" text-anchor="middle" fill="#6c757d">Auto-Reconnect Daemon</text>
    </g>
  </g>

  <!-- Arrow 1 to 2 -->
  <path d="M 245 320 L 275 320" stroke="#111111" stroke-width="2" fill="none"/>
  <polygon points="275,320 269,316 269,324" fill="#111111"/>

  <!-- Col 2: CV Inference -->
  <g transform="translate(280, 95)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="190" height="460" fill="#ffffff" stroke="#212529" stroke-width="2"/>
    <rect width="190" height="32" fill="#212529"/>
    <text x="95" y="21" text-anchor="middle" fill="#ffffff" font-weight="bold">2. CV &amp; INFERENCE [PROPOSED]</text>

    <g transform="translate(15, 50)">
      <rect width="160" height="50" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="22" text-anchor="middle" fill="#111111" font-weight="bold">Preprocessing</text>
      <text x="80" y="38" text-anchor="middle" fill="#6c757d">Letterbox 640x640, Scale</text>

      <rect y="70" width="160" height="50" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="92" text-anchor="middle" fill="#111111" font-weight="bold">YOLO Detection</text>
      <text x="80" y="108" text-anchor="middle" fill="#6c757d">BBoxes + Confidence</text>

      <rect y="140" width="160" height="50" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="162" text-anchor="middle" fill="#111111" font-weight="bold">ByteTrack Associator</text>
      <text x="80" y="178" text-anchor="middle" fill="#6c757d">Kalman Filter Tracks</text>

      <rect y="210" width="160" height="50" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="232" text-anchor="middle" fill="#111111" font-weight="bold">Tracklet Buffer</text>
      <text x="80" y="248" text-anchor="middle" fill="#6c757d">Motion Trajectories</text>

      <!-- Error Branch -->
      <rect y="280" width="160" height="60" fill="#f1f3f5" stroke="#868e96" stroke-width="1" stroke-dasharray="4,3"/>
      <text x="80" y="302" text-anchor="middle" fill="#212529" font-weight="bold">[Error Path] GPU Fault</text>
      <text x="80" y="320" text-anchor="middle" fill="#6c757d">Skip Frame &amp; Telemetry</text>
      <text x="80" y="334" text-anchor="middle" fill="#6c757d">CPU Fallback if Configured</text>
    </g>
  </g>

  <!-- Arrow 2 to 3 -->
  <path d="M 475 320 L 505 320" stroke="#111111" stroke-width="2" fill="none"/>
  <polygon points="505,320 499,316 499,324" fill="#111111"/>

  <!-- Col 3: Rule Engine -->
  <g transform="translate(510, 95)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="190" height="460" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
    <rect width="190" height="32" fill="#495057"/>
    <text x="95" y="21" text-anchor="middle" fill="#ffffff" font-weight="bold">3. RULE ENGINE [PROPOSED]</text>

    <g transform="translate(15, 50)">
      <rect width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="22" text-anchor="middle" fill="#111111" font-weight="bold">Polygon ROI Test</text>
      <text x="80" y="38" text-anchor="middle" fill="#6c757d">Ray-Casting Intersect</text>

      <rect y="70" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="92" text-anchor="middle" fill="#111111" font-weight="bold">Dwell Time Threshold</text>
      <text x="80" y="108" text-anchor="middle" fill="#6c757d">Tracklet Duration in Zone</text>

      <rect y="140" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="162" text-anchor="middle" fill="#111111" font-weight="bold">Density &amp; Queue Logic</text>
      <text x="80" y="178" text-anchor="middle" fill="#6c757d">Headcount per Area</text>

      <rect y="210" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="232" text-anchor="middle" fill="#111111" font-weight="bold">Deduplication Filter</text>
      <text x="80" y="248" text-anchor="middle" fill="#6c757d">Debounce Cooldown Window</text>

      <!-- Discard Branch -->
      <rect y="280" width="160" height="60" fill="#f1f3f5" stroke="#868e96" stroke-width="1" stroke-dasharray="4,3"/>
      <text x="80" y="302" text-anchor="middle" fill="#212529" font-weight="bold">[Filter] Duplicate Event</text>
      <text x="80" y="320" text-anchor="middle" fill="#6c757d">Increment Active Counter</text>
      <text x="80" y="334" text-anchor="middle" fill="#6c757d">Suppress Notification</text>
    </g>
  </g>

  <!-- Arrow 3 to 4 -->
  <path d="M 705 320 L 735 320" stroke="#111111" stroke-width="2" fill="none"/>
  <polygon points="735,320 729,316 729,324" fill="#111111"/>

  <!-- Col 4: Storage & Alerts -->
  <g transform="translate(740, 95)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="190" height="460" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
    <rect width="190" height="32" fill="#495057"/>
    <text x="95" y="21" text-anchor="middle" fill="#ffffff" font-weight="bold">4. DISPATCH [PROPOSED]</text>

    <g transform="translate(15, 50)">
      <rect width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="22" text-anchor="middle" fill="#111111" font-weight="bold">PostgreSQL Database</text>
      <text x="80" y="38" text-anchor="middle" fill="#6c757d">Event &amp; Zone Records</text>

      <rect y="70" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="92" text-anchor="middle" fill="#111111" font-weight="bold">S3 / MinIO Store</text>
      <text x="80" y="108" text-anchor="middle" fill="#6c757d">Keyframe Snapshot JPEGs</text>

      <rect y="140" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="162" text-anchor="middle" fill="#111111" font-weight="bold">WebSocket Hub</text>
      <text x="80" y="178" text-anchor="middle" fill="#6c757d">Broadcast JSON Payload</text>

      <rect y="210" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="232" text-anchor="middle" fill="#111111" font-weight="bold">External Webhooks</text>
      <text x="80" y="248" text-anchor="middle" fill="#6c757d">Slack / Email Dispatch</text>

      <!-- Failure Branch -->
      <rect y="280" width="160" height="60" fill="#f1f3f5" stroke="#868e96" stroke-width="1" stroke-dasharray="4,3"/>
      <text x="80" y="302" text-anchor="middle" fill="#212529" font-weight="bold">[Error Path] Push Failure</text>
      <text x="80" y="320" text-anchor="middle" fill="#6c757d">Dead-Letter Queue</text>
      <text x="80" y="334" text-anchor="middle" fill="#6c757d">Retry with Backoff</text>
    </g>
  </g>

  <!-- Arrow 4 to 5 -->
  <path d="M 935 320 L 965 320" stroke="#111111" stroke-width="2" fill="none"/>
  <polygon points="965,320 959,316 959,324" fill="#111111"/>

  <!-- Col 5: Operator Review -->
  <g transform="translate(970, 95)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="180" height="460" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
    <rect width="180" height="32" fill="#495057"/>
    <text x="90" y="21" text-anchor="middle" fill="#ffffff" font-weight="bold">5. HUMAN REVIEW [PROPOSED]</text>

    <g transform="translate(10, 50)">
      <rect width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="22" text-anchor="middle" fill="#111111" font-weight="bold">Operator UI Grid</text>
      <text x="80" y="38" text-anchor="middle" fill="#6c757d">Live Feed Overlays</text>

      <rect y="70" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="92" text-anchor="middle" fill="#111111" font-weight="bold">Evidence Inspection</text>
      <text x="80" y="108" text-anchor="middle" fill="#6c757d">Review Snapshot Crop</text>

      <rect y="140" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="162" text-anchor="middle" fill="#111111" font-weight="bold">Incident Action</text>
      <text x="80" y="178" text-anchor="middle" fill="#6c757d">Acknowledge / Escalate</text>

      <rect y="210" width="160" height="50" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="80" y="232" text-anchor="middle" fill="#111111" font-weight="bold">Feedback Capture</text>
      <text x="80" y="248" text-anchor="middle" fill="#6c757d">Flag False Positive</text>

      <rect y="280" width="160" height="60" fill="#ffffff" stroke="#343a40" stroke-width="1"/>
      <text x="80" y="302" text-anchor="middle" fill="#111111" font-weight="bold">Audit Logging</text>
      <text x="80" y="320" text-anchor="middle" fill="#6c757d">Timestamp, User ID,</text>
      <text x="80" y="334" text-anchor="middle" fill="#6c757d">Operator Notes</text>
    </g>
  </g>

  <!-- Bottom Principle -->
  <rect x="50" y="580" width="1100" height="40" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
  <text x="600" y="605" text-anchor="middle" fill="#333333" font-family="system-ui, sans-serif" font-size="12">
    Engineering Principle: The automated pipeline directs human operator attention; critical actions require operator review and verification.
  </text>
</svg>'''
    render(svg, "system-workflow.png")

# ========================================================
# 4. SYSTEM ARCHITECTURE (1200x760) - Monochrome Layered
# ========================================================
def gen_system_architecture():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760">
  <rect width="1200" height="760" fill="#ffffff"/>
  <rect x="1" y="1" width="1198" height="758" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="600" y="40" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="20" font-weight="bold">
    Lenz Layered System Architecture [Proposed Architecture]
  </text>
  <text x="600" y="62" text-anchor="middle" fill="#6c757d" font-family="system-ui, sans-serif" font-size="12">
    Decoupled service tiers for multi-stream ingestion, deep learning inference, event routing, and persistence
  </text>

  <!-- Tier 1: Client -->
  <g transform="translate(60, 90)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="1080" height="95" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <rect width="1080" height="24" fill="#343a40"/>
    <text x="20" y="16" fill="#ffffff" font-weight="bold">PRESENTATION &amp; OPERATOR TIER (REACT / VITE / TAILWIND CSS) [PROPOSED]</text>
    
    <g transform="translate(20, 38)">
      <rect width="245" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="122" y="25" text-anchor="middle" fill="#111111" font-weight="bold">Multi-Stream Video Grid</text>

      <rect x="265" width="245" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="387" y="25" text-anchor="middle" fill="#111111" font-weight="bold">Interactive Zone Drawing Tool</text>

      <rect x="530" width="245" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="652" y="25" text-anchor="middle" fill="#111111" font-weight="bold">Real-Time Incident Triage Feed</text>

      <rect x="795" width="265" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="927" y="25" text-anchor="middle" fill="#111111" font-weight="bold">Operational Analytics &amp; Reports</text>
    </g>
  </g>

  <!-- Connectors 1 to 2 -->
  <g stroke="#212529" stroke-width="1.5" stroke-dasharray="4,4">
    <line x1="300" y1="185" x2="300" y2="220"/>
    <line x1="880" y1="185" x2="880" y2="220"/>
  </g>

  <!-- Tier 2: Application / Gateway -->
  <g transform="translate(60, 220)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="1080" height="95" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <rect width="1080" height="24" fill="#343a40"/>
    <text x="20" y="16" fill="#ffffff" font-weight="bold">APPLICATION &amp; GATEWAY TIER (FASTAPI / ASYNC PYTHON) [PROPOSED]</text>
    
    <g transform="translate(20, 38)">
      <rect width="245" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="122" y="25" text-anchor="middle" fill="#111111" font-weight="bold">REST API &amp; OpenAPI Engine</text>

      <rect x="265" width="245" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="387" y="25" text-anchor="middle" fill="#111111" font-weight="bold">Auth, RBAC &amp; API Security</text>

      <rect x="530" width="245" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="652" y="25" text-anchor="middle" fill="#111111" font-weight="bold">WebSocket Push Gateway</text>

      <rect x="795" width="265" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="927" y="25" text-anchor="middle" fill="#111111" font-weight="bold">Job Orchestrator &amp; Upload Handler</text>
    </g>
  </g>

  <!-- Connectors 2 to 3 -->
  <g stroke="#212529" stroke-width="1.5" stroke-dasharray="4,4">
    <line x1="400" y1="315" x2="400" y2="350"/>
    <line x1="750" y1="315" x2="750" y2="350"/>
  </g>

  <!-- Tier 3: AI Inference -->
  <g transform="translate(60, 350)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="1080" height="105" fill="#ffffff" stroke="#111111" stroke-width="2"/>
    <rect width="1080" height="24" fill="#111111"/>
    <text x="20" y="16" fill="#ffffff" font-weight="bold">AI INFERENCE &amp; REASONING TIER (ONNX RUNTIME / TENSORRT) [PROPOSED]</text>
    
    <g transform="translate(20, 36)">
      <rect width="245" height="55" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="122" y="24" text-anchor="middle" fill="#111111" font-weight="bold">Frame Decimation &amp; Buffer</text>
      <text x="122" y="42" text-anchor="middle" fill="#6c757d">5 - 15 FPS Ingestion Pool</text>

      <rect x="265" width="245" height="55" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="387" y="24" text-anchor="middle" fill="#111111" font-weight="bold">Object Detection (YOLO)</text>
      <text x="387" y="42" text-anchor="middle" fill="#6c757d">Person, Vehicle, Bag Classes</text>

      <rect x="530" width="245" height="55" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="652" y="24" text-anchor="middle" fill="#111111" font-weight="bold">ByteTrack Multi-Object Tracker</text>
      <text x="652" y="42" text-anchor="middle" fill="#6c757d">Kalman Filter &amp; ID Persistence</text>

      <rect x="795" width="265" height="55" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="927" y="24" text-anchor="middle" fill="#111111" font-weight="bold">Spatial-Temporal Event Engine</text>
      <text x="927" y="42" text-anchor="middle" fill="#6c757d">Ray-Casting, Dwell &amp; Deduplication</text>
    </g>
  </g>

  <!-- Connectors 3 to 4 -->
  <g stroke="#212529" stroke-width="1.5" stroke-dasharray="4,4">
    <line x1="450" y1="455" x2="450" y2="490"/>
    <line x1="720" y1="455" x2="720" y2="490"/>
  </g>

  <!-- Tier 4: Storage & Messaging -->
  <g transform="translate(60, 490)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="1080" height="95" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <rect width="1080" height="24" fill="#343a40"/>
    <text x="20" y="16" fill="#ffffff" font-weight="bold">PERSISTENCE, MESSAGE BROKER &amp; EVIDENCE TIER [PROPOSED]</text>
    
    <g transform="translate(20, 38)">
      <rect width="330" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="165" y="25" text-anchor="middle" fill="#111111" font-weight="bold">PostgreSQL 16 (Metadata, Events, Zones, Users)</text>

      <rect x="360" width="330" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="525" y="25" text-anchor="middle" fill="#111111" font-weight="bold">Redis 7 (State Cache, Frame Queues, Pub/Sub)</text>

      <rect x="720" width="340" height="42" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
      <text x="890" y="25" text-anchor="middle" fill="#111111" font-weight="bold">S3 / MinIO (Keyframe Snapshots &amp; Clips)</text>
    </g>
  </g>

  <!-- Tier 5: External Feeds & Integrations -->
  <g transform="translate(60, 620)" font-family="system-ui, sans-serif" font-size="11">
    <!-- Left: Sources -->
    <rect width="520" height="95" fill="#ffffff" stroke="#495057" stroke-width="1.5"/>
    <rect width="520" height="22" fill="#495057"/>
    <text x="15" y="15" fill="#ffffff" font-weight="bold">INPUT SOURCES &amp; PROTOCOLS [PROPOSED]</text>
    <text x="15" y="45" fill="#212529">• IP Surveillance Cameras (RTSP / ONVIF H.264 &amp; H.265 Streams)</text>
    <text x="15" y="65" fill="#212529">• Pre-recorded Video Footage Uploads (MP4, MKV, AVI)</text>
    <text x="15" y="85" fill="#6c757d">In-memory ring buffer with packet loss recovery</text>

    <!-- Right: Integrations -->
    <g transform="translate(560, 0)">
      <rect width="520" height="95" fill="#ffffff" stroke="#495057" stroke-width="1.5"/>
      <rect width="520" height="22" fill="#495057"/>
      <text x="15" y="15" fill="#ffffff" font-weight="bold">ALERT INTEGRATIONS &amp; DISPATCH [PROPOSED]</text>
      <text x="15" y="45" fill="#212529">• Real-Time WebSockets to Operator Workstations (&lt;2s latency)</text>
      <text x="15" y="65" fill="#212529">• Automated Webhook Payloads to Enterprise Incident Systems</text>
      <text x="15" y="85" fill="#6c757d">Dead-letter queue with exponential retry on delivery failure</text>
    </g>
  </g>

  <text x="600" y="745" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    All architecture components are proposed specifications for the Lenz platform. Repository is currently at initial setup stage.
  </text>
</svg>'''
    render(svg, "system-architecture.png")

# ========================================================
# 5. DATABASE ER DIAGRAM (1300x850) - Monochrome Design
# ========================================================
def gen_database_er():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="850" viewBox="0 0 1300 850">
  <rect width="1300" height="850" fill="#ffffff"/>
  <rect x="1" y="1" width="1298" height="848" fill="none" stroke="#ced4da" stroke-width="1"/>

  <!-- Prominent Title -->
  <text x="650" y="38" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="20" font-weight="bold">
    Proposed Database Design — Not Yet Implemented
  </text>
  <text x="650" y="60" text-anchor="middle" fill="#6c757d" font-family="system-ui, sans-serif" font-size="12">
    Relational Entity-Relationship Specification for PostgreSQL Event Store and Camera Fleet Registry
  </text>

  <!-- Table 1: USERS -->
  <g transform="translate(50, 90)" font-family="monospace" font-size="11">
    <rect width="250" height="180" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
    <rect width="250" height="26" fill="#f1f3f5" stroke="#343a40" stroke-width="1"/>
    <text x="125" y="17" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">USERS</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="0"><tspan font-weight="bold">PK</tspan> id: uuid</text>
      <text x="0" y="18">   name: varchar(100)</text>
      <text x="0" y="36"><tspan font-weight="bold">UK</tspan> email: varchar(255)</text>
      <text x="0" y="54">   password_hash: varchar</text>
      <text x="0" y="72">   role: enum(admin, op)</text>
      <text x="0" y="90">   created_at: timestamp</text>
    </g>
  </g>

  <!-- Table 2: AUDIT_LOGS -->
  <g transform="translate(50, 310)" font-family="monospace" font-size="11">
    <rect width="250" height="200" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
    <rect width="250" height="26" fill="#f1f3f5" stroke="#343a40" stroke-width="1"/>
    <text x="125" y="17" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">AUDIT_LOGS</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="0"><tspan font-weight="bold">PK</tspan> id: uuid</text>
      <text x="0" y="18"><tspan font-weight="bold">FK</tspan> user_id: uuid</text>
      <text x="0" y="36">   action: varchar(100)</text>
      <text x="0" y="54">   resource_type: varchar</text>
      <text x="0" y="72">   resource_id: varchar</text>
      <text x="0" y="90">   ip_address: varchar(45)</text>
      <text x="0" y="108">   timestamp: timestamp</text>
    </g>
  </g>

  <!-- Table 3: LOCATIONS -->
  <g transform="translate(360, 90)" font-family="monospace" font-size="11">
    <rect width="260" height="170" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
    <rect width="260" height="26" fill="#f1f3f5" stroke="#343a40" stroke-width="1"/>
    <text x="130" y="17" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">LOCATIONS</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="0"><tspan font-weight="bold">PK</tspan> id: uuid</text>
      <text x="0" y="18">   name: varchar(150)</text>
      <text x="0" y="36">   address: text</text>
      <text x="0" y="54">   metadata: jsonb</text>
      <text x="0" y="72">   created_at: timestamp</text>
    </g>
  </g>

  <!-- Table 4: CAMERAS -->
  <g transform="translate(360, 300)" font-family="monospace" font-size="11">
    <rect width="260" height="260" fill="#ffffff" stroke="#111111" stroke-width="2"/>
    <rect width="260" height="26" fill="#111111"/>
    <text x="130" y="17" text-anchor="middle" fill="#ffffff" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">CAMERAS</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="0"><tspan font-weight="bold">PK</tspan> id: uuid</text>
      <text x="0" y="18"><tspan font-weight="bold">FK</tspan> location_id: uuid</text>
      <text x="0" y="36">   name: varchar(100)</text>
      <text x="0" y="54">   stream_url_enc: text</text>
      <text x="0" y="72">   protocol: varchar(10)</text>
      <text x="0" y="90">   status: varchar(20)</text>
      <text x="0" y="108">   resolution: varchar(20)</text>
      <text x="0" y="126">   fps: integer</text>
      <text x="0" y="144">   last_seen_at: timestamp</text>
      <text x="0" y="162">   created_at: timestamp</text>
    </g>
  </g>

  <!-- Table 5: MONITORING_ZONES -->
  <g transform="translate(360, 600)" font-family="monospace" font-size="11">
    <rect width="260" height="210" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
    <rect width="260" height="26" fill="#f1f3f5" stroke="#343a40" stroke-width="1"/>
    <text x="130" y="17" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">MONITORING_ZONES</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="0"><tspan font-weight="bold">PK</tspan> id: uuid</text>
      <text x="0" y="18"><tspan font-weight="bold">FK</tspan> camera_id: uuid</text>
      <text x="0" y="36">   name: varchar(100)</text>
      <text x="0" y="54">   zone_type: varchar(50)</text>
      <text x="0" y="72">   polygon_coords: jsonb</text>
      <text x="0" y="90">   zone_config: jsonb</text>
      <text x="0" y="108">   created_at: timestamp</text>
    </g>
  </g>

  <!-- Table 6: DETECTION_EVENTS -->
  <g transform="translate(680, 280)" font-family="monospace" font-size="11">
    <rect width="270" height="290" fill="#ffffff" stroke="#111111" stroke-width="2"/>
    <rect width="270" height="26" fill="#111111"/>
    <text x="135" y="17" text-anchor="middle" fill="#ffffff" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">DETECTION_EVENTS</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="0"><tspan font-weight="bold">PK</tspan> id: uuid</text>
      <text x="0" y="18"><tspan font-weight="bold">FK</tspan> camera_id: uuid</text>
      <text x="0" y="36"><tspan font-weight="bold">FK</tspan> zone_id: uuid</text>
      <text x="0" y="54">   event_type: varchar(50)</text>
      <text x="0" y="72">   confidence: float</text>
      <text x="0" y="90">   severity: varchar(20)</text>
      <text x="0" y="108">   review_status: varchar</text>
      <text x="0" y="126">   timestamp: timestamp</text>
      <text x="0" y="144">   evidence_snapshot: text</text>
      <text x="0" y="162">   evidence_clip: text</text>
      <text x="0" y="180">   metadata: jsonb</text>
    </g>
  </g>

  <!-- Table 7: ALERTS -->
  <g transform="translate(1000, 280)" font-family="monospace" font-size="11">
    <rect width="250" height="250" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
    <rect width="250" height="26" fill="#f1f3f5" stroke="#343a40" stroke-width="1"/>
    <text x="125" y="17" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">ALERTS</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="0"><tspan font-weight="bold">PK</tspan> id: uuid</text>
      <text x="0" y="18"><tspan font-weight="bold">FK</tspan> event_id: uuid</text>
      <text x="0" y="36"><tspan font-weight="bold">FK</tspan> recipient_id: uuid</text>
      <text x="0" y="54">   channel: varchar(30)</text>
      <text x="0" y="72">   status: varchar(20)</text>
      <text x="0" y="90">   created_at: timestamp</text>
      <text x="0" y="108">   acknowledged_at: timestamp</text>
      <text x="0" y="126"><tspan font-weight="bold">FK</tspan> acknowledged_by: uuid</text>
    </g>
  </g>

  <!-- Table 8: DETECTION_MODELS -->
  <g transform="translate(680, 90)" font-family="monospace" font-size="11">
    <rect width="270" height="160" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
    <rect width="270" height="26" fill="#f1f3f5" stroke="#343a40" stroke-width="1"/>
    <text x="135" y="17" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">DETECTION_MODELS</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="0"><tspan font-weight="bold">PK</tspan> id: uuid</text>
      <text x="0" y="18">   name: varchar(100)</text>
      <text x="0" y="36">   version: varchar(50)</text>
      <text x="0" y="54">   model_type: varchar(50)</text>
      <text x="0" y="72">   configuration: jsonb</text>
      <text x="0" y="90">   active_status: boolean</text>
    </g>
  </g>

  <!-- Table 9: PROCESSING_JOBS -->
  <g transform="translate(680, 610)" font-family="monospace" font-size="11">
    <rect width="270" height="200" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
    <rect width="270" height="26" fill="#f1f3f5" stroke="#343a40" stroke-width="1"/>
    <text x="135" y="17" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">PROCESSING_JOBS</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="0"><tspan font-weight="bold">PK</tspan> id: uuid</text>
      <text x="0" y="18"><tspan font-weight="bold">FK</tspan> camera_id: uuid</text>
      <text x="0" y="36">   video_source: varchar(30)</text>
      <text x="0" y="54">   status: varchar(20)</text>
      <text x="0" y="72">   frames_processed: int</text>
      <text x="0" y="90">   fps_actual: float</text>
      <text x="0" y="108">   started_at: timestamp</text>
      <text x="0" y="126">   error_message: text</text>
    </g>
  </g>

  <!-- Relationship Lines (Monochrome Solid) -->
  <g stroke="#343a40" stroke-width="1.5" fill="none">
    <path d="M 175 270 L 175 310"/>
    <path d="M 490 260 L 490 300"/>
    <path d="M 490 560 L 490 600"/>
    <path d="M 620 425 L 680 425"/>
    <path d="M 950 425 L 1000 425"/>
    <path d="M 590 560 L 590 660 L 680 660"/>
  </g>

  <text x="650" y="835" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Schema status: Proposed architectural design. All entities, attributes, and relationships are subject to implementation verification.
  </text>
</svg>'''
    render(svg, "database-er.png")

# ========================================================
# 6. AI DETECTION PIPELINE (1200x560) - Monochrome Flow
# ========================================================
def gen_ai_pipeline():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560">
  <rect width="1200" height="560" fill="#ffffff"/>
  <rect x="1" y="1" width="1198" height="558" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="600" y="40" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="20" font-weight="bold">
    AI Computer Vision &amp; Tracking Pipeline [Proposed Architecture]
  </text>
  <text x="600" y="62" text-anchor="middle" fill="#6c757d" font-family="system-ui, sans-serif" font-size="12">
    Stage-by-stage transformation of raw video frames into validated, actionable incident events
  </text>

  <!-- 6 Stages -->
  <g transform="translate(30, 95)" font-family="system-ui, sans-serif" font-size="11">
    <!-- Stage 1 -->
    <rect width="175" height="380" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
    <rect width="175" height="28" fill="#495057"/>
    <text x="87" y="18" text-anchor="middle" fill="#ffffff" font-weight="bold">1. INGESTION [PROP]</text>
    <g transform="translate(12, 45)" fill="#212529">
      <text x="0" y="15" font-weight="bold">• Frame Demuxing</text>
      <text x="0" y="32" fill="#495057">OpenCV VideoCapture /</text>
      <text x="0" y="48" fill="#495057">FFmpeg RTSP client</text>

      <text x="0" y="80" font-weight="bold">• FPS Decimation</text>
      <text x="0" y="96" fill="#495057">Downsample 30 FPS</text>
      <text x="0" y="112" fill="#495057">to 5 - 10 FPS target</text>

      <text x="0" y="144" font-weight="bold">• Shared Ring Buffer</text>
      <text x="0" y="160" fill="#495057">Decoupled queue to</text>
      <text x="0" y="176" fill="#495057">prevent stream stalls</text>

      <text x="0" y="208" font-weight="bold">• Stream Health</text>
      <text x="0" y="224" fill="#495057">Heartbeat telemetry</text>
    </g>

    <!-- Stage 2 -->
    <g transform="translate(195, 0)">
      <rect width="175" height="380" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
      <rect width="175" height="28" fill="#495057"/>
      <text x="87" y="18" text-anchor="middle" fill="#ffffff" font-weight="bold">2. PREPROCESS [PROP]</text>
      <g transform="translate(12, 45)" fill="#212529">
        <text x="0" y="15" font-weight="bold">• Color Conversion</text>
        <text x="0" y="32" fill="#495057">BGR to RGB color</text>
        <text x="0" y="48" fill="#495057">space transformation</text>

        <text x="0" y="80" font-weight="bold">• Letterbox Padding</text>
        <text x="0" y="96" fill="#495057">Aspect-preserving</text>
        <text x="0" y="112" fill="#495057">resize to 640x640</text>

        <text x="0" y="144" font-weight="bold">• Normalization</text>
        <text x="0" y="160" fill="#495057">Scale uint8 [0, 255]</text>
        <text x="0" y="176" fill="#495057">to float32 [0.0, 1.0]</text>

        <text x="0" y="208" font-weight="bold">• Tensor Transfer</text>
        <text x="0" y="224" fill="#495057">Device buffer load</text>
      </g>
    </g>

    <!-- Stage 3 -->
    <g transform="translate(390, 0)">
      <rect width="175" height="380" fill="#ffffff" stroke="#111111" stroke-width="2"/>
      <rect width="175" height="28" fill="#111111"/>
      <text x="87" y="18" text-anchor="middle" fill="#ffffff" font-weight="bold">3. DETECTION [PROP]</text>
      <g transform="translate(12, 45)" fill="#212529">
        <text x="0" y="15" font-weight="bold">• YOLO Inference</text>
        <text x="0" y="32" fill="#495057">ONNX / TensorRT</text>
        <text x="0" y="48" fill="#495057">lightweight backbone</text>

        <text x="0" y="80" font-weight="bold">• Target Classes</text>
        <text x="0" y="96" fill="#495057">Person, Car, Truck,</text>
        <text x="0" y="112" fill="#495057">Bicycle, Backpack</text>

        <text x="0" y="144" font-weight="bold">• Raw Predictions</text>
        <text x="0" y="160" fill="#495057">BBoxes (x, y, w, h)</text>
        <text x="0" y="176" fill="#495057">+ class confidences</text>

        <text x="0" y="208" font-weight="bold">• NMS Filtering</text>
        <text x="0" y="224" fill="#495057">IoU threshold = 0.45</text>
      </g>
    </g>

    <!-- Stage 4 -->
    <g transform="translate(585, 0)">
      <rect width="175" height="380" fill="#ffffff" stroke="#111111" stroke-width="2"/>
      <rect width="175" height="28" fill="#111111"/>
      <text x="87" y="18" text-anchor="middle" fill="#ffffff" font-weight="bold">4. TRACKING [PROP]</text>
      <g transform="translate(12, 45)" fill="#212529">
        <text x="0" y="15" font-weight="bold">• ByteTrack Method</text>
        <text x="0" y="32" fill="#495057">Associates high &amp; low</text>
        <text x="0" y="48" fill="#495057">confidence detections</text>

        <text x="0" y="80" font-weight="bold">• Kalman Filtering</text>
        <text x="0" y="96" fill="#495057">State estimation &amp;</text>
        <text x="0" y="112" fill="#495057">velocity prediction</text>

        <text x="0" y="144" font-weight="bold">• Persistent Track IDs</text>
        <text x="0" y="160" fill="#495057">Maintains identities</text>
        <text x="0" y="176" fill="#495057">through occlusions</text>

        <text x="0" y="208" font-weight="bold">• Trajectory Log</text>
        <text x="0" y="224" fill="#495057">Centroid history path</text>
      </g>
    </g>

    <!-- Stage 5 -->
    <g transform="translate(780, 0)">
      <rect width="175" height="380" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
      <rect width="175" height="28" fill="#495057"/>
      <text x="87" y="18" text-anchor="middle" fill="#ffffff" font-weight="bold">5. RULE LOGIC [PROP]</text>
      <g transform="translate(12, 45)" fill="#212529">
        <text x="0" y="15" font-weight="bold">• Ray-Casting Test</text>
        <text x="0" y="32" fill="#495057">Point-in-Polygon check</text>
        <text x="0" y="48" fill="#495057">for restricted boundaries</text>

        <text x="0" y="80" font-weight="bold">• Dwell Duration</text>
        <text x="0" y="96" fill="#495057">Tracklet time in zone</text>
        <text x="0" y="112" fill="#495057">&gt; T seconds threshold</text>

        <text x="0" y="144" font-weight="bold">• Density Estimation</text>
        <text x="0" y="160" fill="#495057">Headcount per polygon</text>
        <text x="0" y="176" fill="#495057">&gt; safety threshold</text>

        <text x="0" y="208" font-weight="bold">• Parking Slot IoU</text>
        <text x="0" y="224" fill="#495057">Overlap ratio &gt; 0.35</text>
      </g>
    </g>

    <!-- Stage 6 -->
    <g transform="translate(975, 0)">
      <rect width="165" height="380" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
      <rect width="165" height="28" fill="#495057"/>
      <text x="82" y="18" text-anchor="middle" fill="#ffffff" font-weight="bold">6. DISPATCH [PROP]</text>
      <g transform="translate(12, 45)" fill="#212529">
        <text x="0" y="15" font-weight="bold">• Debounce Window</text>
        <text x="0" y="32" fill="#495057">Suppresses duplicate</text>
        <text x="0" y="48" fill="#495057">alerts within window</text>

        <text x="0" y="80" font-weight="bold">• Keyframe Capture</text>
        <text x="0" y="96" fill="#495057">Crops bounding box &amp;</text>
        <text x="0" y="112" fill="#495057">saves JPEG to S3</text>

        <text x="0" y="144" font-weight="bold">• Structured Log</text>
        <text x="0" y="160" fill="#495057">Writes JSON event to</text>
        <text x="0" y="176" fill="#495057">PostgreSQL database</text>

        <text x="0" y="208" font-weight="bold">• WebSocket Push</text>
        <text x="0" y="224" fill="#495057">Broadcasts to UI</text>
      </g>
    </g>
  </g>

  <!-- Connectors -->
  <g stroke="#212529" stroke-width="1.5" fill="none">
    <path d="M 205 285 L 225 285"/>
    <path d="M 400 285 L 420 285"/>
    <path d="M 595 285 L 615 285"/>
    <path d="M 790 285 L 810 285"/>
    <path d="M 985 285 L 1005 285"/>
  </g>

  <text x="600" y="525" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Pipeline status: Proposed architectural design. Performance estimates are derived from official YOLOv8 and ByteTrack documentation.
  </text>
</svg>'''
    render(svg, "ai-pipeline.png")

# ========================================================
# 7. USE CASE DIAGRAM (1200x660) - Monochrome UML Style
# ========================================================
def gen_use_case():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="660" viewBox="0 0 1200 660">
  <rect width="1200" height="660" fill="#ffffff"/>
  <rect x="1" y="1" width="1198" height="658" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="600" y="40" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="20" font-weight="bold">
    System Use Case Diagram [Proposed Architecture]
  </text>
  <text x="600" y="62" text-anchor="middle" fill="#6c757d" font-family="system-ui, sans-serif" font-size="12">
    Primary actor interactions with system services across administration, live monitoring, and automated AI processing
  </text>

  <!-- Left: Actors -->
  <g transform="translate(50, 100)" font-family="system-ui, sans-serif">
    <!-- Admin -->
    <rect width="170" height="95" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <text x="85" y="35" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">System Administrator</text>
    <text x="85" y="55" text-anchor="middle" fill="#6c757d" font-size="11">Camera fleet setup,</text>
    <text x="85" y="70" text-anchor="middle" fill="#6c757d" font-size="11">zone configuration</text>

    <!-- Operator -->
    <g transform="translate(0, 140)">
      <rect width="170" height="95" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
      <text x="85" y="35" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">Security Operator</text>
      <text x="85" y="55" text-anchor="middle" fill="#6c757d" font-size="11">Live stream monitor,</text>
      <text x="85" y="70" text-anchor="middle" fill="#6c757d" font-size="11">incident triage</text>
    </g>

    <!-- Manager -->
    <g transform="translate(0, 280)">
      <rect width="170" height="95" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
      <text x="85" y="35" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">Facility Manager</text>
      <text x="85" y="55" text-anchor="middle" fill="#6c757d" font-size="11">Operational analytics,</text>
      <text x="85" y="70" text-anchor="middle" fill="#6c757d" font-size="11">queue/crowd reports</text>
    </g>
  </g>

  <!-- System Boundary -->
  <g transform="translate(270, 90)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="660" height="520" fill="#ffffff" stroke="#212529" stroke-width="1.5"/>
    <text x="330" y="25" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">
      LENZ VIDEO INTELLIGENCE PLATFORM [PROPOSED BOUNDARY]
    </text>

    <!-- Use Cases (Col 1: Operator / Admin actions) -->
    <g transform="translate(30, 45)">
      <rect width="270" height="40" rx="4" fill="#f8f9fa" stroke="#495057" stroke-width="1"/>
      <text x="135" y="24" text-anchor="middle" fill="#111111">UC-01: Register &amp; Manage IP Cameras</text>

      <rect y="55" width="270" height="40" rx="4" fill="#f8f9fa" stroke="#495057" stroke-width="1"/>
      <text x="135" y="79" text-anchor="middle" fill="#111111">UC-02: Draw Monitoring Polygon Zones</text>

      <rect y="110" width="270" height="40" rx="4" fill="#f8f9fa" stroke="#495057" stroke-width="1"/>
      <text x="135" y="134" text-anchor="middle" fill="#111111">UC-03: Upload Pre-recorded Video File</text>

      <rect y="165" width="270" height="40" rx="4" fill="#f8f9fa" stroke="#495057" stroke-width="1"/>
      <text x="135" y="189" text-anchor="middle" fill="#111111">UC-04: Monitor Multi-Camera Live Grid</text>

      <rect y="220" width="270" height="40" rx="4" fill="#f8f9fa" stroke="#495057" stroke-width="1"/>
      <text x="135" y="244" text-anchor="middle" fill="#111111">UC-05: Receive Real-Time Incident Alerts</text>

      <rect y="275" width="270" height="40" rx="4" fill="#f8f9fa" stroke="#495057" stroke-width="1"/>
      <text x="135" y="299" text-anchor="middle" fill="#111111">UC-06: Acknowledge / Escalate Alert</text>

      <rect y="330" width="270" height="40" rx="4" fill="#f8f9fa" stroke="#495057" stroke-width="1"/>
      <text x="135" y="354" text-anchor="middle" fill="#111111">UC-07: Inspect Keyframe Snapshot</text>

      <rect y="385" width="270" height="40" rx="4" fill="#f8f9fa" stroke="#495057" stroke-width="1"/>
      <text x="135" y="409" text-anchor="middle" fill="#111111">UC-08: View Operational Telemetry</text>
    </g>

    <!-- Use Cases (Col 2: Automated AI Tasks) -->
    <g transform="translate(360, 45)">
      <rect width="270" height="40" rx="4" fill="#ffffff" stroke="#111111" stroke-width="1.5"/>
      <text x="135" y="24" text-anchor="middle" fill="#111111" font-weight="bold">UC-09: Decode &amp; Decimate Stream</text>

      <rect y="55" width="270" height="40" rx="4" fill="#ffffff" stroke="#111111" stroke-width="1.5"/>
      <text x="135" y="79" text-anchor="middle" fill="#111111" font-weight="bold">UC-10: Detect Persons &amp; Vehicles</text>

      <rect y="110" width="270" height="40" rx="4" fill="#ffffff" stroke="#111111" stroke-width="1.5"/>
      <text x="135" y="134" text-anchor="middle" fill="#111111" font-weight="bold">UC-11: Assign Persistent Track IDs</text>

      <rect y="165" width="270" height="40" rx="4" fill="#ffffff" stroke="#111111" stroke-width="1.5"/>
      <text x="135" y="189" text-anchor="middle" fill="#111111" font-weight="bold">UC-12: Evaluate Zone Polygon Rules</text>

      <rect y="220" width="270" height="40" rx="4" fill="#ffffff" stroke="#111111" stroke-width="1.5"/>
      <text x="135" y="244" text-anchor="middle" fill="#111111" font-weight="bold">UC-13: Deduplicate Candidate Events</text>

      <rect y="275" width="270" height="40" rx="4" fill="#ffffff" stroke="#111111" stroke-width="1.5"/>
      <text x="135" y="299" text-anchor="middle" fill="#111111" font-weight="bold">UC-14: Store Event &amp; S3 Keyframe</text>

      <rect y="330" width="270" height="40" rx="4" fill="#ffffff" stroke="#111111" stroke-width="1.5"/>
      <text x="135" y="354" text-anchor="middle" fill="#111111" font-weight="bold">UC-15: Broadcast via WebSocket</text>

      <rect y="385" width="270" height="40" rx="4" fill="#ffffff" stroke="#111111" stroke-width="1.5"/>
      <text x="135" y="409" text-anchor="middle" fill="#111111" font-weight="bold">UC-16: Log Security Audit Trail</text>
    </g>
  </g>

  <!-- Right: External Systems -->
  <g transform="translate(980, 100)" font-family="system-ui, sans-serif">
    <rect width="170" height="95" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <text x="85" y="35" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">Camera Stream Source</text>
    <text x="85" y="55" text-anchor="middle" fill="#6c757d" font-size="11">RTSP / ONVIF feeds,</text>
    <text x="85" y="70" text-anchor="middle" fill="#6c757d" font-size="11">network video packets</text>

    <g transform="translate(0, 140)">
      <rect width="170" height="95" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
      <text x="85" y="35" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">External Notifications</text>
      <text x="85" y="55" text-anchor="middle" fill="#6c757d" font-size="11">Slack, Email, SMS,</text>
      <text x="85" y="70" text-anchor="middle" fill="#6c757d" font-size="11">Enterprise Webhooks</text>
    </g>

    <g transform="translate(0, 280)">
      <rect width="170" height="95" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
      <text x="85" y="35" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">Persistence Stores</text>
      <text x="85" y="55" text-anchor="middle" fill="#6c757d" font-size="11">PostgreSQL Database,</text>
      <text x="85" y="70" text-anchor="middle" fill="#6c757d" font-size="11">MinIO / S3 Object Store</text>
    </g>
  </g>

  <!-- Association Lines -->
  <g stroke="#495057" stroke-width="1.5" fill="none">
    <line x1="220" y1="140" x2="300" y2="155"/>
    <line x1="220" y1="280" x2="300" y2="330"/>
    <line x1="220" y1="420" x2="300" y2="500"/>
    <line x1="900" y1="155" x2="980" y2="140"/>
    <line x1="900" y1="330" x2="980" y2="280"/>
    <line x1="900" y1="450" x2="980" y2="420"/>
  </g>

  <text x="600" y="635" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Use case diagram represents proposed functional requirements. Solid association lines indicate primary interactions.
  </text>
</svg>'''
    render(svg, "use-case.png")

# ========================================================
# 8. DATA FLOW DIAGRAM (1200x640) - Monochrome DFD
# ========================================================
def gen_data_flow():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="640" viewBox="0 0 1200 640">
  <rect width="1200" height="640" fill="#ffffff"/>
  <rect x="1" y="1" width="1198" height="638" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="600" y="40" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="20" font-weight="bold">
    Level-1 Data Flow Diagram (DFD) [Proposed Architecture]
  </text>
  <text x="600" y="62" text-anchor="middle" fill="#6c757d" font-family="system-ui, sans-serif" font-size="12">
    Movement of video frames, inference vectors, rule definitions, and incident alerts across system boundaries
  </text>

  <!-- External Entity: Camera -->
  <g transform="translate(50, 110)" font-family="system-ui, sans-serif">
    <rect width="160" height="70" fill="#ffffff" stroke="#111111" stroke-width="2"/>
    <line x1="12" y1="0" x2="12" y2="70" stroke="#111111" stroke-width="1.5"/>
    <text x="86" y="32" text-anchor="middle" fill="#111111" font-size="12" font-weight="bold">IP Camera Feed</text>
    <text x="86" y="50" text-anchor="middle" fill="#6c757d" font-size="11">RTSP Stream Source</text>
  </g>

  <!-- Process 1.0 -->
  <g transform="translate(280, 110)" font-family="system-ui, sans-serif">
    <rect width="120" height="70" rx="10" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <text x="60" y="30" text-anchor="middle" fill="#111111" font-size="11" font-weight="bold">1.0 Ingest</text>
    <text x="60" y="48" text-anchor="middle" fill="#495057" font-size="11">&amp; Decimate</text>
  </g>

  <!-- Data Store D1 -->
  <g transform="translate(460, 115)" font-family="monospace" font-size="11">
    <line x1="0" y1="0" x2="140" y2="0" stroke="#343a40" stroke-width="1.5"/>
    <line x1="0" y1="55" x2="140" y2="55" stroke="#343a40" stroke-width="1.5"/>
    <rect width="140" height="55" fill="#f8f9fa"/>
    <text x="70" y="26" text-anchor="middle" fill="#111111" font-weight="bold">D1: Frame Buffer</text>
    <text x="70" y="44" text-anchor="middle" fill="#6c757d">Decimated Tensors</text>
  </g>

  <!-- Process 2.0 -->
  <g transform="translate(660, 110)" font-family="system-ui, sans-serif">
    <rect width="120" height="70" rx="10" fill="#ffffff" stroke="#111111" stroke-width="2"/>
    <text x="60" y="30" text-anchor="middle" fill="#111111" font-size="11" font-weight="bold">2.0 Detect</text>
    <text x="60" y="48" text-anchor="middle" fill="#495057" font-size="11">&amp; Track (YOLO)</text>
  </g>

  <!-- Process 3.0 -->
  <g transform="translate(850, 110)" font-family="system-ui, sans-serif">
    <rect width="120" height="70" rx="10" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <text x="60" y="30" text-anchor="middle" fill="#111111" font-size="11" font-weight="bold">3.0 Evaluate</text>
    <text x="60" y="48" text-anchor="middle" fill="#495057" font-size="11">Zone Rules</text>
  </g>

  <!-- Data Store D2: Zone Configurations -->
  <g transform="translate(810, 250)" font-family="monospace" font-size="11">
    <line x1="0" y1="0" x2="200" y2="0" stroke="#343a40" stroke-width="1.5"/>
    <line x1="0" y1="55" x2="200" y2="55" stroke="#343a40" stroke-width="1.5"/>
    <rect width="200" height="55" fill="#f8f9fa"/>
    <text x="100" y="26" text-anchor="middle" fill="#111111" font-weight="bold">D2: Zone Rules &amp; Polygons</text>
    <text x="100" y="44" text-anchor="middle" fill="#6c757d">Thresholds, Coordinates</text>
  </g>

  <!-- Data Store D3: Event Store -->
  <g transform="translate(810, 370)" font-family="monospace" font-size="11">
    <line x1="0" y1="0" x2="200" y2="0" stroke="#343a40" stroke-width="1.5"/>
    <line x1="0" y1="55" x2="200" y2="55" stroke="#343a40" stroke-width="1.5"/>
    <rect width="200" height="55" fill="#f8f9fa"/>
    <text x="100" y="26" text-anchor="middle" fill="#111111" font-weight="bold">D3: Event Store (Postgres)</text>
    <text x="100" y="44" text-anchor="middle" fill="#6c757d">BBoxes, Confidence, Time</text>
  </g>

  <!-- Data Store D4: Media Store -->
  <g transform="translate(1030, 370)" font-family="monospace" font-size="11">
    <line x1="0" y1="0" x2="140" y2="0" stroke="#343a40" stroke-width="1.5"/>
    <line x1="0" y1="55" x2="140" y2="55" stroke="#343a40" stroke-width="1.5"/>
    <rect width="140" height="55" fill="#f8f9fa"/>
    <text x="70" y="26" text-anchor="middle" fill="#111111" font-weight="bold">D4: Media (S3)</text>
    <text x="70" y="44" text-anchor="middle" fill="#6c757d">Keyframe Snapshots</text>
  </g>

  <!-- Process 4.0 -->
  <g transform="translate(480, 410)" font-family="system-ui, sans-serif">
    <rect width="130" height="70" rx="10" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <text x="65" y="30" text-anchor="middle" fill="#111111" font-size="11" font-weight="bold">4.0 Dispatch</text>
    <text x="65" y="48" text-anchor="middle" fill="#495057" font-size="11">&amp; WebSocket Push</text>
  </g>

  <!-- External Entity: Operator -->
  <g transform="translate(150, 410)" font-family="system-ui, sans-serif">
    <rect width="180" height="70" fill="#ffffff" stroke="#111111" stroke-width="2"/>
    <line x1="12" y1="0" x2="12" y2="70" stroke="#111111" stroke-width="1.5"/>
    <text x="96" y="32" text-anchor="middle" fill="#111111" font-size="12" font-weight="bold">Operator Dashboard</text>
    <text x="96" y="50" text-anchor="middle" fill="#6c757d" font-size="11">Review &amp; Acknowledge</text>
  </g>

  <!-- Process 5.0 -->
  <g transform="translate(480, 525)" font-family="system-ui, sans-serif">
    <rect width="130" height="60" rx="10" fill="#f8f9fa" stroke="#343a40" stroke-width="1.5"/>
    <text x="65" y="27" text-anchor="middle" fill="#111111" font-size="11" font-weight="bold">5.0 Aggregate</text>
    <text x="65" y="44" text-anchor="middle" fill="#495057" font-size="11">Analytics</text>
  </g>

  <!-- Flow Arrows (Monochrome Lines with Text) -->
  <g stroke="#343a40" stroke-width="1.5" fill="none">
    <path d="M 210 145 L 280 145"/>
    <path d="M 400 145 L 460 145"/>
    <path d="M 600 145 L 660 145"/>
    <path d="M 780 145 L 850 145"/>
    <path d="M 910 250 L 910 180"/>
    <path d="M 910 180 L 910 370"/>
    <path d="M 970 145 L 1100 145 L 1100 370"/>
    <path d="M 810 400 L 610 440"/>
    <path d="M 480 445 L 330 445"/>
    <path d="M 240 410 L 240 275 L 810 275"/>
    <path d="M 810 415 L 610 550"/>
    <path d="M 480 555 L 240 555 L 240 480"/>
  </g>

  <text x="600" y="615" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    DFD Level-1 representation of proposed system. Data stores represent planned persistence layers.
  </text>
</svg>'''
    render(svg, "data-flow.png")

# ========================================================
# 9. DASHBOARD MOCKUP (1280x820) - Technical Wireframe
# ========================================================
def gen_dashboard_mockup():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="820" viewBox="0 0 1280 820">
  <!-- Clean White Background & Outline -->
  <rect width="1280" height="820" fill="#ffffff"/>
  <rect x="1" y="1" width="1278" height="818" fill="none" stroke="#ced4da" stroke-width="1"/>

  <!-- Top Navigation Wireframe -->
  <rect width="1280" height="52" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
  <text x="25" y="32" fill="#111111" font-family="system-ui, sans-serif" font-size="16" font-weight="900" letter-spacing="1">LENZ</text>
  <text x="80" y="32" fill="#6c757d" font-family="system-ui, sans-serif" font-size="12">| Operator Monitoring Portal (Conceptual Wireframe Specification)</text>

  <!-- Nav Links -->
  <g transform="translate(560, 32)" font-family="system-ui, sans-serif" font-size="12" fill="#495057">
    <text x="0" y="0" font-weight="bold" fill="#111111">[Live Feeds]</text>
    <text x="90" y="0">Incident Review</text>
    <text x="200" y="0">Zone Rules</text>
    <text x="290" y="0">Analytics</text>
    <text x="370" y="0">Cameras</text>
  </g>

  <!-- User / Status Box -->
  <g transform="translate(1080, 14)" font-family="system-ui, sans-serif" font-size="11">
    <rect width="175" height="26" fill="#ffffff" stroke="#ced4da" stroke-width="1"/>
    <text x="15" y="17" fill="#212529">Status: Active (12 Feeds)</text>
  </g>

  <!-- Prominent Watermark / Wireframe Notice -->
  <rect x="0" y="52" width="1280" height="24" fill="#e9ecef" stroke="#ced4da" stroke-width="0.5"/>
  <text x="640" y="68" text-anchor="middle" fill="#495057" font-family="system-ui, sans-serif" font-size="11" font-weight="bold" letter-spacing="1">
    PROPOSED DASHBOARD WIREFRAME SPECIFICATION — NOT AN IMPLEMENTED APPLICATION SCREENSHOT
  </text>

  <!-- KPI Row -->
  <g transform="translate(25, 90)" font-family="system-ui, sans-serif">
    <!-- Card 1 -->
    <rect width="285" height="75" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
    <text x="16" y="24" fill="#6c757d" font-size="11" font-weight="bold">ACTIVE CAMERA STREAMS</text>
    <text x="16" y="54" fill="#111111" font-size="22" font-weight="bold">12 / 12 Feeds</text>
    <text x="175" y="52" fill="#495057" font-size="11">100% Operational</text>

    <!-- Card 2 -->
    <g transform="translate(305, 0)">
      <rect width="285" height="75" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="16" y="24" fill="#6c757d" font-size="11" font-weight="bold">RECORDED EVENTS (TODAY)</text>
      <text x="16" y="54" fill="#111111" font-size="22" font-weight="bold">142 Incidents</text>
      <text x="165" y="52" fill="#495057" font-size="11">12 Pending Review</text>
    </g>

    <!-- Card 3 -->
    <g transform="translate(610, 0)">
      <rect width="285" height="75" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="16" y="24" fill="#6c757d" font-size="11" font-weight="bold">AVERAGE DENSITY RATIO</text>
      <text x="16" y="54" fill="#111111" font-size="22" font-weight="bold">0.42 persons/m²</text>
      <text x="185" y="52" fill="#495057" font-size="11">Normal Range</text>
    </g>

    <!-- Card 4 -->
    <g transform="translate(915, 0)">
      <rect width="340" height="75" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
      <text x="16" y="24" fill="#6c757d" font-size="11" font-weight="bold">PARKING OCCUPANCY RATE</text>
      <text x="16" y="54" fill="#111111" font-size="22" font-weight="bold">84% Occupied</text>
      <text x="175" y="52" fill="#495057" font-size="11">168 / 200 Bays</text>
    </g>
  </g>

  <!-- Main Grid: Two Video Tile Outlines (Left) + Incident Feed (Right) -->
  <g transform="translate(25, 180)">
    <!-- Video Tile 1 Wireframe -->
    <g transform="translate(0, 0)" font-family="system-ui, sans-serif">
      <rect width="445" height="280" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
      <rect width="445" height="28" fill="#f1f3f5" stroke="#ced4da" stroke-width="1"/>
      <text x="12" y="18" fill="#111111" font-size="11" font-weight="bold">CAM-01: Railway Platform 2 North [Simulated Feed]</text>
      <text x="430" y="18" text-anchor="end" fill="#6c757d" font-family="monospace" font-size="10">15 FPS | 1080p</text>

      <!-- Video Area Wireframe Elements -->
      <g transform="translate(0, 28)">
        <rect width="445" height="252" fill="#fafafa"/>
        <!-- Perspective grid lines -->
        <line x1="30" y1="240" x2="180" y2="30" stroke="#dee2e6" stroke-width="1"/>
        <line x1="415" y1="240" x2="265" y2="30" stroke="#dee2e6" stroke-width="1"/>

        <!-- BBox 1 -->
        <rect x="70" y="80" width="35" height="85" fill="none" stroke="#212529" stroke-width="1.5"/>
        <rect x="70" y="66" width="65" height="14" fill="#343a40"/>
        <text x="73" y="77" fill="#ffffff" font-family="monospace" font-size="9">Person 0.94</text>

        <!-- BBox 2 -->
        <rect x="130" y="100" width="40" height="90" fill="none" stroke="#212529" stroke-width="1.5"/>
        <rect x="130" y="86" width="65" height="14" fill="#343a40"/>
        <text x="133" y="97" fill="#ffffff" font-family="monospace" font-size="9">Person 0.89</text>

        <!-- BBox 3 (Warning Dwell) -->
        <rect x="220" y="75" width="38" height="80" fill="none" stroke="#212529" stroke-width="2"/>
        <rect x="220" y="61" width="65" height="14" fill="#212529"/>
        <text x="223" y="72" fill="#ffffff" font-family="monospace" font-size="9">Person 0.91</text>

        <!-- Polygon Zone Boundary -->
        <polygon points="50,210 200,75 255,75 120,210" fill="#e9ecef" fill-opacity="0.5" stroke="#495057" stroke-width="1.5" stroke-dasharray="4,2"/>
        <text x="70" y="200" fill="#212529" font-size="10" font-weight="bold">ZONE: Platform Edge (Warning Active)</text>
      </g>
    </g>

    <!-- Video Tile 2 Wireframe -->
    <g transform="translate(465, 0)" font-family="system-ui, sans-serif">
      <rect width="445" height="280" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
      <rect width="445" height="28" fill="#f1f3f5" stroke="#ced4da" stroke-width="1"/>
      <text x="12" y="18" fill="#111111" font-size="11" font-weight="bold">CAM-04: North Parking Lot Zone B [Simulated Feed]</text>
      <text x="430" y="18" text-anchor="end" fill="#6c757d" font-family="monospace" font-size="10">12 FPS | 1080p</text>

      <g transform="translate(0, 28)">
        <rect width="445" height="252" fill="#fafafa"/>

        <!-- Bay 1 (Occupied) -->
        <polygon points="40,150 120,150 100,210 10,210" fill="#e9ecef" stroke="#212529" stroke-width="1.5"/>
        <text x="30" y="190" fill="#111111" font-family="monospace" font-size="9" font-weight="bold">B-01 [OCCUPIED]</text>

        <!-- Bay 2 (Empty) -->
        <polygon points="140,150 220,150 200,210 110,210" fill="#ffffff" stroke="#6c757d" stroke-width="1.5"/>
        <text x="135" y="190" fill="#495057" font-family="monospace" font-size="9" font-weight="bold">B-02 [AVAILABLE]</text>

        <!-- Bay 3 (Occupied Vehicle) -->
        <polygon points="240,150 320,150 300,210 210,210" fill="#e9ecef" stroke="#212529" stroke-width="1.5"/>
        <rect x="235" y="160" width="70" height="40" fill="none" stroke="#212529" stroke-width="1.5"/>
        <text x="235" y="190" fill="#111111" font-family="monospace" font-size="9" font-weight="bold">B-03 [CAR 0.95]</text>

        <!-- Bay 4 (Empty) -->
        <polygon points="340,150 420,150 400,210 310,210" fill="#ffffff" stroke="#6c757d" stroke-width="1.5"/>
        <text x="335" y="190" fill="#495057" font-family="monospace" font-size="9" font-weight="bold">B-04 [AVAILABLE]</text>
      </g>
    </g>

    <!-- Right Side: Real-Time Incident Feed -->
    <g transform="translate(930, 0)" font-family="system-ui, sans-serif">
      <rect width="325" height="580" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
      <rect width="325" height="32" fill="#f1f3f5" stroke="#ced4da" stroke-width="1"/>
      <text x="15" y="21" fill="#111111" font-size="12" font-weight="bold">REAL-TIME INCIDENT FEED [PROPOSED]</text>

      <!-- Incident 1 -->
      <g transform="translate(12, 45)">
        <rect width="300" height="110" fill="#f8f9fa" stroke="#343a40" stroke-width="1"/>
        <rect x="8" y="8" width="60" height="18" fill="#343a40"/>
        <text x="38" y="20" text-anchor="middle" fill="#ffffff" font-size="9" font-weight="bold">CRITICAL</text>
        <text x="75" y="20" fill="#6c757d" font-family="monospace" font-size="10">18:42:15 UTC</text>
        <text x="10" y="44" fill="#111111" font-size="12" font-weight="bold">Zone Intrusion Detected</text>
        <text x="10" y="62" fill="#495057" font-size="11">CAM-07 • Server Room East Corridor</text>
        
        <rect x="10" y="75" width="130" height="24" fill="#ffffff" stroke="#343a40" stroke-width="1"/>
        <text x="75" y="90" text-anchor="middle" fill="#111111" font-size="10" font-weight="bold">Inspect Snapshot</text>
        
        <rect x="150" y="75" width="140" height="24" fill="#343a40"/>
        <text x="220" y="90" text-anchor="middle" fill="#ffffff" font-size="10" font-weight="bold">Acknowledge Alert</text>
      </g>

      <!-- Incident 2 -->
      <g transform="translate(12, 165)">
        <rect width="300" height="110" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
        <rect x="8" y="8" width="60" height="18" fill="#6c757d"/>
        <text x="38" y="20" text-anchor="middle" fill="#ffffff" font-size="9" font-weight="bold">WARNING</text>
        <text x="75" y="20" fill="#6c757d" font-family="monospace" font-size="10">18:39:02 UTC</text>
        <text x="10" y="44" fill="#111111" font-size="12" font-weight="bold">Overcrowding Threshold (85%)</text>
        <text x="10" y="62" fill="#495057" font-size="11">CAM-01 • Platform 2 Gate</text>

        <rect x="10" y="75" width="130" height="24" fill="#ffffff" stroke="#343a40" stroke-width="1"/>
        <text x="75" y="90" text-anchor="middle" fill="#111111" font-size="10" font-weight="bold">Inspect Snapshot</text>

        <rect x="150" y="75" width="140" height="24" fill="#343a40"/>
        <text x="220" y="90" text-anchor="middle" fill="#ffffff" font-size="10" font-weight="bold">Acknowledge Alert</text>
      </g>

      <!-- Incident 3 -->
      <g transform="translate(12, 285)">
        <rect width="300" height="110" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
        <rect x="8" y="8" width="50" height="18" fill="#adb5bd"/>
        <text x="33" y="20" text-anchor="middle" fill="#111111" font-size="9" font-weight="bold">INFO</text>
        <text x="65" y="20" fill="#6c757d" font-family="monospace" font-size="10">18:31:40 UTC</text>
        <text x="10" y="44" fill="#111111" font-size="12" font-weight="bold">Queue Congestion (Length: 9)</text>
        <text x="10" y="62" fill="#495057" font-size="11">CAM-03 • Cafe Checkout Counter</text>

        <rect x="10" y="75" width="130" height="24" fill="#ffffff" stroke="#343a40" stroke-width="1"/>
        <text x="75" y="90" text-anchor="middle" fill="#111111" font-size="10" font-weight="bold">Inspect Snapshot</text>

        <rect x="150" y="75" width="140" height="24" fill="#343a40"/>
        <text x="220" y="90" text-anchor="middle" fill="#ffffff" font-size="10" font-weight="bold">Acknowledge Alert</text>
      </g>

      <!-- Incident 4 -->
      <g transform="translate(12, 405)">
        <rect width="300" height="95" fill="#f8f9fa" stroke="#ced4da" stroke-width="1"/>
        <rect x="8" y="8" width="70" height="18" fill="#495057"/>
        <text x="43" y="20" text-anchor="middle" fill="#ffffff" font-size="9" font-weight="bold">RESOLVED</text>
        <text x="85" y="20" fill="#6c757d" font-family="monospace" font-size="10">18:22:10 UTC</text>
        <text x="10" y="44" fill="#111111" font-size="12" font-weight="bold">Unattended Table Cleared</text>
        <text x="10" y="62" fill="#495057" font-size="11">CAM-02 • Table 14 Serviced by staff</text>
      </g>
    </g>

    <!-- Bottom Left: Fleet Health Table -->
    <g transform="translate(0, 290)" font-family="system-ui, sans-serif" font-size="11">
      <rect width="910" height="290" fill="#ffffff" stroke="#343a40" stroke-width="1.5"/>
      <rect width="910" height="28" fill="#f1f3f5" stroke="#ced4da" stroke-width="1"/>
      <text x="15" y="18" fill="#111111" font-weight="bold">CAMERA FLEET HEALTH &amp; TELEMETRY [PROPOSED UI TABLE]</text>

      <!-- Table Header -->
      <g transform="translate(15, 45)" fill="#6c757d" font-weight="bold">
        <text x="0" y="0">CAMERA ID &amp; NAME</text>
        <text x="240" y="0">LOCATION</text>
        <text x="440" y="0">STATUS</text>
        <text x="560" y="0">FPS</text>
        <text x="650" y="0">LATENCY</text>
        <text x="760" y="0">ACTIVE RULES</text>
      </g>

      <!-- Rows -->
      <g transform="translate(15, 60)" fill="#212529">
        <line x1="0" y1="0" x2="880" y2="0" stroke="#dee2e6" stroke-width="1"/>
        <text x="0" y="20">CAM-01 Platform 2 North</text>
        <text x="240" y="20" fill="#495057">Central Metro Station</text>
        <text x="440" y="20">Streaming</text>
        <text x="560" y="20">15.0</text>
        <text x="650" y="20">38 ms</text>
        <text x="760" y="20">Crowd Surge, Edge Safety</text>

        <line x1="0" y1="35" x2="880" y2="35" stroke="#dee2e6" stroke-width="1"/>
        <text x="0" y="55">CAM-02 Main Dining Room</text>
        <text x="240" y="55" fill="#495057">Downtown Cafe</text>
        <text x="440" y="55">Streaming</text>
        <text x="560" y="55">10.0</text>
        <text x="650" y="55">42 ms</text>
        <text x="760" y="55">Table Turnover, Queue</text>

        <line x1="0" y1="70" x2="880" y2="70" stroke="#dee2e6" stroke-width="1"/>
        <text x="0" y="90">CAM-03 Food Court Queues</text>
        <text x="240" y="90" fill="#495057">Downtown Cafe</text>
        <text x="440" y="90">Streaming</text>
        <text x="560" y="90">12.0</text>
        <text x="650" y="90">35 ms</text>
        <text x="760" y="90">Queue Length &gt; 6</text>

        <line x1="0" y1="105" x2="880" y2="105" stroke="#dee2e6" stroke-width="1"/>
        <text x="0" y="125">CAM-04 North Parking Lot</text>
        <text x="240" y="125" fill="#495057">Commercial Plaza</text>
        <text x="440" y="125">Streaming</text>
        <text x="560" y="125">12.0</text>
        <text x="650" y="125">40 ms</text>
        <text x="760" y="125">Bay Occupancy, No-Parking</text>

        <line x1="0" y1="140" x2="880" y2="140" stroke="#dee2e6" stroke-width="1"/>
        <text x="0" y="160">CAM-07 Server Room East</text>
        <text x="240" y="160" fill="#495057">HQ Facility</text>
        <text x="440" y="160">Streaming</text>
        <text x="560" y="160">15.0</text>
        <text x="650" y="160">34 ms</text>
        <text x="760" y="160">Restricted Zone Entry</text>
      </g>
    </g>
  </g>
</svg>'''
    render(svg, "dashboard-mockup.png")

# ========================================================
# 10. ANALYTICS CHARTS - Strictly Monochrome with Disclaimer
# ========================================================
def gen_monochrome_analytics():
    # 1. crowd-density-trend.png (800x450)
    svg1 = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">
  <rect width="800" height="450" fill="#ffffff"/>
  <rect x="1" y="1" width="798" height="448" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="40" y="34" fill="#111111" font-family="system-ui, sans-serif" font-size="15" font-weight="bold">
    Platform Crowd Density Over 24 Hours [Illustrative Synthetic Data — Not Actual Lenz Results]
  </text>
  <text x="40" y="54" fill="#6c757d" font-family="system-ui, sans-serif" font-size="11">
    Demonstration of threshold evaluation logic; numbers represent simulated scenario testing
  </text>

  <!-- Axes -->
  <line x1="60" y1="360" x2="740" y2="360" stroke="#212529" stroke-width="1.5"/>
  <line x1="60" y1="90" x2="60" y2="360" stroke="#212529" stroke-width="1.5"/>

  <!-- Y axis grid & labels -->
  <g font-family="system-ui, sans-serif" font-size="11" fill="#495057" text-anchor="end">
    <text x="50" y="365">0%</text>
    <line x1="60" y1="360" x2="740" y2="360" stroke="#dee2e6" stroke-width="1"/>
    <text x="50" y="295">25%</text>
    <line x1="60" y1="290" x2="740" y2="290" stroke="#dee2e6" stroke-width="1"/>
    <text x="50" y="225">50%</text>
    <line x1="60" y1="220" x2="740" y2="220" stroke="#dee2e6" stroke-width="1"/>
    <text x="50" y="155">75%</text>
    <line x1="60" y1="150" x2="740" y2="150" stroke="#dee2e6" stroke-width="1"/>
    <text x="50" y="95">100%</text>
    <line x1="60" y1="90" x2="740" y2="90" stroke="#dee2e6" stroke-width="1"/>
  </g>

  <!-- Critical Threshold Line at 80% (Dashed Black Line) -->
  <line x1="60" y1="136" x2="740" y2="136" stroke="#111111" stroke-width="1.5" stroke-dasharray="6,4"/>
  <text x="735" y="128" text-anchor="end" fill="#111111" font-family="system-ui, sans-serif" font-size="11" font-weight="bold">
    Configurable Overcrowding Threshold (80% Capacity)
  </text>

  <!-- X axis labels (Hours) -->
  <g font-family="system-ui, sans-serif" font-size="11" fill="#495057" text-anchor="middle">
    <text x="60" y="380">00:00</text>
    <text x="145" y="380">04:00</text>
    <text x="230" y="380">08:00</text>
    <text x="315" y="380">12:00</text>
    <text x="400" y="380">16:00</text>
    <text x="485" y="380">18:00</text>
    <text x="570" y="380">20:00</text>
    <text x="655" y="380">22:00</text>
    <text x="740" y="380">24:00</text>
  </g>

  <!-- Density Curve (Monochrome Solid Black Line with Light Grey Fill) -->
  <path d="M 60 345 Q 120 350 180 320 T 240 140 T 320 220 T 420 200 T 500 120 T 600 240 T 740 350 L 740 360 L 60 360 Z" fill="#e9ecef" fill-opacity="0.6"/>
  <path d="M 60 345 Q 120 350 180 320 T 240 140 T 320 220 T 420 200 T 500 120 T 600 240 T 740 350" fill="none" stroke="#111111" stroke-width="2.5"/>

  <!-- Peak indicators -->
  <circle cx="240" cy="140" r="4" fill="#111111"/>
  <text x="240" y="125" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="10" font-weight="bold">Simulated Peak (86%)</text>

  <circle cx="500" cy="120" r="4" fill="#111111"/>
  <text x="500" y="105" text-anchor="middle" fill="#111111" font-family="system-ui, sans-serif" font-size="10" font-weight="bold">Simulated Rush (92%)</text>

  <text x="400" y="425" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Synthetic baseline curve demonstrating threshold breach triggering. No real-world telemetry is asserted.
  </text>
</svg>'''
    render(svg1, "crowd-density-trend.png", is_analytics=True)

    # 2. parking-occupancy-trend.png (800x450)
    svg2 = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">
  <rect width="800" height="450" fill="#ffffff"/>
  <rect x="1" y="1" width="798" height="448" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="40" y="34" fill="#111111" font-family="system-ui, sans-serif" font-size="15" font-weight="bold">
    Smart Parking Occupancy by Zone [Illustrative Synthetic Data — Not Actual Lenz Results]
  </text>
  <text x="40" y="54" fill="#6c757d" font-family="system-ui, sans-serif" font-size="11">
    Demonstration of per-zone occupancy tracking across operational hours
  </text>

  <line x1="60" y1="360" x2="740" y2="360" stroke="#212529" stroke-width="1.5"/>
  <line x1="60" y1="90" x2="60" y2="360" stroke="#212529" stroke-width="1.5"/>

  <g font-family="system-ui, sans-serif" font-size="11" fill="#495057" text-anchor="end">
    <text x="50" y="365">0%</text>
    <text x="50" y="295">30%</text>
    <text x="50" y="225">60%</text>
    <text x="50" y="155">90%</text>
  </g>

  <!-- Zone Curves (Monochrome: Solid, Dashed, Dotted) -->
  <path d="M 60 330 Q 180 320 260 160 T 420 180 T 540 150 T 740 310" fill="none" stroke="#111111" stroke-width="2"/>
  <path d="M 60 340 Q 180 335 260 210 T 420 220 T 540 190 T 740 330" fill="none" stroke="#495057" stroke-width="2" stroke-dasharray="6,4"/>
  <path d="M 60 350 Q 180 300 260 240 T 420 150 T 540 260 T 740 340" fill="none" stroke="#868e96" stroke-width="2" stroke-dasharray="2,3"/>

  <!-- Legend -->
  <g transform="translate(430, 40)" font-family="system-ui, sans-serif" font-size="11" fill="#212529">
    <line x1="0" y1="10" x2="25" y2="10" stroke="#111111" stroke-width="2"/>
    <text x="32" y="14">Zone A: Surface (Solid)</text>
    
    <line x1="160" y1="10" x2="185" y2="10" stroke="#495057" stroke-width="2" stroke-dasharray="6,4"/>
    <text x="192" y="14">Zone B: Covered (Dash)</text>
  </g>

  <!-- X axis labels (Hours) -->
  <g font-family="system-ui, sans-serif" font-size="11" fill="#495057" text-anchor="middle">
    <text x="60" y="380">06:00</text>
    <text x="230" y="380">10:00</text>
    <text x="400" y="380">14:00</text>
    <text x="570" y="380">18:00</text>
    <text x="740" y="380">22:00</text>
  </g>

  <text x="400" y="425" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Synthetic scenario illustrating multi-zone vehicle occupancy data model.
  </text>
</svg>'''
    render(svg2, "parking-occupancy-trend.png", is_analytics=True)

    # 3. events-by-category.png (800x450)
    svg3 = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">
  <rect width="800" height="450" fill="#ffffff"/>
  <rect x="1" y="1" width="798" height="448" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="40" y="34" fill="#111111" font-family="system-ui, sans-serif" font-size="15" font-weight="bold">
    Detections by Operational Category [Illustrative Synthetic Data — Not Actual Lenz Results]
  </text>
  <text x="40" y="54" fill="#6c757d" font-family="system-ui, sans-serif" font-size="11">
    Conceptual category distribution schema for event query aggregation
  </text>

  <!-- Horizontal Bars (Monochrome Shading) -->
  <g transform="translate(60, 95)" font-family="system-ui, sans-serif">
    <text x="0" y="24" fill="#111111" font-size="12" font-weight="bold">Crowd Overcrowding Surge</text>
    <rect x="230" y="8" width="340" height="24" fill="#212529"/>
    <text x="585" y="25" fill="#111111" font-size="11">34% (Simulated Test Set)</text>

    <text x="0" y="74" fill="#111111" font-size="12" font-weight="bold">Queue Congestion Delay</text>
    <rect x="230" y="58" width="280" height="24" fill="#495057"/>
    <text x="525" y="75" fill="#111111" font-size="11">28% (Simulated Test Set)</text>

    <text x="0" y="124" fill="#111111" font-size="12" font-weight="bold">Parking Bay Occupancy</text>
    <rect x="230" y="108" width="180" height="24" fill="#6c757d"/>
    <text x="425" y="125" fill="#111111" font-size="11">18% (Simulated Test Set)</text>

    <text x="0" y="174" fill="#111111" font-size="12" font-weight="bold">Unattended Area / Table</text>
    <rect x="230" y="158" width="120" height="24" fill="#adb5bd"/>
    <text x="365" y="175" fill="#111111" font-size="11">12% (Simulated Test Set)</text>

    <text x="0" y="224" fill="#111111" font-size="12" font-weight="bold">Restricted Zone Intrusion</text>
    <rect x="230" y="208" width="80" height="24" fill="#ced4da" stroke="#495057" stroke-width="1"/>
    <text x="325" y="225" fill="#111111" font-size="11">8% (Simulated Test Set)</text>
  </g>

  <text x="400" y="425" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Conceptual category breakdown. Ratios demonstrate aggregation API schema, not deployed production counts.
  </text>
</svg>'''
    render(svg3, "events-by-category.png", is_analytics=True)

    # 4. alert-severity-distribution.png (800x450)
    svg4 = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">
  <rect width="800" height="450" fill="#ffffff"/>
  <rect x="1" y="1" width="798" height="448" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="40" y="34" fill="#111111" font-family="system-ui, sans-serif" font-size="15" font-weight="bold">
    Alert Severity Triage Specification [Illustrative Synthetic Data — Not Actual Lenz Results]
  </text>
  <text x="40" y="54" fill="#6c757d" font-family="system-ui, sans-serif" font-size="11">
    Design specification of operator triage priority levels and acknowledgment targets
  </text>

  <!-- 4 Stat Cards in Monochrome -->
  <g transform="translate(50, 95)" font-family="system-ui, sans-serif">
    <!-- Critical Card -->
    <rect width="160" height="270" fill="#ffffff" stroke="#111111" stroke-width="2"/>
    <rect width="160" height="28" fill="#111111"/>
    <text x="80" y="19" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="bold">LEVEL: CRITICAL</text>
    <text x="80" y="80" text-anchor="middle" fill="#111111" font-size="28" font-weight="bold">5%</text>
    <text x="80" y="110" text-anchor="middle" fill="#495057" font-size="11">Triage Tier 1</text>
    <line x1="20" y1="135" x2="140" y2="135" stroke="#dee2e6" stroke-width="1"/>
    <text x="80" y="160" text-anchor="middle" fill="#6c757d" font-size="10">Target Acknowledge</text>
    <text x="80" y="180" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">&lt; 60 Seconds</text>
    <text x="80" y="225" text-anchor="middle" fill="#495057" font-size="10">Intrusion / Stampede</text>

    <!-- Warning Card -->
    <g transform="translate(180, 0)">
      <rect width="160" height="270" fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>
      <rect width="160" height="28" fill="#495057"/>
      <text x="80" y="19" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="bold">LEVEL: WARNING</text>
      <text x="80" y="80" text-anchor="middle" fill="#111111" font-size="28" font-weight="bold">22%</text>
      <text x="80" y="110" text-anchor="middle" fill="#495057" font-size="11">Triage Tier 2</text>
      <line x1="20" y1="135" x2="140" y2="135" stroke="#dee2e6" stroke-width="1"/>
      <text x="80" y="160" text-anchor="middle" fill="#6c757d" font-size="10">Target Acknowledge</text>
      <text x="80" y="180" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">&lt; 3.0 Minutes</text>
      <text x="80" y="225" text-anchor="middle" fill="#495057" font-size="10">Crowd / Loitering</text>
    </g>

    <!-- Info Card -->
    <g transform="translate(360, 0)">
      <rect width="160" height="270" fill="#f8f9fa" stroke="#ced4da" stroke-width="1.5"/>
      <rect width="160" height="28" fill="#6c757d"/>
      <text x="80" y="19" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="bold">LEVEL: INFO</text>
      <text x="80" y="80" text-anchor="middle" fill="#111111" font-size="28" font-weight="bold">48%</text>
      <text x="80" y="110" text-anchor="middle" fill="#495057" font-size="11">Operational Log</text>
      <line x1="20" y1="135" x2="140" y2="135" stroke="#dee2e6" stroke-width="1"/>
      <text x="80" y="160" text-anchor="middle" fill="#6c757d" font-size="10">Target Action</text>
      <text x="80" y="180" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">Automated Audit</text>
      <text x="80" y="225" text-anchor="middle" fill="#495057" font-size="10">Queue / Table Cycle</text>
    </g>

    <!-- Resolved Card -->
    <g transform="translate(540, 0)">
      <rect width="160" height="270" fill="#ffffff" stroke="#ced4da" stroke-width="1.5"/>
      <rect width="160" height="28" fill="#ced4da"/>
      <text x="80" y="19" text-anchor="middle" fill="#111111" font-size="11" font-weight="bold">STATUS: RESOLVED</text>
      <text x="80" y="80" text-anchor="middle" fill="#111111" font-size="28" font-weight="bold">25%</text>
      <text x="80" y="110" text-anchor="middle" fill="#495057" font-size="11">Operator Verified</text>
      <line x1="20" y1="135" x2="140" y2="135" stroke="#dee2e6" stroke-width="1"/>
      <text x="80" y="160" text-anchor="middle" fill="#6c757d" font-size="10">Feedback Loop</text>
      <text x="80" y="180" text-anchor="middle" fill="#111111" font-size="13" font-weight="bold">Active Learning</text>
      <text x="80" y="225" text-anchor="middle" fill="#495057" font-size="10">Dismissal / Cleared</text>
    </g>
  </g>

  <text x="400" y="425" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Specification model for triage service. Does not represent production incident statistics.
  </text>
</svg>'''
    render(svg4, "alert-severity-distribution.png", is_analytics=True)

    # 5. hourly-event-frequency.png (800x450)
    svg5 = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">
  <rect width="800" height="450" fill="#ffffff"/>
  <rect x="1" y="1" width="798" height="448" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="40" y="34" fill="#111111" font-family="system-ui, sans-serif" font-size="15" font-weight="bold">
    Hourly Event Trigger Distribution [Illustrative Synthetic Data — Not Actual Lenz Results]
  </text>
  <text x="40" y="54" fill="#6c757d" font-family="system-ui, sans-serif" font-size="11">
    Simulated 24-hour event occurrence frequency demonstrating time-bucket aggregation API
  </text>

  <line x1="60" y1="360" x2="740" y2="360" stroke="#212529" stroke-width="1.5"/>

  <!-- 24 Hourly Monochrome Bars -->
  <g transform="translate(60, 0)">
    <!-- Night (Light Grey) -->
    <rect x="0" y="340" width="18" height="20" fill="#ced4da"/>
    <rect x="28" y="345" width="18" height="15" fill="#ced4da"/>
    <rect x="56" y="348" width="18" height="12" fill="#ced4da"/>
    <rect x="84" y="350" width="18" height="10" fill="#ced4da"/>
    <rect x="112" y="340" width="18" height="20" fill="#ced4da"/>
    <rect x="140" y="310" width="18" height="50" fill="#adb5bd"/>
    <!-- Morning Rush (Dark Charcoal) -->
    <rect x="168" y="240" width="18" height="120" fill="#495057"/>
    <rect x="196" y="160" width="18" height="200" fill="#212529"/>
    <rect x="224" y="130" width="18" height="230" fill="#212529"/>
    <rect x="252" y="190" width="18" height="170" fill="#495057"/>
    <rect x="280" y="220" width="18" height="140" fill="#495057"/>
    <!-- Lunch -->
    <rect x="308" y="150" width="18" height="210" fill="#212529"/>
    <rect x="336" y="160" width="18" height="200" fill="#212529"/>
    <rect x="364" y="240" width="18" height="120" fill="#495057"/>
    <rect x="392" y="260" width="18" height="100" fill="#495057"/>
    <rect x="420" y="230" width="18" height="130" fill="#495057"/>
    <!-- Evening Rush -->
    <rect x="448" y="140" width="18" height="220" fill="#212529"/>
    <rect x="476" y="110" width="18" height="250" fill="#212529"/>
    <rect x="504" y="150" width="18" height="210" fill="#212529"/>
    <rect x="532" y="220" width="18" height="140" fill="#495057"/>
    <rect x="560" y="280" width="18" height="80" fill="#495057"/>
    <rect x="588" y="310" width="18" height="50" fill="#adb5bd"/>
    <rect x="616" y="330" width="18" height="30" fill="#ced4da"/>
    <rect x="644" y="340" width="18" height="20" fill="#ced4da"/>
  </g>

  <!-- X axis labels -->
  <g font-family="system-ui, sans-serif" font-size="11" fill="#495057">
    <text x="60" y="380">00:00</text>
    <text x="195" y="380">05:00</text>
    <text x="335" y="380">10:00</text>
    <text x="475" y="380">15:00</text>
    <text x="615" y="380">20:00</text>
    <text x="704" y="380">23:00</text>
  </g>

  <text x="400" y="425" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Synthetic distribution illustrating time-of-day query aggregation. No live deployment telemetry is claimed.
  </text>
</svg>'''
    render(svg5, "hourly-event-frequency.png", is_analytics=True)

    # 6. camera-processing-latency.png (800x450)
    svg6 = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">
  <rect width="800" height="450" fill="#ffffff"/>
  <rect x="1" y="1" width="798" height="448" fill="none" stroke="#ced4da" stroke-width="1"/>

  <text x="40" y="34" fill="#111111" font-family="system-ui, sans-serif" font-size="15" font-weight="bold">
    Projected Latency Scaling vs. Concurrent Streams [Theoretical Design Analysis]
  </text>
  <text x="40" y="54" fill="#6c757d" font-family="system-ui, sans-serif" font-size="11">
    Analytical projection based on published YOLOv8 benchmarks on NVIDIA TensorRT vs. CPU ONNX Runtime
  </text>

  <line x1="60" y1="360" x2="740" y2="360" stroke="#212529" stroke-width="1.5"/>
  <line x1="60" y1="90" x2="60" y2="360" stroke="#212529" stroke-width="1.5"/>

  <g font-family="system-ui, sans-serif" font-size="11" fill="#495057" text-anchor="end">
    <text x="50" y="365">0ms</text>
    <text x="50" y="300">50ms</text>
    <text x="50" y="235">100ms</text>
    <text x="50" y="170">150ms</text>
    <text x="50" y="105">200ms</text>
  </g>

  <!-- Stream counts on X -->
  <g font-family="system-ui, sans-serif" font-size="11" fill="#495057" text-anchor="middle">
    <text x="140" y="380">1 Stream</text>
    <text x="290" y="380">4 Streams</text>
    <text x="440" y="380">8 Streams</text>
    <text x="590" y="380">16 Streams</text>
    <text x="710" y="380">32 Streams</text>
  </g>

  <!-- CPU Curve (Dashed Black Line) -->
  <path d="M 140 280 L 290 220 L 440 160 L 590 120" fill="none" stroke="#495057" stroke-width="2" stroke-dasharray="6,4"/>
  <circle cx="140" cy="280" r="3.5" fill="#495057"/>
  <circle cx="290" cy="220" r="3.5" fill="#495057"/>
  <circle cx="440" cy="160" r="3.5" fill="#495057"/>
  <circle cx="590" cy="120" r="3.5" fill="#495057"/>

  <!-- GPU TensorRT Curve (Solid Black Line) -->
  <path d="M 140 335 L 290 325 L 440 310 L 590 280 L 710 240" fill="none" stroke="#111111" stroke-width="2.5"/>
  <circle cx="140" cy="335" r="4" fill="#111111"/>
  <circle cx="290" cy="325" r="4" fill="#111111"/>
  <circle cx="440" cy="310" r="4" fill="#111111"/>
  <circle cx="590" cy="280" r="4" fill="#111111"/>
  <circle cx="710" cy="240" r="4" fill="#111111"/>

  <!-- Legend -->
  <g transform="translate(420, 38)" font-family="system-ui, sans-serif" font-size="11" fill="#212529">
    <line x1="0" y1="10" x2="25" y2="10" stroke="#111111" stroke-width="2.5"/>
    <text x="32" y="14">GPU TensorRT FP16 (Projected)</text>
    
    <line x1="210" y1="10" x2="235" y2="10" stroke="#495057" stroke-width="2" stroke-dasharray="6,4"/>
    <text x="242" y="14">Host CPU ONNX (Projected)</text>
  </g>

  <text x="400" y="425" text-anchor="middle" fill="#868e96" font-family="system-ui, sans-serif" font-size="10">
    Source: Analytical scaling model based on Ultralytics published YOLOv8 benchmarks. Lenz has not completed physical hardware testing.
  </text>
</svg>'''
    render(svg6, "camera-processing-latency.png", is_analytics=True)

if __name__ == "__main__":
    print("Generating Monochrome Hero Banner...")
    gen_hero()
    print("Generating Monochrome Problem vs. Solution...")
    gen_problem_vs_solution()
    print("Generating Monochrome System Workflow...")
    gen_system_workflow()
    print("Generating Monochrome System Architecture...")
    gen_system_architecture()
    print("Generating Monochrome Database ER Diagram...")
    gen_database_er()
    print("Generating Monochrome AI Pipeline...")
    gen_ai_pipeline()
    print("Generating Monochrome Use Case...")
    gen_use_case()
    print("Generating Monochrome Data Flow...")
    gen_data_flow()
    print("Generating Monochrome Dashboard Mockup...")
    gen_dashboard_mockup()
    print("Generating Monochrome Analytics Charts...")
    gen_monochrome_analytics()
    print("All professional monochrome visual assets successfully generated!")
