# Labelin

**End-to-end traffic object detection suite:** capture from live streams, auto-label with YOLO, refine in a visual workspace, then train your own model—all in one desktop app.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![YOLO](https://img.shields.io/badge/detection-YOLO%20%28Ultralytics%29-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## Overview

Labelin is a **professional, open-source desktop application** for building custom traffic detection datasets and models. It streamlines the full pipeline: **acquire → auto-label → correct → split → train** without leaving the GUI.

| Step | What it does |
|------|----------------|
| **1. Data Acquisition** | Capture frames from M3U8/RTSP streams; optionally pre-annotate with a YOLO model. |
| **2. Labeling Workspace** | Load raw or auto-labeled images, draw or edit bounding boxes (car, motorcycle, bus, truck), save in YOLO format. |
| **3. Dataset Prep** | Split images into train/val and generate a `traffic.yaml` for Ultralytics YOLO. |
| **4. Model Training** | Run YOLO training with configurable epochs, batch size, and image size; monitor logs in-app. |

Built with **PyQt6** and **OpenCV**; detection and training powered by **Ultralytics YOLO**.

---

## Features

- **Stream capture** — M3U8 (HLS) and RTSP; configurable frame count.
- **Auto-labeling** — Pre-annotate with a YOLO `.pt` model (COCO car/motorcycle/bus/truck → custom classes).
- **Visual labeling** — Draw, move, delete boxes; load raw or auto-labeled sets; save to `dataset_manual`.
- **Dataset split** — Train/val split with configurable ratio; auto-generated YAML.
- **Training** — YOLO training from the GUI (epochs, batch, image size); GPU auto-detected.
- **Dark UI** — Catnpp-inspired dark theme for long labeling sessions.

---

## Requirements

- **Python** 3.10+
- **GPU** optional but recommended for training (CUDA); runs on CPU otherwise.
- **YOLO base weights** (e.g. `yolov8n.pt` or `yolo11n.pt`) for auto-label and training.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Labelin.git
cd Labelin
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or:  .venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Ultralytics for auto-label and training

If you use **Step 1 (Auto-Label)** or **Step 4 (Training)**:

```bash
pip install ultralytics
```

---

## Quick Start

```bash
python main.py
```

1. **Tab 1 — Data Acquisition**  
   - Enter stream URL (M3U8/RTSP), set “Frames to Capture”, click **Start Capture**.  
   - Optionally set “YOLO Model Path” and click **Start Auto-Label** to pre-annotate `dataset_raw` → `dataset_labeled`.

2. **Tab 2 — Labeling Workspace**  
   - **Load Raw Images** or **Load Auto-Labeled**, pick a class, draw boxes on the canvas.  
   - Use **Prev/Next** to navigate; **Save (S)** or keyboard **S** to write to `dataset_manual/images` and `dataset_manual/labels`.

3. **Tab 3 — Dataset Prep**  
   - Set “Source Directory” to `dataset_manual` (or your labeled folder).  
   - Set train ratio (e.g. 0.8), click **Generate Train/Val Split & traffic.yaml**.  
   - Output: `dataset/train`, `dataset/val`, and `dataset/traffic.yaml`.

4. **Tab 4 — Model Training**  
   - Set “Traffic YAML Path” to `dataset/traffic.yaml`, choose base model (e.g. `yolo11n.pt`).  
   - Set epochs, batch size, image size, then **Start YOLO Training**.  
   - Best weights: `runs/yolo_traffic_gui/weights/best.pt`.

---

## Project layout (after use)

```
Labelin/
├── main.py                 # Application entry point
├── requirements.txt
├── README.md
├── dataset_raw/            # Captured frames (Step 1)
├── dataset_labeled/        # Auto-labeled images + labels (Step 1)
│   ├── images/
│   └── labels/
├── dataset_manual/        # Manually corrected images + labels (Step 2)
│   ├── images/
│   └── labels/
├── dataset/                # Train/val split + YAML (Step 3)
│   ├── train/images, train/labels
│   ├── val/images, val/labels
│   └── traffic.yaml
└── runs/                   # YOLO training outputs (Step 4)
    └── yolo_traffic_gui/
        └── weights/best.pt
```

---

## Classes (YOLO format)

Default classes and COCO mapping used in the app:

| ID | Class      | COCO ID (for auto-label) |
|----|------------|---------------------------|
| 0  | car        | 2                         |
| 1  | motorcycle | 3                         |
| 2  | bus        | 5                         |
| 3  | truck      | 7                         |

Labels are stored in normalized YOLO format: `class_id x_center y_center width height` (relative to image size).

---

## Keyboard shortcuts (Labeling tab)

- **S** — Save current image and labels to `dataset_manual`.
- **Delete / Backspace** — Remove selected bounding box.
- **Ctrl + Mouse wheel** — Zoom in/out on canvas.

---

## Configuration

- **Stream URL** — Any M3U8 (HLS) or RTSP URL; default example points to a public traffic stream.
- **YOLO model** — Path to a `.pt` file (e.g. `yolo11n.pt`, `yolov8n.pt`). Download from [Ultralytics](https://github.com/ultralytics/ultralytics) or train a previous run’s `best.pt`.
- **Paths** — All directories (e.g. `dataset_raw`, `dataset_manual`, `dataset/traffic.yaml`) can be adjusted in the UI where applicable.

---

## Contributing

Contributions are welcome: bug reports, feature ideas, or pull requests. Please open an issue first for larger changes so we can align on design.

1. Fork the repo.  
2. Create a branch (`git checkout -b feature/your-feature`).  
3. Commit with clear messages (`git commit -m 'Add feature X'`).  
4. Push and open a Pull Request.

---

## License

This project is open source under the [MIT License](LICENSE). You are free to use, modify, and distribute it, with attribution.

---

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detection and training.
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the desktop GUI.
- [OpenCV](https://opencv.org/) for image and video I/O.

---

*Labelin — from stream to trained model, in one place.*
