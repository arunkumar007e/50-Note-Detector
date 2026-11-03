# INR 50 Note Detector

A Python-based computer vision project that detects the new Indian ₹50 banknote in a live webcam feed using YOLO and OpenCV.

The detector identifies the new light blue-green ₹50 note featuring Mahatma Gandhi on the front and the Hampi stone chariot on the back. It is designed to distinguish ₹50 notes from other denominations.

## Features

- ₹50 banknote detection using YOLO
- Live webcam detection using OpenCV
- Bounding-box annotation and confidence score
- Dataset analysis and integrity checking
- YOLO dataset preparation
- Model training and evaluation
- ONNX model export for OpenCV inference
- Optional local browser-based webcam interface

## Dataset

The dataset contains 311 original phone photographs:

- 106 images of ₹50 notes (front and back)
- 205 images of other denominations (₹10, ₹20, ₹100, ₹200 and ₹500)
- Original image resolution: 3000×4000 or 4000×3000
- No corrupt images or exact duplicate files detected during initial inspection

**Download Dataset:** [Google Drive Dataset](https://drive.google.com/file/d/1XxeiSXhBm6tZHcI7kXXh-H97p4wOC4x2/view?usp=sharing)

Download and extract the dataset into the project directory before preparing the training data.

## Requirements

- Python 3.10–3.12
- Webcam
- OpenCV
- Ultralytics YOLO
- Optional NVIDIA GPU for faster training

## Installation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training Workflow

### 1. Analyze Dataset

```powershell
python -m src.analyze_dataset
```

Checks image integrity, dimensions, and duplicate files.

### 2. Annotate ₹50 Notes

```powershell
python -m src.annotate
```

Draw bounding boxes around the complete ₹50 note in every target image and save the annotations.

### 3. Prepare YOLO Dataset

```powershell
python -m src.prepare_dataset
```

Creates the training, validation, and test dataset using the annotations and background images.

### 4. Train Model

```powershell
python -m src.train
```

### 5. Evaluate Model

```powershell
python -m src.evaluate --weights runs/train/inr50/weights/best.pt --split test
```

### 6. Export ONNX Model

```powershell
python -m src.export_onnx --weights runs/train/inr50/weights/best.pt
```

## Run Webcam Detection

After exporting the ONNX model:

```powershell
python test.py --model model.onnx
```

The webcam displays a bounding box labelled ₹50 along with the detection confidence.

Press `q` to exit.

Optional arguments:

```powershell
python test.py --model model.onnx --conf 0.65 --camera 0 --imgsz 640
```

## Optional Web Interface

```powershell
python -m src.web_app
```

Open:

```text
http://127.0.0.1:5000
```

Allow camera access to use the local browser-based detection interface.

## Run Tests

```powershell
python -m pytest
```

## Project Structure

```text
inr50-note-detector/
├── Dataset/              # Original note photographs
├── annotations/          # Bounding-box annotations
├── yolo_data/            # Prepared YOLO dataset
├── src/                  # Training and inference code
├── tests/                 # Unit tests
├── reports/               # Dataset analysis reports
├── test.py                # OpenCV webcam detection
├── requirements.txt
└── README.md

```

## Limitations

- Training requires manually reviewed bounding-box annotations.
- Only the new ₹50 banknote design is considered a positive class.
- This is a banknote detection demo, not a counterfeit detection or currency authentication system.
- Detection accuracy depends on dataset quality, annotation quality, and model training.
- Ultralytics is licensed under AGPL-3.0. Review its license before distributing the project.
