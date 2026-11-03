# INR 50 Note Detector

A Python project for detecting the new light blue-green Indian ₹50 banknote
(Gandhi portrait on the front, Hampi chariot on the back) in a live webcam feed.
It is a local training pipeline with an OpenCV webcam script and an optional
localhost browser interface. The browser interface sends webcam frames to the
local Python detector; no external service or database is used.

The project uses the existing `Dataset/` folder as read-only input. It contains
311 JPEGs (106 in `INR50_Front` and `INR50_Back`, plus 205 images of other
denominations). The Windows image decoder opened all 311 files for metadata
inspection and reported dimensions of 3000×4000 or 4000×3000. The Python
integrity and duplicate scan remains available through the analysis command.
The original photos are not copied into Git.
See [the initial dataset inventory](reports/dataset_inventory.md) for per-folder
counts and inspection notes.

## Requirements

- Python 3.10–3.12
- A webcam for live detection
- Optional NVIDIA GPU for faster training; CPU training is supported

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ultralytics uses a small pretrained YOLO nano checkpoint by default. Its first
use may download the checkpoint. The project dataset itself is the supplied
phone-photo data; no other banknote photos are added.

## Workflow

1. Inspect the dataset and write an exact-duplicate/corruption report:

   ```powershell
   python -m src.analyze_dataset
   ```

2. Label the note boundary in every `INR50_Front` and `INR50_Back` photo:

   ```powershell
   python -m src.annotate
   ```

   Drag a tight box around the complete note and press Enter to save. Escape
   skips an image; skipped images remain unlabeled and prevent preparation.
   Labels are stored separately under `annotations/labels/`, leaving the
   supplied images untouched. Review the saved boxes before continuing. This
   manual review is necessary because the supplied folders have no bounding-box
   annotations and image-based guesses are not reliable labels.

3. Build the one-class YOLO dataset:

   ```powershell
   python -m src.prepare_dataset
   ```

   Photos from the same filename timestamp minute stay together across all
   folders to reduce near-duplicate leakage; the split balances folder counts
   as much as possible. The non-₹50 denominations become background-only
   examples. Images are hard-linked where possible and copied only when hard
   links are unavailable. The generated `yolo_data/` directory must be empty or new.

4. Train and evaluate:

   ```powershell
   python -m src.train
   python -m src.evaluate --weights runs/train/inr50/weights/best.pt --split test
   ```

   Training metrics are not a guarantee of webcam performance. Test with held
   out photos and real notes; raise `--conf` if any non-₹50 object triggers a
   false positive.

5. Export to ONNX and run the webcam detector:

   ```powershell
   python -m src.export_onnx --weights runs/train/inr50/weights/best.pt
   python test.py --model model.onnx
   ```

   Optional inference settings: `--conf 0.65`, `--camera 0`, and `--imgsz 640`.
   Press `q` in the camera window to exit. OpenCV loads and runs the ONNX model
   during export as a compatibility check.

   To use the browser-based webcam interface instead, start the local server
   after exporting the model:

   ```powershell
   python -m src.web_app
   ```

   Open `http://127.0.0.1:5000` in your browser and allow camera access. The
   server binds to localhost only. A trained `model.onnx` is required for
   detection; without it, the page displays the next setup steps.

6. Run unit tests:

   ```powershell
   python -m pytest
   ```

## Important limitations

- The supplied data has no labels. Training cannot run until every target photo
  has a reviewed box.
- Only the provided sample photos have been inspected; confirm all `INR50_*`
  images contain the new ₹50 design and are photos you are allowed to use.
- High precision is important: the webcam script only displays class `note_50`
  and defaults to a confidence threshold of 0.65. This is a demo detector, not
  a currency-authentication or counterfeit-detection system.
- Ultralytics is AGPL-3.0. Review its terms before publishing or distributing
  this project.
