from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
ANNOTATIONS_DIR = ROOT / "annotations" / "labels"
YOLO_DATA_DIR = ROOT / "yolo_data"
RUNS_DIR = ROOT / "runs" / "train"
REPORTS_DIR = ROOT / "reports"
DEFAULT_MODEL = ROOT / "model.onnx"
CLASS_NAME = "note_50"
TARGET_FOLDERS = {"INR50_Front", "INR50_Back"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
