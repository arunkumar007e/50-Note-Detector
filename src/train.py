import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from src.paths import RUNS_DIR, YOLO_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small INR 50 banknote detector.")
    parser.add_argument("--data", default=str(YOLO_DATA_DIR / "data.yaml"))
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--name", default="inr50")
    args = parser.parse_args()

    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("--epochs, --batch, and --imgsz must be positive (imgsz at least 32).")
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO dataset config not found: {data_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)
    model.train(
        data=str(data_path.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=20,
        project=str(RUNS_DIR),
        name=args.name,
        device=device,
        workers=0,
        seed=42,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
