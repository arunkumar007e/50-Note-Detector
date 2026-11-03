import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from src.paths import REPORTS_DIR, YOLO_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained model on held-out data.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=YOLO_DATA_DIR / "data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "evaluation.json")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Trained weights not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"YOLO dataset config not found: {args.data}")

    metrics = YOLO(str(args.weights)).val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        plots=True,
        verbose=True,
    )
    report = {
        "weights": str(args.weights.resolve()),
        "split": args.split,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Metrics written to {args.output}")


if __name__ == "__main__":
    main()
