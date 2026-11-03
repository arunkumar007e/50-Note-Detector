import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.paths import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained weights to OpenCV-compatible ONNX.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Trained weights not found: {args.weights}")

    exported = Path(
        YOLO(str(args.weights)).export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=False,
            opset=12,
            simplify=False,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != args.output.resolve():
        exported.replace(args.output)

    network = cv2.dnn.readNetFromONNX(str(args.output))
    network.setInput(np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32))
    output = network.forward()
    if output.size == 0:
        raise RuntimeError("OpenCV loaded the ONNX file but returned an empty inference output.")
    print(f"OpenCV loaded and ran {args.output} (output shape: {output.shape}).")


if __name__ == "__main__":
    main()
