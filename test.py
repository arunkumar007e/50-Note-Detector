import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.inference import OnnxNoteDetector, open_camera
from src.paths import DEFAULT_MODEL


def load_label_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/Nirmala.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_detections(frame, detections, font):
    for detection in detections:
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (40, 220, 100),
            3,
        )
    canvas = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        label = f"₹50 {detection.confidence:.2f}"
        try:
            draw.text(
                (detection.x1, max(0, detection.y1 - 30)),
                label,
                font=font,
                fill=(30, 190, 80),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        except (UnicodeEncodeError, ValueError):
            draw.text(
                (detection.x1, max(0, detection.y1 - 30)),
                f"INR 50 {detection.confidence:.2f}",
                font=font,
                fill=(30, 190, 80),
            )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect new INR 50 notes in a live webcam feed.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    detector = OnnxNoteDetector(args.model, image_size=args.imgsz, confidence=args.conf)
    camera = open_camera(args.camera)
    font = load_label_font(24)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The camera stopped returning frames.")
            detections = detector.detect(frame)
            frame = draw_detections(frame, detections, font)
            cv2.imshow("INR 50 note detector (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()