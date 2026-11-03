from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class OnnxNoteDetector:
    def __init__(
        self,
        model_path: Path,
        image_size: int = 640,
        confidence: float = 0.65,
        nms_threshold: float = 0.45,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. Train and export the model first."
            )
        if image_size < 32:
            raise ValueError("Image size must be at least 32 pixels.")
        if not 0 <= confidence <= 1 or not 0 <= nms_threshold <= 1:
            raise ValueError("Confidence and NMS thresholds must be between 0 and 1.")
        self.image_size = image_size
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.network = cv2.dnn.readNetFromONNX(str(model_path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if (
            frame is None
            or frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError("Expected a non-empty three-channel BGR image.")

        input_blob, scale, pad_x, pad_y = self._letterbox(frame)
        self.network.setInput(input_blob)
        raw_output = self.network.forward()
        rows = np.asarray(raw_output)
        if rows.ndim == 3:
            rows = rows[0]
        if rows.ndim != 2:
            raise RuntimeError(f"Unexpected ONNX output shape: {raw_output.shape}")
        if rows.shape[0] < rows.shape[1] and rows.shape[0] <= 128:
            rows = rows.T
        if rows.shape[1] < 5:
            raise RuntimeError(f"Expected at least 5 values per detection; got {rows.shape}.")

        height, width = frame.shape[:2]
        boxes = []
        scores = []
        for row in rows:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if class_id != 0 or not np.isfinite(score) or score < self.confidence:
                continue
            center_x, center_y, box_width, box_height = map(float, row[:4])
            x1 = int(round((center_x - box_width / 2 - pad_x) / scale))
            y1 = int(round((center_y - box_height / 2 - pad_y) / scale))
            x2 = int(round((center_x + box_width / 2 - pad_x) / scale))
            y2 = int(round((center_y + box_height / 2 - pad_y) / scale))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)

        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence, self.nms_threshold
        )
        if len(indices) == 0:
            return []
        return [
            Detection(
                x1=boxes[int(index)][0],
                y1=boxes[int(index)][1],
                x2=boxes[int(index)][0] + boxes[int(index)][2],
                y2=boxes[int(index)][1] + boxes[int(index)][3],
                confidence=scores[int(index)],
            )
            for index in np.asarray(indices).reshape(-1)
        ]

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.image_size - resized_width) // 2
        pad_y = (self.image_size - resized_height) // 2
        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, scalefactor=1 / 255.0, size=(self.image_size, self.image_size), swapRB=True
        )
        return blob, scale, pad_x, pad_y


def open_camera(index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {index}. Check that it is connected and available.")
    return camera
