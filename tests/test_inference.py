from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.inference import Detection, OnnxNoteDetector


class FakeNetwork:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def setInput(self, blob: np.ndarray) -> None:
        assert blob.shape == (1, 3, 640, 640)

    def forward(self) -> np.ndarray:
        return self.output


def test_detect_maps_one_class_box_back_to_source_frame(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    output = np.zeros((1, 5, 8400), dtype=np.float32)
    output[0, :, 0] = [320, 320, 200, 100, 0.9]
    with patch.object(cv2.dnn, "readNetFromONNX", return_value=FakeNetwork(output)):
        detector = OnnxNoteDetector(model_path)

    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0] == Detection(220, 190, 420, 290, pytest.approx(0.9))


def test_detect_rejects_non_bgr_input(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    with patch.object(
        cv2.dnn,
        "readNetFromONNX",
        return_value=FakeNetwork(np.zeros((1, 5, 8400), dtype=np.float32)),
    ):
        detector = OnnxNoteDetector(model_path)

    with pytest.raises(ValueError, match="three-channel BGR"):
        detector.detect(np.zeros((480, 640), dtype=np.uint8))
