from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.inference import Detection
from src.web_app import create_app


def test_status_reports_missing_model_and_disables_detection(tmp_path: Path) -> None:
    app = create_app(tmp_path / "missing.onnx")
    client = app.test_client()

    status = client.get("/api/status")
    response = client.post("/api/detect", data={"frame": (b"", "frame.jpg")})

    assert status.json == {
        "model_available": False,
        "message": "Export a trained model to model.onnx to enable detection.",
    }
    assert response.status_code == 503


def test_detect_returns_boxes_from_uploaded_webcam_frame(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    detection = Detection(10, 20, 100, 120, 0.9)
    with patch("src.web_app.OnnxNoteDetector") as detector_type:
        detector_type.return_value.detect.return_value = [detection]
        app = create_app(model_path)

    success, encoded = cv2.imencode(".jpg", np.zeros((160, 200, 3), dtype=np.uint8))
    assert success
    response = app.test_client().post(
        "/api/detect",
        data={"frame": (BytesIO(encoded.tobytes()), "frame.jpg")},
    )

    assert response.status_code == 200
    assert response.json == {
        "width": 200,
        "height": 160,
        "detections": [
            {"x1": 10, "y1": 20, "x2": 100, "y2": 120, "confidence": 0.9}
        ],
    }


def test_detect_rejects_non_image_upload(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    with patch("src.web_app.OnnxNoteDetector"):
        app = create_app(model_path)

    response = app.test_client().post(
        "/api/detect",
        data={"frame": (BytesIO(b"not an image"), "frame.jpg")},
    )

    assert response.status_code == 400
    assert response.json == {"error": "The uploaded file is not a readable image."}
