import argparse
import threading
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from src.inference import OnnxNoteDetector
from src.paths import DEFAULT_MODEL

MAX_FRAME_BYTES = 8 * 1024 * 1024


def create_app(model_path: Path = DEFAULT_MODEL) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_FRAME_BYTES
    detector = (
        OnnxNoteDetector(model_path, image_size=640, confidence=0.65)
        if model_path.is_file()
        else None
    )
    inference_lock = threading.Lock()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify(
            {
                "model_available": detector is not None,
                "message": (
                    "Detector ready."
                    if detector is not None
                    else "Export a trained model to model.onnx to enable detection."
                ),
            }
        )

    @app.post("/api/detect")
    def detect():
        if detector is None:
            return jsonify({"error": "Export a trained model to model.onnx first."}), 503

        uploaded_frame = request.files.get("frame")
        if uploaded_frame is None:
            return jsonify({"error": "Upload a webcam frame using the 'frame' field."}), 400

        encoded_frame = uploaded_frame.read()
        frame = cv2.imdecode(np.frombuffer(encoded_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "The uploaded file is not a readable image."}), 400

        with inference_lock:
            detections = detector.detect(frame)
        return jsonify(
            {
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "detections": [asdict(detection) for detection in detections],
            }
        )

    @app.errorhandler(RequestEntityTooLarge)
    def frame_too_large(_error):
        return jsonify({"error": "Webcam frame exceeds the 8 MB limit."}), 413

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the INR 50 detector on localhost.")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")

    app = create_app()
    print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
