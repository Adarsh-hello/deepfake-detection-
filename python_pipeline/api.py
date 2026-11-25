"""FastAPI inference server loading deepfake_model.keras and serving predictions, metrics, and history."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import tensorflow as tf

from dataset_prep import IMAGE_SIZE

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "deepfake_model.keras"
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
HISTORY_LOG_PATH = ARTIFACTS_DIR / "detection_history.json"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
app = FastAPI(title="DEEPFAKE DETECTOR API", version="1.0.0")

_MODEL: tf.keras.Model | None = None


def get_model() -> tf.keras.Model:
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Trained model deepfake_model.keras not found. Run train.py first.",
            )
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
    return _MODEL


def preprocess_uploaded_image(raw: bytes) -> np.ndarray:
    """Validate image bytes and preprocess to (1, 128, 128, 3) normalized float32."""
    try:
        with Image.open(BytesIO(raw)) as img:
            rgb = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            return np.expand_dims(arr, axis=0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image file: {exc}")


def append_history(record: Dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []
    if HISTORY_LOG_PATH.exists():
        history = json.loads(HISTORY_LOG_PATH.read_text())
    history.insert(0, record)
    HISTORY_LOG_PATH.write_text(json.dumps(history[:50], indent=2))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Evaluation metrics not generated yet.")
    return json.loads(METRICS_PATH.read_text())


@app.get("/history")
def get_history():
    if not HISTORY_LOG_PATH.exists():
        return {"history": []}
    return {"history": json.loads(HISTORY_LOG_PATH.read_text())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only JPG, JPEG, and PNG images are accepted.",
        )

    raw = await file.read()
    batch = preprocess_uploaded_image(raw)
    model = get_model()

    prob_fake = float(model.predict(batch, verbose=0)[0][0])
    prediction = "FAKE" if prob_fake >= 0.5 else "REAL"
    confidence = round(max(prob_fake, 1.0 - prob_fake) * 100.0, 2)

    record = {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": round(prob_fake, 4),
        "real_probability": round(1.0 - prob_fake, 4),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    append_history(record)
    return record
