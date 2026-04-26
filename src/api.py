from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from src.explain_prediction import explain_features
    from src.feature_extractor import extract_features
except ModuleNotFoundError:
    from explain_prediction import explain_features
    from feature_extractor import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "feature_columns.pkl"


def _load_ml_artifacts() -> tuple[Any, Any, List[str]]:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_columns = list(joblib.load(FEATURE_COLUMNS_PATH))
    return model, label_encoder, feature_columns


MODEL, LABEL_ENCODER, FEATURE_COLUMNS = _load_ml_artifacts()


def _feature_frame(features: Dict[str, Any]) -> pd.DataFrame:
    row: Dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        value = features.get(column, 0.0)
        if isinstance(value, bool):
            row[column] = float(int(value))
        else:
            row[column] = float(value)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _safe_probabilities(model: Any, x_frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_frame)
        return np.asarray(probabilities[0], dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_frame), dtype=float)
        if scores.ndim == 1:
            scores = np.array([scores[0], -scores[0]], dtype=float)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        return probs.astype(float)

    predicted = int(model.predict(x_frame)[0])
    one_hot = np.zeros(len(LABEL_ENCODER.classes_), dtype=float)
    if 0 <= predicted < len(one_hot):
        one_hot[predicted] = 1.0
    return one_hot


def _build_legal_flags(features: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if bool(features.get("inside_restricted_polygon", False)):
        flags.append("Inside restricted polygon")
    if bool(features.get("inside_water_polygon", False)):
        flags.append("Inside water polygon")
    return flags


def _build_environmental_flags(features: Dict[str, Any]) -> List[str]:
    flags: List[str] = []

    if int(features.get("within_water_50", 0)):
        flags.append("Water body within 50m")
    elif int(features.get("within_water_150", 0)):
        flags.append("Water body within 150m")

    if int(features.get("within_forest_100", 0)):
        flags.append("Forest area within 100m")
    elif int(features.get("within_forest_300", 0)):
        flags.append("Forest area within 300m")

    if int(features.get("within_restricted_50", 0)):
        flags.append("Restricted land within 50m")
    elif int(features.get("within_restricted_150", 0)):
        flags.append("Restricted land within 150m")

    nearby_count = int(features.get("nearby_sensitive_layer_count", 0))
    if nearby_count >= 2:
        flags.append("Multiple sensitive layers nearby")

    return flags

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount the /data directory so frontend map can load geojson shape overlays
app.mount("/data", StaticFiles(directory="data"), name="data")

class Location(BaseModel):
    latitude: float
    longitude: float


@app.post("/analyze")
def analyze(loc: Location):
    features = extract_features(loc.latitude, loc.longitude)
    x_frame = _feature_frame(features)

    predicted_encoded = int(MODEL.predict(x_frame)[0])
    predicted_label = str(LABEL_ENCODER.inverse_transform([predicted_encoded])[0])

    probabilities = _safe_probabilities(MODEL, x_frame)
    class_labels = [str(label) for label in LABEL_ENCODER.classes_]
    confidence_map = {
        class_labels[index]: round(float(probabilities[index]), 6)
        for index in range(min(len(class_labels), len(probabilities)))
    }
    base_confidence = confidence_map.get(predicted_label, 0.0)

    explanations = explain_features(features)
    top_ai_reasons = explanations.get("top_factors", [])

    legal_flags = _build_legal_flags(features)
    environmental_flags = _build_environmental_flags(features)

    final_label = predicted_label
    if bool(features.get("inside_restricted_polygon", False)) or bool(features.get("inside_water_polygon", False)):
        final_label = "HIGH"

    explanation_parts = [item.get("reason", "") for item in top_ai_reasons if item.get("reason")]
    explanation_parts.extend(legal_flags)
    explanation_text = " | ".join(dict.fromkeys(explanation_parts)) if explanation_parts else "Model-based geospatial assessment."

    distances = {
        "water_distance_m": round(float(features.get("dist_water", 0.0)), 6),
        "forest_distance_m": round(float(features.get("dist_forest", 0.0)), 6),
        "restricted_distance_m": round(float(features.get("dist_restricted", 0.0)), 6),
    }

    response = {
        "coordinates": {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
        },
        "risk": {
            "predicted": predicted_label,
            "final": final_label,
            "override_applied": final_label != predicted_label,
        },
        "confidence": {
            "predicted_class": round(float(base_confidence), 6),
            "class_probabilities": confidence_map,
        },
        "distances": distances,
        "top_ai_reasons": top_ai_reasons,
        "feature_importance": {
            "method": explanations.get("method", "unknown"),
            "top_factors": top_ai_reasons,
        },
        "legal_flags": legal_flags,
        "environmental_flags": environmental_flags,
        "raw_features": features,
        # Backward-compatible fields for existing frontend contract.
        "risk_level": final_label,
        "explanation": explanation_text,
    }

    return response
