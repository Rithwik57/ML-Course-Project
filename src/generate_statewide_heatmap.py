from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform as shapely_transform, unary_union
import fiona

try:
    from src.feature_extractor import extract_features
except ModuleNotFoundError:
    from feature_extractor import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = PROJECT_ROOT / "model.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "feature_columns.pkl"
STATE_SHAPE_PATH = DATA_DIR / "State" / "State.shp"
OUTPUT_PATH = DATA_DIR / "karnataka_ai_risk_surface.geojson"

# Karnataka-ish coarse bounds (fallback)
LAT_MIN, LAT_MAX = 11.5, 18.6
LON_MIN, LON_MAX = 74.0, 78.7

# Coarse grid step for statewide overlay
GRID_STEP_DEGREES = 0.08


def load_model_artifacts() -> Tuple[Any, Any, List[str]]:
    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run `python src/train_model.py` first."
        )

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_columns = list(joblib.load(FEATURE_COLUMNS_PATH))
    return model, label_encoder, feature_columns


def load_karnataka_boundary() -> Any | None:
    if not STATE_SHAPE_PATH.exists():
        return None

    try:
        with fiona.open(STATE_SHAPE_PATH, ignore_fields=["created_da", "last_edi_1"]) as source:
            geometries = [shape(feature["geometry"]) for feature in source if feature.get("geometry") is not None]
            if not geometries:
                return None

            boundary = unary_union(geometries)
            source_crs = source.crs_wkt if source.crs_wkt else source.crs
            if source_crs:
                transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
                boundary = shapely_transform(transformer.transform, boundary)
            return boundary
    except Exception:
        return None


def generate_grid_points(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    step: float,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []

    lat_values = np.arange(lat_min, lat_max + 1e-9, step)
    lon_values = np.arange(lon_min, lon_max + 1e-9, step)

    for lat in lat_values:
        for lon in lon_values:
            points.append((round(float(lat), 6), round(float(lon), 6)))

    return points


def build_feature_frame(features: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    row: Dict[str, float] = {}
    for col in feature_columns:
        value = features.get(col, 0.0)
        if isinstance(value, bool):
            row[col] = float(int(value))
        else:
            row[col] = float(value)
    return pd.DataFrame([row], columns=feature_columns)


def predict_risk_and_confidence(
    model: Any,
    label_encoder: Any,
    x_frame: pd.DataFrame,
) -> Tuple[str, float]:
    encoded_pred = int(model.predict(x_frame)[0])
    predicted_label = str(label_encoder.inverse_transform([encoded_pred])[0])

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x_frame)[0], dtype=float)
        if 0 <= encoded_pred < len(probabilities):
            confidence = float(probabilities[encoded_pred])
        else:
            confidence = float(np.max(probabilities))

    return predicted_label, round(confidence, 6)


def to_geojson_feature(lat: float, lon: float, risk_label: str, confidence: float) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat],
        },
        "properties": {
            "predicted_risk": risk_label,
            "confidence": confidence,
        },
    }


def main() -> None:
    print("Loading model artifacts...")
    model, label_encoder, feature_columns = load_model_artifacts()

    print("Loading Karnataka boundary...")
    boundary = load_karnataka_boundary()

    print("Generating coarse statewide grid...")
    candidate_points = generate_grid_points(
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        step=GRID_STEP_DEGREES,
    )

    print(f"Candidate points: {len(candidate_points)}")

    features_out: List[Dict[str, Any]] = []
    kept = 0
    skipped = 0

    for index, (lat, lon) in enumerate(candidate_points, start=1):
        if boundary is not None and not boundary.covers(Point(lon, lat)):
            skipped += 1
            continue

        feature_dict = extract_features(lat, lon)
        x_row = build_feature_frame(feature_dict, feature_columns)
        risk_label, confidence = predict_risk_and_confidence(model, label_encoder, x_row)

        features_out.append(to_geojson_feature(lat, lon, risk_label, confidence))
        kept += 1

        if index % 500 == 0:
            print(f"Processed {index}/{len(candidate_points)} points (kept={kept}, skipped={skipped})")

    geojson = {
        "type": "FeatureCollection",
        "features": features_out,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(geojson, file_obj)

    print("\nStatewide AI heatmap generation complete.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Features written: {kept}")
    print(f"Skipped points: {skipped}")


if __name__ == "__main__":
    main()
