from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "feature_columns.pkl"

_REASON_MAP = {
    "dist_restricted": "Restricted land extremely close",
    "dist_water": "High water body proximity",
    "dist_forest": "Forest ecological sensitivity nearby",
    "nearby_sensitive_layer_count": "Multiple sensitive zones nearby",
    "weighted_overlap_score": "High cumulative environmental overlap",
}

_MODEL = None
_FEATURE_COLUMNS: List[str] | None = None


def _load_artifacts() -> Tuple[object, List[str]]:
    global _MODEL, _FEATURE_COLUMNS

    if _MODEL is not None and _FEATURE_COLUMNS is not None:
        return _MODEL, _FEATURE_COLUMNS

    if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Please run `python src/train_model.py` first."
        )

    _MODEL = joblib.load(MODEL_PATH)
    _FEATURE_COLUMNS = list(joblib.load(FEATURE_COLUMNS_PATH))
    return _MODEL, _FEATURE_COLUMNS


def _human_reason(feature_name: str) -> str:
    return _REASON_MAP.get(feature_name, feature_name.replace("_", " ").capitalize())


def _normalize_contributions(items: List[Tuple[str, float]]) -> List[Dict[str, float | str]]:
    total = sum(value for _, value in items)
    if total <= 0:
        equal = 100.0 / max(1, len(items))
        return [
            {
                "feature": name,
                "reason": _human_reason(name),
                "percentage": round(equal, 2),
            }
            for name, _ in items
        ]

    return [
        {
            "feature": name,
            "reason": _human_reason(name),
            "percentage": round((value / total) * 100.0, 2),
        }
        for name, value in items
    ]


def _prepare_frame(feature_dict: Dict[str, float | int | bool], feature_columns: List[str]) -> pd.DataFrame:
    row = {}
    for column in feature_columns:
        raw_value = feature_dict.get(column, 0.0)
        if isinstance(raw_value, bool):
            row[column] = int(raw_value)
        else:
            row[column] = float(raw_value)
    return pd.DataFrame([row], columns=feature_columns)


def _explain_with_shap(model: object, x_row: pd.DataFrame, feature_columns: List[str]):
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_row)

    # Multi-class can return list[n_classes][n_features] or ndarray.
    predicted_class_index = int(model.predict(x_row)[0]) if hasattr(model, "predict") else 0

    if isinstance(shap_values, list):
        class_values = np.array(shap_values[predicted_class_index])[0]
    else:
        shap_array = np.array(shap_values)
        if shap_array.ndim == 3:
            class_values = shap_array[0, :, predicted_class_index]
        elif shap_array.ndim == 2:
            class_values = shap_array[0]
        else:
            class_values = shap_array.reshape(-1)

    abs_contrib = np.abs(class_values)
    pairs = list(zip(feature_columns, abs_contrib.tolist()))
    top3 = sorted(pairs, key=lambda pair: pair[1], reverse=True)[:3]
    return _normalize_contributions(top3), "shap"


def _explain_with_importances(model: object, feature_columns: List[str]):
    if not hasattr(model, "feature_importances_"):
        raise RuntimeError("Model does not provide feature importances for fallback explanation.")

    importances = np.abs(np.array(getattr(model, "feature_importances_"), dtype=float))
    pairs = list(zip(feature_columns, importances.tolist()))
    top3 = sorted(pairs, key=lambda pair: pair[1], reverse=True)[:3]
    return _normalize_contributions(top3), "feature_importances"


def explain_features(feature_dict: Dict[str, float | int | bool]) -> Dict[str, object]:
    """
    Explain a single prediction context using top 3 feature contributions.

    Returns:
    {
      "top_factors": [
        {"feature": "dist_restricted", "reason": "...", "percentage": 52.3},
        ...
      ],
      "method": "shap" | "feature_importances"
    }
    """
    model, feature_columns = _load_artifacts()
    x_row = _prepare_frame(feature_dict, feature_columns)

    try:
        top_factors, method = _explain_with_shap(model, x_row, feature_columns)
    except Exception:
        top_factors, method = _explain_with_importances(model, feature_columns)

    return {
        "top_factors": top_factors,
        "method": method,
    }


__all__ = ["explain_features"]
