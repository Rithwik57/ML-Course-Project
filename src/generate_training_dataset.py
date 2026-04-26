from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from src.feature_extractor import extract_features
except ModuleNotFoundError:
    from feature_extractor import extract_features


# Karnataka-ish bounding box (approximate)
KARNATAKA_LAT_RANGE: Tuple[float, float] = (11.5, 18.6)
KARNATAKA_LON_RANGE: Tuple[float, float] = (74.0, 78.7)

# Minimum required sample size
MIN_SAMPLES = 20_000
DEFAULT_SAMPLES = 20_000

# Severity normalization controls (meters)
RESTRICTED_SCALE_M = 300.0
WATER_SCALE_M = 400.0
FOREST_SCALE_M = 500.0

OUTPUT_PATH = Path("data") / "training_dataset.csv"


def inverse_distance_severity(distance_m: float, scale_m: float) -> float:
    """
    Convert distance (meters) to severity in [0, 1].
    Closer distance yields higher severity.
    """
    if distance_m <= 0:
        return 1.0
    severity = 1.0 / (1.0 + (distance_m / scale_m))
    return float(np.clip(severity, 0.0, 1.0))


def compute_risk_score(features: Dict[str, float | int | bool]) -> float:
    """
    RiskScore = 0.4*restricted_component + 0.3*water_component
              + 0.2*forest_component + 0.1*overlap_component
    """
    restricted_component = inverse_distance_severity(
        float(features["dist_restricted"]), RESTRICTED_SCALE_M
    )
    water_component = inverse_distance_severity(
        float(features["dist_water"]), WATER_SCALE_M
    )
    forest_component = inverse_distance_severity(
        float(features["dist_forest"]), FOREST_SCALE_M
    )

    overlap_count = float(features["nearby_sensitive_layer_count"])
    overlap_strength = float(features["weighted_overlap_score"])
    overlap_component = np.clip((overlap_count / 3.0) * 0.5 + (overlap_strength / 3.0) * 0.5, 0.0, 1.0)

    risk_score = (
        0.4 * restricted_component
        + 0.3 * water_component
        + 0.2 * forest_component
        + 0.1 * float(overlap_component)
    )
    return float(np.clip(risk_score, 0.0, 1.0))


def label_from_score(score: float) -> str:
    if score > 0.70:
        return "HIGH"
    if 0.35 <= score <= 0.70:
        return "MEDIUM"
    return "LOW"


def generate_random_points(
    n_samples: int,
    lat_range: Tuple[float, float] = KARNATAKA_LAT_RANGE,
    lon_range: Tuple[float, float] = KARNATAKA_LON_RANGE,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latitudes = rng.uniform(lat_range[0], lat_range[1], size=n_samples)
    longitudes = rng.uniform(lon_range[0], lon_range[1], size=n_samples)
    return latitudes, longitudes


def build_training_dataset(n_samples: int = DEFAULT_SAMPLES) -> pd.DataFrame:
    if n_samples < MIN_SAMPLES:
        raise ValueError(f"n_samples must be at least {MIN_SAMPLES}")

    latitudes, longitudes = generate_random_points(n_samples)

    rows: List[Dict[str, float | int | bool | str]] = []
    for idx, (latitude, longitude) in enumerate(zip(latitudes, longitudes), start=1):
        features = extract_features(float(latitude), float(longitude))
        score = compute_risk_score(features)
        label = label_from_score(score)

        row: Dict[str, float | int | bool | str] = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            **features,
            "risk_score": round(score, 6),
            "risk_label": label,
        }
        rows.append(row)

        if idx % 2000 == 0:
            print(f"Processed {idx}/{n_samples} points...")

    return pd.DataFrame(rows)


def print_class_distribution(df: pd.DataFrame) -> None:
    counts = df["risk_label"].value_counts().sort_index()
    ratios = df["risk_label"].value_counts(normalize=True).sort_index() * 100.0

    print("\nClass distribution summary:")
    for label in ["LOW", "MEDIUM", "HIGH"]:
        count = int(counts.get(label, 0))
        pct = float(ratios.get(label, 0.0))
        print(f"  {label:<6}: {count:>6} ({pct:>6.2f}%)")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic dataset with {DEFAULT_SAMPLES} samples...")
    dataset = build_training_dataset(DEFAULT_SAMPLES)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved dataset to: {OUTPUT_PATH}")
    print(f"Total rows: {len(dataset)}")
    print_class_distribution(dataset)


if __name__ == "__main__":
    main()
