from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "training_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model.pkl"
ENCODER_PATH = PROJECT_ROOT / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "feature_columns.pkl"


def load_artifacts():
    missing = [
        str(path)
        for path in [MODEL_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH, DATASET_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required files not found. Run `python src/train_model.py` first. Missing:\n"
            + "\n".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    df = pd.read_csv(DATASET_PATH)

    return model, label_encoder, feature_columns, df


def run_smoke_test(sample_count: int = 5) -> None:
    model, label_encoder, feature_columns, df = load_artifacts()

    sample_df = df.sample(n=min(sample_count, len(df)), random_state=42).copy()
    x_sample = sample_df[feature_columns]

    encoded_pred = model.predict(x_sample)
    decoded_pred = label_encoder.inverse_transform(encoded_pred)

    results = sample_df[["latitude", "longitude", "risk_label"]].copy()
    results["predicted_label"] = decoded_pred
    results["match"] = results["risk_label"] == results["predicted_label"]

    print("\n=== Inference Smoke Test ===")
    print(f"Loaded model: {type(model).__name__}")
    print(f"Samples tested: {len(results)}")
    print(results.to_string(index=False))
    print(f"\nSample agreement: {results['match'].mean() * 100:.2f}%")


def main() -> None:
    run_smoke_test(sample_count=8)


if __name__ == "__main__":
    main()
