from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "training_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model.pkl"
ENCODER_PATH = PROJECT_ROOT / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "feature_columns.pkl"

TARGET_COLUMN = "risk_label"
EXCLUDED_COLUMNS = {"latitude", "longitude", "risk_score", TARGET_COLUMN}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in dataset.")

    return df


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_columns = [col for col in df.columns if col not in EXCLUDED_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns found after exclusions.")
    return feature_columns


def build_models(label_count: int) -> List[Tuple[str, object]]:
    models: List[Tuple[str, object]] = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softmax",
                    num_class=label_count,
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                ),
            )
        )
    except Exception:
        print("[INFO] xgboost not installed or unavailable. Skipping XGBoost.")

    return models


def print_feature_importances(model_name: str, model: object, feature_columns: List[str]) -> None:
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = getattr(model, "feature_importances_", None)
    elif model_name == "XGBoost" and hasattr(model, "get_booster"):
        booster = model.get_booster()
        score_map = booster.get_score(importance_type="gain")
        if score_map:
            importances = [score_map.get(f"f{idx}", 0.0) for idx in range(len(feature_columns))]

    if importances is None:
        print(f"[INFO] {model_name} does not expose feature importances.")
        return

    feature_importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False)

    print(f"\nTop feature importances for {model_name}:")
    print(feature_importance_df.head(15).to_string(index=False))


def train_and_benchmark(df: pd.DataFrame) -> Dict[str, object]:
    feature_columns = select_feature_columns(df)
    x_data = df[feature_columns].copy()
    y_labels = df[TARGET_COLUMN].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    models = build_models(label_count=len(label_encoder.classes_))
    if not models:
        raise RuntimeError("No models available for training.")

    best_result: Dict[str, object] | None = None

    print(f"Using features ({len(feature_columns)}): {feature_columns}")
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

    for model_name, model in models:
        print(f"\n--- Training {model_name} ---")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            digits=4,
            zero_division=0,
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print("Classification report:")
        print(report)

        print_feature_importances(model_name, model, feature_columns)

        result = {
            "model_name": model_name,
            "model": model,
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "feature_columns": feature_columns,
            "label_encoder": label_encoder,
        }

        if best_result is None or result["weighted_f1"] > best_result["weighted_f1"]:
            best_result = result

    if best_result is None:
        raise RuntimeError("Failed to train/evaluate any model.")

    return best_result


def persist_artifacts(best_result: Dict[str, object]) -> None:
    joblib.dump(best_result["model"], MODEL_PATH)
    joblib.dump(best_result["label_encoder"], ENCODER_PATH)
    joblib.dump(best_result["feature_columns"], FEATURE_COLUMNS_PATH)


def main() -> None:
    print("Loading training dataset...")
    df = load_dataset(DATASET_PATH)

    print("Starting supervised ensemble benchmark...")
    best_result = train_and_benchmark(df)

    print("\n=== Best Model Selected ===")
    print(f"Model: {best_result['model_name']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Weighted F1: {best_result['weighted_f1']:.4f}")

    persist_artifacts(best_result)
    print("\nSaved artifacts:")
    print(f"- {MODEL_PATH}")
    print(f"- {ENCODER_PATH}")
    print(f"- {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()