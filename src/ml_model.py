import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

def predict_risk_ml(result: dict):
    # Create DataFrame with SAME feature names used in training
    df = pd.DataFrame([{
        "water_dist": result["water_distance_m"],
        "forest_dist": result["forest_distance_m"],
        "restricted_dist": result["restricted_distance_m"]
    }])

    prediction = model.predict(df)[0]
    return prediction