import pandas as pd
import xgboost as xgb
import joblib

# -----------------------------
# STEP 1: Create enriched training data
# -----------------------------
# Features: [reservoir_dist, stream_dist, forest_dist, restricted_dist, slope_degrees]
# Risk labels: HIGH, MEDIUM, LOW

data = [
    # reservoir, stream, forest, restricted, slope, risk
    # High Risk Epicenter Examples
    [10,   5000, 5000, 5000, 45.0, "HIGH"], # Inside Reservoir + steep
    [5000, 30,   5000, 5000, 35.0, "HIGH"], # Inside Stream
    [5000, 5000, 5000, 20,   5.0,  "HIGH"], # Inside Restricted Area
    [5000, 5000, 10,   5000, 2.0,  "HIGH"], # Inside Forest
    
    # Medium Risk Buffer Examples
    [120,  5000, 5000, 5000, 2.0,  "MEDIUM"], # Near Reservoir
    [5000, 80,   5000, 5000, 2.0,  "MEDIUM"], # Near Stream
    [5000, 5000, 150,  5000, 10.0, "MEDIUM"], # Near forest
    [5000, 5000, 5000, 100,  5.0,  "MEDIUM"], # Near restricted
    
    # Low Risk Flat Terrain Examples
    [400,  8000, 8000, 8000, 2.0,  "LOW"],
    [600,  10000,10000,10000,0.5,  "LOW"],
    [800,  5000, 5000, 5000, 5.0,  "LOW"],
    
    # High Slope but Normal Area (Not Near Epicenters) Examples
    [5000, 5000, 5000, 5000, 35.0, "LOW"],
    [8000, 8000, 8000, 8000, 45.0, "LOW"],
    [5000, 5000, 5000, 5000, 25.0, "LOW"],
]

df = pd.DataFrame(data, columns=[
    "reservoir_dist",
    "stream_dist",
    "forest_dist",
    "restricted_dist",
    "slope_degrees",
    "risk"
])

# -----------------------------
# STEP 2: Encode labels and Train model
# -----------------------------
X = df[["reservoir_dist", "stream_dist", "forest_dist", "restricted_dist", "slope_degrees"]]

label_mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
y = df["risk"].map(label_mapping)

model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=3, 
    learning_rate=0.1, 
    objective='multi:softprob'
)
model.fit(X, y)

joblib.dump({"model": model, "mapping": {v: k for k, v in label_mapping.items()}}, "model.pkl")

print("XGBoost Model updated with all 4 geometry epicenter rules and saved as model.pkl")