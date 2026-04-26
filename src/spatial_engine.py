import geopandas as gpd
from shapely.geometry import Point
import joblib
import pandas as pd
import os
from src.raster_engine import get_elevation_and_slope, get_landcover_class

# -----------------------------
# LOAD DATA AND MODEL
# -----------------------------
# Load fallback OpenStreetMap vectors
forest = gpd.read_file("data/forest_clean.geojson").to_crs(epsg=3857)
restricted = gpd.read_file("data/restricted_clean.geojson").to_crs(epsg=3857)

# Load High-Resolution Data (if compiled by compile_hydrology.py)
# Otherwise, safely fallback to the OSM layer
hydro_streams = None
hydro_reservoirs = None

if os.path.exists("data/streams_karnataka.parquet"):
    hydro_streams = gpd.read_parquet("data/streams_karnataka.parquet")
if os.path.exists("data/reservoirs_karnataka.parquet"):
    hydro_reservoirs = gpd.read_parquet("data/reservoirs_karnataka.parquet")
if not hydro_streams is not None and not hydro_reservoirs is not None:
    # Safe fallback if ETL script hasn't been run yet
    try:
        hydro_reservoirs = gpd.read_file("data/water_clean.geojson").to_crs(epsg=3857)
    except:
        pass

# Load the enhanced XGBoost model and its label mapping
try:
    saved_bundle = joblib.load("model.pkl")
    ml_model = saved_bundle["model"]
    label_mapping = saved_bundle["mapping"]
except FileNotFoundError:
    print("WARNING: model.pkl not found. Run train_model.py first.")
    ml_model = None


def analyze_location(lat: float, lon: float):
    # -------------------------
    # 1. EXACT VECTOR DISTANCES
    # -------------------------
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)[0]
    
    forest_dist = float(forest.distance(point).min()) if not forest.empty else 9999
    restricted_dist = float(restricted.distance(point).min()) if not restricted.empty else 9999
    
    stream_dist = float(hydro_streams.distance(point).min()) if (hydro_streams is not None and not hydro_streams.empty) else 9999
    reservoir_dist = float(hydro_reservoirs.distance(point).min()) if (hydro_reservoirs is not None and not hydro_reservoirs.empty) else 9999

    # -------------------------
    # 2. RASTER FEATURES
    # -------------------------
    raster_topo = get_elevation_and_slope(lat, lon)
    landcover = get_landcover_class(lat, lon)
    
    # -------------------------
    # 3. XGBoost PREDICTION (SOFT BASELINE)
    # -------------------------
    risk_level = "UNKNOWN"
    explanation = "Model uninitialized."
    flags = []
    
    if ml_model is not None:
        features = pd.DataFrame([{
            "reservoir_dist": reservoir_dist,
            "stream_dist": stream_dist,
            "forest_dist": forest_dist,
            "restricted_dist": restricted_dist,
            "slope_degrees": raster_topo["slope_degrees"]
        }])
        
        try:
            pred_idx = ml_model.predict(features)[0]
            risk_level = label_mapping.get(pred_idx, "UNKNOWN")
        except Exception as e:
            risk_level = "UNKNOWN"

    # -------------------------
    # 4. EPICENTER OVERRIDES (HARD LIMITS)
    # -------------------------
    # Rule: ALL data including reservoirs, streams, forests, restricted areas MUST be HIGH RISK epicenters natively if < 50m.
    
    is_epicenter = False
    
    # HIGH overrides (EPICENTERS)
    if reservoir_dist < 50: 
        risk_level = "HIGH"
        flags.append("Inside major reservoir or tank epicenter (High Flood/Submersion Risk)")
        is_epicenter = True
    if stream_dist < 50:
        risk_level = "HIGH"
        flags.append("Inside stream/river drainage network epicenter")
        is_epicenter = True
    if restricted_dist < 50:
        risk_level = "HIGH"
        flags.append("Inside restricted government property / military zone")
        is_epicenter = True
    if forest_dist < 50:
        risk_level = "HIGH"
        flags.append("Inside natural forest or protected green epicenter")
        is_epicenter = True
        
    # MEDIUM overrides (Buffer Zones)
    if not is_epicenter:
        if reservoir_dist >= 50 and reservoir_dist < 200:
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
            flags.append("Near reservoir/lake catchment zone")
        if stream_dist >= 50 and stream_dist < 150:
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
            flags.append("Near river/stream bed")
        if restricted_dist >= 50 and restricted_dist < 150:
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
            flags.append("Near restricted property buffer")
        if forest_dist >= 50 and forest_dist < 200:
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
            flags.append("Adjacent to protected forest line")

    # Environmental additions
    if raster_topo["slope_degrees"] > 25: flags.append(f"Unstable incline ({raster_topo['slope_degrees']}°)")
    elif raster_topo["slope_degrees"] > 10: flags.append(f"Moderate incline ({raster_topo['slope_degrees']}°)")
    
    if len(flags) > 0:
        explanation = " | ".join(flags)
    else:
        explanation = "Stable terrain; no immediate epicenter proximity."

    # -------------------------
    # RETURN CLEAN OUTPUT
    # -------------------------
    return {
        "risk_level": risk_level,
        "explanation": explanation,
        "distances": {
            "reservoir_distance_m": round(reservoir_dist, 2),
            "stream_distance_m": round(stream_dist, 2),
            "forest_distance_m": round(forest_dist, 2),
            "restricted_distance_m": round(restricted_dist, 2),
        },
        "topography": raster_topo,
        "landcover_class": landcover
    }