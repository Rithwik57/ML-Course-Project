from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform as shapely_transform


# -----------------------------------------------------------------------------
# Module-level configuration
# -----------------------------------------------------------------------------
# We keep all layers loaded once so repeated API calls / training loops are fast.
_MODULE_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _MODULE_ROOT / "data"

_WGS84_CRS = "EPSG:4326"
_METRIC_CRS = "EPSG:3857"

_LAYER_FILES = {
    "water": _DATA_DIR / "water_clean.geojson",
    "forest": _DATA_DIR / "forest_clean.geojson",
    "restricted": _DATA_DIR / "restricted_clean.geojson",
}

# Finite fallback distance for rare cases where a layer is empty/missing.
_FALLBACK_DISTANCE_M = 1_000_000.0


# -----------------------------------------------------------------------------
# Layer loading helpers
# -----------------------------------------------------------------------------
def _load_layer_pair(path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load a layer in WGS84 and in metric CRS for precise distance operations."""
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs(_WGS84_CRS)
    else:
        gdf = gdf.to_crs(_WGS84_CRS)

    gdf_metric = gdf.to_crs(_METRIC_CRS)
    return gdf, gdf_metric


def _safe_unary_union(gdf: gpd.GeoDataFrame):
    """Return unary union geometry if possible, else None."""
    if gdf.empty:
        return None
    return gdf.unary_union


# Load all layers once at import-time (requirement #1).
WATER_GDF_WGS84, WATER_GDF_METRIC = _load_layer_pair(_LAYER_FILES["water"])
FOREST_GDF_WGS84, FOREST_GDF_METRIC = _load_layer_pair(_LAYER_FILES["forest"])
RESTRICTED_GDF_WGS84, RESTRICTED_GDF_METRIC = _load_layer_pair(_LAYER_FILES["restricted"])

WATER_UNION_WGS84 = _safe_unary_union(WATER_GDF_WGS84)
FOREST_UNION_WGS84 = _safe_unary_union(FOREST_GDF_WGS84)
RESTRICTED_UNION_WGS84 = _safe_unary_union(RESTRICTED_GDF_WGS84)


# Reused transformer object for fast point projection.
_TO_METRIC = Transformer.from_crs(_WGS84_CRS, _METRIC_CRS, always_xy=True)


# -----------------------------------------------------------------------------
# Feature helpers
# -----------------------------------------------------------------------------
def _point_from_lat_lon(latitude: float, longitude: float) -> Point:
    """Create a WGS84 point from latitude/longitude with basic validation."""
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        raise ValueError("Latitude and longitude must be finite numbers.")

    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        raise ValueError("Latitude/longitude out of valid geographic range.")

    return Point(longitude, latitude)


def _to_metric_point(point_wgs84: Point) -> Point:
    """Project WGS84 point into metric CRS so distances are in meters."""
    return shapely_transform(_TO_METRIC.transform, point_wgs84)


def _distance_m(layer_metric: gpd.GeoDataFrame, point_metric: Point) -> float:
    """Minimum point-to-layer distance in meters (finite fallback if layer is empty)."""
    if layer_metric.empty:
        return _FALLBACK_DISTANCE_M

    distance = float(layer_metric.distance(point_metric).min())
    if math.isfinite(distance):
        return distance
    return _FALLBACK_DISTANCE_M


def _flag_within(distance_m: float, threshold_m: float) -> int:
    """Return binary proximity flag (1 if within threshold, else 0)."""
    return int(distance_m <= threshold_m)


def _inside_polygon(union_geom, point_wgs84: Point) -> bool:
    """Check whether point is inside (or on boundary of) a layer geometry."""
    if union_geom is None:
        return False
    return bool(union_geom.covers(point_wgs84))


def _proximity_component(distance_m: float, radius_m: float = 300.0) -> float:
    """
    Convert a distance into a normalized proximity value in [0,1].
    1.0 at distance 0, linearly decays to 0 at radius_m and beyond.
    """
    if distance_m >= radius_m:
        return 0.0
    return max(0.0, (radius_m - distance_m) / radius_m)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def extract_features(latitude: float, longitude: float) -> Dict[str, float | int | bool]:
    """
    Extract geospatial ML features from latitude/longitude.

    Returns a single dictionary containing distances, binary threshold flags,
    aggregate overlap metrics, and polygon containment booleans.
    """
    point_wgs84 = _point_from_lat_lon(latitude, longitude)
    point_metric = _to_metric_point(point_wgs84)

    # Distances (meters)
    dist_water = _distance_m(WATER_GDF_METRIC, point_metric)
    dist_forest = _distance_m(FOREST_GDF_METRIC, point_metric)
    dist_restricted = _distance_m(RESTRICTED_GDF_METRIC, point_metric)

    # Binary threshold flags
    within_water_50 = _flag_within(dist_water, 50.0)
    within_water_150 = _flag_within(dist_water, 150.0)
    within_forest_100 = _flag_within(dist_forest, 100.0)
    within_forest_300 = _flag_within(dist_forest, 300.0)
    within_restricted_50 = _flag_within(dist_restricted, 50.0)
    within_restricted_150 = _flag_within(dist_restricted, 150.0)

    # Aggregate proximity metrics
    nearby_sensitive_layer_count = int(
        (dist_water <= 300.0)
        + (dist_forest <= 300.0)
        + (dist_restricted <= 300.0)
    )

    min_distance_all = min(dist_water, dist_forest, dist_restricted)

    # Weighted overlap score: stronger when multiple sensitive layers are nearby.
    base_overlap = (
        _proximity_component(dist_water)
        + _proximity_component(dist_forest)
        + _proximity_component(dist_restricted)
    )
    multi_layer_bonus = 0.5 * max(0, nearby_sensitive_layer_count - 1)
    weighted_overlap_score = round(base_overlap + multi_layer_bonus, 6)

    # Containment booleans (point in polygon)
    inside_water_polygon = _inside_polygon(WATER_UNION_WGS84, point_wgs84)
    inside_forest_polygon = _inside_polygon(FOREST_UNION_WGS84, point_wgs84)
    inside_restricted_polygon = _inside_polygon(RESTRICTED_UNION_WGS84, point_wgs84)

    return {
        "dist_water": round(dist_water, 6),
        "dist_forest": round(dist_forest, 6),
        "dist_restricted": round(dist_restricted, 6),
        "within_water_50": within_water_50,
        "within_water_150": within_water_150,
        "within_forest_100": within_forest_100,
        "within_forest_300": within_forest_300,
        "within_restricted_50": within_restricted_50,
        "within_restricted_150": within_restricted_150,
        "nearby_sensitive_layer_count": nearby_sensitive_layer_count,
        "min_distance_all": round(min_distance_all, 6),
        "weighted_overlap_score": weighted_overlap_score,
        "inside_water_polygon": inside_water_polygon,
        "inside_forest_polygon": inside_forest_polygon,
        "inside_restricted_polygon": inside_restricted_polygon,
    }


__all__ = ["extract_features"]
