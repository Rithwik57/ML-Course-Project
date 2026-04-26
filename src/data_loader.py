"""Utilities for loading GIS GeoJSON datasets."""

from pathlib import Path

import geopandas as gpd
from geopandas import GeoDataFrame


DATA_DIR = Path("data")


def _load_geojson(filename: str, dataset_name: str) -> GeoDataFrame:
    """Load a GeoJSON file from the data directory."""
    file_path = DATA_DIR / filename

    # Validate input file path early so the error is explicit and actionable.
    if not file_path.exists():
        message = f"{dataset_name} file not found: {file_path}"
        print(message)
        raise FileNotFoundError(message)

    # Read the GeoJSON file into a GeoDataFrame.
    gdf = gpd.read_file(file_path)

    # Print the number of loaded features for quick visibility.
    print(f"{dataset_name}: loaded {len(gdf)} features")
    return gdf


def load_lakes() -> GeoDataFrame:
    """Load lakes dataset from data/lakes.geojson."""
    return _load_geojson(filename="lakes.geojson", dataset_name="lakes")


def load_forest() -> GeoDataFrame:
    """Load forest dataset from data/forest.geojson."""
    return _load_geojson(filename="forest.geojson", dataset_name="forest")


def load_govt_land() -> GeoDataFrame:
    """Load government land dataset from data/govt_land.geojson."""
    return _load_geojson(filename="govt_land.geojson", dataset_name="govt_land")
