import rasterio
import numpy as np
import os

ELEVATION_TIF = "data/elevation_map.tif"
LANDCOVER_TIF = "data/landcover_map.tif"

def get_elevation_and_slope(lat: float, lon: float) -> dict:
    """
    Reads elevation from a local GeoTIFF and approximates slope.
    If the file is missing, returns safe default/simulated values.
    """
    if not os.path.exists(ELEVATION_TIF):
        # Fallback simulated response if no raster is downloaded yet
        # Higher elevation simulation near certain lats/lons
        sim_ele = abs(lat * lon) % 1000 
        return {
            "elevation_m": round(sim_ele, 2),
            "slope_degrees": round((sim_ele % 45), 2),
            "raster_source": "SIMULATED (Missing .tif)"
        }
    
    try:
        with rasterio.open(ELEVATION_TIF) as src:
            # Get the pixel coordinates from lat/lon
            # Assuming TIF is in EPSG:4326 for simplicity
            row, col = src.index(lon, lat)
            
            # Read a 3x3 window around the pixel to calculate slope
            window = rasterio.windows.Window(col - 1, row - 1, 3, 3)
            data = src.read(1, window=window)
            
            center_elevation = data[1, 1]
            
            # Simple slope approximation (dz/dx) using 30m standard spacing
            cell_size = 30.0 
            dz_dx = (data[1, 2] - data[1, 0]) / (2 * cell_size)
            dz_dy = (data[2, 1] - data[0, 1]) / (2 * cell_size)
            
            slope_percent = np.sqrt(dz_dx**2 + dz_dy**2)
            slope_degrees = np.degrees(np.arctan(slope_percent))
            
            return {
                "elevation_m": round(float(center_elevation), 2),
                "slope_degrees": round(float(slope_degrees), 2),
                "raster_source": "LOCAL_TIF"
            }
    except Exception as e:
        return {"elevation_m": 0.0, "slope_degrees": 0.0, "raster_source": f"ERROR: {str(e)}"}

def get_landcover_class(lat: float, lon: float) -> str:
    """
    Extracts the land cover class (e.g. 10=Tree cover, 40=Cropland) from ESA WorldCover TIF.
    """
    if not os.path.exists(LANDCOVER_TIF):
        return "UNKNOWN (SIMULATED)"
        
    try:
        with rasterio.open(LANDCOVER_TIF) as src:
            row, col = src.index(lon, lat)
            data = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))
            val = data[0, 0]
            
            # ESA WorldCover classes
            classes = {
                10: "Trees",
                20: "Shrubland",
                30: "Grassland",
                40: "Cropland",
                50: "Built-up",
                60: "Bare / sparse vegetation",
                70: "Snow and ice",
                80: "Permanent water bodies",
                90: "Herbaceous wetland",
                95: "Mangroves",
                100: "Moss and lichen"
            }
            return classes.get(int(val), "UNKNOWN")
    except:
        return "UNKNOWN (ERROR)"
