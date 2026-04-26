import geopandas as gpd

# -----------------------------
# STEP 1: Load South India data
# -----------------------------
water = gpd.read_file("data/southern-zone-260408-free.shp/gis_osm_water_a_free_1.shp")
landuse = gpd.read_file("data/southern-zone-260408-free.shp/gis_osm_landuse_a_free_1.shp")
protected = gpd.read_file("data/southern-zone-260408-free.shp/gis_osm_protected_areas_a_free_1.shp")

# -----------------------------
# STEP 2: Load Karnataka boundary
# -----------------------------
karnataka = gpd.read_file("data/State/State.shp")

# -----------------------------
# STEP 3: Show original sizes
# -----------------------------
print("===== BEFORE FILTERING =====")
print("Water:", water.shape)
print("Landuse:", landuse.shape)
print("Protected:", protected.shape)

# -----------------------------
# STEP 4: Match coordinate system
# -----------------------------
water = water.to_crs(karnataka.crs)
landuse = landuse.to_crs(karnataka.crs)
protected = protected.to_crs(karnataka.crs)

# -----------------------------
# STEP 5: Filter only Karnataka
# -----------------------------
print("\nFiltering... (this may take some time ⏳)")

water_ka = gpd.overlay(water, karnataka, how="intersection")
landuse_ka = gpd.overlay(landuse, karnataka, how="intersection")
protected_ka = gpd.overlay(protected, karnataka, how="intersection")

# -----------------------------
# STEP 6: Show filtered sizes
# -----------------------------
print("\n===== AFTER FILTERING (KARNATAKA ONLY) =====")
print("Water:", water_ka.shape)
print("Landuse:", landuse_ka.shape)
print("Protected:", protected_ka.shape)

# -----------------------------
# STEP 7: Save filtered data (VERY IMPORTANT)
# -----------------------------
water_ka.to_file("data/water_karnataka.geojson", driver="GeoJSON")
landuse_ka.to_file("data/landuse_karnataka.geojson", driver="GeoJSON")
protected_ka.to_file("data/protected_karnataka.geojson", driver="GeoJSON")

print("\n✅ Filtered datasets saved in /data folder")
