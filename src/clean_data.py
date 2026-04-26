import geopandas as gpd

# -----------------------------
# STEP 1: Load filtered datasets
# -----------------------------
water = gpd.read_file("data/water_karnataka.geojson")
landuse = gpd.read_file("data/landuse_karnataka.geojson")
protected = gpd.read_file("data/protected_karnataka.geojson")

print("=== ORIGINAL DATA ===")
print("Water:", water.shape)
print("Landuse:", landuse.shape)
print("Protected:", protected.shape)


# -----------------------------
# STEP 2: Remove NULL / EMPTY geometries
# -----------------------------
water = water[water.geometry.notnull()]
landuse = landuse[landuse.geometry.notnull()]
protected = protected[protected.geometry.notnull()]

water = water[~water.is_empty]
landuse = landuse[~landuse.is_empty]
protected = protected[~protected.is_empty]


# -----------------------------
# STEP 3: Fix invalid geometries (VERY IMPORTANT)
# -----------------------------
water["geometry"] = water.buffer(0)
landuse["geometry"] = landuse.buffer(0)
protected["geometry"] = protected.buffer(0)

# Keep only valid ones after fixing
water = water[water.is_valid]
landuse = landuse[landuse.is_valid]
protected = protected[protected.is_valid]


# -----------------------------
# STEP 4: Keep only important classes
# -----------------------------
# WATER → real water bodies
water = water[water["fclass"].isin(["water", "riverbank", "wetland"])]

# LANDUSE → forest only
forest = landuse[landuse["fclass"] == "forest"]

# PROTECTED → keep all
restricted = protected


# -----------------------------
# STEP 5: Simplify geometry (less aggressive)
# -----------------------------
water["geometry"] = water["geometry"].simplify(0.00001)
forest["geometry"] = forest["geometry"].simplify(0.00001)
restricted["geometry"] = restricted["geometry"].simplify(0.00001)


# -----------------------------
# STEP 6: Final cleanup
# -----------------------------
water = water.reset_index(drop=True)
forest = forest.reset_index(drop=True)
restricted = restricted.reset_index(drop=True)


# -----------------------------
# STEP 7: Final check
# -----------------------------
print("\n=== CLEANED DATA ===")
print("Water:", water.shape)
print("Forest:", forest.shape)
print("Restricted:", restricted.shape)


# -----------------------------
# STEP 8: Save clean datasets
# -----------------------------
water.to_file("data/water_clean.geojson", driver="GeoJSON")
forest.to_file("data/forest_clean.geojson", driver="GeoJSON")
restricted.to_file("data/restricted_clean.geojson", driver="GeoJSON")

print("\n✅ Clean datasets saved successfully!")