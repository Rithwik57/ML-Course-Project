import geopandas as gpd

water = gpd.read_file("data/water_clean.geojson")

# pick first polygon
sample = water.iloc[0]

# get a point INSIDE it
point_inside = sample.geometry.representative_point()

print(point_inside)