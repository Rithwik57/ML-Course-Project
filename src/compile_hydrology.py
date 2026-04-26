import geopandas as gpd
import pandas as pd
import os
import sys

def process_hydrology():
    print("Beginning Optimized ETL Data Processing...")

    # -----------------------------
    # PROCESS STREAMS / RIVERS
    # -----------------------------
    print("1/2 Loading and Compressing River Basins...")
    basins = [
        "data/River Line of Cauvery Basin.geojson",
        "data/River Line of Godavari Basin.geojson",
        "data/River Line of Krishna Basin.geojson",
        "data/River Line of Pennar Basin.geojson",
        "data/River Line of West flowing rivers from Tadri to Kanyakumari Basin.geojson",
        "data/River Line of West flowing rivers from Tapi to Tadri Basin.geojson"
    ]
    
    stream_frames = []
    for f in basins:
        if os.path.exists(f):
            print(f"  -> Processing {f}... (Straight compression)")
            # Load and fix any topology errors implicitly
            try:
                df = gpd.read_file(f).to_crs(epsg=3857)
                df['geometry'] = df.geometry.make_valid()
                stream_frames.append(df)
            except Exception as e:
                print(f"  -> Warning: failed on {f}: {e}")
        else:
            print(f"  -> Skipping {f}: Not found locally.")

    if stream_frames:
        print("Merging streams and dumping to Parquet...")
        all_streams = pd.concat(stream_frames, ignore_index=True)
        # Drop empty geometries to save massive space
        all_streams = all_streams[~all_streams.is_empty]
        all_streams.to_parquet("data/streams_karnataka.parquet")
        print("  -> Saved data/streams_karnataka.parquet.")

    # -----------------------------
    # PROCESS RESERVOIRS
    # -----------------------------
    print("2/2 Processing Reservoirs and Lakes...")
    reservoir_files = [
        "data/Karnataka Waterbody Boundary 2019.geojson",
        "data/Karnataka Waterbody Point 2019.geojson",
        "data/Minor Irrigation First Waterbody Census for Karnataka.geojson",
        "data/Reservoir Region.geojson",
        "data/DWA Waterbodies Ph1 for Karnataka.geojson"
    ]
    
    res_frames = []
    for f in reservoir_files:
        if os.path.exists(f):
            print(f"  -> Processing {f}... (Straight compression)")
            try:
                df = gpd.read_file(f).to_crs(epsg=3857)
                df['geometry'] = df.geometry.make_valid()
                res_frames.append(df)
            except Exception as e:
                print(f"  -> Warning: failed on {f}: {e}")
            
    if res_frames:
        print("Merging reservoirs and dumping to Parquet...")
        all_res = pd.concat(res_frames, ignore_index=True)
        all_res = all_res[~all_res.is_empty]
        all_res.to_parquet("data/reservoirs_karnataka.parquet")
        print("  -> Saved data/reservoirs_karnataka.parquet.")

    print("ETL Process Complete!")

if __name__ == "__main__":
    process_hydrology()
