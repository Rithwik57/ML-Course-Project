import folium
import geopandas as gpd
from spatial_engine import analyze_location
from risk_engine import classify_risk


# -----------------------------
# FIX: Remove problematic columns (like Timestamp)
# -----------------------------
def clean_geojson(gdf):
    return gdf[["geometry"]].copy()


def create_map(lat, lon):

    # -------------------------
    # BASE MAP
    # -------------------------
    m = folium.Map(location=[lat, lon], zoom_start=13, control_scale=True)

    # -------------------------
    # LOAD & CLEAN DATA
    # -------------------------
    water = clean_geojson(gpd.read_file("data/water_clean.geojson"))
    forest = clean_geojson(gpd.read_file("data/forest_clean.geojson"))
    restricted = clean_geojson(gpd.read_file("data/restricted_clean.geojson"))

    # -------------------------
    # ADD LAYERS (WITH STYLE)
    # -------------------------
    folium.GeoJson(
        water,
        name="Water Bodies",
        style_function=lambda x: {
            "fillColor": "blue",
            "color": "blue",
            "weight": 1,
            "fillOpacity": 0.4,
        },
    ).add_to(m)

    folium.GeoJson(
        forest,
        name="Forest Areas",
        style_function=lambda x: {
            "fillColor": "green",
            "color": "green",
            "weight": 1,
            "fillOpacity": 0.4,
        },
    ).add_to(m)

    folium.GeoJson(
        restricted,
        name="Restricted Zones",
        style_function=lambda x: {
            "fillColor": "red",
            "color": "red",
            "weight": 1,
            "fillOpacity": 0.4,
        },
    ).add_to(m)

    # -------------------------
    # ANALYZE LOCATION
    # -------------------------
    result = analyze_location(lat, lon)
    risk = classify_risk(result)

    # -------------------------
    # FLOATING RISK PANEL
    # -------------------------
    risk_panel = f"""
    <div style="
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background: white;
        padding: 12px;
        border-radius: 6px;
        border: 2px solid black;
        font-size: 14px;
    ">
    <b>Risk Level:</b> {risk['risk_level']}<br>
    <b>Reason:</b> {risk['explanation']}
    </div>
    """

    m.get_root().html.add_child(folium.Element(risk_panel))

    # -------------------------
    # COLOR BASED ON RISK
    # -------------------------
    if risk["risk_level"] == "HIGH":
        color = "red"
    elif risk["risk_level"] == "MEDIUM":
        color = "orange"
    else:
        color = "green"

    # -------------------------
    # POPUP CONTENT
    # -------------------------
    popup_html = f"""
    <div style="width:200px">
        <h4>Risk Report</h4>
        <b>Risk Level:</b> {risk['risk_level']}<br>
        <b>Reason:</b> {risk['explanation']}<br><br>

        <b>Distances:</b><br>
        Water: {result['water_distance_m']} m<br>
        Forest: {result['forest_distance_m']} m<br>
        Restricted: {result['restricted_distance_m']} m
    </div>
    """

    # -------------------------
    # ADD MARKER
    # -------------------------
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip="Click for details",
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(m)

    # -------------------------
    # LAYER CONTROL
    # -------------------------
    folium.LayerControl().add_to(m)

    # -------------------------
    # LEGEND (IMPORTANT)
    # -------------------------
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 220px; height: 140px; 
        background-color: white; 
        border:2px solid grey; z-index:9999; font-size:14px;
        padding: 10px;
    ">
    <b>Map Legend</b><br>
    🔵 Water Bodies<br>
    🟢 Forest Areas<br>
    🔴 Restricted Zones<br><br>
    <b>Risk Levels</b><br>
    🔴 High Risk<br>
    🟠 Medium Risk<br>
    🟢 Low Risk
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # -------------------------
    # SAVE MAP
    # -------------------------
    m.save("outputs/map.html")

    print("✅ Map generated: outputs/map.html")