from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from src.spatial_engine import analyze_location
except ModuleNotFoundError:
    from spatial_engine import analyze_location

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount the /data directory so frontend map can load geojson shape overlays
app.mount("/data", StaticFiles(directory="data"), name="data")

class Location(BaseModel):
    latitude: float
    longitude: float

@app.post("/analyze")
def analyze(loc: Location):
    result = analyze_location(loc.latitude, loc.longitude)

    return {
        "risk_level": result["risk_level"],
        "explanation": result["explanation"],
        "distances": result["distances"],
        "topography": result.get("topography", {}),
        "landcover": result.get("landcover_class", "UNKNOWN")
    }
