# GIS Safety Analysis Tool

A geospatial risk screening project that classifies a location as `LOW`, `MEDIUM`, or `HIGH` risk based on proximity to:
- Water bodies
- Forest zones
- Restricted/protected areas

The project includes:
- A Python pipeline to filter and clean GIS datasets
- A rule-based spatial risk engine
- A FastAPI backend (`/analyze`)
- A React + Leaflet frontend for interactive map checks
- A Folium map generator for static HTML output

## Project Goals

- Turn raw GIS layers into analysis-ready geospatial data.
- Compute distance-based risk indicators for user-provided coordinates.
- Return an interpretable result: risk level + explanation + distances.
- Provide both CLI and web-based ways to use the system.

## Repository Structure

- `src/`: backend, GIS processing, risk logic, map generation
- `data/`: raw and cleaned geospatial datasets
- `outputs/`: generated artifacts (for example `map.html`)
- `frontend/`: React/Vite map UI
- `model.pkl`: optional ML model used by experimental scripts
- `PROJECT_WORKING_AND_METHODOLOGY.txt`: detailed technical methodology

## Tech Stack

- Python: GeoPandas, Shapely, Folium, FastAPI, Uvicorn, Pandas, scikit-learn
- Frontend: React, Vite, React-Leaflet, Leaflet
- Data format: Shapefile + GeoJSON

## Prerequisites

- Python `3.10+` (3.11/3.12 recommended)
- Node.js `18+` and npm (for frontend)
- Git

## Setup

From repo root:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Frontend setup:

```bash
cd frontend
npm install
cd ..
```

## Data Preparation Workflow

Run these only when you need to regenerate processed datasets.

1. Filter South India layers to Karnataka boundary:

```bash
source venv/bin/activate
python src/filtered_data.py
```

2. Clean geometries and keep risk-relevant classes:

```bash
source venv/bin/activate
python src/clean_data.py
```

This produces:
- `data/water_clean.geojson`
- `data/forest_clean.geojson`
- `data/restricted_clean.geojson`

## Run Options

### Option A: Generate static Folium map (quick check)

```bash
source venv/bin/activate
python src/main.py --latitude 12.9716 --longitude 77.5946
```

Output file:
- `outputs/map.html`

### Option B: Run backend API only

```bash
source venv/bin/activate
python -m uvicorn src.api:app --reload
```

Backend URL:
- `http://localhost:8000`

Health check (example):

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"latitude":12.9716,"longitude":77.5946}'
```

### Option C: Run full stack (backend + frontend)

Terminal 1:

```bash
source venv/bin/activate
python -m uvicorn src.api:app --reload
```

Terminal 2:

```bash
cd frontend
npm run dev
```

Open:
- `http://localhost:5173`

## Risk Logic (Current)

- `HIGH`:
  - Inside water body (distance < 50m), or
  - Inside restricted/protected area (distance < 50m)
- `MEDIUM` (if not already HIGH):
  - Near water (<150m), near forest (<300m), or near restricted area (<100m)
- `LOW`:
  - None of the above

Implementation files:
- `src/spatial_engine.py`
- `src/risk_engine.py`

## Optional ML Path

The repo includes an experimental ML classifier:
- Train: `python src/train_model.py`
- Compare rule-based vs ML: `python src/test.py`

Note:
- API/Frontend currently use the rule-based engine as primary output.

## Collaboration Notes

- Keep generated files (`outputs/`) out of commits unless intentionally sharing results.
- Re-run `src/clean_data.py` after changing raw GIS inputs.
- Update this README when changing setup commands, ports, or API contracts.
- Very large raw OSM shapefiles are intentionally gitignored to keep the repo pushable on GitHub (100MB file limit). Teammates can still run the app directly using the committed cleaned GeoJSON files.

## Troubleshooting

- If `ModuleNotFoundError` appears, confirm virtual environment is activated.
- If frontend cannot reach backend, ensure backend is running on port `8000`.
- If GeoJSON layers fail to render, regenerate cleaned data with `src/clean_data.py`.
