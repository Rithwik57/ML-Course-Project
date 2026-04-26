import { useEffect, useMemo, useRef, useState } from "react";
import {
  GeoJSON,
  LayersControl,
  MapContainer,
  Marker,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

const defaultCenter = [12.9716, 77.5946];
const BACKEND_BASE_URL =
  import.meta.env.VITE_BACKEND_BASE_URL ?? "http://localhost:8000";
const WATER_GEOJSON_URL =
  import.meta.env.VITE_WATER_GEOJSON_URL ??
  `${BACKEND_BASE_URL}/data/water_clean.geojson`;
const FOREST_GEOJSON_URL =
  import.meta.env.VITE_FOREST_GEOJSON_URL ??
  `${BACKEND_BASE_URL}/data/forest_clean.geojson`;
const RESTRICTED_GEOJSON_URL =
  import.meta.env.VITE_RESTRICTED_GEOJSON_URL ??
  `${BACKEND_BASE_URL}/data/restricted_clean.geojson`;
const AI_RISK_SURFACE_GEOJSON_URL =
  import.meta.env.VITE_AI_RISK_SURFACE_GEOJSON_URL ??
  `${BACKEND_BASE_URL}/data/karnataka_ai_risk_surface.geojson`;

const locationIcon = L.icon({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

function RecenterMap({ center }) {
  const map = useMap();
  const previousCenterRef = useRef(null);

  useEffect(() => {
    const previousCenter = previousCenterRef.current;
    const hasChanged =
      !previousCenter ||
      previousCenter[0] !== center[0] ||
      previousCenter[1] !== center[1];

    if (!hasChanged) {
      return;
    }

    map.setView(center, 14, { animate: false });

    previousCenterRef.current = center;
  }, [center, map]);

  return null;
}

function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click(event) {
      if (typeof onMapClick === "function") {
        onMapClick(event.latlng.lat, event.latlng.lng);
      }
    },
  });

  return null;
}

function createLayerStyle(color) {
  return {
    color,
    weight: 2,
    fillColor: color,
    fillOpacity: 0.3,
  };
}

function riskColor(level) {
  const normalized = String(level ?? "").toUpperCase();
  if (normalized === "HIGH") return "#c62828";
  if (normalized === "MEDIUM") return "#f9a825";
  if (normalized === "LOW") return "#2e7d32";
  return "#546e7a";
}

async function loadGeoJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load layer: ${url}`);
  }
  return response.json();
}

function MapView({ position, onMapClick, onMarkerClick }) {
  const markerPosition = position ? [position.lat, position.lon] : null;
  const center = markerPosition ?? defaultCenter;
  const [waterLayer, setWaterLayer] = useState(null);
  const [forestLayer, setForestLayer] = useState(null);
  const [restrictedLayer, setRestrictedLayer] = useState(null);
  const [aiRiskSurfaceLayer, setAiRiskSurfaceLayer] = useState(null);

  const markerEventHandlers = useMemo(
    () => ({
      click(event) {
        if (typeof onMarkerClick !== "function") {
          return;
        }

        onMarkerClick(event.latlng.lat, event.latlng.lng);
      },
    }),
    [onMarkerClick],
  );

  useEffect(() => {
    let isMounted = true;

    async function loadLayers() {
      const [waterResult, forestResult, restrictedResult, aiSurfaceResult] = await Promise.allSettled([
        loadGeoJson(WATER_GEOJSON_URL),
        loadGeoJson(FOREST_GEOJSON_URL),
        loadGeoJson(RESTRICTED_GEOJSON_URL),
        loadGeoJson(AI_RISK_SURFACE_GEOJSON_URL),
      ]);

      if (!isMounted) {
        return;
      }

      if (waterResult.status === "fulfilled") {
        setWaterLayer(waterResult.value);
      }

      if (forestResult.status === "fulfilled") {
        setForestLayer(forestResult.value);
      }

      if (restrictedResult.status === "fulfilled") {
        setRestrictedLayer(restrictedResult.value);
      }

      if (aiSurfaceResult.status === "fulfilled") {
        setAiRiskSurfaceLayer(aiSurfaceResult.value);
      }
    }

    loadLayers();

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <section className="map-wrapper">
      <MapContainer center={center} zoom={5} className="map-canvas">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapClickHandler onMapClick={onMapClick} />

        <RecenterMap center={center} />

        <LayersControl position="topright">
          {waterLayer && (
            <LayersControl.Overlay checked name="Water Bodies">
              <GeoJSON data={waterLayer} style={createLayerStyle("#1565c0")} />
            </LayersControl.Overlay>
          )}

          {forestLayer && (
            <LayersControl.Overlay checked name="Forest Areas">
              <GeoJSON data={forestLayer} style={createLayerStyle("#2e7d32")} />
            </LayersControl.Overlay>
          )}

          {restrictedLayer && (
            <LayersControl.Overlay checked name="Restricted Areas">
              <GeoJSON data={restrictedLayer} style={createLayerStyle("#c62828")} />
            </LayersControl.Overlay>
          )}

          {aiRiskSurfaceLayer && (
            <LayersControl.Overlay name="AI Statewide Risk Surface">
              <GeoJSON
                data={aiRiskSurfaceLayer}
                pointToLayer={(feature, latlng) => {
                  const predictedRisk = feature?.properties?.predicted_risk;
                  return L.circleMarker(latlng, {
                    radius: 4,
                    color: riskColor(predictedRisk),
                    fillColor: riskColor(predictedRisk),
                    fillOpacity: 0.55,
                    weight: 1,
                  });
                }}
                onEachFeature={(feature, layer) => {
                  const predictedRisk = feature?.properties?.predicted_risk ?? "UNKNOWN";
                  const confidence = Number(feature?.properties?.confidence ?? 0);
                  layer.bindPopup(
                    `<strong>AI Risk:</strong> ${predictedRisk}<br/><strong>Confidence:</strong> ${(confidence * 100).toFixed(2)}%`
                  );
                }}
              />
            </LayersControl.Overlay>
          )}
        </LayersControl>

        {markerPosition && (
          <Marker
            key={`${position.lat}:${position.lon}`}
            position={markerPosition}
            icon={locationIcon}
            eventHandlers={markerEventHandlers}
          />
        )}
      </MapContainer>
    </section>
  );
}

export default MapView;
