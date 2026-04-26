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

const defaultCenter = [20.5937, 78.9629];
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

    if (previousCenter) {
      map.flyTo(center, 14);
    } else {
      map.setView(center, 14);
    }

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

async function loadGeoJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load layer: ${url}`);
  }
  return response.json();
}

function MapView({ position, analysis, onMapClick, onMarkerClick }) {
  const center = position ? [position.lat, position.lon] : defaultCenter;
  const [waterLayer, setWaterLayer] = useState(null);
  const [forestLayer, setForestLayer] = useState(null);
  const [restrictedLayer, setRestrictedLayer] = useState(null);

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
      const [waterResult, forestResult, restrictedResult] = await Promise.allSettled([
        loadGeoJson(WATER_GEOJSON_URL),
        loadGeoJson(FOREST_GEOJSON_URL),
        loadGeoJson(RESTRICTED_GEOJSON_URL),
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
        </LayersControl>

        {position && (
          <Marker
            position={center}
            icon={locationIcon}
            eventHandlers={markerEventHandlers}
          />
        )}
      </MapContainer>
    </section>
  );
}

export default MapView;
