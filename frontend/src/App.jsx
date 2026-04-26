import { useEffect, useState } from "react";
import InputPanel from "./components/InputPanel";
import MapView from "./components/MapView";

const BACKEND_BASE_URL =
  import.meta.env.VITE_BACKEND_BASE_URL ?? "http://localhost:8000";
const ANALYZE_API_URL = `${BACKEND_BASE_URL}/analyze`;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatDistanceLabel(key) {
  return key
    .replace(/_distance_m$/i, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function buildRiskDetailsHtml(details) {
  if (!details) {
    return "";
  }

  if (!details.riskLevel) {
    return `
      <section class="risk-details-card">
        <h3 class="risk-details-title">Selected Location</h3>
        <p class="risk-details-line">Checking risk...</p>
      </section>
    `;
  }

  const distances = details.distances && typeof details.distances === "object"
    ? Object.entries(details.distances)
    : [];

  const distancesHtml =
    distances.length > 0
      ? `<ul class="risk-details-list">${distances
          .map(
            ([key, value]) =>
              `<li><strong>${escapeHtml(formatDistanceLabel(key))}:</strong> ${escapeHtml(value)} m</li>`,
          )
          .join("")}</ul>`
      : `<p class="risk-details-line">No distance data available.</p>`;

  return `
    <section class="risk-details-card">
      <h3 class="risk-details-title">Marker Details</h3>
      <p class="risk-details-line"><strong>Risk Level:</strong> ${escapeHtml(details.riskLevel)}</p>
      <p class="risk-details-line">${escapeHtml(details.explanation || "No explanation provided.")}</p>
      ${distancesHtml}
    </section>
  `;
}

async function fetchRiskAnalysis(latitude, longitude) {
  const response = await fetch(ANALYZE_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      latitude,
      longitude,
    }),
  });

  if (!response.ok) {
    throw new Error("Analyze API returned a non-200 response");
  }

  const payload = await response.json();
  return {
    riskLevel:
      typeof payload.risk_level === "string" && payload.risk_level.trim()
        ? payload.risk_level.toUpperCase()
        : "UNKNOWN",
    explanation:
      typeof payload.explanation === "string" && payload.explanation.trim()
        ? payload.explanation
        : "No explanation provided.",
    distances:
      payload.distances && typeof payload.distances === "object"
        ? payload.distances
        : {},
  };
}

function App() {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [position, setPosition] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [status, setStatus] = useState("Enter coordinates and click Check Risk.");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const setSelectedLocation = (lat, lon) => {
    setLatitude(lat);
    setLongitude(lon);
    setPosition({ lat, lon });
  };

  const renderRiskDetails = (details) => {
    const container = document.getElementById("risk-details");
    if (!container) {
      return;
    }
    container.innerHTML = buildRiskDetailsHtml(details);
  };

  const handleLatitudeChange = (value) => {
    setLatitude(value);
  };

  const handleLongitudeChange = (value) => {
    setLongitude(value);
  };

  const handleMapClick = (nextLatitude, nextLongitude) => {
    setSelectedLocation(nextLatitude, nextLongitude);
    setStatus("Location selected on map. Click Check Risk.");
    if (!isSidebarOpen) setIsSidebarOpen(true);
  };

  const handleMarkerClick = (nextLatitude, nextLongitude) => {
    setSelectedLocation(nextLatitude, nextLongitude);
    setStatus("Marker selected. Click Check Risk.");
    if (!isSidebarOpen) setIsSidebarOpen(true);
  };

  const handleCheckRisk = async () => {
    const targetLat = Number(latitude);
    const targetLon = Number(longitude);

    if (
      !Number.isFinite(targetLat) ||
      !Number.isFinite(targetLon) ||
      targetLat < -90 ||
      targetLat > 90 ||
      targetLon < -180 ||
      targetLon > 180
    ) {
      setStatus("Please enter valid latitude and longitude values.");
      return;
    }

    setSelectedLocation(targetLat, targetLon);
    setStatus("Checking risk level...");
    setAnalysis(null);
    renderRiskDetails({
      riskLevel: null,
      explanation: "Checking risk...",
      distances: {},
    });

    try {
      const nextAnalysis = await fetchRiskAnalysis(targetLat, targetLon);
      setAnalysis(nextAnalysis);
      setStatus(`Risk check complete: ${nextAnalysis.riskLevel}`);
      renderRiskDetails(nextAnalysis);
    } catch (error) {
      const failedAnalysis = {
        riskLevel: "ERROR",
        explanation: "Unable to fetch risk from backend.",
        distances: {},
      };
      setAnalysis(failedAnalysis);
      setStatus("Request failed. Please make sure backend is running.");
      renderRiskDetails(failedAnalysis);
    }
  };

  useEffect(() => {
    const handleDocumentKeyDown = (event) => {
      if (event.key !== "Enter" || event.repeat) {
        return;
      }

      const target = event.target;
      if (target instanceof HTMLElement) {
        const tagName = target.tagName;
        if (
          tagName === "INPUT" ||
          tagName === "TEXTAREA" ||
          target.isContentEditable
        ) {
          return;
        }
      }

      const checkRiskButton = document.getElementById("check-risk-btn");
      if (!(checkRiskButton instanceof HTMLButtonElement)) {
        return;
      }

      event.preventDefault();
      checkRiskButton.click();
    };

    document.addEventListener("keydown", handleDocumentKeyDown);
    return () => {
      document.removeEventListener("keydown", handleDocumentKeyDown);
    };
  }, []);

  return (
    <main className="app-shell">
      <div className={`sidebar ${isSidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <h1 className="title">GIS Risk Checker</h1>
          <button 
            className="toggle-btn" 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            title="Toggle Sidebar"
          >
            {isSidebarOpen ? "◀" : "▶"}
          </button>
        </div>
        
        <div className="sidebar-content">
          <InputPanel
            latitude={latitude}
            longitude={longitude}
            onLatitudeChange={handleLatitudeChange}
            onLongitudeChange={handleLongitudeChange}
            onCheckRisk={handleCheckRisk}
          />
          <p className="status">{status}</p>
          <div id="risk-details" className="risk-details" />
        </div>
      </div>
      
      <div className="map-container">
        <MapView
          position={position}
          analysis={analysis}
          onMapClick={handleMapClick}
          onMarkerClick={handleMarkerClick}
        />
      </div>
    </main>
  );
}

export default App;
