import { useEffect, useState } from "react";
import InputPanel from "./components/InputPanel";
import MapView from "./components/MapView";

const BACKEND_BASE_URL =
  import.meta.env.VITE_BACKEND_BASE_URL ?? "http://localhost:8000";
const ANALYZE_API_URL = `${BACKEND_BASE_URL}/analyze`;

function formatDistanceLabel(key) {
  return key
    .replace(/_m$/i, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function parseAnalyzePayload(payload) {
  const riskPredicted =
    typeof payload?.risk?.predicted === "string" && payload.risk.predicted
      ? payload.risk.predicted.toUpperCase()
      : typeof payload?.risk_level === "string" && payload.risk_level
        ? payload.risk_level.toUpperCase()
        : "UNKNOWN";

  const riskFinal =
    typeof payload?.risk?.final === "string" && payload.risk.final
      ? payload.risk.final.toUpperCase()
      : riskPredicted;

  const confidence = Number(payload?.confidence?.predicted_class);
  const confidencePct = Number.isFinite(confidence)
    ? Math.max(0, Math.min(100, confidence * 100))
    : null;

  const topAiReasons = Array.isArray(payload?.top_ai_reasons)
    ? payload.top_ai_reasons
    : [];

  const featureImportance = Array.isArray(payload?.feature_importance?.top_factors)
    ? payload.feature_importance.top_factors
    : topAiReasons;

  return {
    riskFinal,
    riskPredicted,
    confidencePct,
    classProbabilities:
      payload?.confidence?.class_probabilities &&
      typeof payload.confidence.class_probabilities === "object"
        ? payload.confidence.class_probabilities
        : {},
    topAiReasons,
    featureImportance,
    featureImportanceMethod:
      typeof payload?.feature_importance?.method === "string"
        ? payload.feature_importance.method
        : "unknown",
    legalFlags: Array.isArray(payload?.legal_flags) ? payload.legal_flags : [],
    environmentalFlags: Array.isArray(payload?.environmental_flags)
      ? payload.environmental_flags
      : [],
    distances:
      payload?.distances && typeof payload.distances === "object"
        ? payload.distances
        : {},
    explanation:
      typeof payload?.explanation === "string" && payload.explanation.trim()
        ? payload.explanation
        : "No explanation provided.",
  };
}

function riskClassName(level) {
  if (level === "HIGH") return "risk-pill high";
  if (level === "MEDIUM") return "risk-pill medium";
  if (level === "LOW") return "risk-pill low";
  return "risk-pill unknown";
}

function AnalysisPanel({ analysis }) {
  if (!analysis) {
    return (
      <section className="insight-card">
        <h3 className="insight-title">AI Predicted Risk</h3>
        <p className="insight-muted">Select a location and click Check Risk.</p>
      </section>
    );
  }

  const probabilityRows = Object.entries(analysis.classProbabilities);
  const distanceRows = Object.entries(analysis.distances);

  return (
    <section className="insight-card">
      <h3 className="insight-title">AI Predicted Risk</h3>

      <div className="risk-row">
        <span className={riskClassName(analysis.riskFinal)}>{analysis.riskFinal}</span>
        <span className="risk-sub">Model: {analysis.riskPredicted}</span>
      </div>

      <p className="insight-line">
        <strong>Confidence:</strong>{" "}
        {analysis.confidencePct === null ? "N/A" : `${analysis.confidencePct.toFixed(2)}%`}
      </p>

      <p className="insight-line">{analysis.explanation}</p>

      {probabilityRows.length > 0 && (
        <div className="insight-section">
          <h4>Class Probabilities</h4>
          <ul className="plain-list">
            {probabilityRows.map(([label, value]) => (
              <li key={label}>
                <strong>{label}:</strong> {(Number(value) * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="insight-section">
        <h4>Top AI Influencing Factors</h4>
        {analysis.topAiReasons.length === 0 ? (
          <p className="insight-muted">No AI factors available.</p>
        ) : (
          <ul className="plain-list">
            {analysis.topAiReasons.map((item, index) => (
              <li key={`${item.feature ?? "feature"}-${index}`}>
                <strong>{item.reason ?? item.feature}:</strong>{" "}
                {Number(item.percentage ?? 0).toFixed(2)}%
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="insight-section">
        <h4>Feature Importance ({analysis.featureImportanceMethod})</h4>
        {analysis.featureImportance.length === 0 ? (
          <p className="insight-muted">No feature importance data available.</p>
        ) : (
          <div className="importance-list">
            {analysis.featureImportance.map((item, index) => {
              const pct = Math.max(0, Math.min(100, Number(item.percentage ?? 0)));
              return (
                <div className="importance-item" key={`${item.feature ?? "importance"}-${index}`}>
                  <div className="importance-header">
                    <span>{item.feature ?? "feature"}</span>
                    <span>{pct.toFixed(2)}%</span>
                  </div>
                  <div className="importance-bar-track">
                    <div className="importance-bar-fill" style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="insight-section">
        <h4>Legal Flags</h4>
        {analysis.legalFlags.length === 0 ? (
          <p className="insight-muted">None</p>
        ) : (
          <ul className="plain-list">
            {analysis.legalFlags.map((flag, index) => (
              <li key={`legal-${index}`}>{flag}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="insight-section">
        <h4>Environmental Flags</h4>
        {analysis.environmentalFlags.length === 0 ? (
          <p className="insight-muted">None</p>
        ) : (
          <ul className="plain-list">
            {analysis.environmentalFlags.map((flag, index) => (
              <li key={`env-${index}`}>{flag}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="insight-section">
        <h4>Distances</h4>
        {distanceRows.length === 0 ? (
          <p className="insight-muted">No distance data available.</p>
        ) : (
          <ul className="plain-list">
            {distanceRows.map(([key, value]) => (
              <li key={key}>
                <strong>{formatDistanceLabel(key)}:</strong> {Number(value).toFixed(2)} m
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
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
  return parseAnalyzePayload(payload);
}

function App() {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [status, setStatus] = useState("Enter coordinates and click Check Risk.");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const hasLatitudeInput = String(latitude).trim() !== "";
  const hasLongitudeInput = String(longitude).trim() !== "";
  const parsedLatitude = hasLatitudeInput ? Number(latitude) : Number.NaN;
  const parsedLongitude = hasLongitudeInput ? Number(longitude) : Number.NaN;
  const hasValidPosition =
    Number.isFinite(parsedLatitude) &&
    Number.isFinite(parsedLongitude) &&
    parsedLatitude >= -90 &&
    parsedLatitude <= 90 &&
    parsedLongitude >= -180 &&
    parsedLongitude <= 180;

  const position = hasValidPosition
    ? { lat: parsedLatitude, lon: parsedLongitude }
    : null;

  const setSelectedLocation = (lat, lon) => {
    setLatitude(String(lat));
    setLongitude(String(lon));
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
    if (!hasValidPosition || !position) {
      setStatus("Please enter valid latitude and longitude values.");
      return;
    }

    const targetLat = position.lat;
    const targetLon = position.lon;

    setStatus("Checking risk level...");
    setAnalysis(null);

    try {
      const nextAnalysis = await fetchRiskAnalysis(targetLat, targetLon);
      setAnalysis(nextAnalysis);
      setStatus(`Risk check complete: ${nextAnalysis.riskFinal}`);
    } catch (error) {
      setAnalysis({
        riskFinal: "ERROR",
        riskPredicted: "UNKNOWN",
        confidencePct: null,
        classProbabilities: {},
        topAiReasons: [],
        featureImportance: [],
        featureImportanceMethod: "unknown",
        legalFlags: [],
        environmentalFlags: [],
        distances: {},
        explanation: "Unable to fetch risk from backend.",
      });
      setStatus("Request failed. Please make sure backend is running.");
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
          <AnalysisPanel analysis={analysis} />
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
