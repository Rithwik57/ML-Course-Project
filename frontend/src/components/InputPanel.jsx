function InputPanel({
  latitude,
  longitude,
  onLatitudeChange,
  onLongitudeChange,
  onCheckRisk,
}) {
  return (
    <section className="input-panel">
      <label className="field">
        Latitude
        <input
          type="number"
          step="any"
          value={latitude}
          onChange={(event) => onLatitudeChange(event.target.value)}
          placeholder="e.g. 12.9300"
        />
      </label>

      <label className="field">
        Longitude
        <input
          type="number"
          step="any"
          value={longitude}
          onChange={(event) => onLongitudeChange(event.target.value)}
          placeholder="e.g. 77.6700"
        />
      </label>

      <button id="check-risk-btn" type="button" onClick={onCheckRisk}>
        Check Risk
      </button>
    </section>
  );
}

export default InputPanel;
