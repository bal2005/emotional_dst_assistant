import { useState } from "react";

interface Props {
  alpha: number;
  onChange: (value: number) => void;
}

const PRESETS = [
  { label: "Reactive", value: 0.9, tip: "Latest message dominates" },
  { label: "Balanced", value: 0.4, tip: "Equal weight to new & history" },
  { label: "Stable",   value: 0.1, tip: "History dominates" },
];

export default function AlphaSlider({ alpha, onChange }: Props) {
  const [showTip, setShowTip] = useState(false);

  const handleSlider = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value));
  };

  // Describe current alpha behaviour
  const behaviour =
    alpha >= 0.7
      ? "Latest message has strong influence"
      : alpha >= 0.35
      ? "Balanced — new and history weighted equally"
      : "History dominates — slow to change";

  return (
    <div style={styles.card}>
      {/* Header row */}
      <div style={styles.heading}>
        <span>⚙️</span>
        <span>EMA Alpha</span>
        <span
          style={styles.infoIcon}
          onMouseEnter={() => setShowTip(true)}
          onMouseLeave={() => setShowTip(false)}
          aria-label="What is alpha?"
        >
          ?
        </span>
      </div>

      {/* Tooltip */}
      {showTip && (
        <div style={styles.tooltip}>
          <strong>α (alpha)</strong> controls how much the <em>latest</em> message
          shifts the running emotional state (VAD).
          <br /><br />
          <strong>High α (→ 1.0)</strong> — reacts quickly to each new message.<br />
          <strong>Low α (→ 0.0)</strong> — smooths over many messages; slow to change.
          <br /><br />
          Formula: <code>VAD = α × new + (1−α) × previous</code>
        </div>
      )}

      {/* Slider */}
      <div style={styles.sliderRow}>
        <span style={styles.sliderLabel}>0.01</span>
        <input
          type="range"
          min={0.01}
          max={0.99}
          step={0.01}
          value={alpha}
          onChange={handleSlider}
          style={styles.slider}
          aria-label="Alpha value"
        />
        <span style={styles.sliderLabel}>0.99</span>
      </div>

      {/* Current value badge */}
      <div style={styles.valueRow}>
        <span style={styles.valueBadge}>α = {alpha.toFixed(2)}</span>
        <span style={styles.behaviourText}>{behaviour}</span>
      </div>

      {/* Preset buttons */}
      <div style={styles.presets}>
        {PRESETS.map((p) => (
          <button
            key={p.label}
            style={{
              ...styles.presetBtn,
              ...(Math.abs(alpha - p.value) < 0.01 ? styles.presetBtnActive : {}),
            }}
            onClick={() => onChange(p.value)}
            title={p.tip}
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    padding: "14px 18px",
    borderBottom: "1px solid #e2e8f0",
    position: "relative",
  },
  heading: {
    fontSize: 13,
    fontWeight: 600,
    color: "#4a5568",
    marginBottom: 10,
    display: "flex",
    gap: 6,
    alignItems: "center",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  infoIcon: {
    width: 16,
    height: 16,
    borderRadius: "50%",
    background: "#cbd5e0",
    color: "#4a5568",
    fontSize: 10,
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "help",
    userSelect: "none",
    marginLeft: "auto",
  },
  tooltip: {
    position: "absolute",
    top: 44,
    right: 18,
    left: 18,
    background: "#2d3748",
    color: "#e2e8f0",
    fontSize: 11,
    lineHeight: 1.6,
    padding: "10px 12px",
    borderRadius: 8,
    zIndex: 10,
    boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
  },
  sliderRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
  },
  sliderLabel: {
    fontSize: 10,
    color: "#a0aec0",
    fontFamily: "monospace",
    flexShrink: 0,
  },
  slider: {
    flex: 1,
    accentColor: "#667eea",
    cursor: "pointer",
    height: 4,
  },
  valueRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 10,
  },
  valueBadge: {
    background: "#667eea",
    color: "#fff",
    fontSize: 12,
    fontWeight: 700,
    padding: "2px 10px",
    borderRadius: 10,
    fontFamily: "monospace",
    flexShrink: 0,
  },
  behaviourText: {
    fontSize: 11,
    color: "#718096",
    fontStyle: "italic",
  },
  presets: {
    display: "flex",
    gap: 6,
  },
  presetBtn: {
    flex: 1,
    padding: "5px 0",
    fontSize: 11,
    fontWeight: 500,
    background: "#edf2f7",
    border: "1px solid #cbd5e0",
    borderRadius: 6,
    cursor: "pointer",
    color: "#4a5568",
    transition: "all 0.15s",
  },
  presetBtnActive: {
    background: "#667eea",
    color: "#fff",
    border: "1px solid #667eea",
  },
};
