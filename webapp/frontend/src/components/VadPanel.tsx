import type { Vad } from "../App";

function Bar({ label, value }: { label: string; value?: number }) {
  const pct = value !== undefined ? Math.round(((value + 1) / 2) * 100) : null;
  const colors: Record<string, string> = {
    Valence: "#68d391",
    Arousal: "#f6ad55",
    Dominance: "#76e4f7",
  };
  return (
    <div style={styles.barWrap}>
      <div style={styles.barLabel}>
        <span>{label}</span>
        <span style={styles.barNum}>{value !== undefined ? value.toFixed(3) : "—"}</span>
      </div>
      <div style={styles.track}>
        <div
          style={{
            ...styles.fill,
            width: pct !== null ? `${pct}%` : "0%",
            background: colors[label] || "#a0aec0",
          }}
        />
      </div>
    </div>
  );
}

export default function VadPanel({ vad }: { vad: Vad }) {
  const hasData = vad.valence !== undefined;

  return (
    <div style={styles.card}>
      <div style={styles.heading}>
        <span>📊</span> Running VAD
      </div>
      {!hasData ? (
        <p style={styles.empty}>No VAD data yet.</p>
      ) : (
        <div style={styles.bars}>
          <Bar label="Valence" value={vad.valence} />
          <Bar label="Arousal" value={vad.arousal} />
          <Bar label="Dominance" value={vad.dominance} />
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    padding: "16px 18px",
    borderBottom: "1px solid #e2e8f0",
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
  empty: { fontSize: 12, color: "#a0aec0", fontStyle: "italic" },
  bars: { display: "flex", flexDirection: "column", gap: 10 },
  barWrap: {},
  barLabel: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 12,
    color: "#4a5568",
    marginBottom: 4,
    fontWeight: 500,
  },
  barNum: { color: "#718096", fontFamily: "monospace" },
  track: {
    height: 8,
    background: "#edf2f7",
    borderRadius: 4,
    overflow: "hidden",
  },
  fill: {
    height: "100%",
    borderRadius: 4,
    transition: "width 0.4s ease",
  },
};
