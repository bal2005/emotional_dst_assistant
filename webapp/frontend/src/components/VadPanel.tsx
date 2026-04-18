import type { Vad } from "../App";

function Bar({
  label,
  value,
  color,
  pctOverride,
}: {
  label: string;
  value?: number;
  color: string;
  pctOverride?: number;
}) {
  // VAD values are in [-1, 1] → map to 0–100%
  // context_score / effective_alpha are already in [0, 1]
  const pct =
    pctOverride !== undefined
      ? Math.round(pctOverride * 100)
      : value !== undefined
      ? Math.round(((value + 1) / 2) * 100)
      : null;

  return (
    <div style={styles.barWrap}>
      <div style={styles.barLabel}>
        <span>{label}</span>
        <span style={styles.barNum}>
          {value !== undefined ? value.toFixed(3) : "—"}
        </span>
      </div>
      <div style={styles.track}>
        <div
          style={{
            ...styles.fill,
            width: pct !== null ? `${pct}%` : "0%",
            background: color,
          }}
        />
      </div>
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value?: number | null }) {
  if (value == null) return null;  // catches both null and undefined
  return (
    <div style={styles.metaRow}>
      <span style={styles.metaLabel}>{label}</span>
      <span style={styles.metaVal}>{value.toFixed(3)}</span>
      <div style={styles.metaTrack}>
        <div
          style={{
            ...styles.metaFill,
            width: `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`,
          }}
        />
      </div>
    </div>
  );
}

export default function VadPanel({ vad }: { vad: Vad }) {
  const hasData = vad.valence != null;
  const ctxScore = vad.context_score ?? undefined;
  const effAlpha = vad.effective_alpha ?? undefined;

  return (
    <div style={styles.card}>
      <div style={styles.heading}>
        <span>📊</span> Running VAD
      </div>

      {!hasData ? (
        <p style={styles.empty}>No VAD data yet.</p>
      ) : (
        <>
          {/* VAD bars */}
          <div style={styles.bars}>
            <Bar label="Valence"   value={vad.valence}   color="#68d391" />
            <Bar label="Arousal"   value={vad.arousal}   color="#f6ad55" />
            <Bar label="Dominance" value={vad.dominance} color="#76e4f7" />
          </div>

          {/* DST meta — only shown when available */}
          {(ctxScore !== undefined || effAlpha !== undefined) && (
            <div style={styles.metaSection}>
              <div style={styles.metaHeading}>DST Context</div>
              <MetaRow label="Context score" value={ctxScore} />
              <MetaRow label="Effective α"   value={effAlpha} />
              {ctxScore !== undefined && (
                <div style={styles.stabilityRow}>
                  <span
                    style={{
                      ...styles.stabilityBadge,
                      background:
                        ctxScore >= 0.7 ? "#c6f6d5" :
                        ctxScore >= 0.35 ? "#fefcbf" : "#fed7d7",
                      color:
                        ctxScore >= 0.7 ? "#276749" :
                        ctxScore >= 0.35 ? "#744210" : "#c53030",
                    }}
                  >
                    {ctxScore >= 0.7
                      ? "🔒 Stable — resisting drift"
                      : ctxScore >= 0.35
                      ? "⚖️ Balanced"
                      : "🌊 Reactive — low history"}
                  </span>
                </div>
              )}
            </div>
          )}
        </>
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
  bars: { display: "flex", flexDirection: "column", gap: 10, marginBottom: 12 },
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
  // DST meta section
  metaSection: {
    borderTop: "1px dashed #e2e8f0",
    paddingTop: 10,
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  metaHeading: {
    fontSize: 10,
    fontWeight: 700,
    color: "#a0aec0",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    marginBottom: 2,
  },
  metaRow: {
    display: "flex",
    alignItems: "center",
    gap: 6,
  },
  metaLabel: {
    fontSize: 11,
    color: "#718096",
    width: 100,
    flexShrink: 0,
  },
  metaVal: {
    fontSize: 11,
    fontFamily: "monospace",
    color: "#4a5568",
    width: 44,
    flexShrink: 0,
  },
  metaTrack: {
    flex: 1,
    height: 5,
    background: "#edf2f7",
    borderRadius: 3,
    overflow: "hidden",
  },
  metaFill: {
    height: "100%",
    background: "#667eea",
    borderRadius: 3,
    transition: "width 0.4s ease",
  },
  stabilityRow: { marginTop: 4 },
  stabilityBadge: {
    fontSize: 10,
    fontWeight: 600,
    padding: "3px 8px",
    borderRadius: 8,
  },
};
