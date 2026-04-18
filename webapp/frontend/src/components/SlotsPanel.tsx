import type { Slots, SlotStatus } from "../App";

interface Props {
  slots: Slots;
  slotStatus: SlotStatus;
}

const SLOT_ICONS: Record<string, string> = {
  Emotion: "💭",
  Activity: "🏃",
  Place: "📍",
  Event: "📅",
  Tag: "🏷️",
  Remedy: "💊",
};

const SLOT_COLORS: Record<string, string> = {
  Emotion: "#e9d8fd",
  Activity: "#c6f6d5",
  Place: "#bee3f8",
  Event: "#feebc8",
  Tag: "#fed7d7",
  Remedy: "#fefcbf",
};

export default function SlotsPanel({ slots, slotStatus }: Props) {
  const entries = Object.entries(slots).filter(([, v]) => v);

  return (
    <div style={styles.card}>
      <div style={styles.heading}>
        <span>🗂️</span> Extracted Slots
      </div>
      {entries.length === 0 ? (
        <p style={styles.empty}>No slots collected yet.</p>
      ) : (
        <div style={styles.grid}>
          {entries.map(([key, val]) => {
            const status = slotStatus[key];
            const isUnsure = val === "unsure" || status === "unsure";
            return (
              <div
                key={key}
                style={{
                  ...styles.chip,
                  background: isUnsure ? "#fff5f5" : (SLOT_COLORS[key] || "#edf2f7"),
                  border: isUnsure ? "1px dashed #fc8181" : "1px solid transparent",
                  opacity: isUnsure ? 0.85 : 1,
                }}
              >
                <span style={styles.chipIcon}>{SLOT_ICONS[key] || "•"}</span>
                <div style={{ flex: 1 }}>
                  <div style={styles.chipKey}>{key}</div>
                  <div style={styles.chipVal}>
                    {isUnsure ? <em style={{ color: "#a0aec0" }}>not specified</em> : val}
                  </div>
                </div>
                {/* Status badge */}
                <span
                  style={{
                    ...styles.badge,
                    background: isUnsure ? "#fed7d7" : "#c6f6d5",
                    color: isUnsure ? "#c53030" : "#276749",
                  }}
                >
                  {isUnsure ? "unsure" : "known"}
                </span>
              </div>
            );
          })}
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
  grid: { display: "flex", flexDirection: "column", gap: 6 },
  chip: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "7px 10px",
    borderRadius: 8,
  },
  chipIcon: { fontSize: 16 },
  chipKey: { fontSize: 10, fontWeight: 700, color: "#718096", textTransform: "uppercase" },
  chipVal: { fontSize: 13, color: "#2d3748", fontWeight: 500 },
  badge: {
    fontSize: 9,
    fontWeight: 700,
    padding: "2px 6px",
    borderRadius: 8,
    textTransform: "uppercase",
    letterSpacing: "0.04em",
    flexShrink: 0,
  },
};
