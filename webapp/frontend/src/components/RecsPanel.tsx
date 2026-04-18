import type { Recommendation } from "../App";

export default function RecsPanel({ recs }: { recs: Recommendation[] }) {
  return (
    <div style={styles.card}>
      <div style={styles.heading}>
        <span>📍</span> Recommendations
      </div>
      {recs.length === 0 ? (
        <p style={styles.empty}>No recommendations yet.</p>
      ) : (
        <div style={styles.list}>
          {recs.slice(0, 3).map((r, i) => (
            <div key={i} style={styles.rec}>
              <div style={styles.recHeader}>
                <span style={styles.badge}>{i + 1}</span>
                <span style={styles.place}>{r.place}</span>
                <span style={styles.score}>★ {r.score?.toFixed(1)}</span>
              </div>
              <div style={styles.area}>
                📍 {r.area}, {r.city}
              </div>
              <div style={styles.activity}>🏃 {r.activity}</div>
              {r.remedies?.length > 0 && (
                <div style={styles.remedies}>
                  {r.remedies.slice(0, 2).map((rem, j) => (
                    <span key={j} style={styles.remedyTag}>
                      {rem}
                    </span>
                  ))}
                </div>
              )}
              {r.why?.length > 0 && (
                <div style={styles.why}>
                  {r.why.slice(0, 2).map((w, j) => (
                    <div key={j} style={styles.whyItem}>
                      ✓ {w}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    padding: "16px 18px",
    flex: 1,
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
  list: { display: "flex", flexDirection: "column", gap: 10 },
  rec: {
    background: "#ffffff",
    border: "1px solid #e2e8f0",
    borderRadius: 10,
    padding: "10px 12px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
  },
  recHeader: {
    display: "flex",
    alignItems: "center",
    gap: 7,
    marginBottom: 4,
  },
  badge: {
    width: 20,
    height: 20,
    borderRadius: "50%",
    background: "#667eea",
    color: "#fff",
    fontSize: 11,
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  place: { fontSize: 13, fontWeight: 600, color: "#2d3748", flex: 1 },
  score: { fontSize: 11, color: "#f6ad55", fontWeight: 600 },
  area: { fontSize: 11, color: "#718096", marginBottom: 3 },
  activity: { fontSize: 12, color: "#4a5568", marginBottom: 5 },
  remedies: { display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 4 },
  remedyTag: {
    background: "#e9d8fd",
    color: "#553c9a",
    fontSize: 10,
    padding: "2px 7px",
    borderRadius: 10,
    fontWeight: 500,
  },
  why: { display: "flex", flexDirection: "column", gap: 2 },
  whyItem: { fontSize: 10, color: "#68d391", fontWeight: 500 },
};
