import { useState, useCallback } from "react";
import ChatPanel from "./components/ChatPanel";
import SlotsPanel from "./components/SlotsPanel";
import VadPanel from "./components/VadPanel";
import RecsPanel from "./components/RecsPanel";
import AlphaSlider from "./components/AlphaSlider";

export type Message = { role: "user" | "assistant"; text: string };
export type Slots = Record<string, string>;
export type SlotStatus = Record<string, "known" | "unsure">;
export type Vad = {
  valence?: number;
  arousal?: number;
  dominance?: number;
  context_score?: number;
  effective_alpha?: number;
};
export type Recommendation = {
  place: string;
  area: string;
  city: string;
  activity: string;
  remedies: string[];
  score: number;
  why: string[];
};

const API = import.meta.env.VITE_API_URL ?? "";
const DEFAULT_ALPHA = 0.4;

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", text: "Hi there 🌱 Tell me how you're feeling today." },
  ]);
  const [slots, setSlots] = useState<Slots>({});
  const [slotStatus, setSlotStatus] = useState<SlotStatus>({});
  const [vad, setVad] = useState<Vad>({});
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [alpha, setAlpha] = useState<number>(DEFAULT_ALPHA);

  const sendMessage = useCallback(
    async (text: string) => {
      setMessages((prev) => [...prev, { role: "user", text }]);
      setLoading(true);
      try {
        const res = await fetch(`${API}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, alpha }),
        });
        const data = await res.json();

        const reply =
          data.reply ||
          data.question ||
          "I couldn't process that. Could you try again?";

        setMessages((prev) => [...prev, { role: "assistant", text: reply }]);

        if (data.slots || data.slots_collected) {
          setSlots(data.slots || data.slots_collected || {});
        }
        if (data.slot_status) setSlotStatus(data.slot_status);

        // Update VAD panel on every turn that carries running_vad.
        // Also merge context_score / effective_alpha — they arrive on every
        // response type (clarification, preferences, final) so the DST
        // sidebar stays live throughout the conversation.
        if (data.running_vad) {
          setVad({
            ...data.running_vad,
            context_score:   data.context_score   ?? undefined,
            effective_alpha: data.effective_alpha ?? undefined,
          });
        } else if (data.context_score != null || data.effective_alpha != null) {
          // running_vad unchanged (e.g. preferences-reply turn) but DST
          // meta may have updated — merge into existing vad state
          setVad((prev) => ({
            ...prev,
            context_score:   data.context_score   ?? prev.context_score,
            effective_alpha: data.effective_alpha ?? prev.effective_alpha,
          }));
        }

        if (data.recommendations) setRecs(data.recommendations);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            text: "⚠️ Could not reach the server. Is the backend running?",
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [alpha]
  );

  const handleReset = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/reset`, { method: "POST" });
      const data = await res.json();
      if (data.status === "reset") {
        setMessages([
          { role: "assistant", text: "Hi there 🌱 Tell me how you're feeling today." },
        ]);
        setSlots({});
        setSlotStatus({});
        setVad({});
        setRecs([]);
      }
    } catch {
      setMessages([
        { role: "assistant", text: "Hi there 🌱 Tell me how you're feeling today." },
      ]);
      setSlots({});
      setSlotStatus({});
      setVad({});
      setRecs([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleAlphaChange = useCallback(async (value: number) => {
    setAlpha(value);
    // Sync to backend immediately so it's in effect even before next message
    try {
      await fetch(`${API}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alpha: value }),
      });
    } catch {
      // non-critical — alpha is also sent per-message as fallback
    }
  }, []);

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <span style={styles.logo}>🌿</span>
        <h1 style={styles.title}>FeelWell AI:Emotional Wellness Assistant</h1>
        <button style={styles.resetBtn} onClick={handleReset} disabled={loading}>
          {loading ? "..." : "Reset"}
        </button>
      </header>

      <main style={styles.main}>
        <div style={styles.left}>
          <ChatPanel messages={messages} loading={loading} onSend={sendMessage} />
        </div>
        <div style={styles.right}>
          <AlphaSlider alpha={alpha} onChange={handleAlphaChange} />
          <SlotsPanel slots={slots} slotStatus={slotStatus} />
          <VadPanel vad={vad} />
          <RecsPanel recs={recs} />
        </div>
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    background: "#f5f7fa",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "12px 24px",
    background: "#ffffff",
    borderBottom: "1px solid #e2e8f0",
    boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
  },
  logo: { fontSize: 24 },
  title: {
    fontSize: 18,
    fontWeight: 600,
    color: "#2d3748",
    flex: 1,
  },
  resetBtn: {
    padding: "6px 16px",
    background: "#edf2f7",
    border: "1px solid #cbd5e0",
    borderRadius: 8,
    cursor: "pointer",
    fontSize: 13,
    color: "#4a5568",
    fontWeight: 500,
    transition: "background 0.15s",
  },
  main: {
    display: "flex",
    flex: 1,
    overflow: "hidden",
    gap: 0,
  },
  left: {
    flex: 2,
    display: "flex",
    flexDirection: "column",
    borderRight: "1px solid #e2e8f0",
    minWidth: 0,
  },
  right: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    overflowY: "auto",
    gap: 0,
    minWidth: 280,
    maxWidth: 380,
    background: "#fafbfc",
  },
};
