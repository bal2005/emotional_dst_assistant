import { useEffect, useRef, useState } from "react";
import type { Message } from "../App";

interface Props {
  messages: Message[];
  loading: boolean;
  onSend: (text: string) => void;
}

export default function ChatPanel({ messages, loading, onSend }: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = () => {
    const t = input.trim();
    if (!t || loading) return;
    setInput("");
    onSend(t);
  };

  return (
    <div style={styles.wrap}>
      <div style={styles.messages}>
        {messages.map((m, i) => (
          <div key={i} style={m.role === "user" ? styles.userRow : styles.assistantRow}>
            <div style={m.role === "user" ? styles.userBubble : styles.assistantBubble}>
              {m.text}
            </div>
          </div>
        ))}
        {loading && (
          <div style={styles.assistantRow}>
            <div style={{ ...styles.assistantBubble, ...styles.typing }}>
              <span style={styles.dot} />
              <span style={styles.dot} />
              <span style={styles.dot} />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div style={styles.inputRow}>
        <input
          style={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Tell me how you're feeling..."
          disabled={loading}
        />
        <button style={styles.sendBtn} onClick={submit} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    display: "flex",
    flexDirection: "column",
    flex: 1,
    overflow: "hidden",
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    padding: "20px 24px",
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  userRow: { display: "flex", justifyContent: "flex-end" },
  assistantRow: { display: "flex", justifyContent: "flex-start" },
  userBubble: {
    background: "#667eea",
    color: "#fff",
    padding: "10px 14px",
    borderRadius: "18px 18px 4px 18px",
    maxWidth: "72%",
    fontSize: 14,
    lineHeight: 1.5,
    boxShadow: "0 1px 3px rgba(102,126,234,0.3)",
  },
  assistantBubble: {
    background: "#ffffff",
    color: "#2d3748",
    padding: "10px 14px",
    borderRadius: "18px 18px 18px 4px",
    maxWidth: "72%",
    fontSize: 14,
    lineHeight: 1.5,
    border: "1px solid #e2e8f0",
    boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
    whiteSpace: "pre-wrap",
  },
  typing: {
    display: "flex",
    gap: 5,
    alignItems: "center",
    padding: "12px 16px",
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: "#a0aec0",
    display: "inline-block",
    animation: "bounce 1.2s infinite",
  },
  inputRow: {
    display: "flex",
    gap: 10,
    padding: "14px 24px",
    borderTop: "1px solid #e2e8f0",
    background: "#ffffff",
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    border: "1px solid #cbd5e0",
    borderRadius: 10,
    fontSize: 14,
    outline: "none",
    background: "#f7fafc",
    color: "#2d3748",
  },
  sendBtn: {
    padding: "10px 20px",
    background: "#667eea",
    color: "#fff",
    border: "none",
    borderRadius: 10,
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 600,
    opacity: 1,
    transition: "opacity 0.15s",
  },
};
