import { Component, type ReactNode } from "react";

interface Props { children: ReactNode }
interface State { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div style={styles.wrap}>
          <div style={styles.box}>
            <div style={styles.icon}>⚠️</div>
            <h2 style={styles.title}>Something went wrong</h2>
            <pre style={styles.msg}>{this.state.error.message}</pre>
            <button
              style={styles.btn}
              onClick={() => this.setState({ error: null })}
            >
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    background: "#f5f7fa",
  },
  box: {
    background: "#fff",
    border: "1px solid #e2e8f0",
    borderRadius: 12,
    padding: "32px 40px",
    textAlign: "center",
    maxWidth: 480,
    boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
  },
  icon: { fontSize: 40, marginBottom: 12 },
  title: { fontSize: 18, fontWeight: 600, color: "#2d3748", marginBottom: 10 },
  msg: {
    fontSize: 12,
    color: "#e53e3e",
    background: "#fff5f5",
    border: "1px solid #fed7d7",
    borderRadius: 6,
    padding: "8px 12px",
    textAlign: "left",
    whiteSpace: "pre-wrap",
    marginBottom: 16,
  },
  btn: {
    padding: "8px 20px",
    background: "#667eea",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 600,
  },
};
