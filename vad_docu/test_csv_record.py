import pandas as pd
from emotional_dst import compute_vad_hybrid, get_running_state, set_running_state, ema_update, nearest_emotion

# --- Custom function: process only with LLM ---
def process_utterance_llm(text: str, alpha=0.3):
    # Run compute_vad_hybrid but ignore MWE and Hybrid
    _, vad2, _, infos = compute_vad_hybrid(text)

    # EMA update with LLM values only
    prev = get_running_state()
    updated = ema_update(prev, vad2, alpha)
    set_running_state(updated)

    # Map to nearest emotion using LLM VAD
    emo, conf = nearest_emotion(vad2)

    return {
        "input": text,
        "llm": vad2,
        "running_state": updated,
        "mapped_emotion": emo,
        "mapped_confidence": conf,
        "method": "LLM-based inference"
    }

# -------------------------
# Load CSV
# -------------------------
df = pd.read_csv("validate_slots.csv")

# --- Reset state for isolated testing ---
set_running_state({"valence": 0.0, "arousal": 0.0, "dominance": 0.0})

# -------------------------
# Process every row
# -------------------------
results = []
for idx, row in df.iterrows():
    text = row["text"]
    out = process_utterance_llm(text)
    results.append(out)

# Convert results into DataFrame for analysis/export
results_df = pd.DataFrame(results)

# -------------------------
# Display a sample of results
# -------------------------
print("Processed", len(results_df), "records.\n")
print(results_df.head(5))

# Optionally save results
results_df.to_csv("llm_only_results.csv", index=False)
