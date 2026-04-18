"""
emotional_dst.py

Emotional DST system with Hybrid VAD:
- Compute VAD using MWE+Unigram, LLM (Llama), and merged Hybrid
- Store all results in SQLite DB
- Maintain running EMA-smoothed VAD
- Map to nearest emotion
Usage: python emotional_dst.py "I feel really stressed about my exams"
"""

import os, re, json, sqlite3, math, requests
from datetime import datetime, timezone 
from typing import Dict, Tuple, Optional

# ---------- CONFIG ----------
DB_PATH = os.getenv("EMO_DBDIR", "emotional_state.db")
DEFAULT_ALPHA = 0.8  # EMA smoothing factor

# Llama local API
LLAMA_API_URL = "http://localhost:1234/v1/chat/completions"
LLAMA_MODEL = "meta-llama-3.1-8b-instruct-hf"

# ---------- DATABASE ----------
def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS utterances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        valence REAL,
        arousal REAL,
        dominance REAL,
        confidence REAL,
        method TEXT,
        extra_json TEXT,
        ts TEXT
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS running_state (
        key TEXT PRIMARY KEY,
        valence REAL,
        arousal REAL,
        dominance REAL,
        last_updated TEXT
    );
    """)
    conn.commit(); conn.close()

def insert_utterance(text: str, vad: Dict[str, float], confidence: float,
                     method: str, extra: Optional[Dict]=None, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Ensure tables exist (idempotent)
    c.execute("""
    CREATE TABLE IF NOT EXISTS utterances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        valence REAL,
        arousal REAL,
        dominance REAL,
        confidence REAL,
        method TEXT,
        extra_json TEXT,
        ts TEXT
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS running_state (
        key TEXT PRIMARY KEY,
        valence REAL,
        arousal REAL,
        dominance REAL,
        last_updated TEXT
    );
    """)

    ts = datetime.now(timezone.utc).isoformat()
    c.execute("""
    INSERT INTO utterances (text, valence, arousal, dominance, confidence, method, extra_json, ts)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (text, vad["valence"], vad["arousal"], vad["dominance"],
          confidence, method, json.dumps(extra or {}), ts))

    conn.commit()
    conn.close()

def get_running_state(key="user_current", db_path: str = DB_PATH) -> Optional[Dict]:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT valence, arousal, dominance, last_updated FROM running_state WHERE key = ?", (key,))
    row = c.fetchone(); conn.close()
    if row:
        return {"valence": row[0], "arousal": row[1], "dominance": row[2], "last_updated": row[3]}
    return None

def set_running_state(vad: Dict[str, float], key="user_current", db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    ts = datetime.now(timezone.utc).isoformat()

    c.execute("""
    INSERT INTO running_state (key, valence, arousal, dominance, last_updated)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(key) DO UPDATE SET
      valence=excluded.valence,
      arousal=excluded.arousal,
      dominance=excluded.dominance,
      last_updated=excluded.last_updated
    """, (key, vad["valence"], vad["arousal"], vad["dominance"], ts))
    conn.commit(); conn.close()

# ---------- LOAD LEXICONS ----------
def load_lexicon_txt(path: str):
    lex = {}
    if not os.path.exists(path): return lex
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4: continue
            try:
                v,a,d = map(float, parts[-3:])
            except ValueError:
                continue
            phrase = " ".join(parts[:-3]).lower()
            lex[phrase] = (v,a,d)
    return lex

MWE_LEXICON = load_lexicon_txt("vad_docu/mwe.txt")
UNI_LEXICON = load_lexicon_txt("vad_docu/unigram.txt")

#---------- MWE + UNIGRAM ----------
def compute_vad_mwe_unigram(text: str):
    text_l = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", text_l)
    matched, vs, as_, ds, total = [], 0.0, 0.0, 0.0, 0

    # MWEs
    for phrase, (v,a,d) in MWE_LEXICON.items():
        if phrase in text_l:
            matched.append(phrase)
            vs += v; as_ += a; ds += d; total += 1
            for t in phrase.split():
                if t in tokens: 
                    tokens.remove(t)

    # Unigrams
    for t in tokens:
        if t in UNI_LEXICON:
            v,a,d = UNI_LEXICON[t]
            matched.append(t)
            vs += v; as_ += a; ds += d; total += 1

    if total > 0:
        vad = {"valence": vs/total, "arousal": as_/total, "dominance": ds/total}
        conf = len(matched)/max(1,len(text_l.split()))
        return vad, conf, {"matched": matched, "method": "MWE+Unigram"}
    else:
        return {"valence":0,"arousal":0,"dominance":0}, 0.0, {"matched":[],"method":"MWE-none"}

# 
import re

# Common stopwords to ignore
STOPWORDS = set([
    "i","me","my","am","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","a","an","the","and","or",
    "but","because","so","of","in","on","at","to","for","with","by",
    "from","that","this","it","as","can","will","would","should",
    "could","didn","don","wasn","weren","hasn","haven","not"
])

def compute_vad_unigram_weighted(text: str):
    text_l = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", text_l)
    
    matched, vs, as_, ds, total_weight = [], 0.0, 0.0, 0.0, 0.0
    
    for t in tokens:
        if t in STOPWORDS:   # skip neutral words
            continue
        if t in UNI_LEXICON:
            v,a,d = UNI_LEXICON[t]
            # Weight emotional words higher if strongly non-neutral
            weight = 2.0 if abs(v) > 0.4 or abs(a) > 0.4 or abs(d) > 0.4 else 1.0
            matched.append(f"{t}*{weight}")
            vs += v * weight
            as_ += a * weight
            ds += d * weight
            total_weight += weight
    
    if total_weight > 0:
        vad = {
            "valence": vs/total_weight,
            "arousal": as_/total_weight,
            "dominance": ds/total_weight
        }
        conf = len(matched) / max(1, len(tokens))
        return vad, conf, {"matched": matched, "method": "Unigram-weighted"}
    else:
        return {"valence":0,"arousal":0,"dominance":0}, 0.0, {"matched":[],"method":"Unigram-none"}


# ---------- LLM (Llama) ----------
def compute_vad_llm(text: str, context_summary: str = ""):
    """
    Stateless LLM VAD extractor.
    Accepts an optional context_summary (compact string, NOT full history)
    so the LLM can interpret ambiguous inputs more accurately.
    """
    context_block = ""
    if context_summary:
        context_block = f"Context: {context_summary}\n"

    system_msg = (
        "You are a VAD scoring API. "
        "You ONLY output a single valid JSON object with three keys: "
        "valence, arousal, dominance. "
        "All values are floats in [-1.0, 1.0]. "
        "No explanation. No code. No markdown. No extra text. JSON only."
    )

    user_msg = f"""{context_block}Score the emotional VAD for this text.

Emotion centroids (target VAD values for each emotion):
- happy    → valence  0.60, arousal  0.30, dominance  0.50
- angry    → valence -0.70, arousal  0.80, dominance  0.40
- anxious  → valence -0.55, arousal  0.65, dominance -0.40
- stressed → valence -0.40, arousal  0.50, dominance -0.30
- shocked  → valence  0.05, arousal  0.75, dominance -0.25
- sad      → valence -0.60, arousal -0.30, dominance -0.40
- lonely   → valence -0.50, arousal -0.40, dominance -0.50
- bored    → valence -0.20, arousal -0.60, dominance -0.20
- neutral  → valence  0.05, arousal -0.40, dominance -0.30

Score the text toward the nearest centroid. Interpolate for mixed emotions.

Text: "{text}"

Reply with ONLY this JSON (no other text):
{{"valence": <float>, "arousal": <float>, "dominance": <float>}}"""

    try:
        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 60,
            "stream": False,
        }
        resp = requests.post(LLAMA_API_URL, json=payload)
        data = resp.json()

        raw = data["choices"][0]["message"]["content"].strip()
        print("🔎 LLM Raw Output:", raw)

        # Parse JSON substring safely
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError(f"Invalid JSON response: {raw}")

        json_text = raw[start:end]
        parsed = json.loads(json_text)

        vad = {k: float(parsed[k]) for k in ["valence", "arousal", "dominance"]}
        return vad, 0.9, {"method": "LLM", "raw": raw}

    except Exception as e:
        print("❌ LLM Error:", str(e))  # 🔎 Debug log error reason
        return {"valence": 0, "arousal": 0, "dominance": 0}, 0.0, {
            "method": "LLM-failed",
            "error": str(e)
        }


# ---------- HYBRID ----------
def compute_vad_hybrid(text: str, w_mweuni=0.3, w_llm=0.7):
    vad1, conf1, info1 = compute_vad_mwe_unigram(text)
    vad2, conf2, info2 = compute_vad_llm(text)
    merged = {
        "valence": vad1["valence"]*w_mweuni + vad2["valence"]*w_llm,
        "arousal": vad1["arousal"]*w_mweuni + vad2["arousal"]*w_llm,
        "dominance": vad1["dominance"]*w_mweuni + vad2["dominance"]*w_llm,
    }
    return vad1, vad2, merged, {"mwe_info": info1, "llm_info": info2}

# ---------- EMOTION MAPPING ----------
# Centroids are calibrated so that realistic LLM VAD outputs (not extreme ±1)
# map cleanly to the correct emotion. Each centroid is the target VAD point
# for that emotion in the 3D valence-arousal-dominance space.
#
# Valence  : positive = pleasant,  negative = unpleasant
# Arousal  : positive = activated, negative = calm/deactivated
# Dominance: positive = in control, negative = submissive/overwhelmed

EMOTION_CENTROIDS = {
    # Positive
    "happy":    {"valence":  0.60, "arousal":  0.30, "dominance":  0.50},

    # Negative — high arousal
    "angry":    {"valence": -0.70, "arousal":  0.80, "dominance":  0.40},
    "anxious":  {"valence": -0.55, "arousal":  0.65, "dominance": -0.40},
    "stressed": {"valence": -0.40, "arousal":  0.50, "dominance": -0.30},
    "shocked":  {"valence":  0.05, "arousal":  0.75, "dominance": -0.25},

    # Negative — low arousal
    "sad":      {"valence": -0.60, "arousal": -0.30, "dominance": -0.40},
    "lonely":   {"valence": -0.50, "arousal": -0.40, "dominance": -0.50},
    "bored":    {"valence": -0.20, "arousal": -0.60, "dominance": -0.20},

    # Neutral
    "neutral":  {"valence":  0.05, "arousal": -0.40, "dominance": -0.30},
}


def nearest_emotion(vad: Dict[str, float]) -> Tuple[str, float]:
    """Map a VAD vector to the nearest emotion centroid using Euclidean distance."""
    best, bestd = None, 1e9
    for emo, c in EMOTION_CENTROIDS.items():
        d = math.sqrt(sum((vad[k] - c[k]) ** 2 for k in ["valence", "arousal", "dominance"]))
        if d < bestd:
            bestd, best = d, emo
    return best, max(0.0, 1 - bestd / 3.0)

# ---------- EMA ----------
def ema_update(prev: Optional[Dict[str,float]], curr: Dict[str,float], alpha=DEFAULT_ALPHA) -> Dict[str,float]:
    if prev is None: return curr.copy()
    return {k: alpha*curr[k] + (1-alpha)*prev[k] for k in ["valence","arousal","dominance"]}

# ---------- PROCESS ----------
def process_utterance(text: str, alpha=DEFAULT_ALPHA, maintain_state=True):
    init_db(DB_PATH)
    vad1, vad2, merged, infos = compute_vad_hybrid(text)

    insert_utterance(text, vad1, 0.7, "MWE+Unigram", infos["mwe_info"])
    insert_utterance(text, vad2, 0.9, "LLM", infos["llm_info"])
    insert_utterance(text, merged, 0.85, "Hybrid", {})

    if maintain_state:
        prev = get_running_state()
        updated = ema_update(prev, merged, alpha)
        set_running_state(updated)
    else:
        updated = merged  # just use the merged result directly

    emo, conf = nearest_emotion(merged)
    return {
        "input": text,
        "mwe_unigram": vad1,
        "llm": vad2,
        "merged": merged,
        "running_state": updated,
        "mapped_emotion": emo,
        "mapped_confidence": conf
    }

# ---------- CLI ----------
if __name__ == "__main__":
    import sys
    init_db()
    if len(sys.argv) < 2:
        print("Usage: python emotional_dst.py \"I feel stressed today\"")
        sys.exit(0)

    text = " ".join(sys.argv[1:])
    print("Processing:", text)
    res = process_utterance(text)
    print(json.dumps(res, indent=2))


# =============================================================================
# ██████╗ ███████╗████████╗    ██╗      █████╗ ██╗   ██╗███████╗██████╗
# ██╔══██╗██╔════╝╚══██╔══╝    ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗
# ██║  ██║███████╗   ██║       ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝
# ██║  ██║╚════██║   ██║       ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗
# ██████╔╝███████║   ██║       ███████╗██║  ██║   ██║   ███████╗██║  ██║
# ╚═════╝ ╚══════╝   ╚═╝       ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
#
# Context-Aware Dialogue State Tracking (DST) Layer
# --------------------------------------------------
# Principle:
#   LLM  = stateless signal extractor  (current turn only)
#   DST  = stateful memory + reasoning (across all turns)
#
# Key design decisions:
#   - LLM never receives full chat history (token efficiency)
#   - LLM receives only a compact 3-line context summary
#   - Context score controls how much a new input can shift the state
#   - Strong past emotion resists sudden flips from low-info inputs
# =============================================================================

from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# 1. DialogueState — persists across all turns of a conversation
# ---------------------------------------------------------------------------

@dataclass
class DialogueState:
    """
    Holds the full emotional memory of a conversation.

    Attributes
    ----------
    chat_history  : raw user messages in order
    vad_history   : fused VAD dict per turn  [{valence, arousal, dominance}, ...]
    last_vad      : most recently stored VAD (after context-aware update)
    last_emotion  : emotion label mapped from last_vad
    event         : detected event keyword (e.g. "birthday", "exam", "breakup")
    """
    chat_history:  List[str]             = field(default_factory=list)
    vad_history:   List[Dict[str,float]] = field(default_factory=list)
    last_vad:      Optional[Dict[str,float]] = None
    last_emotion:  str                   = "neutral"
    event:         Optional[str]         = None

    def reset(self):
        """Clear all state (call between independent conversations)."""
        self.chat_history.clear()
        self.vad_history.clear()
        self.last_vad     = None
        self.last_emotion = "neutral"
        self.event        = None


# ---------------------------------------------------------------------------
# 2. context_summary — compact 3-line string fed to the LLM prompt
# ---------------------------------------------------------------------------

# Keywords that indicate a named event in the user's message
_EVENT_KEYWORDS = {
    "birthday", "exam", "interview", "breakup", "wedding", "funeral",
    "promotion", "accident", "party", "anniversary", "graduation",
    "deadline", "presentation", "trip", "vacation", "loss", "grief",
}

def _detect_event(text: str) -> Optional[str]:
    """Return the first event keyword found in text, or None."""
    t = text.lower()
    for kw in _EVENT_KEYWORDS:
        if kw in t:
            return kw
    return None


def context_summary(state: DialogueState) -> str:
    """
    Generate a compact context string for the LLM.
    Uses only the last 3 VAD values — never the raw chat history.

    Called BEFORE the current turn's VAD is stored, so we also include
    the detected event from the current turn (already set in Step 2)
    even when vad_history is empty (Turn 1).

    Example output:
        Previous emotional trend:
        Valence  ~ 0.72
        Emotion  ~ happy
        Event    ~ birthday
    """
    event_str = state.event if state.event else "None"

    # Turn 1: no VAD history yet — still send event context so LLM
    # knows this is a birthday/exam/etc. conversation from the start
    if not state.vad_history:
        if state.event:
            return (
                f"Conversation context:\n"
                f"Event    ~ {event_str}\n"
                f"Note: This is the first message. Interpret emotion accordingly."
            )
        return ""

    # Turn 2+: include average valence trend + last emotion + event
    recent = state.vad_history[-3:]
    avg_valence = sum(v["valence"] for v in recent) / len(recent)

    return (
        f"Previous emotional trend:\n"
        f"Valence  ~ {avg_valence:.2f}\n"
        f"Emotion  ~ {state.last_emotion}\n"
        f"Event    ~ {event_str}"
    )


# ---------------------------------------------------------------------------
# 3. get_context_score — how strongly should history resist the new input?
# ---------------------------------------------------------------------------

def get_context_score(state: DialogueState) -> float:
    """
    Returns a score in [0.0, 1.0] representing emotional stability.

    High score  → history is strong / consistent → resist sudden change
    Low score   → little history or high variance → accept new input freely

    Extra guard: if the established emotion is weakly positive (valence < 0.3),
    cap the context score at 0.4 so the system stays open to correction.
    A "stably wrong" low-valence state should not resist positive signals.
    """
    if len(state.vad_history) < 2:
        return 0.0

    recent   = state.vad_history[-3:]
    valences = [v["valence"] for v in recent]
    mean_v   = sum(valences) / len(valences)
    variance = sum((v - mean_v) ** 2 for v in valences) / len(valences)

    stability = max(0.0, 1.0 - variance / 0.5)

    # If the average valence is weakly positive (< 0.3) and the emotion
    # is not clearly negative, cap stability — don't lock in an uncertain state
    if mean_v < 0.3 and state.last_emotion not in ("angry", "stressed", "lonely"):
        stability = min(stability, 0.4)

    return round(stability, 4)


# ---------------------------------------------------------------------------
# 4. context_aware_vad_update — replaces plain EMA
# ---------------------------------------------------------------------------

def context_aware_vad_update(
    state:       DialogueState,
    fused_vad:   Dict[str, float],
    llm_conf:    float,
    lex_conf:    float,
) -> Dict[str, float]:
    """
    Combine previous state + new fused VAD using context score as a brake.

    Formula (per dimension):
        input_weight   = mean(llm_conf, lex_conf)          # signal quality
        context_score  = get_context_score(state)          # history stability
        resistance     = context_score * (1 - input_weight)
        effective_alpha = input_weight * (1 - resistance)
        new_vad = effective_alpha * fused_vad
                + (1 - effective_alpha) * last_vad

    Behaviour
    ---------
    - Strong past emotion + weak new signal  → small shift  (stability)
    - Weak past emotion   + strong new signal → larger shift (responsiveness)
    - No history at all                       → accept fused_vad directly
    """
    if state.last_vad is None:
        # First turn — no history to blend with
        return fused_vad.copy()

    input_weight  = (llm_conf + lex_conf) / 2.0          # 0–1
    ctx_score     = get_context_score(state)              # 0–1
    resistance    = ctx_score * (1.0 - input_weight)      # 0–1
    eff_alpha     = input_weight * (1.0 - resistance)     # effective blend weight
    # Clamp to a sensible range so we never fully freeze or fully ignore history
    eff_alpha     = max(0.05, min(0.95, eff_alpha))

    updated = {
        k: round(eff_alpha * fused_vad[k] + (1.0 - eff_alpha) * state.last_vad[k], 6)
        for k in ["valence", "arousal", "dominance"]
    }
    return updated


# ---------------------------------------------------------------------------
# 5. process_input — the clean DST pipeline (replaces process_utterance)
# ---------------------------------------------------------------------------

def process_input(text: str, state: DialogueState) -> Dict:
    """
    Full context-aware VAD pipeline.

    Steps
    -----
    1.  Append text to chat history
    2.  Detect & store event keyword (if any)
    3.  Generate compact context summary (3 lines, no raw history)
    4.  Compute LLM VAD  — stateless, receives only current text + summary
    5.  Compute Lexicon VAD (weighted unigram)
    6.  Fuse: confidence-weighted blend of LLM + Lexicon
    7.  Context-aware update (replaces EMA)
    8.  Store updated VAD in state.vad_history and state.last_vad
    9.  Map to nearest emotion label
    10. Persist to SQLite (same DB as process_utterance)
    11. Return structured result dict

    Parameters
    ----------
    text  : current user utterance
    state : DialogueState instance (mutated in-place)

    Returns
    -------
    dict with keys:
        input, llm_vad, lex_vad, fused_vad, context_score,
        effective_alpha (approx), updated_vad,
        mapped_emotion, mapped_confidence,
        context_summary_used, event
    """
    init_db(DB_PATH)

    # ── Step 1: store raw text ──────────────────────────────────────────────
    state.chat_history.append(text)

    # ── Step 2: event detection — MUST happen before context_summary ────────
    # so the event is included in the summary even on Turn 1
    detected_event = _detect_event(text)
    if detected_event:
        state.event = detected_event

    # ── Step 3: context summary (compact, no raw history) ───────────────────
    # On Turn 1: vad_history is empty but event is already set above,
    # so the summary will include the event keyword for the LLM.
    ctx_summary = context_summary(state)

    # ── Step 4: LLM VAD — stateless, gets only current text + summary ───────
    llm_vad, llm_conf, llm_info = compute_vad_llm(text, context_summary=ctx_summary)

    # ── Step 5: Lexicon VAD ──────────────────────────────────────────────────
    lex_vad, lex_conf, lex_info = compute_vad_unigram_weighted(text)

    # ── Step 6: Confidence-weighted fusion ───────────────────────────────────
    total_conf = llm_conf + lex_conf
    if total_conf > 0:
        w_llm = llm_conf / total_conf
        w_lex = lex_conf / total_conf
    else:
        w_llm, w_lex = 0.7, 0.3   # fallback weights

    fused_vad = {
        k: round(w_llm * llm_vad[k] + w_lex * lex_vad[k], 6)
        for k in ["valence", "arousal", "dominance"]
    }

    # ── Step 7: Context-aware update ─────────────────────────────────────────
    # Compute context score BEFORE appending so it reflects history up to
    # the previous turn (correct — current turn hasn't been stored yet)
    ctx_score   = get_context_score(state)
    updated_vad = context_aware_vad_update(state, fused_vad, llm_conf, lex_conf)

    # Approximate effective_alpha for transparency in the return dict
    input_weight = (llm_conf + lex_conf) / 2.0
    resistance   = ctx_score * (1.0 - input_weight)
    eff_alpha    = max(0.05, min(0.95, input_weight * (1.0 - resistance)))

    # ── Step 8: Store in state ────────────────────────────────────────────────
    state.vad_history.append(updated_vad)
    state.last_vad = updated_vad

    # ── Step 8b: Recompute context score AFTER storing so the returned value
    # reflects the current turn being included — this is what the frontend shows
    ctx_score_after = get_context_score(state)

    # ── Step 9: Emotion mapping ───────────────────────────────────────────────
    emotion, conf = nearest_emotion(updated_vad)
    state.last_emotion = emotion

    # ── Step 10: Persist to SQLite ────────────────────────────────────────────
    insert_utterance(text, llm_vad,    llm_conf, "DST-LLM",    llm_info)
    insert_utterance(text, lex_vad,    lex_conf, "DST-Lexicon", lex_info)
    insert_utterance(text, updated_vad, conf,    "DST-Final",   {
        "context_score": ctx_score,
        "effective_alpha": round(eff_alpha, 4),
        "event": state.event,
    })

    # ── Step 11: Return ───────────────────────────────────────────────────────
    return {
        "input":                text,
        "llm_vad":              llm_vad,
        "lex_vad":              lex_vad,
        "fused_vad":            fused_vad,
        "context_score":        ctx_score_after,   # post-append — reflects current turn
        "effective_alpha":      round(eff_alpha, 4),
        "updated_vad":          updated_vad,
        "mapped_emotion":       emotion,
        "mapped_confidence":    conf,
        "merged":               updated_vad,
        "context_summary_used": ctx_summary,
        "event":                state.event,
    }


# ---------------------------------------------------------------------------
# CLI — quick test of the DST pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    init_db()
    if len(sys.argv) < 2:
        print("Usage: python emotional_dst.py \"I feel stressed today\"")
        sys.exit(0)

    text = " ".join(sys.argv[1:])
    print("Processing:", text)
    res = process_utterance(text)
    print(json.dumps(res, indent=2))
