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
DEFAULT_ALPHA = 0.6  # EMA smoothing factor

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
    ts = datetime.now(timezone.utc).isoformat()

    c.execute("""
    INSERT INTO utterances (text, valence, arousal, dominance, confidence, method, extra_json, ts)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (text, vad["valence"], vad["arousal"], vad["dominance"],
          confidence, method, json.dumps(extra or {}), ts))
    conn.commit(); conn.close()

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
def compute_vad_llm(text: str):
    prompt = f"""
You are a JSON API. 
Extract Valence, Arousal, Dominance (VAD) values in [-1,1] for the given text.  

Reference Emotion Centroids (for context only):
{{
    "happy":    {{"valence": 1.000, "arousal": 0.357, "dominance": 1.000}},
    "shocked":  {{"valence": 0.214, "arousal": 0.477, "dominance": 0.219}},
    "neutral":  {{"valence": 0.048, "arousal": -1.000, "dominance": -1.000}},
    "angry":    {{"valence": 0.000, "arousal": 1.000, "dominance": 0.582}},
    "lonely":   {{"valence": 0.145, "arousal": -0.751, "dominance": -0.796}},
    "stressed": {{"valence": 0.045, "arousal": 0.362, "dominance": -0.218}}
}}


⚠️ IMPORTANT: Return ONLY a single valid JSON object. 
Do not include any explanation, code, or extra text.  

Format:
{{
  "valence": <float>,
  "arousal": <float>,
  "dominance": <float>
}}

Text: "{text}"
"""

    try:
        payload = {
            "model": LLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        resp = requests.post(LLAMA_API_URL, json=payload)
        data = resp.json()

        # 🔎 Debug log entire raw response
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
# EMOTION_CENTROIDS = {
#     "anxious": {"valence": -0.6, "arousal": 0.7, "dominance": -0.3},
#     "sad": {"valence": -0.6, "arousal": -0.2, "dominance": -0.4},
#     "lonely": {"valence": -0.4, "arousal": 0.2, "dominance": -0.2},
#     "bored": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
#     "happy": {"valence": 0.8, "arousal": 0.4, "dominance": 0.5},
#     "angry": {"valence": -0.7, "arousal": 0.8, "dominance": 0.4},
#     "stressed": {"valence": -0.3, "arousal": 0.3, "dominance": -0.1},
#}

EMOTION_CENTROIDS = {
    "happy":    {"valence": 1.000, "arousal": 0.357, "dominance": 1.000},
    "shocked":  {"valence": 0.214, "arousal": 0.477, "dominance": 0.219},
    "neutral":  {"valence": 0.048, "arousal": -1.000, "dominance": -1.000},
    "angry":    {"valence": 0.000, "arousal": 1.000, "dominance": 0.582},
    "lonely":   {"valence": 0.145, "arousal": -0.751, "dominance": -0.796},
    "stressed": {"valence": 0.045, "arousal": 0.362, "dominance": -0.218}
}




def nearest_emotion(vad: Dict[str,float]) -> Tuple[str,float]:
    EMOTION_CENTROIDS = {
    "happy":    {"valence": 1.000, "arousal": 0.357, "dominance": 1.000},
    "shocked":  {"valence": 0.214, "arousal": 0.477, "dominance": 0.219},
    "neutral":  {"valence": 0.048, "arousal": -1.000, "dominance": -1.000},
    "angry":    {"valence": 0.000, "arousal": 1.000, "dominance": 0.582},
    "lonely":   {"valence": 0.145, "arousal": -0.751, "dominance": -0.796},
    "stressed": {"valence": 0.045, "arousal": 0.362, "dominance": -0.218}
}
    best, bestd = None, 1e9
    for emo, c in EMOTION_CENTROIDS.items():
        d = math.sqrt(sum((vad[k]-c[k])**2 for k in ["valence","arousal","dominance"]))
        if d < bestd: bestd, best = d, emo
    return best, max(0.0, 1 - bestd/3.0)

# ---------- EMA ----------
def ema_update(prev: Optional[Dict[str,float]], curr: Dict[str,float], alpha=DEFAULT_ALPHA) -> Dict[str,float]:
    if prev is None: return curr.copy()
    return {k: alpha*curr[k] + (1-alpha)*prev[k] for k in ["valence","arousal","dominance"]}

# ---------- PROCESS ----------
def process_utterance(text: str, alpha=DEFAULT_ALPHA, maintain_state=True):
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
