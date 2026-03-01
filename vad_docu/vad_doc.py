import re
import json
import os
import google.generativeai as genai

# ========== LOAD LEXICONS ==========
def load_lexicon_txt(path: str):
    """
    Loads a .txt lexicon file with lines:
    phrase valence arousal dominance
    Example: "burnt out -0.7 -0.6 -0.5"
    Skips headers or malformed lines.
    """
    lex = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                v, a, d = map(float, parts[-3:])
            except ValueError:
                continue  # skip header
            phrase = " ".join(parts[:-3]).lower()
            lex[phrase] = (v, a, d)
    return lex

# Load your files
MWE_LEXICON = load_lexicon_txt("mwe.txt")
UNI_LEXICON = load_lexicon_txt("unigram.txt")

# ========== MWE + UNIGRAM SCORER ==========
def compute_vad_mwe_unigram(text: str):
    text_l = text.lower()
    tokens = re.findall(r"[a-zA-Z']+", text_l)
    
    matched_terms = []
    vs, as_, ds, total = 0.0, 0.0, 0.0, 0
    
    # --- Step 1: MWEs ---
    for phrase, (v,a,d) in MWE_LEXICON.items():
        if phrase in text_l:
            matched_terms.append(phrase)
            vs += v; as_ += a; ds += d
            total += 1
            # Remove matched phrase tokens from further unigram checking
            for t in phrase.split():
                if t in tokens:
                    tokens.remove(t)

    # --- Step 2: Unigrams ---
    for t in tokens:
        if t in UNI_LEXICON:
            v,a,d = UNI_LEXICON[t]
            matched_terms.append(t)
            vs += v; as_ += a; ds += d
            total += 1
    
    if total > 0:
        vad = {"valence": vs/total, "arousal": as_/total, "dominance": ds/total}
        confidence = len(matched_terms)/len(text_l.split())
        return vad, confidence, {"matched": matched_terms, "method": "MWE+Unigram"}
    else:
        return {"valence":0.0,"arousal":0.0,"dominance":0.0}, 0.0, {"matched":[],"method":"none"}

# ========== LLM SCORER ==========
GEMINI_KEY = "AIzaSyB7gKTnDrD4kcjnGbCI72RQbgaioYYMUh0"
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

def compute_vad_llm(text: str, model="gemini-1.5-flash"):
    if not GEMINI_KEY:
        return {"valence":0,"arousal":0,"dominance":0},0.0,{"method":"LLM-skipped"}

    prompt = f"""
Extract Valence, Arousal, Dominance (VAD) in range [-1,1] for the text.
Return ONLY JSON as:
{{"valence": X, "arousal": Y, "dominance": Z}}

Text: "{text}"
"""
    try:
        resp = genai.GenerativeModel(model).generate_content(prompt)
        raw = resp.text.strip()
        # Strip code fences if present
        raw = raw.replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        vad = {k: float(data[k]) for k in ["valence","arousal","dominance"]}
        return vad, 0.9, {"method":"LLM","raw":raw}
    except Exception as e:
        return {"valence":0,"arousal":0,"dominance":0},0.0,{"method":"LLM-failed","error":str(e)}

# ========== HYBRID MERGE ==========
def compute_vad_hybrid(text: str, weight_mweuni=0.6, weight_llm=0.4):
    vad1, conf1, info1 = compute_vad_mwe_unigram(text)
    vad2, conf2, info2 = compute_vad_llm(text)

    # Weighted merge
    merged = {
        "valence": (vad1["valence"]*weight_mweuni + vad2["valence"]*weight_llm),
        "arousal": (vad1["arousal"]*weight_mweuni + vad2["arousal"]*weight_llm),
        "dominance": (vad1["dominance"]*weight_mweuni + vad2["dominance"]*weight_llm),
    }
    return {
        "text": text,
        "mwe_unigram": (vad1, conf1, info1),
        "llm": (vad2, conf2, info2),
        "merged": merged
    }

# ========== DEMO ==========
if __name__ == "__main__":
    texts = [
        "I am feeling low today.",
        "She is on cloud nine after getting the job!",
        "He looks tired and sad.",
        "I'm completely burnt out with this workload.",
        "They are over the moon with excitement!"
    ]
    for t in texts:
        result = compute_vad_hybrid(t)
        print(f"\nText: {t}")
        print(" MWE+Unigram →", result["mwe_unigram"])
        print(" LLM         →", result["llm"])
        print(" Merged      →", result["merged"])
