# ====== Sample Unigram VAD Lexicon ======
# Format: word -> (valence, arousal, dominance)
UNIGRAM_VAD = {
    "happy": (0.9, 0.6, 0.7),
    "sad": (-0.7, -0.4, -0.5),
    "angry": (-0.8, 0.8, 0.6),
    "tired": (-0.4, -0.6, -0.5),
    "relaxed": (0.7, -0.3, 0.5)
}

# ====== Sample MWE VAD Lexicon ======
# Format: phrase -> (valence, arousal, dominance)
MWE_VAD = {
    "down in the dumps": (-0.8, -0.5, -0.6),
    "over the moon": (0.95, 0.7, 0.8),
    "burnt out": (-0.7, -0.6, -0.4),
    "feeling low": (-0.6, -0.3, -0.5),
    "on cloud nine": (0.9, 0.8, 0.7)
}

import re
from typing import Dict, Tuple

def compute_vad(text: str) -> Tuple[Dict[str, float], float]:
    """
    Compute VAD score using unigram + MWE lexicons.
    Returns: VAD dict + confidence
    """
    text_lower = text.lower()
    vad_scores = []
    matched_tokens = []

    # Step 1: Check MWEs first (they override unigram matches)
    for phrase, (v, a, d) in MWE_VAD.items():
        if phrase in text_lower:
            vad_scores.append((v, a, d))
            matched_tokens.append(phrase)
            # Remove the phrase to avoid double-counting
            text_lower = text_lower.replace(phrase, "")

    # Step 2: Check remaining unigrams
    tokens = re.findall(r"[a-z']+", text_lower)
    for tok in tokens:
        if tok in UNIGRAM_VAD:
            vad_scores.append(UNIGRAM_VAD[tok])
            matched_tokens.append(tok)

    if not vad_scores:
        return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}, 0.0

    # Step 3: Average the VAD scores
    val = sum(v for v, _, _ in vad_scores) / len(vad_scores)
    aro = sum(a for _, a, _ in vad_scores) / len(vad_scores)
    dom = sum(d for _, _, d in vad_scores) / len(vad_scores)

    confidence = len(matched_tokens) / (len(tokens) + len(matched_tokens))
    return {"valence": val, "arousal": aro, "dominance": dom}, confidence

examples = [
    "I am feeling low today.",
    "She is on cloud nine after getting the job!",
    "He looks tired and sad.",
    "I'm completely burnt out with this workload.",
    "They are over the moon with excitement!"
]

for ex in examples:
    vad, conf = compute_vad(ex)
    print(f"Text: {ex}")
    print(f"VAD: {vad}, Confidence: {conf:.2f}\n")
