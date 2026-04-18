import json
import asyncio
import requests
from typing import Dict, List
import os
import re
from neo4j import GraphDatabase

# ------------------ Neo4j Config ------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dbpwd@123")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ================== IMPORT UNCHANGED DST CORE ==================
# DO NOT MODIFY emotional_dst.py
from emotional_dst import process_utterance

# ================== LLM CONFIG ==================
LLAMA_API_URL = "http://localhost:1234/v1/chat/completions"
LLAMA_MODEL = "meta-llama-3.1-8b-instruct-hf"

# ================== CONVERSATION STATE ==================
DEFAULT_ORCHESTRATOR_ALPHA = 0.4  # EMA weight for in-memory running VAD

conversation_state: Dict = {
    "history": [],
    "slots": {},                 # Emotion, Activity, Place, Event, Tag, Remedy
    "slot_status": {},           # "known" | "unsure" per slot
    "slots_asked": set(),        # slots already asked once — never ask again
    "running_vad": None,
    "preferences_asked": False,
    "alpha": DEFAULT_ORCHESTRATOR_ALPHA,
}

def reset_conversation_state():
    """Mutates the global conversation_state in-place so all importers see the reset."""
    conversation_state["history"] = []
    conversation_state["slots"] = {}
    conversation_state["slot_status"] = {}
    conversation_state["slots_asked"] = set()
    conversation_state["running_vad"] = None
    conversation_state["preferences_asked"] = False
    # alpha is intentionally preserved across resets

def set_alpha(alpha: float):
    """Update the EMA alpha used for the running VAD. Clamped to [0.01, 0.99]."""
    conversation_state["alpha"] = max(0.01, min(0.99, float(alpha)))

MANDATORY_SLOTS = ["Activity", "Place"]
OPTIONAL_SLOTS = ["Event", "Tag", "Remedy"]

# ================== UNCERTAINTY DETECTION ==================
UNCERTAINTY_PHRASES = {
    "not sure", "no idea", "don't know", "dont know", "idk",
    "maybe", "not really", "no preference", "anything", "whatever",
    "doesn't matter", "doesnt matter", "no clue", "nothing",
    "not certain", "unsure", "i don't mind", "i dont mind",
    "up to you", "you decide", "no specific", "not particular",
}

def is_uncertain(text: str) -> bool:
    """Return True if the user's response signals uncertainty or vagueness."""
    t = text.strip().lower()
    # Short responses (≤ 2 meaningful words)
    words = [w for w in re.findall(r"[a-z]+", t) if len(w) > 1]
    if len(words) <= 2:
        return True
    # Explicit uncertainty phrases
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in t:
            return True
    return False

# ================== SLOT ONTOLOGY ==================
SLOT_ONTOLOGY = {
    "Emotion": "User's emotional state (e.g., stressed, anxious, happy)",
    "Activity": "What the user wants or plans to do",
    "Place": "Location or environment preference",
    "Event": "Specific event or occasion",
    "Tag": "Free-form contextual tag",
    "Remedy": "Action or suggestion that helps emotionally"
}

# ================== UTIL: Safe JSON extraction ==================
def extract_first_json_object(text: str) -> Dict:
    """
    Extract the first {...} JSON object from any LLM output safely.
    Returns {} if not found/invalid.
    """
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end])
    except Exception:
        return {}

# ================== LLM SLOT EXTRACTION ==================
def extract_slots_from_text(user_text: str) -> Dict:
    """
    LLM-only slot extraction.
    Output MUST be JSON.
    """
    prompt = f"""
Extract entities from user input as JSON ONLY.

Slot ontology:
- Emotion
- Activity
- Place
- Event
- Tag
- Remedy

Rules:
- If a slot is not mentioned, set it to null
- Do NOT add explanations
- Do NOT add extra text

Allowed Emotions:
Anxious, Stressed, Sad, Lonely, Bored, Happy, Angry

User input:
"{user_text}"
"""

    try:
        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You extract structured information only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 300,
            "stream": False
        }

        resp = requests.post(LLAMA_API_URL, json=payload, timeout=60)
        raw = resp.json()["choices"][0]["message"]["content"]

        parsed = extract_first_json_object(raw)
        return {k: v for k, v in parsed.items() if v is not None}

    except Exception as e:
        print("❌ Slot extraction failed:", e)
        return {}

# ================== SLOT-FOCUSED CLARIFICATION ==================
def generate_clarification_question(missing_slot: str, emotion: str, history: List[str]) -> str:
    prompt = f"""
You are a Dialogue State Tracking (DST) assistant.

Ask ONE short, direct question to fill the missing information.

Missing info: {missing_slot}
User emotion: {emotion}

Conversation:
{chr(10).join(history[-4:])}

Rules:
- Ask ONLY about the missing information
- One sentence only
- No explanations
"""
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You ask slot-filling questions only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 40,
        "stream": False
    }

    try:
        resp = requests.post(LLAMA_API_URL, json=payload, timeout=60)
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"Could you tell me about the {missing_slot.lower()}?"

# ================== OPTIONAL PREFERENCES QUESTION ==================
def generate_preferences_question(emotion: str) -> str:
    return (
        "Before I suggest something, do you have any other preferences "
        "like quiet/crowd, nearby area, beach/park/temple/library etc.?"
    )

# ------------------ Preference extractor ------------------
def extract_preferences_simple(text: str) -> Dict:
    t = (text or "").lower()
    return {
        "wants_quiet": any(k in t for k in ["quiet", "calm", "peace", "peaceful"]),
        "wants_nature": any(k in t for k in ["park", "garden", "nature", "green", "walk"]),
        "wants_beach": any(k in t for k in ["beach", "marina", "elliot", "besant"]),
        "wants_spiritual": any(k in t for k in ["temple", "church", "mosque", "pray", "prayer"]),
        "wants_library": any(k in t for k in ["library", "reading", "study", "book"]),
        "wants_mall": any(k in t for k in ["mall", "shopping"]),
        "wants_nearby": any(k in t for k in ["near", "nearby", "close", "around", "near me"]),
    }

# ------------------ Normalization helpers ------------------
GENERIC_PLACE_WORDS = {"park", "beach", "temple", "library", "mall", "cafe", "home", "gym"}

def normalize_emotion_for_neo4j(emotion: str) -> str:
    if not emotion:
        return "Stressed"
    e = emotion.strip().lower()
    EMO_MAP = {"neutral": "Stressed", "shocked": "Stressed"}
    e = EMO_MAP.get(e, e)
    return e.title()

def keywordize_activity(text: str) -> str:
    t = (text or "").lower()
    for k in ["walk", "cycling", "cycle", "run", "jog", "meditation", "movie", "museum",
              "temple", "beach", "workout", "gym", "reading", "journal"]:
        if k in t:
            return "cycling" if k == "cycle" else k
    toks = re.findall(r"[a-z]+", t)
    return toks[0] if toks else ""

def normalize_place_hint(place_hint: str) -> Dict:
    p = (place_hint or "").strip().lower()
    if not p:
        return {"area_hint": "", "place_pref": ""}
    if p in GENERIC_PLACE_WORDS:
        return {"area_hint": "", "place_pref": p}
    return {"area_hint": p, "place_pref": ""}

# ================== Neo4j Recommendation (Top 3) ==================
async def neo4j_recommend(slots: Dict) -> List[Dict]:
    context_text = " ".join(conversation_state["history"][-4:]) if conversation_state["history"] else ""
    prefs = extract_preferences_simple(context_text)

    emotion = normalize_emotion_for_neo4j(slots.get("Emotion", "Stressed"))
    activity_kw = keywordize_activity(slots.get("Activity", ""))
    place_norm = normalize_place_hint(slots.get("Place", ""))
    area_hint = place_norm["area_hint"]
    place_pref = place_norm["place_pref"]

    cypher = """
    MATCH (em:Emotion)
    WHERE toLower(em.name) = toLower($emotion)
    MATCH (em)-[:SUGGESTS]->(a:Activity)-[:AT]->(p:Place)
    OPTIONAL MATCH (em)-[:HELPED_BY]->(r:Remedy)
    RETURN em.name AS emotion,
           a.name AS activity,
           a.category AS category,
           p.name AS place,
           p.area AS area,
           p.city AS city,
           collect(DISTINCT r.name) AS remedies
    """

    def run_query():
        with driver.session() as session:
            return session.run(cypher, emotion=emotion).data()

    rows = await asyncio.to_thread(run_query)
    if not rows:
        return []

    scored = []
    for row in rows:
        score = 0.0
        why = []

        score += 3.0
        why.append(f"Matches emotion: {emotion}")

        if activity_kw:
            act = (row.get("activity") or "").lower()
            if activity_kw in act:
                score += 3.0
                why.append(f"Matches activity: {activity_kw}")
            else:
                score += 0.5

        if area_hint:
            area = (row.get("area") or "").lower()
            place = (row.get("place") or "").lower()
            if area_hint in area or area_hint in place:
                score += 2.5
                why.append(f"Near your area: {area_hint}")

        place_name = (row.get("place") or "").lower()

        if place_pref and place_pref in place_name:
            score += 2.0
            why.append(f"Matches your place preference: {place_pref}")

        if prefs["wants_nature"] and any(k in place_name for k in ["park", "poonga", "garden"]):
            score += 1.5
            why.append("Nature-friendly place")

        if prefs["wants_beach"] and "beach" in place_name:
            score += 1.5
            why.append("Beach preference matched")

        if prefs["wants_spiritual"] and any(k in place_name for k in ["temple", "basilica", "church", "mosque"]):
            score += 1.5
            why.append("Spiritual preference matched")

        if prefs["wants_library"] and "library" in place_name:
            score += 1.5
            why.append("Library preference matched")

        if prefs["wants_quiet"] and any(k in place_name for k in ["library", "park", "poonga", "garden"]):
            score += 1.0
            why.append("Likely calm environment")

        category = (row.get("category") or "").lower()
        if category in ["outdoor", "spiritual", "indoor"]:
            score += 0.3

        scored.append({
            "place": row.get("place"),
            "area": row.get("area"),
            "city": row.get("city"),
            "activity": row.get("activity"),
            "remedies": row.get("remedies", []),
            "score": round(score, 2),
            "why": why
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    top = []
    seen = set()
    for s in scored:
        key = (s["place"], s.get("area"), s.get("city"))
        if key in seen:
            continue
        seen.add(key)
        top.append(s)
        if len(top) == 3:
            break

    return top

# ================== LLM: Turn retrieved records into chat response ==================
def llm_recommendation_response(emotion: str, slots: Dict, recs: List[Dict], unsure_slots: List[str] = None) -> str:
    """
    Generates a natural assistant reply using retrieved Neo4j records.
    unsure_slots: list of slot names the user was uncertain about — tone is adjusted accordingly.
    """
    unsure_slots = unsure_slots or []
    recs_compact = []
    for r in recs:
        recs_compact.append({
            "place": r.get("place"),
            "area": r.get("area"),
            "activity": r.get("activity"),
            "remedies": r.get("remedies", [])[:3],
            "why": r.get("why", [])[:3]
        })

    uncertainty_note = ""
    if unsure_slots:
        uncertainty_note = (
            f"\nNote: The user was unsure about: {', '.join(unsure_slots)}. "
            "Do NOT ask about these again. Base suggestions purely on their emotion. "
            "Keep the tone gentle and non-pressuring."
        )

    prompt = f"""
You are an empathetic wellness assistant for Chennai.

User emotion: {emotion}
Collected slots (may be incomplete): {json.dumps(slots, ensure_ascii=False)}
{uncertainty_note}

You MUST base your response ONLY on these 3 retrieved recommendations:
{json.dumps(recs_compact, ensure_ascii=False)}

Write a friendly chat message:
- 1 short validating sentence about their emotion
- Then suggest the 3 options as bullet points:
  • Place (Area) — Activity — 1 quick reason
- Then suggest 1-2 quick remedies (from the remedies list) at the end
- End with ONE soft, optional question (e.g., "Want something closer to you?")
- If the user was unsure about preferences, do NOT ask them to specify — keep it open

If the user gives feedback like he/she is relieved acknowledge that by saying a greeting message and ask for any room to improve their day.

Return ONLY JSON:
{{"reply": "<text>"}}
"""

    try:
        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "Return ONLY JSON. No extra text."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 300,
            "stream": False
        }

        resp = requests.post(LLAMA_API_URL, json=payload, timeout=60)
        raw = resp.json()["choices"][0]["message"]["content"]
        parsed = extract_first_json_object(raw)
        reply = parsed.get("reply")
        if isinstance(reply, str) and reply.strip():
            return reply.strip()

        return deterministic_reply(emotion, recs)

    except Exception as e:
        print("❌ LLM recommendation response failed:", e)
        return deterministic_reply(emotion, recs)

def deterministic_reply(emotion: str, recs: List[Dict]) -> str:
    """
    Fallback if LLM output is messy.
    """
    lines = [f"I hear you — feeling {emotion.lower()} can be heavy. Here are a few options:"]
    for r in recs[:3]:
        place = r.get("place")
        area = r.get("area")
        activity = r.get("activity")
        reason = (r.get("why") or ["Good match for you"])[-1]
        lines.append(f"• {place} ({area}) — {activity} — {reason}")
    remedies = []
    for r in recs:
        remedies.extend(r.get("remedies", []))
    remedies = list(dict.fromkeys(remedies))[:2]
    if remedies:
        lines.append(f"\nQuick reset: try **{remedies[0]}**{(' + ' + remedies[1]) if len(remedies) > 1 else ''}.")
    lines.append("\nDo you want something closer to your area or any specific time (morning/evening)?")
    return "\n".join(lines)

# ================== SLOT COMPLETENESS ==================
def get_missing_slots(slots: Dict) -> List[str]:
    """
    Return mandatory slots that are:
    - not yet filled (not in slots), AND
    - not already marked unsure, AND
    - not already asked once before
    """
    missing = []
    for s in MANDATORY_SLOTS:
        already_asked = s in conversation_state["slots_asked"]
        already_unsure = conversation_state["slot_status"].get(s) == "unsure"
        filled = s in slots and slots[s]
        if not filled and not already_asked and not already_unsure:
            missing.append(s)
    return missing

# ================== STATE RESET AFTER FINAL RECOMMENDATION ==================
def reset_after_recommendation_keep_history():
    conversation_state["slots"] = {}
    conversation_state["slot_status"] = {}
    conversation_state["slots_asked"] = set()
    conversation_state["preferences_asked"] = False

# ================== MAIN TURN HANDLER ==================
async def process_turn(user_text: str, alpha: float = None) -> Dict:
    conversation_state["history"].append(user_text)

    # Use provided alpha or fall back to stored value
    ema_alpha = alpha if alpha is not None else conversation_state["alpha"]
    if alpha is not None:
        conversation_state["alpha"] = max(0.01, min(0.99, float(alpha)))

    # -------- STEP 1: EMOTION + EMA (VAD) --------
    dst_result = process_utterance(user_text, alpha=ema_alpha)

    if conversation_state["running_vad"] is None:
        conversation_state["slots"]["Emotion"] = dst_result["mapped_emotion"]
        conversation_state["running_vad"] = dst_result["merged"]
    else:
        for k in ["valence", "arousal", "dominance"]:
            conversation_state["running_vad"][k] = (
                ema_alpha * dst_result["merged"][k]
                + (1 - ema_alpha) * conversation_state["running_vad"][k]
            )
        conversation_state["slots"]["Emotion"] = dst_result["mapped_emotion"]

    # -------- STEP 2: SLOT EXTRACTION (NO EMOTION OVERRIDE) --------
    extracted_slots = await asyncio.to_thread(extract_slots_from_text, user_text)
    extracted_slots.pop("Emotion", None)

    # -------- STEP 2a: UNCERTAINTY CHECK ON EXTRACTED SLOTS --------
    # If the user's reply is uncertain, mark the slot we last asked about as "unsure"
    # so we never block on it again.
    if is_uncertain(user_text):
        # Mark any mandatory slot we were waiting on as unsure
        for slot in MANDATORY_SLOTS:
            if slot in conversation_state["slots_asked"] and slot not in conversation_state["slots"]:
                conversation_state["slot_status"][slot] = "unsure"
                conversation_state["slots"][slot] = "unsure"   # placeholder so it's "filled"
    else:
        # Normal path — accept extracted values and mark them "known"
        for k, v in extracted_slots.items():
            conversation_state["slots"][k] = v
            conversation_state["slot_status"][k] = "known"

    # -------- STEP 3: MANDATORY SLOT CHECK (ONE ATTEMPT PER SLOT) --------
    missing_slots = get_missing_slots(conversation_state["slots"])
    if missing_slots:
        slot = missing_slots[0]
        # Record that we've asked for this slot — will never ask again
        conversation_state["slots_asked"].add(slot)
        question = await asyncio.to_thread(
            generate_clarification_question,
            slot,
            conversation_state["slots"].get("Emotion", "neutral"),
            conversation_state["history"]
        )
        return {
            "type": "clarification",
            "question": question,
            "slots_collected": conversation_state["slots"],
            "slot_status": dict(conversation_state["slot_status"]),
            "running_vad": conversation_state["running_vad"],
        }

    # -------- STEP 4: ASK OPTIONAL PREFERENCES (ONCE) --------
    # Skip preferences step if both mandatory slots are unsure — go straight to recs
    both_unsure = all(
        conversation_state["slot_status"].get(s) == "unsure" for s in MANDATORY_SLOTS
    )
    if not conversation_state["preferences_asked"] and not both_unsure:
        conversation_state["preferences_asked"] = True
        return {
            "type": "preferences",
            "question": generate_preferences_question(conversation_state["slots"].get("Emotion", "neutral")),
            "slots_collected": conversation_state["slots"],
            "slot_status": dict(conversation_state["slot_status"]),
            "running_vad": conversation_state["running_vad"],
        }

    # -------- STEP 5: Neo4j Recommendations --------
    # Build effective slots — replace "unsure" placeholders with empty string
    # so the recommender falls back to pure emotion-based scoring
    effective_slots = {
        k: ("" if v == "unsure" else v)
        for k, v in conversation_state["slots"].items()
    }
    recommendations = await neo4j_recommend(effective_slots)

    emotion = conversation_state["slots"].get("Emotion", "stressed")

    # -------- STEP 6: LLM Response from retrieved records --------
    # Tell the LLM which slots were unsure so it can be more empathetic
    unsure_slots = [k for k, v in conversation_state["slot_status"].items() if v == "unsure"]
    chat_response = await asyncio.to_thread(
        llm_recommendation_response,
        emotion,
        effective_slots,
        recommendations,
        unsure_slots,
    )

    final_output = {
        "type": "final",
        "emotion": emotion,
        "running_vad": conversation_state["running_vad"],
        "slots": dict(conversation_state["slots"]),
        "slot_status": dict(conversation_state["slot_status"]),
        "recommendations": recommendations,
        "reply": chat_response,
        "alpha": conversation_state["alpha"],
    }

    # -------- STEP 7: RESET FOR NEXT TRACE (KEEP HISTORY) --------
    reset_after_recommendation_keep_history()

    return final_output

# ================== INTERACTIVE LOOP ==================
if __name__ == "__main__":
    reset_conversation_state()

    print("\n🟢 Welcome to the Emotional Wellness Assistant 🌱")
    print("You can talk freely about how you're feeling or what you want to do.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("👋 Take care. I'm here whenever you need.")
            break

        output = asyncio.run(process_turn(user_input))

        if output.get("type") == "final" and output.get("reply"):
            print("\nAssistant:\n" + output["reply"] + "\n")

        print(json.dumps(output, indent=2, ensure_ascii=False))