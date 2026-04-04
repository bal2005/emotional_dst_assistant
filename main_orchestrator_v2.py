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
def reset_conversation_state():
    return {
        "history": [],
        "slots": {},                 # Emotion, Activity, Place, Event, Tag, Remedy
        "running_vad": None,
        "preferences_asked": False
    }

conversation_state = reset_conversation_state()

MANDATORY_SLOTS = ["Activity", "Place"]
OPTIONAL_SLOTS = ["Event", "Tag", "Remedy"]

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
def llm_recommendation_response(emotion: str, slots: Dict, recs: List[Dict]) -> str:
    """
    Generates a natural assistant reply using retrieved Neo4j records.
    Returns a plain string (safe even if model adds extra junk).
    """
    recs_compact = []
    for r in recs:
        recs_compact.append({
            "place": r.get("place"),
            "area": r.get("area"),
            "activity": r.get("activity"),
            "remedies": r.get("remedies", [])[:3],
            "why": r.get("why", [])[:3]
        })

    prompt = f"""
You are an empathetic wellness assistant for Chennai.

User emotion: {emotion}
Collected slots (may be incomplete): {json.dumps(slots, ensure_ascii=False)}

You MUST base your response ONLY on these 3 retrieved recommendations:
{json.dumps(recs_compact, ensure_ascii=False)}

Write a friendly chat message:
- 1 short validating sentence about their emotion
- Then suggest the 3 options as bullet points:
  • Place (Area) — Activity — 1 quick reason
- Then suggest 1-2 quick remedies (from the remedies list) at the end
- End with ONE short question to continue the conversation (e.g., "Want something closer to you?")

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
    return [s for s in MANDATORY_SLOTS if s not in slots]

# ================== STATE RESET AFTER FINAL RECOMMENDATION ==================
def reset_after_recommendation_keep_history():
    """
    Option A — Keep history (recommended)

    Keep:
    - history
    - running_vad

    Reset:
    - slots
    - preferences_asked

    Why:
    - better emotion continuity
    - better preference extraction
    - more natural conversation
    """
    conversation_state["slots"] = {}
    conversation_state["preferences_asked"] = False
    # conversation_state["history"] = []   ❌ don't reset
    # conversation_state["running_vad"] = None   ❌ don't reset

# ================== MAIN TURN HANDLER ==================
async def process_turn(user_text: str) -> Dict:
    conversation_state["history"].append(user_text)

    # -------- STEP 1: EMOTION + EMA (VAD) --------
    dst_result = process_utterance(user_text)

    if conversation_state["running_vad"] is None:
        conversation_state["slots"]["Emotion"] = dst_result["mapped_emotion"]
        conversation_state["running_vad"] = dst_result["merged"]
    else:
        alpha = 0.4
        for k in ["valence", "arousal", "dominance"]:
            conversation_state["running_vad"][k] = (
                alpha * dst_result["merged"][k]
                + (1 - alpha) * conversation_state["running_vad"][k]
            )
        # keep emotion updated for the new trace as well
        conversation_state["slots"]["Emotion"] = dst_result["mapped_emotion"]

    # -------- STEP 2: SLOT EXTRACTION (NO EMOTION OVERRIDE) --------
    extracted_slots = await asyncio.to_thread(extract_slots_from_text, user_text)
    extracted_slots.pop("Emotion", None)
    conversation_state["slots"].update(extracted_slots)

    # -------- STEP 3: MANDATORY SLOT CHECK --------
    missing_slots = get_missing_slots(conversation_state["slots"])
    if missing_slots:
        slot = missing_slots[0]
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
            "running_vad": conversation_state["running_vad"]
        }

    # -------- STEP 4: ASK OPTIONAL PREFERENCES (ONCE) --------
    if not conversation_state["preferences_asked"]:
        conversation_state["preferences_asked"] = True
        return {
            "type": "preferences",
            "question": generate_preferences_question(conversation_state["slots"].get("Emotion", "neutral")),
            "slots_collected": conversation_state["slots"],
            "running_vad": conversation_state["running_vad"]
        }

    # -------- STEP 5: Neo4j Recommendations --------
    recommendations = await neo4j_recommend(conversation_state["slots"])

    emotion = conversation_state["slots"].get("Emotion", "stressed")

    # -------- STEP 6: LLM Response from retrieved records --------
    chat_response = await asyncio.to_thread(
        llm_recommendation_response,
        emotion,
        conversation_state["slots"],
        recommendations
    )

    final_output = {
        "type": "final",
        "emotion": emotion,
        "running_vad": conversation_state["running_vad"],
        "slots": dict(conversation_state["slots"]),
        "recommendations": recommendations,
        "reply": chat_response
    }

    # -------- STEP 7: RESET FOR NEXT TRACE (KEEP HISTORY) --------
    reset_after_recommendation_keep_history()

    return final_output

# ================== INTERACTIVE LOOP ==================
if __name__ == "__main__":
    conversation_state = reset_conversation_state()

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