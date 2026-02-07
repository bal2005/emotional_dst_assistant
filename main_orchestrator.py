import json
import asyncio
import requests
from typing import Dict, List

# ================== IMPORT UNCHANGED DST CORE ==================
# DO NOT MODIFY emotional_dst.py
from emotional_dst import process_utterance

# ================== LLM CONFIG ==================
LLAMA_API_URL = "http://localhost:1234/v1/chat/completions"
LLAMA_MODEL = "meta-llama-3.1-8b-instruct-hf"

# ================== CONVERSATION STATE ==================
conversation_state = {
    "history": [],
    "slots": {},                 # Emotion, Activity, Place, Event, Tag, Remedy
    "running_vad": None,
    "emotion_locked": False      # Emotion inferred only after slots complete
}

MANDATORY_SLOTS = ["Activity", "Place"]

# ================== SLOT → QUESTION MAP ==================
CLARIFICATION_QUESTIONS = {
    "Activity": "What kind of activity are you interested in?",
    "Place": "Do you have a preferred place or location?"
}

# ================== LLM SLOT EXTRACTION ==================
def extract_slots_from_text(user_text: str) -> Dict:
    """
    LLM-only slot extraction.
    Output MUST be JSON.
    """
    prompt = f"""
Extract entities from user input as JSON ONLY.

Ontology labels:
Emotion, Activity, Place, Event, Tag, Remedy

Rules:
- If a slot is not mentioned, set it to null
- Do not add explanations
- Do not add extra text

Allowed Emotions:
Anxious, Stressed, Sad, Lonely, Bored, Happy, Angry

User input:
"{user_text}"
"""

    try:
        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are a precise information extractor."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 300,
            "stream": False
        }

        resp = requests.post(LLAMA_API_URL, json=payload)
        raw = resp.json()["choices"][0]["message"]["content"]

        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return {k: v for k, v in parsed.items() if v is not None}

    except Exception as e:
        print("❌ Slot extraction failed:", e)
        return {}

# ================== EMPTY DB SEARCH (ASYNC STUB) ==================
async def semantic_search_stub(slots: Dict) -> List[Dict]:
    """
    Neo4j intentionally isolated for now.
    """
    await asyncio.sleep(0.05)
    return []

# ================== SLOT COMPLETENESS CHECK ==================
def get_missing_slots(slots: Dict) -> List[str]:
    return [s for s in MANDATORY_SLOTS if s not in slots]

# ================== MAIN TURN HANDLER ==================
async def process_turn(user_text: str) -> Dict:

    conversation_state["history"].append(user_text)

    # -------- STEP 1: EMOTION + VAD (ALWAYS RUN) --------
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

    # -------- STEP 2: SLOT EXTRACTION (NO EMOTION) --------
    extracted_slots = await asyncio.to_thread(
        extract_slots_from_text, user_text
    )

    extracted_slots.pop("Emotion", None)
    conversation_state["slots"].update(extracted_slots)

    # -------- STEP 3: CHECK MISSING SLOTS --------
    missing_slots = get_missing_slots(conversation_state["slots"])

    if missing_slots:
        slot_to_ask = missing_slots[0]
        return {
            "type": "clarification",
            "question": CLARIFICATION_QUESTIONS.get(
                slot_to_ask,
                f"Could you clarify the {slot_to_ask}?"
            ),
            "slots_collected": conversation_state["slots"],
            "running_vad": conversation_state["running_vad"]
        }

    # -------- STEP 4: DB SEARCH (STUB) --------
    recommendations = await semantic_search_stub(conversation_state["slots"])

    return {
        "type": "final",
        "emotion": conversation_state["slots"]["Emotion"],
        "running_vad": conversation_state["running_vad"],
        "slots": conversation_state["slots"],
        "recommendations": recommendations
    }


# ================== INTERACTIVE TEST LOOP ==================
if __name__ == "__main__":
    print("\n🟢 Emotional DST Orchestrator Started")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        output = asyncio.run(process_turn(user_input))
        print(json.dumps(output, indent=2))
