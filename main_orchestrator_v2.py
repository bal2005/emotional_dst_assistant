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
def reset_conversation_state():
    return {
        "history": [],
        "slots": {},                 # Emotion, Activity, Place, Event, Tag, Remedy
        "running_vad": None,
        "preferences_asked": False   # 👈 new flag
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

        resp = requests.post(LLAMA_API_URL, json=payload)
        raw = resp.json()["choices"][0]["message"]["content"]

        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return {k: v for k, v in parsed.items() if v is not None}

    except Exception as e:
        print("❌ Slot extraction failed:", e)
        return {}

# ================== SLOT-FOCUSED CLARIFICATION ==================
def generate_clarification_question(
    missing_slot: str,
    emotion: str,
    history: List[str]
) -> str:
    """
    Generates a clarification question ONLY for the missing slot.
    """

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
        resp = requests.post(LLAMA_API_URL, json=payload)
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"Could you tell me about the {missing_slot.lower()}?"

# ================== OPTIONAL PREFERENCES QUESTION ==================
def generate_preferences_question(emotion: str) -> str:
    return (
        f"Before I suggest something, do you have any other preferences "
        f"like a specific occasion, mood tag, or something that helps you relax?"
    )

# ================== EMPTY DB SEARCH (ASYNC STUB) ==================
async def semantic_search_stub(slots: Dict) -> List[Dict]:
    """
    Neo4j intentionally isolated for now.
    """
    await asyncio.sleep(0.05)
    return []

# ================== SLOT COMPLETENESS ==================
def get_missing_slots(slots: Dict) -> List[str]:
    return [s for s in MANDATORY_SLOTS if s not in slots]

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

    # -------- STEP 2: SLOT EXTRACTION (NO EMOTION OVERRIDE) --------
    extracted_slots = await asyncio.to_thread(
        extract_slots_from_text, user_text
    )

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
            "question": generate_preferences_question(
                conversation_state["slots"].get("Emotion", "neutral")
            ),
            "slots_collected": conversation_state["slots"],
            "running_vad": conversation_state["running_vad"]
        }

    # -------- STEP 5: DB SEARCH (STUB) --------
    recommendations = await semantic_search_stub(conversation_state["slots"])

    return {
        "type": "final",
        "emotion": conversation_state["slots"]["Emotion"],
        "running_vad": conversation_state["running_vad"],
        "slots": conversation_state["slots"],
        "recommendations": recommendations
    }

# ================== INTERACTIVE LOOP ==================
if __name__ == "__main__":

    # 🔁 RESET EMA + STATE AT START
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
        print(json.dumps(output, indent=2))
