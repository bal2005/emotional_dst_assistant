import asyncio
import json

# IMPORT YOUR EXISTING ORCHESTRATOR
from main_orchestrator_v2 import process_turn, conversation_state


# ---------------- RESET STATE ----------------
def reset_conversation_state():
    conversation_state["history"].clear()
    conversation_state["slots"].clear()
    conversation_state["running_vad"] = None


# ---------------- RUN SINGLE TEST ----------------
async def run_test(test_name, user_inputs):
    print("\n" + "=" * 70)
    print(f"🧪 TEST CASE: {test_name}")
    print("=" * 70)

    reset_conversation_state()

    for user_input in user_inputs:
        print(f"\nYou: {user_input}")
        output = await process_turn(user_input)
        print("Assistant:")
        print(json.dumps(output, indent=2))

        if output.get("type") == "final":
            print("\n✅ FINAL STATE REACHED")
            break


# ---------------- ALL DEMO TESTS ----------------
async def run_all_tests():

    # 1️⃣ Stressed → Park
    await run_test(
        "Stressed → Park Walk",
        [
            "I am feeling very stressed",
            "I want to go out",
            "Maybe a park"
        ]
    )

    # 2️⃣ Anxious → Beach
    await run_test(
        "Anxious → Beach Relaxation",
        [
            "I feel anxious these days",
            "I want to relax outside",
            "A beach would be nice"
        ]
    )

    # 3️⃣ Sad → Home
    await run_test(
        "Sad → Stay Home Comfort",
        [
            "I am feeling sad",
            "I just want to rest",
            "At home"
        ]
    )

    # 4️⃣ Angry → Gym
    await run_test(
        "Angry → Gym Workout",
        [
            "I am very angry today",
            "I want to release energy",
            "Maybe the gym"
        ]
    )

    # 5️⃣ Lonely → Cafe
    await run_test(
        "Lonely → Cafe Visit",
        [
            "I feel lonely",
            "I want to be around people",
            "A cafe sounds good"
        ]
    )

    # 6️⃣ Bored → Mall / Outing
    await run_test(
        "Bored → Mall Outing",
        [
            "I am bored",
            "I want to go out",
            "Maybe a mall"
        ]
    )

    # 7️⃣ Mixed Emotion → Temple
    await run_test(
        "Overwhelmed → Temple Visit",
        [
            "Everything feels overwhelming",
            "I want some peace",
            "A temple"
        ]
    )

    # 8️⃣ One-shot Natural Language
    await run_test(
        "One-shot Rich Input",
        [
            "I am stressed from work and want to calm down by walking in a quiet park"
        ]
    )


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    asyncio.run(run_all_tests())
