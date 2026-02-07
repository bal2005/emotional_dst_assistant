import gradio as gr
import asyncio

# 🔗 Import orchestrator
from main_orchestrator_v2 import process_turn, reset_conversation_state

# ------------------------------------------------
# Reset state ONCE when app starts
# ------------------------------------------------
reset_conversation_state()

# ------------------------------------------------
# Async wrapper (Gradio needs sync fn)
# ------------------------------------------------
def run_async(coro):
    return asyncio.run(coro)

# ------------------------------------------------
# Chat handler (message-based)
# ------------------------------------------------
def chat_handler(user_text, history):
    """
    history: List[{'role': 'user'|'assistant', 'content': str}]
    """

    if user_text.strip().lower() == "quit":
        history.append(
            {"role": "assistant", "content": "👋 Take care. I'm here whenever you need."}
        )
        return history

    # Run orchestrator
    result = run_async(process_turn(user_text))

    # Build assistant response
    if result["type"] == "clarification":
        assistant_msg = f"❓ {result['question']}"

    elif result["type"] == "preferences":
        assistant_msg = f"🧩 {result['question']}"

    elif result["type"] == "final":
        assistant_msg = (
            f"🧠 **Emotion:** {result['emotion']}\n\n"
            f"📊 **Running VAD**\n"
            f"- Valence: {result['running_vad']['valence']:.2f}\n"
            f"- Arousal: {result['running_vad']['arousal']:.2f}\n"
            f"- Dominance: {result['running_vad']['dominance']:.2f}\n\n"
            f"📌 **Slots Collected:**\n"
        )
        for k, v in result["slots"].items():
            assistant_msg += f"- {k}: {v}\n"

    else:
        assistant_msg = "⚠️ Something went wrong."

    # Append messages
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_msg})

    return history


# ------------------------------------------------
# Gradio UI
# ------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Emotional Wellness Assistant 🌱")
    gr.Markdown(
        "Emotion-aware Dialogue State Tracking system with VAD smoothing.\n\n"
        "Talk freely about how you feel or what you want to do."
    )

    chatbot = gr.Chatbot(height=450)

    user_input = gr.Textbox(
        placeholder="Tell me how you're feeling...",
        show_label=False
    )

    user_input.submit(
        chat_handler,
        inputs=[user_input, chatbot],
        outputs=chatbot
    )

    user_input.submit(lambda: "", None, user_input)

demo.launch()
