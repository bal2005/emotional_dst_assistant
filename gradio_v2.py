import asyncio
import json
import gradio as gr

# IMPORTANT:
# Point this import to your UPDATED orchestrator file that returns:
# { type: "final", reply: "...", recommendations: [...], ... }
from main_orchestrator_v2 import process_turn, reset_conversation_state


def run_async(coro):
    return asyncio.run(coro)


def safe_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def format_assistant_message(result: dict) -> str:
    t = result.get("type")

    if t == "clarification":
        return f" {result.get('question', 'Could you clarify?')}"

    if t == "preferences":
        return f" {result.get('question', 'Any preferences?')}"

    if t == "final":
        # Use LLM-generated grounded response if available
        reply = result.get("reply")
        if isinstance(reply, str) and reply.strip():
            return reply.strip()

        # fallback
        recs = result.get("recommendations", [])
        if recs:
            lines = ["Here are a few options I found:"]
            for r in recs[:3]:
                lines.append(f"• {r.get('place')} ({r.get('area')}) — {r.get('activity')}")
            return "\n".join(lines)

        return "I couldn't find recommendations yet. Want to try a different place/activity?"

    return " Something went wrong."


def extract_debug_fields(result: dict, state: dict):
    """
    Update and return (slots, running_vad, recommendations) for display.
    Tries to read from result first; falls back to state.
    """
    slots = (
        result.get("slots_collected")
        or result.get("slots")
        or state.get("slots")
        or {}
    )

    running_vad = (
        result.get("running_vad")
        or state.get("running_vad")
        or {}
    )

    recommendations = (
        result.get("recommendations")
        or state.get("recommendations")
        or []
    )

    # Persist into state
    state["slots"] = slots
    state["running_vad"] = running_vad
    state["recommendations"] = recommendations

    return slots, running_vad, recommendations, state


def chat_handler(user_text, history, state):
    user_text = (user_text or "").strip()

    # Keep current panel values if empty input
    if not user_text:
        return (
            history,
            state,
            safe_json(state.get("slots", {})),
            safe_json(state.get("running_vad", {})),
            safe_json(state.get("recommendations", [])),
            ""
        )

    # Reset
    if user_text.lower() in {"reset", "/reset"}:
        state = reset_conversation_state()
        history = [{"role": "assistant", "content": "✅ Reset done. Tell me how you're feeling."}]
        return (
            history,
            state,
            safe_json(state.get("slots", {})),
            safe_json(state.get("running_vad", {})),
            safe_json(state.get("recommendations", [])),
            ""
        )

    # Quit
    if user_text.lower() == "quit":
        history.append({"role": "assistant", "content": "👋 Take care. I'm here whenever you need."})
        return (
            history,
            state,
            safe_json(state.get("slots", {})),
            safe_json(state.get("running_vad", {})),
            safe_json(state.get("recommendations", [])),
            ""
        )

    # Add user message to chat history
    history.append({"role": "user", "content": user_text})

    # Run orchestrator
    try:
        result = run_async(process_turn(user_text))
        assistant_msg = format_assistant_message(result)
    except Exception as e:
        result = {}
        assistant_msg = f"⚠️ Error: {e}"

    # Update debug panels
    slots, running_vad, recommendations, state = extract_debug_fields(result, state)

    # Add assistant response
    history.append({"role": "assistant", "content": assistant_msg})

    return (
        history,
        state,
        safe_json(slots),
        safe_json(running_vad),
        safe_json(recommendations),
        ""
    )


def on_reset():
    state = reset_conversation_state()
    history = [{"role": "assistant", "content": " Reset done. Tell me how you're feeling."}]
    return (
        history,
        state,
        safe_json(state.get("slots", {})),
        safe_json(state.get("running_vad", {})),
        safe_json(state.get("recommendations", [])),
    )


with gr.Blocks() as demo:
    gr.Markdown("##  Emotional Wellness Assistant 🌱")

    state = gr.State(reset_conversation_state())

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=520, label="Conversation")
            txt = gr.Textbox(placeholder="Tell me how you're feeling...", show_label=False)
            with gr.Row():
                reset_btn = gr.Button("Reset")

        with gr.Column(scale=1):
            gr.Markdown("###  Extracted Slots")
            slots_box = gr.Code(value="{}", language="json", lines=12)

            gr.Markdown("###  Running VAD")
            vad_box = gr.Code(value="{}", language="json", lines=8)

            gr.Markdown("###  Top-3 Recommendations (Raw)")
            recs_box = gr.Code(value="[]", language="json", lines=12)

    txt.submit(
        chat_handler,
        inputs=[txt, chatbot, state],
        outputs=[chatbot, state, slots_box, vad_box, recs_box, txt]
    )

    reset_btn.click(
        on_reset,
        inputs=None,
        outputs=[chatbot, state, slots_box, vad_box, recs_box]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)