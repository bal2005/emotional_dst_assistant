"""
FastAPI backend for Emotional Wellness Assistant.
Wraps main_orchestrator_v2.process_turn and reset_conversation_state.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from main_orchestrator_v2 import (
    process_turn,
    reset_conversation_state as _reset,
    set_alpha,
    conversation_state,
    DEFAULT_ORCHESTRATOR_ALPHA,
)

app = FastAPI(title="Emotional Wellness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    text: str
    alpha: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=0.99,
        description="EMA smoothing factor (0.01–0.99). Higher = more weight on latest utterance.",
    )


class AlphaRequest(BaseModel):
    alpha: float = Field(
        ...,
        ge=0.01,
        le=0.99,
        description="EMA smoothing factor (0.01–0.99).",
    )


@app.post("/chat")
async def chat(req: MessageRequest):
    result = await process_turn(req.text, alpha=req.alpha)
    return result


@app.post("/reset")
async def reset():
    _reset()
    return {"status": "reset"}


@app.get("/config")
async def get_config():
    return {
        "alpha": conversation_state["alpha"],
        "alpha_default": DEFAULT_ORCHESTRATOR_ALPHA,
        "alpha_min": 0.01,
        "alpha_max": 0.99,
    }


@app.post("/config")
async def update_config(req: AlphaRequest):
    set_alpha(req.alpha)
    return {"alpha": conversation_state["alpha"]}


@app.get("/health")
async def health():
    return {"status": "ok"}
