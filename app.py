"""
app.py
------
The Flask web app. It wires together:
    - the SAFETY LAYER (safety.py)  -> runs on every message, before the model
    - the SYSTEM PROMPT (prompts.py) -> shapes the model's behavior
    - the LLM (Google Gemini via langchain-google-genai)

Request flow for POST /chat:
    user message
       -> safety layer (sentiment + risk screen)
          -> if HIGH RISK: return fixed escalation reply  (model is NEVER called)
          -> else:        call Gemini with the system prompt, with retry + fallback
       -> JSON: {reply, sentiment, escalated}
"""

import os

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# langchain wrapper around Gemini's chat models.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Our own modules.
from prompts import build_system_prompt
from safety import run_safety_layer


# ---------------------------------------------------------------------------
# CONFIG / STARTUP
# ---------------------------------------------------------------------------
# Load environment variables from .env (e.g. GOOGLE_API_KEY). This must run before
# we read os.environ. The .env file is gitignored; .env.example shows the format.
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# A calm, hand-written reply we fall back to if the LLM errors or returns nothing.
# This guarantees the endpoint NEVER crashes or returns an empty body.
FALLBACK_REPLY = (
    "I'm having a little trouble responding right now, but I'm still here with you. "
    "If something is weighing on you, talking to a trusted person or a mental-health "
    "professional can really help."
)

app = Flask(__name__)


def _make_llm():
    """
    Construct the Gemini chat client.

    We build it lazily (per call is fine for a demo) and keep temperature modest so
    replies stay grounded and consistent rather than wandering. If the API key is
    missing we return None so the caller can fall back gracefully instead of crashing.
    """
    if not GOOGLE_API_KEY:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # fast, inexpensive; fine for a supportive listener
        google_api_key=GOOGLE_API_KEY,
        temperature=0.6,
    )


def generate_llm_reply(message: str, sentiment: dict) -> str:
    """
    Call Gemini for the NORMAL (non-escalated) path.

    Reliability features required by the brief:
      - try/except so an API error never bubbles up as a 500,
      - ONE retry on transient failure,
      - a safe fallback message if the call fails or returns empty.

    The sentiment signal is injected into the system prompt so the model adapts tone.
    """
    llm = _make_llm()
    if llm is None:
        # No API key configured — degrade gracefully.
        return FALLBACK_REPLY

    # Assemble the conversation: standing instructions + this user's message.
    system_prompt = build_system_prompt(sentiment["label"], sentiment["score"])
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message),
    ]

    # Try up to twice (initial attempt + 1 retry) to ride out transient errors.
    attempts = 2
    for attempt in range(attempts):
        try:
            response = llm.invoke(messages)
            text = (response.content or "").strip()
            if text:
                return text
            # Empty response is treated like a soft failure -> retry, then fallback.
        except Exception as exc:  # noqa: BLE001 - we intentionally catch everything here
            # Log to console for the developer; the user still gets a safe reply.
            print(f"[LLM] attempt {attempt + 1}/{attempts} failed: {exc}")

    # Both attempts exhausted (error or empty) -> safe fallback.
    return FALLBACK_REPLY


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serve the chat UI."""
    return render_template("chat.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Lightweight health/readiness check.

    Useful for deployment monitoring and for confirming configuration WITHOUT
    sending a chat message. It reports:
      - status: the service is up and serving requests,
      - llm_configured: whether GOOGLE_API_KEY is present. If this is False the
        normal chat path will return the safe fallback instead of a real reply.

    Note: we never expose the key itself — only whether one is set.
    """
    return jsonify({
        "status": "ok",
        "llm_configured": bool(GOOGLE_API_KEY),
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Expects JSON: {"message": "..."}.
    Returns JSON: {"reply": str, "sentiment": {...}, "escalated": bool}.
    """
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    # Basic input guard.
    if not message:
        return jsonify({
            "reply": "I'm listening — feel free to share what's on your mind.",
            "sentiment": {"score": 0.0, "label": "neutral"},
            "escalated": False,
        })

    # ---- THE SAFETY LAYER RUNS FIRST, BEFORE ANY MODEL CALL ----
    result = run_safety_layer(message)
    sentiment = result["sentiment"]

    if result["escalated"]:
        # High-risk path: return the fixed, safe escalation reply. No LLM involved.
        return jsonify({
            "reply": result["safe_reply"],
            "sentiment": sentiment,
            "escalated": True,
        })

    # Normal path: it's safe to call the model (with retry + fallback inside).
    reply = generate_llm_reply(message, sentiment)
    return jsonify({
        "reply": reply,
        "sentiment": sentiment,
        "escalated": False,
    })


if __name__ == "__main__":
    # debug=True gives auto-reload + tracebacks while you learn/iterate.
    # Bind to localhost only — this is a local demo, not a public service.
    app.run(host="127.0.0.1", port=5000, debug=True)
