"""
safety.py
---------
THE SAFETY LAYER. This is the heart of the demo.

It runs on EVERY user message BEFORE we ever call the LLM, and it does three jobs:

  1. RISK SCREEN  — a fast keyword/pattern check for crisis & distress indicators.
                    Tuned for HIGH RECALL (it would rather over-flag than miss a
                    person in danger). If it flags, we DO NOT call the chat model;
                    we return a fixed, safe escalation message instead.

  2. SENTIMENT    — VADER (rule-based, no model download) scores the emotional tone
                    of the message so the assistant can adapt how gently it speaks.

  3. ESCALATION   — when risk is flagged, we surface crisis resources and log the
                    event to escalations.log as a stand-in for a real human handoff.

Design principle: in a safety system, the costs of the two error types are NOT
equal. A false positive (flagging a safe message) is mildly annoying. A false
negative (missing someone at real risk) is catastrophic. So we deliberately bias
the screen toward flagging. That is what "optimize for recall" means here.
"""

import re
import logging
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# CONFIGURABLE CRISIS RESOURCES
# ---------------------------------------------------------------------------
# >>> FILL THIS IN <<<
# Replace these placeholders with a VERIFIED helpline for YOUR region/country.
# This text is shown to the user verbatim when a message is flagged as high-risk,
# so accuracy literally matters here. Keep entries short and actionable.
CRISIS_RESOURCES = """If you're in immediate danger, please contact your local emergency number now.

You can also reach a trained human who wants to help:
  • [CRISIS LINE NAME]  —  [PHONE NUMBER]   (replace with a verified local helpline)
  • [TEXT/CHAT OPTION]  —  [DETAILS]        (e.g. text HOME to a crisis text line)

You deserve support from a real person, and reaching out is a sign of strength."""


# ---------------------------------------------------------------------------
# LOGGING SETUP  (stand-in for "human handoff")
# ---------------------------------------------------------------------------
# Every escalation is appended to escalations.log. In a real system this is where
# you'd notify an on-call clinician / safety team. Here it's our paper trail.
escalation_logger = logging.getLogger("escalations")
escalation_logger.setLevel(logging.INFO)
# Avoid attaching duplicate handlers if this module is imported more than once.
if not escalation_logger.handlers:
    _file_handler = logging.FileHandler("escalations.log")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    escalation_logger.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# RISK SCREEN
# ---------------------------------------------------------------------------
# These patterns are intentionally broad. We use word-boundary regexes (not exact
# string equality) so that variations and surrounding text still match. High
# recall is the goal: better to catch a borderline phrase than to let it slip.
#
# NOTE: keyword matching is a *coarse* first line of defense, not the whole story.
# It will miss cleverly-worded or metaphorical distress, and it will sometimes
# misfire (e.g. "this kills me" said about a funny joke). For a learning demo
# that's an acceptable trade-off, and over-flagging is the safer direction.
HIGH_RISK_PATTERNS = [
    # Direct statements of suicidal intent / ideation
    r"\bkill myself\b",
    r"\bkilling myself\b",
    r"\bend my life\b",
    r"\bend it all\b",
    r"\btake my (own )?life\b",
    r"\bsuicid",                       # suicide, suicidal, ...
    r"\bwant to die\b",
    r"\bwanna die\b",
    r"\bwish i (was|were) dead\b",
    r"\bbetter off dead\b",
    r"\bdon'?t want to (be here|live|exist)\b",
    r"\bno reason to live\b",
    r"\bcan'?t go on\b",
    r"\bgive up on life\b",

    # Self-harm
    r"\bself[\s-]?harm\b",
    r"\bcut myself\b",
    r"\bcutting myself\b",
    r"\bhurt myself\b",
    r"\bharm myself\b",

    # Hopelessness / planning language often seen near crisis
    r"\bno way out\b",
    r"\bnothing matters\b",
    r"\beveryone (would )?be better off without me\b",
    r"\bgoodbye forever\b",
    r"\bfinal goodbye\b",
    r"\boverdose\b",

    # Harm toward others (also requires escalation, not chatbot chatter)
    r"\bkill (him|her|them|someone|everyone)\b",
    r"\bhurt (someone|somebody|people)\b",
]

# Pre-compile the patterns once (case-insensitive) for speed and clarity.
_COMPILED_RISK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in HIGH_RISK_PATTERNS]


def screen_for_risk(message: str):
    """
    Fast first-pass risk screen.

    Returns a tuple (is_flagged, matched_terms):
        is_flagged    -> True if ANY high-risk pattern matched.
        matched_terms -> list of the patterns that matched (useful for logging).

    Because we OR together many broad patterns, this errs toward flagging — the
    high-recall behavior we want for a safety gate.
    """
    matched_terms = []
    for pattern in _COMPILED_RISK_PATTERNS:
        if pattern.search(message):
            matched_terms.append(pattern.pattern)

    is_flagged = len(matched_terms) > 0
    return is_flagged, matched_terms


# ---------------------------------------------------------------------------
# SENTIMENT DETECTION  (VADER — rule-based)
# ---------------------------------------------------------------------------
# VADER ("Valence Aware Dictionary and sEntiment Reasoner") is a lexicon + rules
# system. It ships its word list inside the package, so there is NO model download
# and inference is instant. It returns a "compound" score in [-1, 1] that we map
# into simple buckets for the prompt.
_vader = SentimentIntensityAnalyzer()


def analyze_sentiment(message: str):
    """
    Score the message's sentiment with VADER.

    Returns a dict:
        {
          "score": float,   # compound score, -1.0 (very neg) .. +1.0 (very pos)
          "label": str,     # "negative" / "neutral" / "positive"
        }
    """
    scores = _vader.polarity_scores(message)
    compound = scores["compound"]

    # Standard VADER thresholds for bucketing the compound score.
    if compound <= -0.05:
        label = "negative"
    elif compound >= 0.05:
        label = "positive"
    else:
        label = "neutral"

    return {"score": compound, "label": label}


# ---------------------------------------------------------------------------
# ESCALATION RESPONSE
# ---------------------------------------------------------------------------
def build_escalation_response() -> str:
    """
    The FIXED, safe response we return when risk is flagged.

    Why fixed (not LLM-generated)? When someone may be in danger we do not want a
    probabilistic model improvising. A hand-written, reviewed message guarantees we:
      - gently acknowledge the person,
      - clearly say we are not a substitute for professional help,
      - show the verified crisis resources.
    """
    return (
        "I'm really glad you reached out, and I want you to be safe. "
        "I'm just a demo and not a substitute for professional help — "
        "but you don't have to go through this alone.\n\n"
        f"{CRISIS_RESOURCES}"
    )


def log_escalation(message: str, matched_terms, sentiment: dict) -> None:
    """
    Record the escalation to escalations.log as a stand-in for human handoff.

    In production this is where you'd page a clinician or open a safety ticket. We
    log the matched terms and sentiment, plus a short snippet of the message for
    triage context. (Be mindful of privacy: we keep only a truncated snippet.)
    """
    snippet = message.strip().replace("\n", " ")[:200]
    escalation_logger.info(
        "ESCALATION | matched=%s | sentiment=%s(%.2f) | message=%r",
        matched_terms,
        sentiment["label"],
        sentiment["score"],
        snippet,
    )


# ---------------------------------------------------------------------------
# ORCHESTRATION  — the single entry point app.py calls
# ---------------------------------------------------------------------------
def run_safety_layer(message: str):
    """
    Run the full safety layer on one user message.

    This is the function app.py calls. It bundles sentiment + risk screening and
    decides whether to escalate. It returns everything app.py needs to build the
    HTTP response, WITHOUT calling the LLM itself (that stays in app.py so the
    safety layer has no dependency on the model).

    Returns a dict:
        {
          "escalated":  bool,          # True -> skip the LLM, use safe_reply
          "safe_reply": str | None,    # the fixed escalation text (if escalated)
          "sentiment":  {"score": float, "label": str},
        }
    """
    # 1. Sentiment first — cheap, and we want it even when we escalate (for logs).
    sentiment = analyze_sentiment(message)

    # 2. Risk screen. This is the gate.
    is_flagged, matched_terms = screen_for_risk(message)

    if is_flagged:
        # 3. Escalate: log it (human-handoff stand-in) and return the SAFE reply.
        #    Critically, we never reach the LLM on this path.
        log_escalation(message, matched_terms, sentiment)
        return {
            "escalated": True,
            "safe_reply": build_escalation_response(),
            "sentiment": sentiment,
        }

    # Not flagged: app.py is cleared to call the LLM with the sentiment signal.
    return {
        "escalated": False,
        "safe_reply": None,
        "sentiment": sentiment,
    }
