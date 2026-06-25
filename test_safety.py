"""
test_safety.py
--------------
Tests for the SAFETY LAYER — the part of this demo where a bug is most costly.

We focus on the behavior that actually keeps people safe:
  - the risk screen must FLAG crisis messages (catching these is the whole point),
  - it must let ordinary messages through (so the bot is still usable),
  - escalation must skip the model and return crisis resources,
  - sentiment must produce a score in the valid range.

Run it with no extra dependencies:
    python test_safety.py
(or, if you have pytest installed:  pytest test_safety.py)
"""

from safety import (
    screen_for_risk,
    analyze_sentiment,
    run_safety_layer,
    CRISIS_RESOURCES,
)


# ---------------------------------------------------------------------------
# RISK SCREEN — recall: every one of these MUST be flagged.
# ---------------------------------------------------------------------------
# These phrase variations exist to guard against regressions where someone edits
# the pattern list and accidentally narrows it. If any of these stops flagging,
# the test fails loudly.
MUST_FLAG = [
    "I want to kill myself",
    "I just want to end my life",
    "honestly I want to die",
    "I wish I were dead",
    "everyone would be better off without me",
    "I've been thinking about suicide",
    "I keep cutting myself",
    "there's no reason to live anymore",
    "I can't go on",
]

# Ordinary, non-crisis messages that should NOT be flagged, so the assistant
# stays usable for everyday venting and small talk.
MUST_NOT_FLAG = [
    "I had a stressful day at work",
    "I'm feeling a bit down but I'll be okay",
    "Can you help me feel less anxious before a meeting?",
    "I love rainy mornings with coffee",
    "My friend and I had an argument",
]


def test_high_risk_messages_are_flagged():
    for msg in MUST_FLAG:
        flagged, matched = screen_for_risk(msg)
        assert flagged, f"FAILED to flag high-risk message: {msg!r}"
        assert matched, f"Flagged but reported no matched terms: {msg!r}"


def test_ordinary_messages_are_not_flagged():
    for msg in MUST_NOT_FLAG:
        flagged, _ = screen_for_risk(msg)
        assert not flagged, f"Over-flagged an ordinary message: {msg!r}"


def test_risk_screen_is_case_insensitive():
    # The screen must not be fooled by capitalization.
    flagged, _ = screen_for_risk("I WANT TO KILL MYSELF")
    assert flagged, "Risk screen should be case-insensitive"


# ---------------------------------------------------------------------------
# SENTIMENT — VADER should return a compound score within [-1, 1].
# ---------------------------------------------------------------------------
def test_sentiment_range_and_labels():
    for msg in ["I am so happy today!", "this is the worst day ever", "the meeting is at 3pm"]:
        result = analyze_sentiment(msg)
        assert -1.0 <= result["score"] <= 1.0, f"Score out of range for {msg!r}"
        assert result["label"] in {"negative", "neutral", "positive"}


def test_negative_message_scores_negative():
    result = analyze_sentiment("I feel hopeless and miserable and everything is awful")
    assert result["label"] == "negative", f"Expected negative, got {result}"


# ---------------------------------------------------------------------------
# ORCHESTRATION — the full safety layer's decision and shape.
# ---------------------------------------------------------------------------
def test_escalation_path_returns_resources_and_skips_model():
    result = run_safety_layer("I want to end my life")
    assert result["escalated"] is True
    # The safe reply must actually contain the crisis-resource text we configured.
    assert CRISIS_RESOURCES in result["safe_reply"]
    # sentiment is still computed even on the escalation path (used for logging).
    assert "score" in result["sentiment"]


def test_safe_path_defers_to_model():
    result = run_safety_layer("I had a long day, just wanted to talk")
    assert result["escalated"] is False
    # On the safe path we do NOT craft a reply here — app.py calls the LLM.
    assert result["safe_reply"] is None


# ---------------------------------------------------------------------------
# Standalone runner so `python test_safety.py` works without pytest installed.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    raise SystemExit(1 if failures else 0)
