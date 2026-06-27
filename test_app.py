"""
test_app.py
-----------
Endpoint-level tests for the Flask app, using Flask's built-in test client.

These run WITHOUT a Gemini API key, because they only exercise paths that don't
reach the model:
  - GET  /health         -> service/readiness info
  - GET  /               -> the chat page (with its disclaimer banner)
  - POST /chat (risk)    -> escalation path, which by design SKIPS the LLM
  - POST /chat (empty)   -> canned prompt, also no model call

Run with no extra dependencies:
    python test_app.py
(or, if you have pytest installed:  pytest test_app.py)
"""

from app import app


def _client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health_reports_ok_and_config_flag():
    resp = _client().get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    # We don't assert True/False (depends on the env) — just that the flag exists
    # and is a boolean, and that the key itself is never leaked.
    assert isinstance(data["llm_configured"], bool)
    assert "GOOGLE_API_KEY" not in resp.get_data(as_text=True)


def test_index_serves_chat_page_with_disclaimer():
    resp = _client().get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # The disclaimer banner is a safety requirement of the UI — make sure it ships.
    assert "demo, not a medical service" in body.lower()


def test_chat_high_risk_escalates_without_model():
    # The escalation path runs entirely in the safety layer, so this works even
    # with no API key configured.
    resp = _client().post("/chat", json={"message": "I want to end my life"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["escalated"] is True
    assert "sentiment" in data
    assert data["reply"]  # must contain the safe escalation text


def test_chat_empty_message_is_handled_gracefully():
    resp = _client().post("/chat", json={"message": "   "})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["escalated"] is False
    assert data["reply"]  # never returns an empty body


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
