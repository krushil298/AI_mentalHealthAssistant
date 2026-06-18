# AI Mental Health Support Assistant — a Demo of LLM Safety Engineering

⚠️ **This is a demonstration project, not a medical service.** It is not a doctor,
therapist, or crisis line and must never be used for real emergency support. Its
purpose is to teach how you build a *layered safety system* around an LLM.

---

## What this demo shows

A warm, supportive-listener chatbot (powered by Google Gemini) wrapped in a
**safety layer that runs on every message _before_ the model is ever called.**
The interesting part isn't the chatbot — it's the engineering around it.

---

## The layered safety architecture

A request travels through several independent layers. Each one can stop or shape
the response. Defense in depth: if one layer misses something, another can catch it.

```
  user message
      │
      ▼
  ┌─────────────────────────────────────────────────────────┐
  │ SAFETY LAYER  (safety.py — runs BEFORE the model)        │
  │                                                          │
  │   1. Sentiment scoring (VADER)                           │
  │        → feeds tone into the system prompt               │
  │                                                          │
  │   2. Risk screen (keyword/pattern, HIGH RECALL)          │
  │        ├─ FLAGGED ─► fixed safe escalation reply         │
  │        │             + crisis resources                  │
  │        │             + log to escalations.log            │
  │        │             (the model is NEVER called)         │
  │        │                                                 │
  │        └─ not flagged ─► continue                        │
  └─────────────────────────────────────────────────────────┘
      │ (only safe messages get here)
      ▼
  ┌─────────────────────────────────────────────────────────┐
  │ LLM LAYER  (app.py + prompts.py)                         │
  │   • constrained system prompt (no diagnosis/medical      │
  │     advice; brief; always point to real help)            │
  │   • try/except + 1 retry + safe fallback reply           │
  └─────────────────────────────────────────────────────────┘
      │
      ▼
  JSON: { reply, sentiment, escalated }
```

### Why the risk screen runs BEFORE the model

Two reasons:

1. **The model is the wrong tool when someone may be in danger.** An LLM is
   probabilistic — it might say something unhelpful or even harmful, and it might
   bury the crisis resources. For a high-risk message we want a **fixed,
   human-reviewed response** that always shows verified help. So we short-circuit:
   when the screen flags a message, **we never call the model at all.**

2. **It's a gate, not a filter.** Checking *after* generation means you've already
   spent a model call and you're now editing a risky output. Checking *before*
   means the dangerous content never reaches the model and never reaches the user.

### Why the screen optimizes for RECALL (it over-flags on purpose)

In a safety classifier the two errors are not equally costly:

| Error | What happens | Cost |
|-------|--------------|------|
| **False positive** | A safe message gets flagged | User sees crisis resources unnecessarily — mildly annoying |
| **False negative** | A person in real distress is missed | The dangerous failure — we lose the chance to escalate |

Because a miss is catastrophic and a false alarm is cheap, we deliberately tune the
screen to **err toward flagging** (high recall). The keyword/pattern list in
`safety.py` is broad and uses word-boundary regexes so variations still match.

> ⚠️ Keyword screening is a *coarse first line of defense.* It will miss subtle,
> metaphorical, or cleverly-worded distress, and it will sometimes misfire. In a
> real product you'd layer an ML classifier and/or a dedicated safety model on top.
> For a learning demo, simple + high-recall is the right starting point.

---

## Rule-based sentiment (VADER) vs. an ML sentiment model

This demo uses **VADER** (Valence Aware Dictionary and sEntiment Reasoner).

| | **VADER (rule-based)** — what we use | **ML sentiment model** |
|---|---|---|
| How it works | A hand-built lexicon of words with sentiment scores, plus rules for negation, intensifiers ("very"), punctuation, and emojis | Learns patterns from labeled training data (e.g. fine-tuned transformer) |
| Setup | Ships its dictionary in the package — **no model download, instant** | Needs a downloaded model (often 100s of MB) and more compute |
| Transparency | Fully explainable — you can read why a score was assigned | Often a black box |
| Strengths | Fast, deterministic, great on short social/informal text | Captures context, sarcasm, and nuance far better |
| Weaknesses | Misses context it has no rule for; only as good as its lexicon | Heavier, slower, can inherit training-data bias |

We pass VADER's **compound score** (range `-1.0` … `+1.0`) into the system prompt so
the assistant softens its tone for negative messages. It's a cheap, transparent
signal that's perfect for a demo — and a good reminder that **not every "AI" feature
needs a neural network.**

---

## Project structure

```
mental_health_assistant/
  app.py              Flask app + routes; the normal LLM path (retry + fallback)
  safety.py           THE safety layer: risk screen + VADER sentiment + escalation/logging
  prompts.py          The constrained system prompt (no diagnosis, brief, point to help)
  templates/chat.html Simple chat UI with the disclaimer banner
  requirements.txt
  .env.example        Copy to .env and add your Gemini key
  README.md
```

---

## Setup

```bash
cd mental_health_assistant

# 1. Create + activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini API key
cp .env.example .env
#    then edit .env and paste your key into GOOGLE_API_KEY
#    (get one at https://aistudio.google.com/app/apikey)

# 4. Fill in CRISIS_RESOURCES at the top of safety.py with a VERIFIED local helpline

# 5. Run
python app.py
#    open http://127.0.0.1:5000
```

## Two things you MUST configure before this is meaningful

1. **`GOOGLE_API_KEY`** in `.env` — without it, the normal chat path returns the
   safe fallback message instead of a real reply.
2. **`CRISIS_RESOURCES`** at the top of `safety.py` — the placeholders must be
   replaced with a **verified helpline for your region.** This text is shown to a
   person in crisis verbatim, so accuracy is not optional.

---

## API

`POST /chat` with `{"message": "..."}` returns:

```json
{ "reply": "…", "sentiment": { "score": -0.42, "label": "negative" }, "escalated": false }
```

When `escalated` is `true`, `reply` is the fixed crisis-resource message and the LLM
was not called. Every escalation is appended to `escalations.log` as a stand-in for
a real human handoff.
