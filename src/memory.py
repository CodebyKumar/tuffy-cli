"""Long-term memory: the store that makes the agent self-evolving.

Layout under data/memory/ (see data/memory/README.md):
  user/profile.json    - allowlisted personal-info fields about the user.
  user/notes.json      - every other durable fact about the user.
  agent/lessons.json   - capped operational lessons learned from tool failures.
  sessions/log.json    - episodic summaries of past sessions.
  quarantine.json      - extraction hits rejected by validation, kept for
                         debugging in Ray mode; never fed back into the prompt.

What does NOT live here: identity ("what model are you", "what's your
purpose") — that's src/identity.py, code-owned and never LLM-written. This
module actively refuses to store identity-shaped keys (see _is_identity_key)
because a small model cannot reliably separate "the user told me a fact
about themselves" from "I just said something about myself" — it kept
storing its own model name/role/purpose as if they were user facts.

Memory updates through three channels:
1. The 'remember' tool, when the user explicitly asks.
2. extract_facts(): a validated reflection pass main.py runs after each turn
   (off the main thread — see main.py's AsyncMemory), so facts the user
   mentions in passing get stored without any tool call.
3. add_session_summary()/add_lesson(): written on exit and by the agent loop
   when a failed tool call is corrected.
"""

import json
import os
import re
from datetime import datetime

from src.identity import RESERVED_IDENTITY_KEYS, is_self_referential_value, is_transcript_key
from src.tools.registry import registry

MEMORY_DIR = "./data/memory"
PROFILE_FILE = os.path.join(MEMORY_DIR, "user", "profile.json")
NOTES_FILE = os.path.join(MEMORY_DIR, "user", "notes.json")
SESSIONS_FILE = os.path.join(MEMORY_DIR, "sessions", "log.json")
LESSONS_FILE = os.path.join(MEMORY_DIR, "agent", "lessons.json")
QUARANTINE_FILE = os.path.join(MEMORY_DIR, "quarantine.json")

# Fixed allowlist of personal-info keys that get routed to profile.json.
# Anything else the model remembers goes to notes.json.
PROFILE_KEYS = {"name", "email", "location", "age", "occupation", "pronouns"}

# Values a reflection pass might extract that carry no real information —
# refusing these stops "goals: unknown" style noise from ever being stored.
_PLACEHOLDER_VALUES = {
    "unknown", "n/a", "na", "none", "null", "not specified", "not provided",
    "not sure", "unclear", "tbd", "?", "-", "",
}

MAX_SESSIONS_KEPT = 20
SESSIONS_IN_PROMPT = 3
MAX_LESSONS = 5
MAX_QUARANTINE_KEPT = 50

for _dir in (os.path.dirname(PROFILE_FILE), os.path.dirname(SESSIONS_FILE), os.path.dirname(LESSONS_FILE)):
    os.makedirs(_dir, exist_ok=True)


def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", key.strip().lower().replace(" ", "_"))


def _is_identity_key(key: str) -> bool:
    normalized = _normalize_key(key)
    return normalized in RESERVED_IDENTITY_KEYS or any(
        normalized == reserved or normalized.endswith(f"_{reserved}")
        for reserved in RESERVED_IDENTITY_KEYS
    )


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_VALUES


def _quarantine(key: str, value: str, reason: str) -> None:
    """Records a rejected extraction for later inspection (Ray mode /memory
    can surface this) instead of silently dropping it with no trace."""
    entries = _load_json(QUARANTINE_FILE, [])
    entries.append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "key": key, "value": value, "reason": reason,
    })
    _save_json(QUARANTINE_FILE, entries[-MAX_QUARANTINE_KEPT:])


def load_memory() -> dict:
    """Merged view of profile + notes, for building the system prompt."""
    return {**_load_json(PROFILE_FILE, {}), **_load_json(NOTES_FILE, {})}


def store_fact(key: str, value: str, source: str = "explicit") -> tuple[bool, str]:
    """Persists one fact, routing profile keys to profile.json.

    Returns (changed, reason). Rejects identity-shaped keys and placeholder
    values outright (quarantined, never stored) — a fact extractor calling
    this with junk should not be able to pollute the user's memory, whether
    the call came from the explicit 'remember' tool or the automatic
    reflection pass.
    """
    normalized = _normalize_key(key)
    value = str(value).strip()

    if not normalized or not value:
        return False, "empty key or value"
    if _is_identity_key(normalized):
        if source == "auto":
            _quarantine(key, value, "identity-shaped key")
        return False, "that's part of my own identity, not something to remember about you"
    if source == "auto" and is_transcript_key(normalized):
        # The extractor echoed the raw exchange back as if it were a fact
        # (e.g. {"user_message": "...", "assistant_reply": "..."}) — that's
        # a transcript, not a durable fact, no matter what it's filed under.
        _quarantine(key, value, "transcript-shaped key, not a fact")
        return False, "that's a copy of the conversation, not a fact to remember"
    if is_self_referential_value(value):
        if source == "auto":
            _quarantine(key, value, "self-referential value under a generic key")
        return False, "that describes me, not you — not something to remember about you"
    if _is_placeholder(value):
        if source == "auto":
            _quarantine(key, value, "placeholder value")
        return False, "placeholder value, nothing to store"

    path = PROFILE_FILE if normalized in PROFILE_KEYS else NOTES_FILE
    data = _load_json(path, {})
    existing = data.get(normalized)
    if existing == value:
        return False, "already stored"
    if existing and _is_placeholder(existing) is False and source == "auto" and len(value) < 2:
        # An auto-extracted value that's suspiciously short shouldn't clobber
        # a real existing value (e.g. don't let a guess overwrite a name).
        _quarantine(key, value, f"too short to override existing '{existing}'")
        return False, "kept existing value"

    data[normalized] = value
    _save_json(path, data)
    return True, "stored"


def load_sessions(n: int = SESSIONS_IN_PROMPT) -> list[str]:
    """The n most recent session summaries, oldest first."""
    sessions = _load_json(SESSIONS_FILE, [])
    return [s["summary"] for s in sessions[-n:]]


def add_session_summary(summary: str) -> None:
    summary = summary.strip()
    if not summary:
        return
    sessions = _load_json(SESSIONS_FILE, [])
    sessions.append({"ts": datetime.now().strftime("%Y-%m-%d %H:%M"), "summary": summary})
    _save_json(SESSIONS_FILE, sessions[-MAX_SESSIONS_KEPT:])


def load_lessons() -> list[str]:
    return _load_json(LESSONS_FILE, [])


def add_lesson(lesson: str) -> None:
    """Appends one operational lesson, deduplicated, capped at MAX_LESSONS
    (oldest dropped first)."""
    lesson = lesson.strip()
    if not lesson:
        return
    lessons = _load_json(LESSONS_FILE, [])
    if lesson in lessons:
        return
    lessons.append(lesson)
    _save_json(LESSONS_FILE, lessons[-MAX_LESSONS:])


def load_quarantine(n: int = 20) -> list[dict]:
    """Rejected extraction attempts, for Ray-mode debugging only — never
    injected into the prompt."""
    return _load_json(QUARANTINE_FILE, [])[-n:]


def clear_memory() -> None:
    for path in (PROFILE_FILE, NOTES_FILE, SESSIONS_FILE, LESSONS_FILE, QUARANTINE_FILE):
        if os.path.exists(path):
            os.remove(path)


_EXTRACT_SYSTEM_PROMPT = (
    "You extract durable facts ABOUT THE USER ONLY from one chat exchange — "
    "never facts about the assistant itself (its name, model, role, or "
    "purpose are off-limits, no matter how they're phrased), and never the "
    "exchange itself. Valid examples: the user's name, preferences, location, "
    "occupation, interests, plans, or corrections they made about THEMSELVES. "
    "INVALID, never do this: keys like 'user_message', 'assistant_reply', "
    "'user_said', 'conversation' — copying what either side said is not a "
    "fact. Output ONLY a flat JSON object of snake_case keys to short string "
    "values, e.g. {\"favorite_color\": \"blue\"}. If the exchange contains no "
    "durable fact about the user, output exactly NONE. Never invent facts, "
    "never echo the "
    "raw exchange back as a fact, never store facts about the assistant."
)


# Set TUFFY_NO_AUTO_MEMORY=1 to disable the reflection pass entirely — on a
# memory/thermal-constrained machine, this is one full extra LLM completion
# per turn, purely for a "nice to have." The 'remember' tool still works.
_AUTO_MEMORY_DISABLED = os.environ.get("TUFFY_NO_AUTO_MEMORY") == "1"

# Below this many words, an exchange essentially never contains a new
# personal fact ("hi", "thanks", "ok", tool-result acknowledgements) — skip
# the extraction completion outright instead of spending GPU time on turns
# that were always going to come back NONE.
_MIN_WORDS_FOR_EXTRACTION = 4


def extract_facts(complete_fn, user_text: str, assistant_text: str) -> dict:
    """Reflection pass: asks the model (via complete_fn, a create_chat_completion
    -compatible callable) whether the last exchange contained durable facts
    about the user, validates and stores any it finds, and returns the newly
    stored {key: value} pairs. Rejected candidates are quarantined, not
    silently dropped, so misbehavior stays inspectable.

    Skips the LLM call entirely (no cost) for disabled/trivially short turns
    — see _AUTO_MEMORY_DISABLED and _MIN_WORDS_FOR_EXTRACTION above.
    """
    if _AUTO_MEMORY_DISABLED or len(user_text.split()) < _MIN_WORDS_FOR_EXTRACTION:
        return {}
    try:
        result = complete_fn(
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"User said: {user_text}\nAssistant replied: {assistant_text[:400]}"},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        text = result["choices"][0]["message"]["content"].strip()
        if not text or text.upper().startswith("NONE"):
            return {}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        facts = json.loads(match.group(0))
        if not isinstance(facts, dict):
            return {}
        stored = {}
        for key, value in facts.items():
            if not isinstance(value, (str, int, float)):
                continue
            changed, _ = store_fact(str(key), str(value), source="auto")
            if changed:
                stored[_normalize_key(str(key))] = str(value).strip()
        return stored
    except Exception:
        return {}  # memory extraction must never break the chat


def summarize_session(complete_fn, messages: list) -> str:
    """Builds a 1-2 sentence summary of this session's conversation for
    episodic memory. Returns '' when there's nothing worth remembering."""
    user_turns = [
        m["content"] for m in messages
        if m["role"] == "user" and isinstance(m["content"], str)
    ]
    if len(user_turns) < 2:
        return ""
    try:
        transcript = "\n".join(f"- {t[:160]}" for t in user_turns[-12:])
        result = complete_fn(
            messages=[
                {"role": "system", "content": (
                    "Summarize what the user talked about and wanted in this chat "
                    "session in 1-2 short plain sentences, past tense. Output only "
                    "the summary."
                )},
                {"role": "user", "content": transcript},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        return result["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


@registry.register(
    name="remember",
    description="Persist a fact about the user to long-term memory, keyed by a short label. Use when the user explicitly asks you to remember something, or tells you a clearly important personal fact. Never call this to store facts about yourself (your own name/model/role) — that's fixed, not memory.",
    parameters={
        "key": {"type": "string", "description": "Short snake_case label for the fact, e.g. 'name', 'favorite_color'."},
        "value": {"type": "string", "description": "The fact to store."}
    },
    required=["key", "value"],
    group="memory",
)
def remember(key: str, value: str) -> str:
    changed, reason = store_fact(key, value, source="explicit")
    if changed:
        return f"Stored '{key}' = '{value}' in long-term memory."
    return f"Didn't store it: {reason}."
