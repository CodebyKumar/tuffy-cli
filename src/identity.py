"""The agent's self-model: static facts about what Tuffy IS, owned entirely
by code and the active model card. This is deliberately NOT part of
long-term memory (data/memory/) — an LLM reflection pass must never write
here, because a small model cannot reliably tell "a fact about the user" from
"a fact I just said about myself" (it kept storing its own model name/role/
purpose into the user's profile). Identity is fixed; memory is learned.

src/prompts/templates.py's self_model() renders this into the system prompt.
"""

AGENT_NAME = "Tuffy"
AGENT_TAGLINE = "a local, tool-using AI agent"

# Keys that must never be written into user memory by the automatic fact
# extractor (src/memory.py's extract_facts) — these describe the agent
# itself, not the user, no matter what phrasing they show up under.
#
# Deliberately excludes bare "name"/"role"/"title": those are legitimate
# facts ABOUT THE USER too (the user's own name, job title, etc.), so a blanket
# ban would reject real profile data. The agent/assistant/ai/model-prefixed
# and framework/hardware-specific variants are unambiguous, so those are safe
# to always reject.
RESERVED_IDENTITY_KEYS = {
    "model", "hardware", "framework", "capabilities", "llm", "identity",
    "who_i_am", "agent_name", "agent_role", "agent_purpose", "agent_model",
    "assistant_name", "assistant_role", "assistant_model",
    "ai_name", "ai_model", "ai_role",
}

# Keys that are conversation-transcript shaped rather than fact shaped — a
# small extractor model sometimes echoes the raw exchange back as if the
# transcript itself were a "fact" ({"user_message": "...", "assistant_reply":
# "..."}). These describe an EXCHANGE, not a durable fact about the user, so
# they're rejected regardless of value.
_TRANSCRIPT_KEY_MARKERS = (
    "user", "assistant", "message", "reply", "replied", "response",
    "responded", "said", "says", "conversation", "exchange", "dialogue",
    "utterance",
)


def is_transcript_key(normalized_key: str) -> bool:
    parts = normalized_key.split("_")
    return any(part in _TRANSCRIPT_KEY_MARKERS for part in parts)

# Substrings that mark a VALUE as describing the agent rather than the user,
# regardless of which key it's filed under (a small model files "I'm a local
# AI agent" under plain "role", "purpose", "title" just as often as under an
# agent_-prefixed key).
_SELF_REFERENTIAL_VALUE_MARKERS = (
    "i'm tuffy", "i am tuffy", "local ai agent", "ai agent", "language model",
    "i run on", "running locally", "llama.cpp", "qwen", "large language model",
)


def is_self_referential_value(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _SELF_REFERENTIAL_VALUE_MARKERS)


def describe(model_card: dict) -> str:
    """One rendered block describing the agent for the system prompt."""
    caps = ", ".join(model_card["capabilities"])
    return (
        f"- You are {AGENT_NAME}, {AGENT_TAGLINE}.\n"
        f"- Currently running on: {model_card['name']} "
        f"({model_card['family']}, {model_card['parameters']} params, "
        f"{model_card['quantization']} quant, capabilities: {caps}), via llama.cpp.\n"
        "- This identity is fixed by your configuration, not something you or the user "
        "can 'remember' or change — never store your own name, model, or role as a "
        "fact about the user."
    )
