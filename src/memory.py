"""Long-term memory adapter wrapping the Elastimem framework."""

import os
import re
import elastimem
from datetime import datetime
from src.identity import RESERVED_IDENTITY_KEYS
from src.tools.registry import registry

# Every model card's persona instructs the model to always open a reply with
# a <think>...</think> block (see personas.yaml) - a habit strong enough
# that the model applies it even to Elastimem's one-off background prompts
# (fact extraction, session/rolling summaries), which never ask for one and
# have no code path to strip it themselves. Left unstripped, raw reasoning
# text gets stored as if it were the actual output - e.g. a session summary
# starting with "<think>\nOkay, let's see..." - and then re-injected into
# future prompts verbatim. Mirrors the stripping src/agent.py's streaming
# path already does for the interactive chat answer.
#
# Background completions use a much tighter max_tokens than the interactive
# chat path (see ElastimemConfig.worker_max_tokens) - a model that rambles in
# its <think> block routinely gets cut off before ever emitting </think>, so
# a second pattern drops a trailing unclosed <think> (open tag with no
# matching close anywhere after it) rather than only ever matching complete
# open/close pairs.
_THINK_PATTERN = re.compile(r"<think>\s*.*?\s*</think>", re.DOTALL)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>.*$", re.DOTALL)

DB_DIR = "./data/memory"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "tuffy.db")

# Initialize singleton store. 4096 matches today's default local model
# (src/models/configs/local.py); attach_llm() below corrects this to the
# real model card the moment a session picks one, and switch_model() keeps
# it correct across model switches (see reconfigure_for_model()).
mem = elastimem.open(
    DB_PATH,
    context_tokens=4096,
    reserved_keys=frozenset(RESERVED_IDENTITY_KEYS),
)

def _adapt(agent_complete):
    def llm(prompt, *, max_tokens, temperature):
        r = agent_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        content = r["choices"][0]["message"]["content"] or ""
        content = _THINK_PATTERN.sub("", content)
        content = _UNCLOSED_THINK_PATTERN.sub("", content)
        return content.strip()
    return llm

_last_complete_fn = None    # remembered so clear_memory() can re-wire a fresh store
_last_model_card = None     # remembered so clear_memory() can re-apply real budgets
_last_static_prompt_tokens = None


def attach_llm(complete_fn):
    global _last_complete_fn
    _last_complete_fn = complete_fn
    if os.environ.get("TUFFY_NO_AUTO_MEMORY") == "1":
        return
    mem.complete_fn = _adapt(complete_fn)

def reconfigure_for_model(model_card: dict, static_prompt_tokens: int = None) -> None:
    """Rebuilds memory's token budgets for the given model card's real
    context length (and, when known, the fixed size of the persona/tools
    portion of the system prompt). Call this whenever the active model
    changes (initial load and every /models switch) - without it, budgets
    silently stay pinned to whatever context_tokens was set at import time
    (4096) even after switching to e.g. a 131k-context API model. See
    elastimem's docs/integration.md ("If your host can switch models
    mid-session") for why this isn't automatic.

    reprobe=True: the memory store is constructed at import time, before any
    model is chosen, so its startup RAM reading predates a local model
    claiming its share. Called here right after a model finishes loading
    (this is that exact moment for both the initial load in main.py and
    every /models switch), so the tier reflects real post-load memory
    pressure from the first turn - not a stale pre-load guess. No-op cost
    for API models (unloading/loading one changes credentials, not RAM), but
    harmless to call every time rather than branching on provider type."""
    global _last_model_card, _last_static_prompt_tokens
    _last_model_card = model_card
    _last_static_prompt_tokens = static_prompt_tokens
    context_length = model_card.get("context_length") or 4096
    overrides = {"context_tokens": context_length}
    if static_prompt_tokens is not None:
        overrides["static_prompt_tokens"] = static_prompt_tokens
    mem.reconfigure(reprobe=True, **overrides)

def store_fact(key: str, value: str, source: str = "explicit") -> tuple[bool, str]:
    return mem.remember(key, value, source)

def add_lesson(lesson: str) -> None:
    mem.add_lesson(lesson)

def clear_memory() -> None:
    """Closes, archives the DB file, and opens a fresh memory database."""
    global mem
    mem.close()
    if os.path.exists(DB_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(DB_DIR, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        archive_path = os.path.join(backup_dir, f"tuffy.db.{timestamp}.bak")
        try:
            os.rename(DB_PATH, archive_path)
            print(f"Archived memory database to {archive_path}")
        except Exception as e:
            print(f"Error archiving memory database: {e}")
            try:
                os.remove(DB_PATH)
            except Exception:
                pass
    # A fresh store starts back at the hardcoded 4096-token default budget
    # (same as import time) regardless of which model is actually active.
    # Without re-deriving real budgets below, a user on a large-context model
    # who runs /purge gets memory silently re-budgeted down to ~4K-context
    # math — facts/episodic/sessions/lessons sections all shrink to a
    # fraction of what they should be — until their next /models switch
    # happens to call reconfigure_for_model() again.
    context_tokens = (
        (_last_model_card.get("context_length") or 4096)
        if _last_model_card is not None else 4096
    )
    mem = elastimem.open(
        DB_PATH,
        context_tokens=context_tokens,
        reserved_keys=frozenset(RESERVED_IDENTITY_KEYS),
    )
    # clear_memory() swaps `mem` for a brand-new store with no complete_fn of
    # its own — without re-attaching here, every background job (fact
    # extraction, summaries) on the fresh store would silently run with no
    # LLM until the next model switch happened to call attach_llm() again.
    if _last_complete_fn is not None:
        attach_llm(_last_complete_fn)
    # Re-derive the rest of the budget (static_prompt_tokens, tier reprobe)
    # the same way a real model switch would, now that the new store exists
    # to reconfigure.
    if _last_model_card is not None:
        reconfigure_for_model(_last_model_card, static_prompt_tokens=_last_static_prompt_tokens)

# --- Register tools ---

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

@registry.register(
    name="recall",
    description=(
        "Search long-term memory (past conversations and stored facts) for something specific "
        "the user is asking you to remember or bring back up — e.g. 'what do you know about me', "
        "'did I mention my birthday', 'what did we talk about last time', or any question about "
        "something from an earlier session that isn't already visible in the current conversation. "
        "Don't call this for facts already stated earlier in THIS conversation (you can already see "
        "those) or for general world knowledge (use web_search for that instead)."
    ),
    parameters={
        "query": {"type": "string", "description": "What to search for, in the user's own words — short queries work fine."}
    },
    required=["query"],
    group="memory",
)
def recall(query: str) -> str:
    hits = mem.recall(query)
    if not hits:
        return "nothing found in long-term memory for that"
    lines = []
    for hit in hits:
        lines.append(f"- [{hit.date}] {hit.text}")
    return "\n".join(lines)
