"""Long-term memory adapter wrapping the Elastimem framework."""

import os
import json
import shutil
import elastimem
from datetime import datetime
from src.identity import RESERVED_IDENTITY_KEYS
from src.tools.registry import registry

DB_DIR = "./data/memory"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "tuffy.db")

is_new_db = not os.path.exists(DB_PATH)

def _migrate_legacy_json(mem_store):
    legacy_dir = os.path.join(DB_DIR, "legacy-json.bak")
    profile_path = os.path.join(DB_DIR, "user", "profile.json")
    notes_path = os.path.join(DB_DIR, "user", "notes.json")
    lessons_path = os.path.join(DB_DIR, "agent", "lessons.json")
    quarantine_path = os.path.join(DB_DIR, "quarantine.json")
    sessions_path = os.path.join(DB_DIR, "sessions", "log.json")
    
    # Check if we have legacy data to migrate
    has_legacy = any(os.path.exists(p) for p in (profile_path, notes_path, lessons_path, quarantine_path, sessions_path))
    if not has_legacy:
        return
        
    os.makedirs(legacy_dir, exist_ok=True)
    
    # 1. Profile facts
    if os.path.exists(profile_path):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        mem_store.remember(k, str(v), source="import")
        except Exception as e:
            print(f"Error migrating profile: {e}")
            
    # 2. Notes facts
    if os.path.exists(notes_path):
        try:
            with open(notes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        mem_store.remember(k, str(v), source="import")
        except Exception as e:
            print(f"Error migrating notes: {e}")
            
    # 3. Lessons
    if os.path.exists(lessons_path):
        try:
            with open(lessons_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for lesson in data:
                        mem_store.add_lesson(str(lesson))
        except Exception as e:
            print(f"Error migrating lessons: {e}")

    # Move files to backup
    for p in (profile_path, notes_path, lessons_path, quarantine_path, sessions_path):
        if os.path.exists(p):
            try:
                dest = os.path.join(legacy_dir, os.path.basename(p))
                if os.path.dirname(p).endswith("user") or os.path.dirname(p).endswith("agent") or os.path.dirname(p).endswith("sessions"):
                    subdir = os.path.basename(os.path.dirname(p))
                    dest = os.path.join(legacy_dir, f"{subdir}_{os.path.basename(p)}")
                shutil.move(p, dest)
            except Exception as e:
                print(f"Error moving {p} to backup: {e}")

# Initialize singleton store. 4096 matches today's default local model
# (src/models/configs/local.py); attach_llm() below corrects this to the
# real model card the moment a session picks one, and switch_model() keeps
# it correct across model switches (see reconfigure_for_model()).
mem = elastimem.open(
    DB_PATH,
    context_tokens=4096,
    reserved_keys=frozenset(RESERVED_IDENTITY_KEYS),
)

if is_new_db:
    _migrate_legacy_json(mem)

def _adapt(agent_complete):
    def llm(prompt, *, max_tokens, temperature):
        r = agent_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return r["choices"][0]["message"]["content"]
    return llm

def attach_llm(complete_fn):
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
    mid-session") for why this isn't automatic."""
    context_length = model_card.get("context_length") or 4096
    overrides = {"context_tokens": context_length}
    if static_prompt_tokens is not None:
        overrides["static_prompt_tokens"] = static_prompt_tokens
    mem.reconfigure(**overrides)

# --- Re-export legacy API ---

def load_memory() -> dict:
    return mem.facts()

def store_fact(key: str, value: str, source: str = "explicit") -> tuple[bool, str]:
    return mem.remember(key, value, source)

def load_sessions(n: int = 3) -> list[str]:
    # Recent session summaries, oldest first
    return [s["summary"] for s in mem.sessions(n) if s.get("summary")][::-1]

def add_session_summary(summary: str) -> None:
    # No-op since session summaries are now generated by mem.end_session()
    pass

def load_lessons() -> list[str]:
    return mem.lessons()

def add_lesson(lesson: str) -> None:
    mem.add_lesson(lesson)

def load_quarantine(n: int = 20) -> list[dict]:
    return mem.quarantine_entries(n)

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
    mem = elastimem.open(
        DB_PATH,
        context_tokens=4096,
        reserved_keys=frozenset(RESERVED_IDENTITY_KEYS),
    )

def extract_facts(complete_fn, user_text: str, assistant_text: str) -> dict:
    # No-op: handled in background by record_turn
    return {}

def summarize_session(complete_fn, messages: list) -> str:
    # No-op: handled by end_session
    return ""

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
    name="memory_search",
    description="Search past conversations and facts in long-term memory for relevant information.",
    parameters={
        "query": {"type": "string", "description": "Search query."}
    },
    required=["query"],
    group="memory",
)
def memory_search(query: str) -> str:
    hits = mem.recall(query)
    if not hits:
        return "nothing found"
    lines = []
    for hit in hits:
        lines.append(f"- [{hit.date}] {hit.text}")
    return "\n".join(lines)
