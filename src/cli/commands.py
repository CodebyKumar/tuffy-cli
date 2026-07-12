"""Slash commands: one function per command, plus the dispatch table that
maps a typed command to its handler. Registering a new command means adding
a COMMANDS entry and a cmd_* function — main.py's input loop never needs to
change."""

import os

import src.memory as memory
from src.tools.registry import registry, group_title
from src.tools.mcp_client import connected_servers, MCP_CONFIG_PATH
from src.models.registry import registry as model_registry
from src.skills.loader import list_skills
from src.vision import encode_image_to_data_uri
from src.cli.session import Session, estimate_tokens
from src.cli.display import C_DIM, C_SUCCESS, C_WARN, C_BLUE, C_BOLD, C_RESET

# --- /help -------------------------------------------------------------

_HELP_SECTIONS = [
    ("Chat", [
        ("/new", "start a fresh conversation (keeps long-term memory)"),
        ("/clear", "wipe current conversation history (keeps long-term memory)"),
        ("/purge", "wipe long-term memory database (archives old DB)"),
        ("/image <path>", "attach an image to your next message (vision models only)"),
    ]),
    ("Inspect", [
        ("/memory", "show summary, facts, sessions, and lessons from memory"),
        ("/memory search <query>", "search past conversations and facts"),
        ("/memory facts <key>", "show all version history of a specific fact key"),
        ("/memory forget <key>", "forget a specific fact key (non-destructive tombstone)"),
        ("/memory quarantine", "show recently rejected/quarantined auto-extractions"),
        ("/tools", "list every tool the agent can call, grouped by domain"),
        ("/skills", "list installed skills (drop new ones in ./.tuffy/skills/<name>/)"),
        ("/mcp", "list connected MCP servers and the tools they registered"),
        ("/status", "show the active model, vision support, and turn count"),
    ]),
    ("Models", [
        ("/models", "list available models (local + API) and the active one"),
        ("/models <id>", "switch to a different model, unloading the current one"),
        ("/models info <id>", "show a model's full model card"),
    ]),
    ("Session", [
        ("/help", "show this message"),
        ("/exit, /quit", "save session memory and terminate"),
    ]),
]


def cmd_help(session: Session):
    print(f"{C_DIM}Commands (type any of these, or just chat normally):{C_RESET}")
    for title, rows in _HELP_SECTIONS:
        print(f"\n{C_BOLD}{C_BLUE}{title}{C_RESET}")
        width = max(len(name) for name, _ in rows)
        for name, desc in rows:
            print(f"  {C_SUCCESS}{name.ljust(width)}{C_RESET}  {desc}")
    print()


# --- /new / /clear -------------------------------------------------------

def cmd_new(session: Session):
    session.messages = [session.system_message()]
    print(f"{C_SUCCESS}Started a new conversation. Long-term memory is unchanged.{C_RESET}\n")


def cmd_clear(session: Session):
    session.messages = [session.system_message()]
    print(f"{C_SUCCESS}Conversation history cleared. Long-term memory is unchanged.{C_RESET}\n")


def cmd_purge(session: Session):
    memory.clear_memory()
    session.messages = [session.system_message()]
    print(f"{C_SUCCESS}Memory database cleared and archived. Conversation history reset.{C_RESET}\n")


# --- /memory -------------------------------------------------------------

def cmd_memory(session: Session, arg: str = ""):
    arg = arg.strip()
    if not arg:
        stats = memory.mem.stats()
        facts = memory.mem.facts()
        sessions = memory.mem.sessions(n=5)
        lessons = memory.mem.lessons()
        
        print(f"{C_SUCCESS}Long-term memory status & contents{C_RESET}")
        profile = memory.mem.profile
        print(f"  {C_DIM}Tier:{C_RESET} {profile.tier.name} | {C_DIM}Facts:{C_RESET} {stats.get('facts', 0)} | {C_DIM}Sessions:{C_RESET} {stats.get('sessions', 0)} | {C_DIM}Lessons:{C_RESET} {stats.get('lessons', 0)}")
        print(f"  {C_DIM}Database file:{C_RESET} {stats.get('path')} ({stats.get('db_bytes', 0) / 1024:.1f} KB)")
        
        print(f"\n{C_DIM}Facts about you:{C_RESET}")
        if facts:
            for k, v in facts.items():
                print(f"  {k}: {v}")
        else:
            print("  (none yet — they accumulate as you chat)")
            
        if sessions:
            print(f"\n{C_DIM}Recent sessions:{C_RESET}")
            for s in reversed(sessions):
                summary = s.get("summary")
                if summary:
                    print(f"  - [{s.get('ended_at') or 'active'}] {summary}")
                    
        if lessons:
            print(f"\n{C_DIM}Lessons learned:{C_RESET}")
            for l in lessons:
                print(f"  - {l}")
        print()
        return

    parts = arg.split(" ", 1)
    subcmd = parts[0].lower()
    subarg = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "search":
        if not subarg:
            print(f"{C_WARN}Usage: /memory search <query>{C_RESET}\n")
            return
        hits = memory.mem.recall(subarg)
        print(f"{C_SUCCESS}Search results for '{subarg}':{C_RESET}")
        if not hits:
            print("  No relevant memories found.")
        for hit in hits:
            print(f"  - [{hit.date}] [{hit.kind}] {hit.text} (score: {hit.score:.2f})")
        print()
        
    elif subcmd == "facts":
        if not subarg:
            print(f"{C_WARN}Usage: /memory facts <key>{C_RESET}\n")
            return
        history = memory.mem.fact_history(subarg)
        print(f"{C_SUCCESS}Version history for fact '{subarg}':{C_RESET}")
        if not history:
            print(f"  No fact found with key '{subarg}'.")
        for fact in history:
            status = "archived" if fact.archived else ("invalidated" if fact.invalidated_at else "active")
            print(f"  - [{fact.valid_from}] {fact.value} ({status})")
        print()
        
    elif subcmd == "forget":
        if not subarg:
            print(f"{C_WARN}Usage: /memory forget <key>{C_RESET}\n")
            return
        found = memory.mem.forget(subarg)
        if found:
            print(f"{C_SUCCESS}Fact '{subarg}' has been tombstoned (forgotten).{C_RESET}\n")
        else:
            print(f"{C_WARN}Fact '{subarg}' was not found.{C_RESET}\n")
            
    elif subcmd == "quarantine":
        entries = memory.mem.quarantine_entries(20)
        print(f"{C_SUCCESS}Recently rejected/quarantined extractions:{C_RESET}")
        if not entries:
            print("  No quarantined entries.")
        for entry in entries:
            print(f"  - [{entry.get('ts')}] '{entry.get('key')}': '{entry.get('value')}' - {entry.get('reason')} (source: {entry.get('source')})")
        print()
    else:
        print(f"{C_WARN}Unknown memory subcommand: {subcmd}. Use search, facts, forget, or quarantine.{C_RESET}\n")


# --- /models ---------------------------------------------------------------

def cmd_models(session: Session, arg: str):
    if not arg:
        print(f"{C_DIM}Available models:{C_RESET}")
        for model_id in model_registry.list_ids():
            card = model_registry.get(model_id)
            marker = f"{C_SUCCESS}*{C_RESET}" if model_id == session.current_model_id else " "
            caps = ", ".join(card["capabilities"])
            provider = card["provider"]
            tag = "local" if provider == "llama_cpp" else f"api: {provider}"
            print(f"  {marker} {model_id} - {card['name']} [{caps}] ({tag})")
        print(f"{C_DIM}  Use '/models info <id>' for full model card, '/models <id>' to switch.{C_RESET}\n")
        return

    if arg.lower().startswith("info "):
        requested_id = arg[len("info "):].strip()
        try:
            card = model_registry.get(requested_id)
        except ValueError as e:
            print(f"{C_DIM}{e}{C_RESET}\n")
            return
        print(f"{C_SUCCESS}{card['name']}{C_RESET}")
        for field in ("id", "family", "parameters", "quantization", "context_length", "provider"):
            print(f"  {field:14}: {card[field]}")
        print(f"  capabilities  : {', '.join(card['capabilities'])}")
        for field in ("license", "source", "description"):
            print(f"  {field:14}: {card[field]}")
        if card["provider"] == "llama_cpp":
            print(f"  {'path':14}: {card['path']}")
        else:
            cfg = card["provider_config"]
            print(f"  {'base_url':14}: {cfg['base_url']}")
            print(f"  {'model_name':14}: {cfg['model_name']}")
            print(f"  {'api_key_env':14}: {cfg['api_key_env']} ({'set' if os.environ.get(cfg['api_key_env']) else 'NOT SET'})")
            limits = card.get("rate_limits") or {}
            if limits:
                print(f"  {'rate limits':14}: {limits['requests_per_minute']} req/min, {limits['requests_per_day']:,} req/day, "
                      f"{limits['tokens_per_minute']:,} tok/min, {limits['tokens_per_day']:,} tok/day")
        print()
        return

    requested_id = arg
    if requested_id == session.current_model_id:
        print(f"{C_DIM}Model '{requested_id}' is already active.{C_RESET}\n")
        return
    try:
        model_registry.get(requested_id)
    except ValueError as e:
        print(f"{C_DIM}{e}{C_RESET}\n")
        return

    try:
        session.switch_model(requested_id)
    except Exception as e:
        print(f"{C_WARN}Couldn't switch to '{requested_id}': {e}{C_RESET}")
        print(f"{C_DIM}Staying on '{session.current_model_id}'.{C_RESET}\n")
        return
    print(f"{C_SUCCESS}Switched to model '{requested_id}'.{C_RESET}\n")


# --- /image ------------------------------------------------------------

def cmd_image(session: Session, image_path: str):
    if not image_path:
        print(f"{C_DIM}Usage: /image <path-to-image>{C_RESET}\n")
        return
    if not session.agent.supports_vision:
        print(f"{C_DIM}Model '{session.current_model_id}' has no vision capability. Switch models with /models <id>.{C_RESET}\n")
        return
    try:
        session.pending_image_data_uri = encode_image_to_data_uri(image_path)
    except PermissionError:
        print(
            f"{C_DIM}macOS blocked access to '{image_path}'. Grant your terminal "
            f"access in System Settings → Privacy & Security → Files and Folders "
            f"(or Full Disk Access), or move the file somewhere accessible.{C_RESET}\n"
        )
        return
    except (ValueError, OSError) as e:
        print(f"{C_DIM}Couldn't load image: {e}{C_RESET}\n")
        return
    print(f"{C_SUCCESS}Image loaded — it will be attached to your next message.{C_RESET}\n")


# --- /tools / /skills ------------------------------------------------------

def cmd_tools(session: Session):
    print(f"{C_DIM}Available tools:{C_RESET}")
    for group, schemas in registry.tools_by_group():
        title = group_title(group)
        print(f"\n{C_BOLD}{C_BLUE}== {title} =={C_RESET}")
        for schema in schemas:
            fn = schema["function"]
            print(f"  {C_SUCCESS}{fn['name']}{C_RESET} - {fn['description']}")
    print()


def cmd_skills(session: Session):
    skills = list_skills()
    if not skills:
        print(f"{C_DIM}No skills installed. Drop a folder with a SKILL.md into ./.tuffy/skills/<name>/ and restart.{C_RESET}\n")
        return
    print(f"{C_DIM}Installed skills:{C_RESET}")
    for name, info in skills.items():
        print(f"  {C_SUCCESS}{name}{C_RESET} - {info['description']}")
    print()


# --- /mcp --------------------------------------------------------------

def cmd_mcp(session: Session):
    servers = connected_servers()
    if not servers:
        print(
            f"{C_DIM}No MCP servers connected. Create {MCP_CONFIG_PATH} to configure some — "
            f"see docs/configure-mcp.md for the config shape and examples.{C_RESET}\n"
        )
        return

    print(f"{C_DIM}Connected MCP servers:{C_RESET}")
    for group, schemas in registry.tools_by_group():
        if not group.startswith("mcp:"):
            continue
        server_name = group[len("mcp:"):]
        print(f"\n{C_BOLD}{C_BLUE}{server_name}{C_RESET}  ({len(schemas)} tool(s))")
        for schema in schemas:
            fn = schema["function"]
            print(f"  {C_SUCCESS}{fn['name']}{C_RESET} - {fn['description']}")
    print()


# --- /status -------------------------------------------------------------

def cmd_status(session: Session):
    card = model_registry.get(session.current_model_id)
    vision = "yes" if session.agent.supports_vision else "no"
    turns = sum(1 for m in session.messages if m["role"] == "user")
    used_tokens = estimate_tokens(session.messages)
    context_length = card.get("context_length")
    print(f"{C_SUCCESS}Session status{C_RESET}")
    print(f"  model       : {card['name']} ({session.current_model_id})")
    print(f"  provider    : {card['provider']}")
    print(f"  vision      : {vision}")
    print(f"  turns so far: {turns}")
    print(f"  pending img : {'yes' if session.pending_image_data_uri else 'no'}")
    if context_length:
        pct = used_tokens / context_length * 100
        print(f"  context used: ~{used_tokens:,} / {context_length:,} tok ({pct:.1f}%, estimated)")
    else:
        print(f"  context used: ~{used_tokens:,} tok (estimated; model has no declared max)")
    limits = card.get("rate_limits") or {}
    if limits:
        print(f"  rate limits : {limits['requests_per_minute']} req/min, {limits['requests_per_day']:,} req/day, "
              f"{limits['tokens_per_minute']:,} tok/min, {limits['tokens_per_day']:,} tok/day")
    print()


# --- dispatch ----------------------------------------------------------

def handle_command(session: Session, stripped: str) -> str:
    """Returns 'exit', 'handled', or 'unhandled' (caller decides what to do
    with an unrecognized command)."""
    command = stripped.lower()

    if command in ("/exit", "/quit"):
        return "exit"

    if command == "/new":
        cmd_new(session)
        return "handled"

    if command == "/clear":
        cmd_clear(session)
        return "handled"

    if command == "/help":
        cmd_help(session)
        return "handled"

    if command == "/purge":
        cmd_purge(session)
        return "handled"

    if command == "/memory" or stripped.lower().startswith("/memory "):
        cmd_memory(session, stripped[len("/memory"):].strip())
        return "handled"

    if command == "/status":
        cmd_status(session)
        return "handled"

    if command == "/models" or command.startswith("/models "):
        cmd_models(session, stripped[len("/models"):].strip())
        return "handled"

    if command == "/image" or command.startswith("/image "):
        cmd_image(session, stripped[len("/image"):].strip())
        return "handled"

    if command == "/tools":
        cmd_tools(session)
        return "handled"

    if command == "/skills":
        cmd_skills(session)
        return "handled"

    if command == "/mcp":
        cmd_mcp(session)
        return "handled"

    return "unhandled"
