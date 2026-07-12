"""Tuffy's entry point: wires up skills/MCP discovery, then hands off to the
CLI's input loop. See src/cli/ for the banner, commands, and turn loop —
this file only owns process startup/shutdown."""

import os
import sys
import traceback

from src.models import DEFAULT_MODEL
from src.models.registry import registry as model_registry
from src.skills.loader import discover_skills, mcp_configs_from_skills
from src.tools.mcp_client import connect_mcp_servers
import src.tools  # noqa: F401 - registers tools (editing/coding/research/system) as a side effect of import
import src.memory  # noqa: F401 - registers the 'remember' tool
import src.skills  # noqa: F401 - registers the read_skill tool

from src.cli.session import Session
from src.cli.commands import handle_command
from src.cli.turn import run_turn
from src.cli.display import print_logo, print_session_info, C_DIM, C_USER, C_RESET

# Scans ./.tuffy/skills/*/ and auto-imports each skill's tools.py before the
# first system prompt is built, so skill descriptions and skill-provided
# tools are both present from the very first turn.
discover_skills()

# Connects to any MCP servers configured in ./.tuffy/mcp.json (gitignored —
# see docs/configure-mcp.md) plus each loaded skill's own mcp.json, if any. A
# no-op when no servers are configured. Must run after discover_skills() (so
# skills' mcp.json files are known) and before the first system prompt is
# built, so MCP tools appear in TOOLS YOU CAN CALL from turn one.
connect_mcp_servers(extra_configs=mcp_configs_from_skills())


def main():
    print_logo()
    session = Session(DEFAULT_MODEL)
    import src.memory as memory
    from src.prompts import build_system_prompt
    memory.attach_llm(session.agent.complete)
    model_card = model_registry.get(session.current_model_id)
    static_tokens = len(build_system_prompt(model_card=model_card)) // 4
    memory.reconfigure_for_model(model_card, static_prompt_tokens=static_tokens)
    model_name = model_card["name"]
    print_session_info(model_name, session.agent.supports_vision)

    while True:
        try:
            user_input = input(f"{C_USER}You ❯{C_RESET} ")
        except (KeyboardInterrupt, EOFError):
            print()
            session.end()
            print(f"{C_DIM}Goodbye!{C_RESET}")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            result = handle_command(session, stripped)
            if result == "exit":
                session.end()
                print(f"{C_DIM}Goodbye!{C_RESET}")
                break
            if result == "handled":
                continue
            print(f"{C_DIM}Unknown command: {stripped}. Type /help for a list of commands.{C_RESET}\n")
            continue

        run_turn(session, user_input)


if __name__ == "__main__":
    # Exit via os._exit on every path — clean or crashed — so llama.cpp's
    # Metal backend never runs its C++ static destructors, which trip a
    # harmless-but-scary GGML_ASSERT backtrace during atexit teardown.
    # Everything real (model, mtmd context, KV cache) is freed by unload().
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        import src.memory as memory
        memory.mem.close()
        os._exit(exit_code)
