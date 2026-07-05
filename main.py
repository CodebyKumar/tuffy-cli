"""Tuffy's CLI. Clean chat — a banner, an animated status line while
  the agent works, and the final answer. No tool JSON, no memory-write
  chatter, no raw observations.
"""

import itertools
import json
import os
import sys
import threading
import time
import traceback

from src.agent import LocalAgent
from src.prompts import build_system_prompt
from src.memory import clear_memory, load_memory, load_sessions, load_lessons, load_quarantine, extract_facts, summarize_session, add_session_summary
from src.tools.registry import registry
from src.models import DEFAULT_MODEL
from src.models.registry import registry as model_registry
from src.vision import encode_image_to_data_uri
import src.tools  # noqa: F401 - registers tools as a side effect of import
import src.workspace  # noqa: F401
import src.memory  # noqa: F401

# --- Layout constants -------------------------------------------------------
C_AI = "\033[38;2;0;255;180m"      # Mint
C_USER = "\033[96m"     # Cyan
C_DIM = "\033[2m"       # Faded gray
C_SUCCESS = "\033[92m"  # Green
C_WARN = "\033[93m"     # Yellow
C_BLUE = "\033[94m"     # Blue
C_BOLD = "\033[1m"
C_RESET = "\033[0m"
CLEAR_LINE = "\r\033[K"

BANNER = f"""{C_SUCCESS}{C_BOLD}
  ████████╗██╗   ██╗███████╗███████╗██╗   ██╗
  ╚══██╔══╝██║   ██║██╔════╝██╔════╝╚██╗ ██╔╝
     ██║   ██║   ██║█████╗  █████╗   ╚████╔╝
     ██║   ██║   ██║██╔══╝  ██╔══╝    ╚██╔╝
     ██║   ╚██████╔╝██║     ██║        ██║
     ╚═╝    ╚═════╝ ╚═╝     ╚═╝        ╚═╝
                      ⚡{C_RESET}"""

# Keep the rolling chat history well under n_ctx (4096 tokens) so the
# system prompt (rules + memory + protocol examples) never gets silently
# truncated by llama.cpp. This is a rough char-based proxy for token count.
MAX_HISTORY_CHARS = 6000


class Spinner:
    """Terminal spinner for AI status updates."""

    MAX_LABEL = 64

    def __init__(self, label: str = "thinking"):
        self.label = label
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def set_label(self, label: str):
        label = " ".join(str(label).split())

        if len(label) > self.MAX_LABEL:
            label = label[: self.MAX_LABEL - 1] + "…"

        with self._lock:
            self.label = label or "thinking"

    def start(self):
        if self._thread is not None:
            return

        self._stop_event.clear()

        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        def run():
            frames = ["", ".", "..", "..."]

            i = 0
            while not self._stop_event.is_set():
                with self._lock:
                    label = self.label

                print(
                    f"{CLEAR_LINE}{C_AI}AI ❯{C_RESET} "
                    f"{C_DIM}{label}{frames[i % len(frames)]}{C_RESET}",
                    end="",
                    flush=True,
                )

                i += 1
                time.sleep(0.4)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join()

        self._thread = None

        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

        print(
            f"{CLEAR_LINE}{C_AI}AI ❯{C_RESET} ",
            end="",
            flush=True,
        )

def keep_only_latest_image(messages: list) -> None:
    """Drops all but the most recent image from history, in place.

    llama.cpp's multimodal handler re-encodes every image still in the
    conversation on every single turn — with several images that gets slower
    each turn and eventually exhausts GPU memory. The newest image stays so
    follow-up questions about it keep working; older ones collapse to their
    text plus a note."""
    latest_seen = False
    for msg in reversed(messages):
        if not isinstance(msg.get("content"), list):
            continue
        if not latest_seen:
            latest_seen = True
            continue
        text = " ".join(
            part.get("text", "") for part in msg["content"] if part.get("type") == "text"
        )
        msg["content"] = f"{text}\n[An image was attached here earlier but has been removed from context.]"


def _content_char_count(content) -> int:
    """Message content is normally a string, but a message carrying an image
    (see LocalAgent.attach_image) has a multimodal content list instead - only
    the text parts count toward the char budget there."""
    if isinstance(content, str):
        return len(content)
    return sum(len(part.get("text", "")) for part in content if part.get("type") == "text")


def trim_history(messages: list) -> list:
    """Keeps the system prompt plus the most recent turns that fit the budget.
    The newest message is always kept even if it alone busts the budget —
    dropping the question the user just asked is never acceptable."""
    system_msg = messages[0]
    convo = messages[1:]

    kept = []
    total_chars = 0
    for msg in reversed(convo):
        total_chars += _content_char_count(msg["content"])
        if kept and total_chars > MAX_HISTORY_CHARS:
            break
        kept.append(msg)
    kept.reverse()

    return [system_msg] + kept


def compact_turn(messages: list, turn_start: int) -> None:
    """After a turn completes, drops its ReAct intermediates (tool-call
    drafts and observation messages) from history, keeping the user's
    question, any image-bearing message, and the final answer. The answer
    already contains what the observations contributed, so keeping raw tool
    dumps around only slows every later turn and bloats the context."""
    del_candidates = range(turn_start + 1, len(messages) - 1)
    keep = [
        i for i in del_candidates
        if isinstance(messages[i].get("content"), list)  # image messages stay
    ]
    messages[turn_start + 1:len(messages) - 1] = [messages[i] for i in keep]


def load_agent(model_id: str) -> LocalAgent:
    model_card = model_registry.get(model_id)
    print(f"{C_DIM}Loading model '{model_id}'...{C_RESET}")
    agent = LocalAgent(model_card)
    vision = "vision on" if agent.supports_vision else "text only"
    print(f"{C_DIM}Ready: {model_card['name']} ({vision}){C_RESET}")
    if agent.vision_disabled_reason:
        print(f"{C_WARN}[Vision disabled] {agent.vision_disabled_reason}{C_RESET}")
    return agent


class Session:
    """Owns the mutable state a running Tuffy session needs: the active
    model/agent, chat history, pending image, and the background memory worker."""

    def __init__(self, model_id: str):
        self.current_model_id = model_id
        self.agent = load_agent(model_id)
        self.pending_image_data_uri = None
        self.active_spinner = None
        self.captured_images = []

        self._wire_agent_callbacks()
        self.messages = [self.system_message()]

    # -- wiring between the agent's live signals and the active UI mode --
    def _wire_agent_callbacks(self):
        self.agent.status_cb = self._on_status
        self.agent.trace_cb = self._render_trace

    def _render_trace(self, event: str, data: dict):
        self.trace_printed = True
        spinner = self.active_spinner
        if spinner is not None:
            spinner.stop()

        if event == "tool_call":
            args_json = json.dumps(data["arguments"], ensure_ascii=False)
            if data.get("thought"):
                print(f"{CLEAR_LINE}{C_BLUE}[thoughts] {data['thought']}{C_RESET}", flush=True)
                print(f"{C_WARN}[tool_call] {data['name']}({args_json}){C_RESET}", flush=True)
            else:
                print(f"{CLEAR_LINE}{C_WARN}[tool_call] {data['name']}({args_json}){C_RESET}", flush=True)
        elif event == "tool_result":
            print(f"{CLEAR_LINE}{C_USER}[response] {data['result']}{C_RESET}", flush=True)
            if data.get("name") == "capture_image":
                res = data.get("result", "")
                prefix = "(image attached, saved at "
                if res.startswith(prefix) and res.endswith(")"):
                    path = res[len(prefix):-1]
                    if os.path.exists(path):
                        self.captured_images.append(path)

        if spinner is not None:
            spinner.start()

    def _on_status(self, label: str):
        if self.active_spinner is not None:
            self.active_spinner.set_label(label)

    def system_message(self) -> dict:
        return {
            "role": "system",
            "content": build_system_prompt(model_card=model_registry.get(self.current_model_id)),
        }

    def switch_model(self, model_id: str):
        self.agent.unload()
        import gc
        gc.collect()
        self.agent = load_agent(model_id)
        self._wire_agent_callbacks()
        self.current_model_id = model_id

    def end(self):
        """Writes this session into episodic memory, then frees the model."""
        summary = summarize_session(self.agent.complete, self.messages)
        if summary:
            add_session_summary(summary)
        self.agent.unload()
        import gc
        gc.collect()

        # Delete any images captured in this session
        for path in self.captured_images:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"{C_DIM}Removed captured session image: {os.path.basename(path)}{C_RESET}")
            except Exception:
                pass


# --- Slash commands (shared by both modes) ----------------------------------

def cmd_help(session: Session):
    print(
        f"{C_DIM}Available commands:\n"
        "  /memory        - show everything in long-term memory (facts, sessions, lessons)\n"
        "  /clear         - wipe long-term memory and conversation history\n"
        "  /tools         - list all tools the agent can call and what they do\n"
        "  /models        - list available models and show which one is active\n"
        "  /models <id>   - switch to a different model, unloading the current one\n"
        "  /models info <id> - show a model's full model card\n"
        "  /image <path>  - attach an image file to your next message (requires a vision-capable model)\n"
        "  /exit, /quit   - save session memory and terminate\n"
        "  /help          - show this message\n"
        f"{C_RESET}"
    )


def cmd_memory(session: Session):
    facts = load_memory()
    sessions = load_sessions(n=5)
    lessons = load_lessons()
    print(f"{C_SUCCESS}Long-term memory{C_RESET}")
    print(f"{C_DIM}Facts about you:{C_RESET}")
    if facts:
        for k, v in facts.items():
            print(f"  {k}: {v}")
    else:
        print("  (none yet — they accumulate as you chat)")
    if sessions:
        print(f"{C_DIM}Recent sessions:{C_RESET}")
        for s in sessions:
            print(f"  - {s}")
    if lessons:
        print(f"{C_DIM}Lessons learned:{C_RESET}")
        for l in lessons:
            print(f"  - {l}")
    print()


def cmd_models(session: Session, arg: str):
    if not arg:
        print(f"{C_DIM}Available models:{C_RESET}")
        for model_id in model_registry.list_ids():
            card = model_registry.get(model_id)
            marker = f"{C_SUCCESS}*{C_RESET}" if model_id == session.current_model_id else " "
            caps = ", ".join(card["capabilities"])
            print(f"  {marker} {model_id} - {card['name']} [{caps}]")
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
        for field in ("id", "family", "parameters", "quantization", "context_length"):
            print(f"  {field:14}: {card[field]}")
        print(f"  capabilities  : {', '.join(card['capabilities'])}")
        for field in ("license", "source", "path", "description"):
            print(f"  {field:14}: {card[field]}")
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

    session.switch_model(requested_id)
    print(f"{C_SUCCESS}Switched to model '{requested_id}'.{C_RESET}\n")


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


def cmd_tools(session: Session):
    print(f"{C_DIM}Available tools:{C_RESET}")
    for schema in registry.schemas:
        fn = schema["function"]
        print(f"  {C_SUCCESS}{fn['name']}{C_RESET} - {fn['description']}")
    print()


def handle_command(session: Session, stripped: str) -> str:
    """Returns 'exit', 'handled', or 'unhandled' (caller decides what to do
    with an unrecognized command)."""
    command = stripped.lower()

    if command in ("/exit", "/quit"):
        return "exit"

    if command == "/clear":
        clear_memory()
        session.messages = [session.system_message()]
        print(f"{C_SUCCESS}Memory and conversation history cleared.{C_RESET}\n")
        return "handled"

    if command == "/help":
        cmd_help(session)
        return "handled"

    if command == "/memory":
        cmd_memory(session)
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

    return "unhandled"


# --- The turn loop, shared by both modes ------------------------------------

def run_turn(session: Session, user_input: str) -> bool:
    """Runs one user turn to completion. Returns False if generation was
    interrupted/failed and the turn was rolled back, True otherwise."""
    messages = session.messages
    messages[0] = session.system_message()
    user_message = {"role": "user", "content": user_input}
    if session.pending_image_data_uri:
        user_message = session.agent.attach_image(user_message, session.pending_image_data_uri)
        session.pending_image_data_uri = None
    messages.append(user_message)
    keep_only_latest_image(messages)
    session.messages = messages = trim_history(messages)
    turn_start = len(messages) - 1

    session.trace_printed = False
    print()  # Line gap between "You ❯" and thoughts/spinner

    spinner = Spinner()
    session.active_spinner = spinner
    spinner.start()
    token_stream = session.agent.run_stream(messages)

    full_response = ""
    try:
        first_token = True
        for token in token_stream:
            # First stop() erases the status text and leaves 'AI ❯ ' in
            # place; later calls are no-ops and never touch the screen.
            if first_token:
                first_token = False
                if session.trace_printed:
                    print(CLEAR_LINE, flush=True)
                spinner.stop()
            else:
                spinner.stop()
            print(token, end="", flush=True)
            full_response += token
    except RuntimeError as e:
        # llama.cpp returns decode errors (e.g. -3) under memory pressure;
        # drop this turn instead of crashing the whole session.
        spinner.stop()
        print(
            f"{C_DIM}[Generation failed: {e}. The machine is likely low on "
            f"memory — close some apps and try again.]{C_RESET}\n"
        )
        del messages[turn_start:]
        return False
    except KeyboardInterrupt:
        spinner.stop()
        print(f"{C_DIM}[Interrupted]{C_RESET}\n")
        del messages[turn_start:]
        return False
    finally:
        spinner.stop()
        session.active_spinner = None
    print("\n")

    messages.append({"role": "assistant", "content": full_response})
    # Drop this turn's ReAct intermediates (tool drafts/observations) — the
    # final answer carries what mattered, and stale tool dumps slow every
    # later turn. Ray mode already showed them live; nothing is lost.
    compact_turn(messages, turn_start)

    # Reflection pass runs synchronously.
    extract_facts(session.agent.complete, user_input, full_response)
    return True


def print_banner(session: Session):
    print(BANNER)
    print(f"{C_DIM}Local AI agent. /help for commands.{C_RESET}\n")


def main():
    session = Session(DEFAULT_MODEL)

    print_banner(session)

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
        os._exit(exit_code)
