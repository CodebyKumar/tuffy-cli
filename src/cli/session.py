"""Owns the mutable state a running Tuffy session needs: the active
model/agent, chat history, pending image, and history-trimming rules. This
is the only place main.py's turn loop reaches into for agent state."""

import gc
import json
import os

from src.agent import LocalAgent
from src.prompts import build_system_prompt
from src.memory import summarize_session, add_session_summary
from src.models.registry import registry as model_registry
from src.cli.display import C_DIM, C_WARN, C_USER, C_BLUE, C_RESET, CLEAR_LINE

# Keep the rolling chat history well under n_ctx (4096 tokens) so the
# system prompt (rules + memory + protocol examples) never gets silently
# truncated by llama.cpp. This is a rough char-based proxy for token count.
MAX_HISTORY_CHARS = 6000


def _limits_line(model_card: dict) -> str:
    parts = []
    if model_card.get("context_length"):
        parts.append(f"context {model_card['context_length']:,} tok")
    limits = model_card.get("rate_limits") or {}
    if limits:
        parts.append(f"{limits['requests_per_minute']} req/min")
        parts.append(f"{limits['requests_per_day']:,} req/day")
        parts.append(f"{limits['tokens_per_minute']:,} tok/min")
        parts.append(f"{limits['tokens_per_day']:,} tok/day")
    return " | ".join(parts)


def load_agent(model_id: str) -> LocalAgent:
    model_card = model_registry.get(model_id)
    print(f"{C_DIM}Loading model '{model_id}'...{C_RESET}")
    agent = LocalAgent(model_card)
    vision = "vision on" if agent.supports_vision else "text only"
    print(f"{C_DIM}Ready: {model_card['name']} ({vision}){C_RESET}")
    limits_line = _limits_line(model_card)
    if limits_line:
        print(f"{C_DIM}{limits_line}{C_RESET}")
    if agent.vision_disabled_reason:
        print(f"{C_WARN}[Vision disabled] {agent.vision_disabled_reason}{C_RESET}")
    return agent


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


def estimate_tokens(messages: list) -> int:
    """Rough token count for the given messages, via the same chars/4 proxy
    trim_history uses for its budget. No provider here returns real usage
    counts mid-stream, so this is what /status and the model-load banner
    show as 'current context usage' — an estimate, not an exact count."""
    return sum(_content_char_count(m["content"]) for m in messages) // 4


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


class Session:
    """Owns the mutable state a running Tuffy session needs: the active
    model/agent, chat history, pending image, and the background memory worker."""

    def __init__(self, model_id: str):
        self.current_model_id = model_id
        self.agent = load_agent(model_id)
        self.pending_image_data_uri = None
        self.active_spinner = None
        self.captured_images = []
        self.trace_printed = False

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
        """Loads the requested model BEFORE unloading the current one, so a
        failed switch (e.g. an API model with a missing API key env var)
        leaves the session on its previous, working model instead of with no
        agent at all."""
        new_agent = load_agent(model_id)
        self.agent.unload()
        gc.collect()
        self.agent = new_agent
        self._wire_agent_callbacks()
        old_model_id = self.current_model_id
        self.current_model_id = model_id
        # Conversation history is kept (not reset) across a model switch, but
        # the new model needs to know a switch happened - otherwise it can
        # mistake a plain "hi" for a nudge to re-answer the last unresolved
        # question in the carried-over history instead of just greeting back.
        self.messages.append({
            "role": "system",
            "content": (
                f"[Model switched from '{old_model_id}' to '{model_id}'. "
                "Prior turns above are already answered - do not re-answer "
                "them unless the user explicitly asks again.]"
            ),
        })

    def end(self):
        """Writes this session into episodic memory, then frees the model."""
        summary = summarize_session(self.agent.complete, self.messages)
        if summary:
            add_session_summary(summary)
        self.agent.unload()
        gc.collect()

        # Delete any images captured in this session
        for path in self.captured_images:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"{C_DIM}Removed captured session image: {os.path.basename(path)}{C_RESET}")
            except OSError:
                pass
