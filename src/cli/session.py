"""Owns the mutable state a running Tuffy session needs: the active
model/agent, chat history, pending image, and history-trimming rules. This
is the only place main.py's turn loop reaches into for agent state."""

import gc
import os
from collections import deque

from src.engine.model_agent import ModelAgent
from src.prompts import build_system_prompt
from src.models.registry import registry as model_registry
from src.cli.display import C_DIM, C_WARN, C_RESET

_HEALTH_WINDOW = 10
_CONSECUTIVE_FAILURE_NUDGE_THRESHOLD = 3


class TurnHealth:
    """Passive rolling record of how the last few turns against the CURRENT
    model went - fed from real Done/Failed outcomes turn.py already sees, no
    extra model calls or background threads. llama.cpp's model object isn't
    thread-safe (a prior bug here came from a background worker racing a
    foreground generation on the same instance - see memory/
    tuffy-elastimem-integration.md), so this only ever reads events the
    normal turn loop already produced instead of polling the model."""

    def __init__(self):
        self._outcomes: deque[str] = deque(maxlen=_HEALTH_WINDOW)  # "ok" | "oom" | "provider" | "interrupted" | other kind

    def record(self, failed_kind: str | None):
        self._outcomes.append(failed_kind or "ok")

    def reset(self):
        self._outcomes.clear()

    def consecutive_failures(self) -> int:
        n = 0
        for outcome in reversed(self._outcomes):
            if outcome == "ok":
                break
            n += 1
        return n

    def should_nudge(self) -> bool:
        return self.consecutive_failures() >= _CONSECUTIVE_FAILURE_NUDGE_THRESHOLD

    def recent_outcomes(self) -> list[str]:
        """Public read of the rolling outcome window, oldest first - for
        callers (e.g. tuffy.AgentSession.status()) that want the raw list
        rather than summary()'s formatted string."""
        return list(self._outcomes)

    def summary(self) -> str:
        if not self._outcomes:
            return "no turns yet"
        ok = sum(1 for o in self._outcomes if o == "ok")
        total = len(self._outcomes)
        tail = self.consecutive_failures()
        line = f"{ok}/{total} of last turns succeeded"
        if tail:
            recent_kind = self._outcomes[-1]
            line += f" ({tail} failure(s) in a row, most recent: {recent_kind})"
        return line

# Prefix marking a model-switch notice injected into history (see
# Session.switch_model) so a later switch can find and replace it instead of
# appending a new one every time.
_MODEL_SWITCH_TAG = "[Model switched from "


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


def load_agent(model_id: str) -> ModelAgent:
    model_card = model_registry.get(model_id)
    print(f"{C_DIM}Loading model '{model_id}'...{C_RESET}")
    agent = ModelAgent(model_card)
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
    (see ModelAgent.attach_image) has a multimodal content list instead - only
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


def trim_history(messages: list, plan) -> list:
    """Keeps the system prompt, plus the newest user message, and the most
    recent user/assistant pairs within plan.keep_last_n_turns. Old turns are
    evicted and reported to elastimem, unless they are image-bearing or the newest user msg."""
    if len(messages) <= 2:
        return messages

    system_msg = messages[0]
    # The last message is the new user query. It must never be evicted.
    newest_user_msg = messages[-1]
    
    # Completed turns are between system message (index 0) and the newest user message (last index).
    convo = messages[1:-1]
    
    # Let's find all completed user/assistant pairs, scanning backwards.
    pairs = []
    i = len(messages) - 2  # start at the last assistant reply (since -1 is the new user query)
    while i > 0:
        if messages[i].get("role") == "assistant" and i - 1 > 0 and messages[i-1].get("role") == "user":
            pairs.append((i - 1, i))
            i -= 2
        else:
            i -= 1

    # Keep the newest plan.keep_last_n_turns pairs
    keep_n = getattr(plan, "keep_last_n_turns", 3)
    pairs_to_keep = pairs[:keep_n]
    pairs_to_evict = pairs[keep_n:]

    indices_to_remove = set()
    evicted_pairs = []

    for u_idx, a_idx in pairs_to_evict:
        u_msg = messages[u_idx]
        a_msg = messages[a_idx]
        # Never evict if either message contains an image
        if isinstance(u_msg.get("content"), list) or isinstance(a_msg.get("content"), list):
            continue
        
        evicted_pairs.append((u_msg["content"], a_msg["content"]))
        indices_to_remove.add(u_idx)
        indices_to_remove.add(a_idx)

    if evicted_pairs:
        # Reverse to report in chronological order (oldest first)
        evicted_pairs.reverse()
        import src.memory as memory
        memory.mem.report_evictions(evicted_pairs)

    # Reconstruct the chat history, filtering out the evicted indices
    new_convo = [msg for idx, msg in enumerate(messages) if idx != 0 and idx != len(messages) - 1 and idx not in indices_to_remove]
    
    return [system_msg] + new_convo + [newest_user_msg]


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
    model/agent, chat history, pending image, and the background memory
    worker. Display concerns (spinner, trace rendering) live entirely in
    src/cli/turn.py, which drives src.engine.turn_engine.run_turn and reacts
    to its events — this class no longer wires any callback into the agent."""

    def __init__(self, model_id: str):
        self.current_model_id = model_id
        self.agent = load_agent(model_id)
        self.pending_image_data_uri = None
        self.captured_images = []
        self.health = TurnHealth()

        self.messages = [self.system_message()]

    def note_captured_image(self, tool_name: str, result_text: str):
        """Called by the turn runner when a tool result comes back, so a
        captured photo (e.g. capture_image) gets cleaned up at session end
        the same as before — just driven by an explicit call instead of
        string-sniffing inside a trace callback."""
        if tool_name != "capture_image":
            return
        marker = "(image attached, saved at "
        if result_text.startswith(marker) and result_text.endswith(")"):
            path = result_text[len(marker):-1]
            if os.path.exists(path):
                self.captured_images.append(path)

    def system_message(self, context_plan=None) -> dict:
        return {
            "role": "system",
            "content": build_system_prompt(
                model_card=model_registry.get(self.current_model_id),
                context_plan=context_plan
            ),
        }

    def switch_model(self, model_id: str):
        """Loads the requested model BEFORE unloading the current one, so a
        failed switch (e.g. an API model with a missing API key env var)
        leaves the session on its previous, working model instead of with no
        agent at all."""
        new_agent = load_agent(model_id)
        import src.memory as memory
        memory.mem.drain()
        self.agent.unload()
        gc.collect()
        self.agent = new_agent
        memory.attach_llm(self.agent.complete)
        new_card = model_registry.get(model_id)
        static_tokens = len(build_system_prompt(model_card=new_card)) // 4
        memory.reconfigure_for_model(new_card, static_prompt_tokens=static_tokens)
        old_model_id = self.current_model_id
        self.current_model_id = model_id
        self.health.reset()
        # Conversation history is kept (not reset) across a model switch, but
        # the new model needs to know a switch happened - otherwise it can
        # mistake a plain "hi" for a nudge to re-answer the last unresolved
        # question in the carried-over history instead of just greeting back.
        # Any earlier switch-notice is REPLACED, not appended to - repeated
        # /models switching within one long session used to leave one of
        # these system messages behind per switch (trim_history's pairing
        # logic only evicts consecutive user/assistant pairs, so a lone
        # injected system message was never eligible for eviction and
        # accumulated forever).
        self.messages = [
            m for m in self.messages
            if not (m.get("role") == "system" and m.get("content", "").startswith(_MODEL_SWITCH_TAG))
        ]
        self.messages.append({
            "role": "system",
            "content": (
                f"{_MODEL_SWITCH_TAG}'{old_model_id}' to '{model_id}'. "
                "Prior turns above are already answered - do not re-answer "
                "them unless the user explicitly asks again.]"
            ),
        })

    def end(self):
        """Writes this session into episodic memory, then frees the model."""
        import src.memory as memory
        memory.mem.end_session()
        
        import contextlib
        @contextlib.contextmanager
        def _suppress_stdout_stderr():
            try:
                null_fd = os.open(os.devnull, os.O_RDWR)
                save_stdout = os.dup(1)
                save_stderr = os.dup(2)
                os.dup2(null_fd, 1)
                os.dup2(null_fd, 2)
                try:
                    yield
                finally:
                    os.dup2(save_stdout, 1)
                    os.dup2(save_stderr, 2)
                    os.close(save_stdout)
                    os.close(save_stderr)
                    os.close(null_fd)
            except Exception:
                yield

        with _suppress_stdout_stderr():
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
