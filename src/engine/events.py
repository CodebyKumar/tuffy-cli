"""Typed events the turn engine emits. Replaces the old dual-channel design
(live tokens via `yield`, final classification via `StopIteration.value`,
side-info via trace_cb/status_cb callbacks) with one flat, ordered stream
any consumer can read without knowing anything about generators.

A turn is exactly one sequence of these, always ending in either Done or
Failed — never both, never neither."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Status:
    """Advisory only — safe to ignore. 'thinking', 'using web_search', etc."""
    label: str


@dataclass(frozen=True)
class Thought:
    """A <think> block the model produced. Never part of the answer."""
    text: str


@dataclass(frozen=True)
class ToolCall:
    """The model decided to invoke a tool. Emitted before execution."""
    name: str
    arguments: dict
    thought: str = ""


@dataclass(frozen=True)
class ToolResult:
    """A tool ran to completion (success or handled failure) and produced an
    observation that was fed back to the model. `ok=False` means the tool
    raised (any exception) or the call was malformed — the loop recovered
    and asked the model to self-correct, nothing crashed."""
    name: str
    result: str
    ok: bool


@dataclass(frozen=True)
class AnswerStart:
    """The model has opened <final_response> - the final answer is about to
    stream as Token events. Fires BEFORE any of that content arrives, so a
    renderer can retire status UI (e.g. a spinner) deterministically instead
    of inferring "answer started" from the first Token itself, which races
    against still-updating status output when the model's first answer
    chars happen to contain symbols/newlines a spinner's own redraw logic
    wasn't expecting mid-frame."""


@dataclass(frozen=True)
class Token:
    """One piece of the final answer, safe to print immediately."""
    text: str


@dataclass(frozen=True)
class Done:
    """Terminal: the turn produced a real answer."""
    full_text: str


@dataclass(frozen=True)
class Failed:
    """Terminal: the turn could not produce an answer. `recoverable=True`
    means the session should keep running (drop this turn and let the user
    try again); `recoverable=False` is reserved for situations so broken
    continuing is unsafe (none currently raise this — kept for callers that
    want to distinguish 'ask user to retry' from 'something is very wrong'
    without inspecting `kind` strings)."""
    kind: str      # "oom" | "provider" | "interrupted" | "empty" | "internal"
    message: str
    recoverable: bool = True


TurnEvent = Status | Thought | ToolCall | ToolResult | AnswerStart | Token | Done | Failed
