"""Incremental scanner that turns a raw token stream into typed events.

Watches for '<think>...</think>' and '<tool_call>...</tool_call>' wherever
they appear in the stream — not just as literal first bytes — and only holds
back the shortest possible tail that could still be the start of one of
those tags, so plain text reaches the caller as early as possible.

Also guards against two known small-model failure modes:
  - a leaked chat-role token opening the reply ("user", "assistant") instead
    of real content — see _DEGENERATE_PREFIX_PATTERN.
  - hand-written foreign script (the model can't render some scripts
    reliably; such text is only trusted when it came out of a tool this
    turn, e.g. via translate).

Returns a ParseResult summarizing what happened; live text/thought events are
yielded through the same generator as they're recognized, so callers get
both a live feed and a final classification without relying on
StopIteration.value smuggling (unlike the previous implementation, callers
here just iterate — do not need `yield from` chained through multiple
frames to get the summary. See TurnEngine for the actual chaining, done
once, in one place)."""

import re
from dataclasses import dataclass, field

from src.engine.events import AnswerStart, Thought, Token

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_FINAL_RESPONSE_PATTERN = re.compile(r"<final_response>\s*(.*?)\s*</final_response>", re.DOTALL)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_CALL_OPEN = "<tool_call>"
_FINAL_RESPONSE_OPEN = "<final_response>"
_FINAL_RESPONSE_CLOSE = "</final_response>"

_TAG_OPENS = (_THINK_OPEN, _TOOL_CALL_OPEN, _FINAL_RESPONSE_OPEN)

# Scripts the model cannot write reliably itself (Greek/Cyrillic through
# Indic through CJK). Symbols, punctuation and emoji are deliberately outside
# these ranges. Text in these scripts is only trusted when it came out of a
# tool (i.e. the translate tool) this turn.
_FOREIGN_SCRIPT_PATTERN = re.compile(r"[Ͱ-῿⺀-퟿]")

# A known small-model failure mode: the reply starts with a leaked chat-role
# token — either the ENTIRE reply is just the bare word (sometimes with
# trailing whitespace/punctuation), or the role word leaks as a prefix
# before the model recovers and continues with real content a line later
# ("user\nI'm Tuffy, ..."). Both come from the same cause: the PROTOCOL
# EXAMPLES block's literal "User:"/"Assistant:" lines pull the next-token
# distribution onto a role-label token right at the start of generation.
# Empty output is the same failure (nothing was said at all). Deliberately
# anchored to the START of the reply and requires a line break or full-string
# match after the role word, so a real answer that merely CONTAINS "user"
# (e.g. "as a user, you can...") is never touched.
_DEGENERATE_PREFIX_PATTERN = re.compile(
    r"^(user|assistant|system)([\s:.,!?]*$|\s*\n)", re.IGNORECASE
)

_DEGENERATE_HOLDBACK_CHARS = len("assistant") + 2

_WORD_CHAR = re.compile(r"\w")


def is_degenerate_reply(text: str) -> bool:
    stripped = text.strip()
    return (
        not stripped
        # A reply with no word characters at all ("...", "…", "?!") is
        # filler, not an answer. Small models sometimes emit a bare "..."
        # and stop — and if that ever gets SAVED as an assistant turn, it
        # poisons every later prompt (episodic retrieval re-injects
        # "Assistant: ..." as an example the model then parrots, a
        # self-reinforcing loop observed in practice). Treat it exactly
        # like an empty reply: suppress and retry.
        or not _WORD_CHAR.search(stripped)
        or bool(_DEGENERATE_PREFIX_PATTERN.match(stripped))
    )


def strip_leading_filler_lines(text: str) -> str:
    """Drops leading lines that contain no word characters (a bare '...'
    or '…' the model emits before its real answer). Only whole lines are
    dropped — inline punctuation inside a real sentence is untouched."""
    while "\n" in text:
        first, rest = text.split("\n", 1)
        if first.strip() and not _WORD_CHAR.search(first):
            text = rest.lstrip("\n")
        else:
            break
    return text


@dataclass
class ParseResult:
    full_text: str = ""          # everything received, think blocks stripped
    is_tool_call: bool = False
    tool_call_json: str | None = None
    suppressed_foreign: bool = False
    degenerate_start: bool = False
    dropped_unclosed_think: bool = False  # generation cut off mid-<think>


class StreamParser:
    """One instance per completion call. Feed raw text deltas via `feed()`;
    each call returns a list of events recognized so far (possibly empty).
    Call `finish()` once the underlying stream ends to flush anything still
    held back and get the final ParseResult."""

    def __init__(self, sourced_text: str = ""):
        self._sourced_text = sourced_text
        self._pending = ""
        self._in_think = False
        self._in_final_response = False
        self._tool_call_found = False
        self._suppressed = False
        self._degenerate_start = False
        self._flushed_anything = False
        self._pending_start = ""
        self._full_text = ""
        self._dropped_unclosed_think = False
        self._answer_start_emitted = False

    def _unsourced_foreign(self, text: str) -> bool:
        chars = set(_FOREIGN_SCRIPT_PATTERN.findall(text))
        return any(c not in self._sourced_text for c in chars)

    def _flush(self, text: str, events: list):
        if not text or self._suppressed:
            return
        if not self._flushed_anything:
            self._pending_start += text
            # A filler line ("...") before the real answer is dropped here,
            # while it's still held back and unshown — once real content has
            # been flushed to the screen it can't be un-shown, so this is
            # the only safe point to do it.
            self._pending_start = strip_leading_filler_lines(self._pending_start)
            has_content = bool(self._pending_start.strip())
            if (not has_content or "\n" not in self._pending_start) and \
                    len(self._pending_start) < _DEGENERATE_HOLDBACK_CHARS:
                return
            if is_degenerate_reply(self._pending_start) or \
                    _DEGENERATE_PREFIX_PATTERN.match(self._pending_start):
                self._suppressed = True
                self._degenerate_start = True
                return
            text = self._pending_start
            self._flushed_anything = True
        if self._unsourced_foreign(text):
            self._suppressed = True
            return
        events.append(Token(text))

    def _flush_final_response(self, text: str, events: list):
        """Streams text known to be inside <final_response>...</final_response>
        directly as Token events - no degenerate-start holdback/buffering,
        since the tag itself is the model's explicit signal this is real
        answer content, not something that might still turn into a stray
        role-label leak. Foreign-script suppression still applies (the
        model still can't reliably hand-write non-Latin script)."""
        if not text or self._suppressed:
            return
        if not self._answer_start_emitted:
            events.append(AnswerStart())
            self._answer_start_emitted = True
            self._flushed_anything = True
        if self._unsourced_foreign(text):
            self._suppressed = True
            return
        events.append(Token(text))

    def feed(self, delta: str) -> list:
        events = []
        if not delta:
            return events
        self._full_text += delta

        if self._suppressed or self._tool_call_found:
            return events

        self._pending += delta

        while True:
            if self._in_think:
                close_idx = self._pending.find(_THINK_CLOSE)
                if close_idx == -1:
                    break
                think_text = self._pending[:close_idx].strip()
                if think_text:
                    events.append(Thought(think_text))
                self._pending = self._pending[close_idx + len(_THINK_CLOSE):]
                self._in_think = False
                continue

            if self._in_final_response:
                close_idx = self._pending.find(_FINAL_RESPONSE_CLOSE)
                if close_idx == -1:
                    # Hold back only the shortest tail that could still be
                    # the start of the closing tag - same trick as the
                    # opener scan below, so real content streams live
                    # instead of waiting for the whole answer to buffer.
                    safe_len = len(self._pending)
                    for k in range(min(len(_FINAL_RESPONSE_CLOSE), len(self._pending)), 0, -1):
                        if self._pending[-k:] == _FINAL_RESPONSE_CLOSE[:k]:
                            safe_len = min(safe_len, len(self._pending) - k)
                            break
                    if safe_len > 0:
                        self._flush_final_response(self._pending[:safe_len], events)
                        self._pending = self._pending[safe_len:]
                    break
                self._flush_final_response(self._pending[:close_idx], events)
                self._pending = self._pending[close_idx + len(_FINAL_RESPONSE_CLOSE):]
                self._in_final_response = False
                continue

            think_idx = self._pending.find(_THINK_OPEN)
            call_idx = self._pending.find(_TOOL_CALL_OPEN)
            final_idx = self._pending.find(_FINAL_RESPONSE_OPEN)
            candidates = [i for i in (think_idx, call_idx, final_idx) if i != -1]
            if not candidates:
                safe_len = len(self._pending)
                for opener in _TAG_OPENS:
                    for k in range(min(len(opener), len(self._pending)), 0, -1):
                        if self._pending[-k:] == opener[:k]:
                            safe_len = min(safe_len, len(self._pending) - k)
                            break
                if safe_len > 0:
                    self._flush(self._pending[:safe_len], events)
                    self._pending = self._pending[safe_len:]
                break

            first_idx = min(candidates)
            # Text before ANY tag (<think>, <tool_call>, or <final_response>
            # itself) still goes through the old buffered path (degenerate-
            # start detection etc.) - only content INSIDE <final_response>
            # gets the new unbuffered treatment, applied separately once
            # _in_final_response is set below.
            self._flush(self._pending[:first_idx], events)
            if self._suppressed:
                break

            if first_idx == think_idx:
                self._pending = self._pending[first_idx + len(_THINK_OPEN):]
                self._in_think = True
                continue
            elif first_idx == final_idx:
                self._pending = self._pending[first_idx + len(_FINAL_RESPONSE_OPEN):]
                self._in_final_response = True
                continue
            else:
                self._tool_call_found = True
                self._pending = ""
                break

        return events

    def finish(self) -> tuple[list, ParseResult]:
        """Call once the underlying token stream is exhausted. Returns
        (trailing_events, result)."""
        events = []

        if self._in_think:
            # Generation was cut off mid-thought (hit max_tokens, provider
            # truncated, ...). The old implementation let this fall through
            # to the generic "leftover pending text" flush below, which
            # passed it through the same degenerate/foreign-script guards as
            # real answer text and could show raw, never-meant-to-be-seen
            # chain-of-thought to the user as if it were the final answer.
            # A <think> block is never answer text, closed or not — drop it
            # and trace it as a (possibly incomplete) thought instead.
            think_text = self._pending.strip()
            if think_text:
                events.append(Thought(think_text))
            # Also strip it from full_text (used for history storage and the
            # tool-call regex) - _THINK_PATTERN below only matches CLOSED
            # <think>...</think> pairs, so an unclosed one would otherwise
            # survive into what gets saved as the assistant's turn.
            open_idx = self._full_text.rfind(_THINK_OPEN)
            if open_idx != -1:
                self._full_text = self._full_text[:open_idx]
            self._pending = ""
            self._in_think = False
            self._dropped_unclosed_think = True

        if self._in_final_response:
            # Generation ended before the closing tag arrived (EOS/stop-
            # string cut it short, or a length cap hit mid-answer) - the
            # tag opened, so this IS the answer; flush whatever's left of
            # it as-is rather than losing it or routing it back through
            # the buffered/degenerate-holdback path meant for un-tagged
            # text.
            self._flush_final_response(self._pending, events)
            self._pending = ""
            self._in_final_response = False

        if not self._tool_call_found and not self._suppressed and self._pending:
            self._flush(self._pending, events)

        if not self._tool_call_found and not self._suppressed and \
                not self._flushed_anything and self._pending_start:
            if is_degenerate_reply(self._pending_start):
                self._suppressed = True
                self._degenerate_start = True
            else:
                self._flushed_anything = True
                events.append(Token(self._pending_start))

        final_response_match = _FINAL_RESPONSE_PATTERN.search(self._full_text)
        if final_response_match:
            # The model used the tag: that's the authoritative answer text,
            # regardless of whatever else surrounds it (a stray sentence
            # before/after the tag some small models occasionally still
            # emit is deliberately NOT included - the tag is the contract).
            full_text = final_response_match.group(1).strip()
        else:
            full_text = _THINK_PATTERN.sub("", self._full_text).strip()
            # An unclosed <final_response> (caught above) leaves its content
            # in _full_text but with no matching closing tag for the regex
            # to find - fall back to stripping from the open tag onward.
            open_idx = full_text.find(_FINAL_RESPONSE_OPEN)
            if open_idx != -1:
                full_text = full_text[open_idx + len(_FINAL_RESPONSE_OPEN):].strip()

        if not self._tool_call_found:
            # Mirror the live-stream cleanup in what gets SAVED as the
            # assistant's turn: drop leading filler lines, and if the whole
            # reply was filler ("...") flag it degenerate so the engine
            # retries instead of persisting it — a saved "Assistant: ..."
            # turn gets re-injected by episodic retrieval into every later
            # prompt and teaches the model to answer "..." (self-reinforcing
            # pollution loop, observed in practice with a 2B local model).
            full_text = strip_leading_filler_lines(full_text).strip()
            if full_text and not _WORD_CHAR.search(full_text):
                full_text = ""
            if not full_text and not self._flushed_anything:
                self._suppressed = True
                self._degenerate_start = True

        tool_call_json = None
        if self._tool_call_found:
            match = _TOOL_CALL_PATTERN.search(self._full_text)
            if match:
                tool_call_json = match.group(1).strip()
            else:
                # The model opened <tool_call> but the closing tag never
                # arrived (EOS/stop-string cut generation short). The JSON
                # payload itself is often complete anyway — hand whatever
                # followed the opening tag to the tool-call parser, which
                # knows how to dig a JSON object out of surrounding noise.
                # Without this, tool_call_json stays None and the model gets
                # a useless "not valid JSON (char 0)" error for a call that
                # was actually 99% well-formed.
                _, _, tail = self._full_text.partition(_TOOL_CALL_OPEN)
                tool_call_json = tail.strip() or None

        result = ParseResult(
            full_text=full_text,
            is_tool_call=self._tool_call_found,
            tool_call_json=tool_call_json,
            suppressed_foreign=self._suppressed and not self._degenerate_start,
            degenerate_start=self._degenerate_start,
            dropped_unclosed_think=self._dropped_unclosed_think,
        )
        return events, result
