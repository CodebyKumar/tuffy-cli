"""Runs one user turn to completion: builds the message list, drives
src.engine.turn_engine.run_turn, and renders each TurnEvent as it arrives —
spinner label updates, trace lines (thoughts/tool calls/results), live
answer tokens — then folds the result back into session history.

Replaces the previous design where the engine pushed side-information out
through session.trace_cb/status_cb callbacks while the token loop here only
saw plain text. Here there is exactly one thing being iterated — the event
stream — and every branch below corresponds to one event type. No hidden
control flow, no values smuggled through StopIteration."""

import json
import os

import src.memory as memory
from src.cli.session import Session, keep_only_latest_image, trim_history, compact_turn
from src.cli.display import Spinner, C_AI, C_BLUE, C_DIM, C_RESET, C_USER, C_WARN, CLEAR_LINE
from src.engine import turn_engine
from src.engine.events import AnswerStart, Done, Failed, Status, Thought, Token, ToolCall, ToolResult

_DEBUG_CONTEXT_ENV = "TUFFY_DEBUG_CONTEXT"


def _dump_debug_context(user_input: str, messages: list) -> None:
    """When TUFFY_DEBUG_CONTEXT=<path> is set, appends the exact system
    prompt and full message history sent for this turn to that file — a
    ground-truth trace of what the model actually saw, for diagnosing
    memory/context bugs (garbled facts, stale retrieval, runaway history
    growth) that are invisible from the rendered chat transcript alone.
    No-op (and no file I/O) when the env var isn't set."""
    path = os.environ.get(_DEBUG_CONTEXT_ENV)
    if not path:
        return
    with open(path, "a") as f:
        f.write(f"\n{'='*80}\nUSER: {user_input}\n{'-'*80}\n")
        f.write(f"SYSTEM MESSAGE:\n{messages[0]['content']}\n")
        f.write(f"{'-'*80}\nFULL HISTORY ({len(messages)} messages):\n")
        for i, m in enumerate(messages):
            content = m.get("content")
            if isinstance(content, list):
                content = "[multipart/image content]"
            f.write(f"  [{i}] {m.get('role')}: {str(content)[:300]}\n")


def run_turn(session: Session, user_input: str) -> bool:
    """Runs one user turn to completion. Returns False if generation was
    interrupted/failed and the turn was rolled back, True otherwise."""
    memory.mem.tick()
    plan = memory.mem.build_context(user_input)

    messages = session.messages
    messages[0] = session.system_message(context_plan=plan)
    _dump_debug_context(user_input, messages)
    user_message = {"role": "user", "content": user_input}
    pending_image = session.pending_image_data_uri
    if pending_image:
        user_message = session.agent.attach_image(user_message, pending_image)
    messages.append(user_message)
    keep_only_latest_image(messages)
    session.messages = messages = trim_history(messages, plan)
    turn_start = len(messages) - 1

    renderer = _TurnRenderer()
    full_response = ""
    outcome_failed = None

    try:
        with memory.mem.foreground():
            event_stream = turn_engine.run_turn(session.agent.provider, session.agent.sampling_params, messages)
            for event in event_stream:
                renderer.render(event)
                if isinstance(event, Token):
                    full_response += event.text
                elif isinstance(event, ToolResult):
                    session.note_captured_image(event.name, event.result)
                elif isinstance(event, Done):
                    full_response = event.full_text
                elif isinstance(event, Failed):
                    outcome_failed = event
    except KeyboardInterrupt:
        outcome_failed = Failed(kind="interrupted", message="Interrupted")
    finally:
        renderer.finish()

    if outcome_failed is not None:
        print()
        _report_failure(outcome_failed)
        session.health.record(outcome_failed.kind)
        if session.health.should_nudge():
            print(
                f"{C_DIM}[{session.health.consecutive_failures()} failed turns in a row on this "
                f"model — try /models to switch, or /status for details.]{C_RESET}\n"
            )
        del messages[turn_start:]
        if outcome_failed.kind == "context_overflow":
            # Rolling back this turn's own additions (above) isn't enough —
            # the PRE-EXISTING history was already large enough to overflow,
            # so the very next turn would hit the same wall immediately.
            # Force a hard trim down to a small fixed window now, rather
            # than waiting for the normal budget-based trim_history (which
            # runs on the next turn anyway, but against the same
            # already-too-large plan.keep_last_n_turns that let this
            # happen).
            session.messages = messages = _force_trim(messages, keep_last_n_turns=1)
        return False

    # The pending image is only cleared once the turn actually succeeded —
    # a failure above leaves it in place so the user doesn't have to
    # re-attach it before retrying.
    session.pending_image_data_uri = None

    print("\n")

    if not full_response.strip():
        # Belt-and-suspenders: run_turn is expected to always force a real
        # answer (see turn_engine._final_answer_guaranteed), but if every
        # guard somehow still lets an empty reply through, don't save a
        # blank turn — it corrupts history (the model sees its own empty
        # reply as an example to repeat) and pollutes elastimem's episodic
        # record.
        del messages[turn_start:]
        session.health.record("empty")
        print(f"{C_DIM}[No response generated — turn discarded, try again]{C_RESET}\n")
        return False

    messages.append({"role": "assistant", "content": full_response})
    # Drop this turn's ReAct intermediates (tool drafts/observations) — the
    # final answer carries what mattered, and stale tool dumps slow every
    # later turn. Trace mode already showed them live; nothing is lost.
    compact_turn(messages, turn_start)
    session.health.record(None)

    memory.mem.record_turn(user_input, full_response)
    return True


def _force_trim(messages: list, keep_last_n_turns: int) -> list:
    """A fixed-budget trim_history call for the context_overflow recovery
    path, where the normal plan-derived budget already proved too generous
    for what actually fit. A plain object with just the one attribute
    trim_history reads, rather than a full context_plan."""
    class _FixedPlan:
        pass
    plan = _FixedPlan()
    plan.keep_last_n_turns = keep_last_n_turns
    return trim_history(messages, plan)


def _report_failure(failure: Failed):
    if failure.kind == "oom":
        memory.mem.report_pressure()
        print(
            f"{C_DIM}[Generation failed: {failure.message}. The machine is likely low on "
            f"memory — close some apps and try again.]{C_RESET}\n"
        )
    elif failure.kind == "context_overflow":
        print(
            f"{C_DIM}[Generation failed: the conversation grew too long for this model's "
            f"context window. Trimming older history — try again.]{C_RESET}\n"
        )
    elif failure.kind == "provider":
        print(f"{C_DIM}[Generation failed: {failure.message}]{C_RESET}\n")
    elif failure.kind == "interrupted":
        print(f"{C_DIM}[{failure.message}]{C_RESET}\n")
    else:
        print(f"{C_DIM}[Generation failed: {failure.message}]{C_RESET}\n")


class _TurnRenderer:
    """Owns the spinner and the 'has AI ❯ been printed yet this turn' state
    — both turn-scoped, both used only here, instead of being split across
    Session attributes (trace_printed, active_spinner) that every other
    piece of code had to know not to touch mid-turn.

    Trace lines (thought/tool_call/tool_result) can arrive interleaved with
    live answer tokens across ReAct hops, so the spinner is stopped just
    long enough to print each trace line and restarted afterward - EXCEPT
    once the final answer's tokens start arriving, at which point the
    spinner is retired for the rest of the turn (a hop can never emit a
    trace line after its own answer tokens have started, since a tool-call
    hop and a final-answer hop are mutually exclusive per hop)."""

    def __init__(self):
        self.spinner = Spinner()
        self.spinner.start()
        self._prompt_printed = False
        self._answer_started = False

    def _print_prompt_prefix(self):
        if not self._prompt_printed:
            print(f"{CLEAR_LINE}{C_AI}AI ❯{C_RESET} ", end="", flush=True)
            self._prompt_printed = True
        else:
            print(CLEAR_LINE, end="", flush=True)

    def render(self, event):
        if isinstance(event, Status):
            self.spinner.set_label(event.label)
        elif isinstance(event, Thought):
            self.spinner.stop(show_prompt=False)
            self._print_prompt_prefix()
            print(f"{C_BLUE}[thought] {event.text}{C_RESET}", flush=True)
            self.spinner.start()
        elif isinstance(event, ToolCall):
            self.spinner.stop(show_prompt=False)
            self._print_prompt_prefix()
            if event.thought:
                # The <tool_call> JSON's own "thought" field is a distinct,
                # short rationale the model writes for THIS call - print it
                # as a real, permanent [thought] line (same as a <think>
                # block) rather than feeding it to the spinner label, where
                # it used to flash for a fraction of a second as
                # "AI ❯ <thought text>..." and then vanish the instant the
                # next event stopped the spinner. Nothing should only ever
                # exist as spinner text.
                print(f"{C_BLUE}[thought] {event.thought}{C_RESET}", flush=True)
            args_json = json.dumps(event.arguments, ensure_ascii=False)
            print(f"{C_WARN}[execute] {event.name}({args_json}){C_RESET}", flush=True)
            self.spinner.start()
        elif isinstance(event, ToolResult):
            self.spinner.stop(show_prompt=False)
            self._print_prompt_prefix()
            color = C_USER if event.ok else C_WARN
            print(f"{color}[result] {event.result}{C_RESET}", flush=True)
            self.spinner.start()
        elif isinstance(event, AnswerStart):
            # Retires the spinner and prints the "AI ❯" prefix BEFORE any
            # answer content streams, rather than inferring "the answer
            # started" from the first Token itself (the old signal). That
            # inference raced against the spinner's own background-thread
            # redraw: if the first Token happened to land while the spinner
            # was mid-frame, and that Token's text contained a newline or
            # multiple lines (a code fence, a list), the spinner's row-count
            # tracking (see Spinner._clear_last_render) could miscount and
            # corrupt the terminal display on the next redraw it never got
            # to do, since answer text was now sharing the same screen
            # region a still-notionally-running spinner thought it owned.
            # This event makes the handoff a hard boundary instead of a race.
            if not self._answer_started:
                self._answer_started = True
                self.spinner.stop(show_prompt=False)
                self._print_prompt_prefix()
        elif isinstance(event, Token):
            if not self._answer_started:
                # Fallback for the (should be rare, given personas.yaml now
                # mandates the tag) case where the model's reply never used
                # <final_response> at all - same handoff as AnswerStart,
                # just triggered by the first Token instead of a dedicated
                # event, exactly as before this fix existed.
                self._answer_started = True
                self.spinner.stop(show_prompt=False)
                self._print_prompt_prefix()
            print(event.text, end="", flush=True)
        # Done/Failed carry no direct rendering here — turn.py prints the
        # final newline/error message itself once the loop ends.

    def finish(self):
        self.spinner.stop()
