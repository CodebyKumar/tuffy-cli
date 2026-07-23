# Tuffy CLI — Upgrade Plan

Status: **proposal.** No code changes made yet.

`tuffy-cli` is not just "the backend `tuffy-ui` imports" — it is a complete, standalone terminal
AI agent with its own users and its own `/mode voice` fallback path. This plan is scoped to
improving `tuffy-cli` **as a terminal product in its own right**: better voice mode, better
memory/model ergonomics, better extensibility — independent of anything `tuffy-ui` needs. Where an
item would also benefit `tuffy-ui` (e.g. a shared voice package), that's noted, but nothing here is
driven by `tuffy-ui`'s roadmap.

See `ARCHITECTURE.md` for the current design this plan builds on top of, and
`tuffy-ui/UPGRADE_PLAN.md` for the sibling repo's plan (voice-first desktop UI, separate concerns).

---

## 0. What's already strong (keep, don't touch)

- The `LLMProvider` / `ToolRegistry` / skills-and-MCP seams (`ARCHITECTURE.md` §1.2) are clean and
  should not be disturbed by anything below.
- The public package surface (`tuffy_core/__init__.py`: `create_session`, `AgentSession`,
  `run_turn_stream`) already exists and is what `tuffy-ui` depends on — this plan does not touch
  its shape, only what's built on top of it inside `tuffy-cli` itself.
- 140+ passing tests, a real `ARCHITECTURE.md`, per-folder `README.md`s — this is an unusually
  well-documented codebase already; upgrades below should match that bar, not lower it.

---

## 1. Terminal voice mode is the weakest part of the product — fix it

`src/voice/voice_cli.py` (200 lines) is a **blocking, turn-based, press-Enter-to-talk** loop:
record → transcribe → run turn → speak, one step at a time, no VAD, no streaming TTS, no barge-in
except a crude `select()`-on-stdin hack during playback. Meanwhile `tuffy-ui/backend/voice/` has
already solved every one of these problems properly: Silero VAD, per-sentence streaming TTS,
resident whisper.cpp, a real interrupt path. Right now the terminal's voice mode is a strictly
worse experience than the desktop app's, on the same underlying agent.

Concretely, in priority order:

1. **Continuous listening via VAD, not press-Enter-to-record.** Port the VAD-gated
   utterance-buffering approach from `tuffy-ui/backend/voice/vad.py` (Silero, ONNX, no torch
   dependency) into `src/voice/`. Replace `AudioInterface.record_audio()`'s manual
   start/stop-on-Enter with "start speaking, pause ~500ms, utterance ends automatically" — this
   alone is the single biggest UX gap versus `tuffy-ui`.
2. **Streaming/incremental TTS**, not "wait for the full reply, then synthesize the whole thing."
   `tuffy-ui/backend/voice/tts_piper.py` already streams PCM per-sentence as it's produced;
   `src/voice/tts.py` today has no equivalent — `clean_text_for_speech()` runs on the full final
   reply only after `run_turn` completes.
3. **A real interrupt/barge-in path**, not the current `select()` poll on stdin during playback
   only. Speaking over Tuffy mid-generation (not just mid-playback) should stop generation, not
   just audio.
4. **Decide the code-sharing question `TUFFY_DESKTOP_PLAN.md` (now removed) deliberately deferred**
   — `tuffy-ui/backend/voice/` and `src/voice/` are two independent implementations of
   VAD/STT/TTS today. Now that both exist and both work, extract the genuinely-shared parts (VAD
   wrapper, whisper.cpp wrapper, Piper wrapper — not the WS-frame-shaped buffering, which is
   `tuffy-ui`-specific) into a small third package (e.g. `tuffy-voice`) both repos depend on,
   *only if* `tuffy-cli`'s `pyproject.toml` staying free of FastAPI/server deps is preserved
   (`ARCHITECTURE.md` principle 5). If that's too disruptive right now, at minimum bring `src/voice/`
   up to feature parity by porting the *techniques*, independent implementations is fine short-term.
5. **`/mode voice` should show live partial transcripts** the way `tuffy-ui` will (once its own
   Phase 6 lands) — printing the partial STT text to the terminal as the user speaks, not just
   after the utterance ends. Cheap to add once VAD-gated streaming STT exists per item 1.

## 2. Model & session ergonomics

- **`/models switch` has no dry-run / compatibility check.** Switching to a model whose
  `context_length` is much smaller than the current conversation's token count should warn
  *before* switching (today it presumably just truncates silently on the next turn via
  `session.trim_history`). Add a pre-switch estimate + confirmation when the active history
  wouldn't fit.
- **No `/models benchmark` or load-time feedback.** Local GGUF loads can take anywhere from
  seconds to a minute depending on size/quantization; there's no progress indicator during
  `ModelAgent.load()`. A simple spinner with elapsed time (reusing `src/cli/display.py`'s existing
  spinner) during model load would remove a real "is it hung?" moment.
- **Session resumption across restarts.** Today, per `ARCHITECTURE.md` §2.2, chat history lives
  only in the in-memory `Session` for the process lifetime — closing the terminal loses the visible
  transcript (long-term facts persist via Elastimem, but the turn-by-turn conversation doesn't).
  Consider an opt-in `--resume` flag that persists/reloads the last N turns of raw conversation
  history to `data/session_history.json`, separate from Elastimem's fact store. This is a real gap
  for a terminal tool used across multiple sessions in a day.

## 3. Tool surface growth

- **`/tools` output doesn't show which tools came from MCP vs. skills vs. native at a glance in a
  way that's easy to scan** when the list grows past ~20 entries (once several MCP servers are
  connected). Consider grouping headers becoming collapsible or a `/tools <group>` filter for
  *display* only (never for hiding tools from the model — `ARCHITECTURE.md` principle 2 is
  correct and should stay).
- **No tool-call cost/latency visibility.** For anyone debugging a slow turn, there's currently no
  per-tool timing surfaced (only the overall spinner). A `TUFFY_DEBUG_CONTEXT`-style env var
  (`TUFFY_DEBUG_TIMING=1`) that appends per-hop/per-tool-call timing to a log would pair well with
  the existing debug-context mechanism (`docs/cli-reference.md`'s "Debugging" section) and require
  no new UI.

## 4. Memory quality-of-life

- **`/memory quarantine` is read-only today** (per `docs/cli-reference.md`) — there's no way to
  review a quarantined item and manually promote it if the rejection was a false positive (e.g. a
  legitimately new fact that happened to collide with a reserved-key heuristic). A
  `/memory promote <id>` command closing that loop would remove the only one-way door in the
  memory system.
- **No memory export.** For a personal agent that accumulates real facts about the user over
  months, there's no `/memory export` to dump the fact store as portable JSON/markdown (for backup,
  audit, or migration to a fresh `tuffy.db`). Low effort, meaningfully de-risks "what if Elastimem's
  format changes" or "I want to see everything it thinks it knows about me" in one place instead of
  paging through `/memory search`.

## 5. Persona / prompt maintainability

`src/prompts/personas.yaml` currently has one active preset (`tuffy`) with a single large
`system_prompt` string. This is fine at current size but worth flagging before it grows:

- If a second persona is ever added (the file's own structure already anticipates this —
  `presets:` is a dict, `active:` selects one), verify `src/prompts/__init__.py`'s
  `build_system_prompt()` doesn't have any accidental single-preset assumptions baked in.
- No action needed now — this is a "watch for it" item, not a current problem.

## 6. Suggested priority order

1. §1.1–1.3 (VAD, streaming TTS, real interrupt) — the highest-impact, most overdue gap; brings
   terminal voice mode to parity with what's already proven working in `tuffy-ui`.
2. §2 model-load spinner + pre-switch context warning — small, high-visibility polish.
3. §4 memory export + quarantine promote — closes real one-way doors, low effort.
4. §1.4 shared-voice-package extraction — only once both voice stacks are feature-complete enough
   to know what's actually common versus incidentally similar.
5. §3 tool-surface display grouping, §2 session resumption — nice-to-haves, do opportunistically.

## Non-goals

- No change to the `LLMProvider`/`ToolRegistry`/skills/MCP architecture — it's sound.
- No new dependency on FastAPI, WebSocket, or anything from `tuffy-ui`'s stack — `tuffy-cli` stays
  a terminal-only package per `ARCHITECTURE.md` principle 5.
- Not a rewrite of `src/voice/` from scratch — porting proven techniques from `tuffy-ui/backend/voice/`,
  not importing that code directly (different transport shape: WS binary frames vs. local mic).
