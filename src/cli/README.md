# src/cli/

Everything that turns the agent core into an interactive terminal chat. `main.py` at the
repo root is a thin entry point — it wires up skills/MCP discovery at startup and then hands
off to this package for the actual input loop.

- [display.py](display.py) - ANSI colors, the startup banner, and `Spinner` (the animated
  `AI ❯ thinking...` status line shown while the agent works).
- [session.py](session.py) - `Session` — owns the active model/agent, chat history, and
  history-trimming rules (`trim_history`, `compact_turn`, `keep_only_latest_image`) that keep
  the rolling conversation under the model's context budget.
- [commands.py](commands.py) - one `cmd_*` function per slash command (`/help`, `/new`,
  `/clear`, `/purge`, `/memory`, `/status`, `/models` (including `/models default`), `/image`,
  `/tools`, `/skills`) plus `handle_command()`, the dispatch table `main.py`'s input loop calls.
- [turn.py](turn.py) - `run_turn()`: builds the message list for one user turn, drives
  `src.engine.turn_engine.run_turn()`, and renders each `TurnEvent` as it arrives (spinner label,
  trace lines, live answer tokens) before folding tool-call intermediates back out of history
  once the turn completes.

## Adding a new slash command

1. Write a `cmd_<name>(session, ...)` function in `commands.py` that prints whatever it needs to.
2. Add a branch to `handle_command()` matching `/<name>`.
3. Add a row to `_HELP_SECTIONS` under whichever category fits (`Chat`, `Inspect`, `Models`,
   `Session`, or a new one) so `/help` picks it up automatically.

No other file needs to change — `main.py` only ever calls `handle_command()`, never a specific
command function.

## Why this is split out

Nothing in this package is imported by `src/engine/` or the tool/model/skill registries —
only `main.py` reaches into it. That keeps the agent core and its registries usable headless
(e.g. a future web or API frontend) without dragging in `print()`/`input()` terminal code.
