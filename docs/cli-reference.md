# CLI Reference

Every slash command, grouped the same way `/help` groups them inside Tuffy. See
[src/cli/commands.py](../src/cli/commands.py) for the implementation.

## Chat

| Command | What it does |
|---|---|
| `/new` | Start a fresh conversation. Long-term memory is untouched. |
| `/clear` | Reset the in-window conversation history. Long-term memory (the Elastimem database) is untouched — same effect as `/new`. Use `/purge` to actually wipe memory. |
| `/purge` | Wipe the long-term memory database: archives the current `tuffy.db` to `data/memory/backups/` and opens a fresh one. Also resets the conversation history. |
| `/image <path>` | Attach an image file to your next message (requires a vision-capable model). |
| `/mode [text|voice]` | Show the current active mode, or switch between text and voice modes. |

## Inspect

| Command | What it does |
|---|---|
| `/memory` | Show a summary of long-term memory: facts about you, recent session summaries, lessons learned. |
| `/memory search <query>` | Search past conversations and stored facts (same recall the `recall` tool uses). |
| `/memory facts <key>` | Show the full version history of one fact key. |
| `/memory forget <key>` | Forget a fact key — a non-destructive tombstone, not a hard delete. |
| `/memory quarantine` | Show recently rejected/quarantined auto-extractions (e.g. self-referential or identity-shaped values `remember`/reflection tried to store). |
| `/tools` | List every tool the agent can call, grouped by domain (native and MCP). |
| `/skills` | List installed skills. Drop a new one in `./.tuffy/skills/<name>/` and restart to add more. |
| `/mcp` | List connected MCP servers and the tools each one registered. |
| `/status` | Show the active model, vision support, turn count, estimated context usage (vs. the model's max), and rate limits (API models). |

## Models

| Command | What it does |
|---|---|
| `/models` | List every registered model (local and API), marking the active one. |
| `/models switch <id>` | Switch to a different model, unloading the current one. Loads the new one first, so a failed switch leaves you on the working model. |
| `/models <id>` | Shorthand for `/models switch <id>`. |
| `/models default <id>` | Switch to a model and persist it as the startup default (`.tuffy/settings.json`, gitignored) — it loads automatically next time Tuffy starts. |
| `/models info <id>` | Show a model's full card: capabilities, context length, license, source, and (for API models) endpoint/key details. |

## Session

| Command | What it does |
|---|---|
| `/help` | Show the categorized command list. |
| `/exit`, `/quit` | Save session memory and terminate. |

## Adding a new command

See [src/cli/README.md](../src/cli/README.md) — write a `cmd_<name>` function, add a branch to
`handle_command()`, add a row to `_HELP_SECTIONS`. No other file needs to change.

## Debugging: TUFFY_DEBUG_CONTEXT

Set `TUFFY_DEBUG_CONTEXT=<path>` before starting Tuffy to append the exact system prompt and full
message history sent to the model, for every turn, to that file:

```bash
TUFFY_DEBUG_CONTEXT=/tmp/tuffy-debug.log uv run main.py
```

This is a ground-truth trace of what the model actually saw — useful for diagnosing
memory/context bugs (a stale or garbled fact, retrieval returning the wrong thing, history
growing unexpectedly large) that aren't visible from the rendered chat transcript alone. No-op
(zero file I/O) when unset. See `_dump_debug_context` in
[src/cli/turn.py](../src/cli/turn.py).
