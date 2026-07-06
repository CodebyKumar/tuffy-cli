# CLI Reference

Every slash command, grouped the same way `/help` groups them inside Tuffy. See
[src/cli/commands.py](../src/cli/commands.py) for the implementation.

## Chat

| Command | What it does |
|---|---|
| `/new` | Start a fresh conversation. Long-term memory is untouched. |
| `/clear` | Wipe long-term memory **and** the conversation history. |
| `/image <path>` | Attach an image file to your next message (requires a vision-capable model). |

## Inspect

| Command | What it does |
|---|---|
| `/memory` | Show everything in long-term memory: facts about you, recent session summaries, lessons learned. |
| `/tools` | List every tool the agent can call, grouped by domain (native and MCP). |
| `/skills` | List installed skills. Drop a new one in `./skills/<name>/` and restart to add more. |
| `/mcp` | List connected MCP servers and the tools each one registered. |
| `/status` | Show the active model, whether it supports vision, and how many turns this session has had. |

## Models

| Command | What it does |
|---|---|
| `/models` | List every registered model (local and API), marking the active one. |
| `/models <id>` | Switch to a different model, unloading the current one. Loads the new one first, so a failed switch leaves you on the working model. |
| `/models info <id>` | Show a model's full card: capabilities, context length, license, source, and (for API models) endpoint/key details. |

## Session

| Command | What it does |
|---|---|
| `/help` | Show the categorized command list. |
| `/exit`, `/quit` | Save session memory and terminate. |

## Adding a new command

See [src/cli/README.md](../src/cli/README.md) — write a `cmd_<name>` function, add a branch to
`handle_command()`, add a row to `_HELP_SECTIONS`. No other file needs to change.
