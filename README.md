# Tuffy ⚡

Tuffy is a direct, capable personal AI agent. It can run fully offline on local model weights,
or switch to any OpenAI-compatible API model — same tools, same chat loop, same `/models`
command either way.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design rationale, [src/README.md](src/README.md)
for a map of every package, and [docs/](docs/) for step-by-step configuration guides (models,
MCP servers, skills, tools). Most folders below have their own README with more detail.

---

## Features

- **Local or API models**: Run fully offline on local model weights, or switch to any
  OpenAI-wire-format API model with `/models <id>` — the model registry and chat loop treat both
  the same way. See [docs/configure-models.md](docs/configure-models.md).
- **Grouped tools**: Tools are organized by domain — Editing & Files, Coding & Execution,
  Research & Lookup, System, Memory — shown with headers in `/tools` and the system prompt.
- **Coding & editing**: `edit_file` for targeted changes (not just whole-file overwrites),
  `run_python`/`run_shell` (sandboxed, allowlisted) and `git_status`/`git_diff`/`git_commit` for
  working on real code in the workspace.
- **Skills**: drop a folder with a `SKILL.md` into `./skills/<name>/` to teach Tuffy how to
  approach a kind of task (optionally shipping its own tools or an MCP server to connect) — no
  core code changes needed. See `/skills` and [docs/configure-skills.md](docs/configure-skills.md).
- **MCP client**: connect any MCP server (filesystem, GitHub, browser, etc.) by listing it in
  `mcp_servers.json` — its tools show up alongside Tuffy's native ones automatically. See `/mcp`
  and [docs/configure-mcp.md](docs/configure-mcp.md).
- **Categorized CLI**: `/help` groups commands by what they do (Chat, Inspect, Models, Session)
  instead of one flat list; `/status` shows the active model and session state at a glance.
- **Clean chat with live traces**: colored step-by-step traces show thoughts, tool calls, and
  responses as they happen, with an animated status spinner while the agent works.
- **Synchronous memory**: Fact extraction and session summary storage run synchronously on turn
  completions, avoiding background threads and memory spikes.

---

## Available Commands

Run `/help` inside Tuffy for the full categorized list, or see
[docs/cli-reference.md](docs/cli-reference.md) for the same list with more detail. Highlights:

**Chat**
- `/new` - Start a fresh conversation, keeping long-term memory intact.
- `/clear` - Wipe long-term memory AND the conversation history.
- `/image <path>` - Attach an image file to your next message (requires a vision model).

**Inspect**
- `/memory` - Show everything in long-term memory (facts about you, recent sessions, lessons learned).
- `/tools` - List all tools the agent can call, grouped by domain.
- `/skills` - List installed skills (drop new ones in `./skills/<name>/`).
- `/mcp` - List connected MCP servers and the tools they registered.
- `/status` - Show the active model, vision support, and turn count for this session.

**Models**
- `/models` - List available models (local and API) and show which one is active.
- `/models <id>` - Switch to a different model (local or API), unloading the current one.
- `/models info <id>` - Show a model's full model card, including provider/API details.

**Session**
- `/help` - Show the full command list.
- `/exit` or `/quit` - Save session memory and close the program.

---

## Setup & Running

### 1. Requirements
- Python 3.11+
- virtualenv (e.g. using `uv` or standard Python `venv`)

### 2. Installation
Set up your virtual environment and install the dependencies:
```bash
# Using uv (recommended)
uv sync

# Or using standard pip
pip install -r pyproject.toml
```

### 3. Add a model
Tuffy ships with no model registered by default — you choose what runs it. Register a local
model (place its weight file under `src/models/weights/`) or an API-provider model in
[src/models/__init__.py](src/models/__init__.py), then set it as `DEFAULT_MODEL` or switch to it
at runtime with `/models <id>`. Full walkthrough (including API-key setup for hosted models):
[docs/configure-models.md](docs/configure-models.md).

### 4. (Optional) Connecting MCP servers
See [docs/configure-mcp.md](docs/configure-mcp.md) — create `mcp_servers.json` (gitignored) and
list any MCP servers you want connected. Tuffy connects to each one at startup and registers its
tools as `<server>_<tool>`; a server that fails to connect is skipped with a warning and never
blocks startup.

### 5. (Optional) Adding skills
See [docs/configure-skills.md](docs/configure-skills.md) — drop a folder into `./skills/<name>/`
with a `SKILL.md` (YAML frontmatter with `name`/`description`, plus a markdown body of
guidance). Optionally add a `tools.py` or an `mcp.json`. See `skills/code-review/` for a working
example.

### 6. Running the Agent
Run the main script to start your chat session:
```bash
python3 main.py
```

---

## Project Structure

```
main.py                 Entry point: startup wiring (skills/MCP discovery), then the input loop
src/
  cli/                  Interactive terminal chat: banner, spinner, slash commands, turn loop
  agent.py              LocalAgent — the provider-agnostic ReAct tool-calling loop
  identity.py           Fixed self-model (name, capabilities) — never LLM-written
  memory.py             Long-term memory (profile/notes/sessions/lessons) + the `remember` tool
  vision.py              Image encoding + IMAGE_SENTINEL protocol for vision tool results
  llm/                   Model-provider interface + adapters (local weights, OpenAI-compatible API)
  models/                Model registry; weights/ holds gitignored model weight files
  prompts/               All system-prompt text: personas.yaml + templates.py
  tools/                 Native tools by domain (editing/coding/research/system) + MCP client
  skills/                Discovery/loading for ./skills/*/ capability packs
skills/                 Droppable capability packs (SKILL.md + optional tools.py/mcp.json)
docs/                   Configuration guides (models, MCP, skills, tools, CLI reference)
data/memory/            JSON-backed long-term memory store
agent_workspace/        Sandboxed file I/O root for the agent's file/code tools
```

Every folder above has its own README with more detail — start at [src/README.md](src/README.md)
for the full map, [docs/](docs/) for configuration how-tos, or [ARCHITECTURE.md](ARCHITECTURE.md)
for the design rationale behind the tool/model-provider/skills systems.
