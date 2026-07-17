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
- **Coding & editing**: `edit_file` for targeted changes (not just whole-file overwrites) and
  `run_python`/`run_shell` (sandboxed, allowlisted — no `git`, see
  [src/tools/README.md](src/tools/README.md)) for working on real code in the workspace.
- **Skills**: drop a folder with a `SKILL.md` into `./.tuffy/skills/<name>/` to teach Tuffy how to
  approach a kind of task (optionally shipping its own tools or an MCP server to connect) — no
  core code changes needed. See `/skills` and [docs/configure-skills.md](docs/configure-skills.md).
- **MCP client**: connect any MCP server (filesystem, GitHub, browser, etc.) by listing it in
  `.tuffy/mcp.json` — its tools show up alongside Tuffy's native ones automatically. See `/mcp`
  and [docs/configure-mcp.md](docs/configure-mcp.md).
- **Categorized CLI**: `/help` groups commands by what they do (Chat, Inspect, Models, Session)
  instead of one flat list; `/status` shows the active model and session state at a glance.
- **Clean chat with live traces**: colored step-by-step traces show thoughts, tool calls, and
  responses as they happen, with an animated status spinner while the agent works.
- **Voice Interactive Mode**: Run the CLI in full voice mode using local Whisper STT (Speech-to-Text) and Piper TTS (Text-to-Speech) for hands-free spoken conversations.
- **Elastimem-backed memory**: a SQLite-backed store with facts, episodic recall, session
  summaries, and lessons learned, extracted by a background worker gated to stay off the
  foreground reply path. See [data/README.md](data/README.md).

---

## Available Commands

Run `/help` inside Tuffy for the full categorized list, or see
[docs/cli-reference.md](docs/cli-reference.md) for the same list with more detail. Highlights:

**Chat**
- `/new` - Start a fresh conversation, keeping long-term memory intact.
- `/clear` - Reset the conversation history (long-term memory is untouched — same as `/new`).
- `/purge` - Wipe long-term memory: archives the memory database and starts a fresh one.
- `/image <path>` - Attach an image file to your next message (requires a vision model).

**Inspect**
- `/memory` - Show a summary of long-term memory (facts about you, recent sessions, lessons learned).
- `/memory search <query>` - Search past conversations and stored facts.
- `/tools` - List all tools the agent can call, grouped by domain.
- `/skills` - List installed skills (drop new ones in `./.tuffy/skills/<name>/`).
- `/mcp` - List connected MCP servers and the tools they registered.
- `/mcp add <github-url>` / `/mcp remove <name>` - Add or remove an MCP server without hand-editing config.
- `/status` - Show the active model, vision support, turn count, estimated context usage, and rate limits (API models).

**Models**
- `/models` - List available models (local and API) and show which one is active.
- `/models switch <id>` - Switch to a different model (local or API), unloading the current one. `/models <id>` also works as shorthand.
- `/models default <id>` - Switch to a model and persist it as the startup default.
- `/models info <id>` - Show a model's full model card, including provider/API details.

**Session**
- `/mode [text|voice]` - View or switch interaction mode (text vs voice).
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

# Or to include optional voice interactive mode packages
uv sync --extra voice

# Or using standard pip
pip install -r pyproject.toml
```

### 3. Add a model
Tuffy ships with no model registered by default — you choose what runs it. Register a local
model (place its weight file under `src/models/weights/`) in
[src/models/configs/local.py](src/models/configs/local.py), or an API-provider model in
[src/models/configs/api.py](src/models/configs/api.py), then set it as `DEFAULT_MODEL` (in
[src/models/__init__.py](src/models/__init__.py)) or switch to it at runtime with `/models <id>`.
For API models, put the provider's API key in `.env` at the repo root (e.g.
`GROQ_API_KEY=...`) — the API provider reads it from there automatically if it's not already
exported in your shell, no manual `export` needed. Full walkthrough (including rate-limit
metadata for hosted models): [docs/configure-models.md](docs/configure-models.md).

### 4. (Optional) Connecting MCP servers
See [docs/configure-mcp.md](docs/configure-mcp.md) — create `.tuffy/mcp.json` (gitignored) and
list any MCP servers you want connected. Tuffy connects to each one at startup and registers its
tools as `<server>_<tool>`; a server that fails to connect is skipped with a warning and never
blocks startup.

### 5. (Optional) Adding skills
See [docs/configure-skills.md](docs/configure-skills.md) — drop a folder into
`./.tuffy/skills/<name>/` with a `SKILL.md` (YAML frontmatter with `name`/`description`, plus a
markdown body of guidance). Optionally add a `tools.py` or an `mcp.json`. See
`.tuffy/skills/code-review/` for a working example.

### 6. Running the Agent
Run the main script to start your chat session:
```bash
python3 main.py
```

To start directly in Voice Mode, pass the `--voice` flag:
```bash
python3 main.py --voice
```

### 7. (Optional) Running Tuffy from anywhere with a `tuffy` command
Add a shell function to your `~/.zshrc` (or `~/.bashrc`) so typing `tuffy` in any terminal starts
the agent, regardless of your current directory:
```bash
tuffy() {
  local project_dir="/absolute/path/to/tuffy"
  local python_bin="$project_dir/.venv/bin/python3"
  [ -x "$python_bin" ] || python_bin="python3"
  (cd "$project_dir" && "$python_bin" main.py)
}
```
Then `source ~/.zshrc` (or open a new terminal tab) once. It runs in a subshell, so your working
directory is untouched after Tuffy exits, and it prefers the project's own `.venv` python if one
exists.

### 8. (Jetson Orin) One-shot setup
If you're deploying Tuffy on a Jetson Orin (e.g. copied over via a pendrive rather than
`git clone`), run [scripts/setup_jetson.sh](scripts/setup_jetson.sh) instead of steps 2–6 above —
it installs `uv` if missing, runs `uv sync`, rebuilds `llama-cpp-python` with CUDA support for
the Tegra GPU (a wheel built on another machine won't have Jetson's GPU backend), and launches
the app:
```bash
bash scripts/setup_jetson.sh
```
Re-run it any time after pulling new dependencies or model weights — it's idempotent.

---

## Project Structure

```
main.py                 Entry point: startup wiring (skills/MCP discovery), then the input loop
src/
  cli/                  Interactive terminal chat: banner, spinner, slash commands, turn loop
  engine/               Provider-agnostic ReAct tool-calling loop (turn_engine, stream_parser, tool_dispatch)
  identity.py           Fixed self-model (name, capabilities) — never LLM-written
  memory.py             Elastimem-backed long-term memory (facts/episodic/lessons) + the `remember`/`recall` tools
  settings.py           Persisted user settings (.tuffy/settings.json) — default model id
  vision.py           Image encoding + IMAGE_SENTINEL protocol for vision tool results
  voice/              Local Speech-to-Text (Whisper) and Text-to-Speech (Piper) wrappers + interactive audio loop
  llm/                   Model-provider interface + adapters (local weights, OpenAI-compatible API)
  models/                Model registry; configs/local.py + configs/api.py hold model cards; weights/ holds gitignored model weight files
  prompts/               All system-prompt text: personas.yaml + templates.py
  tools/                 Native tools by domain (editing/coding/research/system) + MCP client
  skills/                Discovery/loading for ./.tuffy/skills/*/ capability packs
.tuffy/                 Agent-owned config (gitignored where noted), industry-standard dotfolder pattern
  skills/                Droppable capability packs (SKILL.md + optional tools.py/mcp.json)
  mcp.json                MCP server config (gitignored)
docs/                   Configuration guides (models, MCP, skills, tools, CLI reference)
scripts/                Setup scripts (setup_jetson.sh — bootstrap + run on Jetson Orin)
data/memory/            JSON-backed long-term memory store
agent_workspace/        Sandboxed file I/O root for the agent's file/code tools
.env                    API keys (gitignored) — read by the API provider if not already exported
.env.example            Template for .env — copy and fill in your keys
```

Every folder above has its own README with more detail — start at [src/README.md](src/README.md)
for the full map, [docs/](docs/) for configuration how-tos, or [ARCHITECTURE.md](ARCHITECTURE.md)
for the design rationale behind the tool/model-provider/skills systems.
