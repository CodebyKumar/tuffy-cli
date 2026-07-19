# tuffy_core

This package is the **public API surface** for external consumers (e.g., frontend backends like `tuffy-ui/backend` or third-party orchestrators). 

Everything inside the `src/` directory is internal implementation detail. **`tuffy_core` is the only officially supported import path from outside this repository.**

---

## High-Level Purpose

Rather than introducing new dependencies, `tuffy_core` acts as a thin wrapper around the underlying engine and CLI machinery (e.g., `src.cli.session.Session`, `src.engine.turn_engine`, `src.voice`). It exposes clean, structured API objects and returns standard Python datatypes (lists, dicts, iterators of events) instead of formatting and printing to `stdout`.

---

## Key Exports

`tuffy_core` exports the following public classes and helper functions:

### 1. Session & Execution
* **`create_session(model_id: str = None) -> AgentSession`**: Instantiates and returns a ready-to-use session. This handles loading model configs, initializing/attaching the Elastimem database, and computing token budgets.
* **`AgentSession`**: A thin proxy representing the active chat. Exposes methods to run commands, fetch status snapshots, stage image attachments, switch models, and query MCP servers.
* **`run_turn_stream(session: AgentSession, text: str) -> Iterator[TurnEvent]`**: Streams execution events (Thoughts, ToolCalls, Observations, streamed tokens) of a single turn. Callers iterate over this to feed live updates (e.g., to a WebSocket stream).

### 2. Inspecting Registries & Capabilities
* **`list_tools()`**: Returns a list of all registered tools (native, MCP, or skill-based).
* **`list_skills()`**: Returns a list of all installed skill capability packs.
* **`list_models()`**: Returns all registered local and API model cards.

### 3. Long-Term Memory
* **`memory_summary()`**: Returns database stats (number of facts, sessions, lessons, path, and size).
* **`memory_search(query: str)`**: Conducts a semantic recall query against the SQLite/Elastimem memory store.

### 4. Voice Services
* **`WhisperSTT`**: Speech-to-Text transcriber wrapper using local Whisper weights.
* **`PiperTTS`**: Text-to-Speech synthesizer wrapper using local Piper voices.

---

## Lazy-Loading and Dependency Safety

Voice capability is treated as an optional feature. Heavy libraries (like `pywhispercpp` and `piper`) may not be installed on lightweight or text-only client environments.

To prevent `ImportError` when importing `tuffy_core`, **voice classes are designed with lazy imports**:
- Imports of `src.voice` packages are placed inside the constructor (`__init__`) and execution methods rather than at module scope.
- Text-only consumers can import `tuffy_core` and run chat sessions safely without any voice dependencies.
- If an environment tries to instantiate `WhisperSTT` or `PiperTTS` without having the voice packages installed, it will raise a clean `ImportError` at construction time.
