# Tuffy UI — Implementation Plan (for review, not yet approved for coding)

> **This is a point-in-time design proposal, not a current-state reference.** Implementation has
> since progressed well past this plan, and some specifics below (the REST API's exact endpoint
> list, the frontend's file layout, the orb's rendering approach) have changed since this was
> written and are now out of date — see the inline notes marking those sections, and prefer
> reading the actual current code (`tuffy-ui/backend/server/app.py` for the real API surface,
> `tuffy-ui/frontend/src/` for the real layout) over this document for anything that matters for
> correctness. Kept for the architectural reasoning/decisions that are still accurate.

Status: **proposal.** Per the architecture-definition instructions, this document analyzes the
existing Tuffy codebase, answers all open architectural questions, and produces the requested
deliverables. No code changes have been made. This supersedes both the earlier
`VOICE_UI_PLAN.md` (in-process web UI idea) and the first draft of this document (which put
`server/`/`voice/` inside the `tuffy` repo) — the desktop-app framing with a separate,
Tuffy-core-consuming repo is the current direction.

**Naming:** the new repo is named **`tuffy-ui`**, not `tuffy-desktop`, decided in this revision.
It is currently a Tauri desktop app specifically, but the name anticipates that a future web or
mobile client would be a reasonable sibling under the same convention (`tuffy-ui` for this Tauri
shell today; nothing stops a distinct `tuffy-web` repo later if a browser-hosted client turns out
to need a different tech stack entirely — the naming commitment made here is just "not
desktop-specific," not a promise of one monorepo for every future frontend).

## Answers to open questions (confirmed with user)

- **Q1 — Integration:** Backend service. Tauri is a Rust/webview shell; it cannot import a Python
  package directly. Tuffy Core runs as a local FastAPI+WebSocket process; Tuffy UI talks to it as
  a client. This is also the only option consistent with the already-locked WebSocket/FastAPI
  decisions.
- **Q2 — Sessions:** Single session only for v1. Avoids the local-LLM concurrency problem
  (`llama-cpp-python`'s `Llama` instance is not safely callable from concurrent requests) entirely
  rather than building a queue for a capability not yet needed. (See also the `SessionManager`
  abstraction in §3, added this revision, which keeps this a v1 *policy* rather than a
  hard-coded architectural limit.)
- **Q3 — Memory:** Shared. Matches today's behavior (one Elastimem singleton, one user) and keeps
  terminal and desktop sessions continuous with each other.
- **Q4 — Voice location:** Inside the Tuffy UI backend (Python), not Tauri/Rust — see the
  repo-location revision in §0 below for exactly where that Python code now lives. `whisper.cpp`
  and Piper both have workable Python bindings; keeping STT/TTS resident in one process alongside
  the LLM client means one model-loading/resource-management story, and `tuffy --voice` (terminal,
  still inside the `tuffy` repo) can share the exact same voice code the desktop app uses instead
  of a parallel Rust implementation.
- **Q5 — Deployment:** Developer mode first. Plan targets a working `uv run` backend +
  `tauri dev` frontend across Mac/Jetson/Linux. Packaging (Tauri bundler, appliance/kiosk mode on
  Jetson) is a later phase, not designed in detail now.
- **Q6 — Thin Client vs. Integrated Runtime:** Thin Client. This is implied by Q1 but worth
  stating as its own decision since it's arguably the single biggest architectural fork in the
  project. Integrated Runtime (embedding a Python interpreter inside the Tauri/Rust process, e.g.
  via `pyo3`) would let the desktop app skip running a separate backend process, but it reintroduces
  exactly the coupling Q1 rejected — the desktop shell would need to match the backend's Python
  version and native extension ABI (`llama-cpp-python`'s compiled bindings, whisper.cpp's
  bindings) at build time, cross-platform, per architecture (macOS arm64/x86_64, Jetson aarch64,
  Linux x86_64). That is a substantially harder build/CI matrix than "spawn a subprocess and speak
  WebSocket to it." Thin Client — Tauri → FastAPI → Tuffy Core, as a separate OS process — is the
  recommendation, and is what the rest of this document assumes throughout.

---

## 0. Revision: Where the Backend Code Lives

The previous draft put a new `src/server/` and `src/voice/` inside the `tuffy` repo itself. On
reflection (and per reviewer feedback) that's the wrong home for it long-term: `server/` and
`voice/` are exactly the kind of code that grows — new WebSocket message types, new TTS voices,
device-handling edge cases — and if that growth happens inside `tuffy`, the repo stops being
"Tuffy = AI Core" and slowly becomes "Agent + Server + Voice + Desktop support," which is the
opposite of the project's own stated design principle (`ARCHITECTURE.md` principle 5).

**Decision: `src/server/` and `src/voice/` move out of `tuffy` entirely, into `tuffy-ui/backend/`.**
`tuffy` gains nothing except the public package surface described below — no server code, no
voice code, no FastAPI/uvicorn/whisper.cpp/Piper dependencies in its `pyproject.toml`.

This raises the "Tuffy Core Packaging Requirement" from the previous draft from a same-repo
code-review discipline to an actual cross-repo install boundary:

```python
from tuffy import create_session, AgentSession, run_turn_stream
```

- `create_session(model_id: str | None = None) -> AgentSession` — wraps today's `Session`
  construction + `reconfigure_for_model` + memory attachment (currently done ad hoc in `main.py`
  lines 43–61).
- `AgentSession` — a thin public wrapper around the internal `Session`/`ModelAgent`, exposing only
  what a frontend needs (`history`, `switch_model()`, `system_message()`), not every internal
  field.
- `run_turn_stream(session, text) -> Iterator[TurnEvent]` — the public entry to
  `turn_engine.run_turn()`, already generator-based and already frontend-agnostic; this just
  gives it a stable import path.

Because `tuffy-ui/backend/` is now a genuinely separate repo, `tuffy` must become an actually
installable package — `tuffy-ui/backend/pyproject.toml` depends on it the same way it already
depends on `elastimem`:

```toml
[tool.uv.sources]
tuffy = { path = "../../tuffy", editable = true }
```

(A git-URL source, matching the existing `elastimem` pattern, is the equivalent option once
`tuffy-ui` is not always checked out as a literal sibling folder — either is fine for v1 developer
mode; the path-based source is simplest while both repos live in the same workspace.)

This means the previous draft's "held to the same import discipline as an external consumer, even
though it's the same repo" caveat is no longer a caveat — `tuffy-ui/backend/` **is** an external
consumer, enforced by the package boundary itself, not by code review. Any temptation to reach
into `tuffy.src.models.xxx` or `tuffy.src.memory` directly simply won't resolve as an import,
since only `tuffy/__init__.py`'s exports are the public contract.

---

## 1. Integration Strategy

### Architecture

```
┌──────────────────────────────────────┐         ┌───────────────────────┐
│              tuffy-ui                  │         │         tuffy           │
│                                        │         │                       │
│  src-tauri/ (Rust shell)               │         │  from tuffy import     │
│   └─ webview (HTML/CSS/TS)             │         │    create_session,     │
│       - Orb (Canvas)                  │         │    AgentSession,       │
│       - Conversation view              │         │    run_turn_stream     │
│       - Text/voice input               │         │                       │
│                                        │         │  src/engine/           │
│  backend/ (Python, installs `tuffy`)   │  in-proc │  src/llm/              │
│   ├─ server/  FastAPI + WebSocket      │◄────────│  src/tools/            │
│   ├─ voice/   STT/TTS/VAD              │  import │  src/memory.py         │
│   └─ bridge/  SessionManager,          │         │  src/cli/ (terminal,   │
│               event→JSON translation   │         │            unchanged)  │
│                                        │         │                       │
│         ▲ WS/HTTP, ws://localhost:PORT │         │  (installable via     │
│         │ (Tauri ↔ backend, same host) │         │   uv path/git source) │
└──────────┼──────────────────────────┘         └───────────────────────┘
           │
     Tauri process manages backend/ as a child process
```

Two boundaries now exist, not one: Tauri ↔ `backend/` is a WebSocket/HTTP process boundary;
`backend/` ↔ `tuffy` is a Python package-import boundary. Both are real, enforced boundaries —
neither is "just a convention."

### Tradeoffs (Option A vs B vs Hybrid)

| | Direct Python import (A) | Backend service (B) | Hybrid |
|---|---|---|---|
| Feasible with Tauri | No — Tauri's Rust process can't import a Python module in-process | Yes | Only via B underneath |
| Matches locked tech (FastAPI, WebSocket) | Conflicts | Matches directly | Matches |
| Terminal/desktop code sharing | N/A | Both talk to the same `tuffy` package; voice code lives once in `tuffy-ui/backend/voice/`, importable by `tuffy --voice` too if that terminal mode later imports from `tuffy-ui` — see note below | Same as B |
| Coupling | Tight — desktop shell tied to Python runtime | Loose — desktop shell only knows a WebSocket/JSON contract; backend only knows `tuffy`'s public exports | Loose |
| Startup complexity | Lower (no separate process) | Backend process must be spawned/health-checked by the shell | Same as B |

**Recommendation: Option B, backend service, with `backend/` now living in `tuffy-ui` rather than
`tuffy`.** It's the only option compatible with Tauri as specified, reuses `ARCHITECTURE.md`'s
existing "terminal I/O stays out of the core" seam directly (the FastAPI layer is just another
consumer of `tuffy`'s public API, same category as `src/cli/` is today), and now keeps `tuffy`
honestly limited to agent-core code forever, not just "thin by convention."

**Note on `tuffy --voice`:** moving `voice/` into `tuffy-ui/backend/` means the terminal's future
voice mode has a choice: either (a) `tuffy` takes a dependency on `tuffy-ui/backend` for voice
code specifically (inverting the intended dependency direction — avoid this), or (b) `tuffy`'s
terminal voice mode duplicates a small STT/TTS/VAD wrapper, or (c) voice code becomes its own
third small package (`tuffy-voice`?) both `tuffy`'s CLI and `tuffy-ui/backend` depend on. Flagging
this explicitly rather than picking now: Phase 3 (terminal voice, §7) is the right time to decide,
once `backend/voice/`'s actual shape exists to evaluate extracting from.

---

## 2. Repository Layout

```
workspace/
├── tuffy/                              existing repo — strictly agent-core, unchanged in shape
│   ├── main.py                         terminal entry point (unchanged behavior)
│   ├── tuffy/__init__.py               NEW — public package surface (create_session, etc.)
│   ├── src/
│   │   ├── cli/                        existing terminal frontend (unchanged)
│   │   ├── engine/                     unchanged
│   │   ├── llm/                        unchanged
│   │   ├── tools/                      unchanged
│   │   ├── memory.py                   unchanged
│   │   └── ...                         unchanged
│   ├── pyproject.toml                  unchanged deps — no fastapi/uvicorn/whisper/piper here
│   └── tests/
│
└── tuffy-ui/                           NEW, separate repo — Tauri desktop app + its backend
    ├── src-tauri/                      Tauri/Rust shell
    │   ├── Cargo.toml
    │   ├── tauri.conf.json
    │   └── src/
    │       ├── main.rs                 spawns/manages backend/ process, opens window
    │       └── backend_lifecycle.rs    health-check + start/stop of the Python process
    │
    ├── backend/                        Python — installs `tuffy` as a dependency
    │   ├── pyproject.toml              depends on tuffy (path/git source) + fastapi, uvicorn,
    │   │                                whisper.cpp binding, piper, silero-vad
    │   ├── server/
    │   │   ├── __init__.py
    │   │   ├── app.py                  FastAPI app, REST endpoints, standalone-runnable (§7 new)
    │   │   ├── ws.py                   WebSocket handler: event stream bridge
    │   │   └── schemas.py              Pydantic models for REST/WS payloads (the versioned contract)
    │   ├── voice/                      STT/TTS/VAD
    │   │   ├── __init__.py
    │   │   ├── base.py                 VoiceIO interface (transcribe/synthesize/is_speech)
    │   │   ├── stt_whisper.py          whisper.cpp binding implementation
    │   │   ├── tts_piper.py            Piper implementation
    │   │   ├── vad.py                  Silero VAD implementation (webrtcvad fallback)
    │   │   └── wakeword.py             extension point, not implemented in v1 (§ Voice Architecture)
    │   └── bridge/
    │       ├── __init__.py
    │       ├── session_manager.py      SessionManager (NEW — see §3)
    │       └── renderer.py             iterates tuffy.run_turn_stream(), emits WS messages
    │
    ├── frontend/                        webview frontend (vanilla JS/TS, no React/Next/Electron)
    │   ├── index.html
    │   ├── src/                         STALE below — see actual current layout note after this tree
    │   │   ├── main.ts
    │   │   ├── state/                   orb.ts (Three.js shader-blob renderer + state machine), ws-client.ts
    │   │   ├── components/              conversation.ts, settings-sheet.ts, input.ts, device-picker.ts, dropdown.ts, image-attach.ts, slash-palette.ts
    │   │   ├── audio/                   mic.ts, mic-worklet.ts, tts-player.ts
    │   │   └── lib/                     api.ts, icons.ts
    │   └── styles.css
    ├── package.json                    minimal — bundler only (Vite for TS, no framework)
    └── README.md
```

> The `frontend/` subtree above is stale: it shows an early flat `frontend/*.ts` layout. The
> actual current layout groups files under `frontend/src/{state,components,audio,lib}/` as
> annotated inline above — see `tuffy-ui/frontend/src/` directly for the real, current file list.
> The orb is also no longer a plain Canvas renderer — see the note in §6 below.

`tuffy-ui/backend/` depends on `tuffy` only through `from tuffy import ...` (§0) — never reaches
into `tuffy/src/...` internals, enforced by the package boundary itself. `tuffy` has zero
knowledge of `tuffy-ui`'s existence beyond exposing a stable public API that any consumer,
including a future one, could use the same way.

---

## 3. Session Lifecycle

### SessionManager abstraction (new this revision)

Even though v1 is single-session (Q2), the session-handling code should be written behind a
`SessionManager` from day one rather than a single ad hoc global `Session` variable in
`server/app.py`:

```python
class SessionManager:
    def get_active_session(self) -> AgentSession: ...
    def create_session(self, model_id: str | None = None) -> AgentSession: ...
    # v1 policy: get_active_session() always returns the same one session;
    # create_session() is only ever called once, at backend startup.
```

This is a real recommendation, not decoration: today's plan already assumes "one `Session`, one
process" is a *policy* choice (§Q2's reasoning is about local-LLM concurrency, not about
`SessionManager` being architecturally incapable of more). Writing the one-session behavior as a
trivial `SessionManager` implementation now means a future multi-chat feature (Chat A / Chat B /
Chat C) is a change to `SessionManager`'s internals and the addition of an inference queue — not
a rewrite of every call site in `server/` that currently assumes a single global session object.

1. **App startup:** Tauri's `main.rs` launches on user double-click. It spawns the `backend/`
   process as a child process (`uv run python -m server.app` or similar, working directory =
   `tuffy-ui/backend/`, resolved via a configured path or a bundled sidecar in packaged mode
   later). Tauri polls a health endpoint (`GET /health`) until the backend is ready, then opens
   the webview window pointed at the bundled frontend, which immediately opens a WebSocket to the
   backend.
2. **Session creation:** On backend startup (not on WebSocket connect — see §7 Backend
   Independence Requirement below for why this matters), `SessionManager.create_session()` calls
   `tuffy.create_session(...)` (Q2: single-session), which runs `reconfigure_for_model` against
   the shared Elastimem store (Q3) exactly as `main.py` does today. If the desktop app reconnects
   (e.g. window reload), the backend reattaches to the existing session via
   `get_active_session()` rather than creating a new one.

### Session Ownership Rules

Explicit answers, since "session" is used loosely elsewhere in this document:

- **One session per window, or per chat?** Per window for v1 — and since Q2 limits v1 to a single
  window, this is equivalent to "one session per running backend process." Multiple concurrent
  chats within one window (tabs, a chat switcher) are out of scope until multi-session is
  revisited — now explicitly a `SessionManager` extension, not an architecture change (see above).
- **Persistent or temporary?** Persistent by default — the session's chat history and Elastimem
  memory persist across app restarts (matches today's terminal behavior: `data/memory/tuffy.db`
  already survives process restarts). Closing and reopening the desktop app resumes the same
  conversation history via `GET /history`, it does not start fresh.
- **Session creation owner:** the backend, not the frontend — the frontend never constructs
  session state, it only requests the current session's history/status via `SessionManager`. This
  keeps session construction (including `reconfigure_for_model`) a single code path, rather than
  something the frontend could get wrong or duplicate.
- **Session destruction:** the backend keeps the session alive for its own process lifetime,
  independent of WebSocket connect/disconnect — closing the desktop window (WebSocket disconnect)
  does *not* destroy the session, since Tauri also shuts the backend process down at the same time
  in the common case (see Shutdown below). A session is only actually destroyed when the backend
  process exits.
- **Conversation storage / history loading:** chat history lives in the `AgentSession` object in
  memory for the running process (same as today's `Session`), with Elastimem separately
  persisting long-term facts. `GET /history` lets the frontend rehydrate the visible transcript on
  reconnect without the backend needing to persist full turn-by-turn history to disk beyond what
  Elastimem already captures — if full transcript persistence across backend restarts becomes a
  requirement later, that's an explicit additive feature, not assumed here.

3. **Turn execution:** User types or speaks. Voice path: audio frames stream to the backend over
   the WebSocket as **binary frames** (see API Design below — this replaced base64-in-JSON in this
   revision), Silero VAD gates when an utterance is "done," whisper.cpp transcribes to text.
   Either path (typed or transcribed) calls the same `tuffy.run_turn_stream(session, text)` —
   `backend/bridge/renderer.py` iterates that generator and emits each `TurnEvent` as a JSON
   WebSocket message. If voice output is enabled, the final answer text (or streamed token chunks
   in a later phase) is also passed to Piper, and synthesized audio is streamed back as binary
   WebSocket frames.
4. **Voice flow (end-to-end):**
   ```
   mic → Silero VAD (backend) → buffered utterance → whisper.cpp → text
       → tuffy.run_turn_stream() → TurnEvents → WebSocket JSON → orb state + transcript (UI)
       → final answer text → Piper → audio → WebSocket binary frames → webview playback
   ```
   `tuffy`'s engine and every module under `src/engine/`, `src/llm/`, `src/tools/` never see that
   the input was spoken — identical to how `IMAGE_SENTINEL` lets vision attach without the engine
   knowing the source was a webcam. This remains true even with `voice/` now living outside the
   `tuffy` repo: the boundary the brief cares about ("Tuffy Core should remain unaware of voice")
   is actually *strengthened* by voice code no longer being able to reach into engine internals at
   all, only able to call `run_turn_stream(session, text)` like any other consumer.
5. **Shutdown:** Closing the desktop window triggers Tauri to send a shutdown signal to the
   backend child process (graceful WebSocket close + SIGTERM). The backend's own exit path calls
   into `tuffy`'s existing `os._exit()` Metal-teardown workaround indirectly (via however
   `tuffy.create_session`'s underlying `Session`/`ModelAgent` unload path works) — this must be
   verified to still avoid the Metal static-destructor crash when exit is triggered by an external
   signal instead of `KeyboardInterrupt`/`EOFError`, same caveat as the previous draft, now
   slightly more indirect since the backend process isn't `tuffy`'s own `main.py`.

---

## 4. API Design

### REST endpoints

> **Stale — this table reflects an early implementation pass.** See
> `tuffy-ui/backend/server/app.py` for the real, current endpoint list, which has grown to also
> cover tools, memory (including per-fact history/forget/quarantine), skills (including per-skill
> detail), MCP servers, per-model info, session status, running slash commands, and shutdown —
> significantly more surface than the four endpoints originally planned here.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check for Tauri's startup polling, and for standalone debugging (§7) |
| `GET` | `/models` | List available model cards (mirrors `/models` slash command) |
| `POST` | `/models/switch` | Switch active model (mirrors `Session.switch_model()`) |
| `GET` | `/history` | Fetch current session's chat history (for UI reload/reattach) |

### WebSocket: `ws://localhost:<port>/ws`

Single connection for the (single, per Q2) session, via `SessionManager.get_active_session()`.
Two distinct transport modes on the same connection:

**JSON text frames** — every control/event message is a JSON envelope carrying an explicit API
version from day one, so `tuffy-ui` and `tuffy` (now genuinely separately-versioned packages, per
§0) can detect a mismatch instead of silently drifting:

```json
{"type": "assistant_chunk", "api_version": 1, "payload": {"text": "Hello"}}
```

**Binary frames — audio only** (revised this pass): mic input and TTS output are sent as raw
WebSocket **binary frames**, not base64-encoded inside JSON. Base64 adds ~33% size overhead and
extra encode/decode CPU work on every audio chunk, for no benefit — WebSocket already supports
binary frames natively, and audio has no need to be human-readable or JSON-nested. A minimal
framing convention (e.g. a 1-byte frame-type prefix, or two separate logical streams multiplexed
by message type at the WebSocket protocol level) distinguishes mic-audio-in from tts-audio-out;
exact framing is an implementation detail for Phase 2, not decided here, but "binary frames for
audio, JSON only for events/control" is the decided architecture.

**Client → server:**

| type | transport | payload | meaning |
|---|---|---|---|
| `text_input` | JSON | `{"text": str}` | user typed a message |
| `audio_chunk` | **binary frame** | raw PCM bytes | streamed mic audio for VAD/STT |
| `voice_output_toggle` | JSON | `{"enabled": bool}` | turn TTS playback on/off |
| `interrupt` | JSON | `{}` | user wants to stop mid-turn (barge-in) |

**Server → client** (one message per engine `TurnEvent`, plus voice-specific ones):

| type | transport | payload | maps to |
|---|---|---|---|
| `status` | JSON | `{"state": "loading"\|"idle"\|"listening"\|"thinking"\|"tool_running"\|"speaking"\|"error"}` | orb state driver (`loading` added this revision, see Orb Architecture) |
| `assistant_started` | JSON | `{}` | turn begins (before any `Thought`/tool activity) |
| `thought` | JSON | `{"text": str}` | `Thought` event |
| `tool_started` | JSON | `{"tool": str, "group": str, "args": {...}}` | `ToolCall` event |
| `tool_finished` | JSON | `{"tool": str, "result": str}` | `ToolResult` event |
| `assistant_chunk` | JSON | `{"text": str}` | `Token` event (streamed answer text) |
| `assistant_finished` | JSON | `{}` | `Done` event |
| `assistant_failed` | JSON | `{"error": str}` | `Failed` event |
| `transcript_partial` | JSON | `{"text": str}` | Phase-6 streaming STT partial |
| `transcript_final` | JSON | `{"text": str}` | STT finished, text handed to `run_turn_stream` |
| `audio_chunk` | **binary frame** | raw PCM bytes | streamed TTS output |

This is a direct, mechanical translation of `tuffy`'s existing `TurnEvent` types — no new event
types needed on the `tuffy` side, only a JSON serialization at the `tuffy-ui/backend` boundary.
Defining these names now (rather than "whatever `TurnEvent` happens to be called this week") is
what keeps frontend and backend from drifting apart — treat this table as the contract
`backend/server/schemas.py`'s Pydantic models must implement exactly, and as the source
`frontend/ws-client.ts`'s event handling is written against.

### Tool Visibility Model (revised this pass — simplified for normal users)

Tuffy's tool registry (`src/tools/registry.py`) already groups tools (native, MCP, skill-shipped)
for display purposes only (`group` is never used to hide a tool from the model — see
`ARCHITECTURE.md` principle 2). `tool_started`'s payload passes through the existing `group`
metadata, but the **default conversation view does not show raw tool names or arguments** to
normal users — only a short, friendly per-group label:

| `group` | Default user-facing label |
|---|---|
| `research` (web_search, etc.) | "Searching..." |
| `editing` (file read/write) | "Reading file..." / "Writing file..." |
| `memory` (remember/recall) | "Using memory..." |
| anything else / MCP / skills | "Using a tool..." (generic fallback) |

Raw tool name, full arguments, and full results are **not** shown in the default conversation
view at all — that level of detail moves entirely into the Developer Panel (hidden by default,
per the original brief's Core UI Components), which shows the full `tool_started`/`tool_finished`
payload verbatim as a debugging/transparency aid for users who opt into it. This is a tightening
from the previous draft's "summarized mode still shows args/results inline, just more compactly"
— the reviewer's point that normal users shouldn't see tool args/results at all (not even
summarized) is adopted directly; verbose detail is Developer Panel only, not a toggle in the main
conversation view.

---

## 5. Voice Architecture

- **STT:** `whisper.cpp` via a Python binding (e.g. `pywhispercpp`), loaded once at backend
  startup alongside the LLM client, resident in-process — same "load once, keep resident" pattern
  `llama-cpp-python` already uses. Runs on buffered utterances (VAD-delimited), not continuous
  streaming, for v1.
- **TTS:** Piper, invoked per finished answer (v1) via its Python bindings or a lightweight
  subprocess call; output PCM streamed back over the WebSocket as binary frames as it's produced.
- **VAD:** **Silero VAD** (small ONNX model, CPU-only) is the recommendation — `webrtcvad`'s
  energy-based detection is markedly more prone to false triggers on keyboard clatter, cooling
  fans, and Jetson's typically noisier ambient/cheap microphone hardware, all of which would
  otherwise cost real debugging time chasing "why did it start listening to my typing." Silero's
  small model-based approach handles these cases meaningfully better at a still-negligible compute
  cost (single-digit ms per frame on CPU). `webrtcvad` remains a viable lower-effort fallback if
  Silero's ONNX runtime proves awkward to build/ship on Jetson specifically. VAD gates when to
  start/stop buffering microphone audio for STT, and separately drives the "Listening" orb state's
  reactivity to input volume in real time.
- **Wake word (extension point, not implemented in v1):** the brief calls out wake word as a
  future enhancement; this plan makes sure the voice pipeline's *shape* doesn't have to be
  redesigned when that happens, by reserving the slot now:
  ```
  mic → [wake word, not implemented v1] → VAD → whisper.cpp → text → tuffy
  ```
  `backend/voice/wakeword.py` exists in the repo layout (§2) as an empty/stub `WakeWord`
  interface (`listen_for_wake(audio_frame) -> bool`) from day one, even though v1's mic pipeline
  calls straight into VAD without it. Future candidates: **OpenWakeWord** (fully offline, ONNX-based,
  same deployment profile as Silero VAD) or **Porcupine** (commercial, very low false-positive
  rate, but requires a license key — a local-first/offline-by-default tradeoff to weigh against
  OpenWakeWord when this is actually implemented, not decided now). The only architectural
  commitment made in v1 is the interface slot in the pipeline, not a library choice.
- **Audio device handling:** Microphone capture happens in the **webview frontend** via
  `MediaRecorder`/`getUserMedia` (browser/webview-native, works identically on Mac/Linux/Jetson
  webviews without a native audio library dependency in Rust or Python) and is streamed to the
  backend as **binary WebSocket frames** (§4). Playback of TTS output similarly uses the webview's
  native `AudioContext`, receiving binary frames. This sidesteps needing platform-specific audio
  device enumeration code in Tauri or Python entirely — the browser engine already handles device
  selection/permissions per-platform. Specifics to design for rather than discover later:
  - **Input/output device selection:** `navigator.mediaDevices.enumerateDevices()` gives a device
    list in the webview; the UI should expose a simple device picker (mic + speaker) in the
    Developer Panel or a settings affordance, defaulting to the OS default device rather than
    forcing a choice up front.
  - **Bluetooth headphones:** work through the same `getUserMedia`/`AudioContext` path with no
    special-casing needed, but Bluetooth mic profiles (HFP/HSP) typically downgrade audio quality
    significantly compared to a wired/USB mic — worth a plan-level note that STT accuracy may
    visibly degrade on Bluetooth input, not a code problem to solve, a behavior to expect.
  - **USB microphones:** same path, generally the most reliable option on Jetson dev kits that
    lack a built-in mic.
  - **Device hotplugging:** the webview fires a `devicechange` event on
    `navigator.mediaDevices` when devices connect/disconnect; the frontend should listen for this
    and, at minimum, gracefully handle an active recording stream's device disappearing
    mid-utterance (stop cleanly, surface the `error` orb state) rather than crashing — full
    "reconnect and resume" behavior is a nice-to-have, not a v1 requirement.
  - Fallback/edge case: Jetson dev kits without any mic/speaker attached simply can't use voice
    mode; text mode still works, and the UI should detect zero available audio input devices at
    startup and disable the mic affordance rather than showing a control that silently fails.
- **Voice remains a wrapper around text, full stop.** This is one of the strongest constraints in
  the original brief and the reviewer explicitly flagged it as something to keep: there is **one**
  agent, `tuffy.run_turn_stream(session, text)`, and voice is purely an input/output adapter
  around it — never a "Voice Agent" distinct from a "Text Agent." Every design decision above
  (VAD, STT, wake word, TTS) produces or consumes plain text at the `tuffy` boundary; nothing
  about voice ever reaches past that boundary into engine internals.

### Models and Assets (new section this revision)

No asset-location strategy existed in the previous draft — worth fixing before Phase 3 needs
somewhere to actually put files:

```
~/.tuffy/models/
├── llm/       existing location today is src/models/weights/ inside the tuffy repo (gitignored);
│              this plan does not relocate existing LLM weights, listed here only for completeness
├── whisper/   whisper.cpp GGML model files (e.g. ggml-small.en.bin)
└── piper/     Piper voice models (.onnx + .onnx.json per voice)
```

`whisper/` and `piper/` are new, backend-managed download/cache locations, analogous to how
`src/models/weights/` already works for LLM GGUF files today but scoped under `~/.tuffy/` (a
user-home location) rather than inside either repo, since these are runtime assets tied to the
user's machine, not source-controlled artifacts — consistent with `~/.tuffy/logs/` (§ Logging)
already being the established pattern for user-machine state that isn't part of either codebase.

---

## 6. Orb Architecture

### State model (revised this pass — `loading` added)

```
loading ──(backend ready + models loaded)──► idle
idle ──(user starts speaking / VAD trigger)──► listening
listening ──(utterance complete, STT done)──► thinking
thinking ──(tool_call event)──► tool_running ──(tool_result)──► thinking
thinking ──(answer_start/token events)──► speaking
speaking ──(done event)──► idle
any state ──(failed event)──► error ──(next input)──► idle
```

`loading` covers the startup window where the backend process, LLM, whisper.cpp, and Piper are
all initializing — without it, the orb would either show nothing or default to `idle` while the
app is, from the user's perspective, frozen (matching the < 3s startup target in Performance
Targets below, and giving that target somewhere to render feedback if a particular machine/model
combination takes longer). The frontend enters `loading` immediately on window open and leaves it
only once `GET /health` succeeds *and* an initial `status: idle` message arrives over the
WebSocket — a simple health-check pass alone isn't suficient, since the backend process can be
"up" as a process before its model/voice components have finished loading.

States map directly onto the `status` WebSocket message (§4), which is itself a direct
translation of existing `TurnEvent` types plus two voice-specific states (`listening`,
`tool_running` — note `tool_running` is a new refinement of "thinking" the desktop UI can display,
not a new engine event; it's derived client-side from seeing a `tool_call` without a following
`done`/`token`) plus `loading`, which is purely a UI-side concept driven by the health-check/first
WebSocket message sequence rather than any `tuffy` engine event.

### Animation system

> **Stale — the actual implementation uses Three.js, not plain Canvas 2D.** The orb renders as a
> full-screen WebGL shader (a soft-edged, perfectly circular disc with interior motion driven by
> layered noise — no faceted 3D geometry, no per-angle silhouette warp), with a 2D Canvas
> fallback only for environments without WebGL. See `tuffy-ui/frontend/src/state/orb.ts` for the
> real current implementation; the plan below (as originally written) is kept for the state-model
> reasoning, which is still accurate.

Canvas 2D (not WebGL/Three.js, per the "avoid heavy frameworks" instruction) — a single
`<canvas>` element with a `requestAnimationFrame` loop reading current orb state + (for
`listening`) live input volume level from VAD, and (for `speaking`) either token arrival rate or
raw TTS audio amplitude, to drive simple parametric animation (pulse rate/amplitude, distortion).
No 3D engine, no particle library — matches "living assistant, not a sci-fi dashboard."

### State transitions

Implemented as a small explicit state machine in `frontend/orb.ts` (not implicit in rendering
code) so transitions are testable independent of the Canvas drawing logic — `ws-client.ts`
dispatches incoming server messages to the state machine, the state machine emits the current
state + animation parameters, `orb.ts`'s render loop reads only current state.

---

## 7. Development Phases (with effort estimates)

| Phase | Scope | Est. effort |
|---|---|---|
| **1. `tuffy` public package surface** | `tuffy/__init__.py` (`create_session`, `AgentSession`, `run_turn_stream`), inference lock around `LLMProvider`, confirm Elastimem shared-singleton usage needs no change (Q3 already matches today's behavior). No `tuffy-ui` code yet — this phase is entirely inside the `tuffy` repo and is independently testable/usable by the existing terminal. | 2–3 days |
| **2. `tuffy-ui/backend` scaffold + FastAPI/WebSocket bridge** | New repo; `backend/pyproject.toml` depends on `tuffy` (path source); `SessionManager`, `server/` (REST + WS, binary audio framing wired even before voice exists so the framing convention is validated early), `bridge/renderer.py` translating `TurnEvent`→JSON. Manual testing via a WS client (e.g. `websocat`) before any frontend exists. Must be runnable standalone (§ Backend Independence below) from this phase onward. | 4–5 days |
| **3. Voice in the backend (text-mode-testable first)** | `backend/voice/` — VoiceIO interface + whisper.cpp/Piper/Silero VAD implementations, `wakeword.py` stub interface only. Decide `tuffy --voice` terminal-code-sharing question (§1 note) once this exists. Model/asset download-and-cache to `~/.tuffy/models/{whisper,piper}/`. | 4–6 days (includes Jetson wheel/build verification) |
| **4. Tauri shell + webview frontend** | `tuffy-ui` Tauri scaffold, process spawn/health-check/shutdown (including the `loading` orb state's health-check sequencing), orb Canvas + state machine, conversation view (with the simplified default tool-visibility labels), text input, WebSocket client wired to the Phase-2 bridge | 5–7 days |
| **5. Voice in the desktop UI** | Mic capture (`MediaRecorder`) → binary-frame backend audio streaming, TTS binary-frame playback wiring, orb `listening`/`speaking` states driven by real audio, audio device picker + hotplug handling | 3–5 days |
| **6. Streaming refinements** | Partial STT transcripts, incremental TTS playback (speak first sentence while later tokens still generating) | 3–5 days |
| **7. Packaging** | Tauri bundler config, backend distribution strategy (bundled Python env vs. requiring separate `tuffy`+`tuffy-ui/backend` install), Jetson appliance/kiosk mode, wake-word implementation if prioritized | Sized once Phase 1–6 architecture is proven; not estimated now per Q5 |

Total pre-packaging estimate: roughly **21–31 working days**, dominated by Phase 3's platform
verification work and Phase 4's UI build — treat as an order-of-magnitude planning number, not a
commitment, since STT/TTS wheel availability on Jetson (Risk section below) can swing Phase 3
significantly.

### Backend Independence Requirement (new section this revision)

`tuffy-ui/backend/` must be runnable and testable **without Tauri at all**:

```bash
cd tuffy-ui/backend
uv run python -m server.app
```

should start the FastAPI server, load `tuffy`'s session + voice components, and serve
`/health`/`/models`/WebSocket exactly as it does when Tauri spawns it — Tauri is just one possible
process manager for this backend, not a dependency of the backend itself. This matters for:

- **Debugging:** iterating on `server/`/`voice/` without rebuilding/relaunching the Tauri shell
  every time.
- **API testing:** hitting the REST/WebSocket contract directly with `curl`/`websocat`/a test
  script during Phase 2–3 development, before any frontend exists to exercise it.
- **Future clients:** a hypothetical future mobile or web client (§0 naming note) would connect to
  this same standalone backend rather than needing its own bundled copy — the backend's value is
  independent of which frontend happens to be driving it today.

Concretely, this means `server/app.py`'s startup path (spawn `SessionManager`, load voice
components) must not assume any Tauri-provided environment variable, IPC channel, or lifecycle
signal exists — `main.rs`'s `backend_lifecycle.rs` is a caller of this standalone entrypoint, not
a co-designed half of it.

---

## 8. Logging Requirements

No logging strategy exists in the current codebase beyond ad-hoc `print()`/console output — worth
establishing one now rather than debugging a multi-process (Tauri + Python backend) system blind.

- **Location:** `~/.tuffy/logs/`, one subdirectory per source, rotated by size (e.g. 10MB/5 files)
  to avoid unbounded growth on a long-running desktop app or Jetson appliance:
  - `~/.tuffy/logs/backend.log` — FastAPI/WebSocket server, session lifecycle, tool
    dispatch/errors (Python `logging`, replacing today's implicit `print`-based visibility).
  - `~/.tuffy/logs/voice.log` — STT/TTS/VAD activity, model load times, transcription
    confidence/errors — this is the log most needed when triaging "why didn't it hear me."
  - `~/.tuffy/logs/desktop.log` — Tauri/Rust-side: process spawn/health-check/shutdown events,
    webview errors.
  - `~/.tuffy/logs/ws.log` (or a level within `backend.log`) — WebSocket message trace, gated
    behind a debug flag (default off) since full message logging is verbose and only needed when
    diagnosing frontend/backend event-contract drift.
- **Format:** structured (JSON lines) rather than free text, so logs are greppable/parseable
  later without inventing a format under pressure during an incident.
- **Terminal mode:** existing `tuffy` terminal behavior (stdout/ANSI) is unaffected; logging is
  additive file output alongside it, not a replacement for the existing console UX. Since
  `tuffy`'s own `pyproject.toml` gains no new dependencies (§0), this logging setup lives entirely
  in `tuffy-ui/backend/`, not in `tuffy` itself, except for whatever `tuffy`'s existing terminal
  already does.

## 9. Performance Targets

No measurable targets exist today; adding them now gives Phase 1–6 implementation something
concrete to design against rather than "make it fast." Treat these as directional v1 targets, to
be revised once real numbers exist on Mac and Jetson hardware — the point is having a number to
architect toward, not precision:

| Metric | Target | Notes |
|---|---|---|
| Desktop app startup (`loading` → `idle`) | < 3s | Dominated by backend process spawn + model load; the new `loading` orb state (§6) exists specifically so a slower load reads as "starting up," not "frozen" |
| Voice response start (utterance end → first audio out) | < 1s | STT + first-token latency + TTS-first-chunk; likely the hardest target on Jetson specifically — flag if Phase 3 benchmarking shows this is unrealistic for the chosen model sizes, rather than silently missing it |
| Orb animation frame rate | 30+ fps | Canvas 2D `requestAnimationFrame`, should be trivial given no 3D rendering |
| Idle CPU usage | < 5% | With backend running, model loaded, no active turn |
| Idle memory usage | < 500MB excluding loaded model weights | Backend process + Tauri shell + webview, model weights counted separately since they dominate and vary per model card |

These numbers directly inform earlier decisions: the < 1s voice-response target is part of why
STT/TTS/VAD are recommended in-process and resident (§5) rather than spawned per-request, and the
binary-frame audio transport (§4) is chosen partly because base64's ~33% overhead works directly
against this same latency target.

## 10. Packaging / Distribution Targets

Deferred in detail per Q5 (developer mode first), but the distribution *targets* should be named
now so Phase 7 isn't designed from scratch later with no prior thought:

- **Terminal entry points** (in the `tuffy` repo): `tuffy` (existing text mode, unchanged),
  `tuffy --voice` (Phase 3+, terminal voice mode — pending the code-sharing decision in §1's note
  on where voice code should live relative to `tuffy-ui/backend/voice/`).
- **Desktop entry point** (in `tuffy-ui`): launching the Tauri app starts its own backend
  automatically (§3 App startup) — there is no separate `tuffy --desktop` terminal command in this
  revision, since the desktop app is a standalone executable/bundle, not something launched from
  within the `tuffy` terminal. (The previous draft's `tuffy --desktop` convenience launcher is
  dropped as unnecessary complexity — evaluate reintroducing it only if a real workflow need
  for "launch the desktop app from the terminal" shows up.)
- **macOS:** `.app` bundle via Tauri's built-in bundler, code-signing/notarization deferred until
  actual distribution outside the dev machine is needed.
- **Linux:** AppImage via Tauri's bundler (portable, no install-time package manager dependency —
  reasonable default for a personal/small-distribution tool).
- **Jetson:** native launcher (a desktop entry / systemd user service for appliance/kiosk mode),
  building on the AppImage or a native binary once the ARM64 build path is validated; kiosk-style
  auto-launch is explicitly a Phase 7+ concern (Q5), not designed in detail here.
- The core open question for all three (already flagged in Packaging risks below): whether the
  `tuffy-ui/backend` Python environment (now depending on both `tuffy` and the voice libraries)
  ships **embedded inside** the Tauri bundle or as a **separate managed install** the desktop app
  detects/launches. Given the size of the runtime (`llama-cpp-python` + whisper.cpp + Piper +
  model weights, now plus `tuffy` itself as an installed dependency), "separate managed install"
  is the likely direction, but this is explicitly a Phase 7 design decision, not decided here.

## 11. Risk Analysis

### Jetson risks

- `whisper.cpp`/Piper Python binding wheel availability on Jetson's ARM64 + JetPack CUDA
  combination is unverified — `scripts/setup_jetson.sh` (in `tuffy`) already has to source-build
  `llama-cpp-python` for the same underlying reason (no matching prebuilt wheel); `tuffy-ui/backend`
  will likely need an analogous setup script of its own, since these are now separate-repo,
  separate-`pyproject.toml` dependencies. Budget for a source-build path for at least one of
  STT/TTS.
- Running LLM + whisper.cpp + Piper concurrently on an Orin Nano's shared RAM/GPU is the top
  performance risk; needs hands-on benchmarking on real hardware, not spec-based sizing.
- Tauri on Jetson (ARM64 Linux) needs its own webview runtime (WebKitGTK) present — verify it's
  available/installable in the Jetson Linux (L4T) base image before assuming Tauri "just works"
  there.

### Linux risks

- Audio permissioning/device enumeration for `getUserMedia` inside a Tauri webview on Linux
  (WebKitGTK) has historically been less consistent than Chromium/Safari — needs a concrete test,
  not an assumption it matches macOS behavior.
- No Windows target is in scope (matches existing Tuffy platform code, which has no
  Windows-specific paths today) — worth confirming this remains out of scope explicitly.

### Audio risks

- Jetson dev kits commonly ship without a built-in mic/speaker — USB audio device dependency for
  any on-device voice testing.
- Backend-side VAD/STT operates on binary frames streamed over WebSocket from the webview;
  network-like serialization (even over localhost) adds latency and framing complexity compared
  to native audio capture — acceptable for v1 given the "voice as wrapper around text" framing,
  and the binary-frame decision (§4) specifically minimizes the avoidable part of this cost, but
  worth measuring actual round-trip latency early rather than assuming it's negligible.

### Packaging risks

- Deferred by design (Q5), but flagging now: bundling a full Python + `tuffy` (as an installed
  dependency) + `llama-cpp-python` + whisper.cpp + Piper runtime inside a Tauri app is a
  substantially heavier bundle than a typical Tauri app (which usually ships a small Rust binary)
  — packaging strategy will need explicit design work in Phase 7, likely favoring "backend is a
  separate install the desktop app manages/detects" over "backend fully embedded in the app
  bundle," especially for the Jetson appliance case.
- Two separate repos (`tuffy`, `tuffy-ui`) mean two independent version-compatibility surfaces
  now: (a) the WebSocket/REST API in §4 between `tuffy-ui`'s frontend and its own backend, and (b)
  the Python package API in §0 between `tuffy-ui/backend` and `tuffy`. Both need explicit
  versioning so a `tuffy-ui` release doesn't silently break against an incompatible `tuffy`
  package version, or vice versa — not a v1 concern under developer mode (where both are checked
  out from the same workspace at matching commits), but worth pinning a compatible version range
  in `tuffy-ui/backend/pyproject.toml` once `tuffy` starts tagging releases, and keeping the
  `api_version` field (§4) in the WebSocket contract from the start so it's not a retrofit later.

---

## Summary for reviewer

- Confirms `tuffy` stays strictly agent-core, forever: it gains only a public
  `tuffy/__init__.py` export surface (`create_session`, `AgentSession`, `run_turn_stream`) —
  nothing else in the repo changes shape, and no new dependencies (FastAPI, whisper.cpp, Piper)
  land in its `pyproject.toml`.
- `tuffy-ui` (renamed from `tuffy-desktop` this revision) is a genuinely separate repo containing
  **both** the Tauri/frontend shell **and** its own Python backend (`server/`, `voice/`,
  `bridge/`), which installs `tuffy` as a normal package dependency rather than living inside it.
- Terminal (`tuffy`, and later `tuffy --voice`) keeps working unchanged; the exact code-sharing
  path for terminal voice mode vs. `tuffy-ui/backend/voice/` is deliberately left as a Phase 3
  decision rather than guessed at now.
- **Changes in this revision, addressing reviewer feedback (round 2):** moved `server/`/`voice/`
  out of `tuffy` into `tuffy-ui/backend/`, turning the packaging requirement into a real
  cross-repo install boundary (§0); added a `SessionManager` abstraction so multi-chat is a future
  extension, not a rewrite (§3); added a wake-word extension point/interface stub without
  committing to a library (§5); switched audio transport from base64-in-JSON to WebSocket binary
  frames (§4); added a `loading` orb state (§6); added a Models and Assets section
  (`~/.tuffy/models/{whisper,piper}/`, §5); added an explicit Backend Independence Requirement so
  `tuffy-ui/backend` runs and is testable without Tauri (§7); simplified default tool visibility to
  friendly per-group labels only, moving all raw args/results into the Developer Panel exclusively
  (§4); reaffirmed "voice is a wrapper around text, one agent" as a load-bearing constraint (§5);
  renamed the repo to `tuffy-ui` (§ header).
- Awaiting your review before Phase 1 implementation begins.
