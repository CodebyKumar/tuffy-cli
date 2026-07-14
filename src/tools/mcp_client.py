"""MCP client: connects to external MCP servers (stdio transport) configured
in .tuffy/mcp.json (gitignored — see docs/configure-mcp.md) plus any
.tuffy/skills/<name>/mcp.json, and registers each server's tools into the
same ToolRegistry native tools use, under group="mcp:<server-name>". From the
model's point of view an MCP tool and a native tool are indistinguishable —
same registry, same dispatch path in src/agent.py.

The MCP Python SDK is async-only (ClientSession.call_tool is a coroutine over
an anyio stream); Tuffy's tool registry and ReAct loop are synchronous. This
module bridges the two with one dedicated background thread running its own
asyncio event loop, holding every server's ClientSession alive for the
process's lifetime; each registered tool wrapper is a plain sync function
that submits a coroutine onto that loop and blocks for the result.
"""

import asyncio
import json
import os
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.tools.registry import registry

MCP_CONFIG_PATH = "./.tuffy/mcp.json"
_CALL_TIMEOUT_SECONDS = 30

_connected_servers = []  # names of servers successfully connected, for /tools-adjacent diagnostics


def _normalize_configs(data) -> list[dict]:
    """Accepts every server-config shape actually seen in the wild, so a
    snippet copied from an MCP server's README or from Claude Desktop/Code's
    own config works in .tuffy/mcp.json unmodified:

    1. The REAL standard (Claude Desktop/Code, Cursor, and virtually every
       MCP server's own README "add to your config" snippet): an
       {"mcpServers": {"<name>": {"command": ..., "args": [...], "env": {...}}}}
       object keyed by server name — the name lives in the KEY, not a field
       inside the value.
    2. {"servers": [...]}: a bare list of {name, command, args?, env?}
       dicts — Tuffy's own original shape, kept for backward compatibility
       with configs already written against it (and what /mcp add writes).
    3. A bare top-level list, same entry shape as #2.

    "type": "stdio" (present in some real-world configs, including
    Anthropic's own docs) is accepted and ignored — stdio is the only
    transport this client supports, so the field carries no decision."""
    if isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
        return [
            {"name": name, **cfg}
            for name, cfg in data["mcpServers"].items()
            if isinstance(cfg, dict)
        ]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("servers", [])
    return []


def _load_server_configs() -> list[dict]:
    """Reads .tuffy/mcp.json. Missing file means no servers configured —
    not an error. See _normalize_configs for every accepted shape."""
    if not os.path.isfile(MCP_CONFIG_PATH):
        return []
    try:
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[mcp] Failed to read {MCP_CONFIG_PATH}: {e}")
        return []

    servers = _normalize_configs(data)
    valid = []
    for entry in servers:
        if not isinstance(entry, dict) or "name" not in entry or "command" not in entry:
            print(f"[mcp] Skipping malformed server config (needs 'name' and 'command'): {entry}")
            continue
        valid.append(entry)
    return valid


class _MCPBridge:
    """Owns one background thread running one asyncio event loop, and one
    live ClientSession per connected MCP server. run_coro() is the only
    thread-safe entry point the sync tool wrappers use."""

    def __init__(self):
        self._loop = None
        self._thread = None
        self._sessions = {}  # server name -> ClientSession
        self._stack_cms = []  # keeps stdio_client/session async context managers alive

    def start(self):
        ready = threading.Event()

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        ready.wait()

    def run_coro(self, coro, timeout: float = _CALL_TIMEOUT_SECONDS):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def _connect_one(self, config: dict):
        name = config["name"]
        params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env={**os.environ, **config.get("env", {})} if config.get("env") else None,
        )
        stdio_cm = stdio_client(params)
        read_stream, write_stream = await stdio_cm.__aenter__()
        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        await session.initialize()

        self._stack_cms.append(stdio_cm)
        self._stack_cms.append(session_cm)
        self._sessions[name] = session
        return session

    def connect_all(self, configs: list[dict]) -> list[str]:
        """Connects to every configured server, registering each tool it
        lists. Returns the names of servers that connected successfully; a
        server that fails to connect is skipped with a printed warning, not
        a crash — one broken MCP server should never block the rest of Tuffy
        from starting."""
        connected = []
        for config in configs:
            name = config["name"]
            try:
                session = self.run_coro(self._connect_one(config))
                tools_result = self.run_coro(session.list_tools())
            except Exception as e:
                print(f"[mcp] Failed to connect to server '{name}': {e}")
                continue

            for tool in tools_result.tools:
                self._register_tool(name, session, tool)
            connected.append(name)
            print(f"[mcp] Connected to '{name}' ({len(tools_result.tools)} tool(s)).")

        _connected_servers.extend(connected)
        return connected

    async def _disconnect_one(self, name: str) -> None:
        session = self._sessions.pop(name, None)
        if session is None:
            return
        # Best-effort: the mcp SDK's ClientSession/stdio_client context
        # managers don't expose a standalone close() outside their __aexit__
        # protocol, and __aexit__ requires the exact exception-info tuple
        # from the `async with` block that opened them - which no longer
        # exists here (they were entered manually in _connect_one, outside
        # any `async with`, specifically so the session could outlive that
        # function call). Terminating the subprocess is what actually stops
        # the server; a session/stream that never gets a clean __aexit__ is
        # harmless - it's already unreachable garbage once popped above.
        transport = getattr(session, "_transport", None) or getattr(session, "_read_stream", None)
        proc = getattr(transport, "_process", None) if transport else None
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    def disconnect(self, name: str) -> bool:
        """Disconnects a live server by name: terminates its subprocess and
        drops the session. Returns False if no such server was connected."""
        if name not in self._sessions:
            return False
        self.run_coro(self._disconnect_one(name))
        if name in _connected_servers:
            _connected_servers.remove(name)
        return True

    def _register_tool(self, server_name: str, session: ClientSession, tool) -> None:
        schema = tool.inputSchema or {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        parameters = {
            pname: {
                "type": pdef.get("type", "string"),
                "description": pdef.get("description", ""),
            }
            for pname, pdef in properties.items()
        }

        # Prefix the tool name with its server so two servers exposing e.g.
        # 'search' can't collide in the shared registry.
        registered_name = f"{server_name}_{tool.name}"
        bridge = self

        def call_tool(**kwargs):
            try:
                result = bridge.run_coro(session.call_tool(tool.name, kwargs))
            except Exception as e:
                return f"MCP tool '{registered_name}' failed: {e}"
            parts = []
            for block in result.content:
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(text)
            return "\n".join(parts) if parts else "(no output)"

        call_tool.__name__ = registered_name
        registry.register(
            name=registered_name,
            description=f"[{server_name} MCP server] {tool.description or tool.name}",
            parameters=parameters,
            required=required,
            group=f"mcp:{server_name}",
        )(call_tool)


_bridge = _MCPBridge()


def connect_one_server(config: dict) -> None:
    """Connects a single server and registers its tools immediately, without
    restarting Tuffy — used by `/mcp add` so a newly configured server is
    usable in the same session. Starts the bridge thread if this is the
    first MCP connection of the session (mirrors connect_mcp_servers).
    Raises on failure (unlike connect_all, which only warns and skips) so
    the caller can show the user exactly why the new server didn't connect."""
    if _bridge._loop is None:
        _bridge.start()
    name = config["name"]
    session = _bridge.run_coro(_bridge._connect_one(config))
    tools_result = _bridge.run_coro(session.list_tools())
    for tool in tools_result.tools:
        _bridge._register_tool(name, session, tool)
    _connected_servers.append(name)


def connect_mcp_servers(extra_configs: list[dict] = None) -> list[str]:
    """Starts the bridge thread (once) and connects to every server in
    .tuffy/mcp.json plus any extra_configs (e.g. from skills' mcp.json).
    Safe to call with no config file and no extra_configs present — a no-op
    returning an empty list. Call once at startup, after tool/skill
    discovery and before the first system prompt is built, so MCP tools show
    up in TOOLS YOU CAN CALL from turn one."""
    configs = _load_server_configs() + (extra_configs or [])
    if not configs:
        return []

    _bridge.start()
    return _bridge.connect_all(configs)


def connected_servers() -> list[str]:
    return list(_connected_servers)


def disconnect_server(name: str) -> bool:
    """Disconnects a live server by name (terminates its subprocess, drops
    the session) — the live-session half of `/mcp remove`; the caller is
    also responsible for registry.unregister_group(f"mcp:{name}") and
    removing the entry from .tuffy/mcp.json (see mcp_install.remove_server_config).
    Returns False if the server wasn't connected this session (e.g. it was
    only ever in the config file, never successfully connected)."""
    return _bridge.disconnect(name)
