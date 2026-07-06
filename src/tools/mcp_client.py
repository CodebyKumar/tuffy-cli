"""MCP client: connects to external MCP servers (stdio transport) configured
in mcp_servers.json (repo root, gitignored — see docs/configure-mcp.md)
plus any skills/<name>/mcp.json, and registers each server's tools into the
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

MCP_CONFIG_PATH = "./mcp_servers.json"
_CALL_TIMEOUT_SECONDS = 30

_connected_servers = []  # names of servers successfully connected, for /tools-adjacent diagnostics


def _load_server_configs() -> list[dict]:
    """Reads mcp_servers.json (a JSON list of {name, command, args?, env?}),
    matching the Claude Desktop/Code config shape. Missing file means no
    servers configured — not an error."""
    if not os.path.isfile(MCP_CONFIG_PATH):
        return []
    try:
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[mcp] Failed to read {MCP_CONFIG_PATH}: {e}")
        return []

    # Accept either a bare list, or {"servers": [...]} for readability.
    servers = data if isinstance(data, list) else data.get("servers", [])
    valid = []
    for entry in servers:
        if "name" not in entry or "command" not in entry:
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


def connect_mcp_servers(extra_configs: list[dict] = None) -> list[str]:
    """Starts the bridge thread (once) and connects to every server in
    mcp_servers.json plus any extra_configs (e.g. from skills' mcp.json).
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
