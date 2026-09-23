"""
Tools from MCP (Model Context Protocol) servers.

Connects to the servers listed in an `mcp.json` file and registers each of
their tools in the provider-neutral REGISTRY, so they work with both the
Anthropic and the local transformers backend:

    pip install mcp

The config uses the same shape as Claude Desktop. A server is either a local
process spoken to over stdio, or a remote streamable-HTTP endpoint:

    {
      "mcpServers": {
        "fs":     {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]},
        "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ..."}},
        "big":    {"command": "...", "allow": ["only_this_tool"]}
      }
    }

Tools are registered as `<server>__<tool>`. The optional "allow" list keeps a
server from flooding a small local model's prompt with schemas it won't use.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from .baseTool import REGISTRY, Tool

# Anthropic requires tool names to match ^[a-zA-Z0-9_-]{1,64}$.
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


class MCPBridge:
    """Keeps MCP sessions alive on a background event loop and exposes their
    tools as ordinary synchronous functions.

    The MCP SDK is async while Agent is synchronous and runs tools on a thread
    pool, so every call is handed to the loop with run_coroutine_threadsafe.
    """

    def __init__(self, call_timeout: float = 120) -> None:
        self.call_timeout = call_timeout
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        self._stop = asyncio.Event()
        self._tasks: list[concurrent.futures.Future] = []

    def _run(self, coro: Any, timeout: float | None = None) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _serve(self, cfg: dict, ready: concurrent.futures.Future) -> None:
        # anyio requires the transport's contexts to be entered and exited in
        # the same task, so each server gets one long-lived task that holds
        # the session open until close() is called.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        try:
            async with AsyncExitStack() as stack:
                if "command" in cfg:
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=cfg.get("env"),
                        cwd=cfg.get("cwd"),
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                elif "url" in cfg:
                    http = await stack.enter_async_context(
                        create_mcp_http_client(headers=cfg.get("headers"))
                    )
                    read, write = await stack.enter_async_context(
                        streamable_http_client(cfg["url"], http_client=http)
                    )
                else:
                    raise ValueError('server config needs either "command" or "url"')

                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                ready.set_result(session)
                await self._stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                raise

    def connect(self, server: str, cfg: dict, timeout: float = 60) -> list[Tool]:
        """Start one server and return its tools wrapped as Tool objects."""
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self._tasks.append(asyncio.run_coroutine_threadsafe(self._serve(cfg, ready), self.loop))
        session = ready.result(timeout)

        allow = set(cfg["allow"]) if "allow" in cfg else None
        prefix = _INVALID_NAME_CHARS.sub("_", server)
        tools = []
        for spec in self._run(session.list_tools(), timeout).tools:
            if allow is not None and spec.name not in allow:
                continue
            name = f"{prefix}__{_INVALID_NAME_CHARS.sub('_', spec.name)}"[:64]
            tools.append(
                Tool(
                    name=name,
                    description=spec.description or spec.title or spec.name,
                    input_schema=spec.input_schema or {"type": "object", "properties": {}},
                    fn=self._caller(session, spec.name),
                )
            )
        return tools

    def _caller(self, session: Any, tool_name: str):
        def call(**arguments: Any) -> str:
            result = self._run(session.call_tool(tool_name, arguments), self.call_timeout)
            text = _result_text(result)
            if getattr(result, "is_error", False):
                raise RuntimeError(text)  # Agent.execute reports it as is_error=True
            return text

        return call

    def close(self) -> None:
        """Shut down every server session and stop the background loop."""
        self.loop.call_soon_threadsafe(self._stop.set)
        for task in self._tasks:
            try:
                task.result(timeout=10)
            except Exception:
                pass  # a server that failed to start or hangs on exit
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


def _result_text(result: Any) -> str:
    """Flatten a CallToolResult into the plain string ToolResult carries."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        else:
            # Images, audio and embedded resources can't go through a string.
            parts.append(f"[{getattr(block, 'type', 'unknown')} content omitted]")
    structured = getattr(result, "structured_content", None)
    if not parts and structured is not None:
        parts.append(json.dumps(structured, default=str, indent=2))
    return "\n".join(parts) or "(no output)"


# A JSON string (kept as-is) or a // or /* */ comment (dropped). Matching
# strings first stops "https://..." inside a value from reading as a comment.
_JSONC_COMMENT = re.compile(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/', re.DOTALL)
# A comma left dangling once the entry after it has been commented out.
_TRAILING_COMMA = re.compile(r'("(?:\\.|[^"\\])*")|,(?=\s*[}\]])')


def _read_config(path: Path) -> dict:
    """Parse mcp.json, tolerating comments and trailing commas so servers can
    be switched off by commenting them out."""
    text = _JSONC_COMMENT.sub(lambda m: m.group(1) or "", path.read_text(encoding="utf-8"))
    text = _TRAILING_COMMA.sub(lambda m: m.group(1) or "", text)
    if not text.strip():
        return {}
    config = json.loads(text)
    if not isinstance(config, dict):
        raise ValueError("top level must be a JSON object")
    return config


def load_mcp_servers(path: str | Path = "mcp.json", *, verbose: bool = True) -> MCPBridge | None:
    """Connect every server in the config and register its tools.

    Returns None, and the agent runs with only its built-in tools, when the
    config file is missing, empty, unreadable or lists no enabled servers. A
    server that fails to start is reported and skipped so it can't take the
    whole agent down.
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        servers = _read_config(path).get("mcpServers") or {}
        if not isinstance(servers, dict):
            raise ValueError('"mcpServers" must be an object of name -> server config')
    except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        if verbose:
            print(f"Ignoring {path}: {exc}. Running without MCP servers.")
        return None

    servers = {
        name: cfg for name, cfg in servers.items() if isinstance(cfg, dict) and not cfg.get("disabled")
    }
    if not servers:
        return None  # nothing to connect, so don't start the background loop

    bridge = MCPBridge()
    for server, cfg in servers.items():
        try:
            tools = bridge.connect(server, cfg)
        except Exception as exc:
            # anyio wraps transport failures in task-group exception groups.
            while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]
            if verbose:
                print(f"MCP server {server!r} failed to start: {type(exc).__name__}: {exc}")
            continue
        for t in tools:
            REGISTRY[t.name] = t
        if verbose:
            print(f"MCP server {server!r}: {len(tools)} tool(s)")
    return bridge
