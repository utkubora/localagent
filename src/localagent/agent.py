"""
A minimal-but-real AI agent with tool calling, built on the Claude Messages API.

    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent.py            # interactive REPL
    python agent.py "what is 17 * 23, and what time is it?"

The interesting part is the loop in `Agent.run`: call the model, if it asks for
tools, run them, feed the results back, repeat until it stops asking.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from tools.baseTool import Tool,REGISTRY

import anthropic

# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Use a tool whenever it would give you a more reliable answer than reasoning \
alone — arithmetic, the current time, anything touching the filesystem. You may \
call several tools in one turn when they are independent. If a tool returns an \
error, read it, adjust your input, and try again rather than guessing. When you \
have what you need, answer the user directly and concisely."""


class Agent:
    def __init__(
        self,
        tools: dict[str, Tool] | None = None,
        *,
        model: str = "claude-sonnet-5",
        system: str = SYSTEM_PROMPT,
        max_tokens: int = 8192,
        max_steps: int = 20,
        extra_tool_specs: list[dict] | None = None,
        verbose: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.tools = tools if tools is not None else REGISTRY
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        # Server-side tools (e.g. {"type": "web_search_20250305", "name": "web_search"})
        # run on Anthropic's side — pass them here; they never reach execute().
        self.extra_tool_specs = extra_tool_specs or []
        self.verbose = verbose
        self.client = client or anthropic.Anthropic()
        self.messages: list[dict] = []

    # -- tool execution ----------------------------------------------------- #

    def execute(self, block) -> dict:
        """Run one tool_use block and return a tool_result block."""
        self.log(f"  → {block.name}({json.dumps(block.input)})")
        try:
            tool_obj = self.tools[block.name]
        except KeyError:
            return self._result(block.id, f"No such tool: {block.name}", error=True)

        try:
            output = tool_obj.call(block.input)
        except Exception as exc:  # surface the error to the model, don't crash
            self.log(f"  ✗ {type(exc).__name__}: {exc}")
            return self._result(
                block.id, f"{type(exc).__name__}: {exc}", error=True
            )

        self.log(f"  ← {output[:200]}{'…' if len(output) > 200 else ''}")
        return self._result(block.id, output)

    @staticmethod
    def _result(tool_use_id: str, content: str, error: bool = False) -> dict:
        block = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if error:
            block["is_error"] = True
        return block

    # -- the loop ----------------------------------------------------------- #

    def run(self, user_message: str) -> str:
        """Send a message and keep looping until the model stops calling tools."""
        self.messages.append({"role": "user", "content": user_message})

        for step in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=[t.spec for t in self.tools.values()] + self.extra_tool_specs,
                messages=self.messages,
            )
            # Append the assistant turn verbatim. This matters: thinking blocks
            # must be passed back unmodified alongside the tool_use blocks.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "max_tokens":
                raise RuntimeError("Response truncated — raise max_tokens.")

            if response.stop_reason == "pause_turn":
                continue  # server-side tool paused; just ask it to continue

            if response.stop_reason != "tool_use":
                return self.text(response)

            calls = [b for b in response.content if b.type == "tool_use"]
            self.log(f"[step {step + 1}] {len(calls)} tool call(s)")

            if len(calls) == 1:
                results = [self.execute(calls[0])]
            else:  # independent calls in the same turn can run concurrently
                with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                    results = list(pool.map(self.execute, calls))

            self.messages.append({"role": "user", "content": results})

        raise RuntimeError(f"Gave up after {self.max_steps} steps.")

    @staticmethod
    def text(response) -> str:
        return "\n".join(b.text for b in response.content if b.type == "text")

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr)

