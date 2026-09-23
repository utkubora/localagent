"""
An AI agent with tool calling that runs against either the Claude API or a local
Hugging Face model loaded with `transformers`.

    pip install anthropic                                # hosted backend
    pip install "transformers>=4.45" torch accelerate    # local backend

    python agent.py --backend local --model Qwen/Qwen2.5-7B-Instruct
    python agent.py --backend anthropic "what is 17 * 23?"

The tool registry is provider-neutral. Each backend translates it into whatever
that provider expects and owns its own conversation history in native format:

  Claude API    tools=[{name, description, input_schema}]; assistant turns
                contain tool_use blocks and results go back as tool_result
                blocks inside a user message.

  transformers  tools=[{"type":"function","function":{...}}] handed to
                apply_chat_template, which renders them into the prompt. The
                model emits tool calls as *text* (`<tool_call>{...}</tool_call>`,
                `[TOOL_CALLS] [...]`, and so on) which we parse back out.
"""

from __future__ import annotations


import json
import sys



from concurrent.futures import ThreadPoolExecutor

from .tools.baseTool import Tool,ToolCall,ToolResult,REGISTRY
from .processor import Backend,AnthropicBackend


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #


class Agent:
    def __init__(
        self,
        backend: Backend | None = None,
        tools: dict[str, Tool] | None = None,
        *,
        max_steps: int = 20,
        parallel: bool = True,
        verbose: bool = True,
    ) -> None:
        self.backend = backend or AnthropicBackend()
        self.tools = tools if tools is not None else REGISTRY
        self.max_steps = max_steps
        self.parallel = parallel
        self.verbose = verbose

    def execute(self, call: ToolCall) -> ToolResult:
        self.log(f"  → {call.name}({json.dumps(call.arguments, default=str)})")
        tool_obj = self.tools.get(call.name)
        if tool_obj is None:
            # Small local models invent tool names; tell them what actually exists.
            available = ", ".join(self.tools)
            return ToolResult(
                call.id, call.name, f"No such tool: {call.name}. Available: {available}", True
            )
        try:
            output = tool_obj.call(call.arguments)
        except Exception as exc:  # surface it to the model instead of crashing
            self.log(f"  ✗ {type(exc).__name__}: {exc}")
            return ToolResult(call.id, call.name, f"{type(exc).__name__}: {exc}", True)

        self.log(f"  ← {output[:200]}{'…' if len(output) > 200 else ''}")
        return ToolResult(call.id, call.name, output)

    def run(self, user_message: str) -> str:
        """Send a message and loop until the model stops requesting tools."""
        self.backend.add_user(user_message)

        for step in range(self.max_steps):
            self.log(f"generating step {step + 1}...")
            turn = self.backend.generate(list(self.tools.values()))

            if not turn.tool_calls:
                return turn.text

            self.log(f"[step {step + 1}] {len(turn.tool_calls)} tool call(s)")
            if self.parallel and len(turn.tool_calls) > 1:
                with ThreadPoolExecutor(max_workers=len(turn.tool_calls)) as pool:
                    results = list(pool.map(self.execute, turn.tool_calls))
            else:
                results = [self.execute(c) for c in turn.tool_calls]

            self.backend.add_tool_results(results)

        raise RuntimeError(f"Gave up after {self.max_steps} steps.")

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr)


