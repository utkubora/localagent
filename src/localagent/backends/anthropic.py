from typing import Any

from ..tools.baseTool import Tool, ToolCall, ToolResult, Turn
from .common import SYSTEM_PROMPT, Backend


class AnthropicBackend(Backend):
    """Hosted Claude models. Tool calls arrive as structured blocks."""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        system: str = SYSTEM_PROMPT,
        max_tokens: int = 8192,
        extra_tool_specs: list[dict] | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        # Server-side tools, e.g. {"type": "web_search_20250305", "name": "web_search"}.
        # Those execute on Anthropic's side and never reach our executor.
        self.extra_tool_specs = extra_tool_specs or []
        self.messages: list[dict] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def generate(self, tools: list[Tool]) -> Turn:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=[t.anthropic_spec for t in tools] + self.extra_tool_specs,
            messages=self.messages,
        )
        # Appended verbatim: thinking blocks must survive the round trip intact.
        self.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "max_tokens":
            raise RuntimeError("Response truncated — raise max_tokens.")

        return Turn(
            text="\n".join(b.text for b in response.content if b.type == "text"),
            thinking="\n".join(b.thinking for b in response.content if b.type == "thinking"),
            tool_calls=[
                ToolCall(id=b.id, name=b.name, arguments=b.input)
                for b in response.content
                if b.type == "tool_use"
            ],
        )

    def add_tool_results(self, results: list[ToolResult]) -> None:
        blocks: list[dict] = []
        for r in results:
            block = {"type": "tool_result", "tool_use_id": r.id, "content": r.content}
            if r.is_error:
                block["is_error"] = True
            blocks.append(block)
        self.messages.append({"role": "user", "content": blocks})
