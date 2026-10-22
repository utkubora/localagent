import json
from typing import Any

from ..tools.baseTool import Tool, ToolCall, ToolResult, Turn, _new_id
from .common import SYSTEM_PROMPT, Backend, coerce_args, scrape_turn


def _reasoning(msg: Any) -> str:
    """vLLM calls it `reasoning_content`, newer servers just `reasoning`."""
    return getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""


class OpenAICompatibleBackend(Backend):
    """Any server speaking the OpenAI chat-completions API, local or remote.

    Meant for a separate inference box such as a DGX Spark running vLLM,
    SGLang, TensorRT-LLM, NIM or llama.cpp's server. Tool calls need the
    server's tool parser enabled (vLLM: --enable-auto-tool-choice
    --tool-call-parser <family>); without it they come back as text and are
    scraped like the transformers backend does.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str | None = None,
        api_key: str = "EMPTY",
        system: str = SYSTEM_PROMPT,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        stream: bool = False,
        timeout: float = 600.0,
        client: Any = None,
    ) -> None:
        if client is None:
            import openai

            client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.client = client
        # vLLM and friends usually serve exactly one model, so ask for it.
        self.model = model or client.models.list().data[0].id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.stream = stream
        self.messages: list[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def generate(self, tools: list[Tool]) -> Turn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = [t.openai_spec for t in tools]

        if self.stream:
            text, thinking, raw_calls, finish_reason = self._generate_streaming(kwargs)
        else:
            choice = self.client.chat.completions.create(**kwargs).choices[0]
            msg = choice.message
            text = msg.content or ""
            thinking = _reasoning(msg)
            raw_calls = [(c.id, c.function.name, c.function.arguments) for c in msg.tool_calls or []]
            finish_reason = choice.finish_reason

        if finish_reason == "length":
            raise RuntimeError("Response truncated — raise max_new_tokens.")

        calls = [
            ToolCall(call_id or _new_id(), name, coerce_args(args))
            for call_id, name, args in raw_calls
        ]
        if calls:
            turn = Turn(text=text.strip(), thinking=thinking.strip(), tool_calls=calls)
        else:
            turn = scrape_turn(text, thinking)
        self.messages.append(self._assistant_message(turn))
        return turn

    def _generate_streaming(self, kwargs: dict) -> tuple[str, str, list[tuple], str | None]:
        text, thinking, finish_reason = "", "", None
        # Tool calls stream in fragments keyed by index; stitch them back together.
        partial: dict[int, dict[str, str]] = {}
        for chunk in self.client.chat.completions.create(**kwargs, stream=True):
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                print(delta.content, end="", flush=True)
                text += delta.content
            thinking += _reasoning(delta)
            for tc in delta.tool_calls or []:
                slot = partial.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                slot["id"] = tc.id or slot["id"]
                if tc.function:
                    slot["name"] += tc.function.name or ""
                    slot["arguments"] += tc.function.arguments or ""
            finish_reason = choice.finish_reason or finish_reason
        print()
        raw_calls = [(s["id"], s["name"], s["arguments"]) for _, s in sorted(partial.items())]
        return text, thinking, raw_calls, finish_reason

    @staticmethod
    def _assistant_message(turn: Turn) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": turn.text}
        if turn.tool_calls:
            # The OpenAI API wants `arguments` as a JSON string, unlike transformers.
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in turn.tool_calls
            ]
        return message

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.id, "content": r.content})
