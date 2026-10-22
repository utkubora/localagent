from typing import Any

from ..tools.baseTool import Tool, ToolCall, ToolResult, Turn, _new_id
from .common import SYSTEM_PROMPT, Backend, coerce_args, scrape_turn


class OllamaBackend(Backend):
    """A model served by an Ollama daemon (`ollama serve`), local or remote.

    Ollama applies the chat template and parses tool calls server-side, so
    they usually arrive structured. Models without native tool support still
    sometimes write calls into the text; those are scraped like the
    transformers backend does.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        system: str = SYSTEM_PROMPT,
        host: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        think: bool | None = None,
        stream: bool = False,
        client: Any = None,
    ) -> None:
        if client is None:
            import ollama

            client = ollama.Client(host=host)  # host=None honours OLLAMA_HOST
        self.client = client
        self.model = model
        self.options = {"num_predict": max_new_tokens, "temperature": temperature}
        self.think = think
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
            "tools": [t.openai_spec for t in tools],
            "options": self.options,
        }
        if self.think is not None:
            kwargs["think"] = self.think

        text, thinking, raw_calls, done_reason = "", "", [], None
        if self.stream:
            for chunk in self.client.chat(**kwargs, stream=True):
                msg = chunk.message
                if msg.content:
                    print(msg.content, end="", flush=True)
                    text += msg.content
                thinking += msg.thinking or ""
                raw_calls.extend(msg.tool_calls or [])
                done_reason = chunk.done_reason or done_reason
            print()
        else:
            response = self.client.chat(**kwargs)
            text = response.message.content or ""
            thinking = response.message.thinking or ""
            raw_calls = list(response.message.tool_calls or [])
            done_reason = response.done_reason

        if done_reason == "length":
            raise RuntimeError("Response truncated — raise max_new_tokens.")

        calls = [
            ToolCall(_new_id(), c.function.name, coerce_args(c.function.arguments))
            for c in raw_calls
        ]
        if calls:
            turn = Turn(text=text.strip(), thinking=thinking.strip(), tool_calls=calls)
        else:
            turn = scrape_turn(text, thinking)
        self.messages.append(self._assistant_message(turn))
        return turn

    @staticmethod
    def _assistant_message(turn: Turn) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": turn.text}
        if turn.tool_calls:
            message["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in turn.tool_calls
            ]
        return message

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "tool_name": r.name, "content": r.content})
