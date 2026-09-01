

import json
import re
from typing import Any
from abc import ABC, abstractmethod

from .tools.baseTool import Tool,ToolCall,ToolResult,Turn,_new_id

# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Use a tool whenever it would give you a more reliable answer than reasoning \
alone — arithmetic, the current time, anything touching the filesystem. Request \
a tool call properly; never describe one in prose or invent its output. If a \
tool returns an error, read it, fix your arguments, and try again. Once you have \
what you need, answer the user directly and concisely without calling any \
further tools."""


class Backend(ABC):
    """Owns the conversation in provider-native format."""

    @abstractmethod
    def add_user(self, text: str) -> None: ...

    @abstractmethod
    def generate(self, tools: list[Tool]) -> Turn:
        """Produce one assistant turn and append it to the history."""

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None: ...


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


# -- pulling tool calls back out of raw text -------------------------------- #

_THINK = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_MISTRAL = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\]|\{.*\})", re.DOTALL)
_FUNCTION_TAG = re.compile(r"<function=([\w.-]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_TAG = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.DOTALL)


def _coerce_args(raw: Any) -> dict:
    """Arguments come as a dict, or a JSON string, or occasionally garbage."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _normalize(entry: dict) -> ToolCall | None:
    """Flatten the many shapes a single tool call arrives in."""
    if isinstance(entry.get("function"), dict):
        entry = entry["function"]
    name = entry.get("name") or entry.get("tool_name")
    if not isinstance(name, str):
        return None
    for key in ("arguments", "parameters", "args", "input"):
        if key in entry:
            return ToolCall(_new_id(), name, _coerce_args(entry[key]))
    return ToolCall(_new_id(), name, {})


def _calls_from_json(payload: str) -> list[ToolCall]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    entries = data if isinstance(data, list) else [data]
    return [c for e in entries if isinstance(e, dict) and (c := _normalize(e))]


def _calls_from_function_tags(payload: str) -> list[ToolCall]:
    """Qwen3 style: <function=name><parameter=k>v</parameter></function>."""
    calls = []
    for fname, body in _FUNCTION_TAG.findall(payload):
        args: dict[str, Any] = {}
        for key, value in _PARAMETER_TAG.findall(body):
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value
        calls.append(ToolCall(_new_id(), fname, args or _coerce_args(body)))
    return calls


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract tool calls from a raw local-model completion.

    Every model family invented its own syntax and output drifts even within a
    family, so try the known wrappers in order of specificity and return
    whatever text is left over as the assistant's prose.
    """
    for pattern in (_TOOL_CALL_TAG, _MISTRAL, _FUNCTION_TAG, _JSON_FENCE):
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        calls: list[ToolCall] = []
        for match in matches:
            payload = match.group(0) if pattern is _FUNCTION_TAG else match.group(1)
            calls.extend(_calls_from_function_tags(payload) or _calls_from_json(payload))
        if calls:
            return pattern.sub("", text).strip(), calls

    # Llama 3.x sometimes emits a bare JSON object, optionally after <|python_tag|>
    calls = []
    for match in _BARE_JSON.finditer(text):
        entry = _calls_from_json(match.group(0))
        if entry and ('"name"' in match.group(0) or "'name'" in match.group(0)):
            calls.extend(entry)
            text = text.replace(match.group(0), "")

    return text.replace("<|python_tag|>", "").strip(), calls


class TransformersBackend(Backend):
    """A local Hugging Face causal LM.

    Needs a checkpoint whose chat template supports tools — Qwen2.5/Qwen3,
    Llama 3.1+, Mistral/Ministral, Hermes, SmolLM3, Command-R and friends. A
    base model, or a chat model with no tool-use template, will raise on the
    first apply_chat_template call.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        system: str = SYSTEM_PROMPT,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        do_sample: bool = True,
        stream: bool = False,
        quantization_config: Any = None,
        model_kwargs: dict | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {"device_map": device_map, **(model_kwargs or {})}
        if quantization_config is not None:  # e.g. BitsAndBytesConfig(load_in_4bit=True)
            kwargs["quantization_config"] = quantization_config

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **kwargs)
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, **kwargs)
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.stream = stream
        self.messages: list[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def generate(self, tools: list[Tool]) -> Turn:
        import torch

        # The template renders the tool schemas into the prompt itself, so the
        # whole conversation is re-encoded from scratch on every step.
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            tools=[t.openai_spec for t in tools],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        streamer = None
        if self.stream:
            from transformers import TextStreamer

            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                streamer=streamer,
            )

        raw = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=False)
        turn = self.parse(raw, inputs["input_ids"][0])
        self.messages.append(self._assistant_message(turn))
        return turn

    def parse(self, raw: str, prefix: Any = None) -> Turn:
        """Prefer the tokenizer's own response template; fall back to regex.

        transformers ships `parse_response` for checkpoints that define a
        response_template. Where that's missing we scrape the text ourselves.
        """
        try:
            parsed = self.tokenizer.parse_response(raw, prefix=prefix)
            calls = []
            for entry in parsed.get("tool_calls") or []:
                fn = entry.get("function", entry) if isinstance(entry, dict) else {}
                if fn.get("name"):
                    calls.append(ToolCall(_new_id(), fn["name"], _coerce_args(fn.get("arguments"))))
            return Turn(
                text=(parsed.get("content") or "").strip(),
                thinking=(parsed.get("thinking") or "").strip(),
                tool_calls=calls,
            )
        except Exception:
            pass  # no response_template on this tokenizer, or parsing failed

        text = self.tokenizer.decode(
            self.tokenizer.encode(raw, add_special_tokens=False), skip_special_tokens=True
        )
        thinking = " ".join(m.group(1).strip() for m in _THINK.finditer(text))
        text, calls = parse_tool_calls(_THINK.sub("", text))
        return Turn(text=text, thinking=thinking, tool_calls=calls)

    @staticmethod
    def _assistant_message(turn: Turn) -> dict:
        if not turn.tool_calls:
            return {"role": "assistant", "content": turn.text}
        # transformers wants `arguments` as a dict here, unlike the OpenAI API
        # which uses a JSON string. Passing a string confuses many templates.
        return {
            "role": "assistant",
            "content": turn.text,
            "tool_calls": [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in turn.tool_calls
            ],
        }

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "name": r.name, "content": r.content})

