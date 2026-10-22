"""Pieces shared by every backend: the base class, the default system prompt,
and the regex fallbacks for models that write tool calls as plain text."""

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from ..tools.baseTool import Tool, ToolCall, ToolResult, Turn, _new_id

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


# -- pulling tool calls back out of raw text -------------------------------- #

_THINK = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_MISTRAL = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\]|\{.*\})", re.DOTALL)
_FUNCTION_TAG = re.compile(r"<function=([\w.-]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_TAG = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.DOTALL)


def coerce_args(raw: Any) -> dict:
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
            return ToolCall(_new_id(), name, coerce_args(entry[key]))
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
        calls.append(ToolCall(_new_id(), fname, args or coerce_args(body)))
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


def scrape_turn(text: str, thinking: str = "") -> Turn:
    """Build a turn from raw text: pull out <think> blocks, then tool calls."""
    thinking = " ".join([thinking, *(m.group(1).strip() for m in _THINK.finditer(text))]).strip()
    text, calls = parse_tool_calls(_THINK.sub("", text))
    return Turn(text=text, thinking=thinking, tool_calls=calls)
