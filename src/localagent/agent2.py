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

import argparse
import ast
import inspect
import json
import operator
import os
import re
import sys
import types
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

# --------------------------------------------------------------------------- #
# Tool registry (provider-neutral)
# --------------------------------------------------------------------------- #

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> dict:
    """Map a Python annotation to a JSON Schema fragment."""
    origin = get_origin(annotation)

    if origin is Literal:
        return {"type": "string", "enum": [str(v) for v in get_args(annotation)]}

    if origin is Union or origin is types.UnionType:  # Optional[X] -> schema for X
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {}

    if origin in (list, dict):
        return {"type": _PY_TO_JSON[origin]}

    return {"type": _PY_TO_JSON.get(annotation, "string")}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    current: str | None = None

    for raw in (doc or "").strip().splitlines():
        line = raw.strip()
        if line.lower() in ("args:", "arguments:", "params:", "parameters:"):
            in_args = True
            continue
        if not in_args:
            summary_lines.append(line)
        elif ":" in line:
            name, _, desc = line.partition(":")
            current = name.strip()
            params[current] = desc.strip()
        elif line and current:  # continuation of the previous param
            params[current] += " " + line

    return " ".join(summary_lines).strip(), params


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]

    @property
    def anthropic_spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    @property
    def openai_spec(self) -> dict:
        """The `{"type": "function", ...}` shape HF chat templates expect."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def call(self, arguments: dict) -> str:
        result = self.fn(**arguments)
        return result if isinstance(result, str) else json.dumps(result, default=str, indent=2)


REGISTRY: dict[str, Tool] = {}


def tool(fn: Callable) -> Callable:
    """Decorator: turn a typed, Google-docstring'd function into a Tool.

    Type hints become the JSON Schema, the docstring becomes the description.
    Small local models are much more sensitive to vague descriptions than Claude
    is, so err on the side of over-explaining.
    """
    hints = get_type_hints(fn)
    summary, param_docs = _parse_docstring(fn.__doc__ or "")

    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        schema = _json_type(hints.get(name, str))
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    REGISTRY[fn.__name__] = Tool(
        name=fn.__name__,
        description=summary or fn.__name__,
        input_schema={"type": "object", "properties": properties, "required": required},
        fn=fn,
    )
    return fn


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Example tools — delete these and write your own
# --------------------------------------------------------------------------- #

WORKSPACE = Path("./workspace").resolve()

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result. Supports
    + - * / // % and ** with parentheses. Use this for any arithmetic instead of
    computing it yourself. Does not support variables or function calls.

    Args:
        expression: The expression to evaluate, e.g. "(17 * 23) ** 0.5".
    """
    return str(_eval_node(ast.parse(expression, mode="eval").body))


@tool
def current_time(tz: str = "UTC") -> str:
    """Return the current date and time in ISO 8601 format. Use this whenever the
    answer depends on today's date or the current time.

    Args:
        tz: Currently only "UTC" is supported.
    """
    if tz.upper() != "UTC":
        raise ValueError("only UTC is supported")
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_path(name: str) -> Path:
    path = (WORKSPACE / name).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError("path escapes the workspace directory")
    return path


@tool
def list_files(subdirectory: str = ".") -> str:
    """List the files in the agent's workspace directory. Call this before
    reading a file if you are unsure the file exists.

    Args:
        subdirectory: Path relative to the workspace root. Defaults to the root.
    """
    path = _safe_path(subdirectory)
    if not path.is_dir():
        raise ValueError(f"{subdirectory} is not a directory")
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
    return "\n".join(entries) or "(empty)"


@tool
def read_file(filename: str) -> str:
    """Read a UTF-8 text file from the agent's workspace and return its contents.

    Args:
        filename: Path relative to the workspace root, e.g. "notes/todo.md".
    """
    return _safe_path(filename).read_text(encoding="utf-8")


@tool
def write_file(filename: str, content: str) -> str:
    """Write text to a file in the agent's workspace, creating parent directories
    and overwriting any existing file. Returns a confirmation with the byte count.

    Args:
        filename: Path relative to the workspace root.
        content: The full contents to write. This replaces the whole file.
    """
    path = _safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content.encode())} bytes to {filename}"


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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "local":
        return TransformersBackend(
            model=args.model or "Qwen/Qwen2.5-7B-Instruct",
            max_new_tokens=args.max_new_tokens,
            stream=args.stream,
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY, or use --backend local.")
    return AnthropicBackend(model=args.model or "claude-sonnet-5")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-calling agent, hosted or local.")
    parser.add_argument("prompt", nargs="*", help="run once and exit; omit for a REPL")
    parser.add_argument("--backend", choices=("anthropic", "local"), default="anthropic")
    parser.add_argument("--model", help="API model id or HF checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--stream", action="store_true", help="local backend only")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    WORKSPACE.mkdir(exist_ok=True)
    agent = Agent(build_backend(args), verbose=not args.quiet)

    if args.prompt:
        print(agent.run(" ".join(args.prompt)))
        return

    print(f"Tools: {', '.join(agent.tools)}\nCtrl-C or 'exit' to quit.\n")
    while True:
        try:
            user_input = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user_input.lower() in {"exit", "quit"}:
            return
        if user_input:
            print(f"\nmodel › {agent.run(user_input)}\n")


if __name__ == "__main__":
    main()