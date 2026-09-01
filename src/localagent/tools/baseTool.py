import json
import uuid
import inspect
import types
from dataclasses import dataclass, field
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

