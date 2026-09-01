import ast
import operator

from datetime import datetime, timezone
from pathlib import Path
from .baseTool import tool

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


