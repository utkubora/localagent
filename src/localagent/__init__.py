import argparse
import os
import sys
from pathlib import Path

from .tools.tools import WORKSPACE
from .processor import Backend,TransformersBackend,AnthropicBackend
from .agent import Agent

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
    parser.add_argument("--backend", choices=("anthropic", "local"), default="local")
    parser.add_argument("--model", help="API model id or HF checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--stream", action="store_true", help="local backend only")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--mcp-config", default="mcp.json", help="MCP server config; skipped if the file is missing"
    )
    args = parser.parse_args()

    WORKSPACE.mkdir(exist_ok=True)
    bridge = connect_mcp(args.mcp_config, verbose=not args.quiet)
    try:
        agent = Agent(build_backend(args), verbose=not args.quiet)
        repl(agent, args.prompt)
    finally:
        if bridge is not None:
            bridge.close()


def connect_mcp(config: str, verbose: bool):
    if not Path(config).exists():
        return None
    try:
        from .tools.mcp_bridge import load_mcp_servers
    except ImportError:
        sys.exit(f'{config} found but the MCP SDK is missing: pip install "localagent[mcp]"')
    return load_mcp_servers(config, verbose=verbose)


def repl(agent: Agent, prompt: list[str]) -> None:
    if prompt:
        print(agent.run(" ".join(prompt)))
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