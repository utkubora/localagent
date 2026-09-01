from tools.tools import WORKSPACE
import os
import sys
from agent import Agent

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")

    WORKSPACE.mkdir(exist_ok=True)
    agent = Agent()

    if len(sys.argv) > 1:
        print(agent.run(" ".join(sys.argv[1:])))
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
            print(f"\nclaude › {agent.run(user_input)}\n")


if __name__ == "__main__":
    main()
