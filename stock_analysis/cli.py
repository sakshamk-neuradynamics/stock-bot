from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from .agent import build_agent


def _read_text(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


async def run(principles_path: Optional[str] = None) -> None:
    # If not provided, try to load default principles.txt from CWD
    principles = _read_text(principles_path) or _read_text("principles.txt")
    agent = await build_agent(principles=principles)

    # Start with an empty conversation; user will drive the topic/ticker interactively
    messages: List[Dict[str, Any]] = []

    # Interactive loop
    print("\nType 'exit' to quit.")
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            print("Exiting.")
            break
        messages.append({"role": "user", "content": user_input})

        last_assistant_text = None
        async for chunk in agent.astream({"messages": messages}, stream_mode="values"):
            if "messages" in chunk:
                msg = chunk["messages"][-1]
                last_assistant_text = str(msg)
                if hasattr(msg, "pretty_print"):
                    msg.pretty_print()
                else:
                    print(str(msg))
        if last_assistant_text:
            messages.append({"role": "assistant", "content": last_assistant_text})


if __name__ == "__main__":
    # Optional single arg: path to principles file; otherwise defaults to ./principles.txt if present
    principles_file = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run(principles_file))


