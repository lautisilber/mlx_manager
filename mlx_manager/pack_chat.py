#!/usr/bin/env python3
"""Interactive text chat loop for MLX model packs that ship their own runtime loader.

See pack_common.py for how these packs are detected and loaded. This mirrors
mlx_lm.chat's REPL for such packs.
"""

import sys
from pathlib import Path

from pack_common import make_generator, render

PACK = Path(sys.argv[1]).resolve()

COMMAND_HELP = "- 'q' to exit\n- 'r' to reset the chat\n- 'h' to display these commands"


def main():
    print(f"[INFO] Starting chat session with custom runtime pack at {PACK}")
    print(COMMAND_HELP)
    generate = make_generator(PACK)
    messages = []
    while True:
        try:
            query = input(">> ")
        except EOFError:
            break
        if query == "q":
            break
        if query == "r":
            messages = []
            continue
        if query == "h":
            print(COMMAND_HELP)
            continue
        messages.append({"role": "user", "content": query})
        text, _, _ = generate(render(PACK, messages), on_token=lambda t: print(t, end="", flush=True))
        print()
        messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
