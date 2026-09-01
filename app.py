"""QuantTerm entrypoint. The product desk is Vite/React, not Streamlit.

Start: bash scripts/run_quantterm_complete.sh
Open:  http://127.0.0.1:5173
scripts/run_desk.sh is a compatibility wrapper that execs the same command.
"""
from __future__ import annotations

_DESK = "http://127.0.0.1:5173"
_START = "bash scripts/run_quantterm_complete.sh"

_MESSAGE = (
    "QuantTerm does not use Streamlit.\n"
    f"Start the desk with: {_START}\n"
    f"Then open {_DESK}\n"
)


def main() -> int:
    print(_MESSAGE, end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
