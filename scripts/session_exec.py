"""Exec a command in a fresh POSIX session with terminal-like signal defaults.

CI starts background commands with SIGINT/SIGQUIT ignored. Ignored dispositions survive
exec(), so a backgrounded launcher cannot be used to test terminal Ctrl-C faithfully
unless those signals are restored first. This tiny wrapper establishes a new session,
restores interactive termination signals, then execs the requested command.
"""
from __future__ import annotations

import os
import signal
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("usage: session_exec.py COMMAND [ARG ...]")
    os.setsid()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        signal.signal(sig, signal.SIG_DFL)
    os.execvpe(args[0], args, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
