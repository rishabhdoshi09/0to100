"""python -m research.feature002  — resolve + status (no production side effects)."""
from __future__ import annotations

import argparse

from research.feature002.constants import UNTIL_MATURE
from research.feature002.evaluate import summarize
from research.feature002.ledger import counts
from research.feature002.report import write_all
from research.feature002.resolve import resolve_due


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FEATURE-002 resolver / status")
    p.add_argument("--resolve", action="store_true")
    p.add_argument("--health", action="store_true")
    p.add_argument("--watchdog", action="store_true")
    args = p.parse_args(argv)
    if args.resolve:
        print(resolve_due(), flush=True)
    if args.health or args.watchdog:
        from research.feature002.health import write_health, write_status_md
        from research.feature002.watchdog import run as run_watchdog
        h = write_health()
        write_status_md(h)
        print(h.get("user_summary"), flush=True)
        if args.watchdog:
            print(run_watchdog(persist=False, alert=False), flush=True)
        return 0
    write_all()
    summary = summarize()
    print(summary.get("status") or UNTIL_MATURE, flush=True)
    print(counts(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
