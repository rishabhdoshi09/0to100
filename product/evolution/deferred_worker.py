"""CLI worker for one deferred Evolution work item.

Invoked only by product.evolution.deferred_work in a separate OS process so
a hung Challenger can be terminated without blocking the autonomy supervisor.
"""
from __future__ import annotations

import argparse

from product.evolution.deferred_work import process_work_item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--work-id", required=True)
    args = parser.parse_args()
    process_work_item(args.work_id, root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
