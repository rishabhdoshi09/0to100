"""python -m research.sepa003 — reconstruct + analyse. No orders."""
from __future__ import annotations

import json
import sys


def main(argv=None) -> int:
    from research.sepa003.study import run_study_003
    max_setups = None
    if argv and argv[0].isdigit():
        max_setups = int(argv[0])
    out = run_study_003(max_setups=max_setups)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
