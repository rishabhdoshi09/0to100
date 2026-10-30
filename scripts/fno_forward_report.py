#!/usr/bin/env python3
"""Read-only net-R F&O paper-forward evidence; never triggers scans or orders."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.fno_forward_scorecard import read_forward_scorecard


if __name__ == "__main__":
    report = read_forward_scorecard()
    print(json.dumps(report, indent=2, sort_keys=True))
    sys.exit(0 if report.get("eligible_trades") is not None else 2)
