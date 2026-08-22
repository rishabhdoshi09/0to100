"""Restore bhav-inferred default membership; keep official as v2 only."""
from __future__ import annotations

import shutil
from pathlib import Path

from data.universe_history import build_from_bhav, history_path, ledger_status

ROOT = Path(__file__).resolve().parents[2]
SIDECAR = ROOT / "logs" / "universe_history_bhav_inferred.json"
V2 = ROOT / "logs" / "universe_history_v2.json"


def restore() -> dict:
    sidecar = build_from_bhav(path=SIDECAR, force=True)
    default = history_path()
    default_st = ledger_status(default)
    restored = False
    if SIDECAR.exists() and sidecar.get("built"):
        # Official incomplete archives must not remain the silent default.
        src = str(default_st.get("source") or "")
        completeness = default_st.get("completeness") or {}
        omitted = int(completeness.get("delisted_omitted_no_listed_date") or 0)
        complete = bool(completeness.get("survivorship_complete")) and omitted == 0
        if src.startswith("nse_equity_l") and not complete:
            shutil.copy2(SIDECAR, default)
            restored = True
    return {
        "sidecar": sidecar,
        "default_before": {k: default_st.get(k) for k in ("source", "rows", "research_grade")},
        "default_after": ledger_status(default),
        "v2": ledger_status(V2) if V2.exists() else None,
        "restored_default_from_bhav": restored,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(restore(), indent=2, default=str))
