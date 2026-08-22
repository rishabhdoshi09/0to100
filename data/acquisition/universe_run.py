"""Official listing/delist/symbol-change ingest → identity + universe v2 candidate."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from data.acquisition.cache import write_manifest, write_raw
from data.acquisition.http import HEADERS
from data.nse_universe_ingest import materialize_universe_from_nse
from data.security_identity import (
    _DELISTED_URL,
    _EQUITY_L_URL,
    _SYMBOLCHANGE_URL,
    materialize_from_nse,
)
from data.universe_history import ledger_status


def ingest_official_identity_and_universe() -> dict[str, Any]:
    import requests

    sess = requests.Session()
    sess.headers.update(HEADERS)
    raw_meta = {}
    for name, url in (
        ("equity_l.csv", _EQUITY_L_URL),
        ("delisted.csv", _DELISTED_URL),
        ("symbolchange.csv", _SYMBOLCHANGE_URL),
    ):
        try:
            resp = sess.get(url, timeout=60)
            if resp.status_code == 200 and resp.content:
                rec = write_raw(f"universe/{name}", resp.content, meta={"url": url})
                raw_meta[name] = rec
        except Exception as exc:
            raw_meta[name] = {"error": str(exc)}

    from pathlib import Path
    v2_path = Path(__file__).resolve().parents[2] / "logs" / "universe_history_v2.json"
    ident = materialize_from_nse()
    uni = materialize_universe_from_nse(session=sess, path=v2_path)
    ust = ledger_status(v2_path)
    # Official v2 is a listing-date improvement for *current* EQ, not a
    # complete survivorship archive. Do not overwrite the default research
    # membership file unless official completeness is actually complete.
    if ust.get("research_grade") and (ust.get("completeness") or {}).get("survivorship_complete"):
        materialize_universe_from_nse(session=sess)
        promoted = True
    else:
        promoted = False
    bhav_sidecar = Path(__file__).resolve().parents[2] / "logs" / "universe_history_bhav_inferred.json"
    try:
        from data.universe_history import build_from_bhav
        bhav_st = build_from_bhav(path=bhav_sidecar, force=True)
    except Exception as exc:
        bhav_st = {"built": False, "error": str(exc)}
    man = {
        "source": "nse_equity_l+delisted+symbolchange",
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "raw": raw_meta,
        "identity": {k: ident.get(k) for k in ("n_securities", "n_symbol_changes", "n_delisted") if k in ident}
        if isinstance(ident, dict) else ident,
        "universe_v2_path": str(v2_path),
        "universe_status": ust,
        "v2_created": True,
        "v2_research_grade": bool(ust.get("research_grade")),
        "promoted_to_default": promoted,
        "bhav_inferred_sidecar": bhav_st,
        "ingest": {k: uni.get(k) for k in ("rows", "source", "research_grade", "note") if k in (uni or {})},
    }
    write_manifest("universe_official", man)
    return man
