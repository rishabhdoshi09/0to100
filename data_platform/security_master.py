"""Authoritative NSE security master projection (one symbol identity owner)."""
from __future__ import annotations

from typing import Any

from data_platform.contracts import CompanyProfile, ObservationMeta, QualityStatus, utc_now_iso


def _sector_of(symbol: str) -> str:
    try:
        from scan.sector_heat import sector_of
        return str(sector_of(symbol) or "")
    except Exception:
        return ""


def _fno_eligible(symbol: str) -> bool:
    try:
        from pathlib import Path
        import json
        path = Path("logs/product/fno_universe.json")
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in list(payload.get("underlyings", []) or []):
            if str(item.get("symbol", "")).upper() == symbol:
                return True
    except Exception:
        pass
    return False


def profile_for_symbol(symbol: str) -> CompanyProfile:
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise ValueError("symbol required")
    sector = _sector_of(sym)
    fno = _fno_eligible(sym)
    name = sym
    try:
        from fundamentals.cache import FundamentalsCache
        cached = FundamentalsCache().get(sym)
        if cached and isinstance(cached, dict):
            name = str(cached.get("company_name") or cached.get("name") or sym)
    except Exception:
        pass
    meta = ObservationMeta(
        symbol=sym,
        source="nse_universe+fundamentals_cache+fno_universe",
        retrieved_at=utc_now_iso(),
        quality_status=QualityStatus.FRESH if sector else QualityStatus.PARTIAL,
        missing_reason="" if sector else "sector mapping unavailable",
    )
    return CompanyProfile(
        symbol=sym,
        company_name=name,
        sector=sector,
        industry=sector,
        fno_eligible=bool(fno),
        index_membership=[],
        meta=meta,
    )


def supported_symbols(limit: int = 5000) -> list[str]:
    try:
        from data.nse_universe import get_nse_universe
        rows = list(get_nse_universe() or [])
        return rows[:limit]
    except Exception:
        try:
            from data.bhavcopy_store import store_symbols
            return store_symbols()[:limit]
        except Exception:
            return []


def security_master_payload(limit: int = 200) -> dict[str, Any]:
    symbols = supported_symbols(limit=limit)
    profiles = [profile_for_symbol(s).__dict__ for s in symbols[:limit]]
    for p in profiles:
        m = p.get("meta")
        if m is not None and hasattr(m, "quality_status"):
            p["meta"] = {
                **m.__dict__,
                "quality_status": m.quality_status.value,
            }
    return {
        "generated_at": utc_now_iso(),
        "count": len(profiles),
        "profiles": profiles,
    }
